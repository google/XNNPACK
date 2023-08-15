#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <numeric>
#include <utility>

#include <xnnpack/assembler.h>
#include <xnnpack/leb128.h>
#include <xnnpack/wasm-assembler.h>


using ::xnnpack::internal::Function;
using ::xnnpack::internal::FuncType;
using ::xnnpack::internal::VectorEncodingLength;
using ::xnnpack::internal::WidthEncodedU32;

namespace xnnpack {
namespace internal {
std::array<uint8_t, 16> MakeLanesForI8x16Shuffle(const uint8_t* lanes, size_t num_lanes) {
  std::array<uint8_t, 16> i8x16_lanes;
  auto it = i8x16_lanes.begin();
  for (int lane_index = 0; lane_index < num_lanes; lane_index++) {
    const uint8_t lane = lanes[lane_index];
    assert((lane < num_lanes * 2) && "Lane index is too large");
    const size_t i8x16_lanes_per_original_lane = 16 / num_lanes;
    for (int i = 0; i < i8x16_lanes_per_original_lane; i++) *it++ = i + lane * i8x16_lanes_per_original_lane;
  }
  return i8x16_lanes;
}

template <typename Indices>
auto FindIndicesForType(Indices&& indices, ValType type) {
  auto it = std::find_if(indices.begin(), indices.end(),
                         [type](const auto& indices_for_type) { return indices_for_type.type == type; });
  assert((it != indices.end()) && "Unknown ValType");
  return it;
}

void LocalsManager::ResetLocalsManager(uint32_t parameters_count, const ValTypesToInt& locals_declaration_count) {
  uint32_t curr = parameters_count;
  size_t i = 0;
  for (const auto type_to_count : locals_declaration_count) {
    const ValType type = type_to_count.type;
    const uint32_t size = type_to_count.value;
    indices_[i] = ValTypeIndices(type, curr, size);
    curr += size;
    i++;
  }
}

uint32_t LocalsManager::GetNewLocalIndex(ValType type) {
  return FindIndicesForType(indices_, type)->GetNewIndex();
}

void LocalsManager::DestructLocal(ValType type, uint32_t index) {
  FindIndicesForType(indices_, type)->DestroyLocal(index);
}

uint32_t LocalsManager::ValTypeIndices::GetNewIndex() {
  for (size_t i = 0; i < kMaxLocalsOfSameType; i++) {
    if (!bitset[i]) {
      bitset[i].flip();
      assert((i < size) && "The number of local variables is exceeded");
      return first_index + i;
    }
  }
  XNN_UNREACHABLE;
}

void LocalsManager::ValTypeIndices::DestroyLocal(uint32_t index) {
  bitset[index - first_index].flip();
}

}  // namespace internal

void WasmAssembler::EmitMagicVersion() {
  EmitByteArray(kMagic);
  EmitByteArray(kVersion);
}

// Functions emitting types section
void WasmAssembler::EmitParamsType(const Params& type) {
  EmitEncodedU32(type.size());
  for (ValType val_type : type) {
    emit8(val_type.code);
  }
}

void WasmAssembler::EmitResultType(const ResultType& type) {
  if (type.IsVoid()) {
    EmitEncodedU32(0);
  } else {
    EmitEncodedU32(1);
    emit8(type.type.code);
  }
}

void WasmAssembler::EmitFuncType(const FuncType& type) {
  static constexpr byte kFunctionByte = 0x60;
  emit8(kFunctionByte);
  EmitParamsType(type.params);
  EmitResultType(type.result);
}

static uint32_t FuncTypeEncodingSize(const FuncType& type) {
  const uint32_t args_count = type.params.size();
  const uint32_t args_encoding_length = WidthEncodedU32(args_count) + args_count;
  // a function can return at  most one value
  const uint32_t result_encoding_length = type.result.IsVoid() ? 1 : 2;
  return 1 +  // the function byte 0x60
         args_encoding_length + result_encoding_length;
}

void WasmAssembler::EmitTypeSection() {
  EmitSection(kTypeSectionCode, func_types_, FuncTypeEncodingSize,
              [this](const FuncType& type) { EmitFuncType(type); });
}

void WasmAssembler::EmitImportSection() {
  EmitByteArray(kImportSection);
}

// Functions emitting Function section
void WasmAssembler::EmitFunctionSection() {
  EmitSection(
    kFunctionSectionCode, functions_, [](const Function& func) { return WidthEncodedU32(func.type_index); },
    [this](const Function& func) { EmitEncodedU32(func.type_index); });
}

// Functions emitting Export section
void WasmAssembler::EmitExport(const Function& func) {
  constexpr static byte kFunctionExportCode = 0x0;
  EmitEncodedU32(func.name_length);
  for (int i = 0; i < func.name_length; i++) {
    emit8(func.name[i]);
  }
  emit8(kFunctionExportCode);
  EmitEncodedU32(func.function_index);
}

static uint32_t ExportEncodingLength(const Function& exp) {
  return WidthEncodedU32(exp.name_length) + exp.name_length + 1 + WidthEncodedU32(exp.function_index);
}

void WasmAssembler::EmitExportsSection() {
  EmitSection(kExportsSectionCode, functions_, ExportEncodingLength,
              [this](const Function& func) { EmitExport(func); });
}

// Functions emitting Code section
static uint32_t LocalsDeclarationSize(const Function& func) {
  const auto& locals_declaration = func.locals_declaration;
  return VectorEncodingLength(locals_declaration, [](const auto& decl) { return WidthEncodedU32(decl.value); }) +
         locals_declaration.size();
}

static uint32_t FunctionEncodingSize(const Function& func) {
  const uint32_t decl_encoding_size = LocalsDeclarationSize(func);
  const uint32_t decl_size = func.locals_declaration.size();
  return WidthEncodedU32(func.body.size() + decl_size) + decl_encoding_size + func.body.size();
}

void WasmAssembler::EmitFunction(const Function& func) {
  const auto& locals_declaration = func.locals_declaration;
  uint32_t decl_encoding_size = LocalsDeclarationSize(func);
  EmitEncodedU32(func.body.size() + decl_encoding_size);

  EmitEncodedU32(locals_declaration.size());
  for (const auto& type_to_count : locals_declaration) {
    EmitEncodedU32(type_to_count.value);
    emit8(type_to_count.type.code);
  }
  move_emitted(func.body.begin_offset, func.body.size());
}

void WasmAssembler::EmitCodeSection() {
  EmitSection(kCodeSectionCode, functions_, FunctionEncodingSize, [this](const Function& func) { EmitFunction(func); });
}

void WasmAssembler::RegisterFunction(const ResultType& result, const char* name, const Params& params,
                                     const ValTypesToInt& locals_declaration_count, Code code) {
  functions_.emplace_back(name, functions_.size(), FindOrAddFuncType(FuncType(params, result)),
                          locals_declaration_count, code);
}

uint32_t WasmAssembler::FindOrAddFuncType(const FuncType& type) {
  const auto it = std::find(func_types_.begin(), func_types_.end(), type);
  if (it != func_types_.end()) return std::distance(func_types_.begin(), it);
  func_types_.push_back(type);
  return func_types_.size() - 1;
}
}  // namespace xnnpack
