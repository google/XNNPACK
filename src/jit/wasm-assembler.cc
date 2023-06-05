#include <xnnpack/assembler.h>
#include <xnnpack/leb128.h>
#include <xnnpack/wasm-assembler.h>

#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

namespace xnnpack {

namespace internal {
std::array<uint8_t, 16> MakeLanesForI8x16Shuffle(const uint8_t* lanes,
                                                 size_t num_lanes) {
  std::array<uint8_t, 16> i8x16_lanes;
  auto it = i8x16_lanes.begin();
  for (int lane_index = 0; lane_index < num_lanes; lane_index++) {
    const uint8_t lane = lanes[lane_index];
    assert((lane < num_lanes * 2) && "Lane index is too large");
    const size_t i8x16_lanes_per_original_lane = 16 / num_lanes;
    for (int i = 0; i < i8x16_lanes_per_original_lane; i++)
      *it++ = i + lane * i8x16_lanes_per_original_lane;
  }
  return i8x16_lanes;
}
}  // namespace internal

using internal::AppendEncodedU32;
using internal::Export;
using internal::Function;
using internal::FuncType;
using internal::WidthEncodedU32;

void WasmAssembler::EmitMagicVersion() {
  EmitByteArray(kMagic);
  EmitByteArray(kVersion);
}

template <typename Array, typename Appender>
static auto AppendArray(Array&& arr, Appender&& appender) {
  return [&](std::vector<byte>& out) {
    AppendEncodedU32(arr.size(), out);
    for (const auto& elem : arr) {
      appender(elem, out);
    }
  };
}

// Functions emitting types section
static void AppendResultType(const std::vector<ValType>& type,
                             std::vector<byte>& out) {
  AppendEncodedU32(type.size(), out);
  for (ValType val_type : type) {
    out.push_back(val_type.code);
  }
}

static void AppendFuncType(const Function& func, std::vector<byte>& out) {
  const FuncType& type = func.type;
  static constexpr byte kFunctionByte = 0x60;
  out.push_back(kFunctionByte);
  AppendResultType(type.param, out);
  AppendResultType(type.result, out);
}

void WasmAssembler::EmitTypeSection() {
  EmitSection(kTypeSectionCode, AppendArray(functions_, AppendFuncType));
}

void WasmAssembler::EmitImportSection() { EmitByteArray(kImportSection); }

// Functions emitting Function section
void WasmAssembler::AppendFuncs(std::vector<byte>& out) {
  AppendEncodedU32(functions_.size(), out);
  for (int i = 0; i < functions_.size(); i++) {
    AppendEncodedU32(i, out);
  }
}

void WasmAssembler::EmitFunctionSection() {
  EmitSection(kFunctionSectionCode,
              [this](std::vector<byte>& out) { AppendFuncs(out); });
}

// Functions emitting Export section
static void AppendExport(const Export& exp, std::vector<byte>& out) {
  constexpr static byte kFunctionExportCode = 0x0;
  const size_t name_length = strlen(exp.name);
  AppendEncodedU32(name_length, out);
  for (int i = 0; i < name_length; i++) {
    out.push_back(exp.name[i]);
  }
  out.push_back(kFunctionExportCode);
  AppendEncodedU32(exp.function_index, out);
}

void WasmAssembler::EmitExportsSection() {
  EmitSection(kExportsSectionCode, AppendArray(exports_, AppendExport));
}

// Functions emitting Code section
static void AppendFunctionBody(const Function& func, std::vector<byte>& out) {
  const auto& locals_declaration = func.locals_declaration;
  const uint32_t locals_declaration_size =
      std::accumulate(locals_declaration.begin(),
                      locals_declaration.end(), 0,
                      [](uint32_t sum, const auto& declaration) {
                        return sum + WidthEncodedU32(declaration.second);
                      }) +
      locals_declaration.size() +
      WidthEncodedU32(locals_declaration.size());
  AppendEncodedU32(func.body.size() + locals_declaration_size, out);

  AppendEncodedU32(locals_declaration.size(), out);
  for (const auto& type_to_count : locals_declaration) {
    AppendEncodedU32(type_to_count.second, out);
    out.push_back(type_to_count.first.code);
  }
  out.insert(out.end(), func.body.begin(), func.body.end());
}

void WasmAssembler::EmitCodeSection() {
  EmitSection(kCodeSectionCode, AppendArray(functions_, AppendFunctionBody));
}
}  // namespace xnnpack
