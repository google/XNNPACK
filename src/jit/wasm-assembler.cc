#include <xnnpack/assembler.h>
#include <xnnpack/wasm-assembler.h>

#include <vector>

namespace xnnpack {

using internal::AppendEncodedU32;
using internal::Export;
using internal::Function;
using internal::FuncType;

void WasmAssembler::EmitMagicVersionAndDlynkSection() {
  EmitByteArray(kMagic);
  EmitByteArray(kVersion);
  EmitByteArray(kDLynk);
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
  out.push_back(func.body.size() + 2 * func.locals_declaration.size() + 1);

  AppendEncodedU32(func.locals_declaration.size(), out);
  for (const auto& type_to_count : func.locals_declaration) {
    AppendEncodedU32(type_to_count.second, out);
    out.push_back(type_to_count.first.code);
  }
  out.insert(out.end(), func.body.begin(), func.body.end());
}

void WasmAssembler::EmitCodeSection() {
  EmitSection(kCodeSectionCode, AppendArray(functions_, AppendFunctionBody));
}
}  // namespace xnnpack
