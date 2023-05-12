// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack/assembler.h>
#include <xnnpack/leb128.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace xnnpack {

struct ValType {
  constexpr explicit ValType(byte code) : code(code) {}
  byte code;
};

namespace internal {
struct FuncType {
  std::vector<ValType> param;
  std::vector<ValType> result;
};

struct Function {
  FuncType type;
  std::vector<byte> body;
};

struct Export {
  const char* name;
  size_t function_index;
};

template <typename Derived, typename Intermediate>
class WasmOpsBase {
 public:
  void Emit8(byte b) const { GetDerived()->Emit8(b); }

  void EmitEncodedS32(int32_t value) const {
    GetDerived()->EmitEncodedS32(value);
  }

  void EmitEncodedU32(int32_t value) const {
    GetDerived()->EmitEncodedU32(value);
  }

 private:
  const Derived* GetDerived() const {
    return static_cast<const Derived*>(this);
  }
};

template <typename Derived>
class LocalWasmOps : public WasmOpsBase<Derived, LocalWasmOps<Derived>> {
 public:
  void local_get(uint32_t indx) const {
    this->Emit8(0x20);
    this->EmitEncodedU32(indx);
  }
};

template <typename Derived>
class I32WasmOps : public WasmOpsBase<Derived, I32WasmOps<Derived>> {
 public:
  void i32_add() const { this->Emit8(0x6a); }
  void i32_const(int32_t value) const {
    this->Emit8(0x41);
    this->EmitEncodedS32(value);
  }
};

template <typename Derived>
class ControlFlowWasmOps
    : public WasmOpsBase<Derived, ControlFlowWasmOps<Derived>> {
 public:
  void end() const { this->Emit8(0x0b); }
};

class WasmOps : public LocalWasmOps<WasmOps>,
                public I32WasmOps<WasmOps>,
                public ControlFlowWasmOps<WasmOps> {
 public:
  WasmOps() = default;
  void SetOut(std::vector<byte>* out) { out_ = out; }
  void Emit8(byte b) const { out_->push_back(b); }
  void EmitEncodedS32(int32_t value) const {
    internal::AppendEncodedS32(value, *out_);
  }
  void EmitEncodedU32(uint32_t value) const {
    internal::AppendEncodedU32(value, *out_);
  }

 private:
  std::vector<byte>* out_ = nullptr;
};
}  // namespace internal

class WasmAssembler : public AssemblerBase, protected internal::WasmOps {
 private:
  using Export = internal::Export;
  using Function = internal::Function;
  using FuncType = internal::FuncType;

 public:
  explicit WasmAssembler(xnn_code_buffer* buf) : AssemblerBase(buf) {}

  template <typename Body>
  void AddFunc(const std::vector<ValType>& result, const char* name,
               const std::vector<ValType>& param, Body&& body) {
    std::vector<byte> code;
    SetOut(&code);
    body();
    RegisterFunction(result, name, param, std::move(code));
  }

  void Emit() {
    EmitMagicVersionAndDlynkSection();
    EmitTypeSection();
    EmitFunctionSection();
    EmitExportsSection();
    EmitCodeSection();
  }

 protected:
  static constexpr ValType i32{0x7F};

 private:
  static constexpr std::array<byte, 4> kMagic = {0x00, 0x61, 0x73, 0x6d};
  static constexpr std::array<byte, 4> kVersion = {0x01, 0x00, 0x00, 0x00};
  static constexpr std::array<byte, 17> kDLynk = {
      0x00, 0x0f, 0x08, 0x64, 0x79, 0x6c, 0x69, 0x6e, 0x6b,
      0x2e, 0x30, 0x01, 0x04, 0x00, 0x00, 0x00, 0x00};

  constexpr static byte kTypeSectionCode = 0x01;
  constexpr static byte kFunctionSectionCode = 0x03;
  constexpr static byte kFunctionExportCode = 0x0;
  constexpr static byte kExportsSectionCode = 0x07;
  constexpr static byte kCodeSectionCode = 0x0a;

  void RegisterFunction(const std::vector<ValType>& result, const char* name,
                        const std::vector<ValType>& param,
                        std::vector<byte> code) {
    exports_.push_back(Export{name, functions_.size()});
    functions_.push_back(Function{FuncType{param, result}, std::move(code)});
  }

  template <typename Array>
  void EmitByteArray(Array&& array) {
    for (byte b : array) {
      emit8(b);
    }
  }

  template <typename AppendSection>
  void EmitSection(byte section_code, AppendSection&& append_section) {
    std::vector<byte> out;
    append_section(out);

    emit8(section_code);
    EmitEncodedU32(out.size());
    EmitByteArray(out);
  }

  void EmitMagicVersionAndDlynkSection();

  void EmitTypeSection();

  void EmitFunctionSection();
  void AppendFuncs(std::vector<byte>& out);

  void EmitExportsSection();

  void EmitCodeSection();

  void EmitEncodedU32(uint32_t n) {
    internal::StoreEncodedU32(n, [this](byte b) { emit8(b); });
  }

  std::vector<Function> functions_;
  std::vector<Export> exports_;
};

}  // namespace xnnpack
