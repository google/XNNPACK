// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack/array-apply.h>
#include <xnnpack/assembler.h>
#include <xnnpack/leb128.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace xnnpack {

struct ValType {
  ValType() = delete;
  constexpr explicit ValType(byte code) : code(code) {}
  byte code;
};

inline bool operator==(const ValType& lhs, const ValType& rhs) {
  return lhs.code == rhs.code;
}

using ValTypesToInt = std::vector<std::pair<ValType, uint32_t>>;

namespace internal {
struct FuncType {
  std::vector<ValType> param;
  std::vector<ValType> result;
};

inline uint32_t& At(ValTypesToInt& map, ValType type) {
  const auto it =
      std::find_if(map.begin(), map.end(), [type](const auto& type_to_int) {
        return type_to_int.first.code == type.code;
      });
  assert((it != map.end()) && "Unknown ValType");
  return it->second;
}

struct Function {
  FuncType type;
  ValTypesToInt locals_declaration;
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
class I32WasmOps : public WasmOpsBase<Derived, I32WasmOps<Derived>> {
 public:
  void i32_add() const { this->Emit8(0x6a); }
  void i32_lt_s() const { this->Emit8(0x48); }
  void i32_const(int32_t value) const {
    this->Emit8(0x41);
    this->EmitEncodedS32(value);
  }
};

template <typename Derived>
class ControlFlowWasmOps
    : public WasmOpsBase<Derived, ControlFlowWasmOps<Derived>> {
 public:
  template <typename Cond, typename If, typename Else>
  void IfElse(Cond&& cond, If&& if_block, Else&& else_block) const {
    cond();
    this->Emit8(kIfCode);
    this->Emit8(kEpsilonCode);  // Fallthru elements are not supported
    if_block();
    this->Emit8(kElseCode);
    else_block();
    end();
  }

  template <typename Cond, typename If>
  void If(Cond&& cond, If&& if_block) const {
    cond();
    this->Emit8(kIfCode);
    this->Emit8(kEpsilonCode);  // Fallthru elements are not supported
    if_block();
    end();
  }

  template <typename Cond, typename Body>
  void DoWhile(Body&& body, Cond&& cond) {
    this->Emit8(kLoopCode);
    this->Emit8(kEpsilonCode);  // Fallthru elements are not supported
    body();
    cond();
    this->Emit8(kBrIfCode);
    this->EmitEncodedU32(0);
    end();
  }

  template <typename Cond, typename Body>
  void While(Cond&& cond, Body&& body) {
    If(cond, [&] { DoWhile(std::forward<Body>(body), cond); });
  }

  void end() const { this->Emit8(0x0b); }

 private:
  static constexpr byte kIfCode = 0x04;
  static constexpr byte kElseCode = 0x05;
  static constexpr byte kEpsilonCode = 0x40;
  static constexpr byte kLoopCode = 0x03;
  static constexpr byte kBrIfCode = 0x0d;
};

class LocalsManager {
 public:
  void ResetLocalsManager(uint32_t parameters_count,
                          const ValTypesToInt& locals_declaration_count) {
    next_index_ = locals_declaration_count;
    max_index_ = locals_declaration_count;
    uint32_t curr = parameters_count;
    for (const auto type_to_count : locals_declaration_count) {
      const ValType type = type_to_count.first;
      At(next_index_, type) = curr;
      curr += type_to_count.second;
      At(max_index_, type) = curr - 1;
    }
  }

  uint32_t GetNewLocalIndex(ValType type) {
    uint32_t& next = At(next_index_, type);
    assert((At(max_index_, type) >= next) &&
           "The number of local variables is exceeded");
    return next++;
  }

  void DestructLocal(ValType type) { At(next_index_, type)--; }

 private:
  ValTypesToInt next_index_;
  ValTypesToInt max_index_;
};

template <typename Derived>
class LocalWasmOps : public I32WasmOps<Derived>, public LocalsManager {
 public:
  struct ValueOnStack {
    ValType type;
    LocalWasmOps<Derived>* ops;
  };

  class Local {
   public:
    Local() = default;

    Local(const ValType& type, uint32_t index, bool is_managed,
          LocalWasmOps<Derived>* ops)
        : type_(type), index_(index), is_managed_(is_managed), ops_(ops) {}

    Local(const Local& other) = delete;

    Local(Local&& other) = default;

    Local& operator=(const Local& other) {
      assert((type_ == other.type_) &&
             "Assignment of locals of different type");
      ops_ = other.ops_;
      ops_->local_get(other.index_);
      ops_->local_set(index_);
      return *this;
    }

    Local& operator=(Local&& other) {
      type_ = other.type_;
      index_ = other.index_;
      is_managed_ = other.is_managed_;
      ops_ = other.ops_;
      return *this;
    }

    Local& operator=(const ValueOnStack& value_on_stack) {
      assert((type_ == value_on_stack.type) &&
             "The type of the local and the type of the value on stack don't "
             "match");
      type_ = value_on_stack.type;
      ops_ = value_on_stack.ops;
      value_on_stack.ops->local_set(index_);
      return *this;
    }

    ~Local() {
      if (is_managed_) ops_->DestructLocal(type_);
    }

    ValType type_{0};
    uint32_t index_{};

   private:
    bool is_managed_ = false;
    LocalWasmOps<Derived>* ops_ = nullptr;
  };

  Local MakeLocal(ValType type) {
    return Local{type, GetNewLocalIndex(type), /*is_managed=*/true, this};
  }

  void local_get(uint32_t index) const {
    this->Emit8(0x20);
    this->EmitEncodedU32(index);
  }

  void local_set(uint32_t index) const {
    this->Emit8(0x21);
    this->EmitEncodedU32(index);
  }

  void local_get(const Local& local) const { local_get(local.index_); }

  void local_set(const Local& local) const { local_set(local.index_); }

  ValueOnStack I32Add(const Local& a, const Local& b) {
    return BinaryOp(a, b, &I32WasmOps<Derived>::i32_add);
  }

  ValueOnStack I32LtS(const Local& a, const Local& b) {
    return BinaryOp(a, b, &I32WasmOps<Derived>::i32_lt_s);
  }

  ValueOnStack I32Const(uint32_t value) {
    this->i32_const(value);
    return {i32, this};
  }

 protected:
  static constexpr ValType i32{0x7F};

 private:
  template <typename Op>
  ValueOnStack BinaryOp(const Local& a, const Local& b, Op&& op) {
    assert((a.type_ == b.type_) &&
           "Binary operation on locals of different types");
    local_get(a);
    local_get(b);
    std::mem_fn(op)(*this);
    return {a.type_, this};
  }
};

class WasmOps : public LocalWasmOps<WasmOps>,
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
  using LocalsManager = internal::LocalsManager;

 public:
  explicit WasmAssembler(xnn_code_buffer* buf) : AssemblerBase(buf) {}

  template <size_t InSize, typename Body>
  void AddFunc(const std::vector<ValType>& result, const char* name,
               const std::array<ValType, InSize>& param,
               const ValTypesToInt& locals_declaration_count, Body&& body) {
    std::vector<byte> code;
    SetOut(&code);
    ResetLocalsManager(InSize, locals_declaration_count);
    std::array<Local, InSize> input_locals{};
    for (uint32_t index = 0; index < InSize; index++) {
      input_locals[index] =
          Local{param[index], index, /*is_managed=*/false, this};
    }
    internal::ArrayApply(std::move(input_locals), std::forward<Body>(body));
    RegisterFunction(result, name, std::vector(param.begin(), param.end()),
                     locals_declaration_count, std::move(code));
  }

  void Emit() {
    EmitMagicVersionAndDlynkSection();
    EmitTypeSection();
    EmitFunctionSection();
    EmitExportsSection();
    EmitCodeSection();
  }

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
                        ValTypesToInt locals_declaration_count,

                        std::vector<byte> code) {
    exports_.push_back(Export{name, functions_.size()});
    functions_.push_back(Function{FuncType{param, result},
                                  std::move(locals_declaration_count),
                                  std::move(code)});
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
