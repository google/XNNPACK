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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace xnnpack {

struct ValType {
  ValType() = delete;
  ValType(const ValType&) = default;
  ValType& operator=(const ValType&) = default;
  constexpr explicit ValType(byte code) : code(code) {}
  byte code;
};

inline bool operator==(const ValType& lhs, const ValType& rhs) {
  return lhs.code == rhs.code;
}

using ValTypesToInt = std::vector<std::pair<ValType, uint32_t>>;

namespace internal {
template <typename Array, typename ElementEncodingLength>
static uint32_t VectorEncodingLength(
    Array&& array, ElementEncodingLength&& element_encoding_length) {
  const auto add_encoding_length = [&](uint32_t acc, const auto& element) {
    return acc + element_encoding_length(element);
  };
  const uint32_t total_length_of_element_encodings = std::accumulate(
      array.begin(), array.end(), uint32_t{0}, add_encoding_length);
  return WidthEncodedU32(array.size()) + total_length_of_element_encodings;
}

struct ResultType {
  ResultType(std::initializer_list<ValType> codes) : type(kNoTypeCode) {
    switch (codes.size()) {
      case 0:
        break;
      case 1:
        type = *codes.begin();
        break;
      default:
        XNN_UNREACHABLE;
    }
  }

  bool IsVoid() const { return type.code == kNoTypeCode; }

  ValType type;

 private:
  static constexpr byte kNoTypeCode = 0;
};

inline bool operator==(const ResultType& lhs, const ResultType& rhs) {
  return lhs.type == rhs.type;
}

struct FuncType {
  FuncType(std::vector<ValType> param, ResultType result)
      : param(std::move(param)), result(result) {}
  std::vector<ValType> param;
  ResultType result;
};

inline bool operator==(const FuncType& lhs, const FuncType& rhs) {
  return lhs.param == rhs.param && lhs.result == rhs.result;
}

inline uint32_t& At(ValTypesToInt& map, ValType type) {
  const auto it =
      std::find_if(map.begin(), map.end(), [type](const auto& type_to_int) {
        return type_to_int.first.code == type.code;
      });
  assert((it != map.end()) && "Unknown ValType");
  return it->second;
}

struct Function {
  Function(uint32_t type_index, ValTypesToInt locals_declaration,
           std::vector<byte> body)
      : type_index(type_index),
        locals_declaration(std::move(locals_declaration)),
        body(std::move(body)) {}
  uint32_t type_index;
  ValTypesToInt locals_declaration;
  std::vector<byte> body;
};

struct Export {
  Export(const char* name, size_t function_index)
      : name(name), name_length(strlen(name)), function_index(function_index) {}

  const char* name;
  size_t name_length;
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

 protected:
  const Derived* GetDerived() const {
    return static_cast<const Derived*>(this);
  }
};

template <typename Derived>
class I32WasmOps : public WasmOpsBase<Derived, I32WasmOps<Derived>> {
 public:
  void i32_add() const { this->Emit8(0x6A); }
  void i32_sub() const { this->Emit8(0x6B); }
  void i32_and() const { this->Emit8(0x71); }
  void i32_lt_s() const { this->Emit8(0x48); }
  void i32_le_s() const { this->Emit8(0x4C); }
  void i32_ge_u() const { this->Emit8(0x4F); }
  void i32_shl() const { this->Emit8(0x74); }
  void i32_shr_u() const { this->Emit8(0x76); }
  void i32_ne() const { this->Emit8(0x47); }
  void i32_const(int32_t value) const {
    this->Emit8(0x41);
    this->EmitEncodedS32(value);
  }
};

template <typename Derived>
class F32WasmOps : public WasmOpsBase<Derived, F32WasmOps<Derived>> {
 public:
  void f32_const(float value) const {
    this->Emit8(0x43);
    this->EmitEncodedF32(value);
  }

 private:
  void EmitEncodedF32(float value) const {
    std::array<byte, sizeof(float)> encoding;
    memcpy(encoding.data(), &value, sizeof(float));
    for (byte b : encoding) this->Emit8(b);
  }
};

template <typename Derived>
class V128WasmOps : public WasmOpsBase<Derived, V128WasmOps<Derived>> {
 public:
  void f32x4_splat() const { EmitVectorOpcode(0x13); }
  void f32x4_mul() const { EmitVectorOpcode(0xE6); }
  void f32x4_add() const { EmitVectorOpcode(0xE4); }
  void f32x4_pmax() const { EmitVectorOpcode(0xEB); }
  void f32x4_pmin() const { EmitVectorOpcode(0xEA); }
  void i8x16_shuffle(const std::array<uint8_t, 16>& lanes) const {
    EmitVectorOpcode(0x0D);
    for (auto lane : lanes) {
      assert(lane < 32);
      this->Emit8(lane);
    }
  }

  void EmitVectorOpcodePrefix() const { this->Emit8(0xFD); }

 private:
  void EmitVectorOpcode(uint32_t code) const {
    EmitVectorOpcodePrefix();
    this->EmitEncodedU32(code);
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

  void end() const { this->Emit8(0x0B); }

 private:
  static constexpr byte kIfCode = 0x04;
  static constexpr byte kElseCode = 0x05;
  static constexpr byte kEpsilonCode = 0x40;
  static constexpr byte kLoopCode = 0x03;
  static constexpr byte kBrIfCode = 0x0D;
};

template <typename Derived>
class MemoryWasmOps : public WasmOpsBase<Derived, MemoryWasmOps<Derived>> {
 public:
  void i32_load(uint32_t offset = 0, uint32_t alignment = 4) const {
    load_or_store(0x28, offset, alignment);
  }

  void i32_store(uint32_t offset = 0, uint32_t alignment = 4) const {
    load_or_store(0x36, offset, alignment);
  }

  void v128_load(uint32_t offset = 0, uint32_t alignment = 4) const {
    vector_load_or_store(0x00, offset, alignment);
  }

  void v128_load32_splat(uint32_t offset = 0, uint32_t alignment = 4) const {
    vector_load_or_store(0x09, offset, alignment);
  }

  void v128_load64_splat(uint32_t offset = 0, uint32_t alignment = 4) const {
    vector_load_or_store(0x0A, offset, alignment);
  }

  void v128_store(uint32_t offset = 0, uint32_t alignment = 4) const {
    vector_load_or_store(0x0B, offset, alignment);
  }

  void v128_store64_lane(uint8_t lane, uint32_t offset = 0,
                         uint32_t alignment = 4) const {
    v128_store_lane(0x5B, lane, /*max_lane=*/4, offset, alignment);
  }

  void v128_store32_lane(uint8_t lane, uint32_t offset = 0,
                         uint32_t alignment = 4) const {
    v128_store_lane(0x5A, lane, /*max_lane=*/8, offset, alignment);
  }

 private:
  void load_or_store(byte opcode, uint32_t offset, uint32_t alignment) const {
    this->EmitEncodedU32(opcode);
    this->EmitEncodedU32(log2(alignment));
    this->EmitEncodedU32(offset);
  }

  void vector_load_or_store(byte opcode, uint32_t offset,
                            uint32_t alignment) const {
    this->GetDerived()->EmitVectorOpcodePrefix();
    load_or_store(opcode, offset, alignment);
  }

  void v128_store_lane(byte opcode, uint8_t lane, uint8_t max_lane,
                       uint32_t offset = 0, uint32_t alignment = 4) const {
    assert(lane < max_lane);
    vector_load_or_store(opcode, offset, alignment);
    this->Emit8(lane);
  }
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

std::array<uint8_t, 16> MakeLanesForI8x16Shuffle(const uint8_t* lanes,
                                                 size_t num_lanes);

template <typename Derived>
class LocalWasmOps : public LocalsManager {
 public:
  class Local;

  struct ValueOnStack {
    ValueOnStack(const ValType type, Derived* ops) : type(type), ops(ops) {}
    ValueOnStack(const Local& local) : type(local.type_), ops(local.ops_) {
      ops->local_get(local);
    }
    ValType type;
    Derived* ops;
  };

  class Local {
   public:
    Local() = default;

    Local(const ValType& type, uint32_t index, bool is_managed, Derived* ops)
        : type_(type), index_(index), ops_(ops), is_managed_(is_managed) {}

    Local(const Local& other) = delete;

    Local(Local&& other)
        : type_(other.type_),
          index_(other.index_),
          ops_(other.ops_),
          is_managed_(other.is_managed_) {
      other.index_ = kInvalidIndex;
    }

    Local& operator=(const Local& other) {
      assert((type_ == other.type_) &&
             "Assignment of locals of different type");
      ops_ = other.ops_;
      ops_->local_get(other.index_);
      ops_->local_set(index_);
      return *this;
    }

    Local& operator=(Local&& other) {
      assert((index_ == kInvalidIndex) &&
             "The local already binds to something");
      type_ = other.type_;
      index_ = other.index_;
      other.index_ = kInvalidIndex;
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
      if (is_managed_ && index_ != kInvalidIndex) ops_->DestructLocal(type_);
    }

    ValType type_{0};
    uint32_t index_{kInvalidIndex};
    Derived* ops_ = nullptr;

   private:
    bool is_managed_ = false;
    static constexpr uint32_t kInvalidIndex = -1;
  };

  constexpr static size_t kMaxLocalsSize = 32;

  struct LocalsArray {
    explicit LocalsArray(size_t size) : size(size) {
      assert(size <= kMaxLocalsSize);
    }
    auto begin() { return arr.begin(); }
    auto begin() const { return arr.cbegin(); }
    auto end() {
      auto result = arr.begin();
      std::advance(result, size);
      return result;
    }
    auto end() const {
      auto result = arr.cbegin();
      std::advance(result, size);
      return result;
    }
    auto& operator[](size_t index) {
      assert(index < size);
      return arr[index];
    }
    const auto& operator[](size_t index) const {
      assert(index < size);
      return arr[index];
    }

    size_t size;
    std::array<Local, kMaxLocalsSize> arr;
  };

  Local MakeLocal(ValType type) {
    return Local{type, GetNewLocalIndex(type), /*is_managed=*/true,
                 GetMutableDerived()};
  }

  Local MakeLocal(const ValueOnStack& rhs) {
    auto result = MakeLocal(rhs.type);
    result = rhs;
    return result;
  }

  LocalsArray MakeLocalsArray(size_t size, ValType type) {
    LocalsArray locals(size);
    for (auto& local : locals) local = MakeLocal(type);
    return locals;
  }

  void local_get(uint32_t index) const {
    GetDerived()->Emit8(0x20);
    GetDerived()->EmitEncodedU32(index);
  }

  void local_set(uint32_t index) const {
    GetDerived()->Emit8(0x21);
    GetDerived()->EmitEncodedU32(index);
  }

  void local_get(const Local& local) const { local_get(local.index_); }

  void local_set(const Local& local) const { local_set(local.index_); }

  ValueOnStack I32Add(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::i32_add);
  }

  ValueOnStack I32Sub(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::i32_sub);
  }

  ValueOnStack I32And(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::i32_and);
  }

  ValueOnStack I32LtS(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::i32_lt_s);
  }

  ValueOnStack I32LeS(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::i32_le_s);
  }

  ValueOnStack I32GeU(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::i32_ge_u);
  }

  ValueOnStack I32Shl(const ValueOnStack& value, const ValueOnStack& bits_num) {
    return BinaryOp(value, bits_num, &Derived::i32_shl);
  }

  ValueOnStack I32Ne(const ValueOnStack& lhs, const ValueOnStack& rhs) {
    return BinaryOp(lhs, rhs, &Derived::i32_ne);
  }

  ValueOnStack I32ShrU(const ValueOnStack& value,
                       const ValueOnStack& bits_num) {
    return BinaryOp(value, bits_num, &Derived::i32_shr_u);
  }

  ValueOnStack I32Const(uint32_t value) {
    GetDerived()->i32_const(value);
    return MakeValueOnStack(i32);
  }

  ValueOnStack F32Const(float value) {
    GetDerived()->f32_const(value);
    return MakeValueOnStack(f32);
  }

  ValueOnStack I32Load(const ValueOnStack& address, uint32_t offset = 0,
                       uint32_t alignment = 4) {
    return LoadOp(i32, offset, alignment, &Derived::i32_load);
  }

  ValueOnStack I32Load(const ValueOnStack& base,
                       const ValueOnStack& dynamic_offset,
                       uint32_t static_offset = 0, uint32_t alignment = 4) {
    return I32Load(I32Add(base, I32Shl(dynamic_offset, I32Const(2))),
                   static_offset, alignment);
  }

  void I32Store(const ValueOnStack& address, const ValueOnStack& value,
                uint32_t offset = 0, uint32_t alignment = 4) {
    GetDerived()->i32_store(offset, alignment);
  }

  void I32Store(const ValueOnStack& base, const ValueOnStack& dynamic_offset,
                const Local& value, uint32_t static_offset = 0,
                uint32_t alignment = 4) {
    I32Store(I32Add(base, I32Shl(dynamic_offset, I32Const(2))), value,
             static_offset, alignment);
  }

  ValueOnStack F32x4Add(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::f32x4_add);
  }

  ValueOnStack F32x4Pmax(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::f32x4_pmax);
  }

  ValueOnStack F32x4Pmin(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::f32x4_pmin);
  }

  ValueOnStack I64x2Shuffle(const ValueOnStack& a, const ValueOnStack& b,
                            const std::array<uint8_t, 2>& lanes) {
    return BinaryOp(a, b, [&](const Derived* derived) {
      derived->i8x16_shuffle(
          MakeLanesForI8x16Shuffle(lanes.data(), lanes.size()));
    });
  }

  ValueOnStack F32x4Splat(const ValueOnStack& a) {
    assert(a.type == f32);
    GetDerived()->f32x4_splat();
    return MakeValueOnStack(v128);
  }

  ValueOnStack F32x4Mul(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::f32x4_mul);
  }

  ValueOnStack V128Load(const ValueOnStack& address, uint32_t offset = 0,
                        uint32_t alignment = 4) {
    return LoadOp(v128, offset, alignment, &Derived::v128_load);
  }

  ValueOnStack V128Load32Splat(const ValueOnStack& address, uint32_t offset = 0,
                               uint32_t alignment = 4) {
    return LoadOp(v128, offset, alignment, &Derived::v128_load32_splat);
  }

  ValueOnStack V128Load64Splat(const ValueOnStack& address, uint32_t offset = 0,
                               uint32_t alignment = 4) {
    return LoadOp(v128, offset, alignment, &Derived::v128_load64_splat);
  }

  void V128Store(const ValueOnStack& address, const ValueOnStack& value,
                 uint32_t offset = 0, uint32_t alignment = 4) {
    GetDerived()->v128_store(offset, alignment);
  }

  void V128Store64Lane(const ValueOnStack& address, const ValueOnStack& value,
                       uint8_t lane, uint32_t offset = 0,
                       uint32_t alignment = 4) {
    GetDerived()->v128_store64_lane(lane, offset, alignment);
  }

  void V128Store32Lane(const ValueOnStack& address, const ValueOnStack& value,
                       uint8_t lane, uint32_t offset = 0,
                       uint32_t alignment = 4) {
    GetDerived()->v128_store32_lane(lane, offset, alignment);
  }

 protected:
  static constexpr ValType i32{0x7F};
  static constexpr ValType f32{0x7D};
  static constexpr ValType v128{0x7B};

 private:
  template <typename Op>
  ValueOnStack BinaryOp(const ValueOnStack& a, const ValueOnStack& b, Op&& op) {
    assert((a.type == b.type) &&
           "Binary operation on locals of different types");
    CallOnDerived(std::forward<Op>(op));
    return MakeValueOnStack(a.type);
  }

  template <typename Op, typename = std::enable_if_t<
                             std::is_member_function_pointer<Op>::value>>
  void CallOnDerived(Op&& op) {
    std::mem_fn(std::forward<Op>(op))(*GetDerived());
  }

  template <
      typename Op,
      typename = std::enable_if_t<!std::is_member_function_pointer<Op>::value>,
      typename = void>
  void CallOnDerived(Op&& op) {
    op(GetDerived());
  }

  template <typename Op>
  ValueOnStack LoadOp(const ValType& type, uint32_t offset, uint32_t alignment,
                      Op&& op) {
    std::mem_fn(op)(*GetDerived(), offset, alignment);
    return MakeValueOnStack(type);
  }

  const Derived* GetDerived() const {
    return static_cast<const Derived*>(this);
  }

  Derived* GetMutableDerived() { return static_cast<Derived*>(this); }

  ValueOnStack MakeValueOnStack(const ValType& type) {
    return {type, GetMutableDerived()};
  }
};

class WasmOps : public LocalWasmOps<WasmOps>,
                public I32WasmOps<WasmOps>,
                public F32WasmOps<WasmOps>,
                public V128WasmOps<WasmOps>,
                public MemoryWasmOps<WasmOps>,
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
  using ResultType = internal::ResultType;

 public:
  explicit WasmAssembler(xnn_code_buffer* buf) : AssemblerBase(buf) {}

  template <size_t InSize, typename Body>
  void AddFunc(const ResultType& result, const char* name,
               const std::array<ValType, InSize>& param,
               ValTypesToInt locals_declaration_count, Body&& body) {
    std::vector<byte> code;
    SetOut(&code);
    ResetLocalsManager(InSize, locals_declaration_count);
    std::array<Local, InSize> input_locals{};
    for (uint32_t index = 0; index < InSize; index++) {
      input_locals[index] =
          Local{param[index], index, /*is_managed=*/false, this};
    }
    internal::ArrayApply(std::move(input_locals), std::forward<Body>(body));
    end();
    RegisterFunction(result, name, std::vector(param.begin(), param.end()),
                     std::move(locals_declaration_count), std::move(code));
  }

  void Emit() {
    EmitMagicVersion();
    EmitTypeSection();
    EmitImportSection();
    EmitFunctionSection();
    EmitExportsSection();
    EmitCodeSection();
  }

 private:
  static constexpr std::array<byte, 4> kMagic = {0x00, 0x61, 0x73, 0x6D};
  static constexpr std::array<byte, 4> kVersion = {0x01, 0x00, 0x00, 0x00};
  static constexpr std::array<byte, 17> kImportSection = {
      0x02, 0x0F, 0x01, 0x03, 0x65, 0x6E, 0x76, 0x06, 0x6D,
      0x65, 0x6D, 0x6F, 0x72, 0x79, 0x02, 0x00, 0x00};

  constexpr static byte kTypeSectionCode = 0x01;
  constexpr static byte kFunctionSectionCode = 0x03;
  constexpr static byte kFunctionExportCode = 0x0;
  constexpr static byte kExportsSectionCode = 0x07;
  constexpr static byte kCodeSectionCode = 0x0A;

  void RegisterFunction(const ResultType& result, const char* name,
                        std::vector<ValType>&& param,
                        ValTypesToInt&& locals_declaration_count,
                        std::vector<byte> code);
  uint32_t FindOrAddFuncType(FuncType&& type);

  template <typename Array>
  void EmitByteArray(Array&& array) {
    for (byte b : array) {
      emit8(b);
    }
  }

  template <typename Array, typename SizeCalculator, typename EmitElement>
  void EmitSection(byte section_code, Array&& array,
                   SizeCalculator&& size_calculator,
                   EmitElement&& emit_element) {
    emit8(section_code);
    EmitEncodedU32(VectorEncodingLength(
        array, std::forward<SizeCalculator>(size_calculator)));
    EmitEncodedU32(array.size());
    for (const auto& element : array) emit_element(element);
  }

  void EmitMagicVersion();

  void EmitTypeSection();
  void EmitFuncType(const FuncType& type);
  void EmitParamType(const std::vector<ValType>& type);
  void EmitResultType(const ResultType& type);

  void EmitImportSection();

  void EmitFunctionSection();

  void EmitExportsSection();
  void EmitExport(const Export& exp);

  void EmitCodeSection();
  void EmitFunction(const Function& func);

  void EmitEncodedU32(uint32_t n) {
    internal::StoreEncodedU32(n, [this](byte b) { emit8(b); });
  }

  std::vector<Function> functions_;
  std::vector<FuncType> func_types_;
  std::vector<Export> exports_;
};

}  // namespace xnnpack
