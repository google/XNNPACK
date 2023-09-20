// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <type_traits>
#include <utility>
#include <xnnpack/array-helpers.h>
#include <xnnpack/assembler.h>
#include <xnnpack/common.h>
#include <xnnpack/leb128.h>


namespace xnnpack {

struct ValType {
  ValType() = delete;
  ValType(const ValType&) = default;
  ValType& operator=(const ValType&) = default;
  constexpr explicit ValType(byte code) : code(code) {}
  byte code{};
};

inline bool operator==(const ValType& lhs, const ValType& rhs) {
  return lhs.code == rhs.code;
}

namespace internal {

static constexpr ValType kPlaceholderValType(0);

struct ValTypeToInt {
  template <typename Int>
  constexpr ValTypeToInt(ValType type, Int value) : type(type), value(value) {}
  constexpr ValTypeToInt() : ValTypeToInt(kPlaceholderValType, 0) {}
  ValType type;
  uint32_t value;
};

}  // namespace internal

static constexpr uint32_t kI32DefaultAlignment = 1;
static constexpr uint32_t kV128DefaultAlignment = 1;
static constexpr size_t kMaxNumTypes = 5;

class ValTypesToInt : public internal::ArrayPrefix<internal::ValTypeToInt, kMaxNumTypes> {
 public:
  using ArrayPrefix::ArrayPrefix;
  constexpr ValTypesToInt() : ArrayPrefix(0, {ValType(0), 0}) {}
};

namespace internal {
template <typename Array, typename ElementEncodingLength>
static uint32_t VectorEncodingLength(Array&& array, ElementEncodingLength&& element_encoding_length) {
  const auto add_encoding_length = [&](uint32_t acc, const auto& element) {
    return acc + element_encoding_length(element);
  };
  const uint32_t total_length_of_element_encodings =
    std::accumulate(array.begin(), array.end(), uint32_t{0}, add_encoding_length);
  return WidthEncodedU32(array.size()) + total_length_of_element_encodings;
}

struct ResultType {
  constexpr ResultType() : type(kNoTypeCode) {}
  constexpr ResultType(std::initializer_list<ValType> codes) : ResultType() {
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

static constexpr size_t kMaxParamsCount = 16;
struct Params : ArrayPrefix<ValType, kMaxParamsCount> {
  using ArrayPrefix::ArrayPrefix;
  constexpr Params() : Params(0, kPlaceholderValType){};
};

struct FuncType {
  constexpr FuncType() = default;
  constexpr FuncType(const Params& params, ResultType result) : params(params), result(result) {}
  Params params;
  ResultType result;
};

inline bool operator==(const FuncType& lhs, const FuncType& rhs) {
  return lhs.result == rhs.result &&
         std::equal(lhs.params.begin(), lhs.params.end(), rhs.params.begin(), rhs.params.end());
}

struct Code {
  constexpr explicit Code(size_t begin_offset) : begin_offset(begin_offset), end_offset(begin_offset) {}

  size_t size() const { return end_offset - begin_offset; }

  size_t begin_offset;
  size_t end_offset;
};

struct Function {
  constexpr Function() : Function(nullptr, 0, 0, 0, {}, Code(0)) {}
  constexpr Function(const char* name, size_t function_index, uint32_t type_index,
                     const ValTypesToInt& locals_declaration, Code body)
      : Function(name, strlen(name), function_index, type_index, locals_declaration, body) {}
  constexpr Function(const char* name, size_t name_length, size_t function_index, uint32_t type_index,
                     const ValTypesToInt& locals_declaration, Code body)
      : name(name),
        name_length(name_length),
        function_index(function_index),
        type_index(type_index),
        locals_declaration(locals_declaration),
        body(body) {}
  const char* name;
  size_t name_length;
  size_t function_index;
  uint32_t type_index;
  ValTypesToInt locals_declaration;
  Code body;
};

template <typename Derived, typename Intermediate>
class WasmOpsBase {
 public:
  void Encode8(byte b) { GetDerived()->Encode8Impl(b); }

  void EncodeS32(int32_t value) { GetDerived()->EncodeS32Impl(value); }

  void EncodeU32(int32_t value) { GetDerived()->EncodeU32Impl(value); }

 protected:
  Derived* GetDerived() { return static_cast<Derived*>(this); }
};

template <typename Derived>
class I32WasmOps : public WasmOpsBase<Derived, I32WasmOps<Derived>> {
 public:
  void i32_add() { this->Encode8(0x6A); }
  void i32_sub() { this->Encode8(0x6B); }
  void i32_and() { this->Encode8(0x71); }
  void i32_lt_s() { this->Encode8(0x48); }
  void i32_le_s() { this->Encode8(0x4C); }
  void i32_ge_u() { this->Encode8(0x4F); }
  void i32_shl() { this->Encode8(0x74); }
  void i32_shr_u() { this->Encode8(0x76); }
  void i32_ne() { this->Encode8(0x47); }
  void i32_const(int32_t value) {
    this->Encode8(0x41);
    this->EncodeS32(value);
  }
};

template <typename Derived>
class F32WasmOps : public WasmOpsBase<Derived, F32WasmOps<Derived>> {
 public:
  void f32_const(float value) {
    this->Encode8(0x43);
    this->EmitEncodedF32(value);
  }

 private:
  void EmitEncodedF32(float value) {
    std::array<byte, sizeof(float)> encoding;
    memcpy(encoding.data(), &value, sizeof(float));
    for (byte b : encoding) this->Encode8(b);
  }
};

template <typename Derived>
class V128WasmOps : public WasmOpsBase<Derived, V128WasmOps<Derived>> {
 public:
  void f32x4_splat() { EncodeVectorOpcode(0x13); }
  void f32x4_mul() { EncodeVectorOpcode(0xE6); }
  void f32x4_add() { EncodeVectorOpcode(0xE4); }
  void f32x4_pmax() { EncodeVectorOpcode(0xEB); }
  void f32x4_pmin() { EncodeVectorOpcode(0xEA); }
  void f32x4_eq() { EncodeVectorOpcode(0x41); }
  void v128_andnot() { EncodeVectorOpcode(0x4F); }
  void i32x4_splat() { EncodeVectorOpcode(0x11); }
  void i32x4_max_s() { EncodeVectorOpcode(0xB8); }
  void i8x16_shuffle(const std::array<uint8_t, 16>& lanes) {
    EncodeVectorOpcode(0x0D);
    for (auto lane : lanes) {
      assert(lane < 32);
      this->Encode8(lane);
    }
  }

#if XNN_ARCH_WASMRELAXEDSIMD
  void f32x4_relaxed_wasmsimd() { EncodeVectorOpcode(0x105); }
  void f32x4_relaxed_max() { EncodeVectorOpcode(0x10E); }
  void f32x4_relaxed_min() { EncodeVectorOpcode(0x10D); }
#endif

  void v128_const(std::array<byte, 16>& values) {
    EncodeVectorOpcode(0x0C);
    for (byte v : values) {
      this->Encode8(v);
    }
  }

  void EncodeVectorOpcodePrefix() { this->Encode8(0xFD); }

 private:
  void EncodeVectorOpcode(uint32_t code) {
    EncodeVectorOpcodePrefix();
    this->EncodeU32(code);
  }
};

template <typename Derived>
class ControlFlowWasmOps : public WasmOpsBase<Derived, ControlFlowWasmOps<Derived>> {
 public:
  template <typename Cond, typename If, typename Else>
  void IfElse(Cond&& cond, If&& if_block, Else&& else_block) {
    cond();
    this->Encode8(kIfCode);
    this->Encode8(kEpsilonCode);  // Fallthru elements are not supported
    if_block();
    this->Encode8(kElseCode);
    else_block();
    end();
  }

  template <typename Cond, typename IfBlock>
  void If(Cond&& cond, IfBlock&& if_block) {
    cond();
    this->Encode8(kIfCode);
    this->Encode8(kEpsilonCode);  // Fallthru elements are not supported
    if_block();
    end();
  }

  template <typename Cond, typename Body>
  void DoWhile(Body&& body, Cond&& cond) {
    this->Encode8(kLoopCode);
    this->Encode8(kEpsilonCode);  // Fallthru elements are not supported
    body();
    cond();
    this->Encode8(kBrIfCode);
    this->EncodeU32(0);
    end();
  }

  template <typename Cond, typename Body>
  void While(Cond&& cond, Body&& body) {
    If(cond, [&] { DoWhile(std::forward<Body>(body), cond); });
  }

  void end() { this->Encode8(0x0B); }
  void Return() { this->Encode8(0x0F); }

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
  void i32_load(uint32_t offset = 0, uint32_t alignment = kI32DefaultAlignment) {
    load_or_store(0x28, offset, alignment);
  }

  void i32_store(uint32_t offset = 0, uint32_t alignment = kI32DefaultAlignment) {
    load_or_store(0x36, offset, alignment);
  }

  void v128_load(uint32_t offset = 0, uint32_t alignment = kV128DefaultAlignment) {
    vector_load_or_store(0x00, offset, alignment);
  }

  void v128_load32_splat(uint32_t offset = 0, uint32_t alignment = kV128DefaultAlignment) {
    vector_load_or_store(0x09, offset, alignment);
  }

  void v128_load64_splat(uint32_t offset = 0, uint32_t alignment = kV128DefaultAlignment) {
    vector_load_or_store(0x0A, offset, alignment);
  }

  void v128_store(uint32_t offset = 0, uint32_t alignment = kV128DefaultAlignment) {
    vector_load_or_store(0x0B, offset, alignment);
  }

  void v128_store64_lane(uint8_t lane, uint32_t offset = 0, uint32_t alignment = kV128DefaultAlignment) {
    v128_store_lane(0x5B, lane, /*max_lane=*/4, offset, alignment);
  }

  void v128_store32_lane(uint8_t lane, uint32_t offset = 0, uint32_t alignment = kV128DefaultAlignment) {
    v128_store_lane(0x5A, lane, /*max_lane=*/8, offset, alignment);
  }

 private:
  void load_or_store(byte opcode, uint32_t offset, uint32_t alignment) {
    this->EncodeU32(opcode);
    this->EncodeU32(log2(alignment));
    this->EncodeU32(offset);
  }

  void vector_load_or_store(byte opcode, uint32_t offset, uint32_t alignment) {
    this->GetDerived()->EncodeVectorOpcodePrefix();
    load_or_store(opcode, offset, alignment);
  }

  void v128_store_lane(byte opcode, uint8_t lane, uint8_t max_lane, uint32_t offset, uint32_t alignment) {
    assert(lane < max_lane);
    vector_load_or_store(opcode, offset, alignment);
    this->Encode8(lane);
  }
};

class LocalsManager {
 public:
  void ResetLocalsManager(uint32_t parameters_count, const ValTypesToInt& locals_declaration_count);

  uint32_t GetNewLocalIndex(ValType type);

  void DestructLocal(ValType type, uint32_t index);

 private:
  struct ValTypeIndices {
    static constexpr size_t kMaxLocalsOfSameType = 512;

    ValTypeIndices() : type(0), first_index(0), size(0) {}
    explicit ValTypeIndices(ValType type, uint32_t first_index, uint32_t size)
        : type(type), first_index(first_index), size(size) {}

    uint32_t GetNewIndex();
    void DestroyLocal(uint32_t index);

    ValType type;
    std::bitset<kMaxLocalsOfSameType> bitset;
    uint32_t first_index;
    uint32_t size;
  };

  std::array<ValTypeIndices, kMaxNumTypes> indices_ =
    MakeArray<kMaxNumTypes>(ValTypeIndices{kPlaceholderValType, 0, 0});
};

std::array<uint8_t, 16> MakeLanesForI8x16Shuffle(const uint8_t* lanes, size_t num_lanes);

template <typename Derived>
class LocalWasmOps : public LocalsManager, public WasmOpsBase<Derived, LocalWasmOps<Derived>> {
  using WasmOpsBase<Derived, LocalWasmOps<Derived>>::GetDerived;

 public:
  class Local;

  struct ValueOnStack {
    ValueOnStack(const ValType type, Derived* ops) : type(type), ops(ops) {}
    ValueOnStack(const Local& local) : type(local.type_), ops(local.ops_) { ops->local_get(local); }  // NOLINT
    ValType type;
    Derived* ops;
  };

  class Local {
   public:
    constexpr Local() = default;

    Local(const ValType& type, uint32_t index, bool is_managed, Derived* ops)
        : type_(type), index_(index), ops_(ops), is_managed_(is_managed) {}

    Local(const Local& other) = delete;

    Local(Local&& other) : type_(other.type_), index_(other.index_), ops_(other.ops_), is_managed_(other.is_managed_) {
      other.index_ = kInvalidIndex;
    }

    Local& operator=(const Local& other) {
      assert((type_ == other.type_) && "Assignment of locals of different type");
      ops_ = other.ops_;
      ops_->local_get(other.index_);
      ops_->local_set(index_);
      return *this;
    }

    Local& operator=(Local&& other) {
      assert((index_ == kInvalidIndex) && "The local already binds to something");
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
      if (is_managed_ && index_ != kInvalidIndex) ops_->DestructLocal(type_, index_);
    }

    ValType type_{0};
    uint32_t index_{kInvalidIndex};
    Derived* ops_ = nullptr;

   private:
    bool is_managed_ = false;
    static constexpr uint32_t kInvalidIndex = -1;
  };

  constexpr static size_t kMaxLocalsSize = 32;

  using LocalsArray = ArrayPrefix<Local, kMaxLocalsSize>;

  Local MakeLocal(ValType type) { return Local{type, GetNewLocalIndex(type), /*is_managed=*/true, GetDerived()}; }

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

  void local_get(uint32_t index) { Encode8AndU32(0x20, index); }

  void local_set(uint32_t index) { Encode8AndU32(0x21, index); }

  void local_tee(uint32_t index) { Encode8AndU32(0x22, index); }

  void local_get(const Local& local) { local_get(local.index_); }

  void local_set(const Local& local) { local_set(local.index_); }

  void local_tee(const Local& local) { local_tee(local.index_); }

  ValueOnStack I32Add(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::i32_add); }

  ValueOnStack I32Sub(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::i32_sub); }

  ValueOnStack I32And(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::i32_and); }

  ValueOnStack I32LtS(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::i32_lt_s); }

  ValueOnStack I32LeS(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::i32_le_s); }

  ValueOnStack I32GeU(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::i32_ge_u); }

  ValueOnStack I32Shl(const ValueOnStack& value, const ValueOnStack& bits_num) {
    return BinaryOp(value, bits_num, &Derived::i32_shl);
  }

  ValueOnStack I32Ne(const ValueOnStack& lhs, const ValueOnStack& rhs) { return BinaryOp(lhs, rhs, &Derived::i32_ne); }

  ValueOnStack I32NeZ(const ValueOnStack& value) { return I32Ne(value, I32Const(0)); }

  ValueOnStack I32ShrU(const ValueOnStack& value, const ValueOnStack& bits_num) {
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

  ValueOnStack I32Load(const ValueOnStack& address, uint32_t offset = 0, uint32_t alignment = kI32DefaultAlignment) {
    return LoadOp(i32, offset, alignment, &Derived::i32_load);
  }

  ValueOnStack I32Load(const ValueOnStack& base, const ValueOnStack& dynamic_offset, uint32_t static_offset = 0,
                       uint32_t alignment = kI32DefaultAlignment) {
    return I32Load(I32Add(base, I32Shl(dynamic_offset, I32Const(2))), static_offset, alignment);
  }

  void I32Store(const ValueOnStack& address, const ValueOnStack& value, uint32_t offset = 0,
                uint32_t alignment = kI32DefaultAlignment) {
    GetDerived()->i32_store(offset, alignment);
  }

  void I32Store(const ValueOnStack& base, const ValueOnStack& dynamic_offset, const Local& value,
                uint32_t static_offset = 0, uint32_t alignment = kI32DefaultAlignment) {
    I32Store(I32Add(base, I32Shl(dynamic_offset, I32Const(2))), value, static_offset, alignment);
  }

  ValueOnStack F32x4Add(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::f32x4_add); }

  ValueOnStack F32x4Pmax(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::f32x4_pmax); }

  ValueOnStack F32x4Pmin(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::f32x4_pmin); }

  ValueOnStack F32x4Eq(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::f32x4_eq); }

#if XNN_ARCH_WASMRELAXEDSIMD
  ValueOnStack F32x4RelaxedMadd(const ValueOnStack& a, const ValueOnStack& b, const ValueOnStack& c) {
    GetDerived()->f32x4_relaxed_wasmsimd();
    return MakeValueOnStack(v128);
  }

  ValueOnStack F32x4RelaxedMax(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::f32x4_relaxed_max);
  }

  ValueOnStack F32x4RelaxedMin(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::f32x4_relaxed_min);
  }
#endif

  ValueOnStack V128Andnot(const ValueOnStack& a, const ValueOnStack& b) {
    return BinaryOp(a, b, &Derived::v128_andnot);
  }

  ValueOnStack I64x2Shuffle(const ValueOnStack& a, const ValueOnStack& b, const std::array<uint8_t, 2>& lanes) {
    return BinaryOp(
      a, b, [&](Derived* derived) { derived->i8x16_shuffle(MakeLanesForI8x16Shuffle(lanes.data(), lanes.size())); });
  }

  ValueOnStack I32x4Shuffle(const ValueOnStack& a, const ValueOnStack& b, const std::array<uint8_t, 4>& lanes) {
    return BinaryOp(
      a, b, [&](Derived* derived) { derived->i8x16_shuffle(MakeLanesForI8x16Shuffle(lanes.data(), lanes.size())); });
  }

  ValueOnStack F32x4Splat(const ValueOnStack& a) {
    assert(a.type == f32);
    GetDerived()->f32x4_splat();
    return MakeValueOnStack(v128);
  }

  ValueOnStack I32x4Splat(const ValueOnStack& a) {
    assert(a.type == i32);
    GetDerived()->i32x4_splat();
    return MakeValueOnStack(v128);
  }

  ValueOnStack I32x4MaxS(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::i32x4_max_s); }

  ValueOnStack F32x4Mul(const ValueOnStack& a, const ValueOnStack& b) { return BinaryOp(a, b, &Derived::f32x4_mul); }

  ValueOnStack V128Const(float value) {
    const uint8_t* value_int = reinterpret_cast<uint8_t*>(&value);
    std::array<byte, 16> values;
    for (size_t offset = 0; offset < 16 / sizeof(float); offset++) {
      std::copy(value_int, value_int + sizeof(float), values.data() + offset * sizeof(float));
    }
    GetDerived()->v128_const(values);
    return MakeValueOnStack(v128);
  }

  ValueOnStack V128Load(const ValueOnStack& address, uint32_t offset = 0, uint32_t alignment = kV128DefaultAlignment) {
    return LoadOp(v128, offset, alignment, &Derived::v128_load);
  }

  ValueOnStack V128Load32Splat(const ValueOnStack& address, uint32_t offset = 0,
                               uint32_t alignment = kV128DefaultAlignment) {
    return LoadOp(v128, offset, alignment, &Derived::v128_load32_splat);
  }

  ValueOnStack V128Load64Splat(const ValueOnStack& address, uint32_t offset = 0,
                               uint32_t alignment = kV128DefaultAlignment) {
    return LoadOp(v128, offset, alignment, &Derived::v128_load64_splat);
  }

  void V128Store(const ValueOnStack& address, const ValueOnStack& value, uint32_t offset = 0,
                 uint32_t alignment = kV128DefaultAlignment) {
    GetDerived()->v128_store(offset, alignment);
  }

  void V128Store64Lane(const ValueOnStack& address, const ValueOnStack& value, uint8_t lane, uint32_t offset = 0,
                       uint32_t alignment = kV128DefaultAlignment) {
    GetDerived()->v128_store64_lane(lane, offset, alignment);
  }

  void V128Store32Lane(const ValueOnStack& address, const ValueOnStack& value, uint8_t lane, uint32_t offset = 0,
                       uint32_t alignment = kV128DefaultAlignment) {
    GetDerived()->v128_store32_lane(lane, offset, alignment);
  }

  ValueOnStack Select(const ValueOnStack& if_true, const ValueOnStack& if_false, const ValueOnStack& cond) {
    this->Encode8(0x1B);
    return MakeValueOnStack(if_true.type);
  }

  static constexpr ValType i32{0x7F};
  static constexpr ValType f32{0x7D};
  static constexpr ValType v128{0x7B};

 private:
  template <typename Op>
  ValueOnStack BinaryOp(const ValueOnStack& a, const ValueOnStack& b, Op&& op) {
    assert((a.type == b.type) && "Binary operation on locals of different types");
    CallOnDerived(std::forward<Op>(op));
    return MakeValueOnStack(a.type);
  }

  template <typename Op, typename = std::enable_if_t<std::is_member_function_pointer<Op>::value>>
  void CallOnDerived(Op&& op) {
    std::mem_fn(std::forward<Op>(op))(*GetDerived());
  }

  template <typename Op, typename = std::enable_if_t<!std::is_member_function_pointer<Op>::value>, typename = void>
  void CallOnDerived(Op&& op) {
    op(GetDerived());
  }

  template <typename Op>
  ValueOnStack LoadOp(const ValType& type, uint32_t offset, uint32_t alignment, Op&& op) {
    std::mem_fn(op)(*GetDerived(), offset, alignment);
    return MakeValueOnStack(type);
  }

  void Encode8AndU32(byte b, uint32_t value) {
    this->Encode8(b);
    this->EncodeU32(value);
  }

  ValueOnStack MakeValueOnStack(const ValType& type) { return {type, GetDerived()}; }
};

}  // namespace internal

class WasmAssembler : public AssemblerBase,
                      public internal::LocalWasmOps<WasmAssembler>,
                      public internal::I32WasmOps<WasmAssembler>,
                      public internal::F32WasmOps<WasmAssembler>,
                      public internal::V128WasmOps<WasmAssembler>,
                      public internal::MemoryWasmOps<WasmAssembler>,
                      public internal::ControlFlowWasmOps<WasmAssembler> {
 private:
  using Function = internal::Function;
  using Params = internal::Params;
  using FuncType = internal::FuncType;
  using LocalsManager = internal::LocalsManager;
  using ResultType = internal::ResultType;
  using Code = internal::Code;

  auto MakeStoreToCode(Code& out) {
    return [&](uint8_t b) { emit8(b, &out.end_offset); };
  }

 public:
  explicit WasmAssembler(xnn_code_buffer* buf)
      : AssemblerBase(buf), next_func_body_begin_(code_size_in_bytes() + kInitialCodeOffset) {}

  template <size_t InSize, typename Body>
  void AddFunc(const ResultType& result, const char* name, ValTypesToInt locals_declaration_count, Body&& body) {
    const auto params = internal::MakeArray<InSize>(i32);
    AddFunc<InSize>(result, name, params, locals_declaration_count, std::forward<Body>(body));
  }

  template <size_t InSize, typename Body>
  void AddFunc(const ResultType& result, const char* name, const std::array<ValType, InSize>& params,
               const ValTypesToInt& locals_declaration_count, Body&& body) {
    if (functions_.size() == kMaxNumFuncs) {
      error_ = Error::kMaxNumberOfFunctionsExceeded;
      return;
    }
    Code code(next_func_body_begin_);
    SetOut(&code);
    ResetLocalsManager(InSize, locals_declaration_count);
    std::array<Local, InSize> input_locals{};
    for (uint32_t index = 0; index < InSize; index++) {
      input_locals[index] = Local{params[index], index, /*is_managed=*/false, this};
    }
    internal::ArrayApply(std::move(input_locals), std::forward<Body>(body));
    end();

    if (error_ != Error::kNoError) {
      return;
    }
    RegisterFunction(result, name, Params(params, internal::kPlaceholderValType), locals_declaration_count, code);
    next_func_body_begin_ = code.end_offset;
  }

  void Emit() {
    EmitMagicVersion();
    EmitTypeSection();
    EmitImportSection();
    EmitFunctionSection();
    EmitExportsSection();
    assert(code_size_in_bytes() < kInitialCodeOffset && "Initial code offset is insufficient");
    EmitCodeSection();
  }

  void SetOut(Code* out) { out_ = out; }
  void Encode8Impl(byte b) { emit8(b, &out_->end_offset); }
  void EncodeS32Impl(int32_t value) { internal::StoreEncodedS32(value, MakeStoreToCode(*out_)); }
  void EncodeU32Impl(uint32_t value) { internal::StoreEncodedU32(value, MakeStoreToCode(*out_)); }

 private:
  static constexpr std::array<byte, 4> kMagic = {0x00, 0x61, 0x73, 0x6D};
  static constexpr std::array<byte, 4> kVersion = {0x01, 0x00, 0x00, 0x00};
  static constexpr std::array<byte, 17> kImportSection = {0x02, 0x0F, 0x01, 0x03, 0x65, 0x6E, 0x76, 0x06, 0x6D,
                                                          0x65, 0x6D, 0x6F, 0x72, 0x79, 0x02, 0x00, 0x00};

  constexpr static byte kTypeSectionCode = 0x01;
  constexpr static byte kFunctionSectionCode = 0x03;
  constexpr static byte kFunctionExportCode = 0x0;
  constexpr static byte kExportsSectionCode = 0x07;
  constexpr static byte kCodeSectionCode = 0x0A;

  void RegisterFunction(const ResultType& result, const char* name, const Params& params,
                        const ValTypesToInt& locals_declaration_count, Code code);
  uint32_t FindOrAddFuncType(const FuncType& type);

  template <typename Array>
  void EmitByteArray(Array&& array) {
    for (byte b : array) {
      emit8(b);
    }
  }

  template <typename Array, typename SizeCalculator, typename EmitElement>
  void EmitSection(byte section_code, Array&& array, SizeCalculator&& size_calculator, EmitElement&& emit_element) {
    emit8(section_code);
    EmitEncodedU32(VectorEncodingLength(array, std::forward<SizeCalculator>(size_calculator)));
    EmitEncodedU32(array.size());
    for (const auto& element : array) emit_element(element);
  }

  void EmitMagicVersion();

  void EmitTypeSection();
  void EmitFuncType(const FuncType& type);
  void EmitParamsType(const Params& type);
  void EmitResultType(const ResultType& type);

  void EmitImportSection();

  void EmitFunctionSection();

  void EmitExportsSection();
  void EmitExport(const Function& func);

  void EmitCodeSection();
  void EmitFunction(const Function& func);

  void EmitEncodedU32(uint32_t n) {
    internal::StoreEncodedU32(n, [this](byte b) { emit8(b); });
  }

  static constexpr size_t kMaxNumFuncTypes = 16;
  static constexpr size_t kInitialCodeOffset = 1024;
  static constexpr size_t kMaxNumFuncs = 16;

  internal::ArrayPrefix<Function, kMaxNumFuncs> functions_{0};
  internal::ArrayPrefix<FuncType, kMaxNumFuncTypes> func_types_{0};
  size_t next_func_body_begin_;
  Code* out_ = nullptr;
};

}  // namespace xnnpack
