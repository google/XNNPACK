/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef LITERT_TENSOR_DATATYPES_H_
#define LITERT_TENSOR_DATATYPES_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <type_traits>

#include "litert/tensor/internal/fp16.h"

namespace litert::tensor {

enum class Type {
  kUnknown = 0,
  kBOOL,
  kI2,
  kI4,
  kI8,
  kI16,
  kI32,
  kI64,
  kU4,
  kU8,
  kU16,
  kU32,
  kU64,
  kFP16,
  kFP32,
  kFP64,
  kBF16,
};

template <Type>
struct NativeStorage;

namespace internal {

// Implements the NativeStorage interface.
//
// - T: the native type used to store data.
// - Bits: a power of 2, for data that is smaller than 1 byte.
template <Type t, class T, size_t Bits = 8 * sizeof(T)>
struct StorageImpl {
  static_assert((Bits & (Bits - 1)) == 0, "Bits must be a power of 2.");
  using type = T;
  static constexpr Type value = t;
  static constexpr size_t BufferSize(size_t count) {
    return (Bits * count + 7) / 8;
  }
};

}  // namespace internal

// We're defining this for consistency. Actually using for anything else is an
// error.
template <>
struct NativeStorage<Type::kUnknown> {
  using type = void;
  static constexpr size_t BufferSize(size_t count) { return 0; }
};

struct int2_t {
  int8_t a : 2;
  int8_t b : 2;
  int8_t c : 2;
  int8_t d : 2;
};

static_assert(sizeof(int2_t) == sizeof(int8_t));
static_assert(alignof(int2_t) == alignof(int8_t));

struct int4_t {
  int8_t a : 4;
  int8_t b : 4;
};

static_assert(sizeof(int4_t) == sizeof(int8_t));
static_assert(alignof(int4_t) == alignof(int8_t));

struct uint4_t {
  uint8_t a : 4;
  uint8_t b : 4;
};

static_assert(sizeof(uint4_t) == sizeof(uint8_t));
static_assert(alignof(uint4_t) == alignof(uint8_t));

struct bf16_t {
  constexpr bf16_t(const bf16_t& v) = default;
  // NOLINTNEXTLINE(*-explicit-constructor): bf16_t can be built from a float.
  constexpr bf16_t(float v)
      : val(__builtin_bit_cast(uint32_t, v* kRoundingMultiplier) >> 16) {}
  // NOLINTNEXTLINE(*-explicit-constructor): bf16_t can be converted to a float.
  constexpr operator float() const {
    return __builtin_bit_cast(float, static_cast<uint32_t>(val) << 16);
  }
  friend constexpr bool operator==(bf16_t a, bf16_t b) {
    return a.val == b.val;
  }
  friend constexpr bool operator==(bf16_t a, float b) { return a == bf16_t(b); }
  friend constexpr bool operator==(float a, bf16_t b) { return bf16_t(a) == b; }
  uint16_t val;

  // When rounding a float to bfloat16, we want to add 1 to the bit after the
  // last bit in the mantissa. We can make the floating point hardware do this
  // for us, by multiplying by 1 + 0.5*epsilon:
  // a*rounding_multiplier = a*(1 + 0.5*epsilon) = a + a*0.5*epsilon.
  // 0.5*epsilon is a power of 2, so this is effectively moving the leading 1
  // from the mantissa to the bit after the last bit of the mantissa, and then
  // adding it.
  static constexpr float kRoundingMultiplier = 1.0f + 0.5f / 128.0f;
};

static_assert(bf16_t(1) == 1.0f);
static_assert(bf16_t(2) == 2.0f);

static_assert(sizeof(bf16_t) == sizeof(int16_t));
static_assert(alignof(bf16_t) == alignof(int16_t));

struct fp16_t {
  constexpr fp16_t() : val(0) {}
  // NOLINTNEXTLINE(*-explicit-constructor): fp16_t can be built from a float.
  constexpr fp16_t(float f) : val(fp16_ieee_from_fp32_value(f)) {}
  // NOLINTNEXTLINE(*-explicit-constructor): fp16_t can be converted to a float.
  constexpr operator float() const { return fp16_ieee_to_fp32_value(val); }

  uint16_t val;
};

static_assert(sizeof(fp16_t) == sizeof(int16_t));
static_assert(alignof(fp16_t) == alignof(int16_t));

template <>
struct NativeStorage<Type::kI2>
    : internal::StorageImpl<Type::kI2, int2_t, /*Bits=*/2> {};

template <>
struct NativeStorage<Type::kI4>
    : internal::StorageImpl<Type::kI4, int4_t, /*Bits=*/4> {};

template <>
struct NativeStorage<Type::kI8> : internal::StorageImpl<Type::kI8, int8_t> {};

template <>
struct NativeStorage<Type::kI16> : internal::StorageImpl<Type::kI16, int16_t> {
};

template <>
struct NativeStorage<Type::kI32> : internal::StorageImpl<Type::kI32, int32_t> {
};

template <>
struct NativeStorage<Type::kI64> : internal::StorageImpl<Type::kI64, int64_t> {
};

template <>
struct NativeStorage<Type::kU4>
    : internal::StorageImpl<Type::kU4, uint4_t, /*Bits=*/4> {};

template <>
struct NativeStorage<Type::kU8> : internal::StorageImpl<Type::kU8, uint8_t> {};

template <>
struct NativeStorage<Type::kU16> : internal::StorageImpl<Type::kU16, uint16_t> {
};

template <>
struct NativeStorage<Type::kU32> : internal::StorageImpl<Type::kU32, uint32_t> {
};

template <>
struct NativeStorage<Type::kU64> : internal::StorageImpl<Type::kU64, uint64_t> {
};

template <>
struct NativeStorage<Type::kBF16> : internal::StorageImpl<Type::kBF16, bf16_t> {
};

template <>
struct NativeStorage<Type::kFP16> : internal::StorageImpl<Type::kFP16, fp16_t> {
};

template <>
struct NativeStorage<Type::kFP32> : internal::StorageImpl<Type::kFP32, float> {
};

template <>
struct NativeStorage<Type::kFP64> : internal::StorageImpl<Type::kFP64, double> {
};

template <>
struct NativeStorage<Type::kBOOL> : internal::StorageImpl<Type::kBOOL, bool> {};

template <class T>
struct ApiType;

template <>
struct ApiType<int2_t> : internal::StorageImpl<Type::kI2, int2_t, /*Bits=*/2> {
};

template <>
struct ApiType<int4_t> : internal::StorageImpl<Type::kI4, int4_t, /*Bits=*/4> {
};

template <>
struct ApiType<int8_t> : internal::StorageImpl<Type::kI8, int8_t> {};

template <>
struct ApiType<int16_t> : internal::StorageImpl<Type::kI16, int16_t> {};

template <>
struct ApiType<int32_t> : internal::StorageImpl<Type::kI32, int32_t> {};

template <>
struct ApiType<int64_t> : internal::StorageImpl<Type::kI64, int64_t> {};

template <>
struct ApiType<uint4_t>
    : internal::StorageImpl<Type::kU4, uint4_t, /*Bits=*/4> {};

template <>
struct ApiType<uint8_t> : internal::StorageImpl<Type::kU8, uint8_t> {};

template <>
struct ApiType<uint16_t> : internal::StorageImpl<Type::kU16, uint16_t> {};

template <>
struct ApiType<uint32_t> : internal::StorageImpl<Type::kU32, uint32_t> {};

template <>
struct ApiType<uint64_t> : internal::StorageImpl<Type::kU64, uint64_t> {};

template <>
struct ApiType<bf16_t> : internal::StorageImpl<Type::kBF16, bf16_t> {};

template <>
struct ApiType<fp16_t> : internal::StorageImpl<Type::kFP16, fp16_t> {};

template <>
struct ApiType<float> : internal::StorageImpl<Type::kFP32, float> {};

template <>
struct ApiType<double> : internal::StorageImpl<Type::kFP64, double> {};

template <>
struct ApiType<bool> : internal::StorageImpl<Type::kBOOL, bool> {};

template <Type out, Type in, class = void>
struct Convertion {
  static constexpr typename NativeStorage<out>::type Call(
      typename NativeStorage<in>::type val) {
    return static_cast<typename NativeStorage<out>::type>(val);
  }
};

template <Type inout>
struct Convertion<inout, inout, void> {
  static constexpr typename NativeStorage<inout>::type Call(
      typename NativeStorage<inout>::type val) {
    return val;
  }
};

template <Type type>
struct Convertion<
    Type::kBF16, type,
    std::enable_if_t<type != Type::kFP32 && type != Type::kBF16>> {
  static constexpr typename NativeStorage<Type::kBF16>::type Call(
      typename NativeStorage<type>::type val) {
    return Convertion<Type::kBF16, Type::kFP32>::Call(static_cast<float>(val));
  }
};

template <Type type>
struct Convertion<
    type, Type::kBF16,
    std::enable_if_t<type != Type::kFP32 && type != Type::kBF16>> {
  static constexpr typename NativeStorage<type>::type Call(
      typename NativeStorage<Type::kBF16>::type val) {
    return Convertion<type, Type::kFP32>::Call(static_cast<float>(val));
  }
};

template <Type type>
struct Convertion<Type::kFP16, type,
                  std::enable_if_t<type != Type::kFP32 && type != Type::kFP16 &&
                                   type != Type::kBF16>> {
  static constexpr typename NativeStorage<Type::kFP16>::type Call(
      typename NativeStorage<type>::type val) {
    return Convertion<Type::kFP16, Type::kFP32>::Call(static_cast<float>(val));
  }
};

template <Type type>
struct Convertion<type, Type::kFP16,
                  std::enable_if_t<type != Type::kFP32 && type != Type::kFP16 &&
                                   type != Type::kBF16>> {
  static constexpr typename NativeStorage<type>::type Call(
      typename NativeStorage<Type::kFP16>::type val) {
    return Convertion<type, Type::kFP32>::Call(static_cast<float>(val));
  }
};

template <Type to, class T>
auto ConvertTo(T value) {
  constexpr Type from = ApiType<T>::value;
  return Convertion<to, from>(value);
}

inline const char* ToString(Type t) {
#define LITERT_TENSOR_TYPE_TO_STRING_CASE(name) \
  case Type::k##name:                           \
    return #name
  switch (t) {
    LITERT_TENSOR_TYPE_TO_STRING_CASE(Unknown);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(BOOL);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I2);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I4);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I8);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I16);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I32);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I64);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U4);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U8);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U16);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U32);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U64);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(FP16);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(FP32);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(FP64);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(BF16);
  }
#undef LITERT_TENSOR_TYPE_TO_STRING_CASE
  // This return should never be reached.
  return "ERROR: litert::tensor::ToString(Type) failed.";
}

template <class Sink>
void AbslStringify(Sink& sink, Type t) {
  sink.Append(ToString(t));
}

inline std::ostream& operator<<(std::ostream& os, const Type t) {
  return os << ToString(t);
}

inline constexpr size_t BufferSize(Type t, size_t count) {
#define LITERT_TENSOR_TYPE_BUFFER_SIZE(name) \
  case Type::k##name:                        \
    return NativeStorage<Type::k##name>::BufferSize(count);
  switch (t) {
    LITERT_TENSOR_TYPE_BUFFER_SIZE(Unknown);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(BOOL);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I2);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I4);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I8);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I16);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I32);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(I64);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(U4);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(U8);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(U16);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(U32);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(U64);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(FP16);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(FP32);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(FP64);
    LITERT_TENSOR_TYPE_BUFFER_SIZE(BF16);
  }
#undef LITERT_TENSOR_TYPE_BUFFER_SIZE
  return 0;
}

}  // namespace litert::tensor

#endif  // LITERT_TENSOR_DATATYPES_H_
