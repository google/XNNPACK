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
#include <ostream>

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
template <class T, size_t Bits = 8 * sizeof(T)>
struct NativeStorageImpl {
  static_assert((Bits & (Bits - 1)) == 0, "Bits must be a power of 2.");
  using type = T;
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

template <>
struct NativeStorage<Type::kI2>
    : internal::NativeStorageImpl<int8_t, /*Bits=*/2> {};

template <>
struct NativeStorage<Type::kI4>
    : internal::NativeStorageImpl<int8_t, /*Bits=*/4> {};

template <>
struct NativeStorage<Type::kI8> : internal::NativeStorageImpl<int8_t> {};

template <>
struct NativeStorage<Type::kI16> : internal::NativeStorageImpl<int16_t> {};

template <>
struct NativeStorage<Type::kI32> : internal::NativeStorageImpl<int32_t> {};

template <>
struct NativeStorage<Type::kI64> : internal::NativeStorageImpl<int64_t> {};

template <>
struct NativeStorage<Type::kU4>
    : internal::NativeStorageImpl<uint8_t, /*Bits=*/4> {};

template <>
struct NativeStorage<Type::kU8> : internal::NativeStorageImpl<uint8_t> {};

template <>
struct NativeStorage<Type::kU16> : internal::NativeStorageImpl<uint16_t> {};

template <>
struct NativeStorage<Type::kU32> : internal::NativeStorageImpl<uint32_t> {};

template <>
struct NativeStorage<Type::kU64> : internal::NativeStorageImpl<uint64_t> {};

template <>
struct NativeStorage<Type::kBF16> : internal::NativeStorageImpl<uint16_t> {};

template <>
struct NativeStorage<Type::kFP16> : internal::NativeStorageImpl<uint16_t> {};

template <>
struct NativeStorage<Type::kFP32> : internal::NativeStorageImpl<float> {};

template <>
struct NativeStorage<Type::kFP64> : internal::NativeStorageImpl<double> {};

template <>
struct NativeStorage<Type::kBOOL> : internal::NativeStorageImpl<bool> {};

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
