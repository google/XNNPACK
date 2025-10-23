// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_TYPE_H_
#define XNNPACK_YNNPACK_BASE_TYPE_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

using int8x1_t = int8_t;
using uint8x1_t = uint8_t;
using int16x1_t = int16_t;
using uint16x1_t = uint16_t;
using int32x1_t = int32_t;
using uint32x1_t = uint32_t;
using float16x1_t = half;
using float32x1_t = float;

// Returns true if the type is an integer type.
bool type_is_integral(ynn_type t);

// Returns the size of an element of the type.
size_t type_size_bytes(ynn_type t);

// Returns how many elements are contained in one instance of the type. We
// assume that datatypes with a non-integer number of bytes per element can be
// represented by a struct that contains multiple elements. `type_size_bytes`
// returns the size in bytes of one instance of that struct, and this function
// returns the number of elements stored in that struct.
inline size_t type_element_count(ynn_type t) {
  switch (t) {
    case ynn_type_int4:
    case ynn_type_uint4:
      return 2;
    default:
      return 1;
  }
}

const char* to_string(ynn_type type);

inline std::ostream& operator<<(std::ostream& os, ynn_type type) {
  return os << to_string(type);
}

// Two int4 values stored in an int8.
struct int4x2 {
  uint8_t values;

  int4x2() = default;
  int4x2(uint8_t values) : values(values) {}  // NOLINT
  int4x2(int8_t x0, int8_t x1) : values((x1 << 4) | (x0 & 0x0f)) {}

  YNN_ALWAYS_INLINE int8_t get(size_t i) const {
    switch (i) {
      case 0:
        // Left shifting first implements sign extension for the lower 4 bits.
        return static_cast<int8_t>((values << 4) & 0xf0) >> 4;
      case 1:
        return static_cast<int8_t>(values & 0xf0) >> 4;
      default:
        YNN_UNREACHABLE;
    }
  }
  YNN_ALWAYS_INLINE void set(size_t i, int8_t value) {
    switch (i) {
      case 0:
        values = (values & 0xf0) | (value & 0x0f);
        return;
      case 1:
        values = (value << 4) | (values & 0x0f);
        return;
      default:
        YNN_UNREACHABLE;
    }
  }

  bool operator==(const int4x2& other) const { return values == other.values; }
  bool operator!=(const int4x2& other) const { return values != other.values; }
};

struct uint4x2 {
  uint8_t values;

  uint4x2() = default;
  uint4x2(uint8_t values) : values(values) {}  // NOLINT
  uint4x2(uint8_t x0, uint8_t x1) : values((x1 << 4) | (x0 & 0x0f)) {}

  YNN_ALWAYS_INLINE uint8_t get(size_t i) const {
    switch (i) {
      case 0:
        return static_cast<uint8_t>(values & 0x0f);
      case 1:
        return static_cast<uint8_t>(values & 0xf0) >> 4;
      default:
        YNN_UNREACHABLE;
    }
  }
  YNN_ALWAYS_INLINE void set(size_t i, uint8_t value) {
    switch (i) {
      case 0:
        values = (values & 0xf0) | (value & 0x0f);
        return;
      case 1:
        values = (value << 4) | (values & 0x0f);
        return;
      default:
        YNN_UNREACHABLE;
    }
  }

  bool operator==(const uint4x2& other) const { return values == other.values; }
  bool operator!=(const uint4x2& other) const { return values != other.values; }
};

// We need a type that distinguishes an intX_t from a quantized intX_t. We can't
// do arithmetic on these, because we don't know the quantization parameters.
template <typename T>
struct quantized {
  T value;
  using type = T;

  operator T() const { return value; }  // NOLINT
  // Forward operator[] in case T is a sub-byte packed value.
  auto operator[](size_t i) const { return value[i]; }

  quantized() = default;
  quantized(T t) : value(t) {}  // NOLINT
  quantized<T>& operator=(T t) {
    value = t;
    return *this;
  }
};

template <typename T>
struct is_quantized : std::false_type {};

template <typename T>
struct is_quantized<quantized<T>> : std::true_type {};

template <typename T>
struct unwrap_quantized {
  using type = T;
};

template <>
struct unwrap_quantized<quantized<int8_t>> {
  using type = int8_t;
};

template <>
struct unwrap_quantized<quantized<uint8_t>> {
  using type = uint8_t;
};

template <>
struct unwrap_quantized<quantized<int32_t>> {
  using type = int32_t;
};

template <typename T>
ynn_type type_of() {
  if (std::is_same<T, half>::value) {
    return ynn_type_fp16;
  } else if (std::is_same<T, bfloat16>::value) {
    return ynn_type_bf16;
  } else if (std::is_same<T, float>::value) {
    return ynn_type_fp32;
  } else if (std::is_same<T, int8_t>::value ||
             std::is_same<T, quantized<int8_t>>::value) {
    return ynn_type_int8;
  } else if (std::is_same<T, uint8_t>::value ||
             std::is_same<T, quantized<uint8_t>>::value) {
    return ynn_type_uint8;
  } else if (std::is_same<T, int4x2>::value ||
             std::is_same<T, quantized<int4x2>>::value) {
    return ynn_type_int4;
  } else if (std::is_same<T, uint4x2>::value ||
             std::is_same<T, quantized<uint4x2>>::value) {
    return ynn_type_uint4;
  } else if (std::is_same<T, int32_t>::value ||
             std::is_same<T, quantized<int32_t>>::value) {
    return ynn_type_int32;
  } else {
    return ynn_type_invalid;
  }
}

template <typename F>
constexpr decltype(auto) SwitchRealType(ynn_type type, F&& f) {
  switch (type) {
    case ynn_type_int8:
      return std::forward<F>(f)(quantized<int8_t>());
    case ynn_type_uint8:
      return std::forward<F>(f)(quantized<uint8_t>());
    case ynn_type_int32:
      return std::forward<F>(f)(quantized<int32_t>());
    case ynn_type_fp16:
      return std::forward<F>(f)(half());
    case ynn_type_bf16:
      return std::forward<F>(f)(bfloat16());
    case ynn_type_fp32:
      return std::forward<F>(f)(float());
    default:
      YNN_UNREACHABLE;
  }
}

template <typename F>
constexpr decltype(auto) SwitchIntegerType(ynn_type type, F&& f) {
  switch (type) {
    case ynn_type_int8:
      return std::forward<F>(f)(int8_t());
    case ynn_type_uint8:
      return std::forward<F>(f)(uint8_t());
    case ynn_type_int32:
      return std::forward<F>(f)(int32_t());
    default:
      YNN_UNREACHABLE;
  }
}

template <typename T>
class type_info {
 public:
  static constexpr T epsilon() { return std::numeric_limits<T>::epsilon(); }
  static constexpr T infinity() { return std::numeric_limits<T>::infinity(); }
  static constexpr T min() { return std::numeric_limits<T>::lowest(); }
  static constexpr T max() { return std::numeric_limits<T>::max(); }
  static constexpr T smallest_normal() { return std::numeric_limits<T>::min(); }

  // These are the identity values (values such that f(x, identity) = x) for a
  // min or max operation.
  static constexpr T min_identity() { return infinity(); }
  static constexpr T max_identity() { return -infinity(); }
  static constexpr T sum_identity() { return 0; }

  static constexpr size_t element_count() { return 1; }

  // Get the `i`th element from an array `x` of instances of type `T`.
  YNN_ALWAYS_INLINE static T get(const T* x, size_t i) { return x[i]; }
  YNN_ALWAYS_INLINE static void set(T* x, size_t i, T value) { x[i] = value; }
};

template <>
class type_info<half> {
 public:
  static constexpr half epsilon() { return half::epsilon(); }
  static constexpr half infinity() { return half::infinity(); }
  static constexpr half min() { return half::min(); }
  static constexpr half max() { return half::max(); }
  static constexpr half smallest_normal() { return half::smallest_normal(); }
  static constexpr half min_identity() { return half::min_identity(); }
  static constexpr half max_identity() { return half::max_identity(); }
  static constexpr half sum_identity() { return half::sum_identity(); }

  static constexpr size_t element_count() { return 1; }

  YNN_ALWAYS_INLINE static half get(const half* x, size_t i) { return x[i]; }
  YNN_ALWAYS_INLINE static void set(half* x, size_t i, half value) {
    x[i] = value;
  }
};

template <>
class type_info<bfloat16> {
 public:
  static constexpr bfloat16 epsilon() {
    return bfloat16::from_bits(0x3c00);  // 2^-7 = 0.0078125
  }
  static constexpr bfloat16 infinity() { return bfloat16::from_bits(0x7f80); }
  static constexpr bfloat16 min() { return bfloat16::from_bits(0xff7f); }
  static constexpr bfloat16 max() { return bfloat16::from_bits(0x7f7f); }
  static constexpr bfloat16 smallest_normal() {
    return bfloat16::from_bits(0x0080);  // 2^-126
  }
  static constexpr bfloat16 min_identity() {
    return bfloat16::from_bits(0x7f80);
  }
  static constexpr bfloat16 max_identity() {
    return bfloat16::from_bits(0xff80);
  }
  static constexpr bfloat16 sum_identity() { return bfloat16::from_bits(0); }

  static constexpr size_t element_count() { return 1; }

  YNN_ALWAYS_INLINE static bfloat16 get(const bfloat16* x, size_t i) {
    return x[i];
  }
  YNN_ALWAYS_INLINE static void set(bfloat16* x, size_t i, bfloat16 value) {
    x[i] = value;
  }
};

template <>
class type_info<int8_t> {
 public:
  static constexpr int8_t epsilon() { return 1; }
  static constexpr int8_t min() { return -128; }
  static constexpr int8_t max() { return 127; }
  static constexpr int8_t smallest_normal() { return 0; }
  static constexpr int8_t min_identity() { return max(); }
  static constexpr int8_t max_identity() { return min(); }
  static constexpr int8_t sum_identity() { return 0; }

  static constexpr size_t element_count() { return 1; }

  YNN_ALWAYS_INLINE static int8_t get(const int8_t* x, size_t i) {
    return x[i];
  }
  YNN_ALWAYS_INLINE static void set(int8_t* x, size_t i, int8_t value) {
    x[i] = value;
  }
};

template <>
class type_info<uint8_t> {
 public:
  static constexpr uint8_t epsilon() { return 1; }
  static constexpr uint8_t min() { return 0; }
  static constexpr uint8_t max() { return 255; }
  static constexpr uint8_t smallest_normal() { return 0; }
  static constexpr uint8_t min_identity() { return max(); }
  static constexpr uint8_t max_identity() { return min(); }
  static constexpr uint8_t sum_identity() { return 0; }

  static constexpr size_t element_count() { return 1; }

  YNN_ALWAYS_INLINE static uint8_t get(const uint8_t* x, size_t i) {
    return x[i];
  }
  YNN_ALWAYS_INLINE static void set(uint8_t* x, size_t i, uint8_t value) {
    x[i] = value;
  }
};

template <>
class type_info<int4x2> {
 public:
  static int32_t min() { return -8; }
  static int32_t max() { return 7; }
  static int32_t smallest_normal() { return 0; }
  static int32_t min_identity() { return max(); }
  static int32_t max_identity() { return min(); }

  static constexpr size_t element_count() { return 2; }

  YNN_ALWAYS_INLINE static int get(const int4x2* x, size_t i) {
    return x[i >> 1].get(i & 1);
  }
  YNN_ALWAYS_INLINE static void set(int4x2* x, size_t i, int8_t value) {
    x[i >> 1].set(i & 1, value);
  }
};

template <>
class type_info<uint4x2> {
 public:
  static int32_t min() { return 0; }
  static int32_t max() { return 15; }
  static int32_t smallest_normal() { return 0; }
  static int32_t min_identity() { return max(); }
  static int32_t max_identity() { return min(); }

  static constexpr size_t element_count() { return 2; }

  YNN_ALWAYS_INLINE static int get(const uint4x2* x, size_t i) {
    return x[i >> 1].get(i & 1);
  }
  YNN_ALWAYS_INLINE static void set(uint4x2* x, size_t i, uint8_t value) {
    x[i >> 1].set(i & 1, value);
  }
};

template <typename T>
class type_info<quantized<T>> {
 public:
  static quantized<T> min() { return {type_info<T>::min()}; }
  static quantized<T> max() { return {type_info<T>::max()}; }
  static quantized<T> smallest_normal() { return {0}; }
  static quantized<T> min_identity() { return max(); }
  static quantized<T> max_identity() { return min(); }

  static constexpr size_t element_count() {
    return type_info<T>::element_count();
  }

  YNN_ALWAYS_INLINE static quantized<T> get(const quantized<T>* x, size_t i) {
    return x[i];
  }
  YNN_ALWAYS_INLINE static void set(quantized<T>* x, size_t i,
                                    quantized<T> value) {
    x[i] = value;
  }
};

inline float epsilon(ynn_type type) {
  switch (type) {
    case ynn_type_fp32:
      return type_info<float>::epsilon();
    case ynn_type_fp16:
      return type_info<half>::epsilon();
    case ynn_type_bf16:
      return type_info<bfloat16>::epsilon();
    default:
      return 1.0f;
  }
}

struct quantization_params {
  int32_t zero_point = 0;
  float scale = 1.0f;
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_TYPE_H_
