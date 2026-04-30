// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_TYPE_H_
#define XNNPACK_YNNPACK_BASE_TYPE_H_

#include <algorithm>
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
using bfloat16x1_t = bfloat16;
using float32x1_t = float;

// Returns true if the type is an integer type.
bool type_is_integral(ynn_type t);

// Returns true if the type is a floating point type.
bool type_is_floating_point(ynn_type t);

// Returns the size of an element of the type.
size_t type_size_bytes(ynn_type t);

// Returns the size of an element of the type in bits.
size_t type_size_bits(ynn_type t);

// Returns the number of bits in the mantissa of the type, including the implied
// leading one for float types.
size_t type_mantissa_bits(ynn_type t);

// Returns the number of bits in the exponent of the type.
size_t type_exponent_bits(ynn_type t);

// Returns true if converting a value of `from` to `to` can be done without
// losing information.
bool is_convert_lossless(ynn_type from, ynn_type to);

// Returns how many elements are contained in one instance of the type. We
// assume that datatypes with a non-integer number of bytes per element can be
// represented by a struct that contains multiple elements. `type_size_bytes`
// returns the size in bytes of one instance of that struct, and this function
// returns the number of elements stored in that struct.
inline size_t type_element_count(ynn_type t) {
  switch (t) {
    case ynn_type_int2:
    case ynn_type_uint2:
      return 4;
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
  int4x2(int8_t x) : int4x2(x, x) {}  // NOLINT
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

// Four int2 values stored in an int8.
struct int2x4 {
  uint8_t values;

  int2x4() = default;
  int2x4(int8_t x) : int2x4(x, x, x, x) {}  // NOLINT
  int2x4(int8_t x0, int8_t x1, int8_t x2, int8_t x3)
      : values(((x3 & 0x3) << 6) | ((x2 & 0x3) << 4) | ((x1 & 0x3) << 2) |
               (x0 & 0x3)) {}

  YNN_ALWAYS_INLINE int8_t get(size_t i) const {
    switch (i) {
      case 0:
        return static_cast<int8_t>(values << 6) >> 6;
      case 1:
        return static_cast<int8_t>((values << 4) & 0xc0) >> 6;
      case 2:
        return static_cast<int8_t>((values << 2) & 0xc0) >> 6;
      case 3:
        return static_cast<int8_t>(values & 0xc0) >> 6;
      default:
        YNN_UNREACHABLE;
    }
  }
  YNN_ALWAYS_INLINE void set(size_t i, int8_t value) {
    switch (i) {
      case 0:
        values = (values & 0xfc) | (value & 0x03);
        return;
      case 1:
        values = (values & 0xf3) | ((value & 0x03) << 2);
        return;
      case 2:
        values = (values & 0xcf) | ((value & 0x03) << 4);
        return;
      case 3:
        values = (values & 0x3f) | ((value & 0x03) << 6);
        return;
      default:
        YNN_UNREACHABLE;
    }
  }

  bool operator==(const int2x4& other) const { return values == other.values; }
  bool operator!=(const int2x4& other) const { return values != other.values; }
};

struct uint4x2 {
  uint8_t values;

  uint4x2() = default;
  uint4x2(uint8_t x) : uint4x2(x, x) {}  // NOLINT
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

struct uint2x4 {
  uint8_t values;

  uint2x4() = default;
  uint2x4(uint8_t x) : uint2x4(x, x, x, x) {}  // NOLINT
  uint2x4(uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3)
      : values((x3 << 6) | ((x2 & 0x3) << 4) | ((x1 & 0x3) << 2) | (x0 & 0x3)) {
  }

  YNN_ALWAYS_INLINE uint8_t get(size_t i) const {
    switch (i) {
      case 0:
        return static_cast<uint8_t>(values & 0x03);
      case 1:
        return static_cast<uint8_t>(values & 0x0c) >> 2;
      case 2:
        return static_cast<uint8_t>(values & 0x30) >> 4;
      case 3:
        return static_cast<uint8_t>(values & 0xc0) >> 6;
      default:
        YNN_UNREACHABLE;
    }
  }
  YNN_ALWAYS_INLINE void set(size_t i, uint8_t value) {
    switch (i) {
      case 0:
        values = (values & 0xfc) | (value & 0x03);
        return;
      case 1:
        values = (values & 0xf3) | ((value & 0x03) << 2);
        return;
      case 2:
        values = (values & 0xcf) | ((value & 0x03) << 4);
        return;
      case 3:
        values = (values & 0x3f) | ((value & 0x03) << 6);
        return;
      default:
        YNN_UNREACHABLE;
    }
  }

  bool operator==(const uint2x4& other) const { return values == other.values; }
  bool operator!=(const uint2x4& other) const { return values != other.values; }
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
  } else if (std::is_same<T, double>::value) {
    return ynn_type_fp64;
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
  } else if (std::is_same<T, int2x4>::value ||
             std::is_same<T, quantized<int2x4>>::value) {
    return ynn_type_int2;
  } else if (std::is_same<T, uint2x4>::value ||
             std::is_same<T, quantized<uint2x4>>::value) {
    return ynn_type_uint2;
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
  using element_type = T;

  static constexpr T epsilon() { return std::numeric_limits<T>::epsilon(); }
  static constexpr T infinity() { return std::numeric_limits<T>::infinity(); }
  static constexpr T nan() { return std::numeric_limits<T>::quiet_NaN(); }
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
  YNN_ALWAYS_INLINE static T& ref(T* x, size_t i) { return x[i]; }
};

template <>
class type_info<half> {
 public:
  using element_type = half;

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
  YNN_ALWAYS_INLINE static half& ref(half* x, size_t i) { return x[i]; }
};

template <>
class type_info<bfloat16> {
 public:
  using element_type = bfloat16;

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
  YNN_ALWAYS_INLINE static bfloat16& ref(bfloat16* x, size_t i) { return x[i]; }
};

template <>
class type_info<int8_t> {
 public:
  using element_type = int8_t;

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
  YNN_ALWAYS_INLINE static int8_t& ref(int8_t* x, size_t i) { return x[i]; }
};

template <>
class type_info<uint8_t> {
 public:
  using element_type = uint8_t;

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
  YNN_ALWAYS_INLINE static uint8_t& ref(uint8_t* x, size_t i) { return x[i]; }
};

// This wrapper allows forming a "reference" to a sub-byte datatype.
template <typename T>
class element_ref {
 public:
  YNN_ALWAYS_INLINE element_ref(T* values, size_t offset)
      : values_(values), offset_(offset) {}

  YNN_ALWAYS_INLINE operator int() const {  // NOLINT
    return type_info<T>::get(values_, offset_);
  }
  YNN_ALWAYS_INLINE element_ref& operator=(const element_ref& other) {
    type_info<T>::set(values_, offset_, other);
    return *this;
  }
  template <typename U>
  YNN_ALWAYS_INLINE element_ref& operator=(U value) {
    type_info<T>::set(values_, offset_, value);
    return *this;
  }

  T* address_of() {
    assert(offset_ % type_info<T>::element_count() == 0);
    return values_ + offset_ / type_info<T>::element_count();
  }

 private:
  T* values_;
  size_t offset_;
};

template <>
class type_info<int4x2> {
 public:
  using element_type = int8_t;

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
  YNN_ALWAYS_INLINE static element_ref<int4x2> ref(int4x2* x, size_t i) {
    return element_ref<int4x2>(x, i);
  }
};

template <>
class type_info<int2x4> {
 public:
  using element_type = int8_t;

  static int32_t min() { return -2; }
  static int32_t max() { return 1; }
  static int32_t smallest_normal() { return 0; }
  static int32_t min_identity() { return max(); }
  static int32_t max_identity() { return min(); }

  static constexpr size_t element_count() { return 4; }

  YNN_ALWAYS_INLINE static int get(const int2x4* x, size_t i) {
    return x[i >> 2].get(i & 3);
  }
  YNN_ALWAYS_INLINE static void set(int2x4* x, size_t i, int8_t value) {
    x[i >> 2].set(i & 3, value);
  }
  YNN_ALWAYS_INLINE static element_ref<int2x4> ref(int2x4* x, size_t i) {
    return element_ref<int2x4>(x, i);
  }
};

template <>
class type_info<uint4x2> {
 public:
  using element_type = uint8_t;

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
  YNN_ALWAYS_INLINE static element_ref<uint4x2> ref(uint4x2* x, size_t i) {
    return element_ref<uint4x2>(x, i);
  }
};

template <>
class type_info<uint2x4> {
 public:
  using element_type = uint8_t;

  static int32_t min() { return 0; }
  static int32_t max() { return 3; }
  static int32_t smallest_normal() { return 0; }
  static int32_t min_identity() { return max(); }
  static int32_t max_identity() { return min(); }

  static constexpr size_t element_count() { return 4; }

  YNN_ALWAYS_INLINE static int get(const uint2x4* x, size_t i) {
    return x[i >> 2].get(i & 3);
  }
  YNN_ALWAYS_INLINE static void set(uint2x4* x, size_t i, uint8_t value) {
    x[i >> 2].set(i & 3, value);
  }
  YNN_ALWAYS_INLINE static element_ref<uint2x4> ref(uint2x4* x, size_t i) {
    return element_ref<uint2x4>(x, i);
  }
};

template <typename T>
class type_info<quantized<T>> {
 public:
  using element_type = quantized<typename type_info<T>::element_type>;

  static quantized<T> min() { return {type_info<T>::min()}; }
  static quantized<T> max() { return {type_info<T>::max()}; }
  static quantized<T> smallest_normal() { return {0}; }
  static quantized<T> min_identity() { return max(); }
  static quantized<T> max_identity() { return min(); }

  static constexpr size_t element_count() {
    return type_info<T>::element_count();
  }

  YNN_ALWAYS_INLINE static element_type get(const quantized<T>* x, size_t i) {
    return {type_info<T>::get(reinterpret_cast<const T*>(x), i)};
  }
  YNN_ALWAYS_INLINE static void set(quantized<T>* x, size_t i,
                                    element_type value) {
    type_info<T>::set(reinterpret_cast<T*>(x), i, value.value);
  }
  YNN_ALWAYS_INLINE static auto ref(quantized<T>* x, size_t i) {
    return type_info<T>::ref(reinterpret_cast<T*>(x), i);
  }
};

template <typename T>
T* address_of(T& value) {
  return &value;
}

template <typename T>
T* address_of(element_ref<T> value) {
  return value.address_of();
}

// Extend std::is_integral for our sub-byte types.
template <typename T>
struct is_integral {
  static constexpr bool value = std::is_integral<T>::value;
};

template <>
struct is_integral<int4x2> {
  static constexpr bool value = true;
};

template <>
struct is_integral<uint4x2> {
  static constexpr bool value = true;
};

template <>
struct is_integral<int2x4> {
  static constexpr bool value = true;
};

template <>
struct is_integral<uint2x4> {
  static constexpr bool value = true;
};

inline float epsilon(ynn_type type) {
  switch (type) {
    case ynn_type_fp64:
      return type_info<double>::epsilon();
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

// Implement std::copy_n, std::generate_n with support for sub-byte types.
template <typename Dst, typename F>
void generate_n(Dst* dst, size_t offset, size_t n, F&& f) {
  using DstInfo = type_info<Dst>;
  if (DstInfo::element_count() == 1) {
    std::generate_n(dst + offset, n, std::forward<F>(f));
  } else {
    // We could relax these requirements if needed...
    assert(offset % DstInfo::element_count() == 0);
    assert(n % DstInfo::element_count() == 0);
    dst += offset / DstInfo::element_count();
    n /= DstInfo::element_count();
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < DstInfo::element_count(); ++j) {
        type_info<Dst>::set(dst, j, f());
      }
      ++dst;
    }
  }
}

template <typename Src, typename Dst>
void copy_n(Src* src, size_t src_offset, size_t n, Dst* dst,
            size_t dst_offset) {
  using SrcInfo = type_info<std::remove_cv_t<Src>>;
  using DstInfo = type_info<std::remove_cv_t<Dst>>;
  if (SrcInfo::element_count() == 1 && DstInfo::element_count() == 1) {
    std::copy_n(src + src_offset, n, dst + dst_offset);
  } else {
    // We could relax these requirements if needed...
    assert(src_offset % SrcInfo::element_count() == 0);
    assert(dst_offset % DstInfo::element_count() == 0);
    assert(n % DstInfo::element_count() == 0);
    src += src_offset / SrcInfo::element_count();
    dst += dst_offset / DstInfo::element_count();
    n /= DstInfo::element_count();
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < DstInfo::element_count(); ++j) {
        DstInfo::set(dst, j,
                     SrcInfo::get(src, i * DstInfo::element_count() + j));
      }
      ++dst;
    }
  }
}

// Convert `n` floats from `src` to `type` stored in `dst`.
void convert_n(const float* src, size_t n, ynn_type type, void* dst);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_TYPE_H_
