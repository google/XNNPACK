// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_ARITHMETIC_H_
#define XNNPACK_YNNPACK_BASE_ARITHMETIC_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "ynnpack/base/type.h"

namespace ynn {

// Clamp a float to the range of the given integer or quantized integer type.
template <typename Int>
float clamp_float_to_int(float x) {
  using Unwrapped = typename unwrap_quantized<Int>::type;
  // It's tricky to do this with std::max/std::min, because the min/max values
  // might not be exactly representable as floats, and so are ineffective to
  // avoid converting to an out of bounds integer. To avoid this problem, we've
  // determined a constant that when added to the min/max float values, results
  // in the upper bound of the integer range.
  constexpr int half_mantissa = sizeof(Unwrapped) * 8 > 23 ? 127 : 0;
  x = std::max<float>(x, std::numeric_limits<Unwrapped>::min());
  x = std::min<float>(x, std::numeric_limits<Unwrapped>::max() - half_mantissa);
  return x;
}

// A cast that:
// - Rounds to nearest integer
// - Replaces NaN with 0
// - Saturates to the bounds of the result type
template <typename Result>
Result round_float_to_int(float x) {
  using Unwrapped = typename unwrap_quantized<Result>::type;
  x = std::isnan(x) ? 0.0f : x;
  x = std::round(x);
  x = clamp_float_to_int<Result>(x);
  return static_cast<Unwrapped>(x);
}

template <typename T>
float dequantize(T x, float scale, float zero_point) {
  return (static_cast<float>(x) - zero_point) * scale;
}
template <typename T>
float dequantize(T x, const quantization_params& params) {
  return dequantize(x, params.scale, params.zero_point);
}

template <typename T>
T quantize(float x, float inv_scale, float zero_point) {
  return round_float_to_int<T>(x * inv_scale + zero_point);
}
template <typename T>
T quantize(float x, const quantization_params& params) {
  return quantize<T>(x, 1.0f / params.scale, params.zero_point);
}

template <typename T>
void quantize(const float* in, T* out, size_t n, float inv_scale,
              float zero_point) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = quantize<T>(in[i], inv_scale, zero_point);
  }
}

template <typename T>
void quantize(const float* in, T* out, size_t n,
              const quantization_params& params) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = quantize<T>(in[i], params);
  }
}

inline float fake_quantize(float x, float inv_scale, float zero_point) {
  return std::round(x * inv_scale + zero_point);
}
inline float fake_quantize(float x, const quantization_params& params) {
  return fake_quantize(x, 1.0f / params.scale, params.zero_point);
}

// These help to implement integer arithmetic without signed integer overflow.
inline int64_t widen(int32_t x) { return static_cast<int64_t>(x); }
inline int32_t widen(int16_t x) { return static_cast<int32_t>(x); }
inline int16_t widen(int8_t x) { return static_cast<int16_t>(x); }
inline int32_t narrow(int64_t x) { return static_cast<int32_t>(x); }
inline int16_t narrow(int32_t x) { return static_cast<int16_t>(x); }
inline int8_t narrow(int16_t x) { return static_cast<int8_t>(x); }

// This implements "Euclidean division", which is the way integer division
// should be: (a / b) * b + r = a, where r is always in [0, |b|). This is
// unlike "computer division" where, annoyingly, a / b is rounded towards 0,
// and the remainder may be positive or negative accordingly. This
// implementation of Euclidean integer division is taken from
// https://github.com/dsharlet/slinky/blob/5020dae47ecb176bcd917ecd07d37e19615b955b/base/arithmetic.h#L12-L26
template <typename T>
T euclidean_div(T a, T b) {
  if (b == 0) {
    return 0;
  }
  T q = a / b;
  T r = a - q * b;
  T bs = b >> (sizeof(T) * 8 - 1);
  T rs = r >> (sizeof(T) * 8 - 1);
  return q - (rs & bs) + (rs & ~bs);
}

inline size_t euclidean_div(size_t a, size_t b) { return b != 0 ? a / b : 0; }

template <typename T>
T euclidean_mod(T a, T b) {
  if (b == 0) {
    return 0;
  }
  T r = a % b;
  return r >= 0 ? r : (b < 0 ? r - b : r + b);
}

inline size_t euclidean_mod(size_t a, size_t b) { return b != 0 ? a % b : 0; }

template <typename T>
bool is_power_of_two(T x) {
  return (x & (x - 1)) == 0;
}

template <typename T>
T align_down(T value, T alignment) {
  assert(is_power_of_two(alignment));
  return value & ~(alignment - 1);
}

template <typename T>
T align_up(T value, T alignment) {
  return align_down(value + alignment - 1, alignment);
}

template <typename T>
T floor_div(T a, T b) {
  return euclidean_div(a, b);
}

template <typename T>
T ceil_div(T a, T b) {
  assert(b > 0);
  return euclidean_div(a + b - 1, b);
}

template <typename T>
T integer_pow(T a, T b) {
  if (b < 0) {
    // 1 / a^b is either 0 or 1 (if a^b is positive), or -1 or 0 (if a^b is
    // negative).
    if ((b & 1) == 0) {
      return euclidean_div<T>(1, narrow(widen(a) * widen(a)));
    } else {
      return euclidean_div<T>(1, a);
    }
  }
  T result = 1;
  for (; b; b >>= 1) {
    if (b & 1) {
      result = widen(result) * widen(a);
    }
    a = widen(a) * widen(a);
  }
  return result;
}

template <typename T>
T* offset_bytes(T* ptr, ptrdiff_t offset) {
  assert(ptr || offset == 0);
  return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr) + offset);
}

template <typename T>
const T* offset_bytes(const T* ptr, ptrdiff_t offset) {
  assert(ptr || offset == 0);
  return reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(ptr) +
                                    offset);
}

// std::sub_sat is in C++26, we can use that in a few decades maybe.
inline size_t sub_sat(size_t a, size_t b) { return a > b ? a - b : 0; }

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_ARITHMETIC_H_
