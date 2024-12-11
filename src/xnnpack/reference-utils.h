// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "xnnpack/datatype.h"

namespace xnnpack {

// A cast that:
// - Rounds to nearest integer
// - Replaces NaN with 0
// - Saturates to the bounds of the result type
template <typename Result, typename std::enable_if<!xnnpack::is_quantized<
                               Result>::value>::type* = nullptr>
Result round_float_to_int(float x) {
  x = std::isnan(x) ? 0.0f : x;
  x = std::round(x);
  // It's tricky to do this with std::max/std::min, because the min/max values
  // might not be exactly representable as floats, and so are ineffective to
  // avoid converting to an out of bounds integer. To avoid this problem, we've
  // determined a constant that when added to the min/max float values, results
  // in the upper bound of the integer range.
  constexpr int half_mantissa = sizeof(Result) * 8 > 23 ? 127 : 0;
  x = std::max<float>(x, std::numeric_limits<Result>::min());
  x = std::min<float>(x, std::numeric_limits<Result>::max() - half_mantissa);
  return static_cast<Result>(x);
}

template <typename Result,
          typename std::enable_if<xnnpack::is_quantized<Result>::value>::type* =
              nullptr>
Result round_float_to_int(float x) {
  return round_float_to_int<typename Result::type>(x);
}

template <typename T>
float dequantize(T x, float scale, float zero_point) {
  return (static_cast<float>(x) - zero_point) * scale;
}

template <typename T>
T quantize(float x, float inv_scale, float zero_point) {
  return round_float_to_int<T>(x * inv_scale + zero_point);
}

// These help to implement integer arithmetic without signed integer overflow.
inline int64_t widen(int32_t x) {
  return static_cast<int64_t>(x);
}
inline int32_t widen(int16_t x) {
  return static_cast<int32_t>(x);
}
inline int16_t widen(int8_t x) {
  return static_cast<int16_t>(x);
}

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

template <typename T>
T euclidean_mod(T a, T b) {
  if (b == 0) {
    return 0;
  }
  T r = a % b;
  return r >= 0 ? r : (b < 0 ? r - b : r + b);
}

template <typename T>
T integer_pow(T a, T b) {
  if (b < 0) {
    return euclidean_div<T>(1, integer_pow(a, -b));
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

}  // namespace xnnpack
