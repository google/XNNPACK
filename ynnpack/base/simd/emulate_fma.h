// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_EMULATE_FMA_H_
#define XNNPACK_YNNPACK_BASE_SIMD_EMULATE_FMA_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/bit_cast.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {

namespace simd {

namespace internal {

// Computes f32(a + b) as if the intermediate sum were computed with infinite
// precision, i.e. there is only a single rounding. This approach was derived by
// Andrew Adams.
template <size_t N>
inline vec<float, N> narrowing_add(vec<double, N> a, vec<double, N> b) {
  using s64x = vec<int64_t, N>;
  using f64x = vec<double, N>;
  f64x c = a + b;
  f64x a2 = c - b;
  f64x b2 = c - a;

  // At least one of a - a2 or b - b2 is zero.
  f64x err = bit_cast<f64x>(bit_cast<s64x>(a - a2) | bit_cast<s64x>(b - b2));

  // How big is a one in the LSB of the mantissa?
  f64x eps = bit_cast<f64x>(bit_cast<s64x>(c) | s64x{1}) - c;

  // If there is both a 1 in the LSB of the mantissa, and the error is not zero,
  // then this operation would have a double rounding. We can fix this by adding
  // a correction such that the conversion to float performs the only rounding.

  // We want the magnitude of epsilon with the sign of the error.
  f64x correction = copysign(eps, err);

  // Only add the correction if the error is not zero and the result is finite.
  correction = select((err != f64x{0.0}) & isfinite(c), correction, f64x{0.0});

  return cast(c + correction, float{});
}

}  // namespace internal

template <size_t N>
inline vec<float, N> emulate_fma(vec<float, N> a, vec<float, N> b,
                                 vec<float, N> acc) {
  // This product has no rounding error, because a double mantissa is more than
  // 2x bigger than a float mantissa.
  vec<double, N> product = cast(a, double{}) * cast(b, double{});
  return internal::narrowing_add(product, cast(acc, double{}));
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_EMULATE_FMA_H_
