// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_FMA_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_FMA_H_

#include <immintrin.h>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bit_cast.h"
#include "ynnpack/base/simd/x86_sse2.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

namespace internal {

YNN_ALWAYS_INLINE f64x4& operator|=(f64x4& a, f64x4 b) {
  a.v[0].v = _mm_or_pd(a.v[0].v, b.v[0].v);
  a.v[1].v = _mm_or_pd(a.v[1].v, b.v[1].v);
  return a;
}
YNN_ALWAYS_INLINE f64x4& operator&=(f64x4& a, f64x4 b) {
  a.v[0].v = _mm_and_pd(a.v[0].v, b.v[0].v);
  a.v[1].v = _mm_and_pd(a.v[1].v, b.v[1].v);
  return a;
}
YNN_ALWAYS_INLINE f64x4 operator|(f64x4 a, f64x4 b) { return a |= b; }
YNN_ALWAYS_INLINE f64x4 operator&(f64x4 a, f64x4 b) { return a &= b; }
YNN_ALWAYS_INLINE f64x4 operator!=(f64x4 a, f64x4 b) {
  a.v[0].v = _mm_cmpneq_pd(a.v[0].v, b.v[0].v);
  a.v[1].v = _mm_cmpneq_pd(a.v[1].v, b.v[1].v);
  return a;
}

YNN_ALWAYS_INLINE f64x4 is_finite(f64x4 a) {
  // If a is infinite, then this will be NaN, and then comparing NaN to itself
  // will be false.
  a = a - a;
  a.v[0].v = _mm_cmpeq_pd(a.v[0].v, a.v[0].v);
  a.v[1].v = _mm_cmpeq_pd(a.v[1].v, a.v[1].v);
  return a;
}

// Computes f32(a + b) as if the intermediate sum were computed with infinite
// precision, i.e. there is only a single rounding. This approach was derived by
// Andrew Adams.
inline f32x4 narrowing_add(f64x4 a, f64x4 b) {
  f64x4 c = a + b;
  f64x4 a2 = c - b;
  f64x4 b2 = c - a;

  // At least one of a - a2 or b - b2 is zero.
  f64x4 err = (a - a2) | (b - b2);

  // How big is a one in the LSB of the mantissa?
  f64x4 eps = (c | f64x4{bit_cast<double>(1ll)}) - c;

  // If there is both a 1 in the LSB of the mantissa, and the error is not zero,
  // then this operation would have a double rounding. We can fix this by adding
  // a correction such that the conversion to float performs the only rounding.

  // We want the magnitude of epsilon with the sign of the error.
  f64x4 correction = copysign(eps, err);

  // Only add the correction if the error is not zero and the result is finite.
  correction &= err != f64x4{-0.0};
  correction &= is_finite(c);

  return convert(c + correction, float{});
}

}  // namespace internal

inline f32x4 fma(f32x4 a, f32x4 b, f32x4 acc) {
  // This product has no rounding error, because a double mantissa is more than
  // 2x bigger than a float mantissa.
  f64x4 product = convert(a, double{}) * convert(b, double{});
  return internal::narrowing_add(product, convert(acc, double{}));
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_FMA_H_
