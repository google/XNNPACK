// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_avx_base.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_avx_partial_load_store.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_sse2_partial_load_store.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

using f32x16 = vec<float, 16>;
using s32x16 = vec<int32_t, 16>;
using bf16x32 = vec<bfloat16, 32>;
using f16x32 = vec<half, 32>;
using s8x64 = vec<int8_t, 64>;
using u8x64 = vec<uint8_t, 64>;

YNN_ALWAYS_INLINE f64x4 cast(f32x4 a, double) {
  return f64x4{_mm256_cvtps_pd(a.v)};
}
YNN_ALWAYS_INLINE f32x4 cast(f64x4 a, float) {
  return f32x4{_mm256_cvtpd_ps(a.v)};
}

YNN_ALWAYS_INLINE s32x8 select(s32x8 cond, s32x8 a, s32x8 b) {
  __m256 mc = _mm256_castsi256_ps(cond.v);
  __m256 ma = _mm256_castsi256_ps(a.v);
  __m256 mb = _mm256_castsi256_ps(b.v);
  return s32x8{_mm256_castps_si256(
      _mm256_or_ps(_mm256_and_ps(mc, ma), _mm256_andnot_ps(mc, mb)))};
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_
