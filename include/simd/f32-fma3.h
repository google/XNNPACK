// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_FMA3_H_
#define THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_FMA3_H_

#include <simd/f32-avx-base.h>
#include <xnnpack/common.h>


static XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return _mm256_fmadd_ps(a, b, c);
}

static XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b,
                                                xnn_simd_f32_t c) {
  return _mm256_fnmadd_ps(a, b, c);
}

#endif  // THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_FMA3_H_
