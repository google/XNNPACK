// Auto-generated file. Do not edit!
//   Template: src/f32-rminmax/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"


void xnn_f32_rminmax_ukernel__avx_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  __m256 vmin0 = _mm256_broadcast_ss(input);
  __m256 vmax0 = vmin0;
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vt = _mm256_loadu_ps(input);
    input += 8;

    vmin0 = _mm256_min_ps(vmin0, vt);
    vmax0 = _mm256_max_ps(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    const __m256 vt = _mm256_maskload_ps(input, vmask);

    vmin0 = _mm256_blendv_ps(vmin0, _mm256_min_ps(vmin0, vt), _mm256_castsi256_ps(vmask));
    vmax0 = _mm256_blendv_ps(vmax0, _mm256_max_ps(vmax0, vt), _mm256_castsi256_ps(vmask));
  }
  __m128 vmin = _mm_min_ps(_mm256_castps256_ps128(vmin0), _mm256_extractf128_ps(vmin0, 1));
  __m128 vmax = _mm_max_ps(_mm256_castps256_ps128(vmax0), _mm256_extractf128_ps(vmax0, 1));
  vmin = _mm_min_ps(vmin, _mm_movehl_ps(vmin, vmin));
  vmax = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
  vmin = _mm_min_ss(vmin, _mm_movehdup_ps(vmin));
  vmax = _mm_max_ss(vmax, _mm_movehdup_ps(vmax));
  _mm_store_ss(output, vmin);
  _mm_store_ss(output + 1, vmax);
}
