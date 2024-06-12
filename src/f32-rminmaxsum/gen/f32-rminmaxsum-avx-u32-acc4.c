// Auto-generated file. Do not edit!
//   Template: src/f32-rminmaxsum/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/reduce.h>


void xnn_f32_rminmaxsum_ukernel__avx_u32_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m256 vmin0 = _mm256_broadcast_ss(input);
  __m256 vmax0 = vmin0;
  __m256 vsum0 = _mm256_setzero_ps();
  __m256 vmin1 = vmin0;
  __m256 vmax1 = vmax0;
  __m256 vsum1 = vsum0;
  __m256 vmin2 = vmin0;
  __m256 vmax2 = vmax0;
  __m256 vsum2 = vsum0;
  __m256 vmin3 = vmin0;
  __m256 vmax3 = vmax0;
  __m256 vsum3 = vsum0;
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const __m256 vt0 = _mm256_loadu_ps(input);
    const __m256 vt1 = _mm256_loadu_ps(input + 8);
    const __m256 vt2 = _mm256_loadu_ps(input + 16);
    const __m256 vt3 = _mm256_loadu_ps(input + 24);
    input += 32;

    vmin0 = _mm256_min_ps(vmin0, vt0);
    vmax0 = _mm256_max_ps(vmax0, vt0);
    vsum0 = _mm256_add_ps(vsum0, vt0);
    vmin1 = _mm256_min_ps(vmin1, vt1);
    vmax1 = _mm256_max_ps(vmax1, vt1);
    vsum1 = _mm256_add_ps(vsum1, vt1);
    vmin2 = _mm256_min_ps(vmin2, vt2);
    vmax2 = _mm256_max_ps(vmax2, vt2);
    vsum2 = _mm256_add_ps(vsum2, vt2);
    vmin3 = _mm256_min_ps(vmin3, vt3);
    vmax3 = _mm256_max_ps(vmax3, vt3);
    vsum3 = _mm256_add_ps(vsum3, vt3);
  }
  vmin0 = _mm256_min_ps(vmin0, vmin1);
  vmax0 = _mm256_max_ps(vmax0, vmax1);
  vsum0 = _mm256_add_ps(vsum0, vsum1);
  vmin2 = _mm256_min_ps(vmin2, vmin3);
  vmax2 = _mm256_max_ps(vmax2, vmax3);
  vsum2 = _mm256_add_ps(vsum2, vsum3);
  vmin0 = _mm256_min_ps(vmin0, vmin2);
  vmax0 = _mm256_max_ps(vmax0, vmax2);
  vsum0 = _mm256_add_ps(vsum0, vsum2);
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vt = _mm256_loadu_ps(input);
    input += 8;

    vmin0 = _mm256_min_ps(vmin0, vt);
    vmax0 = _mm256_max_ps(vmax0, vt);
    vsum0 = _mm256_add_ps(vsum0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - batch));

    const __m256 vt = _mm256_maskload_ps(input, vmask);

    vmin0 = _mm256_blendv_ps(vmin0, _mm256_min_ps(vmin0, vt), _mm256_castsi256_ps(vmask));
    vmax0 = _mm256_blendv_ps(vmax0, _mm256_max_ps(vmax0, vt), _mm256_castsi256_ps(vmask));
    vsum0 = _mm256_blendv_ps(vsum0, _mm256_add_ps(vsum0, vt), _mm256_castsi256_ps(vmask));
  }
  __m128 vmin = _mm_min_ps(_mm256_castps256_ps128(vmin0), _mm256_extractf128_ps(vmin0, 1));
  __m128 vmax = _mm_max_ps(_mm256_castps256_ps128(vmax0), _mm256_extractf128_ps(vmax0, 1));
  __m128 vsum = _mm_add_ps(_mm256_castps256_ps128(vsum0), _mm256_extractf128_ps(vsum0, 1));

  vmin = _mm_min_ps(vmin, _mm_movehl_ps(vmin, vmin));
  vmax = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
  vsum = _mm_add_ps(vsum, _mm_movehl_ps(vsum, vsum));
  vmin = _mm_min_ss(vmin, _mm_movehdup_ps(vmin));
  vmax = _mm_max_ss(vmax, _mm_movehdup_ps(vmax));
  vsum = _mm_add_ss(vsum, _mm_movehdup_ps(vsum));
  _mm_store_ss(output, vmin);
  _mm_store_ss(output + 1, vmax);
  _mm_store_ss(output + 2, vsum);
}
