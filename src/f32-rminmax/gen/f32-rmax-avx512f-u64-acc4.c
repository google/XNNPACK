// Auto-generated file. Do not edit!
//   Template: src/f32-rminmax/avx512f.c.in
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


void xnn_f32_rmax_ukernel__avx512f_u64_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m512 vmax0 = _mm512_set1_ps(*input);
  __m512 vmax1 = vmax0;
  __m512 vmax2 = vmax0;
  __m512 vmax3 = vmax0;
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m512 vt0 = _mm512_loadu_ps(input);
    const __m512 vt1 = _mm512_loadu_ps(input + 16);
    const __m512 vt2 = _mm512_loadu_ps(input + 32);
    const __m512 vt3 = _mm512_loadu_ps(input + 48);
    input += 64;

    vmax0 = _mm512_max_ps(vmax0, vt0);
    vmax1 = _mm512_max_ps(vmax1, vt1);
    vmax2 = _mm512_max_ps(vmax2, vt2);
    vmax3 = _mm512_max_ps(vmax3, vt3);
  }
  vmax0 = _mm512_max_ps(vmax0, vmax1);
  vmax2 = _mm512_max_ps(vmax2, vmax3);
  vmax0 = _mm512_max_ps(vmax0, vmax2);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vt = _mm512_loadu_ps(input);
    input += 16;

    vmax0 = _mm512_max_ps(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vt = _mm512_maskz_loadu_ps(vmask, input);

    vmax0 = _mm512_mask_max_ps(vmax0, vmask, vmax0, vt);
  }
  __m256 vmax256 = _mm256_max_ps(_mm512_castps512_ps256(vmax0), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vmax0), 1)));
  __m128 vmax = _mm_max_ps(_mm256_castps256_ps128(vmax256), _mm256_extractf128_ps(vmax256, 1));
  vmax = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
  vmax = _mm_max_ss(vmax, _mm_movehdup_ps(vmax));
  _mm_store_ss(output, vmax);
}
