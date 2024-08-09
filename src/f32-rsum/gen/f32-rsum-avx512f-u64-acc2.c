// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/avx512f.c.in
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


void xnn_f32_rsum_ukernel__avx512f_u64_acc2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m512 vt0 = _mm512_loadu_ps(input);
    const __m512 vt1 = _mm512_loadu_ps(input + 16);
    const __m512 vt2 = _mm512_loadu_ps(input + 32);
    const __m512 vt3 = _mm512_loadu_ps(input + 48);
    input += 64;

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc1 = _mm512_add_ps(vacc1, vt1);
    vacc0 = _mm512_add_ps(vacc0, vt2);
    vacc1 = _mm512_add_ps(vacc1, vt3);
  }
  vacc0 = _mm512_add_ps(vacc0, vacc1);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vt = _mm512_loadu_ps(input);
    input += 16;

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vt = _mm512_maskz_loadu_ps(vmask, input);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }

  __m256 vacc256 = _mm256_add_ps(_mm512_castps512_ps256(vacc0), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256), _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));
  vacc = _mm_max_ss(vacc, _mm_load_ss(&params->scalar.min));
  vacc = _mm_min_ss(vacc, _mm_load_ss(&params->scalar.max));
  *output += _mm_cvtss_f32(vacc);
}
