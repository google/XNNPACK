// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/avx.c.in
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


void xnn_f32_rsum_ukernel__avx_u32_acc4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m256 vacc0 = _mm256_setzero_ps();
  __m256 vacc1 = _mm256_setzero_ps();
  __m256 vacc2 = _mm256_setzero_ps();
  __m256 vacc3 = _mm256_setzero_ps();
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const __m256 vt0 = _mm256_loadu_ps(input);
    const __m256 vt1 = _mm256_loadu_ps(input + 8);
    const __m256 vt2 = _mm256_loadu_ps(input + 16);
    const __m256 vt3 = _mm256_loadu_ps(input + 24);
    input += 32;

    vacc0 = _mm256_add_ps(vacc0, vt0);
    vacc1 = _mm256_add_ps(vacc1, vt1);
    vacc2 = _mm256_add_ps(vacc2, vt2);
    vacc3 = _mm256_add_ps(vacc3, vt3);
  }
  vacc0 = _mm256_add_ps(vacc0, vacc1);
  vacc2 = _mm256_add_ps(vacc2, vacc3);
  vacc0 = _mm256_add_ps(vacc0, vacc2);
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vt = _mm256_loadu_ps(input);
    input += 8;

    vacc0 = _mm256_add_ps(vacc0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));
    const __m256 vt = _mm256_maskload_ps(input, vmask);
    vacc0 = _mm256_add_ps(vacc0, vt);
  }
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc0), _mm256_extractf128_ps(vacc0, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));
  *output += _mm_cvtss_f32(vacc);
}
