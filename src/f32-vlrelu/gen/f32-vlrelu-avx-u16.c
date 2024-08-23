// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vlrelu_ukernel__avx_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vslope = _mm256_set1_ps(params->scalar.slope);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(input);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(input + 8);
    input += 16;

    __m256 vacc01234567 = _mm256_mul_ps(vx01234567, vslope);
    __m256 vacc89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vslope);

    vacc01234567 = _mm256_blendv_ps(vx01234567, vacc01234567, vx01234567);
    vacc89ABCDEF = _mm256_blendv_ps(vx89ABCDEF, vacc89ABCDEF, vx89ABCDEF);

    _mm256_storeu_ps(output, vacc01234567);
    _mm256_storeu_ps(output + 8, vacc89ABCDEF);
    output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;
    __m256 vacc = _mm256_mul_ps(vx, vslope);
    vacc = _mm256_blendv_ps(vx, vacc, vx);
    _mm256_storeu_ps(output, vacc);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);
    __m256 vacc = _mm256_mul_ps(vx, vslope);
    vacc = _mm256_blendv_ps(vx, vacc, vx);

    __m128 vacc_lo = _mm256_castps256_ps128(vacc);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vacc_lo);
      vacc_lo = _mm256_extractf128_ps(vacc, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc_lo);
      vacc_lo = _mm_movehl_ps(vacc_lo, vacc_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc_lo);
    }
  }
}
