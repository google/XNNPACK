// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-avx.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vaddc_minmax_ukernel__avx_u16(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 voutput_min = _mm256_set1_ps(params->avx.min);
  const __m256 voutput_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);
  const __m256 vb = _mm256_broadcast_ss(input_b);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m256 vacc0 = _mm256_loadu_ps(input_a);
    __m256 vacc1 = _mm256_loadu_ps(input_a + 8);
    input_a += 16;

    vacc0 = _mm256_add_ps(vacc0, vb);
    vacc1 = _mm256_add_ps(vacc1, vb);


    vacc0 = _mm256_max_ps(voutput_min, vacc0);
    vacc1 = _mm256_max_ps(voutput_min, vacc1);

    vacc0 = _mm256_min_ps(voutput_max, vacc0);
    vacc1 = _mm256_min_ps(voutput_max, vacc1);

    _mm256_storeu_ps(output, vacc0);
    _mm256_storeu_ps(output + 8, vacc1);
    output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256 vacc = _mm256_loadu_ps(input_a);
    input_a += 8;

    vacc = _mm256_add_ps(vacc, vb);
    vacc = _mm256_max_ps(voutput_min, vacc);
    vacc = _mm256_min_ps(voutput_max, vacc);
    _mm256_storeu_ps(output, vacc);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    __m256 vacc = _mm256_maskload_ps(input_a, vmask);

    vacc = _mm256_add_ps(vacc, vb);
    vacc = _mm256_max_ps(voutput_min, vacc);
    vacc = _mm256_min_ps(voutput_max, vacc);

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
