// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdio.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/reduce.h>

void xnn_qs8_rsum_minmax_fp32_ukernel__avx2_u16(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  __m256i vacc0 = _mm256_setzero_si256();
  // 256 int8s may be summed into an int16 before overflowing.
  // There are 8 lanes in the accumulator register and 1 registers.
  int num_batches = batch >> 12;
  for (; num_batches > 0; --num_batches) {
    __m256i vacc16_0 = _mm256_setzero_si256();
    for (size_t current_batch = 256; current_batch > 0; current_batch -= 16) {
      const __m256i vt0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) input)); input += 16;

      vacc16_0 = _mm256_add_epi16(vacc16_0, vt0);
    }
    __m256i left0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vacc16_0));
    __m256i right0 = _mm256_cvtepi16_epi32(_mm256_extractf128_si256(vacc16_0, 1));
    vacc0 = _mm256_add_epi32(vacc0, _mm256_add_epi32(left0, right0));
    batch -= 256;
  }
  if (XNN_UNLIKELY(batch != 0)) {
    __m256i vacc16_0 = _mm256_setzero_si256();
    for (; batch >= 16; batch -= 16) {
      const __m256i vt0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) input)); input += 16;
      vacc16_0 = _mm256_add_epi16(vacc16_0, vt0);
    }


    for (; batch >= 16; batch -= 16) {
      const __m256i vt0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) input)); input += 16;
      vacc16_0 = _mm256_add_epi16(vacc16_0, vt0);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      const __m128i vt = _mm_loadu_si128((const __m128i*) input);
      const __m128i vmask = _mm_loadu_si128((const __m128i*) &params->fp32_avx2.mask_table[15 - batch]);
      const __m256i vtl = _mm256_cvtepi8_epi16(_mm_and_si128(vt, vmask));
      vacc16_0 = _mm256_add_epi16(vacc16_0, vtl);
    }
    __m256i left = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vacc16_0));
    __m256i right = _mm256_cvtepi16_epi32(_mm256_extractf128_si256(vacc16_0, 1));
    vacc0 = _mm256_add_epi32(vacc0, _mm256_add_epi32(left, right));
  }


  __m128i vacc_lo = _mm_add_epi32(_mm256_castsi256_si128(vacc0), _mm256_extractf128_si256(vacc0, 1));
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);

  const int32_t vinit_bias = params->fp32_avx2.init_bias[0];
  const float vscale = params->fp32_avx2.scale[0];
  const int32_t output_min = params->fp32_avx2.output_min[0];
  const int32_t output_max = params->fp32_avx2.output_max[0];
  const float vmagic_bias = params->fp32_avx2.magic_bias[0];
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_avx2.magic_bias_less_output_zero_point[0];

  float vfpacc = (float) (_mm_cvtsi128_si32(vacc_lo) + vinit_bias) * vscale;
  vfpacc += vmagic_bias;
  int32_t vout = (int32_t) float_as_uint32(vfpacc);
  vout -= vmagic_bias_less_output_zero_point;
  vout = math_max_s32(vout, output_min);
  vout = math_min_s32(vout, output_max);
  *output += (int8_t) vout;
}
