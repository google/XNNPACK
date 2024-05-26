// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/reduce.h>

void xnn_qs8_rsum_minmax_fp32_ukernel__avx512skx_u64_acc2(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  __m512i vacc0 = _mm512_setzero_si512();
  __m512i vacc1 = _mm512_setzero_si512();
  // 256 int8s may be summed into an int16 before overflowing.
  // There are 8 lanes in the accumulator register and 2 registers.
  int num_batches = batch >> 14;
  for (; num_batches > 0; --num_batches) {
    __m512i vacc16_0 = _mm512_setzero_si512();
    __m512i vacc16_1 = _mm512_setzero_si512();
    for (size_t current_batch = 512; current_batch > 0; current_batch -= 64) {
      const __m512i vt0 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) input)); input += 32;
      const __m512i vt1 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) input)); input += 32;

      vacc16_0 = _mm512_add_epi16(vacc16_0, vt0);
      vacc16_1 = _mm512_add_epi16(vacc16_1, vt1);
    }
    __m512i left0 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(vacc16_0));
    __m512i right0 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(vacc16_0, 1));
    vacc0 = _mm512_add_epi32(vacc0, _mm512_add_epi32(left0, right0));
    __m512i left1 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(vacc16_1));
    __m512i right1 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(vacc16_1, 1));
    vacc1 = _mm512_add_epi32(vacc1, _mm512_add_epi32(left1, right1));
    batch -= 512;
  }
  if (XNN_UNLIKELY(batch != 0)) {
    __m512i vacc16_0 = _mm512_setzero_si512();
    __m512i vacc16_1 = _mm512_setzero_si512();
    for (; batch >= 64; batch -= 64) {
      const __m512i vt0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) input)); input += 32;
      const __m512i vt1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) input)); input += 32;
      vacc16_0 = _mm512_add_epi16(vacc16_0, vt0);
      vacc16_1 = _mm512_add_epi16(vacc16_1, vt1);
    }
    vacc16_0 = _mm512_add_epi16(vacc16_0, vacc16_1);
    for (; batch >= 32; batch -= 32) {
      const __m512i vt = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) input)); input += 32;
      vacc16_0 = _mm512_add_epi16(vacc16_0, vt);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << (batch & 31)) - UINT32_C(1)));
      const __m512i vt = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(vmask, (const __m256i*) input));
      vacc16_0 = _mm512_add_epi16(vacc16_0, vt);
    }
    __m512i left = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(vacc16_0));
    __m512i right = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(vacc16_0, 1));
    vacc0 = _mm512_add_epi32(vacc0, _mm512_add_epi32(left, right));
  }
  vacc0 = _mm512_add_epi32(vacc0, vacc1);

  __m256i vsomething = _mm256_add_epi32(_mm512_castsi512_si256(vacc0), _mm512_extracti32x8_epi32(vacc0, 1));
  __m128i vacc_lo = _mm_add_epi32(_mm256_castsi256_si128(vsomething), _mm256_extractf128_si256(vsomething, 1));
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
