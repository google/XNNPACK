// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vpreluc/avx2.c.in
//   Generator: tools/xngen
//
// Copyright (C) 2024 Intel Corporation
//  
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//  
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//  
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//  
//  
// SPDX-License-Identifier: BSD-3-Clause


#include <assert.h>
#include <immintrin.h>
#include <emmintrin.h>
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/vbinary.h"

void xnn_qs8_vpreluc_ukernel__avx2_u16(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_vprelu_scalar_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m256i vinput_zero_point = _mm256_set1_epi32(params->scalar.input_zero_point);
  const __m256 vpositive_multiplier = _mm256_set1_ps(params->scalar.positive_multiplier);
  const __m256 vnegative_multiplier = _mm256_set1_ps(params->scalar.negative_multiplier);
  const __m256 voutput_min_less_zero_point = _mm256_set1_ps((int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point);
  const __m256 voutput_max_less_zero_point = _mm256_set1_ps((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point);
  const __m256 vmagic_bias = _mm256_set1_ps(12582912.0f);
  const __m256i vmagic_bias_less_output_zero_point = _mm256_set1_epi32(INT32_C(0x4B400000) - (int32_t)params->scalar.output_zero_point);
  const int32_t slope = (int32_t) *input_b - params->scalar.slope_zero_point;
  const __m256i vslope = _mm256_set1_epi32(slope);
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    __m256i va0 = _mm256_cvtepi8_epi32(_mm_loadu_si64((const __m128i*) input_a));
    
    __m256i va1 = _mm256_cvtepi8_epi32(_mm_loadu_si64((const __m128i*) (input_a + 8)));
    input_a += 16;
    
    __m256i va0_sub = _mm256_sub_epi32(va0, vinput_zero_point);
    __m256i vcompare0 = _mm256_cmpgt_epi32(_mm256_setzero_si256(), va0_sub);
    __m256i vacc0 = _mm256_blendv_epi8(va0_sub, _mm256_mullo_epi32(va0_sub, vslope), vcompare0);
    __m256i va1_sub = _mm256_sub_epi32(va1, vinput_zero_point);
    __m256i vcompare1 = _mm256_cmpgt_epi32(_mm256_setzero_si256(), va1_sub);
    __m256i vacc1 = _mm256_blendv_epi8(va1_sub, _mm256_mullo_epi32(va1_sub, vslope), vcompare1);

    __m256 vscale0 = _mm256_blendv_ps(vpositive_multiplier, vnegative_multiplier, _mm256_castsi256_ps(vcompare0));
    __m256 vfpacc0 = _mm256_mul_ps(_mm256_cvtepi32_ps(vacc0), vscale0);
    __m256 vscale1 = _mm256_blendv_ps(vpositive_multiplier, vnegative_multiplier, _mm256_castsi256_ps(vcompare1));
    __m256 vfpacc1 = _mm256_mul_ps(_mm256_cvtepi32_ps(vacc1), vscale1);

    __m256 vfpacc_clamped0 = _mm256_min_ps(_mm256_max_ps(vfpacc0, voutput_min_less_zero_point), voutput_max_less_zero_point);
    __m256 vfpacc_biased0 = _mm256_add_ps(vfpacc_clamped0, vmagic_bias);
    __m256i vout0 = _mm256_sub_epi32(_mm256_castps_si256(vfpacc_biased0), vmagic_bias_less_output_zero_point);
    __m256 vfpacc_clamped1 = _mm256_min_ps(_mm256_max_ps(vfpacc1, voutput_min_less_zero_point), voutput_max_less_zero_point);
    __m256 vfpacc_biased1 = _mm256_add_ps(vfpacc_clamped1, vmagic_bias);
    __m256i vout1 = _mm256_sub_epi32(_mm256_castps_si256(vfpacc_biased1), vmagic_bias_less_output_zero_point);

    const __m128i vout_low0 = _mm256_castsi256_si128(vout0);
    const __m128i vout_high0 = _mm256_extracti128_si256(vout0, 1);
    const __m128i vout_packed0 = _mm_packs_epi32(vout_low0, vout_high0);
    __m128i vout_final0 = _mm_packs_epi16(vout_packed0, vout_packed0);
    const __m128i vout_low1 = _mm256_castsi256_si128(vout1);
    const __m128i vout_high1 = _mm256_extracti128_si256(vout1, 1);
    const __m128i vout_packed1 = _mm_packs_epi32(vout_low1, vout_high1);
    __m128i vout_final1 = _mm_packs_epi16(vout_packed1, vout_packed1);

    _mm_storeu_si64((__m128i*)(output), vout_final0);

    _mm_storeu_si64((__m128i*)(output + 8), vout_final1);

    output += 16;
  }

  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m256i va = _mm256_cvtepi8_epi32(_mm_loadu_si64((const __m128i*) input_a));
    __m256i va_sub = _mm256_sub_epi32(va, vinput_zero_point);
    __m256i vcompare = _mm256_cmpgt_epi32(_mm256_setzero_si256(), va_sub);
    __m256i vacc = _mm256_blendv_epi8(va_sub, _mm256_mullo_epi32(va_sub, vslope), vcompare);
    __m256 vscale = _mm256_blendv_ps(vpositive_multiplier, vnegative_multiplier, _mm256_castsi256_ps(vcompare));
    __m256 vfpacc = _mm256_mul_ps(_mm256_cvtepi32_ps(vacc), vscale);
    __m256 vfpacc_clamped = _mm256_min_ps(_mm256_max_ps(vfpacc, voutput_min_less_zero_point), voutput_max_less_zero_point);
    __m256 vfpacc_biased = _mm256_add_ps(vfpacc_clamped, vmagic_bias);
    __m256i vout = _mm256_sub_epi32(_mm256_castps_si256(vfpacc_biased), vmagic_bias_less_output_zero_point);
    input_a+=8;
    const __m128i vout_low = _mm256_castsi256_si128(vout);
    const __m128i vout_high = _mm256_extracti128_si256(vout, 1);
    const __m128i vout_packed = _mm_packs_epi32(vout_low, vout_high);
    __m128i vout_final = _mm_packs_epi16(vout_packed, vout_packed);
    _mm_storeu_si64((__m128i*) output, vout_final);
    output+=8;
  }

  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const __m256i va = _mm256_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input_a));
    const __m256i va_sub = _mm256_sub_epi32(va, vinput_zero_point);
    __m256i vcompare = _mm256_cmpgt_epi32(_mm256_setzero_si256(), va_sub);
    const __m256i vacc = _mm256_blendv_epi8(va_sub, _mm256_mullo_epi32(va_sub, vslope), vcompare);
    const __m256 vscale = _mm256_blendv_ps(vpositive_multiplier, vnegative_multiplier, _mm256_castsi256_ps(vcompare));
    const __m256 vfpacc = _mm256_mul_ps(_mm256_cvtepi32_ps(vacc), vscale);
    const __m256 vfpacc_clamped = _mm256_min_ps(_mm256_max_ps(vfpacc, voutput_min_less_zero_point), voutput_max_less_zero_point);
    const __m256 vfpacc_biased = _mm256_add_ps(vfpacc_clamped, vmagic_bias);
    const __m256i vout = _mm256_sub_epi32(_mm256_castps_si256(vfpacc_biased), vmagic_bias_less_output_zero_point);
    const __m128i vout_low = _mm256_castsi256_si128(vout);
    const __m128i vout_high = _mm256_extracti128_si256(vout, 1);
    const __m128i vout_packed = _mm_packs_epi32(vout_low, vout_high);
    __m128i vout_final = _mm_packs_epi16(vout_packed, vout_packed);
   
    if (batch & (4 * sizeof(int8_t))) {
      _mm_storeu_si32(output, vout_final);
      vout_final = _mm_srli_epi64(vout_final, 32);
      output += 4;
    }

    if (batch & (2 * sizeof(int8_t))) {
     _mm_storeu_si16(output, vout_final);
      vout_final = _mm_srli_epi32(vout_final, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) _mm_extract_epi8(vout_final, 0);
    }
  }
}
