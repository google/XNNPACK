// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vhswish/avx2.c.in
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
#include "src/xnnpack/vunary.h"



void xnn_qu8_vhswish_ukernel__avx2_u16(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qs8_vhswish_scalar_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vthree = _mm256_set1_ps(3.0f);
  const __m256 vsix = _mm256_set1_ps(6.0f);
  const __m256 vzero = _mm256_set1_ps(0.0f);
  const __m256 vsixth = _mm256_set1_ps(0x1.555556p-3f);

  const __m256 voutput_min= _mm256_set1_ps((int32_t) params->scalar.output_min);
  const __m256 voutput_max= _mm256_set1_ps((int32_t) params->scalar.output_max);
  const __m256i vinput_zero_point = _mm256_set1_epi32(params->scalar.input_zero_point);
  const __m256i voutput_zero_point = _mm256_set1_epi32(params->scalar.output_zero_point);
  const __m256 vinput_scale = _mm256_set1_ps(params->scalar.input_scale);
  const __m256 voutput_scale = _mm256_set1_ps(params->scalar.output_scale);
   const __m256 voutput_min_less_zero_point = _mm256_set1_ps((int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point);
  const __m256 voutput_max_less_zero_point = _mm256_set1_ps((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point);
  const __m256 vmagic_bias = _mm256_set1_ps(12582912.0f);
  const __m256i vmagic_bias_less_output_zero_point = _mm256_set1_epi32(INT32_C(0x4B400000) - (int32_t)params->scalar.output_zero_point);

  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    __m256i va0 = _mm256_cvtepu8_epi32(_mm_loadu_si64((const __m128i*) input));
    
    __m256i va1 = _mm256_cvtepu8_epi32(_mm_loadu_si64((const __m128i*) (input + 8)));
    input += 16;

     __m256i va0_sub = _mm256_sub_epi32(va0, vinput_zero_point);
     __m256 va0_mul = _mm256_mul_ps(_mm256_cvtepi32_ps(va0_sub), vinput_scale);
     __m256i va1_sub = _mm256_sub_epi32(va1, vinput_zero_point);
     __m256 va1_mul = _mm256_mul_ps(_mm256_cvtepi32_ps(va1_sub), vinput_scale);

    __m256 va0_sum = _mm256_add_ps(va0_mul, vthree);
    __m256 va0_clamped = _mm256_min_ps(_mm256_max_ps(va0_sum, vzero), vsix);
    __m256 vacc0 = _mm256_mul_ps(_mm256_mul_ps(va0_mul, vsixth), va0_clamped);
    vacc0 = _mm256_div_ps(vacc0, voutput_scale);
    vacc0 = _mm256_min_ps(_mm256_max_ps(vacc0, voutput_min_less_zero_point), voutput_max_less_zero_point);
    __m256 vfpacc_biased0 = _mm256_add_ps(vacc0, vmagic_bias);
    __m256i vout0 = _mm256_sub_epi32(_mm256_castps_si256(vfpacc_biased0), vmagic_bias_less_output_zero_point);
    __m256 va1_sum = _mm256_add_ps(va1_mul, vthree);
    __m256 va1_clamped = _mm256_min_ps(_mm256_max_ps(va1_sum, vzero), vsix);
    __m256 vacc1 = _mm256_mul_ps(_mm256_mul_ps(va1_mul, vsixth), va1_clamped);
    vacc1 = _mm256_div_ps(vacc1, voutput_scale);
    vacc1 = _mm256_min_ps(_mm256_max_ps(vacc1, voutput_min_less_zero_point), voutput_max_less_zero_point);
    __m256 vfpacc_biased1 = _mm256_add_ps(vacc1, vmagic_bias);
    __m256i vout1 = _mm256_sub_epi32(_mm256_castps_si256(vfpacc_biased1), vmagic_bias_less_output_zero_point);

    const __m128i vout_low0 = _mm256_castsi256_si128(vout0);
    const __m128i vout_high0 = _mm256_extracti128_si256(vout0, 1);
    const __m128i vout_packed0 = _mm_packs_epi32(vout_low0, vout_high0);
    __m128i vout_final0 = _mm_packus_epi16(vout_packed0, vout_packed0);
    const __m128i vout_low1 = _mm256_castsi256_si128(vout1);
    const __m128i vout_high1 = _mm256_extracti128_si256(vout1, 1);
    const __m128i vout_packed1 = _mm_packs_epi32(vout_low1, vout_high1);
    __m128i vout_final1 = _mm_packus_epi16(vout_packed1, vout_packed1);

    _mm_storeu_si64((__m128i*)(output), vout_final0);

    _mm_storeu_si64((__m128i*)(output + 8), vout_final1);

    output += 16;
  }

  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    __m256i va = _mm256_cvtepu8_epi32(_mm_loadu_si64((const __m128i*) input));
    __m256i va_sub = _mm256_sub_epi32(va, vinput_zero_point);
    __m256 va_mul = _mm256_mul_ps(_mm256_cvtepi32_ps(va_sub), vinput_scale);
    __m256 va_sum = _mm256_add_ps(va_mul, vthree);
    __m256 va_clamped = _mm256_min_ps(_mm256_max_ps(va_sum, vzero), vsix);
    __m256 vacc = _mm256_mul_ps(_mm256_mul_ps(va_mul, vsixth), va_clamped);
    vacc = _mm256_div_ps(vacc, voutput_scale);
    vacc = _mm256_min_ps(_mm256_max_ps(vacc, voutput_min_less_zero_point), voutput_max_less_zero_point);
    __m256 vfpacc_biased = _mm256_add_ps(vacc, vmagic_bias);
    __m256i vout = _mm256_sub_epi32(_mm256_castps_si256(vfpacc_biased), vmagic_bias_less_output_zero_point);
    input += 8;
    const __m128i vout_low = _mm256_castsi256_si128(vout);
    const __m128i vout_high = _mm256_extracti128_si256(vout, 1);
    const __m128i vout_packed = _mm_packs_epi32(vout_low, vout_high);
    __m128i vout_final = _mm_packus_epi16(vout_packed, vout_packed);
    _mm_storeu_si64((__m128i*) output, vout_final);
    output += 8;
  }

  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    __m256i va = _mm256_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input));
    __m256i va_sub = _mm256_sub_epi32(va, vinput_zero_point);
    __m256 va_mul = _mm256_mul_ps(_mm256_cvtepi32_ps(va_sub), vinput_scale);
    __m256 va_sum = _mm256_add_ps(va_mul, vthree);
    __m256 va_clamped = _mm256_min_ps(_mm256_max_ps(va_sum, vzero), vsix);
    __m256 vacc = _mm256_mul_ps(_mm256_mul_ps(va_mul, vsixth), va_clamped);
    vacc = _mm256_div_ps(vacc, voutput_scale);
    vacc = _mm256_min_ps(_mm256_max_ps(vacc, voutput_min_less_zero_point), voutput_max_less_zero_point);
    __m256 vfpacc_biased = _mm256_add_ps(vacc, vmagic_bias);
    __m256i vout = _mm256_sub_epi32(_mm256_castps_si256(vfpacc_biased), vmagic_bias_less_output_zero_point);

    const __m128i vout_low = _mm256_castsi256_si128(vout);
    const __m128i vout_high = _mm256_extracti128_si256(vout, 1);
    const __m128i vout_packed = _mm_packs_epi32(vout_low, vout_high);
    __m128i vout_final = _mm_packus_epi16(vout_packed, vout_packed);

    if (batch & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(output, vout_final);
      vout_final = _mm_srli_epi64(vout_final, 32);
      output += 4;
    }

    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storeu_si16(output, vout_final);
      vout_final = _mm_srli_epi32(vout_final, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) _mm_extract_epi8(vout_final, 0);
    }
  }
}
