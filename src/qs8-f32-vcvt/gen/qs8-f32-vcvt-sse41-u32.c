// Auto-generated file. Do not edit!
//   Template: src/qs8-f32-vcvt/sse4.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vcvt.h"


void xnn_qs8_f32_vcvt_ukernel__sse41_u32(
    size_t batch,
    const int8_t* input,
    float* output,
    const union xnn_qs8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vzero_point = _mm_set1_epi32(params->scalar.zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  XNN_FORCE_REALIZATION(vzero_point);
  XNN_FORCE_REALIZATION(vscale);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    __m128i vx0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input)));
    __m128i vx4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 4)));
    __m128i vx89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 8)));
    __m128i vxCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 12)));
    __m128i vxGHIJ = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 16)));
    __m128i vxKLMN = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 20)));
    __m128i vxOPQR = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 24)));
    __m128i vxSTUV = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 28)));
    input += 32;

    vx0123 = _mm_sub_epi32(vx0123, vzero_point);
    vx4567 = _mm_sub_epi32(vx4567, vzero_point);
    vx89AB = _mm_sub_epi32(vx89AB, vzero_point);
    vxCDEF = _mm_sub_epi32(vxCDEF, vzero_point);
    vxGHIJ = _mm_sub_epi32(vxGHIJ, vzero_point);
    vxKLMN = _mm_sub_epi32(vxKLMN, vzero_point);
    vxOPQR = _mm_sub_epi32(vxOPQR, vzero_point);
    vxSTUV = _mm_sub_epi32(vxSTUV, vzero_point);

    __m128 vy0123 = _mm_cvtepi32_ps(vx0123);
    __m128 vy4567 = _mm_cvtepi32_ps(vx4567);
    __m128 vy89AB = _mm_cvtepi32_ps(vx89AB);
    __m128 vyCDEF = _mm_cvtepi32_ps(vxCDEF);
    __m128 vyGHIJ = _mm_cvtepi32_ps(vxGHIJ);
    __m128 vyKLMN = _mm_cvtepi32_ps(vxKLMN);
    __m128 vyOPQR = _mm_cvtepi32_ps(vxOPQR);
    __m128 vySTUV = _mm_cvtepi32_ps(vxSTUV);

    vy0123 = _mm_mul_ps(vy0123, vscale);
    vy4567 = _mm_mul_ps(vy4567, vscale);
    vy89AB = _mm_mul_ps(vy89AB, vscale);
    vyCDEF = _mm_mul_ps(vyCDEF, vscale);
    vyGHIJ = _mm_mul_ps(vyGHIJ, vscale);
    vyKLMN = _mm_mul_ps(vyKLMN, vscale);
    vyOPQR = _mm_mul_ps(vyOPQR, vscale);
    vySTUV = _mm_mul_ps(vySTUV, vscale);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    _mm_storeu_ps(output + 8, vy89AB);
    _mm_storeu_ps(output + 12, vyCDEF);
    _mm_storeu_ps(output + 16, vyGHIJ);
    _mm_storeu_ps(output + 20, vyKLMN);
    _mm_storeu_ps(output + 24, vyOPQR);
    _mm_storeu_ps(output + 28, vySTUV);
    output += 32;
  }
  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    __m128i vx = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input)));
    vx = _mm_sub_epi32(vx, vzero_point);
    input += 4;

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, vscale);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 3 * sizeof(int8_t));

    __m128i vx = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input)));
    vx = _mm_sub_epi32(vx, vzero_point);

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, vscale);

    if (batch & (2 * sizeof(int8_t))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      _mm_store_ss(output, vy);
    }
  }
}
