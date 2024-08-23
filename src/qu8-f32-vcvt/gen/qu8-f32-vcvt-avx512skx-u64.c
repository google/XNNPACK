// Auto-generated file. Do not edit!
//   Template: src/qs8-f32-vcvt/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vcvt.h"


void xnn_qu8_f32_vcvt_ukernel__avx512skx_u64(
    size_t batch,
    const uint8_t* input,
    float* output,
    const union xnn_qu8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vzero_point = _mm512_set1_epi32(params->scalar.zero_point);
  const __m512 vscale = _mm512_set1_ps(params->scalar.scale);
  XNN_FORCE_REALIZATION(vzero_point);
  XNN_FORCE_REALIZATION(vscale);
  for (; batch >= 64 * sizeof(uint8_t); batch -= 64 * sizeof(uint8_t)) {
    __m512i vx0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input));
    __m512i vxGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (input + 16)));
    __m512i vxWXYZ = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (input + 32)));
    __m512i vx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (input + 48)));
    input += 64;

    vx0123456789ABCDEF = _mm512_sub_epi32(vx0123456789ABCDEF, vzero_point);
    vxGHIJKLMNOPQRSTUV = _mm512_sub_epi32(vxGHIJKLMNOPQRSTUV, vzero_point);
    vxWXYZ = _mm512_sub_epi32(vxWXYZ, vzero_point);
    vx = _mm512_sub_epi32(vx, vzero_point);

    __m512 vy0123456789ABCDEF = _mm512_cvtepi32_ps(vx0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vxGHIJKLMNOPQRSTUV);
    __m512 vyWXYZ = _mm512_cvtepi32_ps(vxWXYZ);
    __m512 vy = _mm512_cvtepi32_ps(vx);

    vy0123456789ABCDEF = _mm512_mul_ps(vy0123456789ABCDEF, vscale);
    vyGHIJKLMNOPQRSTUV = _mm512_mul_ps(vyGHIJKLMNOPQRSTUV, vscale);
    vyWXYZ = _mm512_mul_ps(vyWXYZ, vscale);
    vy = _mm512_mul_ps(vy, vscale);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    _mm512_storeu_ps(output + 16, vyGHIJKLMNOPQRSTUV);
    _mm512_storeu_ps(output + 32, vyWXYZ);
    _mm512_storeu_ps(output + 48, vy);
    output += 64;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    __m512i vx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input));
    vx = _mm512_sub_epi32(vx, vzero_point);
    input += 16;

    __m512 vy = _mm512_cvtepi32_ps(vx);
    vy = _mm512_mul_ps(vy, vscale);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    // Prepare mask for valid elements (depends on batch).
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512i vx = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(vmask, input));
    vx = _mm512_sub_epi32(vx, vzero_point);

    __m512 vy = _mm512_cvtepi32_ps(vx);
    vy = _mm512_mul_ps(vy, vscale);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}
