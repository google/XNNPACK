// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-vcvt/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <emmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vcvt.h"


void xnn_bf16_f32_vcvt_ukernel__sse2_u24(
    size_t batch,
    const xnn_bfloat16* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  const __m128i vzero = _mm_setzero_si128();
  for (; batch >= 24 * sizeof(xnn_bfloat16); batch -= 24 * sizeof(xnn_bfloat16)) {
    const __m128i vbf0 = _mm_loadu_si128((const __m128i*) i);
    const __m128i vbf1 = _mm_loadu_si128((const __m128i*) (i + 8));
    const __m128i vbf2 = _mm_loadu_si128((const __m128i*) (i + 16));
    i += 24;

    const __m128 vf0 = _mm_castsi128_ps(_mm_unpacklo_epi16(vzero, vbf0));
    const __m128 vf1 = _mm_castsi128_ps(_mm_unpackhi_epi16(vzero, vbf0));
    const __m128 vf2 = _mm_castsi128_ps(_mm_unpacklo_epi16(vzero, vbf1));
    const __m128 vf3 = _mm_castsi128_ps(_mm_unpackhi_epi16(vzero, vbf1));
    const __m128 vf4 = _mm_castsi128_ps(_mm_unpacklo_epi16(vzero, vbf2));
    const __m128 vf5 = _mm_castsi128_ps(_mm_unpackhi_epi16(vzero, vbf2));

    _mm_storeu_ps(output, vf0);
    _mm_storeu_ps(output + 4, vf1);
    _mm_storeu_ps(output + 8, vf2);
    _mm_storeu_ps(output + 12, vf3);
    _mm_storeu_ps(output + 16, vf4);
    _mm_storeu_ps(output + 20, vf5);
    output += 24;
  }
  for (; batch >= 8 * sizeof(xnn_bfloat16); batch -= 8 * sizeof(xnn_bfloat16)) {
    const __m128i vbf = _mm_loadu_si128((const __m128i*) i);
    i += 8;

    const __m128 vf_lo = _mm_castsi128_ps(_mm_unpacklo_epi16(vzero, vbf));
    const __m128 vf_hi = _mm_castsi128_ps(_mm_unpackhi_epi16(vzero, vbf));

    _mm_storeu_ps(output, vf_lo);
    _mm_storeu_ps(output + 4, vf_hi);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(xnn_bfloat16));
    assert(batch <= 7 * sizeof(xnn_bfloat16));
    const __m128i vbf = _mm_loadu_si128((const __m128i*) i);

    __m128 vf = _mm_castsi128_ps(_mm_unpacklo_epi16(vzero, vbf));
    if (batch & (4 * sizeof(xnn_bfloat16))) {
      _mm_storeu_ps(output, vf);
      output += 4;
      vf = _mm_castsi128_ps(_mm_unpackhi_epi16(vzero, vbf));
    }
    if (batch & (2 * sizeof(xnn_bfloat16))) {
      _mm_storel_pi((__m64*) output, vf);
      vf = _mm_movehl_ps(vf, vf);
      output += 2;
    }
    if (batch & (1 * sizeof(xnn_bfloat16))) {
      _mm_store_ss(output, vf);
    }
  }
}
