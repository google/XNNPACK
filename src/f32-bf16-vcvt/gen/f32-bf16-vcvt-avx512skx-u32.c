// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-bf16-vcvt/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/vcvt.h"


void xnn_f32_bf16_vcvt_ukernel__avx512skx_u32(
    size_t batch,
    const float* input,
    xnn_bfloat16* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vbias = _mm512_set1_epi32(0x7FFFu);
  const __m512i vone = _mm512_set1_epi32(1u);
  const __m512i vabs_mask = _mm512_set1_epi32(0x7FFFFFFFu);
  const __m512i vexp_mask = _mm512_set1_epi32(0x7F800000u);
  const __m512i vquiet = _mm512_set1_epi32(0x00400000u);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512i vi0 = _mm512_castps_si512(_mm512_loadu_ps(input + 0));
    __m512i vi1 = _mm512_castps_si512(_mm512_loadu_ps(input + 16));
    input += 32;

    const __m512i vlsb0 = _mm512_and_si512(_mm512_srli_epi32(vi0, 16), vone);
    const __m512i vlsb1 = _mm512_and_si512(_mm512_srli_epi32(vi1, 16), vone);

    const __m512i vrounded0 = _mm512_add_epi32(_mm512_add_epi32(vi0, vbias), vlsb0);
    const __m512i vrounded1 = _mm512_add_epi32(_mm512_add_epi32(vi1, vbias), vlsb1);

    const __mmask16 vnanmask0 = _mm512_cmpgt_epu32_mask(_mm512_and_si512(vi0, vabs_mask), vexp_mask);
    const __mmask16 vnanmask1 = _mm512_cmpgt_epu32_mask(_mm512_and_si512(vi1, vabs_mask), vexp_mask);

    vi0 = _mm512_mask_or_epi32(vrounded0, vnanmask0, vi0, vquiet);
    vi1 = _mm512_mask_or_epi32(vrounded1, vnanmask1, vi1, vquiet);

    const __m256i vbf0 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(vi0, 16));
    const __m256i vbf1 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(vi1, 16));

    _mm256_storeu_si256((__m256i*) (o + 0), vbf0);
    _mm256_storeu_si256((__m256i*) (o + 16), vbf1);
    o += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512i vi = _mm512_castps_si512(_mm512_loadu_ps(input));
    input += 16;

    const __m512i vlsb = _mm512_and_si512(_mm512_srli_epi32(vi, 16), vone);
    const __m512i vrounded = _mm512_add_epi32(_mm512_add_epi32(vi, vbias), vlsb);
    const __mmask16 vnanmask = _mm512_cmpgt_epu32_mask(_mm512_and_si512(vi, vabs_mask), vexp_mask);
    vi = _mm512_mask_or_epi32(vrounded, vnanmask, vi, vquiet);
    const __m256i vbf = _mm512_cvtepi32_epi16(_mm512_srli_epi32(vi, 16));

    _mm256_storeu_si256((__m256i*) o, vbf);
    o += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512i vi = _mm512_castps_si512(_mm512_maskz_loadu_ps(vmask, input));
    const __m512i vlsb = _mm512_and_si512(_mm512_srli_epi32(vi, 16), vone);
    const __m512i vrounded = _mm512_add_epi32(_mm512_add_epi32(vi, vbias), vlsb);
    const __mmask16 vnanmask = _mm512_cmpgt_epu32_mask(_mm512_and_si512(vi, vabs_mask), vexp_mask);
    vi = _mm512_mask_or_epi32(vrounded, vnanmask, vi, vquiet);
    const __m256i vbf = _mm512_cvtepi32_epi16(_mm512_srli_epi32(vi, 16));

    _mm256_mask_storeu_epi16(o, vmask, vbf);
  }
}
