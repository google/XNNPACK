// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-vbinary/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vbinary.h"


static inline __m512 load_bf16_as_f32(const uint16_t* input) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*) input)), 16));
}

static inline __m512 maskz_load_bf16_as_f32(
    __mmask16 mask, const uint16_t* input) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask, input)), 16));
}

static inline __m256i convert_f32_to_bf16(__m512 value) {
  const __m512i vbias = _mm512_set1_epi32(0x7FFF);
  const __m512i vone = _mm512_set1_epi32(1);
  const __m512i vabs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512i vexp_mask = _mm512_set1_epi32(0x7F800000);
  const __m512i vquiet = _mm512_set1_epi32(0x00400000);

  __m512i vi = _mm512_castps_si512(value);
  const __m512i vlsb =
      _mm512_and_si512(_mm512_srli_epi32(vi, 16), vone);
  const __m512i vrounded =
      _mm512_add_epi32(_mm512_add_epi32(vi, vbias), vlsb);
  const __mmask16 vnanmask = _mm512_cmpgt_epu32_mask(
      _mm512_and_si512(vi, vabs_mask), vexp_mask);
  vi = _mm512_mask_or_epi32(vrounded, vnanmask, vi, vquiet);
  return _mm512_cvtepi32_epi16(_mm512_srli_epi32(vi, 16));
}

void xnn_bf16_vmaxc_ukernel__avx512skx_u32(
    size_t batch,
    const xnn_bfloat16* input_a,
    const xnn_bfloat16* input_b,
    xnn_bfloat16* output,
    const struct xnn_bf16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  uint16_t* o = (uint16_t*) output;
  const __m512 vb = _mm512_castsi512_ps(
      _mm512_set1_epi32((int) ((uint32_t) *((const uint16_t*) input_b) << 16)));
  for (; batch >= 32 * sizeof(xnn_bfloat16);
       batch -= 32 * sizeof(xnn_bfloat16)) {
    const __m512 va0 = load_bf16_as_f32(a + 0);
    const __m512 va1 = load_bf16_as_f32(a + 16);
    a += 32;

    __m512 vy0 = _mm512_max_ps(va0, vb);
    __m512 vy1 = _mm512_max_ps(va1, vb);

    _mm256_storeu_si256((__m256i*) (o + 0), convert_f32_to_bf16(vy0));
    _mm256_storeu_si256((__m256i*) (o + 16), convert_f32_to_bf16(vy1));
    o += 32;
  }
  for (; batch >= 16 * sizeof(xnn_bfloat16); batch -= 16 * sizeof(xnn_bfloat16)) {
    const __m512 va0 = load_bf16_as_f32(a);
    a += 16;

    __m512 vy0 = _mm512_max_ps(va0, vb);

    _mm256_storeu_si256((__m256i*) o, convert_f32_to_bf16(vy0));
    o += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    const size_t elements = batch / sizeof(xnn_bfloat16);
    assert(elements != 0 && elements < 16);
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << elements) - 1));

    const __m512 va0 = maskz_load_bf16_as_f32(vmask, a);

    __m512 vy0 = _mm512_max_ps(va0, vb);

    _mm256_mask_storeu_epi16(o, vmask, convert_f32_to_bf16(vy0));
  }
}
