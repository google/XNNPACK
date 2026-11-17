// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-vbinary/avx512bf16.c.in
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
  __m256i result = _mm512_cvtneps_pbh(value);
  const __m512i bits = _mm512_castps_si512(value);
  const __m512i abs_bits =
      _mm512_and_si512(bits, _mm512_set1_epi32(0x7FFFFFFF));
  const __mmask16 subnormal_mask = _mm512_cmplt_epu32_mask(
      _mm512_sub_epi32(abs_bits, _mm512_set1_epi32(1)),
      _mm512_set1_epi32(0x007FFFFF));
  // VCVTNEPS2BF16 flushes FP32 subnormals; preserve software rounding for them.
  if XNN_UNLIKELY(subnormal_mask != 0) {
    const __m512i lsb = _mm512_and_si512(
        _mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    const __m512i rounded = _mm512_add_epi32(
        _mm512_add_epi32(bits, _mm512_set1_epi32(0x7FFF)), lsb);
    result = _mm256_mask_mov_epi16(
        result, subnormal_mask,
        _mm512_cvtepi32_epi16(_mm512_srli_epi32(rounded, 16)));
  }
  return result;
}

void xnn_bf16_vminc_ukernel__avx512bf16_u16(
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
  for (; batch >= 16 * sizeof(xnn_bfloat16);
       batch -= 16 * sizeof(xnn_bfloat16)) {
    const __m512 va0 = load_bf16_as_f32(a + 0);
    a += 16;

    __m512 vy0 = _mm512_min_ps(va0, vb);

    _mm256_storeu_si256((__m256i*) (o + 0), convert_f32_to_bf16(vy0));
    o += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    const size_t elements = batch / sizeof(xnn_bfloat16);
    assert(elements != 0 && elements < 16);
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << elements) - 1));

    const __m512 va0 = maskz_load_bf16_as_f32(vmask, a);

    __m512 vy0 = _mm512_min_ps(va0, vb);

    _mm256_mask_storeu_epi16(o, vmask, convert_f32_to_bf16(vy0));
  }
}
