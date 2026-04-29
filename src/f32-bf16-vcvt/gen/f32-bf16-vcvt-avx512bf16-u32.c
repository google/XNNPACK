// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-bf16-vcvt/avx512bf16.c.in
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


void xnn_f32_bf16_vcvt_ukernel__avx512bf16_u32(
    size_t batch,
    const float* input,
    xnn_bfloat16* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const __m512 vf0 = _mm512_loadu_ps(input);
    const __m512 vf1 = _mm512_loadu_ps(input + 16);
    input += 32;

    _mm256_storeu_si256((__m256i*) o, (__m256i) _mm512_cvtneps_pbh(vf0));
    _mm256_storeu_si256((__m256i*) (o + 16), (__m256i) _mm512_cvtneps_pbh(vf1));
    o += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vf = _mm512_loadu_ps(input);
    input += 16;

    _mm256_storeu_si256((__m256i*) o, (__m256i) _mm512_cvtneps_pbh(vf));
    o += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vf = _mm512_maskz_loadu_ps(vmask, input);
    const __m256i vbf = (__m256i) _mm512_cvtneps_pbh(vf);
    _mm256_mask_storeu_epi16(o, vmask, vbf);
  }
}
