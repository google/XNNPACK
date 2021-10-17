// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vcvt.h>


void xnn_f16_f32_vcvt_ukernel__avx512skx_x32(
    size_t n,
    const void* input,
    float* output,
    const void* params)
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const __m512 vacc0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    const __m512 vacc1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    i += 32;

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vacc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));

    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vacc = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}
