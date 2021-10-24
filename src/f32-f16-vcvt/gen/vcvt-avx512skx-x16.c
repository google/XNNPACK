// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/avx512skx.c.in
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


void xnn_f32_f16_vcvt_ukernel__avx512skx_x16(
    size_t n,
    const float* input,
    void* output,
    const void* params)
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint16_t* o = (uint16_t*) output;
  for (; n >= 16 * sizeof(uint16_t); n -= 16 * sizeof(uint16_t)) {
    const __m512 vf = _mm512_loadu_ps(input);
    input += 16;

    _mm256_storeu_si256((__m256i*) o, _mm512_cvtps_ph(vf, _MM_FROUND_NO_EXC));
    o += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint16_t));
    assert(n <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on n).
    n >>= 1 /* log2(sizeof(uint16_t)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vf = _mm512_maskz_loadu_ps(vmask, input);
    const __m256i vh = _mm512_cvtps_ph(vf, _MM_FROUND_NO_EXC);
    _mm256_mask_storeu_epi16(o, vmask, vh);
  }
}
