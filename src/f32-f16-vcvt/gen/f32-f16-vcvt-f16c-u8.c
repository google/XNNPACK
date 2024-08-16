// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/f16c.c.in
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


void xnn_f32_f16_vcvt_ukernel__f16c_u8(
    size_t batch,
    const float* input,
    void* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  static const uint32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vf = _mm256_loadu_ps(input);
    input += 8;

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    const __m256 vf = _mm256_maskload_ps(input, vmask);

    __m128 vf_lo = _mm256_castps256_ps128(vf);
    if (batch & (4 * sizeof(float))) {
      _mm_storel_epi64((__m128i*) o, _mm_cvtps_ph(vf_lo, _MM_FROUND_TO_NEAREST_INT));
      vf_lo = _mm256_extractf128_ps(vf, 1);
      o += 4;
    }
    __m128i vh = _mm_cvtps_ph(vf_lo, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (2 * sizeof(float))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(float))) {
      *((uint16_t*) o) = _mm_extract_epi16(vh, 0);
    }
  }
}
