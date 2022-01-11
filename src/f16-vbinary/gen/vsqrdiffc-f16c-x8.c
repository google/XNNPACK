// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vopc-f16c.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


void xnn_f16_vsqrdiffc_ukernel__f16c_x8(
    size_t n,
    const void* restrict a_ptr,
    const void* restrict b_ptr,
    void* restrict y_ptr,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(a_ptr != NULL);
  assert(b_ptr != NULL);
  assert(y_ptr != NULL);

  const uint16_t* a = (const uint16_t*) a_ptr;
  const uint16_t* b = (const uint16_t*) b_ptr;
  uint16_t* y = (uint16_t*) y_ptr;


  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va, vb), _MM_FROUND_NO_EXC));
    vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vy, vy), _MM_FROUND_NO_EXC));


    _mm_storeu_si128((__m128i*) y, _mm256_cvtps_ph(vy, _MM_FROUND_NO_EXC));
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va, vb), _MM_FROUND_NO_EXC));
    vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vy, vy), _MM_FROUND_NO_EXC));


    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_NO_EXC);
    if (n & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) y, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      y += 4;
    }
    if (n & (2 * sizeof(uint16_t))) {
      *((uint32_t*) y) = (uint32_t) _mm_cvtsi128_si32(vh);
      vh = _mm_srli_epi64(vh, 32);
      y += 2;
    }
    if (n & (1 * sizeof(uint16_t))) {
      *y = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
