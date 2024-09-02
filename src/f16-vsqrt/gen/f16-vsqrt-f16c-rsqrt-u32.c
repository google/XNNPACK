// Auto-generated file. Do not edit!
//   Template: src/f16-vsqrt/f16c-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


// In the following, instead of computing `sqrt(x)` on the converted `float`
// values, we compute `x * rsqrt(x)` where `rsqrt(x)` is the 12-bit
// approximation of the reciprocal square root.
//
// Since the result will be converted back to an `f16` value with a 10-bit
// mantissa, the 12-bit `rsqrt` approximation is more than sufficiently
// accurate.

void xnn_f16_vsqrt_ukernel__f16c_rsqrt_u32(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  const __m256 vinf = _mm256_set1_ps(INFINITY);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    __m256 vacc2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    __m256 vacc3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    i += 32;

    const __m256 vt0_0 = _mm256_rsqrt_ps(vacc0);
    const __m256 vt0_1 = _mm256_rsqrt_ps(vacc1);
    const __m256 vt0_2 = _mm256_rsqrt_ps(vacc2);
    const __m256 vt0_3 = _mm256_rsqrt_ps(vacc3);
    const __m256 vt1_0 = _mm256_cmp_ps(vt0_0, vinf, _CMP_LT_OQ);
    const __m256 vt1_1 = _mm256_cmp_ps(vt0_1, vinf, _CMP_LT_OQ);
    const __m256 vt1_2 = _mm256_cmp_ps(vt0_2, vinf, _CMP_LT_OQ);
    const __m256 vt1_3 = _mm256_cmp_ps(vt0_3, vinf, _CMP_LT_OQ);
    const __m256 vt2_0 = _mm256_and_ps(vt0_0, vt1_0);
    const __m256 vt2_1 = _mm256_and_ps(vt0_1, vt1_1);
    const __m256 vt2_2 = _mm256_and_ps(vt0_2, vt1_2);
    const __m256 vt2_3 = _mm256_and_ps(vt0_3, vt1_3);
    vacc0 = _mm256_mul_ps(vacc0, vt2_0);
    vacc1 = _mm256_mul_ps(vacc1, vt2_1);
    vacc2 = _mm256_mul_ps(vacc2, vt2_2);
    vacc3 = _mm256_mul_ps(vacc3, vt2_3);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vacc2, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vacc3, _MM_FROUND_TO_NEAREST_INT));
    o += 32;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    const __m256 vt0 = _mm256_rsqrt_ps(vacc);
    const __m256 vt1 = _mm256_cmp_ps(vt0, vinf, _CMP_LT_OQ);
    const __m256 vt2 = _mm256_and_ps(vt0, vt1);
    vacc = _mm256_mul_ps(vacc, vt2);
    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vt0 = _mm256_rsqrt_ps(vacc);
    const __m256 vt1 = _mm256_cmp_ps(vt0, vinf, _CMP_LT_OQ);
    const __m256 vt2 = _mm256_and_ps(vt0, vt1);
    vacc = _mm256_mul_ps(vacc, vt2);
    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      o += 4;
      vh = _mm_unpackhi_epi64(vh, vh);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      o += 2;
      vh = _mm_srli_epi64(vh, 32);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
