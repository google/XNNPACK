// Auto-generated file. Do not edit!
//   Template: src/f16-vtanh/avx-polynomial.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <math.h>

#include <immintrin.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u16(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const union xnn_f16_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vneg_sat_cutoff = _mm256_set1_ps(-0x1.1F0000p+2f);
  const __m256 vpos_sat_cutoff = _mm256_set1_ps(0x1.1F0000p+2f);
  XNN_FORCE_REALIZATION(vneg_sat_cutoff);
  XNN_FORCE_REALIZATION(vpos_sat_cutoff);
  const __m256 vc19 = _mm256_set1_ps(-0x1.1D841Cp-32f);
  XNN_FORCE_REALIZATION(vc19);
  const __m256 vc17 = _mm256_set1_ps(0x1.C4FC88p-26f);
  XNN_FORCE_REALIZATION(vc17);
  const __m256 vc15 = _mm256_set1_ps(-0x1.332066p-20f);
  XNN_FORCE_REALIZATION(vc15);
  const __m256 vc13 = _mm256_set1_ps(0x1.D1AEA2p-16f);
  XNN_FORCE_REALIZATION(vc13);
  const __m256 vc11 = _mm256_set1_ps(-0x1.B2782Ep-12f);
  XNN_FORCE_REALIZATION(vc11);
  const __m256 vc9 = _mm256_set1_ps(0x1.03CAEAp-8f);
  XNN_FORCE_REALIZATION(vc9);
  const __m256 vc7 = _mm256_set1_ps(-0x1.967628p-6f);
  XNN_FORCE_REALIZATION(vc7);
  const __m256 vc5 = _mm256_set1_ps(0x1.ABC35Cp-4f);
  XNN_FORCE_REALIZATION(vc5);
  const __m256 vc3 = _mm256_set1_ps(-0x1.499D08p-2f);
  XNN_FORCE_REALIZATION(vc3);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m256 vx0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vx1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    vx0 = _mm256_max_ps(vneg_sat_cutoff, vx0);
    vx1 = _mm256_max_ps(vneg_sat_cutoff, vx1);
    vx0 = _mm256_min_ps(vpos_sat_cutoff, vx0);
    vx1 = _mm256_min_ps(vpos_sat_cutoff, vx1);

    const __m256 vt0 = _mm256_mul_ps(vx0, vx0);
    const __m256 vt1 = _mm256_mul_ps(vx1, vx1);

    __m256 vp0 = _mm256_add_ps(_mm256_mul_ps(vc19, vt0), vc17);
    __m256 vp1 = _mm256_add_ps(_mm256_mul_ps(vc19, vt1), vc17);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc15);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc15);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc13);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc13);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc11);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc11);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc9);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc9);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc7);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc7);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc5);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc5);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc3);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc3);

    const __m256 vxt0 = _mm256_mul_ps(vx0, vt0);
    const __m256 vxt1 = _mm256_mul_ps(vx1, vt1);
    const __m256 vy0 = _mm256_add_ps(_mm256_mul_ps(vp0, vxt0), vx0);
    const __m256 vy1 = _mm256_add_ps(_mm256_mul_ps(vp1, vxt1), vx1);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy1, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vx = _mm256_max_ps(vneg_sat_cutoff, vx);
    vx = _mm256_min_ps(vpos_sat_cutoff, vx);

    const __m256 vt = _mm256_mul_ps(vx, vx);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc19, vt), vc17);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc15);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc13);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc11);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc9);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc7);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc5);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc3);

    const __m256 vxt = _mm256_mul_ps(vx, vt);
    const __m256 vy = _mm256_add_ps(_mm256_mul_ps(vp, vxt), vx);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if (batch != 0) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    vx = _mm256_max_ps(vneg_sat_cutoff, vx);
    vx = _mm256_min_ps(vpos_sat_cutoff, vx);

    const __m256 vt = _mm256_mul_ps(vx, vx);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc19, vt), vc17);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc15);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc13);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc11);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc9);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc7);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc5);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc3);

    const __m256 vxt = _mm256_mul_ps(vx, vt);
    const __m256 vy = _mm256_add_ps(_mm256_mul_ps(vp, vxt), vx);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);

    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
