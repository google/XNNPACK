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


void xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u80(
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
  for (; batch >= 80 * sizeof(uint16_t); batch -= 80 * sizeof(uint16_t)) {
    __m256 vx0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vx1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    __m256 vx2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    __m256 vx3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    __m256 vx4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 32)));
    __m256 vx5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 40)));
    __m256 vx6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 48)));
    __m256 vx7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 56)));
    __m256 vx8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 64)));
    __m256 vx9 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 72)));
    i += 80;

    vx0 = _mm256_max_ps(vneg_sat_cutoff, vx0);
    vx1 = _mm256_max_ps(vneg_sat_cutoff, vx1);
    vx2 = _mm256_max_ps(vneg_sat_cutoff, vx2);
    vx3 = _mm256_max_ps(vneg_sat_cutoff, vx3);
    vx4 = _mm256_max_ps(vneg_sat_cutoff, vx4);
    vx5 = _mm256_max_ps(vneg_sat_cutoff, vx5);
    vx6 = _mm256_max_ps(vneg_sat_cutoff, vx6);
    vx7 = _mm256_max_ps(vneg_sat_cutoff, vx7);
    vx8 = _mm256_max_ps(vneg_sat_cutoff, vx8);
    vx9 = _mm256_max_ps(vneg_sat_cutoff, vx9);
    vx0 = _mm256_min_ps(vpos_sat_cutoff, vx0);
    vx1 = _mm256_min_ps(vpos_sat_cutoff, vx1);
    vx2 = _mm256_min_ps(vpos_sat_cutoff, vx2);
    vx3 = _mm256_min_ps(vpos_sat_cutoff, vx3);
    vx4 = _mm256_min_ps(vpos_sat_cutoff, vx4);
    vx5 = _mm256_min_ps(vpos_sat_cutoff, vx5);
    vx6 = _mm256_min_ps(vpos_sat_cutoff, vx6);
    vx7 = _mm256_min_ps(vpos_sat_cutoff, vx7);
    vx8 = _mm256_min_ps(vpos_sat_cutoff, vx8);
    vx9 = _mm256_min_ps(vpos_sat_cutoff, vx9);

    const __m256 vt0 = _mm256_mul_ps(vx0, vx0);
    const __m256 vt1 = _mm256_mul_ps(vx1, vx1);
    const __m256 vt2 = _mm256_mul_ps(vx2, vx2);
    const __m256 vt3 = _mm256_mul_ps(vx3, vx3);
    const __m256 vt4 = _mm256_mul_ps(vx4, vx4);
    const __m256 vt5 = _mm256_mul_ps(vx5, vx5);
    const __m256 vt6 = _mm256_mul_ps(vx6, vx6);
    const __m256 vt7 = _mm256_mul_ps(vx7, vx7);
    const __m256 vt8 = _mm256_mul_ps(vx8, vx8);
    const __m256 vt9 = _mm256_mul_ps(vx9, vx9);

    __m256 vp0 = vc19;
    __m256 vp1 = vc19;
    __m256 vp2 = vc19;
    __m256 vp3 = vc19;
    __m256 vp4 = vc19;
    __m256 vp5 = vc19;
    __m256 vp6 = vc19;
    __m256 vp7 = vc19;
    __m256 vp8 = vc19;
    __m256 vp9 = vc19;
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc17);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc17);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc17);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc17);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc17);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc17);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc17);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc17);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc17);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc17);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc15);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc15);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc15);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc15);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc15);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc15);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc15);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc15);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc15);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc15);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc13);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc13);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc13);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc13);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc13);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc13);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc13);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc13);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc13);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc13);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc11);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc11);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc11);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc11);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc11);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc11);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc11);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc11);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc11);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc11);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc9);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc9);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc9);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc9);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc9);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc9);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc9);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc9);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc9);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc9);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc7);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc7);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc7);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc7);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc7);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc7);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc7);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc7);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc7);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc7);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc5);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc5);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc5);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc5);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc5);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc5);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc5);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc5);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc5);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc5);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc3);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc3);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc3);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc3);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc3);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc3);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc3);

    const __m256 vxt0 = _mm256_mul_ps(vx0, vt0);
    const __m256 vxt1 = _mm256_mul_ps(vx1, vt1);
    const __m256 vxt2 = _mm256_mul_ps(vx2, vt2);
    const __m256 vxt3 = _mm256_mul_ps(vx3, vt3);
    const __m256 vxt4 = _mm256_mul_ps(vx4, vt4);
    const __m256 vxt5 = _mm256_mul_ps(vx5, vt5);
    const __m256 vxt6 = _mm256_mul_ps(vx6, vt6);
    const __m256 vxt7 = _mm256_mul_ps(vx7, vt7);
    const __m256 vxt8 = _mm256_mul_ps(vx8, vt8);
    const __m256 vxt9 = _mm256_mul_ps(vx9, vt9);
    const __m256 vy0 = _mm256_fmadd_ps(vp0, vxt0, vx0);
    const __m256 vy1 = _mm256_fmadd_ps(vp1, vxt1, vx1);
    const __m256 vy2 = _mm256_fmadd_ps(vp2, vxt2, vx2);
    const __m256 vy3 = _mm256_fmadd_ps(vp3, vxt3, vx3);
    const __m256 vy4 = _mm256_fmadd_ps(vp4, vxt4, vx4);
    const __m256 vy5 = _mm256_fmadd_ps(vp5, vxt5, vx5);
    const __m256 vy6 = _mm256_fmadd_ps(vp6, vxt6, vx6);
    const __m256 vy7 = _mm256_fmadd_ps(vp7, vxt7, vx7);
    const __m256 vy8 = _mm256_fmadd_ps(vp8, vxt8, vx8);
    const __m256 vy9 = _mm256_fmadd_ps(vp9, vxt9, vx9);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy1, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vy2, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vy3, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 32), _mm256_cvtps_ph(vy4, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 40), _mm256_cvtps_ph(vy5, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 48), _mm256_cvtps_ph(vy6, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 56), _mm256_cvtps_ph(vy7, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 64), _mm256_cvtps_ph(vy8, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 72), _mm256_cvtps_ph(vy9, _MM_FROUND_TO_NEAREST_INT));
    o += 80;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vx = _mm256_max_ps(vneg_sat_cutoff, vx);
    vx = _mm256_min_ps(vpos_sat_cutoff, vx);

    const __m256 vt = _mm256_mul_ps(vx, vx);

    __m256 vp = vc19;
    vp = _mm256_fmadd_ps(vp, vt, vc17);
    vp = _mm256_fmadd_ps(vp, vt, vc15);
    vp = _mm256_fmadd_ps(vp, vt, vc13);
    vp = _mm256_fmadd_ps(vp, vt, vc11);
    vp = _mm256_fmadd_ps(vp, vt, vc9);
    vp = _mm256_fmadd_ps(vp, vt, vc7);
    vp = _mm256_fmadd_ps(vp, vt, vc5);
    vp = _mm256_fmadd_ps(vp, vt, vc3);

    const __m256 vxt = _mm256_mul_ps(vx, vt);
    const __m256 vy = _mm256_fmadd_ps(vp, vxt, vx);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if (batch != 0) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    vx = _mm256_max_ps(vneg_sat_cutoff, vx);
    vx = _mm256_min_ps(vpos_sat_cutoff, vx);

    const __m256 vt = _mm256_mul_ps(vx, vx);

    __m256 vp = vc19;
    vp = _mm256_fmadd_ps(vp, vt, vc17);
    vp = _mm256_fmadd_ps(vp, vt, vc15);
    vp = _mm256_fmadd_ps(vp, vt, vc13);
    vp = _mm256_fmadd_ps(vp, vt, vc11);
    vp = _mm256_fmadd_ps(vp, vt, vc9);
    vp = _mm256_fmadd_ps(vp, vt, vc7);
    vp = _mm256_fmadd_ps(vp, vt, vc5);
    vp = _mm256_fmadd_ps(vp, vt, vc3);

    const __m256 vxt = _mm256_mul_ps(vx, vt);
    const __m256 vy = _mm256_fmadd_ps(vp, vxt, vx);

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
