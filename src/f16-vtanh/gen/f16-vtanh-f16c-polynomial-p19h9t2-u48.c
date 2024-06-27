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


void xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u48(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);


  const __m256 vneg_sat_cutoff = _mm256_load_ps(params->avx_polynomial_p19h9t2.neg_sat_cutoff);
  const __m256 vpos_sat_cutoff = _mm256_load_ps(params->avx_polynomial_p19h9t2.pos_sat_cutoff);
  const __m256 vc19 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c19);
  const __m256 vc17 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c17);
  const __m256 vc15 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c15);
  const __m256 vc13 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c13);
  const __m256 vc11 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c11);
  const __m256 vc9 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c9);
  const __m256 vc7 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c7);
  const __m256 vc5 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c5);
  const __m256 vc3 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c3);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 48 * sizeof(uint16_t); batch -= 48 * sizeof(uint16_t)) {
    __m256 vx0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vx1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    __m256 vx2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    __m256 vx3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    __m256 vx4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 32)));
    __m256 vx5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 40)));
    i += 48;

    vx0 = _mm256_max_ps(vneg_sat_cutoff, vx0);
    vx1 = _mm256_max_ps(vneg_sat_cutoff, vx1);
    vx2 = _mm256_max_ps(vneg_sat_cutoff, vx2);
    vx3 = _mm256_max_ps(vneg_sat_cutoff, vx3);
    vx4 = _mm256_max_ps(vneg_sat_cutoff, vx4);
    vx5 = _mm256_max_ps(vneg_sat_cutoff, vx5);
    vx0 = _mm256_min_ps(vpos_sat_cutoff, vx0);
    vx1 = _mm256_min_ps(vpos_sat_cutoff, vx1);
    vx2 = _mm256_min_ps(vpos_sat_cutoff, vx2);
    vx3 = _mm256_min_ps(vpos_sat_cutoff, vx3);
    vx4 = _mm256_min_ps(vpos_sat_cutoff, vx4);
    vx5 = _mm256_min_ps(vpos_sat_cutoff, vx5);

    const __m256 vt0 = _mm256_mul_ps(vx0, vx0);
    const __m256 vt1 = _mm256_mul_ps(vx1, vx1);
    const __m256 vt2 = _mm256_mul_ps(vx2, vx2);
    const __m256 vt3 = _mm256_mul_ps(vx3, vx3);
    const __m256 vt4 = _mm256_mul_ps(vx4, vx4);
    const __m256 vt5 = _mm256_mul_ps(vx5, vx5);

    __m256 vp0 = _mm256_add_ps(_mm256_mul_ps(vc19, vt0), vc17);
    __m256 vp1 = _mm256_add_ps(_mm256_mul_ps(vc19, vt1), vc17);
    __m256 vp2 = _mm256_add_ps(_mm256_mul_ps(vc19, vt2), vc17);
    __m256 vp3 = _mm256_add_ps(_mm256_mul_ps(vc19, vt3), vc17);
    __m256 vp4 = _mm256_add_ps(_mm256_mul_ps(vc19, vt4), vc17);
    __m256 vp5 = _mm256_add_ps(_mm256_mul_ps(vc19, vt5), vc17);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc15);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc15);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc15);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc15);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc15);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vc15);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc13);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc13);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc13);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc13);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc13);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vc13);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc11);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc11);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc11);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc11);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc11);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vc11);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc9);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc9);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc9);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc9);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc9);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vc9);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc7);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc7);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc7);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc7);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc7);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vc7);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc5);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc5);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc5);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc5);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc5);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vc5);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc3);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc3);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc3);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc3);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc3);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vc3);

    const __m256 vxt0 = _mm256_mul_ps(vx0, vt0);
    const __m256 vxt1 = _mm256_mul_ps(vx1, vt1);
    const __m256 vxt2 = _mm256_mul_ps(vx2, vt2);
    const __m256 vxt3 = _mm256_mul_ps(vx3, vt3);
    const __m256 vxt4 = _mm256_mul_ps(vx4, vt4);
    const __m256 vxt5 = _mm256_mul_ps(vx5, vt5);
    const __m256 vy0 = _mm256_add_ps(_mm256_mul_ps(vp0, vxt0), vx0);
    const __m256 vy1 = _mm256_add_ps(_mm256_mul_ps(vp1, vxt1), vx1);
    const __m256 vy2 = _mm256_add_ps(_mm256_mul_ps(vp2, vxt2), vx2);
    const __m256 vy3 = _mm256_add_ps(_mm256_mul_ps(vp3, vxt3), vx3);
    const __m256 vy4 = _mm256_add_ps(_mm256_mul_ps(vp4, vxt4), vx4);
    const __m256 vy5 = _mm256_add_ps(_mm256_mul_ps(vp5, vxt5), vx5);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy1, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vy2, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vy3, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 32), _mm256_cvtps_ph(vy4, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 40), _mm256_cvtps_ph(vy5, _MM_FROUND_TO_NEAREST_INT));
    o += 48;
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
