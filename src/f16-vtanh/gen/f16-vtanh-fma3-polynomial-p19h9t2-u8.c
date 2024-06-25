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


void xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u8(
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
