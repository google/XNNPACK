// Auto-generated file. Do not edit!
//   Template: src/f16-vtanh/avx-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"

void xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u48(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vsign_mask = _mm_load_si128((const __m128i*) params->avx_expm1minus_rr1_p3h2.sign_mask);
  const __m256 vsat_cutoff = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.sat_cutoff);
  const __m256 vlog2e = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.log2e);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.magic_bias);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.minus_ln2);
  const __m256 vc3 = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.c2);
  const __m256 vtwo = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.two);
  const __m256 vminus_one = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.minus_one);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 48 * sizeof(uint16_t); batch -= 48 * sizeof(uint16_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) i);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (i + 8));
    const __m128i vx2 = _mm_loadu_si128((const __m128i*) (i + 16));
    const __m128i vx3 = _mm_loadu_si128((const __m128i*) (i + 24));
    const __m128i vx4 = _mm_loadu_si128((const __m128i*) (i + 32));
    const __m128i vx5 = _mm_loadu_si128((const __m128i*) (i + 40));
    i += 48;

    const __m128i vabsx0 = _mm_or_si128(vx0, vsign_mask);
    const __m128i vabsx1 = _mm_or_si128(vx1, vsign_mask);
    const __m128i vabsx2 = _mm_or_si128(vx2, vsign_mask);
    const __m128i vabsx3 = _mm_or_si128(vx3, vsign_mask);
    const __m128i vabsx4 = _mm_or_si128(vx4, vsign_mask);
    const __m128i vabsx5 = _mm_or_si128(vx5, vsign_mask);

    __m256 vz0 = _mm256_cvtph_ps(vabsx0);
    const __m128i vinvsignx0 = _mm_xor_si128(vx0, vabsx0);
    __m256 vz1 = _mm256_cvtph_ps(vabsx1);
    const __m128i vinvsignx1 = _mm_xor_si128(vx1, vabsx1);
    __m256 vz2 = _mm256_cvtph_ps(vabsx2);
    const __m128i vinvsignx2 = _mm_xor_si128(vx2, vabsx2);
    __m256 vz3 = _mm256_cvtph_ps(vabsx3);
    const __m128i vinvsignx3 = _mm_xor_si128(vx3, vabsx3);
    __m256 vz4 = _mm256_cvtph_ps(vabsx4);
    const __m128i vinvsignx4 = _mm_xor_si128(vx4, vabsx4);
    __m256 vz5 = _mm256_cvtph_ps(vabsx5);
    const __m128i vinvsignx5 = _mm_xor_si128(vx5, vabsx5);

    vz0 = _mm256_max_ps(vsat_cutoff, vz0);
    __m256 vn0 = _mm256_fmadd_ps(vz0, vlog2e, vmagic_bias);
    vz1 = _mm256_max_ps(vsat_cutoff, vz1);
    __m256 vn1 = _mm256_fmadd_ps(vz1, vlog2e, vmagic_bias);
    vz2 = _mm256_max_ps(vsat_cutoff, vz2);
    __m256 vn2 = _mm256_fmadd_ps(vz2, vlog2e, vmagic_bias);
    vz3 = _mm256_max_ps(vsat_cutoff, vz3);
    __m256 vn3 = _mm256_fmadd_ps(vz3, vlog2e, vmagic_bias);
    vz4 = _mm256_max_ps(vsat_cutoff, vz4);
    __m256 vn4 = _mm256_fmadd_ps(vz4, vlog2e, vmagic_bias);
    vz5 = _mm256_max_ps(vsat_cutoff, vz5);
    __m256 vn5 = _mm256_fmadd_ps(vz5, vlog2e, vmagic_bias);

    const __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn0), 23));
    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    const __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn1), 23));
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    const __m256 vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn2), 23));
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    const __m256 vs3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn3), 23));
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    const __m256 vs4 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn4), 23));
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);
    const __m256 vs5 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn5), 23));
    vn5 = _mm256_sub_ps(vn5, vmagic_bias);

    const __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vz0);
    const __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vz1);
    const __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2, vz2);
    const __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2, vz3);
    const __m256 vt4 = _mm256_fmadd_ps(vn4, vminus_ln2, vz4);
    const __m256 vt5 = _mm256_fmadd_ps(vn5, vminus_ln2, vz5);

    __m256 vp0 = vc3;
    __m256 vp1 = vc3;
    __m256 vp2 = vc3;
    __m256 vp3 = vc3;
    __m256 vp4 = vc3;
    __m256 vp5 = vc3;
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc2);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc2);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc2);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vtwo);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vtwo);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vtwo);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vtwo);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vtwo);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vtwo);

    const __m256 vts0 = _mm256_mul_ps(vt0, vs0);
    const __m256 vsmo0 = _mm256_add_ps(vs0, vminus_one);
    const __m256 vts1 = _mm256_mul_ps(vt1, vs1);
    const __m256 vsmo1 = _mm256_add_ps(vs1, vminus_one);
    const __m256 vts2 = _mm256_mul_ps(vt2, vs2);
    const __m256 vsmo2 = _mm256_add_ps(vs2, vminus_one);
    const __m256 vts3 = _mm256_mul_ps(vt3, vs3);
    const __m256 vsmo3 = _mm256_add_ps(vs3, vminus_one);
    const __m256 vts4 = _mm256_mul_ps(vt4, vs4);
    const __m256 vsmo4 = _mm256_add_ps(vs4, vminus_one);
    const __m256 vts5 = _mm256_mul_ps(vt5, vs5);
    const __m256 vsmo5 = _mm256_add_ps(vs5, vminus_one);
    const __m256 vemo0 = _mm256_fmadd_ps(vp0, vts0, vsmo0);
    const __m256 vemo1 = _mm256_fmadd_ps(vp1, vts1, vsmo1);
    const __m256 vemo2 = _mm256_fmadd_ps(vp2, vts2, vsmo2);
    const __m256 vemo3 = _mm256_fmadd_ps(vp3, vts3, vsmo3);
    const __m256 vemo4 = _mm256_fmadd_ps(vp4, vts4, vsmo4);
    const __m256 vemo5 = _mm256_fmadd_ps(vp5, vts5, vsmo5);

    const __m256 vepo0 = _mm256_add_ps(vemo0, vtwo);
    const __m256 vepo1 = _mm256_add_ps(vemo1, vtwo);
    const __m256 vepo2 = _mm256_add_ps(vemo2, vtwo);
    const __m256 vepo3 = _mm256_add_ps(vemo3, vtwo);
    const __m256 vepo4 = _mm256_add_ps(vemo4, vtwo);
    const __m256 vepo5 = _mm256_add_ps(vemo5, vtwo);

    __m256 vy0 = _mm256_div_ps(vemo0, vepo0);
    __m256 vy1 = _mm256_div_ps(vemo1, vepo1);
    __m256 vy2 = _mm256_div_ps(vemo2, vepo2);
    __m256 vy3 = _mm256_div_ps(vemo3, vepo3);
    __m256 vy4 = _mm256_div_ps(vemo4, vepo4);
    __m256 vy5 = _mm256_div_ps(vemo5, vepo5);


    __m128i vh0 = _mm256_cvtps_ph(vy0, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh1 = _mm256_cvtps_ph(vy1, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh2 = _mm256_cvtps_ph(vy2, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh3 = _mm256_cvtps_ph(vy3, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh4 = _mm256_cvtps_ph(vy4, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh5 = _mm256_cvtps_ph(vy5, _MM_FROUND_TO_NEAREST_INT);
    vh0 = _mm_xor_si128(vh0, vinvsignx0);
    vh1 = _mm_xor_si128(vh1, vinvsignx1);
    vh2 = _mm_xor_si128(vh2, vinvsignx2);
    vh3 = _mm_xor_si128(vh3, vinvsignx3);
    vh4 = _mm_xor_si128(vh4, vinvsignx4);
    vh5 = _mm_xor_si128(vh5, vinvsignx5);

    _mm_storeu_si128((__m128i*) o, vh0);
    _mm_storeu_si128((__m128i*) (o + 8), vh1);
    _mm_storeu_si128((__m128i*) (o + 16), vh2);
    _mm_storeu_si128((__m128i*) (o + 24), vh3);
    _mm_storeu_si128((__m128i*) (o + 32), vh4);
    _mm_storeu_si128((__m128i*) (o + 40), vh5);
    o += 48;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) i);
    i += 8;

    const __m128i vabsx = _mm_or_si128(vx, vsign_mask);
    __m256 vz = _mm256_cvtph_ps(vabsx);

    const __m128i vinvsignx = _mm_xor_si128(vx, vabsx);
    vz = _mm256_max_ps(vsat_cutoff, vz);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = vc3;
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_fmadd_ps(vp, vts, vsmo);

    const __m256 vepo = _mm256_add_ps(vemo, vtwo);

    __m256 vy = _mm256_div_ps(vemo, vepo);


    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
    vh = _mm_xor_si128(vh, vinvsignx);

    _mm_storeu_si128((__m128i*) o, vh);
    o += 8;
  }
  if (batch != 0) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) i);

    const __m128i vabsx = _mm_or_si128(vx, vsign_mask);
    __m256 vz = _mm256_cvtph_ps(vabsx);

    const __m128i vinvsignx = _mm_xor_si128(vx, vabsx);
    vz = _mm256_max_ps(vsat_cutoff, vz);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = vc3;
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_fmadd_ps(vp, vts, vsmo);

    const __m256 vepo = _mm256_add_ps(vemo, vtwo);

    __m256 vy = _mm256_div_ps(vemo, vepo);


    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
    vh = _mm_xor_si128(vh, vinvsignx);

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
