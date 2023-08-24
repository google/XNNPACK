// Auto-generated file. Do not edit!
//   Template: src/f32-velu/sse-rr2-p6.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include <xnnpack/vunary.h>
#include <xnnpack/common.h>


void xnn_f32_velu_ukernel__sse41_rr2_p6_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vprescale = _mm_load_ps(params->sse2_rr2_p6.prescale);
  const __m128 valpha = _mm_load_ps(params->sse2_rr2_p6.alpha);
  const __m128 vbeta = _mm_load_ps(params->sse2_rr2_p6.beta);
  const __m128 vsat_cutoff = _mm_load_ps(params->sse2_rr2_p6.sat_cutoff);
  const __m128 vmagic_bias = _mm_load_ps(params->sse2_rr2_p6.magic_bias);
  const __m128 vlog2e = _mm_load_ps(params->sse2_rr2_p6.log2e);
  const __m128 vminus_ln2_hi = _mm_load_ps(params->sse2_rr2_p6.minus_ln2_hi);
  const __m128 vminus_ln2_lo = _mm_load_ps(params->sse2_rr2_p6.minus_ln2_lo);
  const __m128 vc6 = _mm_load_ps(params->sse2_rr2_p6.c6);
  const __m128 vc5 = _mm_load_ps(params->sse2_rr2_p6.c5);
  const __m128 vc4 = _mm_load_ps(params->sse2_rr2_p6.c4);
  const __m128 vc3 = _mm_load_ps(params->sse2_rr2_p6.c3);
  const __m128 vc2 = _mm_load_ps(params->sse2_rr2_p6.c2);
  const __m128 vone = _mm_load_ps(params->sse2_rr2_p6.one);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128 vz = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx, vprescale));

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);
    __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc6, vt), vc5);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_mul_ps(vp, vt);

    vt = _mm_mul_ps(vt, vs);
    vs = _mm_sub_ps(vs, vone);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vt);
    const __m128 ve = _mm_mul_ps(_mm_add_ps(vp, vs), valpha);

    vx = _mm_mul_ps(vx, vbeta);
    const __m128 vy = _mm_blendv_ps(vx, ve, vx);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128 vx = _mm_loadu_ps(input);

    const __m128 vz = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx, vprescale));

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);
    __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc6, vt), vc5);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_mul_ps(vp, vt);

    vt = _mm_mul_ps(vt, vs);
    vs = _mm_sub_ps(vs, vone);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vt);
    const __m128 ve = _mm_mul_ps(_mm_add_ps(vp, vs), valpha);

    vx = _mm_mul_ps(vx, vbeta);
    __m128 vy = _mm_blendv_ps(vx, ve, vx);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}
