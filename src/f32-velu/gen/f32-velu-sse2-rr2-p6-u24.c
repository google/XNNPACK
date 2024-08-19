// Auto-generated file. Do not edit!
//   Template: src/f32-velu/sse-rr2-p6.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/vunary.h"
#include "xnnpack/common.h"


void xnn_f32_velu_ukernel__sse2_rr2_p6_u24(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsat_cutoff = _mm_set1_ps(-0x1.154246p+4f);
  const __m128 vmagic_bias = _mm_set1_ps(0x1.8000FEp23f);
  const __m128 vlog2e = _mm_set1_ps(0x1.715476p+0f);
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.62E440p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(0x1.0105C6p-21f);
  const __m128 vc6 = _mm_set1_ps(0x1.6b7338p-10f);
  const __m128 vc5 = _mm_set1_ps(0x1.12278Ep-7f);
  const __m128 vc4 = _mm_set1_ps(0x1.555716p-5f);
  const __m128 vc3 = _mm_set1_ps(0x1.5554B0p-3f);
  const __m128 vc2 = _mm_set1_ps(0x1.FFFFFEp-2f);
  const __m128 vone = _mm_set1_ps(1.0f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc6);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);
  
  const __m128 vprescale = _mm_set1_ps(params->scalar.prescale);
  const __m128 valpha = _mm_set1_ps(params->scalar.alpha);
  const __m128 vbeta = _mm_set1_ps(params->scalar.beta);

  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    __m128 vx0123 = _mm_loadu_ps(input);
    __m128 vx4567 = _mm_loadu_ps(input + 4);
    __m128 vx89AB = _mm_loadu_ps(input + 8);
    __m128 vxCDEF = _mm_loadu_ps(input + 12);
    __m128 vxGHIJ = _mm_loadu_ps(input + 16);
    __m128 vxKLMN = _mm_loadu_ps(input + 20);
    input += 24;

    const __m128 vz0123 = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx0123, vprescale));
    const __m128 vz4567 = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx4567, vprescale));
    const __m128 vz89AB = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx89AB, vprescale));
    const __m128 vzCDEF = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vxCDEF, vprescale));
    const __m128 vzGHIJ = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vxGHIJ, vprescale));
    const __m128 vzKLMN = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vxKLMN, vprescale));

    __m128 vn0123 = _mm_add_ps(_mm_mul_ps(vz0123, vlog2e), vmagic_bias);
    __m128 vn4567 = _mm_add_ps(_mm_mul_ps(vz4567, vlog2e), vmagic_bias);
    __m128 vn89AB = _mm_add_ps(_mm_mul_ps(vz89AB, vlog2e), vmagic_bias);
    __m128 vnCDEF = _mm_add_ps(_mm_mul_ps(vzCDEF, vlog2e), vmagic_bias);
    __m128 vnGHIJ = _mm_add_ps(_mm_mul_ps(vzGHIJ, vlog2e), vmagic_bias);
    __m128 vnKLMN = _mm_add_ps(_mm_mul_ps(vzKLMN, vlog2e), vmagic_bias);

    __m128 vs0123 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn0123), 23));
    __m128 vs4567 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn4567), 23));
    __m128 vs89AB = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn89AB), 23));
    __m128 vsCDEF = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vnCDEF), 23));
    __m128 vsGHIJ = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vnGHIJ), 23));
    __m128 vsKLMN = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vnKLMN), 23));

    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);
    vn89AB = _mm_sub_ps(vn89AB, vmagic_bias);
    vnCDEF = _mm_sub_ps(vnCDEF, vmagic_bias);
    vnGHIJ = _mm_sub_ps(vnGHIJ, vmagic_bias);
    vnKLMN = _mm_sub_ps(vnKLMN, vmagic_bias);

    __m128 vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_hi), vz0123);
    __m128 vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_hi), vz4567);
    __m128 vt89AB = _mm_add_ps(_mm_mul_ps(vn89AB, vminus_ln2_hi), vz89AB);
    __m128 vtCDEF = _mm_add_ps(_mm_mul_ps(vnCDEF, vminus_ln2_hi), vzCDEF);
    __m128 vtGHIJ = _mm_add_ps(_mm_mul_ps(vnGHIJ, vminus_ln2_hi), vzGHIJ);
    __m128 vtKLMN = _mm_add_ps(_mm_mul_ps(vnKLMN, vminus_ln2_hi), vzKLMN);

    vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_lo), vt0123);
    vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_lo), vt4567);
    vt89AB = _mm_add_ps(_mm_mul_ps(vn89AB, vminus_ln2_lo), vt89AB);
    vtCDEF = _mm_add_ps(_mm_mul_ps(vnCDEF, vminus_ln2_lo), vtCDEF);
    vtGHIJ = _mm_add_ps(_mm_mul_ps(vnGHIJ, vminus_ln2_lo), vtGHIJ);
    vtKLMN = _mm_add_ps(_mm_mul_ps(vnKLMN, vminus_ln2_lo), vtKLMN);

    __m128 vp0123 = _mm_add_ps(_mm_mul_ps(vc6, vt0123), vc5);
    __m128 vp4567 = _mm_add_ps(_mm_mul_ps(vc6, vt4567), vc5);
    __m128 vp89AB = _mm_add_ps(_mm_mul_ps(vc6, vt89AB), vc5);
    __m128 vpCDEF = _mm_add_ps(_mm_mul_ps(vc6, vtCDEF), vc5);
    __m128 vpGHIJ = _mm_add_ps(_mm_mul_ps(vc6, vtGHIJ), vc5);
    __m128 vpKLMN = _mm_add_ps(_mm_mul_ps(vc6, vtKLMN), vc5);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc4);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc4);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc4);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc4);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc4);
    vpKLMN = _mm_add_ps(_mm_mul_ps(vpKLMN, vtKLMN), vc4);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc3);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc3);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc3);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc3);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc3);
    vpKLMN = _mm_add_ps(_mm_mul_ps(vpKLMN, vtKLMN), vc3);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc2);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc2);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc2);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc2);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc2);
    vpKLMN = _mm_add_ps(_mm_mul_ps(vpKLMN, vtKLMN), vc2);

    vp0123 = _mm_mul_ps(vp0123, vt0123);
    vp4567 = _mm_mul_ps(vp4567, vt4567);
    vp89AB = _mm_mul_ps(vp89AB, vt89AB);
    vpCDEF = _mm_mul_ps(vpCDEF, vtCDEF);
    vpGHIJ = _mm_mul_ps(vpGHIJ, vtGHIJ);
    vpKLMN = _mm_mul_ps(vpKLMN, vtKLMN);

    vt0123 = _mm_mul_ps(vt0123, vs0123);
    vs0123 = _mm_sub_ps(vs0123, vone);
    vt4567 = _mm_mul_ps(vt4567, vs4567);
    vs4567 = _mm_sub_ps(vs4567, vone);
    vt89AB = _mm_mul_ps(vt89AB, vs89AB);
    vs89AB = _mm_sub_ps(vs89AB, vone);
    vtCDEF = _mm_mul_ps(vtCDEF, vsCDEF);
    vsCDEF = _mm_sub_ps(vsCDEF, vone);
    vtGHIJ = _mm_mul_ps(vtGHIJ, vsGHIJ);
    vsGHIJ = _mm_sub_ps(vsGHIJ, vone);
    vtKLMN = _mm_mul_ps(vtKLMN, vsKLMN);
    vsKLMN = _mm_sub_ps(vsKLMN, vone);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vt0123);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vt4567);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vt89AB);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vtCDEF);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vtGHIJ);
    vpKLMN = _mm_add_ps(_mm_mul_ps(vpKLMN, vtKLMN), vtKLMN);

    const __m128 ve0123 = _mm_mul_ps(_mm_add_ps(vp0123, vs0123), valpha);
    const __m128 ve4567 = _mm_mul_ps(_mm_add_ps(vp4567, vs4567), valpha);
    const __m128 ve89AB = _mm_mul_ps(_mm_add_ps(vp89AB, vs89AB), valpha);
    const __m128 veCDEF = _mm_mul_ps(_mm_add_ps(vpCDEF, vsCDEF), valpha);
    const __m128 veGHIJ = _mm_mul_ps(_mm_add_ps(vpGHIJ, vsGHIJ), valpha);
    const __m128 veKLMN = _mm_mul_ps(_mm_add_ps(vpKLMN, vsKLMN), valpha);

    const __m128 vm0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx0123)));
    vx0123 = _mm_mul_ps(vx0123, vbeta);
    const __m128 vm4567 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx4567)));
    vx4567 = _mm_mul_ps(vx4567, vbeta);
    const __m128 vm89AB = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx89AB)));
    vx89AB = _mm_mul_ps(vx89AB, vbeta);
    const __m128 vmCDEF = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vxCDEF)));
    vxCDEF = _mm_mul_ps(vxCDEF, vbeta);
    const __m128 vmGHIJ = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vxGHIJ)));
    vxGHIJ = _mm_mul_ps(vxGHIJ, vbeta);
    const __m128 vmKLMN = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vxKLMN)));
    vxKLMN = _mm_mul_ps(vxKLMN, vbeta);

    const __m128 vy0123 = _mm_or_ps(_mm_and_ps(ve0123, vm0123), _mm_andnot_ps(vm0123, vx0123));
    const __m128 vy4567 = _mm_or_ps(_mm_and_ps(ve4567, vm4567), _mm_andnot_ps(vm4567, vx4567));
    const __m128 vy89AB = _mm_or_ps(_mm_and_ps(ve89AB, vm89AB), _mm_andnot_ps(vm89AB, vx89AB));
    const __m128 vyCDEF = _mm_or_ps(_mm_and_ps(veCDEF, vmCDEF), _mm_andnot_ps(vmCDEF, vxCDEF));
    const __m128 vyGHIJ = _mm_or_ps(_mm_and_ps(veGHIJ, vmGHIJ), _mm_andnot_ps(vmGHIJ, vxGHIJ));
    const __m128 vyKLMN = _mm_or_ps(_mm_and_ps(veKLMN, vmKLMN), _mm_andnot_ps(vmKLMN, vxKLMN));

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    _mm_storeu_ps(output + 8, vy89AB);
    _mm_storeu_ps(output + 12, vyCDEF);
    _mm_storeu_ps(output + 16, vyGHIJ);
    _mm_storeu_ps(output + 20, vyKLMN);
    output += 24;
  }
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

    const __m128 vm = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx)));
    vx = _mm_mul_ps(vx, vbeta);
    const __m128 vy = _mm_or_ps(_mm_and_ps(ve, vm), _mm_andnot_ps(vm, vx));

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

    const __m128 vm = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx)));
    vx = _mm_mul_ps(vx, vbeta);
    __m128 vy = _mm_or_ps(_mm_and_ps(ve, vm), _mm_andnot_ps(vm, vx));

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
