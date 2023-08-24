// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/sse-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <smmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>


void xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u20(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsign_mask = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.sign_mask);
  const __m128 vsat_cutoff = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.sat_cutoff);
  const __m128 vlog2e = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.log2e);
  const __m128 vmagic_bias = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.magic_bias);
  const __m128 vminus_ln2 = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.minus_ln2);
  const __m128 vc6 = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.c6);
  const __m128 vc5 = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.c5);
  const __m128 vc4 = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.c4);
  const __m128 vc3 = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.c3);
  const __m128 vc2 = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.c2);
  const __m128 vminus_two = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.minus_two);
  const __m128 vminus_one = _mm_load_ps(params->sse_expm1minus_rr1_p6h5.minus_one);

  for (; batch >= 20 * sizeof(float); batch -= 20 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    const __m128 vx89AB = _mm_loadu_ps(input + 8);
    const __m128 vxCDEF = _mm_loadu_ps(input + 12);
    const __m128 vxGHIJ = _mm_loadu_ps(input + 16);
    input += 20;

    const __m128 vz0123 = _mm_or_ps(vx0123, vsign_mask);
    const __m128 vz4567 = _mm_or_ps(vx4567, vsign_mask);
    const __m128 vz89AB = _mm_or_ps(vx89AB, vsign_mask);
    const __m128 vzCDEF = _mm_or_ps(vxCDEF, vsign_mask);
    const __m128 vzGHIJ = _mm_or_ps(vxGHIJ, vsign_mask);

    const __m128 vinvsignx0123 = _mm_xor_ps(vx0123, vz0123);
    const __m128 vinvsignx4567 = _mm_xor_ps(vx4567, vz4567);
    const __m128 vinvsignx89AB = _mm_xor_ps(vx89AB, vz89AB);
    const __m128 vinvsignxCDEF = _mm_xor_ps(vxCDEF, vzCDEF);
    const __m128 vinvsignxGHIJ = _mm_xor_ps(vxGHIJ, vzGHIJ);

    const __m128 vm0123 = _mm_cmple_ps(vz0123, vsat_cutoff);
    const __m128 vm4567 = _mm_cmple_ps(vz4567, vsat_cutoff);
    const __m128 vm89AB = _mm_cmple_ps(vz89AB, vsat_cutoff);
    const __m128 vmCDEF = _mm_cmple_ps(vzCDEF, vsat_cutoff);
    const __m128 vmGHIJ = _mm_cmple_ps(vzGHIJ, vsat_cutoff);

    __m128 vn0123 = _mm_add_ps(_mm_mul_ps(vz0123, vlog2e), vmagic_bias);
    __m128 vn4567 = _mm_add_ps(_mm_mul_ps(vz4567, vlog2e), vmagic_bias);
    __m128 vn89AB = _mm_add_ps(_mm_mul_ps(vz89AB, vlog2e), vmagic_bias);
    __m128 vnCDEF = _mm_add_ps(_mm_mul_ps(vzCDEF, vlog2e), vmagic_bias);
    __m128 vnGHIJ = _mm_add_ps(_mm_mul_ps(vzGHIJ, vlog2e), vmagic_bias);

    const __m128 vs0123 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn0123), 23));
    const __m128 vs4567 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn4567), 23));
    const __m128 vs89AB = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn89AB), 23));
    const __m128 vsCDEF = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vnCDEF), 23));
    const __m128 vsGHIJ = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vnGHIJ), 23));

    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);
    vn89AB = _mm_sub_ps(vn89AB, vmagic_bias);
    vnCDEF = _mm_sub_ps(vnCDEF, vmagic_bias);
    vnGHIJ = _mm_sub_ps(vnGHIJ, vmagic_bias);

    const __m128 vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2), vz0123);
    const __m128 vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2), vz4567);
    const __m128 vt89AB = _mm_add_ps(_mm_mul_ps(vn89AB, vminus_ln2), vz89AB);
    const __m128 vtCDEF = _mm_add_ps(_mm_mul_ps(vnCDEF, vminus_ln2), vzCDEF);
    const __m128 vtGHIJ = _mm_add_ps(_mm_mul_ps(vnGHIJ, vminus_ln2), vzGHIJ);

    __m128 vp0123 = _mm_add_ps(_mm_mul_ps(vc6, vt0123), vc5);
    __m128 vp4567 = _mm_add_ps(_mm_mul_ps(vc6, vt4567), vc5);
    __m128 vp89AB = _mm_add_ps(_mm_mul_ps(vc6, vt89AB), vc5);
    __m128 vpCDEF = _mm_add_ps(_mm_mul_ps(vc6, vtCDEF), vc5);
    __m128 vpGHIJ = _mm_add_ps(_mm_mul_ps(vc6, vtGHIJ), vc5);
    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc4);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc4);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc4);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc4);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc4);
    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc3);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc3);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc3);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc3);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc3);
    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc2);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc2);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vc2);
    vpCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtCDEF), vc2);
    vpGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vc2);
    vp0123 = _mm_sub_ps(_mm_mul_ps(vp0123, vt0123), vminus_two);
    vp4567 = _mm_sub_ps(_mm_mul_ps(vp4567, vt4567), vminus_two);
    vp89AB = _mm_sub_ps(_mm_mul_ps(vp89AB, vt89AB), vminus_two);
    vpCDEF = _mm_sub_ps(_mm_mul_ps(vpCDEF, vtCDEF), vminus_two);
    vpGHIJ = _mm_sub_ps(_mm_mul_ps(vpGHIJ, vtGHIJ), vminus_two);

    const __m128 vts0123 = _mm_mul_ps(vt0123, vs0123);
    const __m128 vsmo0123 = _mm_add_ps(vs0123, vminus_one);
    const __m128 vts4567 = _mm_mul_ps(vt4567, vs4567);
    const __m128 vsmo4567 = _mm_add_ps(vs4567, vminus_one);
    const __m128 vts89AB = _mm_mul_ps(vt89AB, vs89AB);
    const __m128 vsmo89AB = _mm_add_ps(vs89AB, vminus_one);
    const __m128 vtsCDEF = _mm_mul_ps(vtCDEF, vsCDEF);
    const __m128 vsmoCDEF = _mm_add_ps(vsCDEF, vminus_one);
    const __m128 vtsGHIJ = _mm_mul_ps(vtGHIJ, vsGHIJ);
    const __m128 vsmoGHIJ = _mm_add_ps(vsGHIJ, vminus_one);
    const __m128 vemo0123 = _mm_add_ps(_mm_mul_ps(vp0123, vts0123), vsmo0123);
    const __m128 vemo4567 = _mm_add_ps(_mm_mul_ps(vp4567, vts4567), vsmo4567);
    const __m128 vemo89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vts89AB), vsmo89AB);
    const __m128 vemoCDEF = _mm_add_ps(_mm_mul_ps(vpCDEF, vtsCDEF), vsmoCDEF);
    const __m128 vemoGHIJ = _mm_add_ps(_mm_mul_ps(vpGHIJ, vtsGHIJ), vsmoGHIJ);

    const __m128 vepo0123 = _mm_sub_ps(vminus_two, vemo0123);
    const __m128 vepo4567 = _mm_sub_ps(vminus_two, vemo4567);
    const __m128 vepo89AB = _mm_sub_ps(vminus_two, vemo89AB);
    const __m128 vepoCDEF = _mm_sub_ps(vminus_two, vemoCDEF);
    const __m128 vepoGHIJ = _mm_sub_ps(vminus_two, vemoGHIJ);

    __m128 vrepo0123 = _mm_rcp_ps(vepo0123);
    __m128 vrepo4567 = _mm_rcp_ps(vepo4567);
    __m128 vrepo89AB = _mm_rcp_ps(vepo89AB);
    __m128 vrepoCDEF = _mm_rcp_ps(vepoCDEF);
    __m128 vrepoGHIJ = _mm_rcp_ps(vepoGHIJ);
    vrepo0123 = _mm_mul_ps(vrepo0123, _mm_add_ps(_mm_mul_ps(vrepo0123, vepo0123), vminus_two));
    vrepo4567 = _mm_mul_ps(vrepo4567, _mm_add_ps(_mm_mul_ps(vrepo4567, vepo4567), vminus_two));
    vrepo89AB = _mm_mul_ps(vrepo89AB, _mm_add_ps(_mm_mul_ps(vrepo89AB, vepo89AB), vminus_two));
    vrepoCDEF = _mm_mul_ps(vrepoCDEF, _mm_add_ps(_mm_mul_ps(vrepoCDEF, vepoCDEF), vminus_two));
    vrepoGHIJ = _mm_mul_ps(vrepoGHIJ, _mm_add_ps(_mm_mul_ps(vrepoGHIJ, vepoGHIJ), vminus_two));
    vrepo0123 = _mm_mul_ps(vrepo0123, _mm_sub_ps(_mm_mul_ps(vrepo0123, vepo0123), vminus_two));
    vrepo4567 = _mm_mul_ps(vrepo4567, _mm_sub_ps(_mm_mul_ps(vrepo4567, vepo4567), vminus_two));
    vrepo89AB = _mm_mul_ps(vrepo89AB, _mm_sub_ps(_mm_mul_ps(vrepo89AB, vepo89AB), vminus_two));
    vrepoCDEF = _mm_mul_ps(vrepoCDEF, _mm_sub_ps(_mm_mul_ps(vrepoCDEF, vepoCDEF), vminus_two));
    vrepoGHIJ = _mm_mul_ps(vrepoGHIJ, _mm_sub_ps(_mm_mul_ps(vrepoGHIJ, vepoGHIJ), vminus_two));

    __m128 vy0123 = _mm_mul_ps(vemo0123, vrepo0123);
    __m128 vy4567 = _mm_mul_ps(vemo4567, vrepo4567);
    __m128 vy89AB = _mm_mul_ps(vemo89AB, vrepo89AB);
    __m128 vyCDEF = _mm_mul_ps(vemoCDEF, vrepoCDEF);
    __m128 vyGHIJ = _mm_mul_ps(vemoGHIJ, vrepoGHIJ);

    vy0123 = _mm_blendv_ps(vy0123, vminus_one, vm0123);
    vy4567 = _mm_blendv_ps(vy4567, vminus_one, vm4567);
    vy89AB = _mm_blendv_ps(vy89AB, vminus_one, vm89AB);
    vyCDEF = _mm_blendv_ps(vyCDEF, vminus_one, vmCDEF);
    vyGHIJ = _mm_blendv_ps(vyGHIJ, vminus_one, vmGHIJ);

    vy0123 = _mm_xor_ps(vy0123, vinvsignx0123);
    vy4567 = _mm_xor_ps(vy4567, vinvsignx4567);
    vy89AB = _mm_xor_ps(vy89AB, vinvsignx89AB);
    vyCDEF = _mm_xor_ps(vyCDEF, vinvsignxCDEF);
    vyGHIJ = _mm_xor_ps(vyGHIJ, vinvsignxGHIJ);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    _mm_storeu_ps(output + 8, vy89AB);
    _mm_storeu_ps(output + 12, vyCDEF);
    _mm_storeu_ps(output + 16, vyGHIJ);
    output += 20;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128 vz = _mm_or_ps(vx, vsign_mask);

    const __m128 vinvsignx = _mm_xor_ps(vx, vz);

    const __m128 vm = _mm_cmple_ps(vz, vsat_cutoff);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));

    vn = _mm_sub_ps(vn, vmagic_bias);

    const __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2), vz);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc6, vt), vc5);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_sub_ps(_mm_mul_ps(vp, vt), vminus_two);

    const __m128 vts = _mm_mul_ps(vt, vs);
    const __m128 vsmo = _mm_add_ps(vs, vminus_one);
    const __m128 vemo = _mm_add_ps(_mm_mul_ps(vp, vts), vsmo);

    const __m128 vepo = _mm_sub_ps(vminus_two, vemo);

    __m128 vrepo = _mm_rcp_ps(vepo);
    vrepo = _mm_mul_ps(vrepo, _mm_add_ps(_mm_mul_ps(vrepo, vepo), vminus_two));
    vrepo = _mm_mul_ps(vrepo, _mm_sub_ps(_mm_mul_ps(vrepo, vepo), vminus_two));

    __m128 vy = _mm_mul_ps(vemo, vrepo);

    vy = _mm_blendv_ps(vy, vminus_one, vm);

    vy = _mm_xor_ps(vy, vinvsignx);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);

    const __m128 vz = _mm_or_ps(vx, vsign_mask);

    const __m128 vinvsignx = _mm_xor_ps(vx, vz);

    const __m128 vm = _mm_cmple_ps(vz, vsat_cutoff);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));

    vn = _mm_sub_ps(vn, vmagic_bias);

    const __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2), vz);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc6, vt), vc5);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_sub_ps(_mm_mul_ps(vp, vt), vminus_two);

    const __m128 vts = _mm_mul_ps(vt, vs);
    const __m128 vsmo = _mm_add_ps(vs, vminus_one);
    const __m128 vemo = _mm_add_ps(_mm_mul_ps(vp, vts), vsmo);

    const __m128 vepo = _mm_sub_ps(vminus_two, vemo);

    __m128 vrepo = _mm_rcp_ps(vepo);
    vrepo = _mm_mul_ps(vrepo, _mm_add_ps(_mm_mul_ps(vrepo, vepo), vminus_two));
    vrepo = _mm_mul_ps(vrepo, _mm_sub_ps(_mm_mul_ps(vrepo, vepo), vminus_two));

    __m128 vy = _mm_mul_ps(vemo, vrepo);

    vy = _mm_blendv_ps(vy, vminus_one, vm);

    vy = _mm_xor_ps(vy, vinvsignx);

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
