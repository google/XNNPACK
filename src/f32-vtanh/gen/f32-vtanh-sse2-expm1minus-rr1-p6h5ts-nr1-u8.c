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

#include <emmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>


void xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u8(
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

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128 vz0123 = _mm_or_ps(vx0123, vsign_mask);
    const __m128 vz4567 = _mm_or_ps(vx4567, vsign_mask);

    const __m128 vinvsignx0123 = _mm_xor_ps(vx0123, vz0123);
    const __m128 vinvsignx4567 = _mm_xor_ps(vx4567, vz4567);

    const __m128 vm0123 = _mm_cmple_ps(vz0123, vsat_cutoff);
    const __m128 vm4567 = _mm_cmple_ps(vz4567, vsat_cutoff);

    __m128 vn0123 = _mm_add_ps(_mm_mul_ps(vz0123, vlog2e), vmagic_bias);
    __m128 vn4567 = _mm_add_ps(_mm_mul_ps(vz4567, vlog2e), vmagic_bias);

    const __m128 vs0123 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn0123), 23));
    const __m128 vs4567 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn4567), 23));

    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);

    const __m128 vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2), vz0123);
    const __m128 vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2), vz4567);

    __m128 vp0123 = _mm_add_ps(_mm_mul_ps(vc6, vt0123), vc5);
    __m128 vp4567 = _mm_add_ps(_mm_mul_ps(vc6, vt4567), vc5);
    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc4);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc4);
    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc3);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc3);
    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc2);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc2);
    vp0123 = _mm_sub_ps(_mm_mul_ps(vp0123, vt0123), vminus_two);
    vp4567 = _mm_sub_ps(_mm_mul_ps(vp4567, vt4567), vminus_two);

    const __m128 vts0123 = _mm_mul_ps(vt0123, vs0123);
    const __m128 vsmo0123 = _mm_add_ps(vs0123, vminus_one);
    const __m128 vts4567 = _mm_mul_ps(vt4567, vs4567);
    const __m128 vsmo4567 = _mm_add_ps(vs4567, vminus_one);
    const __m128 vemo0123 = _mm_add_ps(_mm_mul_ps(vp0123, vts0123), vsmo0123);
    const __m128 vemo4567 = _mm_add_ps(_mm_mul_ps(vp4567, vts4567), vsmo4567);

    const __m128 vepo0123 = _mm_sub_ps(vminus_two, vemo0123);
    const __m128 vepo4567 = _mm_sub_ps(vminus_two, vemo4567);

    __m128 vrepo0123 = _mm_rcp_ps(vepo0123);
    __m128 vrepo4567 = _mm_rcp_ps(vepo4567);
    vrepo0123 = _mm_mul_ps(vrepo0123, _mm_add_ps(_mm_mul_ps(vrepo0123, vepo0123), vminus_two));
    vrepo4567 = _mm_mul_ps(vrepo4567, _mm_add_ps(_mm_mul_ps(vrepo4567, vepo4567), vminus_two));

    __m128 vy0123 = _mm_mul_ps(vemo0123, vrepo0123);
    __m128 vy4567 = _mm_mul_ps(vemo4567, vrepo4567);

    vy0123 = _mm_or_ps(_mm_andnot_ps(vm0123, vy0123), _mm_and_ps(vminus_one, vm0123));
    vy4567 = _mm_or_ps(_mm_andnot_ps(vm4567, vy4567), _mm_and_ps(vminus_one, vm4567));

    vy0123 = _mm_xor_ps(vy0123, vinvsignx0123);
    vy4567 = _mm_xor_ps(vy4567, vinvsignx4567);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
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

    __m128 vy = _mm_mul_ps(vemo, vrepo);

    vy = _mm_or_ps(_mm_andnot_ps(vm, vy), _mm_and_ps(vminus_one, vm));

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

    __m128 vy = _mm_mul_ps(vemo, vrepo);

    vy = _mm_or_ps(_mm_andnot_ps(vm, vy), _mm_and_ps(vminus_one, vm));

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
