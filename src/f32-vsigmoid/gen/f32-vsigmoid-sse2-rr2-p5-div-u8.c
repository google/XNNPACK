// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/sse-rr2-p5-div.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsign_mask = _mm_set1_ps(-0.0f);
  const __m128 vmagic_bias = _mm_set1_ps(0x1.8000FEp23f);
  const __m128 vlog2e = _mm_set1_ps(0x1.715476p0f);
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.62E400p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(-0x1.7F7D1Cp-20f);
  const __m128 vc5 = _mm_set1_ps(0x1.0F9F9Cp-7f);
  const __m128 vc4 = _mm_set1_ps(0x1.573A1Ap-5f);
  const __m128 vc3 = _mm_set1_ps(0x1.555A80p-3f);
  const __m128 vc2 = _mm_set1_ps(0x1.FFFDC6p-2f);
  const __m128 vc1 = _mm_set1_ps(0x1.FFFFF6p-1f);
  const __m128 vone = _mm_set1_ps(1.0f);
  const __m128 vdenorm_cutoff = _mm_set1_ps(-0x1.5D589Ep+6f);

  XNN_FORCE_REALIZATION(vsign_mask);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);

    const __m128 vz0123 = _mm_or_ps(vx0123, vsign_mask);
    const __m128 vz4567 = _mm_or_ps(vx4567, vsign_mask);

    __m128 vn0123 = _mm_add_ps(_mm_mul_ps(vz0123, vlog2e), vmagic_bias);
    __m128 vn4567 = _mm_add_ps(_mm_mul_ps(vz4567, vlog2e), vmagic_bias);

    const __m128 vs0123 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn0123), 23));
    const __m128 vs4567 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn4567), 23));

    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);

    __m128 vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_hi), vz0123);
    __m128 vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_hi), vz4567);

    vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_lo), vt0123);
    vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_lo), vt4567);

    __m128 vp0123 = _mm_add_ps(_mm_mul_ps(vc5, vt0123), vc4);
    __m128 vp4567 = _mm_add_ps(_mm_mul_ps(vc5, vt4567), vc4);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc3);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc3);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc2);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc2);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc1);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc1);

    vt0123 = _mm_mul_ps(vt0123, vs0123);
    vt4567 = _mm_mul_ps(vt4567, vs4567);

    __m128 ve0123 = _mm_add_ps(_mm_mul_ps(vt0123, vp0123), vs0123);
    __m128 ve4567 = _mm_add_ps(_mm_mul_ps(vt4567, vp4567), vs4567);

    __m128 vd0123 = _mm_add_ps(ve0123, vone);
    __m128 vd4567 = _mm_add_ps(ve4567, vone);

    __m128 vf0123 = _mm_div_ps(ve0123, vd0123);
    __m128 vf4567 = _mm_div_ps(ve4567, vd4567);

    vf0123 = _mm_andnot_ps(_mm_cmplt_ps(vz0123, vdenorm_cutoff), vf0123);
    vf4567 = _mm_andnot_ps(_mm_cmplt_ps(vz4567, vdenorm_cutoff), vf4567);

    const __m128 vm0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx0123)));
    const __m128 vm4567 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx4567)));

    vf0123 = _mm_or_ps(_mm_and_ps(vf0123, vm0123), _mm_andnot_ps(vm0123, _mm_sub_ps(vone, vf0123)));
    vf4567 = _mm_or_ps(_mm_and_ps(vf4567, vm4567), _mm_andnot_ps(vm4567, _mm_sub_ps(vone, vf4567)));

    _mm_storeu_ps(output, vf0123);
    _mm_storeu_ps(output + 4, vf4567);

    input += 8;
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);

    const __m128 vz = _mm_or_ps(vx, vsign_mask);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);
    const __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc5, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc1);

    vt = _mm_mul_ps(vt, vs);
    __m128 ve = _mm_add_ps(_mm_mul_ps(vt, vp), vs);

    __m128 vd = _mm_add_ps(ve, vone);
    __m128 vf = _mm_div_ps(ve, vd);

    vf = _mm_andnot_ps(_mm_cmplt_ps(vz, vdenorm_cutoff), vf);
    const __m128 vm = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx)));
    vf = _mm_or_ps(_mm_and_ps(vf, vm), _mm_andnot_ps(vm, _mm_sub_ps(vone, vf)));

    _mm_storeu_ps(output, vf);

    input += 4;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);

    const __m128 vz = _mm_or_ps(vx, vsign_mask);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);
    const __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc5, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc1);

    vt = _mm_mul_ps(vt, vs);
    __m128 ve = _mm_add_ps(_mm_mul_ps(vt, vp), vs);

    __m128 vd = _mm_add_ps(ve, vone);
    __m128 vf = _mm_div_ps(ve, vd);

    vf = _mm_andnot_ps(_mm_cmplt_ps(vz, vdenorm_cutoff), vf);
    const __m128 vm = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx)));
    vf = _mm_or_ps(_mm_and_ps(vf, vm), _mm_andnot_ps(vm, _mm_sub_ps(vone, vf)));

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vf);
      vf = _mm_movehl_ps(vf, vf);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vf);
    }
  }
}
