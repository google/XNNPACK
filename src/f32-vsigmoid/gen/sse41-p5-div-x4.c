// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/sse-p5-div.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vsigmoid_ukernel__sse41_p5_div_x4(
    size_t n,
    const float* x,
    float* y,
    const void* params) XNN_DISABLE_TSAN
{
  assert(n % sizeof(float) == 0);

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

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(x);

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
    vf = _mm_blendv_ps(_mm_sub_ps(vone, vf), vf, vx);

    _mm_storeu_ps(y, vf);

    x += 4;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const __m128 vx = _mm_loadu_ps(x);

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
    vf = _mm_blendv_ps(_mm_sub_ps(vone, vf), vf, vx);

    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vf);
      vf = _mm_movehl_ps(vf, vf);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vf);
    }
  }
}
