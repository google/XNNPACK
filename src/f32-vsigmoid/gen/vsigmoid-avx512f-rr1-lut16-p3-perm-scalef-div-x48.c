// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vunary.h>


void xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x48(
    size_t n,
    const float* x,
    float* y,
    const void* params)
{
  assert(n % sizeof(float) == 0);

  const __m512i vsign_mask = _mm512_set1_epi32(0x80000000);
  const __m512 vmagic_bias = _mm512_set1_ps(0x1.800000p19f);
  const __m512 vlog2e = _mm512_set1_ps(0x1.715476p0f);
  const __m512 vtable = _mm512_set_ps(
    0x1.EA4AFAp+0f, 0x1.D5818Ep+0f, 0x1.C199BEp+0f, 0x1.AE89FAp+0f,
    0x1.9C4918p+0f, 0x1.8ACE54p+0f, 0x1.7A1148p+0f, 0x1.6A09E6p+0f,
    0x1.5AB07Ep+0f, 0x1.4BFDAEp+0f, 0x1.3DEA64p+0f, 0x1.306FE0p+0f,
    0x1.2387A6p+0f, 0x1.172B84p+0f, 0x1.0B5586p+0f, 0x1.000000p+0f);
  const __m512 vminus_ln2 = _mm512_set1_ps(-0x1.62E43p-1f);
  const __m512 vc3 = _mm512_set1_ps(0x1.55559Ap-3f);
  const __m512 vc2 = _mm512_set1_ps(0x1.00021Ep-1f);
  const __m512 vone = _mm512_set1_ps(1.0f);

  for (; n >= 48 * sizeof(float); n -= 48 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(x);
    const __m512 vx1 = _mm512_loadu_ps(x + 16);
    const __m512 vx2 = _mm512_loadu_ps(x + 32);
    x += 48;

    const __m512 vz0 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx0), vsign_mask));
    const __m512 vz1 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx1), vsign_mask));
    const __m512 vz2 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx2), vsign_mask));

    __m512 vn0 = _mm512_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vz2, vlog2e, vmagic_bias);

    const __m512 vl0 = _mm512_permutexvar_ps(_mm512_castps_si512(vn0), vtable);
    const __m512 vl1 = _mm512_permutexvar_ps(_mm512_castps_si512(vn1), vtable);
    const __m512 vl2 = _mm512_permutexvar_ps(_mm512_castps_si512(vn2), vtable);

    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);

    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2, vz0);
    __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2, vz1);
    __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2, vz2);

    __m512 vp0 = _mm512_fmadd_ps(vt0, vc3, vc2);
    __m512 vp1 = _mm512_fmadd_ps(vt1, vc3, vc2);
    __m512 vp2 = _mm512_fmadd_ps(vt2, vc3, vc2);

    vp0 = _mm512_mul_ps(vp0, vt0);
    vp1 = _mm512_mul_ps(vp1, vt1);
    vp2 = _mm512_mul_ps(vp2, vt2);

    vp0 = _mm512_fmadd_ps(vt0, vp0, vt0);
    vp1 = _mm512_fmadd_ps(vt1, vp1, vt1);
    vp2 = _mm512_fmadd_ps(vt2, vp2, vt2);

    vp0 = _mm512_fmadd_ps(vl0, vp0, vl0);
    vp1 = _mm512_fmadd_ps(vl1, vp1, vl1);
    vp2 = _mm512_fmadd_ps(vl2, vp2, vl2);

    const __m512 ve0 = _mm512_scalef_ps(vp0, vn0);
    const __m512 ve1 = _mm512_scalef_ps(vp1, vn1);
    const __m512 ve2 = _mm512_scalef_ps(vp2, vn2);

    const __m512 vd0 = _mm512_add_ps(ve0, vone);
    const __m512 vd1 = _mm512_add_ps(ve1, vone);
    const __m512 vd2 = _mm512_add_ps(ve2, vone);

    __m512 vf0 = _mm512_div_ps(ve0, vd0);
    __m512 vf1 = _mm512_div_ps(ve1, vd1);
    __m512 vf2 = _mm512_div_ps(ve2, vd2);

    vf0 = _mm512_mask_sub_ps(vf0, _mm512_testn_epi32_mask(_mm512_castps_si512(vx0), vsign_mask), vone, vf0);
    vf1 = _mm512_mask_sub_ps(vf1, _mm512_testn_epi32_mask(_mm512_castps_si512(vx1), vsign_mask), vone, vf1);
    vf2 = _mm512_mask_sub_ps(vf2, _mm512_testn_epi32_mask(_mm512_castps_si512(vx2), vsign_mask), vone, vf2);

    _mm512_storeu_ps(y, vf0);
    _mm512_storeu_ps(y + 16, vf1);
    _mm512_storeu_ps(y + 32, vf2);
    y += 48;
  }
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(x);
    x += 16;

    const __m512 vz = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx), vsign_mask));

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512 vl = _mm512_permutexvar_ps(_mm512_castps_si512(vn), vtable);
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vt, vc3, vc2);
    vp = _mm512_mul_ps(vp, vt);
    vp = _mm512_fmadd_ps(vt, vp, vt);
    vp = _mm512_fmadd_ps(vl, vp, vl);

    const __m512 ve = _mm512_scalef_ps(vp, vn);
    const __m512 vd = _mm512_add_ps(ve, vone);

    __m512 vf = _mm512_div_ps(ve, vd);

    vf = _mm512_mask_sub_ps(vf, _mm512_testn_epi32_mask(_mm512_castps_si512(vx), vsign_mask), vone, vf);

    _mm512_storeu_ps(y, vf);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 15 * sizeof(float));

    // Prepare mask for valid 32-bit elements (depends on n).
    n >>= 2 /* log2(sizeof(float)) */;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << n) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, x);
    const __m512 vz = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx), vsign_mask));

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512 vl = _mm512_permutexvar_ps(_mm512_castps_si512(vn), vtable);
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vt, vc3, vc2);
    vp = _mm512_mul_ps(vp, vt);
    vp = _mm512_fmadd_ps(vt, vp, vt);
    vp = _mm512_fmadd_ps(vl, vp, vl);

    const __m512 ve = _mm512_scalef_ps(vp, vn);
    const __m512 vd = _mm512_add_ps(ve, vone);

    __m512 vf = _mm512_div_ps(ve, vd);

    vf = _mm512_mask_sub_ps(vf, _mm512_testn_epi32_mask(_mm512_castps_si512(vx), vsign_mask), vone, vf);

    _mm512_mask_storeu_ps(y, vmask, vf);
  }
}
