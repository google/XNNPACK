// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/avx2-expm1minus.c.in
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

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>


void xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vsign_mask = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3_perm.sign_mask);
  const __m256 vsat_cutoff = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3_perm.sat_cutoff);
  const __m256 vlog2e = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3_perm.log2e);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3_perm.magic_bias);
  const __m256i vtable = _mm256_load_si256((const __m256i*) params->avx_expm1minus_rr1_lut8_p4h3_perm.table);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3_perm.minus_ln2);
  const __m256 vc4 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3_perm.c4);
  const __m256 vc3 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3_perm.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3_perm.c2);
  const __m256 vtwo = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3_perm.two);
  const __m256 vminus_one = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3_perm.minus_one);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const __m256 vx0 = _mm256_loadu_ps(input);
    const __m256 vx1 = _mm256_loadu_ps(input + 8);
    const __m256 vx2 = _mm256_loadu_ps(input + 16);
    const __m256 vx3 = _mm256_loadu_ps(input + 24);
    input += 32;

    __m256 vz0 = _mm256_or_ps(vx0, vsign_mask);
    __m256 vz1 = _mm256_or_ps(vx1, vsign_mask);
    __m256 vz2 = _mm256_or_ps(vx2, vsign_mask);
    __m256 vz3 = _mm256_or_ps(vx3, vsign_mask);

    const __m256 vinvsignx0 = _mm256_xor_ps(vx0, vz0);
    vz0 = _mm256_max_ps(vsat_cutoff, vz0);
    const __m256 vinvsignx1 = _mm256_xor_ps(vx1, vz1);
    vz1 = _mm256_max_ps(vsat_cutoff, vz1);
    const __m256 vinvsignx2 = _mm256_xor_ps(vx2, vz2);
    vz2 = _mm256_max_ps(vsat_cutoff, vz2);
    const __m256 vinvsignx3 = _mm256_xor_ps(vx3, vz3);
    vz3 = _mm256_max_ps(vsat_cutoff, vz3);

    __m256 vn0 = _mm256_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m256 vn1 = _mm256_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m256 vn2 = _mm256_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m256 vn3 = _mm256_fmadd_ps(vz3, vlog2e, vmagic_bias);

    const __m256i ve0 = _mm256_slli_epi32(_mm256_castps_si256(vn0), 20);
    const __m256i ve1 = _mm256_slli_epi32(_mm256_castps_si256(vn1), 20);
    const __m256i ve2 = _mm256_slli_epi32(_mm256_castps_si256(vn2), 20);
    const __m256i ve3 = _mm256_slli_epi32(_mm256_castps_si256(vn3), 20);

    const __m256i vl0 = _mm256_permutevar8x32_epi32(vtable, _mm256_castps_si256(vn0));
    const __m256i vl1 = _mm256_permutevar8x32_epi32(vtable, _mm256_castps_si256(vn1));
    const __m256i vl2 = _mm256_permutevar8x32_epi32(vtable, _mm256_castps_si256(vn2));
    const __m256i vl3 = _mm256_permutevar8x32_epi32(vtable, _mm256_castps_si256(vn3));

    const __m256 vs0 = _mm256_castsi256_ps(_mm256_add_epi32(vl0, ve0));
    const __m256 vs1 = _mm256_castsi256_ps(_mm256_add_epi32(vl1, ve1));
    const __m256 vs2 = _mm256_castsi256_ps(_mm256_add_epi32(vl2, ve2));
    const __m256 vs3 = _mm256_castsi256_ps(_mm256_add_epi32(vl3, ve3));

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);

    const __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vz0);
    const __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vz1);
    const __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2, vz2);
    const __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2, vz3);

    __m256 vp0 = vc4;
    __m256 vp1 = vc4;
    __m256 vp2 = vc4;
    __m256 vp3 = vc4;
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc3);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc2);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vtwo);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vtwo);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vtwo);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vtwo);

    const __m256 vts0 = _mm256_mul_ps(vt0, vs0);
    const __m256 vsmo0 = _mm256_add_ps(vs0, vminus_one);
    const __m256 vts1 = _mm256_mul_ps(vt1, vs1);
    const __m256 vsmo1 = _mm256_add_ps(vs1, vminus_one);
    const __m256 vts2 = _mm256_mul_ps(vt2, vs2);
    const __m256 vsmo2 = _mm256_add_ps(vs2, vminus_one);
    const __m256 vts3 = _mm256_mul_ps(vt3, vs3);
    const __m256 vsmo3 = _mm256_add_ps(vs3, vminus_one);
    const __m256 vemo0 = _mm256_fmadd_ps(vp0, vts0, vsmo0);
    const __m256 vemo1 = _mm256_fmadd_ps(vp1, vts1, vsmo1);
    const __m256 vemo2 = _mm256_fmadd_ps(vp2, vts2, vsmo2);
    const __m256 vemo3 = _mm256_fmadd_ps(vp3, vts3, vsmo3);
    const __m256 vepo0 = _mm256_add_ps(vemo0, vtwo);
    const __m256 vepo1 = _mm256_add_ps(vemo1, vtwo);
    const __m256 vepo2 = _mm256_add_ps(vemo2, vtwo);
    const __m256 vepo3 = _mm256_add_ps(vemo3, vtwo);

    __m256 vrepo0 = _mm256_rcp_ps(vepo0);
    __m256 vrepo1 = _mm256_rcp_ps(vepo1);
    __m256 vrepo2 = _mm256_rcp_ps(vepo2);
    __m256 vrepo3 = _mm256_rcp_ps(vepo3);
    const __m256 verepo0 = _mm256_fnmsub_ps(vrepo0, vepo0, vminus_one);
    const __m256 verepo1 = _mm256_fnmsub_ps(vrepo1, vepo1, vminus_one);
    const __m256 verepo2 = _mm256_fnmsub_ps(vrepo2, vepo2, vminus_one);
    const __m256 verepo3 = _mm256_fnmsub_ps(vrepo3, vepo3, vminus_one);
    vrepo0 = _mm256_fmadd_ps(verepo0, vrepo0, vrepo0);
    vrepo1 = _mm256_fmadd_ps(verepo1, vrepo1, vrepo1);
    vrepo2 = _mm256_fmadd_ps(verepo2, vrepo2, vrepo2);
    vrepo3 = _mm256_fmadd_ps(verepo3, vrepo3, vrepo3);

    __m256 vy0 = _mm256_mul_ps(vemo0, vrepo0);
    __m256 vy1 = _mm256_mul_ps(vemo1, vrepo1);
    __m256 vy2 = _mm256_mul_ps(vemo2, vrepo2);
    __m256 vy3 = _mm256_mul_ps(vemo3, vrepo3);

    const __m256 vey0 = _mm256_fnmadd_ps(vy0, vepo0, vemo0);
    const __m256 vey1 = _mm256_fnmadd_ps(vy1, vepo1, vemo1);
    const __m256 vey2 = _mm256_fnmadd_ps(vy2, vepo2, vemo2);
    const __m256 vey3 = _mm256_fnmadd_ps(vy3, vepo3, vemo3);
    vy0 = _mm256_fmadd_ps(vey0, vrepo0, vy0);
    vy1 = _mm256_fmadd_ps(vey1, vrepo1, vy1);
    vy2 = _mm256_fmadd_ps(vey2, vrepo2, vy2);
    vy3 = _mm256_fmadd_ps(vey3, vrepo3, vy3);

    vy0 = _mm256_xor_ps(vy0, vinvsignx0);
    vy1 = _mm256_xor_ps(vy1, vinvsignx1);
    vy2 = _mm256_xor_ps(vy2, vinvsignx2);
    vy3 = _mm256_xor_ps(vy3, vinvsignx3);

    _mm256_storeu_ps(output, vy0);
    _mm256_storeu_ps(output + 8, vy1);
    _mm256_storeu_ps(output + 16, vy2);
    _mm256_storeu_ps(output + 24, vy3);
    output += 32;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    __m256 vz = _mm256_or_ps(vx, vsign_mask);

    const __m256 vinvsignx = _mm256_xor_ps(vx, vz);
    vz = _mm256_max_ps(vsat_cutoff, vz);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

    const __m256i ve = _mm256_slli_epi32(_mm256_castps_si256(vn), 20);

    const __m256i vl = _mm256_permutevar8x32_epi32(vtable, _mm256_castps_si256(vn));

    const __m256 vs = _mm256_castsi256_ps(_mm256_add_epi32(vl, ve));

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = vc4;
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_fmadd_ps(vp, vts, vsmo);
    const __m256 vepo = _mm256_add_ps(vemo, vtwo);

    __m256 vrepo = _mm256_rcp_ps(vepo);
    const __m256 verepo = _mm256_fnmsub_ps(vrepo, vepo, vminus_one);
    vrepo = _mm256_fmadd_ps(verepo, vrepo, vrepo);

    __m256 vy = _mm256_mul_ps(vemo, vrepo);

    const __m256 vey = _mm256_fnmadd_ps(vy, vepo, vemo);
    vy = _mm256_fmadd_ps(vey, vrepo, vy);

    vy = _mm256_xor_ps(vy, vinvsignx);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx_expm1minus_rr1_lut8_p4h3_perm.mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);

    __m256 vz = _mm256_or_ps(vx, vsign_mask);

    const __m256 vinvsignx = _mm256_xor_ps(vx, vz);
    vz = _mm256_max_ps(vsat_cutoff, vz);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

    const __m256i ve = _mm256_slli_epi32(_mm256_castps_si256(vn), 20);

    const __m256i vl = _mm256_permutevar8x32_epi32(vtable, _mm256_castps_si256(vn));

    const __m256 vs = _mm256_castsi256_ps(_mm256_add_epi32(vl, ve));

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = vc4;
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_fmadd_ps(vp, vts, vsmo);
    const __m256 vepo = _mm256_add_ps(vemo, vtwo);

    __m256 vrepo = _mm256_rcp_ps(vepo);
    const __m256 verepo = _mm256_fnmsub_ps(vrepo, vepo, vminus_one);
    vrepo = _mm256_fmadd_ps(verepo, vrepo, vrepo);

    __m256 vy = _mm256_mul_ps(vemo, vrepo);

    const __m256 vey = _mm256_fnmadd_ps(vy, vepo, vemo);
    vy = _mm256_fmadd_ps(vey, vrepo, vy);

    vy = _mm256_xor_ps(vy, vinvsignx);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}
