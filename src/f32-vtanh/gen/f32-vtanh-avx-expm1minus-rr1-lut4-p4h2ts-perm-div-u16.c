// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/avx-expm1minus.c.in
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

void xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vsign_mask = _mm256_load_ps(params->avx_expm1minus_rr1_lut4_p4h2_perm.sign_mask);
  const __m256 vsat_cutoff = _mm256_load_ps(params->avx_expm1minus_rr1_lut4_p4h2_perm.sat_cutoff);
  const __m256 vlog2e = _mm256_load_ps(params->avx_expm1minus_rr1_lut4_p4h2_perm.log2e);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx_expm1minus_rr1_lut4_p4h2_perm.magic_bias);
  const __m128 vtable = _mm_load_ps(params->avx_expm1minus_rr1_lut4_p4h2_perm.table);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx_expm1minus_rr1_lut4_p4h2_perm.minus_ln2);
  const __m256 vc4 = _mm256_load_ps(params->avx_expm1minus_rr1_lut4_p4h2_perm.c4);
  const __m256 vc3 = _mm256_load_ps(params->avx_expm1minus_rr1_lut4_p4h2_perm.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx_expm1minus_rr1_lut4_p4h2_perm.c2);
  const __m256 vtwo = _mm256_load_ps(params->avx_expm1minus_rr1_lut4_p4h2_perm.two);
  const __m256 vminus_one = _mm256_load_ps(params->avx_expm1minus_rr1_lut4_p4h2_perm.minus_one);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m256 vx0 = _mm256_loadu_ps(input);
    const __m256 vx1 = _mm256_loadu_ps(input + 8);
    input += 16;

    __m256 vz0 = _mm256_or_ps(vx0, vsign_mask);
    __m256 vz1 = _mm256_or_ps(vx1, vsign_mask);

    const __m256 vinvsignx0 = _mm256_xor_ps(vx0, vz0);
    vz0 = _mm256_max_ps(vsat_cutoff, vz0);
    const __m256 vinvsignx1 = _mm256_xor_ps(vx1, vz1);
    vz1 = _mm256_max_ps(vsat_cutoff, vz1);

    __m256 vn0 = _mm256_add_ps(_mm256_mul_ps(vz0, vlog2e), vmagic_bias);
    __m256 vn1 = _mm256_add_ps(_mm256_mul_ps(vz1, vlog2e), vmagic_bias);

    const __m128 vn0_hi = _mm256_extractf128_ps(vn0, 1);
    __m128i ve0_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn0)), 21);
    const __m128 vn1_hi = _mm256_extractf128_ps(vn1, 1);
    __m128i ve1_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn1)), 21);

    __m128i ve0_hi = _mm_slli_epi32(_mm_castps_si128(vn0_hi), 21);
    const __m128i vl0_lo = _mm_castps_si128(_mm_permutevar_ps(vtable, _mm_castps_si128(_mm256_castps256_ps128(vn0))));
    const __m128i vl0_hi = _mm_castps_si128(_mm_permutevar_ps(vtable, _mm_castps_si128(vn0_hi)));
    __m128i ve1_hi = _mm_slli_epi32(_mm_castps_si128(vn1_hi), 21);
    const __m128i vl1_lo = _mm_castps_si128(_mm_permutevar_ps(vtable, _mm_castps_si128(_mm256_castps256_ps128(vn1))));
    const __m128i vl1_hi = _mm_castps_si128(_mm_permutevar_ps(vtable, _mm_castps_si128(vn1_hi)));

    const __m128 vs0_lo = _mm_castsi128_ps(_mm_add_epi32(ve0_lo, vl0_lo));
    const __m128 vs0_hi = _mm_castsi128_ps(_mm_add_epi32(ve0_hi, vl0_hi));
    const __m128 vs1_lo = _mm_castsi128_ps(_mm_add_epi32(ve1_lo, vl1_lo));
    const __m128 vs1_hi = _mm_castsi128_ps(_mm_add_epi32(ve1_hi, vl1_hi));

    const __m256 vs0 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs0_lo), vs0_hi, 1);
    const __m256 vs1 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs1_lo), vs1_hi, 1);

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);

    const __m256 vt0 = _mm256_add_ps(_mm256_mul_ps(vn0, vminus_ln2), vz0);
    const __m256 vt1 = _mm256_add_ps(_mm256_mul_ps(vn1, vminus_ln2), vz1);

    __m256 vp0 = _mm256_add_ps(_mm256_mul_ps(vc4, vt0), vc3);
    __m256 vp1 = _mm256_add_ps(_mm256_mul_ps(vc4, vt1), vc3);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc2);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc2);
    vp0 = _mm256_mul_ps(vp0, vt0);
    vp1 = _mm256_mul_ps(vp1, vt1);

    const __m256 vts0 = _mm256_mul_ps(vt0, vs0);
    const __m256 vsmo0 = _mm256_add_ps(vs0, vminus_one);
    const __m256 vts1 = _mm256_mul_ps(vt1, vs1);
    const __m256 vsmo1 = _mm256_add_ps(vs1, vminus_one);

    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vts0), vts0);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vts1), vts1);
    const __m256 vemo0 = _mm256_add_ps(_mm256_mul_ps(vp0, vtwo), vsmo0);
    const __m256 vemo1 = _mm256_add_ps(_mm256_mul_ps(vp1, vtwo), vsmo1);

    const __m256 vepo0 = _mm256_add_ps(vemo0, vtwo);
    const __m256 vepo1 = _mm256_add_ps(vemo1, vtwo);
    __m256 vy0 = _mm256_div_ps(vemo0, vepo0);
    __m256 vy1 = _mm256_div_ps(vemo1, vepo1);

    vy0 = _mm256_xor_ps(vy0, vinvsignx0);
    vy1 = _mm256_xor_ps(vy1, vinvsignx1);

    _mm256_storeu_ps(output, vy0);
    _mm256_storeu_ps(output + 8, vy1);
    output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    __m256 vz = _mm256_or_ps(vx, vsign_mask);

    const __m256 vinvsignx = _mm256_xor_ps(vx, vz);
    vz = _mm256_max_ps(vsat_cutoff, vz);

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128 vn_hi = _mm256_extractf128_ps(vn, 1);
    __m128i ve_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 21);
    __m128i ve_hi = _mm_slli_epi32(_mm_castps_si128(vn_hi), 21);

    const __m128i vl_lo = _mm_castps_si128(_mm_permutevar_ps(vtable, _mm_castps_si128(_mm256_castps256_ps128(vn))));
    const __m128i vl_hi = _mm_castps_si128(_mm_permutevar_ps(vtable, _mm_castps_si128(vn_hi)));

    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(ve_lo, vl_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(ve_hi, vl_hi));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2), vz);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc4, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_mul_ps(vp, vt);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vts), vts);
    const __m256 vemo = _mm256_add_ps(_mm256_mul_ps(vp, vtwo), vsmo);

    const __m256 vepo = _mm256_add_ps(vemo, vtwo);
    __m256 vy = _mm256_div_ps(vemo, vepo);

    vy = _mm256_xor_ps(vy, vinvsignx);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx_expm1minus_rr1_lut4_p4h2_perm.mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);

    __m256 vz = _mm256_or_ps(vx, vsign_mask);

    const __m256 vinvsignx = _mm256_xor_ps(vx, vz);
    vz = _mm256_max_ps(vsat_cutoff, vz);

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128 vn_hi = _mm256_extractf128_ps(vn, 1);
    __m128i ve_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 21);
    __m128i ve_hi = _mm_slli_epi32(_mm_castps_si128(vn_hi), 21);

    const __m128i vl_lo = _mm_castps_si128(_mm_permutevar_ps(vtable, _mm_castps_si128(_mm256_castps256_ps128(vn))));
    const __m128i vl_hi = _mm_castps_si128(_mm_permutevar_ps(vtable, _mm_castps_si128(vn_hi)));

    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(ve_lo, vl_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(ve_hi, vl_hi));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2), vz);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc4, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_mul_ps(vp, vt);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vts), vts);
    const __m256 vemo = _mm256_add_ps(_mm256_mul_ps(vp, vtwo), vsmo);

    const __m256 vepo = _mm256_add_ps(vemo, vtwo);
    __m256 vy = _mm256_div_ps(vemo, vepo);

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
