// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/avx-rr2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u56(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p0f);
  const __m256 vminus_ln2_hi = _mm256_set1_ps(-0x1.62E400p-1f);
  const __m256 vminus_ln2_lo = _mm256_set1_ps(-0x1.7F7D1Cp-20f);
  const __m256 vc5 = _mm256_set1_ps(0x1.0F9F9Cp-7f);
  const __m256 vc4 = _mm256_set1_ps(0x1.573A1Ap-5f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555A80p-3f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFDC6p-2f);
  const __m256 vc1 = _mm256_set1_ps(0x1.FFFFF6p-1f);
  const __m256 vone = _mm256_set1_ps(1.0f);
  const __m256 vdenorm_cutoff = _mm256_set1_ps(-0x1.5D589Ep+6f);

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


  for (; batch >= 56 * sizeof(float); batch -= 56 * sizeof(float)) {
    const __m256 vx0 = _mm256_loadu_ps(input);
    const __m256 vx1 = _mm256_loadu_ps(input + 8);
    const __m256 vx2 = _mm256_loadu_ps(input + 16);
    const __m256 vx3 = _mm256_loadu_ps(input + 24);
    const __m256 vx4 = _mm256_loadu_ps(input + 32);
    const __m256 vx5 = _mm256_loadu_ps(input + 40);
    const __m256 vx6 = _mm256_loadu_ps(input + 48);
    input += 56;

    const __m256 vz0 = _mm256_or_ps(vx0, vsign_mask);
    const __m256 vz1 = _mm256_or_ps(vx1, vsign_mask);
    const __m256 vz2 = _mm256_or_ps(vx2, vsign_mask);
    const __m256 vz3 = _mm256_or_ps(vx3, vsign_mask);
    const __m256 vz4 = _mm256_or_ps(vx4, vsign_mask);
    const __m256 vz5 = _mm256_or_ps(vx5, vsign_mask);
    const __m256 vz6 = _mm256_or_ps(vx6, vsign_mask);

    __m256 vn0 = _mm256_add_ps(_mm256_mul_ps(vz0, vlog2e), vmagic_bias);
    __m256 vn1 = _mm256_add_ps(_mm256_mul_ps(vz1, vlog2e), vmagic_bias);
    __m256 vn2 = _mm256_add_ps(_mm256_mul_ps(vz2, vlog2e), vmagic_bias);
    __m256 vn3 = _mm256_add_ps(_mm256_mul_ps(vz3, vlog2e), vmagic_bias);
    __m256 vn4 = _mm256_add_ps(_mm256_mul_ps(vz4, vlog2e), vmagic_bias);
    __m256 vn5 = _mm256_add_ps(_mm256_mul_ps(vz5, vlog2e), vmagic_bias);
    __m256 vn6 = _mm256_add_ps(_mm256_mul_ps(vz6, vlog2e), vmagic_bias);

    const __m128 vs_lo0 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn0)), 23));
    const __m128 vs_hi0 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn0, 1)), 23));
    const __m256 vs0 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo0), vs_hi0, 1);
    const __m128 vs_lo1 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn1)), 23));
    const __m128 vs_hi1 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn1, 1)), 23));
    const __m256 vs1 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo1), vs_hi1, 1);
    const __m128 vs_lo2 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn2)), 23));
    const __m128 vs_hi2 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn2, 1)), 23));
    const __m256 vs2 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo2), vs_hi2, 1);
    const __m128 vs_lo3 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn3)), 23));
    const __m128 vs_hi3 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn3, 1)), 23));
    const __m256 vs3 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo3), vs_hi3, 1);
    const __m128 vs_lo4 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn4)), 23));
    const __m128 vs_hi4 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn4, 1)), 23));
    const __m256 vs4 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo4), vs_hi4, 1);
    const __m128 vs_lo5 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn5)), 23));
    const __m128 vs_hi5 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn5, 1)), 23));
    const __m256 vs5 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo5), vs_hi5, 1);
    const __m128 vs_lo6 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn6)), 23));
    const __m128 vs_hi6 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn6, 1)), 23));
    const __m256 vs6 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo6), vs_hi6, 1);

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);
    vn5 = _mm256_sub_ps(vn5, vmagic_bias);
    vn6 = _mm256_sub_ps(vn6, vmagic_bias);

    __m256 vt0 = _mm256_add_ps(_mm256_mul_ps(vn0, vminus_ln2_hi), vz0);
    __m256 vt1 = _mm256_add_ps(_mm256_mul_ps(vn1, vminus_ln2_hi), vz1);
    __m256 vt2 = _mm256_add_ps(_mm256_mul_ps(vn2, vminus_ln2_hi), vz2);
    __m256 vt3 = _mm256_add_ps(_mm256_mul_ps(vn3, vminus_ln2_hi), vz3);
    __m256 vt4 = _mm256_add_ps(_mm256_mul_ps(vn4, vminus_ln2_hi), vz4);
    __m256 vt5 = _mm256_add_ps(_mm256_mul_ps(vn5, vminus_ln2_hi), vz5);
    __m256 vt6 = _mm256_add_ps(_mm256_mul_ps(vn6, vminus_ln2_hi), vz6);

    vt0 = _mm256_add_ps(_mm256_mul_ps(vn0, vminus_ln2_lo), vt0);
    vt1 = _mm256_add_ps(_mm256_mul_ps(vn1, vminus_ln2_lo), vt1);
    vt2 = _mm256_add_ps(_mm256_mul_ps(vn2, vminus_ln2_lo), vt2);
    vt3 = _mm256_add_ps(_mm256_mul_ps(vn3, vminus_ln2_lo), vt3);
    vt4 = _mm256_add_ps(_mm256_mul_ps(vn4, vminus_ln2_lo), vt4);
    vt5 = _mm256_add_ps(_mm256_mul_ps(vn5, vminus_ln2_lo), vt5);
    vt6 = _mm256_add_ps(_mm256_mul_ps(vn6, vminus_ln2_lo), vt6);

    __m256 vp0 = _mm256_add_ps(_mm256_mul_ps(vc5, vt0), vc4);
    __m256 vp1 = _mm256_add_ps(_mm256_mul_ps(vc5, vt1), vc4);
    __m256 vp2 = _mm256_add_ps(_mm256_mul_ps(vc5, vt2), vc4);
    __m256 vp3 = _mm256_add_ps(_mm256_mul_ps(vc5, vt3), vc4);
    __m256 vp4 = _mm256_add_ps(_mm256_mul_ps(vc5, vt4), vc4);
    __m256 vp5 = _mm256_add_ps(_mm256_mul_ps(vc5, vt5), vc4);
    __m256 vp6 = _mm256_add_ps(_mm256_mul_ps(vc5, vt6), vc4);

    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc3);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc3);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc3);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc3);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc3);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vc3);
    vp6 = _mm256_add_ps(_mm256_mul_ps(vp6, vt6), vc3);

    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc2);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc2);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc2);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc2);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc2);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vc2);
    vp6 = _mm256_add_ps(_mm256_mul_ps(vp6, vt6), vc2);

    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vc1);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vc1);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vc1);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vc1);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vc1);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vc1);
    vp6 = _mm256_add_ps(_mm256_mul_ps(vp6, vt6), vc1);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vt3 = _mm256_mul_ps(vt3, vs3);
    vt4 = _mm256_mul_ps(vt4, vs4);
    vt5 = _mm256_mul_ps(vt5, vs5);
    vt6 = _mm256_mul_ps(vt6, vs6);

    const __m256 ve0 = _mm256_add_ps(_mm256_mul_ps(vt0, vp0), vs0);
    const __m256 ve1 = _mm256_add_ps(_mm256_mul_ps(vt1, vp1), vs1);
    const __m256 ve2 = _mm256_add_ps(_mm256_mul_ps(vt2, vp2), vs2);
    const __m256 ve3 = _mm256_add_ps(_mm256_mul_ps(vt3, vp3), vs3);
    const __m256 ve4 = _mm256_add_ps(_mm256_mul_ps(vt4, vp4), vs4);
    const __m256 ve5 = _mm256_add_ps(_mm256_mul_ps(vt5, vp5), vs5);
    const __m256 ve6 = _mm256_add_ps(_mm256_mul_ps(vt6, vp6), vs6);

    const __m256 vd0 = _mm256_add_ps(ve0, vone);
    const __m256 vd1 = _mm256_add_ps(ve1, vone);
    const __m256 vd2 = _mm256_add_ps(ve2, vone);
    const __m256 vd3 = _mm256_add_ps(ve3, vone);
    const __m256 vd4 = _mm256_add_ps(ve4, vone);
    const __m256 vd5 = _mm256_add_ps(ve5, vone);
    const __m256 vd6 = _mm256_add_ps(ve6, vone);

    __m256 vf0 = _mm256_div_ps(ve0, vd0);
    __m256 vf1 = _mm256_div_ps(ve1, vd1);
    __m256 vf2 = _mm256_div_ps(ve2, vd2);
    __m256 vf3 = _mm256_div_ps(ve3, vd3);
    __m256 vf4 = _mm256_div_ps(ve4, vd4);
    __m256 vf5 = _mm256_div_ps(ve5, vd5);
    __m256 vf6 = _mm256_div_ps(ve6, vd6);

    vf0 = _mm256_andnot_ps(_mm256_cmp_ps(vz0, vdenorm_cutoff, _CMP_LT_OS), vf0);
    vf1 = _mm256_andnot_ps(_mm256_cmp_ps(vz1, vdenorm_cutoff, _CMP_LT_OS), vf1);
    vf2 = _mm256_andnot_ps(_mm256_cmp_ps(vz2, vdenorm_cutoff, _CMP_LT_OS), vf2);
    vf3 = _mm256_andnot_ps(_mm256_cmp_ps(vz3, vdenorm_cutoff, _CMP_LT_OS), vf3);
    vf4 = _mm256_andnot_ps(_mm256_cmp_ps(vz4, vdenorm_cutoff, _CMP_LT_OS), vf4);
    vf5 = _mm256_andnot_ps(_mm256_cmp_ps(vz5, vdenorm_cutoff, _CMP_LT_OS), vf5);
    vf6 = _mm256_andnot_ps(_mm256_cmp_ps(vz6, vdenorm_cutoff, _CMP_LT_OS), vf6);

    vf0 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf0), vf0, vx0);
    vf1 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf1), vf1, vx1);
    vf2 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf2), vf2, vx2);
    vf3 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf3), vf3, vx3);
    vf4 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf4), vf4, vx4);
    vf5 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf5), vf5, vx5);
    vf6 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf6), vf6, vx6);

    _mm256_storeu_ps(output, vf0);
    _mm256_storeu_ps(output + 8, vf1);
    _mm256_storeu_ps(output + 16, vf2);
    _mm256_storeu_ps(output + 24, vf3);
    _mm256_storeu_ps(output + 32, vf4);
    _mm256_storeu_ps(output + 40, vf5);
    _mm256_storeu_ps(output + 48, vf6);
    output += 56;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128 vs_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 23));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 23));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc5, vt), vc4);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc1);

    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_add_ps(_mm256_mul_ps(vt, vp), vs);

    const __m256 vd = _mm256_add_ps(ve, vone);
    __m256 vf = _mm256_div_ps(ve, vd);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    _mm256_storeu_ps(output, vf);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);

    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);
    const __m128 vs_lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 23));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 23));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc5, vt), vc4);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc1);

    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_add_ps(_mm256_mul_ps(vt, vp), vs);

    const __m256 vd = _mm256_add_ps(ve, vone);
    __m256 vf = _mm256_div_ps(ve, vd);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    __m128 vf_lo = _mm256_castps256_ps128(vf);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vf_lo);
      vf_lo = _mm256_extractf128_ps(vf, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vf_lo);
      vf_lo = _mm_movehl_ps(vf_lo, vf_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vf_lo);
    }
  }
}
