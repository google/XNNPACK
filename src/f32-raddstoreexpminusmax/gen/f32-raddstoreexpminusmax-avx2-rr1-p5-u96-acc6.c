// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/raddstoreexpminusmax.h"


void xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc6(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E430p-1f);
  const __m256 vc5 = _mm256_set1_ps(0x1.0F9F9Cp-7f);
  const __m256 vc4 = _mm256_set1_ps(0x1.573A1Ap-5f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555A80p-3f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFDC6p-2f);
  const __m256 vc1 = _mm256_set1_ps(0x1.FFFFF6p-1f);
  const __m256 vdenorm_cutoff = _mm256_set1_ps(-0x1.5D589Ep6f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const __m256 vi_max = _mm256_broadcast_ss(max);

  __m256 vacc0 = _mm256_setzero_ps();
  __m256 vacc1 = _mm256_setzero_ps();
  __m256 vacc2 = _mm256_setzero_ps();
  __m256 vacc3 = _mm256_setzero_ps();
  __m256 vacc4 = _mm256_setzero_ps();
  __m256 vacc5 = _mm256_setzero_ps();
  for (; batch >= 96 * sizeof(float); batch -= 96 * sizeof(float)) {
    const __m256 vi0 = _mm256_loadu_ps(input);
    const __m256 vi1 = _mm256_loadu_ps(input + 8);
    const __m256 vi2 = _mm256_loadu_ps(input + 16);
    const __m256 vi3 = _mm256_loadu_ps(input + 24);
    const __m256 vi4 = _mm256_loadu_ps(input + 32);
    const __m256 vi5 = _mm256_loadu_ps(input + 40);
    const __m256 vi6 = _mm256_loadu_ps(input + 48);
    const __m256 vi7 = _mm256_loadu_ps(input + 56);
    const __m256 vi8 = _mm256_loadu_ps(input + 64);
    const __m256 vi9 = _mm256_loadu_ps(input + 72);
    const __m256 vi10 = _mm256_loadu_ps(input + 80);
    const __m256 vi11 = _mm256_loadu_ps(input + 88);
    input += 96;

    const __m256 vx0 = _mm256_sub_ps(vi0, vi_max);
    const __m256 vx1 = _mm256_sub_ps(vi1, vi_max);
    const __m256 vx2 = _mm256_sub_ps(vi2, vi_max);
    const __m256 vx3 = _mm256_sub_ps(vi3, vi_max);
    const __m256 vx4 = _mm256_sub_ps(vi4, vi_max);
    const __m256 vx5 = _mm256_sub_ps(vi5, vi_max);
    const __m256 vx6 = _mm256_sub_ps(vi6, vi_max);
    const __m256 vx7 = _mm256_sub_ps(vi7, vi_max);
    const __m256 vx8 = _mm256_sub_ps(vi8, vi_max);
    const __m256 vx9 = _mm256_sub_ps(vi9, vi_max);
    const __m256 vx10 = _mm256_sub_ps(vi10, vi_max);
    const __m256 vx11 = _mm256_sub_ps(vi11, vi_max);

    __m256 vn0 = _mm256_fmadd_ps(vx0, vlog2e, vmagic_bias);
    __m256 vn1 = _mm256_fmadd_ps(vx1, vlog2e, vmagic_bias);
    __m256 vn2 = _mm256_fmadd_ps(vx2, vlog2e, vmagic_bias);
    __m256 vn3 = _mm256_fmadd_ps(vx3, vlog2e, vmagic_bias);
    __m256 vn4 = _mm256_fmadd_ps(vx4, vlog2e, vmagic_bias);
    __m256 vn5 = _mm256_fmadd_ps(vx5, vlog2e, vmagic_bias);
    __m256 vn6 = _mm256_fmadd_ps(vx6, vlog2e, vmagic_bias);
    __m256 vn7 = _mm256_fmadd_ps(vx7, vlog2e, vmagic_bias);
    __m256 vn8 = _mm256_fmadd_ps(vx8, vlog2e, vmagic_bias);
    __m256 vn9 = _mm256_fmadd_ps(vx9, vlog2e, vmagic_bias);
    __m256 vn10 = _mm256_fmadd_ps(vx10, vlog2e, vmagic_bias);
    __m256 vn11 = _mm256_fmadd_ps(vx11, vlog2e, vmagic_bias);

    const __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn0), 23));
    const __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn1), 23));
    const __m256 vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn2), 23));
    const __m256 vs3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn3), 23));
    const __m256 vs4 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn4), 23));
    const __m256 vs5 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn5), 23));
    const __m256 vs6 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn6), 23));
    const __m256 vs7 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn7), 23));
    const __m256 vs8 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn8), 23));
    const __m256 vs9 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn9), 23));
    const __m256 vs10 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn10), 23));
    const __m256 vs11 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn11), 23));

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);
    vn5 = _mm256_sub_ps(vn5, vmagic_bias);
    vn6 = _mm256_sub_ps(vn6, vmagic_bias);
    vn7 = _mm256_sub_ps(vn7, vmagic_bias);
    vn8 = _mm256_sub_ps(vn8, vmagic_bias);
    vn9 = _mm256_sub_ps(vn9, vmagic_bias);
    vn10 = _mm256_sub_ps(vn10, vmagic_bias);
    vn11 = _mm256_sub_ps(vn11, vmagic_bias);

    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vx0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vx1);
    __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2, vx2);
    __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2, vx3);
    __m256 vt4 = _mm256_fmadd_ps(vn4, vminus_ln2, vx4);
    __m256 vt5 = _mm256_fmadd_ps(vn5, vminus_ln2, vx5);
    __m256 vt6 = _mm256_fmadd_ps(vn6, vminus_ln2, vx6);
    __m256 vt7 = _mm256_fmadd_ps(vn7, vminus_ln2, vx7);
    __m256 vt8 = _mm256_fmadd_ps(vn8, vminus_ln2, vx8);
    __m256 vt9 = _mm256_fmadd_ps(vn9, vminus_ln2, vx9);
    __m256 vt10 = _mm256_fmadd_ps(vn10, vminus_ln2, vx10);
    __m256 vt11 = _mm256_fmadd_ps(vn11, vminus_ln2, vx11);

    __m256 vp0 = _mm256_fmadd_ps(vc5, vt0, vc4);
    __m256 vp1 = _mm256_fmadd_ps(vc5, vt1, vc4);
    __m256 vp2 = _mm256_fmadd_ps(vc5, vt2, vc4);
    __m256 vp3 = _mm256_fmadd_ps(vc5, vt3, vc4);
    __m256 vp4 = _mm256_fmadd_ps(vc5, vt4, vc4);
    __m256 vp5 = _mm256_fmadd_ps(vc5, vt5, vc4);
    __m256 vp6 = _mm256_fmadd_ps(vc5, vt6, vc4);
    __m256 vp7 = _mm256_fmadd_ps(vc5, vt7, vc4);
    __m256 vp8 = _mm256_fmadd_ps(vc5, vt8, vc4);
    __m256 vp9 = _mm256_fmadd_ps(vc5, vt9, vc4);
    __m256 vp10 = _mm256_fmadd_ps(vc5, vt10, vc4);
    __m256 vp11 = _mm256_fmadd_ps(vc5, vt11, vc4);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc3);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc3);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc3);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc3);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc3);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc3);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc3);
    vp10 = _mm256_fmadd_ps(vp10, vt10, vc3);
    vp11 = _mm256_fmadd_ps(vp11, vt11, vc3);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc2);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc2);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc2);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc2);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc2);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc2);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc2);
    vp10 = _mm256_fmadd_ps(vp10, vt10, vc2);
    vp11 = _mm256_fmadd_ps(vp11, vt11, vc2);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc1);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc1);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc1);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc1);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc1);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc1);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc1);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc1);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc1);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc1);
    vp10 = _mm256_fmadd_ps(vp10, vt10, vc1);
    vp11 = _mm256_fmadd_ps(vp11, vt11, vc1);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vt3 = _mm256_mul_ps(vt3, vs3);
    vt4 = _mm256_mul_ps(vt4, vs4);
    vt5 = _mm256_mul_ps(vt5, vs5);
    vt6 = _mm256_mul_ps(vt6, vs6);
    vt7 = _mm256_mul_ps(vt7, vs7);
    vt8 = _mm256_mul_ps(vt8, vs8);
    vt9 = _mm256_mul_ps(vt9, vs9);
    vt10 = _mm256_mul_ps(vt10, vs10);
    vt11 = _mm256_mul_ps(vt11, vs11);

    __m256 vf0 = _mm256_fmadd_ps(vt0, vp0, vs0);
    __m256 vf1 = _mm256_fmadd_ps(vt1, vp1, vs1);
    __m256 vf2 = _mm256_fmadd_ps(vt2, vp2, vs2);
    __m256 vf3 = _mm256_fmadd_ps(vt3, vp3, vs3);
    __m256 vf4 = _mm256_fmadd_ps(vt4, vp4, vs4);
    __m256 vf5 = _mm256_fmadd_ps(vt5, vp5, vs5);
    __m256 vf6 = _mm256_fmadd_ps(vt6, vp6, vs6);
    __m256 vf7 = _mm256_fmadd_ps(vt7, vp7, vs7);
    __m256 vf8 = _mm256_fmadd_ps(vt8, vp8, vs8);
    __m256 vf9 = _mm256_fmadd_ps(vt9, vp9, vs9);
    __m256 vf10 = _mm256_fmadd_ps(vt10, vp10, vs10);
    __m256 vf11 = _mm256_fmadd_ps(vt11, vp11, vs11);

    vf0 = _mm256_andnot_ps(_mm256_cmp_ps(vx0, vdenorm_cutoff, _CMP_LT_OS), vf0);
    vf1 = _mm256_andnot_ps(_mm256_cmp_ps(vx1, vdenorm_cutoff, _CMP_LT_OS), vf1);
    vf2 = _mm256_andnot_ps(_mm256_cmp_ps(vx2, vdenorm_cutoff, _CMP_LT_OS), vf2);
    vf3 = _mm256_andnot_ps(_mm256_cmp_ps(vx3, vdenorm_cutoff, _CMP_LT_OS), vf3);
    vf4 = _mm256_andnot_ps(_mm256_cmp_ps(vx4, vdenorm_cutoff, _CMP_LT_OS), vf4);
    vf5 = _mm256_andnot_ps(_mm256_cmp_ps(vx5, vdenorm_cutoff, _CMP_LT_OS), vf5);
    vf6 = _mm256_andnot_ps(_mm256_cmp_ps(vx6, vdenorm_cutoff, _CMP_LT_OS), vf6);
    vf7 = _mm256_andnot_ps(_mm256_cmp_ps(vx7, vdenorm_cutoff, _CMP_LT_OS), vf7);
    vf8 = _mm256_andnot_ps(_mm256_cmp_ps(vx8, vdenorm_cutoff, _CMP_LT_OS), vf8);
    vf9 = _mm256_andnot_ps(_mm256_cmp_ps(vx9, vdenorm_cutoff, _CMP_LT_OS), vf9);
    vf10 = _mm256_andnot_ps(_mm256_cmp_ps(vx10, vdenorm_cutoff, _CMP_LT_OS), vf10);
    vf11 = _mm256_andnot_ps(_mm256_cmp_ps(vx11, vdenorm_cutoff, _CMP_LT_OS), vf11);

    _mm256_storeu_ps(output, vf0);
    _mm256_storeu_ps(output + 8, vf1);
    _mm256_storeu_ps(output + 16, vf2);
    _mm256_storeu_ps(output + 24, vf3);
    _mm256_storeu_ps(output + 32, vf4);
    _mm256_storeu_ps(output + 40, vf5);
    _mm256_storeu_ps(output + 48, vf6);
    _mm256_storeu_ps(output + 56, vf7);
    _mm256_storeu_ps(output + 64, vf8);
    _mm256_storeu_ps(output + 72, vf9);
    _mm256_storeu_ps(output + 80, vf10);
    _mm256_storeu_ps(output + 88, vf11);
    output += 96;

    vacc0 = _mm256_add_ps(vacc0, vf0);
    vacc1 = _mm256_add_ps(vacc1, vf1);
    vacc2 = _mm256_add_ps(vacc2, vf2);
    vacc3 = _mm256_add_ps(vacc3, vf3);
    vacc4 = _mm256_add_ps(vacc4, vf4);
    vacc5 = _mm256_add_ps(vacc5, vf5);
    vacc0 = _mm256_add_ps(vacc0, vf6);
    vacc1 = _mm256_add_ps(vacc1, vf7);
    vacc2 = _mm256_add_ps(vacc2, vf8);
    vacc3 = _mm256_add_ps(vacc3, vf9);
    vacc4 = _mm256_add_ps(vacc4, vf10);
    vacc5 = _mm256_add_ps(vacc5, vf11);
  }
  vacc0 = _mm256_add_ps(vacc0, vacc1);
  vacc2 = _mm256_add_ps(vacc2, vacc3);
  vacc4 = _mm256_add_ps(vacc4, vacc5);
  vacc0 = _mm256_add_ps(vacc0, vacc2);
  vacc0 = _mm256_add_ps(vacc0, vacc4);

  __m256 vacc = vacc0;
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vi = _mm256_loadu_ps(input);
    input += 8;

    const __m256 vx = _mm256_sub_ps(vi, vi_max);

    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vx);

    __m256 vp = _mm256_fmadd_ps(vc5, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);

    vt = _mm256_mul_ps(vt, vs);
    __m256 vf = _mm256_fmadd_ps(vt, vp, vs);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vx, vdenorm_cutoff, _CMP_LT_OS), vf);

    _mm256_storeu_ps(output, vf);
    output += 8;

    vacc = _mm256_add_ps(vacc, vf);
  }
  if (batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    const __m256 vi = _mm256_maskload_ps(input, vmask);

    const __m256 vx = _mm256_sub_ps(vi, vi_max);

    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vx);

    __m256 vp = _mm256_fmadd_ps(vc5, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);

    vt = _mm256_mul_ps(vt, vs);
    __m256 vf = _mm256_fmadd_ps(vt, vp, vs);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vx, vdenorm_cutoff, _CMP_LT_OS), vf);

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

    vacc = _mm256_add_ps(vacc, _mm256_and_ps(vf, _mm256_castsi256_ps(vmask)));
  }
  __m128 vacc_lo = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
  vacc_lo = _mm_add_ps(vacc_lo, _mm_movehl_ps(vacc_lo, vacc_lo));
  vacc_lo = _mm_add_ss(vacc_lo, _mm_movehdup_ps(vacc_lo));
  _mm_store_ss(sum, vacc_lo);
  _mm256_zeroupper();
}
