// Auto-generated file. Do not edit!
//   Template: src/f32-vscaleexpminusmax/avx2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vscaleexpminusmax.h"


static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

void xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u96(
    size_t batch,
    const float* input,
    float* output,
    float scale,
    float max)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  // The smallest x for which expf(x) is normalized.
  const __m256 vdenorm_cutoff = _mm256_set1_ps(-0x1.5D589Ep6f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 vminus_ln2_hi = _mm256_set1_ps(-0x1.62E43p-1f);
  const __m256 vminus_ln2_lo = _mm256_set1_ps(0x1.05C61p-29f);

  const __m256 vc1 = _mm256_set1_ps(0x1.FFFFF6p-1f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFDC6p-2f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555A80p-3f);
  const __m256 vc4 = _mm256_set1_ps(0x1.573A1Ap-5f);
  const __m256 vc5 = _mm256_set1_ps(0x1.0F9F9Cp-7f);

  const __m256 vscale = _mm256_set1_ps(scale);
  const __m256 vi_max = _mm256_set1_ps(max);

  for (; batch >= 96 * sizeof(float); batch -= 96 * sizeof(float)) {
    // Load 96 (12x8) inputs at a time.
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

    // Subtract maximum input x := i - i_max. This implies x <= 0.
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

    // Compute reduced argument batch := round(x / log(2)).
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

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
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

    // Subtract the large number back to get final batch := round(x / log(2)).
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

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2_hi, vx0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2_hi, vx1);
    __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2_hi, vx2);
    __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2_hi, vx3);
    __m256 vt4 = _mm256_fmadd_ps(vn4, vminus_ln2_hi, vx4);
    __m256 vt5 = _mm256_fmadd_ps(vn5, vminus_ln2_hi, vx5);
    __m256 vt6 = _mm256_fmadd_ps(vn6, vminus_ln2_hi, vx6);
    __m256 vt7 = _mm256_fmadd_ps(vn7, vminus_ln2_hi, vx7);
    __m256 vt8 = _mm256_fmadd_ps(vn8, vminus_ln2_hi, vx8);
    __m256 vt9 = _mm256_fmadd_ps(vn9, vminus_ln2_hi, vx9);
    __m256 vt10 = _mm256_fmadd_ps(vn10, vminus_ln2_hi, vx10);
    __m256 vt11 = _mm256_fmadd_ps(vn11, vminus_ln2_hi, vx11);

    vt0 = _mm256_fmadd_ps(vn0, vminus_ln2_lo, vt0);
    vt1 = _mm256_fmadd_ps(vn1, vminus_ln2_lo, vt1);
    vt2 = _mm256_fmadd_ps(vn2, vminus_ln2_lo, vt2);
    vt3 = _mm256_fmadd_ps(vn3, vminus_ln2_lo, vt3);
    vt4 = _mm256_fmadd_ps(vn4, vminus_ln2_lo, vt4);
    vt5 = _mm256_fmadd_ps(vn5, vminus_ln2_lo, vt5);
    vt6 = _mm256_fmadd_ps(vn6, vminus_ln2_lo, vt6);
    vt7 = _mm256_fmadd_ps(vn7, vminus_ln2_lo, vt7);
    vt8 = _mm256_fmadd_ps(vn8, vminus_ln2_lo, vt8);
    vt9 = _mm256_fmadd_ps(vn9, vminus_ln2_lo, vt9);
    vt10 = _mm256_fmadd_ps(vn10, vminus_ln2_lo, vt10);
    vt11 = _mm256_fmadd_ps(vn11, vminus_ln2_lo, vt11);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
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

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
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

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
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

    // Multiply by scale.
    vf0 = _mm256_mul_ps(vf0, vscale);
    vf1 = _mm256_mul_ps(vf1, vscale);
    vf2 = _mm256_mul_ps(vf2, vscale);
    vf3 = _mm256_mul_ps(vf3, vscale);
    vf4 = _mm256_mul_ps(vf4, vscale);
    vf5 = _mm256_mul_ps(vf5, vscale);
    vf6 = _mm256_mul_ps(vf6, vscale);
    vf7 = _mm256_mul_ps(vf7, vscale);
    vf8 = _mm256_mul_ps(vf8, vscale);
    vf9 = _mm256_mul_ps(vf9, vscale);
    vf10 = _mm256_mul_ps(vf10, vscale);
    vf11 = _mm256_mul_ps(vf11, vscale);

    // Store 96 (12x8) outputs at a time.
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
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    // Load 8 inputs at a time.
    const __m256 vi = _mm256_loadu_ps(input);
    input += 8;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m256 vx = _mm256_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm256_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m256 vp = _mm256_fmadd_ps(vc5, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm256_mul_ps(vt, vs);
    __m256 vf = _mm256_fmadd_ps(vt, vp, vs);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vx, vdenorm_cutoff, _CMP_LT_OS), vf);

    // Multiply by scale.
    vf = _mm256_mul_ps(vf, vscale);

    // Store 64 (8x8) outputs at a time.
    _mm256_storeu_ps(output, vf);
    output += 8;
  }
  if (batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    // Load up to 7 inputs at a time.
    const __m256 vi = _mm256_maskload_ps(input, vmask);

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m256 vx = _mm256_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm256_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m256 vp = _mm256_fmadd_ps(vc5, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm256_mul_ps(vt, vs);
    __m256 vf = _mm256_fmadd_ps(vt, vp, vs);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vx, vdenorm_cutoff, _CMP_LT_OS), vf);

    // Multiply by scale.
    vf = _mm256_mul_ps(vf, vscale);

    // Store up to 7 outputs at a time.
    _mm256_maskstore_ps(output, vmask, vf);
  }
}
