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

void xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u8(
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

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    // Load 8 (1x8) inputs at a time.
    const __m256 vi0 = _mm256_loadu_ps(input);
    input += 8;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m256 vx0 = _mm256_sub_ps(vi0, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m256 vn0 = _mm256_fmadd_ps(vx0, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn0), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn0 = _mm256_sub_ps(vn0, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2_hi, vx0);

    vt0 = _mm256_fmadd_ps(vn0, vminus_ln2_lo, vt0);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m256 vp0 = _mm256_fmadd_ps(vc5, vt0, vc4);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc3);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc2);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0 = _mm256_mul_ps(vt0, vs0);

    __m256 vf0 = _mm256_fmadd_ps(vt0, vp0, vs0);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0 = _mm256_andnot_ps(_mm256_cmp_ps(vx0, vdenorm_cutoff, _CMP_LT_OS), vf0);

    // Multiply by scale.
    vf0 = _mm256_mul_ps(vf0, vscale);

    // Store 8 (1x8) outputs at a time.
    _mm256_storeu_ps(output, vf0);
    output += 8;
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
