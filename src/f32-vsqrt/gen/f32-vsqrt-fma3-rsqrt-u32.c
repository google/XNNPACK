// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/fma3-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <xmmintrin.h>
#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


// In the following, we first compute the _reciprocal_ square root of an input
// `a` and then multiply it with `a` to produce the square root.
//
// We compute the reciprocal square root using a single Newton-Raphson step on
// the equation $x^{-2} - a$, which expands to:
//
//  $$x_{k+1} = 0.5 * x_k * (3.0 - a * x_k^2)$$
//
// So we do the following steps:
//
//  1. t0 = x_k
//  2. t1 = t0 * t0       (x_k^2)
//  3. t3 = a * t1 - 3.0  (a * x_k^2 - 3.0)
//  4. t4 = 0.5 * t0      (-0.5 * x_k)
//  5. t5  = t3 * t4      ((-0.5 * x_k) * (a * x_k^2 - 3.0))
//  6. y = a * t5         (a * a^{-1/2})
//
// Where $x_k$ is the original 14-bit approximation and `t5` contains the final
// 24-bit approximation $x_{k+1}$.

void xnn_f32_vsqrt_ukernel__fma3_rsqrt_u32(
    size_t batch, const float* input, float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
    XNN_OOB_READS {
  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const __m256 vthree = _mm256_set1_ps(3.0f);
  const __m256 vneg_half = _mm256_set1_ps(-0.5f);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const __m256 vx0 = _mm256_loadu_ps(input);
    const __m256 vx1 = _mm256_loadu_ps(input + 8);
    const __m256 vx2 = _mm256_loadu_ps(input + 16);
    const __m256 vx3 = _mm256_loadu_ps(input + 24);
    input += 32;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m256 vinf_mask_0 = _mm256_cmp_ps(vx0, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 vinf_mask_1 = _mm256_cmp_ps(vx1, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 vinf_mask_2 = _mm256_cmp_ps(vx2, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 vinf_mask_3 = _mm256_cmp_ps(vx3, _mm256_setzero_ps(), _CMP_EQ_OQ);

    // Generate the initial 12-bit approximation.
    const __m256 vt0_0 = _mm256_rsqrt_ps(vx0);
    const __m256 vt0_1 = _mm256_rsqrt_ps(vx1);
    const __m256 vt0_2 = _mm256_rsqrt_ps(vx2);
    const __m256 vt0_3 = _mm256_rsqrt_ps(vx3);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1_0 = _mm256_mul_ps(vt0_0, vt0_0);
    const __m256 vt1_1 = _mm256_mul_ps(vt0_1, vt0_1);
    const __m256 vt1_2 = _mm256_mul_ps(vt0_2, vt0_2);
    const __m256 vt1_3 = _mm256_mul_ps(vt0_3, vt0_3);
    const __m256 vt3_0 = _mm256_fmsub_ps(vx0, vt1_0, vthree);
    const __m256 vt3_1 = _mm256_fmsub_ps(vx1, vt1_1, vthree);
    const __m256 vt3_2 = _mm256_fmsub_ps(vx2, vt1_2, vthree);
    const __m256 vt3_3 = _mm256_fmsub_ps(vx3, vt1_3, vthree);
    const __m256 vt4_0 = _mm256_mul_ps(vneg_half, vt0_0);
    const __m256 vt4_1 = _mm256_mul_ps(vneg_half, vt0_1);
    const __m256 vt4_2 = _mm256_mul_ps(vneg_half, vt0_2);
    const __m256 vt4_3 = _mm256_mul_ps(vneg_half, vt0_3);
    const __m256 vt5_0 = _mm256_mul_ps(vt3_0, vt4_0);
    const __m256 vt5_1 = _mm256_mul_ps(vt3_1, vt4_1);
    const __m256 vt5_2 = _mm256_mul_ps(vt3_2, vt4_2);
    const __m256 vt5_3 = _mm256_mul_ps(vt3_3, vt4_3);
    const __m256 vt6_0 = _mm256_andnot_ps(vinf_mask_0, vt5_0);
    const __m256 vt6_1 = _mm256_andnot_ps(vinf_mask_1, vt5_1);
    const __m256 vt6_2 = _mm256_andnot_ps(vinf_mask_2, vt5_2);
    const __m256 vt6_3 = _mm256_andnot_ps(vinf_mask_3, vt5_3);
    const __m256 vy0 = _mm256_mul_ps(vx0, vt6_0);
    const __m256 vy1 = _mm256_mul_ps(vx1, vt6_1);
    const __m256 vy2 = _mm256_mul_ps(vx2, vt6_2);
    const __m256 vy3 = _mm256_mul_ps(vx3, vt6_3);

    // Store the results.
    _mm256_storeu_ps(output, vy0);
    _mm256_storeu_ps(output + 8, vy1);
    _mm256_storeu_ps(output + 16, vy2);
    _mm256_storeu_ps(output + 24, vy3);
    output += 32;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m256 vinf_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);

    // Generate the initial 12-bit approximation.
    const __m256 vt0 = _mm256_rsqrt_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1 = _mm256_mul_ps(vt0, vt0);
    const __m256 vt3 = _mm256_fmsub_ps(vx, vt1, vthree);
    const __m256 vt4 = _mm256_mul_ps(vneg_half, vt0);
    const __m256 vt5 = _mm256_mul_ps(vt3, vt4);
    const __m256 vt6 = _mm256_andnot_ps(vinf_mask, vt5);
    const __m256 vy = _mm256_mul_ps(vx, vt6);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256(
        (const __m256i*)((uintptr_t)&mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m256 vinf_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);

    // Generate the initial 12-bit approximation.
    const __m256 vt0 = _mm256_rsqrt_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1 = _mm256_mul_ps(vt0, vt0);
    const __m256 vt3 = _mm256_fmsub_ps(vx, vt1, vthree);
    const __m256 vt4 = _mm256_mul_ps(vneg_half, vt0);
    const __m256 vt5 = _mm256_mul_ps(vt3, vt4);
    const __m256 vt6 = _mm256_andnot_ps(vinf_mask, vt5);
    __m256 vy = _mm256_mul_ps(vx, vt6);

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
