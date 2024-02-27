// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/fma3-rsqrt.c.in
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
#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>


// In the following, we do a single Newton-Raphson step on the equation
// $x^{-2} - a$, which expands to:
//
//  $$x_{k+1} = 0.5 * x_k * (3.0 - a * x_k^2)$$
//
// So we do the following steps:
//
//  1. t0 = x_k
//  2. t1 = t0 * t0       (x_k^2)
//  3. t3 = a * t1 - 3.0  (a * x_k^2 - 3.0)
//  4. t4 = 0.5 * t0      (-0.5 * x_k)
//  5. y  = t3 * t4       ((-0.5 * x_k) * (a * x_k^2 - 3.0))
//
// Where $x_k$ is the original 12-bit approximation and `y` contains the final
// 24-bit approximation $x_{k+1}$.

void xnn_f32_vrsqrt_ukernel__fma3_rsqrt_u64(
    size_t batch, const float* input, float* output,
    const union xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
    XNN_OOB_READS {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const __m256 kThree = _mm256_load_ps(params->fma3.three);
  const __m256 kNegHalf = _mm256_load_ps(params->fma3.neg_half);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m256 va0 = _mm256_loadu_ps(input);
    const __m256 va1 = _mm256_loadu_ps(input + 8);
    const __m256 va2 = _mm256_loadu_ps(input + 16);
    const __m256 va3 = _mm256_loadu_ps(input + 24);
    const __m256 va4 = _mm256_loadu_ps(input + 32);
    const __m256 va5 = _mm256_loadu_ps(input + 40);
    const __m256 va6 = _mm256_loadu_ps(input + 48);
    const __m256 va7 = _mm256_loadu_ps(input + 56);
    input += 64;

    // Generate the initial 12-bit approximation.
    const __m256 vt0_0 = _mm256_rsqrt_ps(va0);
    const __m256 vt0_1 = _mm256_rsqrt_ps(va1);
    const __m256 vt0_2 = _mm256_rsqrt_ps(va2);
    const __m256 vt0_3 = _mm256_rsqrt_ps(va3);
    const __m256 vt0_4 = _mm256_rsqrt_ps(va4);
    const __m256 vt0_5 = _mm256_rsqrt_ps(va5);
    const __m256 vt0_6 = _mm256_rsqrt_ps(va6);
    const __m256 vt0_7 = _mm256_rsqrt_ps(va7);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1_0 = _mm256_mul_ps(vt0_0, vt0_0);
    const __m256 vt1_1 = _mm256_mul_ps(vt0_1, vt0_1);
    const __m256 vt1_2 = _mm256_mul_ps(vt0_2, vt0_2);
    const __m256 vt1_3 = _mm256_mul_ps(vt0_3, vt0_3);
    const __m256 vt1_4 = _mm256_mul_ps(vt0_4, vt0_4);
    const __m256 vt1_5 = _mm256_mul_ps(vt0_5, vt0_5);
    const __m256 vt1_6 = _mm256_mul_ps(vt0_6, vt0_6);
    const __m256 vt1_7 = _mm256_mul_ps(vt0_7, vt0_7);
    const __m256 vt3_0 = _mm256_fmsub_ps(va0, vt1_0, kThree);
    const __m256 vt3_1 = _mm256_fmsub_ps(va1, vt1_1, kThree);
    const __m256 vt3_2 = _mm256_fmsub_ps(va2, vt1_2, kThree);
    const __m256 vt3_3 = _mm256_fmsub_ps(va3, vt1_3, kThree);
    const __m256 vt3_4 = _mm256_fmsub_ps(va4, vt1_4, kThree);
    const __m256 vt3_5 = _mm256_fmsub_ps(va5, vt1_5, kThree);
    const __m256 vt3_6 = _mm256_fmsub_ps(va6, vt1_6, kThree);
    const __m256 vt3_7 = _mm256_fmsub_ps(va7, vt1_7, kThree);
    const __m256 vt4_0 = _mm256_mul_ps(kNegHalf, vt0_0);
    const __m256 vt4_1 = _mm256_mul_ps(kNegHalf, vt0_1);
    const __m256 vt4_2 = _mm256_mul_ps(kNegHalf, vt0_2);
    const __m256 vt4_3 = _mm256_mul_ps(kNegHalf, vt0_3);
    const __m256 vt4_4 = _mm256_mul_ps(kNegHalf, vt0_4);
    const __m256 vt4_5 = _mm256_mul_ps(kNegHalf, vt0_5);
    const __m256 vt4_6 = _mm256_mul_ps(kNegHalf, vt0_6);
    const __m256 vt4_7 = _mm256_mul_ps(kNegHalf, vt0_7);
    const __m256 vy0 = _mm256_mul_ps(vt3_0, vt4_0);
    const __m256 vy1 = _mm256_mul_ps(vt3_1, vt4_1);
    const __m256 vy2 = _mm256_mul_ps(vt3_2, vt4_2);
    const __m256 vy3 = _mm256_mul_ps(vt3_3, vt4_3);
    const __m256 vy4 = _mm256_mul_ps(vt3_4, vt4_4);
    const __m256 vy5 = _mm256_mul_ps(vt3_5, vt4_5);
    const __m256 vy6 = _mm256_mul_ps(vt3_6, vt4_6);
    const __m256 vy7 = _mm256_mul_ps(vt3_7, vt4_7);

    // Store the results.
    _mm256_storeu_ps(output, vy0);
    _mm256_storeu_ps(output + 8, vy1);
    _mm256_storeu_ps(output + 16, vy2);
    _mm256_storeu_ps(output + 24, vy3);
    _mm256_storeu_ps(output + 32, vy4);
    _mm256_storeu_ps(output + 40, vy5);
    _mm256_storeu_ps(output + 48, vy6);
    _mm256_storeu_ps(output + 56, vy7);
    output += 64;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(input);
    input += 8;

    // Generate the initial 12-bit approximation.
    const __m256 vt0 = _mm256_rsqrt_ps(va);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1 = _mm256_mul_ps(vt0, vt0);
    const __m256 vt3 = _mm256_fmsub_ps(va, vt1, kThree);
    const __m256 vt4 = _mm256_mul_ps(kNegHalf, vt0);
    const __m256 vy = _mm256_mul_ps(vt3, vt4);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256(
        (const __m256i*)((uintptr_t)&params->fma3.mask_table[7] - batch));

    const __m256 va = _mm256_maskload_ps(input, vmask);

    // Generate the initial 12-bit approximation.
    const __m256 vt0 = _mm256_rsqrt_ps(va);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1 = _mm256_mul_ps(vt0, vt0);
    const __m256 vt3 = _mm256_fmsub_ps(va, vt1, kThree);
    const __m256 vt4 = _mm256_mul_ps(kNegHalf, vt0);
    __m256 vy = _mm256_mul_ps(vt3, vt4);

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
