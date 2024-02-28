// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/sse-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


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
//  2. t1 = t0 * t0   (x_k^2)
//  3. t2 = a * t1    (a * x_k^2)
//  4. t3 = 3.0 - t2  (3.0 - a * x_k^2)
//  5. t4 = 0.5 * t0  (0.5 * x_k)
//  6. t5 = t3 * t4   (0.5 * x_k * (3.0 - a * x_k^2))
//  7. y = a * t5     (a * a^{-1/2})
//
// Where $x_k$ is the original 14-bit approximation and `t5` contains the final
// 24-bit approximation $x_{k+1}$.

void xnn_f32_vsqrt_ukernel__sse_rsqrt_u32(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const __m128 kThree = _mm_load_ps(params->sse.three);
  const __m128 kHalf = _mm_load_ps(params->sse.half);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input);
    const __m128 va1 = _mm_loadu_ps(input + 4);
    const __m128 va2 = _mm_loadu_ps(input + 8);
    const __m128 va3 = _mm_loadu_ps(input + 12);
    const __m128 va4 = _mm_loadu_ps(input + 16);
    const __m128 va5 = _mm_loadu_ps(input + 20);
    const __m128 va6 = _mm_loadu_ps(input + 24);
    const __m128 va7 = _mm_loadu_ps(input + 28);
    input += 32;

    // Generate the initial 12-bit approximation.
    const __m128 vt0_0 = _mm_rsqrt_ps(va0);
    const __m128 vt0_1 = _mm_rsqrt_ps(va1);
    const __m128 vt0_2 = _mm_rsqrt_ps(va2);
    const __m128 vt0_3 = _mm_rsqrt_ps(va3);
    const __m128 vt0_4 = _mm_rsqrt_ps(va4);
    const __m128 vt0_5 = _mm_rsqrt_ps(va5);
    const __m128 vt0_6 = _mm_rsqrt_ps(va6);
    const __m128 vt0_7 = _mm_rsqrt_ps(va7);

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1_0 = _mm_mul_ps(vt0_0, vt0_0);
    const __m128 vt1_1 = _mm_mul_ps(vt0_1, vt0_1);
    const __m128 vt1_2 = _mm_mul_ps(vt0_2, vt0_2);
    const __m128 vt1_3 = _mm_mul_ps(vt0_3, vt0_3);
    const __m128 vt1_4 = _mm_mul_ps(vt0_4, vt0_4);
    const __m128 vt1_5 = _mm_mul_ps(vt0_5, vt0_5);
    const __m128 vt1_6 = _mm_mul_ps(vt0_6, vt0_6);
    const __m128 vt1_7 = _mm_mul_ps(vt0_7, vt0_7);
    const __m128 vt2_0 = _mm_mul_ps(va0, vt1_0);
    const __m128 vt2_1 = _mm_mul_ps(va1, vt1_1);
    const __m128 vt2_2 = _mm_mul_ps(va2, vt1_2);
    const __m128 vt2_3 = _mm_mul_ps(va3, vt1_3);
    const __m128 vt2_4 = _mm_mul_ps(va4, vt1_4);
    const __m128 vt2_5 = _mm_mul_ps(va5, vt1_5);
    const __m128 vt2_6 = _mm_mul_ps(va6, vt1_6);
    const __m128 vt2_7 = _mm_mul_ps(va7, vt1_7);
    const __m128 vt3_0 = _mm_sub_ps(kThree, vt2_0);
    const __m128 vt3_1 = _mm_sub_ps(kThree, vt2_1);
    const __m128 vt3_2 = _mm_sub_ps(kThree, vt2_2);
    const __m128 vt3_3 = _mm_sub_ps(kThree, vt2_3);
    const __m128 vt3_4 = _mm_sub_ps(kThree, vt2_4);
    const __m128 vt3_5 = _mm_sub_ps(kThree, vt2_5);
    const __m128 vt3_6 = _mm_sub_ps(kThree, vt2_6);
    const __m128 vt3_7 = _mm_sub_ps(kThree, vt2_7);
    const __m128 vt4_0 = _mm_mul_ps(kHalf, vt0_0);
    const __m128 vt4_1 = _mm_mul_ps(kHalf, vt0_1);
    const __m128 vt4_2 = _mm_mul_ps(kHalf, vt0_2);
    const __m128 vt4_3 = _mm_mul_ps(kHalf, vt0_3);
    const __m128 vt4_4 = _mm_mul_ps(kHalf, vt0_4);
    const __m128 vt4_5 = _mm_mul_ps(kHalf, vt0_5);
    const __m128 vt4_6 = _mm_mul_ps(kHalf, vt0_6);
    const __m128 vt4_7 = _mm_mul_ps(kHalf, vt0_7);
    const __m128 vt5_0 = _mm_mul_ps(vt3_0, vt4_0);
    const __m128 vt5_1 = _mm_mul_ps(vt3_1, vt4_1);
    const __m128 vt5_2 = _mm_mul_ps(vt3_2, vt4_2);
    const __m128 vt5_3 = _mm_mul_ps(vt3_3, vt4_3);
    const __m128 vt5_4 = _mm_mul_ps(vt3_4, vt4_4);
    const __m128 vt5_5 = _mm_mul_ps(vt3_5, vt4_5);
    const __m128 vt5_6 = _mm_mul_ps(vt3_6, vt4_6);
    const __m128 vt5_7 = _mm_mul_ps(vt3_7, vt4_7);
    const __m128 vt6_0 = _mm_cmpeq_ps(vt5_0, vt5_0);
    const __m128 vt6_1 = _mm_cmpeq_ps(vt5_1, vt5_1);
    const __m128 vt6_2 = _mm_cmpeq_ps(vt5_2, vt5_2);
    const __m128 vt6_3 = _mm_cmpeq_ps(vt5_3, vt5_3);
    const __m128 vt6_4 = _mm_cmpeq_ps(vt5_4, vt5_4);
    const __m128 vt6_5 = _mm_cmpeq_ps(vt5_5, vt5_5);
    const __m128 vt6_6 = _mm_cmpeq_ps(vt5_6, vt5_6);
    const __m128 vt6_7 = _mm_cmpeq_ps(vt5_7, vt5_7);
    const __m128 vt7_0 = _mm_and_ps(vt5_0, vt6_0);
    const __m128 vt7_1 = _mm_and_ps(vt5_1, vt6_1);
    const __m128 vt7_2 = _mm_and_ps(vt5_2, vt6_2);
    const __m128 vt7_3 = _mm_and_ps(vt5_3, vt6_3);
    const __m128 vt7_4 = _mm_and_ps(vt5_4, vt6_4);
    const __m128 vt7_5 = _mm_and_ps(vt5_5, vt6_5);
    const __m128 vt7_6 = _mm_and_ps(vt5_6, vt6_6);
    const __m128 vt7_7 = _mm_and_ps(vt5_7, vt6_7);
    const __m128 vy0 = _mm_mul_ps(va0, vt7_0);
    const __m128 vy1 = _mm_mul_ps(va1, vt7_1);
    const __m128 vy2 = _mm_mul_ps(va2, vt7_2);
    const __m128 vy3 = _mm_mul_ps(va3, vt7_3);
    const __m128 vy4 = _mm_mul_ps(va4, vt7_4);
    const __m128 vy5 = _mm_mul_ps(va5, vt7_5);
    const __m128 vy6 = _mm_mul_ps(va6, vt7_6);
    const __m128 vy7 = _mm_mul_ps(va7, vt7_7);

    // Store the results.
    _mm_storeu_ps(output, vy0);
    _mm_storeu_ps(output + 4, vy1);
    _mm_storeu_ps(output + 8, vy2);
    _mm_storeu_ps(output + 12, vy3);
    _mm_storeu_ps(output + 16, vy4);
    _mm_storeu_ps(output + 20, vy5);
    _mm_storeu_ps(output + 24, vy6);
    _mm_storeu_ps(output + 28, vy7);
    output += 32;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input);
    input += 4;
    // Generate the initial 12-bit approximation.
    const __m128 vt0 = _mm_rsqrt_ps(va);

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1 = _mm_mul_ps(vt0, vt0);
    const __m128 vt2 = _mm_mul_ps(va, vt1);
    const __m128 vt3 = _mm_sub_ps(kThree, vt2);
    const __m128 vt4 = _mm_mul_ps(kHalf, vt0);
    const __m128 vt5 = _mm_mul_ps(vt3, vt4);
    const __m128 vt6 = _mm_cmpeq_ps(vt5, vt5);
    const __m128 vt7 = _mm_and_ps(vt5, vt6);
    const __m128 vy = _mm_mul_ps(va, vt7);
    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input);
    // Generate the initial 12-bit approximation.
    const __m128 vt0 = _mm_rsqrt_ps(va);

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1 = _mm_mul_ps(vt0, vt0);
    const __m128 vt2 = _mm_mul_ps(va, vt1);
    const __m128 vt3 = _mm_sub_ps(kThree, vt2);
    const __m128 vt4 = _mm_mul_ps(kHalf, vt0);
    const __m128 vt5 = _mm_mul_ps(vt3, vt4);
    const __m128 vt6 = _mm_cmpeq_ps(vt5, vt5);
    const __m128 vt7 = _mm_and_ps(vt5, vt6);
    __m128 vy = _mm_mul_ps(va, vt7);
    
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}
