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

#include "xnnpack/common.h"
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
//  2. t1 = t0 * t0   (x_k^2)
//  3. t2 = a * t1    (a * x_k^2)
//  4. t3 = 3.0 - t2  (3.0 - a * x_k^2)
//  5. t4 = 0.5 * t0  (0.5 * x_k)
//  6. t5 = t3 * t4   (0.5 * x_k * (3.0 - a * x_k^2))
//  7. y = a * t5     (a * a^{-1/2})
//
// Where $x_k$ is the original 14-bit approximation and `t5` contains the final
// 24-bit approximation $x_{k+1}$.


void xnn_f32_vsqrt_ukernel__sse_rsqrt_u8(
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
  const __m128 vthree = _mm_set1_ps(3.0f);
  const __m128 vhalf = _mm_set1_ps(0.5f);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0 = _mm_loadu_ps(input);
    const __m128 vx1 = _mm_loadu_ps(input + 4);
    input += 8;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m128 vmask0 = _mm_cmpeq_ps(vx0, _mm_setzero_ps());
    const __m128 vmask1 = _mm_cmpeq_ps(vx1, _mm_setzero_ps());

    // Generate the initial 12-bit approximation.
    const __m128 vt0_0 = _mm_rsqrt_ps(vx0);
    const __m128 vt0_1 = _mm_rsqrt_ps(vx1);

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1_0 = _mm_mul_ps(vt0_0, vt0_0);
    const __m128 vt1_1 = _mm_mul_ps(vt0_1, vt0_1);
    const __m128 vt2_0 = _mm_mul_ps(vx0, vt1_0);
    const __m128 vt2_1 = _mm_mul_ps(vx1, vt1_1);
    const __m128 vt3_0 = _mm_sub_ps(vthree, vt2_0);
    const __m128 vt3_1 = _mm_sub_ps(vthree, vt2_1);
    const __m128 vt4_0 = _mm_mul_ps(vhalf, vt0_0);
    const __m128 vt4_1 = _mm_mul_ps(vhalf, vt0_1);
    const __m128 vt5_0 = _mm_mul_ps(vt3_0, vt4_0);
    const __m128 vt5_1 = _mm_mul_ps(vt3_1, vt4_1);
    const __m128 vt6_0 = _mm_andnot_ps(vmask0, vt5_0);
    const __m128 vt6_1 = _mm_andnot_ps(vmask1, vt5_1);
    const __m128 vy0 = _mm_mul_ps(vx0, vt6_0);
    const __m128 vy1 = _mm_mul_ps(vx1, vt6_1);

    // Store the results.
    _mm_storeu_ps(output, vy0);
    _mm_storeu_ps(output + 4, vy1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    // Generate the initial 12-bit approximation.
    const __m128 vt0 = _mm_rsqrt_ps(vx);

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m128 vmask = _mm_cmpeq_ps(vx, _mm_setzero_ps());

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1 = _mm_mul_ps(vt0, vt0);
    const __m128 vt2 = _mm_mul_ps(vx, vt1);
    const __m128 vt3 = _mm_sub_ps(vthree, vt2);
    const __m128 vt4 = _mm_mul_ps(vhalf, vt0);
    const __m128 vt5 = _mm_mul_ps(vt3, vt4);
    const __m128 vt6 = _mm_andnot_ps(vmask, vt5);
    const __m128 vy = _mm_mul_ps(vx, vt6);
    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);
    // Generate the initial 12-bit approximation.
    const __m128 vt0 = _mm_rsqrt_ps(vx);

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m128 vmask = _mm_cmpeq_ps(vx, _mm_setzero_ps());

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1 = _mm_mul_ps(vt0, vt0);
    const __m128 vt2 = _mm_mul_ps(vx, vt1);
    const __m128 vt3 = _mm_sub_ps(vthree, vt2);
    const __m128 vt4 = _mm_mul_ps(vhalf, vt0);
    const __m128 vt5 = _mm_mul_ps(vt3, vt4);
    const __m128 vt6 = _mm_andnot_ps(vmask, vt5);
    __m128 vy = _mm_mul_ps(vx, vt6);

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
