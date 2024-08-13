// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/sse-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"

// In the following, we do a single Newton-Raphson step on the equation
// $x^{-2} - a$, which expands to:
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
//  6. y  = t3 * t4   (0.5 * x_k * (3.0 - a * x_k^2))
//
// Where $x_k$ is the original 12-bit approximation and `y` contains the final
// 24-bit approximation $x_{k+1}$.


void xnn_f32_vrsqrt_ukernel__sse_rsqrt_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const __m128 vthree = _mm_set1_ps(3.0f);
  const __m128 vhalf = _mm_set1_ps(0.5f);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    const __m128 vx89AB = _mm_loadu_ps(input + 8);
    const __m128 vxCDEF = _mm_loadu_ps(input + 12);
    input += 16;

    // Generate the initial 12-bit approximation.
    const __m128 vt0_0123 = _mm_rsqrt_ps(vx0123);
    const __m128 vt0_4567 = _mm_rsqrt_ps(vx4567);
    const __m128 vt0_89AB = _mm_rsqrt_ps(vx89AB);
    const __m128 vt0_CDEF = _mm_rsqrt_ps(vxCDEF);

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1_0123 = _mm_mul_ps(vt0_0123, vt0_0123);
    const __m128 vt1_4567 = _mm_mul_ps(vt0_4567, vt0_4567);
    const __m128 vt1_89AB = _mm_mul_ps(vt0_89AB, vt0_89AB);
    const __m128 vt1_CDEF = _mm_mul_ps(vt0_CDEF, vt0_CDEF);
    const __m128 vt2_0123 = _mm_mul_ps(vx0123, vt1_0123);
    const __m128 vt2_4567 = _mm_mul_ps(vx4567, vt1_4567);
    const __m128 vt2_89AB = _mm_mul_ps(vx89AB, vt1_89AB);
    const __m128 vt2_CDEF = _mm_mul_ps(vxCDEF, vt1_CDEF);
    const __m128 vt3_0123 = _mm_sub_ps(vthree, vt2_0123);
    const __m128 vt3_4567 = _mm_sub_ps(vthree, vt2_4567);
    const __m128 vt3_89AB = _mm_sub_ps(vthree, vt2_89AB);
    const __m128 vt3_CDEF = _mm_sub_ps(vthree, vt2_CDEF);
    const __m128 vt4_0123 = _mm_mul_ps(vhalf, vt0_0123);
    const __m128 vt4_4567 = _mm_mul_ps(vhalf, vt0_4567);
    const __m128 vt4_89AB = _mm_mul_ps(vhalf, vt0_89AB);
    const __m128 vt4_CDEF = _mm_mul_ps(vhalf, vt0_CDEF);
    const __m128 vy0123 = _mm_mul_ps(vt3_0123, vt4_0123);
    const __m128 vy4567 = _mm_mul_ps(vt3_4567, vt4_4567);
    const __m128 vy89AB = _mm_mul_ps(vt3_89AB, vt4_89AB);
    const __m128 vyCDEF = _mm_mul_ps(vt3_CDEF, vt4_CDEF);

    // Store the results.
    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    _mm_storeu_ps(output + 8, vy89AB);
    _mm_storeu_ps(output + 12, vyCDEF);
    output += 16;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    // Generate the initial 12-bit approximation.
    const __m128 vt0 = _mm_rsqrt_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1 = _mm_mul_ps(vt0, vt0);
    const __m128 vt2 = _mm_mul_ps(vx, vt1);
    const __m128 vt3 = _mm_sub_ps(vthree, vt2);
    const __m128 vt4 = _mm_mul_ps(vhalf, vt0);
    const __m128 vy = _mm_mul_ps(vt3, vt4);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);

    // Generate the initial 12-bit approximation.
    const __m128 vt0 = _mm_rsqrt_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1 = _mm_mul_ps(vt0, vt0);
    const __m128 vt2 = _mm_mul_ps(vx, vt1);
    const __m128 vt3 = _mm_sub_ps(vthree, vt2);
    const __m128 vt4 = _mm_mul_ps(vhalf, vt0);
    __m128 vy = _mm_mul_ps(vt3, vt4);

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
