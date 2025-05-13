// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/simd/f32-hvx.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"

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


void xnn_f32_vrsqrt_ukernel__hvx_u128(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  XNN_SIMD_CONST_F32(vthree, 3.0f);
  XNN_SIMD_CONST_F32(vneg_half, -0.5f);

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input + 0);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 32);
    const xnn_simd_f32_t vx2 = xnn_loadu_f32(input + 64);
    const xnn_simd_f32_t vx3 = xnn_loadu_f32(input + 96);
    input += 128;

    // Generate the initial approximation.
    const xnn_simd_f32_t vt0_0 = xnn_rsqrt_f32(vx0);
    const xnn_simd_f32_t vt0_1 = xnn_rsqrt_f32(vx1);
    const xnn_simd_f32_t vt0_2 = xnn_rsqrt_f32(vx2);
    const xnn_simd_f32_t vt0_3 = xnn_rsqrt_f32(vx3);

    // Do a single Newton-Raphson step as described above.
    const xnn_simd_f32_t vt1_0 = xnn_mul_f32(vt0_0, vt0_0);
    const xnn_simd_f32_t vt1_1 = xnn_mul_f32(vt0_1, vt0_1);
    const xnn_simd_f32_t vt1_2 = xnn_mul_f32(vt0_2, vt0_2);
    const xnn_simd_f32_t vt1_3 = xnn_mul_f32(vt0_3, vt0_3);
    const xnn_simd_f32_t vt3_0 = xnn_fmsub_f32(vx0, vt1_0, vthree);
    const xnn_simd_f32_t vt3_1 = xnn_fmsub_f32(vx1, vt1_1, vthree);
    const xnn_simd_f32_t vt3_2 = xnn_fmsub_f32(vx2, vt1_2, vthree);
    const xnn_simd_f32_t vt3_3 = xnn_fmsub_f32(vx3, vt1_3, vthree);
    const xnn_simd_f32_t vt4_0 = xnn_mul_f32(vneg_half, vt0_0);
    const xnn_simd_f32_t vt4_1 = xnn_mul_f32(vneg_half, vt0_1);
    const xnn_simd_f32_t vt4_2 = xnn_mul_f32(vneg_half, vt0_2);
    const xnn_simd_f32_t vt4_3 = xnn_mul_f32(vneg_half, vt0_3);
    const xnn_simd_f32_t vy0 = xnn_mul_f32(vt3_0, vt4_0);
    const xnn_simd_f32_t vy1 = xnn_mul_f32(vt3_1, vt4_1);
    const xnn_simd_f32_t vy2 = xnn_mul_f32(vt3_2, vt4_2);
    const xnn_simd_f32_t vy3 = xnn_mul_f32(vt3_3, vt4_3);

    // Store the results.
    xnn_storeu_f32(output + 0, vy0);
    xnn_storeu_f32(output + 32, vy1);
    xnn_storeu_f32(output + 64, vy2);
    xnn_storeu_f32(output + 96, vy3);
    output += 128;
  }
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += 32;

    // Generate the initial approximation.
    const xnn_simd_f32_t vt0 = xnn_rsqrt_f32(vx);

    // Do a single Newton-Raphson step as described above.
    const xnn_simd_f32_t vt1 = xnn_mul_f32(vt0, vt0);
    const xnn_simd_f32_t vt3 = xnn_fmsub_f32(vx, vt1, vthree);
    const xnn_simd_f32_t vt4 = xnn_mul_f32(vneg_half, vt0);
    const xnn_simd_f32_t vy = xnn_mul_f32(vt3, vt4);

    xnn_storeu_f32(output, vy);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Generate the initial 12-bit approximation.
    const xnn_simd_f32_t vt0 = xnn_rsqrt_f32(vx);

    // Do a single Newton-Raphson step as described above.
    const xnn_simd_f32_t vt1 = xnn_mul_f32(vt0, vt0);
    const xnn_simd_f32_t vt3 = xnn_fmsub_f32(vx, vt1, vthree);
    const xnn_simd_f32_t vt4 = xnn_mul_f32(vneg_half, vt0);
    xnn_simd_f32_t vy = xnn_mul_f32(vt3, vt4);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
