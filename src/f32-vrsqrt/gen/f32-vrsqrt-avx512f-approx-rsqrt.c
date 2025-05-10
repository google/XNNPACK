// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/simd-approx-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/simd/f32-avx512f.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"

// In the following, we use Newton-Raphson steps on the equation
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
// Where $x_k$ is the original approximation and `y` contains the improved
// approximation $x_{k+1}$.


void xnn_f32_vrsqrt_ukernel__avx512f_approx_rsqrt_u16(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 16);

  XNN_SIMD_CONST_F32(vthree, 3.0f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);

  // Register pressure isn't much of an issue in these kernels, but for some
  // reason this is necessary to avoid up to *~10x* slower code.
  // Stack alignment...?
  XNN_FORCE_REALIZATION(vthree);
  XNN_FORCE_REALIZATION(vhalf);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += 16;

    xnn_simd_f32_t vy = xnn_approx_reciprocal_sqrt_f32(vx);

    xnn_simd_f32_t vt1, vt2, vt3, vt4;
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    xnn_storeu_f32(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_approx_reciprocal_sqrt_f32(vx);

    xnn_simd_f32_t vt1, vt2, vt3, vt4;
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vrsqrt_ukernel__avx512f_approx_rsqrt_u32(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 16);

  XNN_SIMD_CONST_F32(vthree, 3.0f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);

  // Register pressure isn't much of an issue in these kernels, but for some
  // reason this is necessary to avoid up to *~10x* slower code.
  // Stack alignment...?
  XNN_FORCE_REALIZATION(vthree);
  XNN_FORCE_REALIZATION(vhalf);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 32;

    xnn_simd_f32_t vy_0 = xnn_approx_reciprocal_sqrt_f32(vx_0);
    xnn_simd_f32_t vy_1 = xnn_approx_reciprocal_sqrt_f32(vx_1);

    xnn_simd_f32_t vt1_0, vt2_0, vt3_0, vt4_0;
    xnn_simd_f32_t vt1_1, vt2_1, vt3_1, vt4_1;
    // Do a Newton-Raphson step as described above.
    vt1_0 = xnn_mul_f32(vy_0, vy_0);
    vt1_1 = xnn_mul_f32(vy_1, vy_1);
    vt2_0 = xnn_mul_f32(vx_0, vt1_0);
    vt2_1 = xnn_mul_f32(vx_1, vt1_1);
    vt3_0 = xnn_sub_f32(vthree, vt2_0);
    vt3_1 = xnn_sub_f32(vthree, vt2_1);
    vt4_0 = xnn_mul_f32(vhalf, vy_0);
    vt4_1 = xnn_mul_f32(vhalf, vy_1);
    vy_0 = xnn_mul_f32(vt3_0, vt4_0);
    vy_1 = xnn_mul_f32(vt3_1, vt4_1);
    // Do a Newton-Raphson step as described above.
    vt1_0 = xnn_mul_f32(vy_0, vy_0);
    vt1_1 = xnn_mul_f32(vy_1, vy_1);
    vt2_0 = xnn_mul_f32(vx_0, vt1_0);
    vt2_1 = xnn_mul_f32(vx_1, vt1_1);
    vt3_0 = xnn_sub_f32(vthree, vt2_0);
    vt3_1 = xnn_sub_f32(vthree, vt2_1);
    vt4_0 = xnn_mul_f32(vhalf, vy_0);
    vt4_1 = xnn_mul_f32(vhalf, vy_1);
    vy_0 = xnn_mul_f32(vt3_0, vt4_0);
    vy_1 = xnn_mul_f32(vt3_1, vt4_1);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 32;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += 16;

    xnn_simd_f32_t vy = xnn_approx_reciprocal_sqrt_f32(vx);

    xnn_simd_f32_t vt1, vt2, vt3, vt4;
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    xnn_storeu_f32(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_approx_reciprocal_sqrt_f32(vx);

    xnn_simd_f32_t vt1, vt2, vt3, vt4;
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vrsqrt_ukernel__avx512f_approx_rsqrt_u64(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 16);

  XNN_SIMD_CONST_F32(vthree, 3.0f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);

  // Register pressure isn't much of an issue in these kernels, but for some
  // reason this is necessary to avoid up to *~10x* slower code.
  // Stack alignment...?
  XNN_FORCE_REALIZATION(vthree);
  XNN_FORCE_REALIZATION(vhalf);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vx_3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 64;

    xnn_simd_f32_t vy_0 = xnn_approx_reciprocal_sqrt_f32(vx_0);
    xnn_simd_f32_t vy_1 = xnn_approx_reciprocal_sqrt_f32(vx_1);
    xnn_simd_f32_t vy_2 = xnn_approx_reciprocal_sqrt_f32(vx_2);
    xnn_simd_f32_t vy_3 = xnn_approx_reciprocal_sqrt_f32(vx_3);

    xnn_simd_f32_t vt1_0, vt2_0, vt3_0, vt4_0;
    xnn_simd_f32_t vt1_1, vt2_1, vt3_1, vt4_1;
    xnn_simd_f32_t vt1_2, vt2_2, vt3_2, vt4_2;
    xnn_simd_f32_t vt1_3, vt2_3, vt3_3, vt4_3;
    // Do a Newton-Raphson step as described above.
    vt1_0 = xnn_mul_f32(vy_0, vy_0);
    vt1_1 = xnn_mul_f32(vy_1, vy_1);
    vt1_2 = xnn_mul_f32(vy_2, vy_2);
    vt1_3 = xnn_mul_f32(vy_3, vy_3);
    vt2_0 = xnn_mul_f32(vx_0, vt1_0);
    vt2_1 = xnn_mul_f32(vx_1, vt1_1);
    vt2_2 = xnn_mul_f32(vx_2, vt1_2);
    vt2_3 = xnn_mul_f32(vx_3, vt1_3);
    vt3_0 = xnn_sub_f32(vthree, vt2_0);
    vt3_1 = xnn_sub_f32(vthree, vt2_1);
    vt3_2 = xnn_sub_f32(vthree, vt2_2);
    vt3_3 = xnn_sub_f32(vthree, vt2_3);
    vt4_0 = xnn_mul_f32(vhalf, vy_0);
    vt4_1 = xnn_mul_f32(vhalf, vy_1);
    vt4_2 = xnn_mul_f32(vhalf, vy_2);
    vt4_3 = xnn_mul_f32(vhalf, vy_3);
    vy_0 = xnn_mul_f32(vt3_0, vt4_0);
    vy_1 = xnn_mul_f32(vt3_1, vt4_1);
    vy_2 = xnn_mul_f32(vt3_2, vt4_2);
    vy_3 = xnn_mul_f32(vt3_3, vt4_3);
    // Do a Newton-Raphson step as described above.
    vt1_0 = xnn_mul_f32(vy_0, vy_0);
    vt1_1 = xnn_mul_f32(vy_1, vy_1);
    vt1_2 = xnn_mul_f32(vy_2, vy_2);
    vt1_3 = xnn_mul_f32(vy_3, vy_3);
    vt2_0 = xnn_mul_f32(vx_0, vt1_0);
    vt2_1 = xnn_mul_f32(vx_1, vt1_1);
    vt2_2 = xnn_mul_f32(vx_2, vt1_2);
    vt2_3 = xnn_mul_f32(vx_3, vt1_3);
    vt3_0 = xnn_sub_f32(vthree, vt2_0);
    vt3_1 = xnn_sub_f32(vthree, vt2_1);
    vt3_2 = xnn_sub_f32(vthree, vt2_2);
    vt3_3 = xnn_sub_f32(vthree, vt2_3);
    vt4_0 = xnn_mul_f32(vhalf, vy_0);
    vt4_1 = xnn_mul_f32(vhalf, vy_1);
    vt4_2 = xnn_mul_f32(vhalf, vy_2);
    vt4_3 = xnn_mul_f32(vhalf, vy_3);
    vy_0 = xnn_mul_f32(vt3_0, vt4_0);
    vy_1 = xnn_mul_f32(vt3_1, vt4_1);
    vy_2 = xnn_mul_f32(vt3_2, vt4_2);
    vy_3 = xnn_mul_f32(vt3_3, vt4_3);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 64;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += 16;

    xnn_simd_f32_t vy = xnn_approx_reciprocal_sqrt_f32(vx);

    xnn_simd_f32_t vt1, vt2, vt3, vt4;
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    xnn_storeu_f32(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_approx_reciprocal_sqrt_f32(vx);

    xnn_simd_f32_t vt1, vt2, vt3, vt4;
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    // Do a Newton-Raphson step as described above.
    vt1 = xnn_mul_f32(vy, vy);
    vt2 = xnn_mul_f32(vx, vt1);
    vt3 = xnn_sub_f32(vthree, vt2);
    vt4 = xnn_mul_f32(vhalf, vy);
    vy = xnn_mul_f32(vt3, vt4);
    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
