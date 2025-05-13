// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/simd-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
    
// Arch-specific SIMD wrapper.
#include "src/xnnpack/simd/f32-sse2.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


// In the following, we first compute the _reciprocal_ square root of an input
// `a` and then multiply it with `a` to produce the square root.
//
// We compute the reciprocal square root using a single Newton-Raphson step on
// the equation $x^{-2} - a$, which expands to:
//
//  $$x_{k+1} = x_k - 0.5 * x_k * (a * x_k^2 - 1.0)$$
//
// Note that we don't further simplify this expression, e.g. by factoring out
// `x_k`, so that the iteration _updates_ `x_k` in a numerically consistent way.
//
// So we do the following steps:
//
//  1. t0 = x_k
//  2. t1 = t0 * t0   (x_k^2)
//  3. t2 = a * t1    (a * x_k^2)
//  4. t3 = t2 - 1.0  (a * x_k^2 - 1.0)
//  5. t4 = 0.5 * t0  (0.5 * x_k)
//  6. t5 = t3 * t4   (0.5 * x_k * (a * x_k^2 - 1.0))
//  7. t6 = t0 - t5   (x_k - (0.5 * x_k * (a * x_k^2 - 1.0)))
//  7. y = a * t6     (a * a^{-1/2})
//
// Where $x_k$ is the original 12-bit (or 14-bit on AVX512f) approximation and
// `t6` contains the final 24-bit approximation $x_{k+1}$.
//
// In the implementation below, steps 3+4 and 6+7 could be merged into a single
// fused multiply-add/sub operation, but we chose not to as they are not
// performance critical and thus numerical consistency across microarchitectures
// is prefered.


void xnn_f32_vsqrt_ukernel__sse2_rsqrt_u4(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)]) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const xnn_simd_f32_t kOne = xnn_set1_f32(1.0f);
  const xnn_simd_f32_t kHalf = xnn_set1_f32(0.5f);


  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const xnn_simd_f32_t vinf_mask = xnn_cmpneq_f32(vx, xnn_zero_f32());

    // Generate the initial 12-bit approximation.
    xnn_simd_f32_t vt0 = xnn_rsqrt_f32(vx);

    // Do a fixed number of Newton-Raphson steps as described above.
    for (size_t i = 0; i < XNN_SIMD_NUM_RSQRT_ITER_F32; i++) {
      const xnn_simd_f32_t vt1 = xnn_mul_f32(vt0, vt0);
      const xnn_simd_f32_t vt2 = xnn_mul_f32(vx, vt1);
      const xnn_simd_f32_t vt3 = xnn_sub_f32(vt2, kOne);
      const xnn_simd_f32_t vt4 = xnn_mul_f32(kHalf, vt0);
      const xnn_simd_f32_t vt5 = xnn_mul_f32(vt3, vt4);
      vt0 = xnn_sub_f32(vt0, vt5);
    }

    // Mask out the inputs that were zero.
    const xnn_simd_f32_t vt7 = xnn_and_f32(vinf_mask, vt0);

    // Extract the square root by multiplying the original value with the
    // inverse square root.
    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vt7);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const xnn_simd_f32_t vinf_mask = xnn_cmpneq_f32(vx, xnn_zero_f32());

    // Generate the initial 12-bit approximation.
    xnn_simd_f32_t vt0 = xnn_rsqrt_f32(vx);

    // Do a fixed number of Newton-Raphson steps as described above.
    for (size_t i = 0; i < XNN_SIMD_NUM_RSQRT_ITER_F32; i++) {
      const xnn_simd_f32_t vt1 = xnn_mul_f32(vt0, vt0);
      const xnn_simd_f32_t vt2 = xnn_mul_f32(vx, vt1);
      const xnn_simd_f32_t vt3 = xnn_sub_f32(vt2, kOne);
      const xnn_simd_f32_t vt4 = xnn_mul_f32(kHalf, vt0);
      const xnn_simd_f32_t vt5 = xnn_mul_f32(vt3, vt4);
      vt0 = xnn_sub_f32(vt0, vt5);
    }

    // Mask out the inputs that were zero.
    const xnn_simd_f32_t vt7 = xnn_and_f32(vinf_mask, vt0);

    // Extract the square root by multiplying the original value with the
    // inverse square root.
    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vt7);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vsqrt_ukernel__sse2_rsqrt_u8(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)]) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const xnn_simd_f32_t kOne = xnn_set1_f32(1.0f);
  const xnn_simd_f32_t kHalf = xnn_set1_f32(0.5f);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 4);
    input += 8;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const xnn_simd_f32_t vinf_mask_0 = xnn_cmpneq_f32(vx0, xnn_zero_f32());
    const xnn_simd_f32_t vinf_mask_1 = xnn_cmpneq_f32(vx1, xnn_zero_f32());

    // Generate the initial 12-bit approximation.
    xnn_simd_f32_t vt0_0 = xnn_rsqrt_f32(vx0);
    xnn_simd_f32_t vt0_1 = xnn_rsqrt_f32(vx1);

    // Do a fixed number of Newton-Raphson steps as described above.
    for (size_t i = 0; i < XNN_SIMD_NUM_RSQRT_ITER_F32; i++) {
      const xnn_simd_f32_t vt1_0 = xnn_mul_f32(vt0_0, vt0_0);
      const xnn_simd_f32_t vt1_1 = xnn_mul_f32(vt0_1, vt0_1);
      const xnn_simd_f32_t vt2_0 = xnn_mul_f32(vx0, vt1_0);
      const xnn_simd_f32_t vt2_1 = xnn_mul_f32(vx1, vt1_1);
      const xnn_simd_f32_t vt3_0 = xnn_sub_f32(vt2_0, kOne);
      const xnn_simd_f32_t vt3_1 = xnn_sub_f32(vt2_1, kOne);
      const xnn_simd_f32_t vt4_0 = xnn_mul_f32(kHalf, vt0_0);
      const xnn_simd_f32_t vt4_1 = xnn_mul_f32(kHalf, vt0_1);
      const xnn_simd_f32_t vt5_0 = xnn_mul_f32(vt3_0, vt4_0);
      const xnn_simd_f32_t vt5_1 = xnn_mul_f32(vt3_1, vt4_1);
      vt0_0 = xnn_sub_f32(vt0_0, vt5_0);
      vt0_1 = xnn_sub_f32(vt0_1, vt5_1);
    }

    // Mask out the inputs that were zero.
    const xnn_simd_f32_t vt7_0 = xnn_and_f32(vinf_mask_0, vt0_0);
    const xnn_simd_f32_t vt7_1 = xnn_and_f32(vinf_mask_1, vt0_1);

    // Extract the square root by multiplying the original value with the
    // inverse square root.
    const xnn_simd_f32_t vy0 = xnn_mul_f32(vx0, vt7_0);
    const xnn_simd_f32_t vy1 = xnn_mul_f32(vx1, vt7_1);

    // Store the results.
    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 4, vy1);
    output += 8;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const xnn_simd_f32_t vinf_mask = xnn_cmpneq_f32(vx, xnn_zero_f32());

    // Generate the initial 12-bit approximation.
    xnn_simd_f32_t vt0 = xnn_rsqrt_f32(vx);

    // Do a fixed number of Newton-Raphson steps as described above.
    for (size_t i = 0; i < XNN_SIMD_NUM_RSQRT_ITER_F32; i++) {
      const xnn_simd_f32_t vt1 = xnn_mul_f32(vt0, vt0);
      const xnn_simd_f32_t vt2 = xnn_mul_f32(vx, vt1);
      const xnn_simd_f32_t vt3 = xnn_sub_f32(vt2, kOne);
      const xnn_simd_f32_t vt4 = xnn_mul_f32(kHalf, vt0);
      const xnn_simd_f32_t vt5 = xnn_mul_f32(vt3, vt4);
      vt0 = xnn_sub_f32(vt0, vt5);
    }

    // Mask out the inputs that were zero.
    const xnn_simd_f32_t vt7 = xnn_and_f32(vinf_mask, vt0);

    // Extract the square root by multiplying the original value with the
    // inverse square root.
    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vt7);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const xnn_simd_f32_t vinf_mask = xnn_cmpneq_f32(vx, xnn_zero_f32());

    // Generate the initial 12-bit approximation.
    xnn_simd_f32_t vt0 = xnn_rsqrt_f32(vx);

    // Do a fixed number of Newton-Raphson steps as described above.
    for (size_t i = 0; i < XNN_SIMD_NUM_RSQRT_ITER_F32; i++) {
      const xnn_simd_f32_t vt1 = xnn_mul_f32(vt0, vt0);
      const xnn_simd_f32_t vt2 = xnn_mul_f32(vx, vt1);
      const xnn_simd_f32_t vt3 = xnn_sub_f32(vt2, kOne);
      const xnn_simd_f32_t vt4 = xnn_mul_f32(kHalf, vt0);
      const xnn_simd_f32_t vt5 = xnn_mul_f32(vt3, vt4);
      vt0 = xnn_sub_f32(vt0, vt5);
    }

    // Mask out the inputs that were zero.
    const xnn_simd_f32_t vt7 = xnn_and_f32(vinf_mask, vt0);

    // Extract the square root by multiplying the original value with the
    // inverse square root.
    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vt7);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vsqrt_ukernel__sse2_rsqrt_u16(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)]) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const xnn_simd_f32_t kOne = xnn_set1_f32(1.0f);
  const xnn_simd_f32_t kHalf = xnn_set1_f32(0.5f);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 4);
    const xnn_simd_f32_t vx2 = xnn_loadu_f32(input + 8);
    const xnn_simd_f32_t vx3 = xnn_loadu_f32(input + 12);
    input += 16;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const xnn_simd_f32_t vinf_mask_0 = xnn_cmpneq_f32(vx0, xnn_zero_f32());
    const xnn_simd_f32_t vinf_mask_1 = xnn_cmpneq_f32(vx1, xnn_zero_f32());
    const xnn_simd_f32_t vinf_mask_2 = xnn_cmpneq_f32(vx2, xnn_zero_f32());
    const xnn_simd_f32_t vinf_mask_3 = xnn_cmpneq_f32(vx3, xnn_zero_f32());

    // Generate the initial 12-bit approximation.
    xnn_simd_f32_t vt0_0 = xnn_rsqrt_f32(vx0);
    xnn_simd_f32_t vt0_1 = xnn_rsqrt_f32(vx1);
    xnn_simd_f32_t vt0_2 = xnn_rsqrt_f32(vx2);
    xnn_simd_f32_t vt0_3 = xnn_rsqrt_f32(vx3);

    // Do a fixed number of Newton-Raphson steps as described above.
    for (size_t i = 0; i < XNN_SIMD_NUM_RSQRT_ITER_F32; i++) {
      const xnn_simd_f32_t vt1_0 = xnn_mul_f32(vt0_0, vt0_0);
      const xnn_simd_f32_t vt1_1 = xnn_mul_f32(vt0_1, vt0_1);
      const xnn_simd_f32_t vt1_2 = xnn_mul_f32(vt0_2, vt0_2);
      const xnn_simd_f32_t vt1_3 = xnn_mul_f32(vt0_3, vt0_3);
      const xnn_simd_f32_t vt2_0 = xnn_mul_f32(vx0, vt1_0);
      const xnn_simd_f32_t vt2_1 = xnn_mul_f32(vx1, vt1_1);
      const xnn_simd_f32_t vt2_2 = xnn_mul_f32(vx2, vt1_2);
      const xnn_simd_f32_t vt2_3 = xnn_mul_f32(vx3, vt1_3);
      const xnn_simd_f32_t vt3_0 = xnn_sub_f32(vt2_0, kOne);
      const xnn_simd_f32_t vt3_1 = xnn_sub_f32(vt2_1, kOne);
      const xnn_simd_f32_t vt3_2 = xnn_sub_f32(vt2_2, kOne);
      const xnn_simd_f32_t vt3_3 = xnn_sub_f32(vt2_3, kOne);
      const xnn_simd_f32_t vt4_0 = xnn_mul_f32(kHalf, vt0_0);
      const xnn_simd_f32_t vt4_1 = xnn_mul_f32(kHalf, vt0_1);
      const xnn_simd_f32_t vt4_2 = xnn_mul_f32(kHalf, vt0_2);
      const xnn_simd_f32_t vt4_3 = xnn_mul_f32(kHalf, vt0_3);
      const xnn_simd_f32_t vt5_0 = xnn_mul_f32(vt3_0, vt4_0);
      const xnn_simd_f32_t vt5_1 = xnn_mul_f32(vt3_1, vt4_1);
      const xnn_simd_f32_t vt5_2 = xnn_mul_f32(vt3_2, vt4_2);
      const xnn_simd_f32_t vt5_3 = xnn_mul_f32(vt3_3, vt4_3);
      vt0_0 = xnn_sub_f32(vt0_0, vt5_0);
      vt0_1 = xnn_sub_f32(vt0_1, vt5_1);
      vt0_2 = xnn_sub_f32(vt0_2, vt5_2);
      vt0_3 = xnn_sub_f32(vt0_3, vt5_3);
    }

    // Mask out the inputs that were zero.
    const xnn_simd_f32_t vt7_0 = xnn_and_f32(vinf_mask_0, vt0_0);
    const xnn_simd_f32_t vt7_1 = xnn_and_f32(vinf_mask_1, vt0_1);
    const xnn_simd_f32_t vt7_2 = xnn_and_f32(vinf_mask_2, vt0_2);
    const xnn_simd_f32_t vt7_3 = xnn_and_f32(vinf_mask_3, vt0_3);

    // Extract the square root by multiplying the original value with the
    // inverse square root.
    const xnn_simd_f32_t vy0 = xnn_mul_f32(vx0, vt7_0);
    const xnn_simd_f32_t vy1 = xnn_mul_f32(vx1, vt7_1);
    const xnn_simd_f32_t vy2 = xnn_mul_f32(vx2, vt7_2);
    const xnn_simd_f32_t vy3 = xnn_mul_f32(vx3, vt7_3);

    // Store the results.
    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 4, vy1);
    xnn_storeu_f32(output + 8, vy2);
    xnn_storeu_f32(output + 12, vy3);
    output += 16;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const xnn_simd_f32_t vinf_mask = xnn_cmpneq_f32(vx, xnn_zero_f32());

    // Generate the initial 12-bit approximation.
    xnn_simd_f32_t vt0 = xnn_rsqrt_f32(vx);

    // Do a fixed number of Newton-Raphson steps as described above.
    for (size_t i = 0; i < XNN_SIMD_NUM_RSQRT_ITER_F32; i++) {
      const xnn_simd_f32_t vt1 = xnn_mul_f32(vt0, vt0);
      const xnn_simd_f32_t vt2 = xnn_mul_f32(vx, vt1);
      const xnn_simd_f32_t vt3 = xnn_sub_f32(vt2, kOne);
      const xnn_simd_f32_t vt4 = xnn_mul_f32(kHalf, vt0);
      const xnn_simd_f32_t vt5 = xnn_mul_f32(vt3, vt4);
      vt0 = xnn_sub_f32(vt0, vt5);
    }

    // Mask out the inputs that were zero.
    const xnn_simd_f32_t vt7 = xnn_and_f32(vinf_mask, vt0);

    // Extract the square root by multiplying the original value with the
    // inverse square root.
    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vt7);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const xnn_simd_f32_t vinf_mask = xnn_cmpneq_f32(vx, xnn_zero_f32());

    // Generate the initial 12-bit approximation.
    xnn_simd_f32_t vt0 = xnn_rsqrt_f32(vx);

    // Do a fixed number of Newton-Raphson steps as described above.
    for (size_t i = 0; i < XNN_SIMD_NUM_RSQRT_ITER_F32; i++) {
      const xnn_simd_f32_t vt1 = xnn_mul_f32(vt0, vt0);
      const xnn_simd_f32_t vt2 = xnn_mul_f32(vx, vt1);
      const xnn_simd_f32_t vt3 = xnn_sub_f32(vt2, kOne);
      const xnn_simd_f32_t vt4 = xnn_mul_f32(kHalf, vt0);
      const xnn_simd_f32_t vt5 = xnn_mul_f32(vt3, vt4);
      vt0 = xnn_sub_f32(vt0, vt5);
    }

    // Mask out the inputs that were zero.
    const xnn_simd_f32_t vt7 = xnn_and_f32(vinf_mask, vt0);

    // Extract the square root by multiplying the original value with the
    // inverse square root.
    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vt7);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
