// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/simd-sqrt.c.in
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
#include "src/xnnpack/simd/f32-scalar.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vrsqrt_ukernel__scalar_sqrt_u1(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)]) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_SIMD_CONST_F32(vone, 1.0f);


  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_sqrt_f32(vx);
    vy = xnn_div_f32(vone, vy);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 0 * sizeof(float));
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_sqrt_f32(vx);
    vy = xnn_div_f32(vone, vy);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vrsqrt_ukernel__scalar_sqrt_u2(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)]) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 1);
    input += 2;

    xnn_simd_f32_t vy0 = xnn_sqrt_f32(vx0);
    xnn_simd_f32_t vy1 = xnn_sqrt_f32(vx1);
    vy0 = xnn_div_f32(vone, vy0);
    vy1 = xnn_div_f32(vone, vy1);

    // Store the results.
    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1, vy1);
    output += 2;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_sqrt_f32(vx);
    vy = xnn_div_f32(vone, vy);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 0 * sizeof(float));
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_sqrt_f32(vx);
    vy = xnn_div_f32(vone, vy);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vrsqrt_ukernel__scalar_sqrt_u4(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)]) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_SIMD_CONST_F32(vone, 1.0f);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 1);
    const xnn_simd_f32_t vx2 = xnn_loadu_f32(input + 2);
    const xnn_simd_f32_t vx3 = xnn_loadu_f32(input + 3);
    input += 4;

    xnn_simd_f32_t vy0 = xnn_sqrt_f32(vx0);
    xnn_simd_f32_t vy1 = xnn_sqrt_f32(vx1);
    xnn_simd_f32_t vy2 = xnn_sqrt_f32(vx2);
    xnn_simd_f32_t vy3 = xnn_sqrt_f32(vx3);
    vy0 = xnn_div_f32(vone, vy0);
    vy1 = xnn_div_f32(vone, vy1);
    vy2 = xnn_div_f32(vone, vy2);
    vy3 = xnn_div_f32(vone, vy3);

    // Store the results.
    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1, vy1);
    xnn_storeu_f32(output + 2, vy2);
    xnn_storeu_f32(output + 3, vy3);
    output += 4;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_sqrt_f32(vx);
    vy = xnn_div_f32(vone, vy);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 0 * sizeof(float));
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_sqrt_f32(vx);
    vy = xnn_div_f32(vone, vy);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
