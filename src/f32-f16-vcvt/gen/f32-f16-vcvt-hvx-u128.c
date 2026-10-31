// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/vcvt.h"


void xnn_f32_f16_vcvt_ukernel__hvx_u128(
    size_t batch,
    const float* input,
    xnn_float16* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input + 0);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 32);
    const xnn_simd_f32_t vx2 = xnn_loadu_f32(input + 64);
    const xnn_simd_f32_t vx3 = xnn_loadu_f32(input + 96);
    input += 128;

    const xnn_simd_f16_t vy0 = xnn_cvt_f16_f32_f32(vx0, vx1);
    const xnn_simd_f16_t vy1 = xnn_cvt_f16_f32_f32(vx2, vx3);

    xnn_storeu_f16(output + 0, vy0);
    xnn_storeu_f16(output + 64, vy1);
    output += 128;
  }
  if (batch >= 64 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 32);
    input += 64;

    const xnn_simd_f16_t vy0 = xnn_cvt_f16_f32_f32(vx0, vx1);

    xnn_storeu_f16(output, vy0);
    output += 64;
    batch -= 64 * sizeof(float);
  }
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += 32;

    const xnn_simd_f16_t vy = xnn_cvt_f16_f32(vx);

    xnn_store_tail_f16(output, vy, 32);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 31 * sizeof(float));

    const size_t num_elements = batch >> XNN_LOG2_SIZEOF_FLOAT;
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, num_elements);
    const xnn_simd_f16_t vy = xnn_cvt_f16_f32(vx);
    xnn_store_tail_f16(output, vy, num_elements);
  }
}
