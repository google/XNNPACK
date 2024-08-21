// Auto-generated file. Do not edit!
//   Template: src/f32-vrem/f32-vrem.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/simd/f32-hvx.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_f32_vrem_ukernel__hvx_u32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 32);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vin1 = xnn_loadu_f32(input_a);
    input_a += xnn_simd_size_f32;

    xnn_simd_f32_t vin2 = xnn_loadu_f32(input_b);
    input_b += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_rem_f32(vin1, vin2);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vin1 = xnn_load_tail_f32(input_a, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vin2 = xnn_load_tail_f32(input_b, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_rem_f32(vin1, vin2);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vrem_ukernel__hvx_u64(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 32);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    xnn_simd_f32_t vin1_0 = xnn_loadu_f32(input_a);
    xnn_simd_f32_t vin1_1 = xnn_loadu_f32(input_a + 1 * xnn_simd_size_f32);
    input_a += 64;

    xnn_simd_f32_t vin2_0 = xnn_loadu_f32(input_b);
    xnn_simd_f32_t vin2_1 = (xnn_loadu_f32(input_b + 1 * xnn_simd_size_f32));
    input_b += 64;

    xnn_simd_f32_t vy_0 = xnn_rem_f32(vin1_0, vin2_0);
    xnn_simd_f32_t vy_1 = xnn_rem_f32(vin1_1, vin2_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 64;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vin1 = xnn_loadu_f32(input_a);
    input_a += xnn_simd_size_f32;

    xnn_simd_f32_t vin2 = xnn_loadu_f32(input_b);
    input_b += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_rem_f32(vin1, vin2);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vin1 = xnn_load_tail_f32(input_a, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vin2 = xnn_load_tail_f32(input_b, batch >> XNN_LOG2_SIZEOF_FLOAT);

    xnn_simd_f32_t vy = xnn_rem_f32(vin1, vin2);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
