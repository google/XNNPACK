// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vsqrt/simd-sqrt.c.in
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
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vsqrt_ukernel__wasmrelaxedsimd_sqrt_u8(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 8);

  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(input);
    input += 8;

    xnn_simd_f16_t vy = xnn_sqrt_f16(vx);

    xnn_storeu_f16(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);
    xnn_simd_f16_t vy = xnn_sqrt_f16(vx);
    xnn_store_tail_f16(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_vsqrt_ukernel__wasmrelaxedsimd_sqrt_u16(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 8);

  for (; batch >= 16 * sizeof(xnn_float16); batch -= 16 * sizeof(xnn_float16)) {
    xnn_simd_f16_t vx_0 = xnn_loadu_f16(input + 0 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    input += 16;

    xnn_simd_f16_t vy_0 = xnn_sqrt_f16(vx_0);
    xnn_simd_f16_t vy_1 = xnn_sqrt_f16(vx_1);

    xnn_storeu_f16(output + 0 * xnn_simd_size_f16, vy_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_1);
    output += 16;
  }
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(input);
    input += 8;

    xnn_simd_f16_t vy = xnn_sqrt_f16(vx);

    xnn_storeu_f16(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);
    xnn_simd_f16_t vy = xnn_sqrt_f16(vx);
    xnn_store_tail_f16(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}

void xnn_f16_vsqrt_ukernel__wasmrelaxedsimd_sqrt_u32(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f16 == 8);

  for (; batch >= 32 * sizeof(xnn_float16); batch -= 32 * sizeof(xnn_float16)) {
    xnn_simd_f16_t vx_0 = xnn_loadu_f16(input + 0 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_1 = xnn_loadu_f16(input + 1 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_2 = xnn_loadu_f16(input + 2 * xnn_simd_size_f16);
    xnn_simd_f16_t vx_3 = xnn_loadu_f16(input + 3 * xnn_simd_size_f16);
    input += 32;

    xnn_simd_f16_t vy_0 = xnn_sqrt_f16(vx_0);
    xnn_simd_f16_t vy_1 = xnn_sqrt_f16(vx_1);
    xnn_simd_f16_t vy_2 = xnn_sqrt_f16(vx_2);
    xnn_simd_f16_t vy_3 = xnn_sqrt_f16(vx_3);

    xnn_storeu_f16(output + 0 * xnn_simd_size_f16, vy_0);
    xnn_storeu_f16(output + 1 * xnn_simd_size_f16, vy_1);
    xnn_storeu_f16(output + 2 * xnn_simd_size_f16, vy_2);
    xnn_storeu_f16(output + 3 * xnn_simd_size_f16, vy_3);
    output += 32;
  }
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    xnn_simd_f16_t vx = xnn_loadu_f16(input);
    input += 8;

    xnn_simd_f16_t vy = xnn_sqrt_f16(vx);

    xnn_storeu_f16(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f16_t vx = xnn_load_tail_f16(input, batch >> XNN_LOG2_SIZEOF_FLOAT16);
    xnn_simd_f16_t vy = xnn_sqrt_f16(vx);
    xnn_store_tail_f16(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT16);
  }
}
