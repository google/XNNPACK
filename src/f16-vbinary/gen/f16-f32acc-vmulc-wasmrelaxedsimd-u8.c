// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/f16-f32acc-vopc.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/simd/f32-wasmrelaxedsimd.h"
#include "src/xnnpack/vbinary.h"

#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"


static XNN_INLINE xnn_simd_f32_t load_f16_f32(const xnn_float16* input) {
  return xnn_cvt_f32_f16(xnn_loadu_f16(input));
}

static XNN_INLINE xnn_simd_f32_t load_tail_f16_f32(
    const xnn_float16* input, size_t num_elements)
{
  return xnn_cvt_f32_f16(xnn_load_tail_f16(input, num_elements));
}

static XNN_INLINE void store_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value)
{
  xnn_store_tail_f16(output, xnn_cvt_f16_f32(value), xnn_simd_size_f32);
}

static XNN_INLINE void store_tail_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value, size_t num_elements)
{
  xnn_store_tail_f16(output, xnn_cvt_f16_f32(value), num_elements);
}

void xnn_f16_f32acc_vmulc_ukernel__wasmrelaxedsimd_u8(
    size_t batch,
    const xnn_float16* restrict input_a,
    const xnn_float16* restrict input_b,
    xnn_float16* restrict output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const xnn_simd_f32_t vb = xnn_set1_f32(xnn_float16_to_float(*input_b));
  for (; batch >= 8 * sizeof(xnn_float16);
       batch -= 8 * sizeof(xnn_float16)) {
    const xnn_simd_f32_t va0 = load_f16_f32(input_a + 0);
    const xnn_simd_f32_t va1 = load_f16_f32(input_a + 4);
    input_a += 8;
    const xnn_simd_f32_t vy0 = xnn_mul_f32(va0, vb);
    store_f32_f16(output + 0, vy0);
    const xnn_simd_f32_t vy1 = xnn_mul_f32(va1, vb);
    store_f32_f16(output + 4, vy1);
    output += 8;
  }

  for (; batch >= 4 * sizeof(xnn_float16);
       batch -= 4 * sizeof(xnn_float16)) {
    const xnn_simd_f32_t va = load_f16_f32(input_a);
    input_a += 4;
    store_f32_f16(output, xnn_mul_f32(va, vb));
    output += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    const size_t num_elements = batch / sizeof(xnn_float16);
    const xnn_simd_f32_t va = load_tail_f16_f32(input_a, num_elements);
    store_tail_f32_f16(output, xnn_mul_f32(va, vb), num_elements);
  }
}
