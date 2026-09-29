// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vopc.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"
#include "src/xnnpack/vbinary.h"


void xnn_f16_vaddc_ukernel__wasmrelaxedsimdfp16_u8(
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

  const xnn_simd_f16_t vb = xnn_set1_f16(*input_b);
  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    const xnn_simd_f16_t va = xnn_loadu_f16(input_a);
    input_a += xnn_simd_size_f16;
    const xnn_simd_f16_t vy = xnn_add_f16(va, vb);
    xnn_storeu_f16(output, vy);
    output += xnn_simd_size_f16;
  }

  if XNN_UNLIKELY(batch != 0) {
    const size_t num_elements = batch / sizeof(xnn_float16);
    const xnn_simd_f16_t va = xnn_load_tail_f16(input_a, num_elements);
    const xnn_simd_f16_t vy = xnn_add_f16(va, vb);
    xnn_store_tail_f16(output, vy, num_elements);
  }
}
