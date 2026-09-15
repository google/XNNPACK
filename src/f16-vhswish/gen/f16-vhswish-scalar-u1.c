// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vhswish/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/simd/f16-scalar.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vhswish_ukernel__scalar_u1(
    size_t batch,
    const xnn_float16* restrict input,
    xnn_float16* restrict output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_SIMD_CONST_F16_FROM_INT16(vsixth, 0x3155);
  XNN_SIMD_CONST_F16_FROM_INT16(vthree, 0x4200);
  XNN_SIMD_CONST_F16_FROM_INT16(vsix, 0x4600);
  XNN_SIMD_CONST_F16_FROM_INT16(vzero, 0x0000);

  for (; batch >= xnn_simd_bytes_f16; batch -= xnn_simd_bytes_f16) {
    const xnn_simd_f16_t vx = xnn_loadu_f16(input);
    input += xnn_simd_size_f16;
    xnn_simd_f16_t vacc = xnn_add_f16(vx, vthree);
    const xnn_simd_f16_t vx_scaled = xnn_mul_f16(vx, vsixth);
    vacc = xnn_max_f16(vacc, vzero);
    vacc = xnn_min_f16(vacc, vsix);
    vacc = xnn_mul_f16(vacc, vx_scaled);
    xnn_storeu_f16(output, vacc);
    output += xnn_simd_size_f16;
  }

}
