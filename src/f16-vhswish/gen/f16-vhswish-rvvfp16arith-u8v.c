// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vhswish/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vhswish_ukernel__rvvfp16arith_u8v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const xnn_float16 vsixth = 0x1.555556p-3f;
  const xnn_float16 vthree = 3.0f;
  const xnn_float16 vsix = 6.0f;
  const xnn_float16 vzero = 0.0f;

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;
  do {
    const size_t vl = __riscv_vsetvl_e16m8(batch);
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(input, vl);
    input += vl;
    vfloat16m8_t vacc = __riscv_vfadd(vx, vthree, vl);
    vx = __riscv_vfmul(vx, vsixth, vl);
    vacc = __riscv_vfmax(vacc, vzero, vl);
    vacc = __riscv_vfmin(vacc, vsix, vl);
    vacc = __riscv_vfmul(vacc, vx, vl);
    __riscv_vse16(output, vacc, vl);
    output += vl;

    batch -= vl;
  } while (batch != 0);
}
