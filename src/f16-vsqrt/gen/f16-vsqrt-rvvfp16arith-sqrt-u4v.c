// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vsqrt/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vsqrt_ukernel__rvvfp16arith_sqrt_u4v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;
  do {
    const size_t vl = __riscv_vsetvl_e16m4(batch);
    vfloat16m4_t vx = __riscv_vle16_v_f16m4(input, vl);
    input += vl;
    vfloat16m4_t vacc = __riscv_vfsqrt(vx, vl);
    __riscv_vse16(output, vacc, vl);
    output += vl;

    batch -= vl;
  } while (batch != 0);
}
