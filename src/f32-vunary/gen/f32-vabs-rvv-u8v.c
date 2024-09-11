// Auto-generated file. Do not edit!
//   Template: src/f32-vunary/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vabs_ukernel__rvv_u8v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t n = __riscv_vsetvl_e32m8(batch);
    const vfloat32m8_t vi = __riscv_vle32_v_f32m8(input, n);
    input += n;
    const vfloat32m8_t vo = __riscv_vfabs_v_f32m8(vi, n);
    __riscv_vse32_v_f32m8(output, vo, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
