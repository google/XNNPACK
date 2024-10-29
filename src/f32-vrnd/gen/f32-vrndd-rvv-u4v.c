// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"


void xnn_f32_vrndd_ukernel__rvv_u4v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t n = __riscv_vsetvl_e32m4(batch);
    vfloat32m4_t x_f32v = __riscv_vle32_v_f32m4(input, n); input += n;
    vint32m4_t x_rnd_i32v = __riscv_vfcvt_x_f_v_i32m4_rm(x_f32v, __RISCV_FRM_RDN, n);
    vfloat32m4_t out_f32v = __riscv_vfcvt_f_x_v_f32m4(x_rnd_i32v, n);
    __riscv_vse32_v_f32m4(output, out_f32v, n); output += n;
    batch -= n;
  } while (batch != 0);
}
