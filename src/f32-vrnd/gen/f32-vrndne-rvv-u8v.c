// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vrndne_ukernel__rvv_u8v(
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
    const size_t n = __riscv_vsetvl_e32m8(batch);
    vfloat32m8_t x_f32v = __riscv_vle32_v_f32m8(input, n); input += n;
    // We need to remember which values are infinity, so we can preserve them
    // after rounding.
    // TODO: We should also preserve NaN.
    vbool4_t inf_bv = __riscv_vmfeq_vf_f32m8_b4(x_f32v, INFINITY, n);
    vbool4_t ninf_bv = __riscv_vmfeq_vf_f32m8_b4(x_f32v, -INFINITY, n);
    vbool4_t mask_bv = __riscv_vmor_mm_b4(inf_bv, ninf_bv, n);
    vint32m8_t x_rnd_i32v = __riscv_vfcvt_x_f_v_i32m8_rm(x_f32v, __RISCV_FRM_RNE, n);
    vfloat32m8_t out_f32v = __riscv_vfcvt_f_x_v_f32m8(x_rnd_i32v, n);
    out_f32v = __riscv_vmerge_vvm_f32m8(out_f32v, x_f32v, mask_bv, n);
    __riscv_vse32_v_f32m8(output, out_f32v, n); output += n;
    batch -= n;
  } while (batch != 0);
}
