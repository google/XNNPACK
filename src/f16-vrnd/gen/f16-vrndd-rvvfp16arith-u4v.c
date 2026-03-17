// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vrnd/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Imagination Technologies inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vrndd_ukernel__rvvfp16arith_u4v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;
  do {
    const size_t vl = __riscv_vsetvl_e16m4(batch);
    vfloat16m4_t x_f16v = __riscv_vle16_v_f16m4(input, vl); input += vl;

    // preserve NaN
    vbool4_t nan_bv = __riscv_vmfeq(x_f16v, x_f16v, vl);
    // magnitude < (1 << FLT16_MANT_DIG)
    vfloat16m4_t mag = __riscv_vfabs(x_f16v, vl);
    vbool4_t mag_bv = __riscv_vmflt(mag, (1 << __FLT16_MANT_DIG__), vl);
    vbool4_t mask_bv = __riscv_vmnand(nan_bv, mag_bv, vl);

    vint16m4_t x_rnd_i16v = __riscv_vfcvt_x(x_f16v, __RISCV_FRM_RDN, vl);
    vfloat16m4_t out_f16v = __riscv_vfcvt_f(x_rnd_i16v, vl);
    out_f16v = __riscv_vmerge(out_f16v, x_f16v, mask_bv, vl);
    __riscv_vse16(output, out_f16v, vl); output += vl;
    batch -= vl;
  } while (batch != 0);
}
