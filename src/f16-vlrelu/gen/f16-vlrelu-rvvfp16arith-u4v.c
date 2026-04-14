// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vlrelu/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Imagination Technologies, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"

void xnn_f16_vlrelu_ukernel__rvvfp16arith_u4v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_lrelu_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const xnn_float16 slope = params->scalar.slope;
  batch >>= XNN_LOG2_SIZEOF_FLOAT16;

  do {
    size_t vl = __riscv_vsetvl_e16m4(batch);
    vfloat16m4_t in_f16v = __riscv_vle16_v_f16m4(input, vl); input += vl;
    vbool4_t mask_f16v = __riscv_vmflt(in_f16v, 0.0f, vl);
    vfloat16m4_t out_f16v = __riscv_vfmul(mask_f16v, in_f16v, slope, vl);
    __riscv_vse16(output, out_f16v, vl); output += vl;

    batch -= vl;
  } while (batch != 0);
}
