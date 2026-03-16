// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-rminmax/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2023 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"

void xnn_f16_rmax_ukernel__rvvfp16arith_u8v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t N = batch >> XNN_LOG2_SIZEOF_FLOAT16;
  size_t avl;
  size_t vl = __riscv_vsetvl_e16m8(N);

  vfloat16m8_t t0 = __riscv_vle16_v_f16m8(input, vl);
  input += vl;

  for (avl = N - vl; avl; avl -= vl, input += vl) {
    vl = __riscv_vsetvl_e16m8(avl);
    vfloat16m8_t vec = __riscv_vle16_v_f16m8(input, vl);
    t0 = __riscv_vfmax_vv_f16m8_tu(t0, t0, vec, vl);
  }

  vfloat16m1_t fmax = __riscv_vle16_v_f16m1(output + 0, 1);
  output[0] = __riscv_vfmv_f_s_f16m1_f16(__riscv_vfredmax_vs_f16m8_f16m1(t0, fmax, N));
}
