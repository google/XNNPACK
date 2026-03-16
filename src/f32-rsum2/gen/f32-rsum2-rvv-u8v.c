// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rsum2/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <riscv_vector.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"


void xnn_f32_rsum2_ukernel__rvv_u8v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_scale_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
  size_t vlmax = __riscv_vsetvl_e32m8(batch);
  vfloat32m8_t vsum2 = __riscv_vfmv_v_f_f32m8(0.f, vlmax);

  do {
    size_t vl = __riscv_vsetvl_e32m8(batch); batch -= vl;

    vfloat32m8_t vinput = __riscv_vle32_v_f32m8(input, vl); input += vl;

    vsum2 = __riscv_vfmacc(vsum2, vinput, vinput, vl);

  } while (batch > 0);

  vfloat32m1_t vred = __riscv_vfredosum(vsum2, vzero, vlmax);
  const float vscale = params->scalar.scale;
  *output += __riscv_vfmv_f_s_f32m1_f32(vred) * vscale;
}
