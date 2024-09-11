// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2023 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vmulc_ukernel__rvv_u4v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m4(n);
    n -= vl;
    vfloat32m4_t va = __riscv_vle32_v_f32m4(input_a, vl);
    input_a += vl;
    vfloat32m4_t vacc = __riscv_vfmul_vf_f32m4(va, b, vl);
    __riscv_vse32_v_f32m4(output, vacc, vl);
    output += vl;
  } while (n > 0);
}
