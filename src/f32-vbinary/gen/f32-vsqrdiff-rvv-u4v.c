// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


void xnn_f32_vsqrdiff_ukernel__rvv_u4v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m4(n);
    n -= vl;
    vfloat32m4_t va = __riscv_vle32_v_f32m4(input_a, vl);
    input_a += vl;
    vfloat32m4_t vb = __riscv_vle32_v_f32m4(input_b, vl);
    input_b += vl;
    vfloat32m4_t vacc = __riscv_vfsub_vv_f32m4(va, vb, vl);
    vacc = __riscv_vfmul_vv_f32m4(vacc, vacc, vl);
    __riscv_vse32_v_f32m4(output, vacc, vl);
    output += vl;
  } while (n > 0);
}
