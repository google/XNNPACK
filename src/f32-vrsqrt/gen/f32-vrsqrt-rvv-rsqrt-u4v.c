// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"

// In the following, we do a single Newton-Raphson step on the equation
// $x^{-2} - a$, which expands to:
//
//  $$x_{k+1} = 0.5 * x_k * (3.0 - a * x_k^2)$$
//
// So we do the following steps:
//
//  1. t0 = x_k
//  2. t1 = t0 * t0   (x_k^2)
//  3. t2 = a * t1    (a * x_k^2)
//  4. t3 = 3.0 - t2  (3.0 - a * x_k^2)
//  5. t4 = 0.5 * t0  (0.5 * x_k)
//  6. y  = t3 * t4   (0.5 * x_k * (3.0 - a * x_k^2))
//
// Where $x_k$ is the original approximation and `y` contains the improved
// approximation $x_{k+1}$.
//
// Note also that the initial approximation computed by the `vfrsqrt7`
// instruction is only accurate to 7 bits (as opposed to 12 or 14 for x86_64),
// which requires us to do two steps of the above.

void xnn_f32_vrsqrt_ukernel__rvv_rsqrt_u4v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  vfloat32m4_t onephalf_f32v = __riscv_vfmv_v_f_f32m4(1.5f, __riscv_vsetvl_e32m4(batch));
  vfloat32m4_t zero_f32v = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvl_e32m4(batch));
  for (; batch > 0; ) {
    int32_t n = __riscv_vsetvl_e32m4(batch); batch -= n;
    vfloat32m4_t in_f32v = __riscv_vle32_v_f32m4(input, n); input += n;

    vfloat32m4_t t0_f32v = __riscv_vfrsqrt7_v_f32m4(in_f32v, n);
    vfloat32m4_t in_half_f32v = __riscv_vfmul_vf_f32m4(in_f32v, 0.5f, n);

    // First Newton-Raphson iteration
    vfloat32m4_t t1_f32v = __riscv_vfmul_vv_f32m4(t0_f32v, t0_f32v, n);
    vfloat32m4_t t2_f32v = __riscv_vfnmsub_vv_f32m4(in_half_f32v, t1_f32v, onephalf_f32v, n);
    t0_f32v = __riscv_vfmul_vv_f32m4(t0_f32v, t2_f32v, n);

    // Second Newton-Raphson iteration
    t1_f32v = __riscv_vfmul_vv_f32m4(t0_f32v, t0_f32v, n);
    t2_f32v = __riscv_vfnmsub_vv_f32m4(in_half_f32v, t1_f32v, onephalf_f32v, n);
    vfloat32m4_t y_f32v = __riscv_vfmul_vv_f32m4(t0_f32v, t2_f32v, n);

    // Set output to 0 where the input is infinity (and not NaN)
    vbool8_t inf_bv = __riscv_vmfeq_vf_f32m4_b8(in_f32v, INFINITY, n);
    y_f32v = __riscv_vmerge_vvm_f32m4(y_f32v, zero_f32v, inf_bv, n);

    __riscv_vse32_v_f32m4(output, y_f32v, n); output += n;
  }
}
