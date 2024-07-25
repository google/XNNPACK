// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"

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

void xnn_f32_vrsqrt_ukernel__rvv_rsqrt_u2v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  vfloat32m2_t onephalf_f32v = __riscv_vfmv_v_f_f32m2(1.5f, __riscv_vsetvl_e32m2(batch));
  for (; batch > 0; ) {
    int32_t n = __riscv_vsetvl_e32m2(batch); batch -= n;
    vfloat32m2_t in_f32v = __riscv_vle32_v_f32m2(input, n); input += n;

    vfloat32m2_t t0_f32v = __riscv_vfrsqrt7_v_f32m2(in_f32v, n);
    vfloat32m2_t in_half_f32v = __riscv_vfmul_vf_f32m2(in_f32v, 0.5f, n);

    // First Newton-Raphson iteration
    vfloat32m2_t t1_f32v = __riscv_vfmul_vv_f32m2(t0_f32v, t0_f32v, n);
    vfloat32m2_t t2_f32v = __riscv_vfnmsub_vv_f32m2(in_half_f32v, t1_f32v, onephalf_f32v, n);
    t0_f32v = __riscv_vfmul_vv_f32m2(t0_f32v, t2_f32v, n);

    // Second Newton-Raphson iteration
    t1_f32v = __riscv_vfmul_vv_f32m2(t0_f32v, t0_f32v, n);
    t2_f32v = __riscv_vfnmsub_vv_f32m2(in_half_f32v, t1_f32v, onephalf_f32v, n);
    vfloat32m2_t y_f32v = __riscv_vfmul_vv_f32m2(t0_f32v, t2_f32v, n);

    __riscv_vse32_v_f32m2(output, y_f32v, n); output += n;
  }
}
