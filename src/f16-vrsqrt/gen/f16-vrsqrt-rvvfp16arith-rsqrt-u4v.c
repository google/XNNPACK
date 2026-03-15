// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vrsqrt/rvv.c.in
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

void xnn_f16_vrsqrt_ukernel__rvvfp16arith_rsqrt_u4v(
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

  vfloat16m4_t onephalf_f16v = __riscv_vfmv_v_f_f16m4(1.5f, __riscv_vsetvl_e16m4(batch));
  vfloat16m4_t zero_f16v = __riscv_vfmv_v_f_f16m4(0.0f, __riscv_vsetvl_e16m4(batch));

  for (; batch > 0; ) {
    size_t n = __riscv_vsetvl_e16m4(batch); batch -= n;
    vfloat16m4_t in_f16v = __riscv_vle16_v_f16m4(input, n); input += n;

    vfloat16m4_t t0_f16v = __riscv_vfrsqrt7(in_f16v, n);
    vfloat16m4_t in_half_f16v = __riscv_vfmul(in_f16v, 0.5f, n);

    // First Newton-Raphson iteration
    vfloat16m4_t t1_f16v = __riscv_vfmul(t0_f16v, t0_f16v, n);
    vfloat16m4_t t2_f16v = __riscv_vfnmsub_vv_f16m4(in_half_f16v, t1_f16v, onephalf_f16v, n);
    t0_f16v = __riscv_vfmul(t0_f16v, t2_f16v, n);

    // Second Newton-Raphson iteration
    t1_f16v = __riscv_vfmul(t0_f16v, t0_f16v, n);
    t2_f16v = __riscv_vfnmsub_vv_f16m4(in_half_f16v, t1_f16v, onephalf_f16v, n);
    vfloat16m4_t y_f16v = __riscv_vfmul(t0_f16v, t2_f16v, n);

    // Set output to 0 where the input is infinity (and not NaN)
    vbool4_t inf_bv = __riscv_vmfeq(in_f16v, INFINITY, n);
    y_f16v = __riscv_vmerge(y_f16v, zero_f16v, inf_bv, n);

    __riscv_vse16(output, y_f16v, n); output += n;
  }
}
