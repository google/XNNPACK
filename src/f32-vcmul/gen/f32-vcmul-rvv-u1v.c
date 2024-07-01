// Auto-generated file. Do not edit!
//   Template: src/f32-vcmul/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/vbinary.h"
#include <riscv_vector.h>

void xnn_f32_vcmul_ukernel__rvv_u1v(
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

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  const float* ar = input_a;
  const float* ai = input_a + batch;
  const float* br = input_b;
  const float* bi = input_b + batch;
  float* or = output;
  float* oi = output + batch;

  for (; batch > 0; ) {
    size_t n = __riscv_vsetvl_e32m1(batch); batch -= n;
    vfloat32m1_t ar_f32v = __riscv_vle32_v_f32m1(ar, n); ar += n;
    vfloat32m1_t ai_f32v = __riscv_vle32_v_f32m1(ai, n); ai += n;
    vfloat32m1_t br_f32v = __riscv_vle32_v_f32m1(br, n); br += n;
    vfloat32m1_t bi_f32v = __riscv_vle32_v_f32m1(bi, n); bi += n;
    vfloat32m1_t ai_bi_f32v = __riscv_vfmul_vv_f32m1(ai_f32v, bi_f32v, n);
    vfloat32m1_t ai_br_f32v = __riscv_vfmul_vv_f32m1(ai_f32v, br_f32v, n);
    vfloat32m1_t ar_br_sub_ai_bi_f32v = __riscv_vfmsac_vv_f32m1(ai_bi_f32v, ar_f32v, br_f32v, n);
    vfloat32m1_t ar_bi_plus_ai_br_f32v = __riscv_vfmacc_vv_f32m1(ai_br_f32v, ar_f32v, bi_f32v, n);
    __riscv_vse32_v_f32m1(or, ar_br_sub_ai_bi_f32v, n); or += n;
    __riscv_vse32_v_f32m1(oi, ar_bi_plus_ai_br_f32v, n); oi += n;
  }
}
