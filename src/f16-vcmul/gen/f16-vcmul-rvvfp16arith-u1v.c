// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vcmul/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/vbinary.h"

void xnn_f16_vcmul_ukernel__rvvfp16arith_u1v(
    size_t batch,
    const xnn_float16* input_a,
    const xnn_float16* input_b,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;

  const xnn_float16* ar = input_a;
  const xnn_float16* ai = input_a + batch;
  const xnn_float16* br = input_b;
  const xnn_float16* bi = input_b + batch;
  xnn_float16* or = output;
  xnn_float16* oi = output + batch;

  do {
    size_t n = __riscv_vsetvl_e16m1(batch); batch -= n;
    vfloat16m1_t var = __riscv_vle16_v_f16m1(ar, n); ar += n;
    vfloat16m1_t vai = __riscv_vle16_v_f16m1(ai, n); ai += n;
    vfloat16m1_t vbr = __riscv_vle16_v_f16m1(br, n); br += n;
    vfloat16m1_t vbi = __riscv_vle16_v_f16m1(bi, n); bi += n;
    vfloat16m1_t vaccr = __riscv_vfmul(var, vbr, n);
    vfloat16m1_t vacci = __riscv_vfmul(var, vbi, n);
    vaccr = __riscv_vfnmsac(vaccr, vai, vbi, n);
    vacci = __riscv_vfmacc(vacci, vai, vbr, n);
    __riscv_vse16(or, vaccr, n); or += n;
    __riscv_vse16(oi, vacci, n); oi += n;
  } while (batch > 0);
}
