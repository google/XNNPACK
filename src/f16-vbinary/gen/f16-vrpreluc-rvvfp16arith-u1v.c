// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vopc-rvvfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vbinary.h"


void xnn_f16_vrpreluc_ukernel__rvvfp16arith_u1v(
    size_t batch,
    const xnn_float16* input_a,
    const xnn_float16* input_b,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const _Float16 b = *(const _Float16*) input_b;
  uint16_t* o = (uint16_t*) output;

  size_t n = batch >> XNN_LOG2_SIZEOF_FLOAT16;

  if XNN_UNLIKELY(b >= 0.0f) {
    size_t vl = __riscv_vsetvl_e16m1(n);
    vfloat16m1_t vacc = __riscv_vfmv_v_f_f16m1(b, vl);
    do {
      size_t vl = __riscv_vsetvl_e16m1(n);
      n -= vl;
      __riscv_vse16_v_f16m1((void *) o, vacc, vl);
      o += vl;
    } while (n > 0);
    return;
  }

  do {
    size_t vl = __riscv_vsetvl_e16m1(n);
    n -= vl;
    vfloat16m1_t va = __riscv_vle16_v_f16m1((const void *) a, vl);
    a += vl;
    vfloat16m1_t vacc = __riscv_vfmul_vf_f16m1(va, b, vl);
    __riscv_vse16_v_f16m1((void *) o, vacc, vl);
    o += vl;
  } while (n > 0);
}
