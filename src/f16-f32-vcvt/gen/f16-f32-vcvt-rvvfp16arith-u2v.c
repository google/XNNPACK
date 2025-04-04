// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/rvvfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright (c) 2025 Institute of Software Chinese Academy of Sciences (ISCAS).
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/vcvt.h"


void xnn_f16_f32_vcvt_ukernel__rvvfp16arith_u2v(
   size_t batch,
   const xnn_float16* input,
   float* output,
   const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_HALF;

  const _Float16* i = (const _Float16*) input;
  for (; batch > 0;) {
    const int32_t n = __riscv_vsetvl_e16m2(batch); batch -= n;
    
    vfloat16m2_t x_f16v = __riscv_vle16_v_f16m2(i, n); i += n;

    vfloat32m4_t y_f32v = __riscv_vfwcvt_f_f_v_f32m4(x_f16v, n);

    __riscv_vse32_v_f32m4(output, y_f32v, n); output += n;
  }
}
