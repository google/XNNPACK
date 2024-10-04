// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/rvvfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/vcvt.h"


void xnn_f32_f16_vcvt_ukernel__rvvfp16arith_u4v(
   size_t batch,
   const float* input,
   xnn_float16* output,
   const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  _Float16* o = (_Float16*) output;
  for (; batch > 0;) {
    const int32_t n = __riscv_vsetvl_e32m4(batch); batch -= n;
    
    vfloat32m4_t x_f32v = __riscv_vle32_v_f32m4(input, n); input += n;

    vfloat16m2_t y_f16v = __riscv_vfncvt_f_f_w_f16m2(x_f32v, n);

    __riscv_vse16_v_f16m2(o, y_f16v, n); o += n;
  }
}
