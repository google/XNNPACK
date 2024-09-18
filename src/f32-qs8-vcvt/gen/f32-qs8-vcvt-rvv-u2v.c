// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/math.h"
#include "xnnpack/vcvt.h"


void xnn_f32_qs8_vcvt_ukernel__rvv_u2v(
    size_t batch,
    const float* input,
    int8_t* output,
    const struct xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  const float scale = params->scalar.scale;
  const float output_min_less_zero_point = (float) ((int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point);
  const int32_t output_zero_point = params->scalar.output_zero_point;

  for (; batch > 0; ) {
    const int32_t n = __riscv_vsetvl_e32m2(batch); batch -= n;

    vfloat32m2_t x_f32v = __riscv_vle32_v_f32m2(input, n); input += n;

    x_f32v = __riscv_vfmul_vf_f32m2(x_f32v, scale, n);
    x_f32v = __riscv_vfmax_vf_f32m2(x_f32v, output_min_less_zero_point, n);
    x_f32v = __riscv_vfmin_vf_f32m2(x_f32v, output_max_less_zero_point, n);

    vint32m2_t y_i32v = __riscv_vfcvt_x_f_v_i32m2(x_f32v, n);
    y_i32v = __riscv_vadd_vx_i32m2(y_i32v, output_zero_point, n);

    __riscv_vse8_v_i8mf2(output, __riscv_vncvt_x_x_w_i8mf2(__riscv_vncvt_x_x_w_i16m1(y_i32v, n), n), n); output += n;
  }
}
