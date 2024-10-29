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


void xnn_f32_qu8_vcvt_ukernel__rvv_u1v(
    size_t batch,
    const float* input,
    uint8_t* output,
    const struct xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  const float scale = params->scalar.scale;
  // TODO: Clamp may not be necessary. RISCV spec doesn't say if vncvt saturates...
  const float output_min_less_zero_point = (float) ((int32_t) 0 - (int32_t) params->scalar.output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) 255 - (int32_t) params->scalar.output_zero_point);
  const int32_t output_zero_point = params->scalar.output_zero_point;

  for (; batch > 0; ) {
    const int32_t n = __riscv_vsetvl_e32m1(batch); batch -= n;

    vfloat32m1_t x_f32v = __riscv_vle32_v_f32m1(input, n); input += n;

    x_f32v = __riscv_vfmul_vf_f32m1(x_f32v, scale, n);
    x_f32v = __riscv_vfmax_vf_f32m1(x_f32v, output_min_less_zero_point, n);
    x_f32v = __riscv_vfmin_vf_f32m1(x_f32v, output_max_less_zero_point, n);

    vint32m1_t y_i32v = __riscv_vfcvt_x_f_v_i32m1(x_f32v, n);
    y_i32v = __riscv_vadd_vx_i32m1(y_i32v, output_zero_point, n);

    __riscv_vse8_v_u8mf4(output, __riscv_vncvt_x_x_w_u8mf4(__riscv_vncvt_x_x_w_u16mf2(__riscv_vreinterpret_v_i32m1_u32m1(y_i32v), n), n), n); output += n;
  }
}
