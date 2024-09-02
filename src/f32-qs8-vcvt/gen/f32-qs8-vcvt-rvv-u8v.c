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


void xnn_f32_qs8_vcvt_ukernel__rvv_u8v(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  const float scale = params->scalar.scale;
  const vfloat32m8_t magic_bias_f32v = __riscv_vfmv_v_f_f32m8(12582912.0f, __riscv_vsetvl_e32m8(batch));
  const float output_min_less_zero_point = (float) ((int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point);
  const int32_t magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  const int32_t magic_max = (int32_t) float_as_uint32(12582912.0f + output_max_less_zero_point);
  const int32_t magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;

  for (; batch > 0; ) {
    const int32_t n = __riscv_vsetvl_e32m8(batch); batch -= n;

    vfloat32m8_t x_f32v = __riscv_vle32_v_f32m8(input, n); input += n;

    x_f32v = __riscv_vfmadd_vf_f32m8(x_f32v, scale, magic_bias_f32v, n);

    vint32m8_t y_i32v = __riscv_vreinterpret_v_f32m8_i32m8(x_f32v);
    y_i32v = __riscv_vmax_vx_i32m8(y_i32v, magic_min, n);
    y_i32v = __riscv_vmin_vx_i32m8(y_i32v, magic_max, n);
    y_i32v = __riscv_vssub_vx_i32m8(y_i32v, magic_bias_less_zero_point, n);

    __riscv_vse8_v_i8m2(output, __riscv_vncvt_x_x_w_i8m2(__riscv_vncvt_x_x_w_i16m4(y_i32v, n), n), n); output += n;
  }
}
