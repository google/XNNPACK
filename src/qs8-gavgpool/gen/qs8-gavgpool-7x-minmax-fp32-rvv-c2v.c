// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/unipass-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/gavgpool.h"


void xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__rvv_c2v(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const int32_t init_bias = params->fp32_scalar_fmagic.init_bias;
  const float scale = params->fp32_scalar_fmagic.scale;
  const float output_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float output_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float magic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t magic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
  vint32m8_t init_bias_i32v = __riscv_vmv_v_x_i32m8(init_bias, __riscv_vsetvl_e8m2(channels));

  do {
    int32_t n = __riscv_vsetvl_e8m2(channels); channels -= n;

    vint8m2_t i0_i8v = __riscv_vle8_v_i8m2(i0, n); i0 += n;
    vint8m2_t i1_i8v = __riscv_vle8_v_i8m2(i1, n); i1 += n;
    vint16m4_t acc_i16v = __riscv_vwadd_vv_i16m4(i0_i8v, i1_i8v, n);
    i0_i8v = __riscv_vle8_v_i8m2(i2, n); i2 += n;
    acc_i16v = __riscv_vwadd_wv_i16m4(acc_i16v, i0_i8v, n);
    i0_i8v = __riscv_vle8_v_i8m2(i3, n); i3 += n;
    acc_i16v = __riscv_vwadd_wv_i16m4(acc_i16v, i0_i8v, n);
    i0_i8v = __riscv_vle8_v_i8m2(i4, n); i4 += n;
    acc_i16v = __riscv_vwadd_wv_i16m4(acc_i16v, i0_i8v, n);
    i0_i8v = __riscv_vle8_v_i8m2(i5, n); i5 += n;
    acc_i16v = __riscv_vwadd_wv_i16m4(acc_i16v, i0_i8v, n);
    i0_i8v = __riscv_vle8_v_i8m2(i6, n); i6 += n;
    acc_i16v = __riscv_vwadd_wv_i16m4(acc_i16v, i0_i8v, n);

    vint32m8_t acc_i32v = __riscv_vwcvt_x_x_v_i32m8(acc_i16v, n);
    acc_i32v = __riscv_vadd_vv_i32m8(acc_i32v, init_bias_i32v, n);
    vfloat32m8_t acc_f32v = __riscv_vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = __riscv_vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = __riscv_vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vint32m8_t out_i32v = __riscv_vfcvt_x_f_v_i32m8(acc_f32v, n);
    vint16m4_t out_i16v = __riscv_vncvt_x_x_w_i16m4(out_i32v, n);
    out_i16v = __riscv_vsub_vx_i16m4(out_i16v, magic_bias_less_output_zero_point, n);
    vint8m2_t out_i8v = __riscv_vncvt_x_x_w_i8m2(out_i16v, n);
    __riscv_vse8_v_i8m2(output, out_i8v, n); output += n;
  } while (channels != 0);
}
