// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/multipass-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/math.h"


void xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__rvv_c1v(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows > 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }
  const size_t input_increment = 7 * input_stride - channels;

  const int32_t init_bias = params->fp32_scalar_fmagic.init_bias;
  int32_t* b = buffer;
  int32_t c = (int32_t) channels;
  do {
    int32_t n = __riscv_vsetvl_e8m1(c); c -= n;
    vuint8m1_t i0_u8v = __riscv_vle8_v_u8m1(i0, n); i0 += n;
    vuint8m1_t i1_u8v = __riscv_vle8_v_u8m1(i1, n); i1 += n;
    vuint16m2_t acc_u16v = __riscv_vwaddu_vv_u16m2(i0_u8v, i1_u8v, n);
    i0_u8v = __riscv_vle8_v_u8m1(i2, n); i2 += n;
    acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
    i0_u8v = __riscv_vle8_v_u8m1(i3, n); i3 += n;
    acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
    i0_u8v = __riscv_vle8_v_u8m1(i4, n); i4 += n;
    acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
    i0_u8v = __riscv_vle8_v_u8m1(i5, n); i5 += n;
    acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
    i0_u8v = __riscv_vle8_v_u8m1(i6, n); i6 += n;
    acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
    vint16m2_t acc_i16v = __riscv_vreinterpret_v_u16m2_i16m2(acc_u16v);

    vint32m4_t acc_i32v = __riscv_vwadd_vx_i32m4(acc_i16v, init_bias, n);
    __riscv_vse32_v_i32m4(b, acc_i32v, n); b += n;
  } while (c > 0);

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    int32_t c = (int32_t) channels;
    do {
      int32_t n = __riscv_vsetvl_e8m1(c); c -= n;
      vuint8m1_t i0_u8v = __riscv_vle8_v_u8m1(i0, n); i0 += n;
      vuint8m1_t i1_u8v = __riscv_vle8_v_u8m1(i1, n); i1 += n;
      vuint16m2_t acc_u16v = __riscv_vwaddu_vv_u16m2(i0_u8v, i1_u8v, n);
      i0_u8v = __riscv_vle8_v_u8m1(i2, n); i2 += n;
      acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
      i0_u8v = __riscv_vle8_v_u8m1(i3, n); i3 += n;
      acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
      i0_u8v = __riscv_vle8_v_u8m1(i4, n); i4 += n;
      acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
      i0_u8v = __riscv_vle8_v_u8m1(i5, n); i5 += n;
      acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
      i0_u8v = __riscv_vle8_v_u8m1(i6, n); i6 += n;
      acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
      vint16m2_t acc_i16v = __riscv_vreinterpret_v_u16m2_i16m2(acc_u16v);

      vint32m4_t acc_i32v = __riscv_vle32_v_i32m4(b, n);
      acc_i32v = __riscv_vwadd_wv_i32m4(acc_i32v, acc_i16v, n);
      __riscv_vse32_v_i32m4(b, acc_i32v, n); b += n;
    } while (c > 0);
  }

  i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const float scale = params->fp32_scalar_fmagic.scale;
  const float output_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float output_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float magic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t magic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;

  do {
    int32_t n = __riscv_vsetvl_e8m1(channels); channels -= n;
    vuint8m1_t i0_u8v = __riscv_vle8_v_u8m1(i0, n); i0 += n;
    vuint8m1_t i1_u8v = __riscv_vle8_v_u8m1(i1, n); i1 += n;
    vuint16m2_t acc_u16v = __riscv_vwaddu_vv_u16m2(i0_u8v, i1_u8v, n);
    i0_u8v = __riscv_vle8_v_u8m1(i2, n); i2 += n;
    acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
    i0_u8v = __riscv_vle8_v_u8m1(i3, n); i3 += n;
    acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
    i0_u8v = __riscv_vle8_v_u8m1(i4, n); i4 += n;
    acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
    i0_u8v = __riscv_vle8_v_u8m1(i5, n); i5 += n;
    acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
    i0_u8v = __riscv_vle8_v_u8m1(i6, n); i6 += n;
    acc_u16v = __riscv_vwaddu_wv_u16m2(acc_u16v, i0_u8v, n);
    vint16m2_t acc_i16v = __riscv_vreinterpret_v_u16m2_i16m2(acc_u16v);

    vint32m4_t acc_i32v = __riscv_vle32_v_i32m4(buffer, n); buffer += n;
    acc_i32v = __riscv_vwadd_wv_i32m4(acc_i32v, acc_i16v, n);
    vfloat32m4_t acc_f32v = __riscv_vfcvt_f_x_v_f32m4(acc_i32v, n);
    acc_f32v = __riscv_vfmul_vf_f32m4(acc_f32v, scale, n);
    acc_f32v = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = __riscv_vfadd_vf_f32m4(acc_f32v, magic_bias, n);

    vuint32m4_t out_u32v = __riscv_vfcvt_xu_f_v_u32m4(acc_f32v, n);
    vuint16m2_t out_u16v = __riscv_vncvt_x_x_w_u16m2(out_u32v, n);
    out_u16v = __riscv_vsub_vx_u16m2(out_u16v, magic_bias_less_output_zero_point, n);
    vuint8m1_t out_u8v = __riscv_vncvt_x_x_w_u8m1(out_u16v, n);
    __riscv_vse8_v_u8m1(output, out_u8v, n); output += n;
  } while (channels != 0);
}
