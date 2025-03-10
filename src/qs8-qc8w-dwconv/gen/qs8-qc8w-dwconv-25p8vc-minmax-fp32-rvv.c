// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Microchip
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>
#include "src/xnnpack/dwconv.h"


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8vc__rvv(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float voutput_min_less_zero_point = (int32_t) params->fp32_scalar.output_min - (int32_t) params->fp32_scalar.output_zero_point;
  const float voutput_max_less_zero_point = (int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point;
  const int16_t voutput_zero_point = params->fp32_scalar.output_zero_point;

  do {
    const int8_t* i[25];
    i[0] = input[0];
    assert(i[0] != NULL);
    if XNN_UNPREDICTABLE(i[0] != zero) {
      i[0] = (const int8_t*) ((uintptr_t) i[0] + input_offset);
    }
    i[1] = input[1];
    assert(i[1] != NULL);
    if XNN_UNPREDICTABLE(i[1] != zero) {
      i[1] = (const int8_t*) ((uintptr_t) i[1] + input_offset);
    }
    i[2] = input[2];
    assert(i[2] != NULL);
    if XNN_UNPREDICTABLE(i[2] != zero) {
      i[2] = (const int8_t*) ((uintptr_t) i[2] + input_offset);
    }
    i[3] = input[3];
    assert(i[3] != NULL);
    if XNN_UNPREDICTABLE(i[3] != zero) {
      i[3] = (const int8_t*) ((uintptr_t) i[3] + input_offset);
    }
    i[4] = input[4];
    assert(i[4] != NULL);
    if XNN_UNPREDICTABLE(i[4] != zero) {
      i[4] = (const int8_t*) ((uintptr_t) i[4] + input_offset);
    }
    i[5] = input[5];
    assert(i[5] != NULL);
    if XNN_UNPREDICTABLE(i[5] != zero) {
      i[5] = (const int8_t*) ((uintptr_t) i[5] + input_offset);
    }
    i[6] = input[6];
    assert(i[6] != NULL);
    if XNN_UNPREDICTABLE(i[6] != zero) {
      i[6] = (const int8_t*) ((uintptr_t) i[6] + input_offset);
    }
    i[7] = input[7];
    assert(i[7] != NULL);
    if XNN_UNPREDICTABLE(i[7] != zero) {
      i[7] = (const int8_t*) ((uintptr_t) i[7] + input_offset);
    }
    i[8] = input[8];
    assert(i[8] != NULL);
    if XNN_UNPREDICTABLE(i[8] != zero) {
      i[8] = (const int8_t*) ((uintptr_t) i[8] + input_offset);
    }
    i[9] = input[9];
    assert(i[9] != NULL);
    if XNN_UNPREDICTABLE(i[9] != zero) {
      i[9] = (const int8_t*) ((uintptr_t) i[9] + input_offset);
    }
    i[10] = input[10];
    assert(i[10] != NULL);
    if XNN_UNPREDICTABLE(i[10] != zero) {
      i[10] = (const int8_t*) ((uintptr_t) i[10] + input_offset);
    }
    i[11] = input[11];
    assert(i[11] != NULL);
    if XNN_UNPREDICTABLE(i[11] != zero) {
      i[11] = (const int8_t*) ((uintptr_t) i[11] + input_offset);
    }
    i[12] = input[12];
    assert(i[12] != NULL);
    if XNN_UNPREDICTABLE(i[12] != zero) {
      i[12] = (const int8_t*) ((uintptr_t) i[12] + input_offset);
    }
    i[13] = input[13];
    assert(i[13] != NULL);
    if XNN_UNPREDICTABLE(i[13] != zero) {
      i[13] = (const int8_t*) ((uintptr_t) i[13] + input_offset);
    }
    i[14] = input[14];
    assert(i[14] != NULL);
    if XNN_UNPREDICTABLE(i[14] != zero) {
      i[14] = (const int8_t*) ((uintptr_t) i[14] + input_offset);
    }
    i[15] = input[15];
    assert(i[15] != NULL);
    if XNN_UNPREDICTABLE(i[15] != zero) {
      i[15] = (const int8_t*) ((uintptr_t) i[15] + input_offset);
    }
    i[16] = input[16];
    assert(i[16] != NULL);
    if XNN_UNPREDICTABLE(i[16] != zero) {
      i[16] = (const int8_t*) ((uintptr_t) i[16] + input_offset);
    }
    i[17] = input[17];
    assert(i[17] != NULL);
    if XNN_UNPREDICTABLE(i[17] != zero) {
      i[17] = (const int8_t*) ((uintptr_t) i[17] + input_offset);
    }
    i[18] = input[18];
    assert(i[18] != NULL);
    if XNN_UNPREDICTABLE(i[18] != zero) {
      i[18] = (const int8_t*) ((uintptr_t) i[18] + input_offset);
    }
    i[19] = input[19];
    assert(i[19] != NULL);
    if XNN_UNPREDICTABLE(i[19] != zero) {
      i[19] = (const int8_t*) ((uintptr_t) i[19] + input_offset);
    }
    i[20] = input[20];
    assert(i[20] != NULL);
    if XNN_UNPREDICTABLE(i[20] != zero) {
      i[20] = (const int8_t*) ((uintptr_t) i[20] + input_offset);
    }
    i[21] = input[21];
    assert(i[21] != NULL);
    if XNN_UNPREDICTABLE(i[21] != zero) {
      i[21] = (const int8_t*) ((uintptr_t) i[21] + input_offset);
    }
    i[22] = input[22];
    assert(i[22] != NULL);
    if XNN_UNPREDICTABLE(i[22] != zero) {
      i[22] = (const int8_t*) ((uintptr_t) i[22] + input_offset);
    }
    i[23] = input[23];
    assert(i[23] != NULL);
    if XNN_UNPREDICTABLE(i[23] != zero) {
      i[23] = (const int8_t*) ((uintptr_t) i[23] + input_offset);
    }
    i[24] = input[24];
    assert(i[24] != NULL);
    if XNN_UNPREDICTABLE(i[24] != zero) {
      i[24] = (const int8_t*) ((uintptr_t) i[24] + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    size_t vl;

    do {
      vl = __riscv_vsetvl_e32m8(c);
      // load bias to vacc
      vint32m8_t vacc = __riscv_vle32_v_i32m8(w, vl);
      w = (const void*) ((uintptr_t) w + vlmax * sizeof(int32_t));

      for (size_t k = 0; k < 25; ++k) {
        vint8m2_t vi = __riscv_vle8_v_i8m2(i[k], vl);
        vint8m2_t vk = __riscv_vle8_v_i8m2((const int8_t*) w, vl);
        w = (const void*) ((uintptr_t) w + vlmax * sizeof(int8_t));

        i[k] += vlmax;
        vint16m4_t vtmp16 =  __riscv_vwmul_vv_i16m4(vi, vk, vl);
        vacc =  __riscv_vwadd_wv_i32m8(vacc, vtmp16, vl);
      }

      vfloat32m8_t vfpacc = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);

      vfloat32m8_t vscale = __riscv_vle32_v_f32m8(w, vl);
      w = (const void*) ((const float*) w + vl);
      vfpacc = __riscv_vfmul_vv_f32m8(vfpacc, vscale, vl);

      vfpacc = __riscv_vfmax_vf_f32m8(vfpacc, voutput_min_less_zero_point, vl);
      vfpacc = __riscv_vfmin_vf_f32m8(vfpacc, voutput_max_less_zero_point, vl);

      vint16m4_t vout16 = __riscv_vfncvt_x(vfpacc, vl);
      vout16 = __riscv_vadd_vx_i16m4(vout16, voutput_zero_point, vl);
      vint8m2_t vout8 = __riscv_vncvt_x_x_w_i8m2(vout16, vl);
      __riscv_vse8_v_i8m2(output, vout8, vl);

      output += vl;
      c -= vl;
    } while(c != 0);
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
 
