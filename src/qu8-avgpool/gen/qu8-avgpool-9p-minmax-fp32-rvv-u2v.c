// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-avgpool/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/avgpool.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_qu8_avgpool_minmax_fp32_ukernel_9p__rvv_u2v(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    size_t input_pixel_stride,
    const uint8_t* zero,
    const float* multiplier,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_qu8_avgpool_minmax_params* restrict params)
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const int32_t vinit_bias = params->scalar.init_bias;
  float vscale = params->scalar.scale;
  const float voutput_min_less_zero_point =
      (float) ((int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point);
  const float voutput_max_less_zero_point =
      (float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point);
  const int16_t voutput_zero_point = params->scalar.output_zero_point;

  do {
    if (multiplier != NULL) {
      vscale = *multiplier++;
    }

    uint8_t* o = output;
    size_t c_offset = 0;
    size_t c = channels;
    do {
      const size_t vl = __riscv_vsetvl_e8m2(c);
      const uint8_t** i = input;
      size_t k = kernel_elements;
      vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);
      do {
        vacc = __riscv_vadd_vx_i32m8(vacc, vinit_bias, vl);

        const uint8_t* i0 = 0 < k ? *i++ : zero;
        assert(i0 != NULL);
        if XNN_UNPREDICTABLE(i0 != zero) {
          i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
        }
        i0 += c_offset;
        const uint8_t* i1 = 1 < k ? *i++ : zero;
        assert(i1 != NULL);
        if XNN_UNPREDICTABLE(i1 != zero) {
          i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
        }
        i1 += c_offset;
        const uint8_t* i2 = 2 < k ? *i++ : zero;
        assert(i2 != NULL);
        if XNN_UNPREDICTABLE(i2 != zero) {
          i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
        }
        i2 += c_offset;
        const uint8_t* i3 = 3 < k ? *i++ : zero;
        assert(i3 != NULL);
        if XNN_UNPREDICTABLE(i3 != zero) {
          i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
        }
        i3 += c_offset;
        const uint8_t* i4 = 4 < k ? *i++ : zero;
        assert(i4 != NULL);
        if XNN_UNPREDICTABLE(i4 != zero) {
          i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
        }
        i4 += c_offset;
        const uint8_t* i5 = 5 < k ? *i++ : zero;
        assert(i5 != NULL);
        if XNN_UNPREDICTABLE(i5 != zero) {
          i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
        }
        i5 += c_offset;
        const uint8_t* i6 = 6 < k ? *i++ : zero;
        assert(i6 != NULL);
        if XNN_UNPREDICTABLE(i6 != zero) {
          i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
        }
        i6 += c_offset;
        const uint8_t* i7 = 7 < k ? *i++ : zero;
        assert(i7 != NULL);
        if XNN_UNPREDICTABLE(i7 != zero) {
          i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
        }
        i7 += c_offset;
        const uint8_t* i8 = 8 < k ? *i++ : zero;
        assert(i8 != NULL);
        if XNN_UNPREDICTABLE(i8 != zero) {
          i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
        }
        i8 += c_offset;

        vuint8m2_t vi0 = __riscv_vle8_v_u8m2(i0, vl);
        vuint8m2_t vi1 = __riscv_vle8_v_u8m2(i1, vl);
        vuint16m4_t vsum = __riscv_vwaddu_vv_u16m4(vi0, vi1, vl);
        vuint8m2_t vi2 = __riscv_vle8_v_u8m2(i2, vl);
        vsum = __riscv_vwaddu_wv_u16m4(vsum, vi2, vl);
        vuint8m2_t vi3 = __riscv_vle8_v_u8m2(i3, vl);
        vsum = __riscv_vwaddu_wv_u16m4(vsum, vi3, vl);
        vuint8m2_t vi4 = __riscv_vle8_v_u8m2(i4, vl);
        vsum = __riscv_vwaddu_wv_u16m4(vsum, vi4, vl);
        vuint8m2_t vi5 = __riscv_vle8_v_u8m2(i5, vl);
        vsum = __riscv_vwaddu_wv_u16m4(vsum, vi5, vl);
        vuint8m2_t vi6 = __riscv_vle8_v_u8m2(i6, vl);
        vsum = __riscv_vwaddu_wv_u16m4(vsum, vi6, vl);
        vuint8m2_t vi7 = __riscv_vle8_v_u8m2(i7, vl);
        vsum = __riscv_vwaddu_wv_u16m4(vsum, vi7, vl);
        vuint8m2_t vi8 = __riscv_vle8_v_u8m2(i8, vl);
        vsum = __riscv_vwaddu_wv_u16m4(vsum, vi8, vl);
        vacc = __riscv_vwadd_wv_i32m8(vacc, __riscv_vreinterpret_v_u16m4_i16m4(vsum), vl);

        k = doz(k, 9);
      } while (k != 0);

      vfloat32m8_t vfpacc = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
      vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, vscale, vl);
      vfpacc = __riscv_vfmax_vf_f32m8(vfpacc, voutput_min_less_zero_point, vl);
      vfpacc = __riscv_vfmin_vf_f32m8(vfpacc, voutput_max_less_zero_point, vl);

      vint16m4_t vout16 = __riscv_vfncvt_x_f_w_i16m4(vfpacc, vl);
      vout16 = __riscv_vadd_vx_i16m4(vout16, voutput_zero_point, vl);
      vuint8m2_t vout8 = __riscv_vncvt_x_x_w_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(vout16), vl);
      __riscv_vse8_v_u8m2(o, vout8, vl);

      o += vl;
      c_offset += vl;
      c -= vl;
    } while (c != 0);

    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
