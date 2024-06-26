// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"


void xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const int32x4_t vinit_bias = vld1q_dup_s32(&params->fp32_neon.init_bias);
  const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neon.scale);
  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->fp32_neon.magic_bias);
  const int32x4_t vmagic_bias_less_output_zero_point = vld1q_dup_s32(&params->fp32_neon.magic_bias_less_output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&params->fp32_neon.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&params->fp32_neon.output_max);

  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];
    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }

    size_t c = channels;
    while (c >= 8) {
      const uint8x8_t vi0 = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi1 = vld1_u8(i1); i1 += 8;
      const uint8x8_t vi2 = vld1_u8(i2); i2 += 8;
      const uint8x8_t vi3 = vld1_u8(i3); i3 += 8;
      const uint8x8_t vi4 = vld1_u8(i4); i4 += 8;
      const uint8x8_t vi5 = vld1_u8(i5); i5 += 8;
      const uint8x8_t vi6 = vld1_u8(i6); i6 += 8;
      const uint8x8_t vi7 = vld1_u8(i7); i7 += 8;
      const uint8x8_t vi8 = vld1_u8(i8); i8 += 8;

      const uint16x8_t vsum018 = vaddw_u8(vaddl_u8(vi0, vi1), vi8);
      const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
      const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);
      const uint16x8_t vsum67 = vaddl_u8(vi6, vi7);

      const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);
      const uint16x8_t vsum01678 = vaddq_u16(vsum018, vsum67);
      const uint16x8_t vsum = vaddq_u16(vsum2345, vsum01678);

      int32x4_t vacc_lo = vaddw_s16(vinit_bias, vreinterpret_s16_u16(vget_low_u16(vsum)));
      int32x4_t vacc_hi = vaddw_s16(vinit_bias, vreinterpret_s16_u16(vget_high_u16(vsum)));

      float32x4_t vfpacc_lo = vcvtq_f32_s32(vacc_lo);
      float32x4_t vfpacc_hi = vcvtq_f32_s32(vacc_hi);

      vfpacc_lo = vmulq_f32(vfpacc_lo, vscale);
      vfpacc_hi = vmulq_f32(vfpacc_hi, vscale);

      vacc_lo = vreinterpretq_s32_f32(vaddq_f32(vfpacc_lo, vmagic_bias));
      vacc_hi = vreinterpretq_s32_f32(vaddq_f32(vfpacc_hi, vmagic_bias));

      vacc_lo = vqsubq_s32(vacc_lo, vmagic_bias_less_output_zero_point);
      vacc_hi = vqsubq_s32(vacc_hi, vmagic_bias_less_output_zero_point);

      #if XNN_ARCH_ARM64
        int16x8_t vacc = vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi);
      #else
        int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
      #endif

      uint8x8_t vout = vqmovun_s16(vacc);
      vout = vmax_u8(vout, voutput_min);
      vout = vmin_u8(vout, voutput_max);

      vst1_u8(output, vout); output += 8;

      c -= 8;
    }
    if (c != 0) {
      const uint8x8_t vi0 = vld1_u8(i0);
      const uint8x8_t vi1 = vld1_u8(i1);
      const uint8x8_t vi2 = vld1_u8(i2);
      const uint8x8_t vi3 = vld1_u8(i3);
      const uint8x8_t vi4 = vld1_u8(i4);
      const uint8x8_t vi5 = vld1_u8(i5);
      const uint8x8_t vi6 = vld1_u8(i6);
      const uint8x8_t vi7 = vld1_u8(i7);
      const uint8x8_t vi8 = vld1_u8(i8);

      const uint16x8_t vsum018 = vaddw_u8(vaddl_u8(vi0, vi1), vi8);
      const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
      const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);
      const uint16x8_t vsum67 = vaddl_u8(vi6, vi7);

      const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);
      const uint16x8_t vsum01678 = vaddq_u16(vsum018, vsum67);
      const uint16x8_t vsum = vaddq_u16(vsum2345, vsum01678);

      int32x4_t vacc_lo = vaddw_s16(vinit_bias, vreinterpret_s16_u16(vget_low_u16(vsum)));
      int32x4_t vacc_hi = vaddw_s16(vinit_bias, vreinterpret_s16_u16(vget_high_u16(vsum)));

      float32x4_t vfpacc_lo = vcvtq_f32_s32(vacc_lo);
      float32x4_t vfpacc_hi = vcvtq_f32_s32(vacc_hi);

      vfpacc_lo = vmulq_f32(vfpacc_lo, vscale);
      vfpacc_hi = vmulq_f32(vfpacc_hi, vscale);

      vacc_lo = vreinterpretq_s32_f32(vaddq_f32(vfpacc_lo, vmagic_bias));
      vacc_hi = vreinterpretq_s32_f32(vaddq_f32(vfpacc_hi, vmagic_bias));

      vacc_lo = vqsubq_s32(vacc_lo, vmagic_bias_less_output_zero_point);
      vacc_hi = vqsubq_s32(vacc_hi, vmagic_bias_less_output_zero_point);

      #if XNN_ARCH_ARM64
        int16x8_t vacc = vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi);
      #else
        int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
      #endif

      uint8x8_t vout = vqmovun_s16(vacc);
      vout = vmax_u8(vout, voutput_min);
      vout = vmin_u8(vout, voutput_max);

      if (c & 4) {
        vst1_lane_u32((void*) output, vreinterpret_u32_u8(vout), 0); output += 4;
        vout = vext_u8(vout, vout, 4);
      }
      if (c & 2) {
        vst1_lane_u16((void*) output, vreinterpret_u16_u8(vout), 0); output += 2;
        vout = vext_u8(vout, vout, 2);
      }
      if (c & 1) {
        vst1_lane_u8(output, vout, 0); output += 1;
      }
    }
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
