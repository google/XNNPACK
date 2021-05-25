// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>


void xnn_qu8_dwconv_minmax_ukernel_up8x9__neon(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const uint8x8_t vkernel_zero_point = vld1_dup_u8((const uint8_t*) &params->neon.kernel_zero_point);
  const int32x4_t vmultiplier = vld1q_dup_s32(&params->neon.multiplier);
  const int32x4_t vright_shift = vld1q_dup_s32(&params->neon.right_shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&params->neon.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&params->neon.output_max);

  do {
    const uint8_t* i0 = input[0];
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }

    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 8; c -= 8) {
      int32x4_t vaccX1_lo = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));
      int32x4_t vaccX1_hi = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));

      const uint8x8_t vk0 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi0 = vld1_u8(i0); i0 += 8;
      const int16x8_t vxk0 = vreinterpretq_s16_u16(vsubl_u8(vk0, vkernel_zero_point));
      const int16x8_t vxi0 = vreinterpretq_s16_u16(vmovl_u8(vi0));
      int32x4_t vaccX0_lo = vmull_s16(vget_low_s16(vxk0), vget_low_s16(vxi0));
      int32x4_t vaccX0_hi = vmull_s16(vget_high_s16(vxk0), vget_high_s16(vxi0));

      const uint8x8_t vk1 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi1 = vld1_u8(i1); i1 += 8;
      const int16x8_t vxk1 = vreinterpretq_s16_u16(vsubl_u8(vk1, vkernel_zero_point));
      const int16x8_t vxi1 = vreinterpretq_s16_u16(vmovl_u8(vi1));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk1), vget_low_s16(vxi1));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk1), vget_high_s16(vxi1));

      const uint8x8_t vk2 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi2 = vld1_u8(i2); i2 += 8;
      const int16x8_t vxk2 = vreinterpretq_s16_u16(vsubl_u8(vk2, vkernel_zero_point));
      const int16x8_t vxi2 = vreinterpretq_s16_u16(vmovl_u8(vi2));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk2), vget_low_s16(vxi2));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk2), vget_high_s16(vxi2));

      const uint8x8_t vk3 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi3 = vld1_u8(i3); i3 += 8;
      const int16x8_t vxk3 = vreinterpretq_s16_u16(vsubl_u8(vk3, vkernel_zero_point));
      const int16x8_t vxi3 = vreinterpretq_s16_u16(vmovl_u8(vi3));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk3), vget_low_s16(vxi3));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk3), vget_high_s16(vxi3));

      const uint8x8_t vk4 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi4 = vld1_u8(i4); i4 += 8;
      const int16x8_t vxk4 = vreinterpretq_s16_u16(vsubl_u8(vk4, vkernel_zero_point));
      const int16x8_t vxi4 = vreinterpretq_s16_u16(vmovl_u8(vi4));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk4), vget_low_s16(vxi4));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk4), vget_high_s16(vxi4));

      const uint8x8_t vk5 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi5 = vld1_u8(i5); i5 += 8;
      const int16x8_t vxk5 = vreinterpretq_s16_u16(vsubl_u8(vk5, vkernel_zero_point));
      const int16x8_t vxi5 = vreinterpretq_s16_u16(vmovl_u8(vi5));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk5), vget_low_s16(vxi5));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk5), vget_high_s16(vxi5));

      const uint8x8_t vk6 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi6 = vld1_u8(i6); i6 += 8;
      const int16x8_t vxk6 = vreinterpretq_s16_u16(vsubl_u8(vk6, vkernel_zero_point));
      const int16x8_t vxi6 = vreinterpretq_s16_u16(vmovl_u8(vi6));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk6), vget_low_s16(vxi6));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk6), vget_high_s16(vxi6));

      const uint8x8_t vk7 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi7 = vld1_u8(i7); i7 += 8;
      const int16x8_t vxk7 = vreinterpretq_s16_u16(vsubl_u8(vk7, vkernel_zero_point));
      const int16x8_t vxi7 = vreinterpretq_s16_u16(vmovl_u8(vi7));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk7), vget_low_s16(vxi7));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk7), vget_high_s16(vxi7));

      const uint8x8_t vk8 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi8 = vld1_u8(i8); i8 += 8;
      const int16x8_t vxk8 = vreinterpretq_s16_u16(vsubl_u8(vk8, vkernel_zero_point));
      const int16x8_t vxi8 = vreinterpretq_s16_u16(vmovl_u8(vi8));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk8), vget_low_s16(vxi8));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk8), vget_high_s16(vxi8));

      int32x4_t vacc_lo = vaddq_s32(vaccX0_lo, vaccX1_lo);
      int32x4_t vacc_hi = vaddq_s32(vaccX0_hi, vaccX1_hi);

      vacc_lo = vqrdmulhq_s32(vacc_lo, vmultiplier);
      vacc_hi = vqrdmulhq_s32(vacc_hi, vmultiplier);

      const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
      vacc_lo = vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
      vacc_hi = vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

      vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
      vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

#if XNN_ARCH_ARM64
      const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
#else
      const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), voutput_zero_point);
#endif
      uint8x8_t vout = vqmovun_s16(vacc);
      vout = vmax_u8(vout, voutput_min);
      vout = vmin_u8(vout, voutput_max);

      vst1_u8(output, vout); output += 8;
    }
    if (c != 0) {
      int32x4_t vaccX1_lo = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));
      int32x4_t vaccX1_hi = vld1q_s32(w); w = (void*) ((uintptr_t) w + sizeof(int32x4_t));

      const uint8x8_t vk0 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi0 = vld1_u8(i0);
      const int16x8_t vxk0 = vreinterpretq_s16_u16(vsubl_u8(vk0, vkernel_zero_point));
      const int16x8_t vxi0 = vreinterpretq_s16_u16(vmovl_u8(vi0));
      int32x4_t vaccX0_lo = vmull_s16(vget_low_s16(vxk0), vget_low_s16(vxi0));
      int32x4_t vaccX0_hi = vmull_s16(vget_high_s16(vxk0), vget_high_s16(vxi0));

      const uint8x8_t vk1 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi1 = vld1_u8(i1);
      const int16x8_t vxk1 = vreinterpretq_s16_u16(vsubl_u8(vk1, vkernel_zero_point));
      const int16x8_t vxi1 = vreinterpretq_s16_u16(vmovl_u8(vi1));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk1), vget_low_s16(vxi1));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk1), vget_high_s16(vxi1));

      const uint8x8_t vk2 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi2 = vld1_u8(i2);
      const int16x8_t vxk2 = vreinterpretq_s16_u16(vsubl_u8(vk2, vkernel_zero_point));
      const int16x8_t vxi2 = vreinterpretq_s16_u16(vmovl_u8(vi2));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk2), vget_low_s16(vxi2));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk2), vget_high_s16(vxi2));

      const uint8x8_t vk3 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi3 = vld1_u8(i3);
      const int16x8_t vxk3 = vreinterpretq_s16_u16(vsubl_u8(vk3, vkernel_zero_point));
      const int16x8_t vxi3 = vreinterpretq_s16_u16(vmovl_u8(vi3));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk3), vget_low_s16(vxi3));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk3), vget_high_s16(vxi3));

      const uint8x8_t vk4 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi4 = vld1_u8(i4);
      const int16x8_t vxk4 = vreinterpretq_s16_u16(vsubl_u8(vk4, vkernel_zero_point));
      const int16x8_t vxi4 = vreinterpretq_s16_u16(vmovl_u8(vi4));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk4), vget_low_s16(vxi4));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk4), vget_high_s16(vxi4));

      const uint8x8_t vk5 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi5 = vld1_u8(i5);
      const int16x8_t vxk5 = vreinterpretq_s16_u16(vsubl_u8(vk5, vkernel_zero_point));
      const int16x8_t vxi5 = vreinterpretq_s16_u16(vmovl_u8(vi5));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk5), vget_low_s16(vxi5));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk5), vget_high_s16(vxi5));

      const uint8x8_t vk6 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi6 = vld1_u8(i6);
      const int16x8_t vxk6 = vreinterpretq_s16_u16(vsubl_u8(vk6, vkernel_zero_point));
      const int16x8_t vxi6 = vreinterpretq_s16_u16(vmovl_u8(vi6));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk6), vget_low_s16(vxi6));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk6), vget_high_s16(vxi6));

      const uint8x8_t vk7 = vld1_u8(w); w = (void*) ((uintptr_t) w + sizeof(uint8x8_t));
      const uint8x8_t vi7 = vld1_u8(i7);
      const int16x8_t vxk7 = vreinterpretq_s16_u16(vsubl_u8(vk7, vkernel_zero_point));
      const int16x8_t vxi7 = vreinterpretq_s16_u16(vmovl_u8(vi7));
      vaccX1_lo = vmlal_s16(vaccX1_lo, vget_low_s16(vxk7), vget_low_s16(vxi7));
      vaccX1_hi = vmlal_s16(vaccX1_hi, vget_high_s16(vxk7), vget_high_s16(vxi7));

      const uint8x8_t vk8 = vld1_u8(w);
      const uint8x8_t vi8 = vld1_u8(i8);
      const int16x8_t vxk8 = vreinterpretq_s16_u16(vsubl_u8(vk8, vkernel_zero_point));
      const int16x8_t vxi8 = vreinterpretq_s16_u16(vmovl_u8(vi8));
      vaccX0_lo = vmlal_s16(vaccX0_lo, vget_low_s16(vxk8), vget_low_s16(vxi8));
      vaccX0_hi = vmlal_s16(vaccX0_hi, vget_high_s16(vxk8), vget_high_s16(vxi8));

      int32x4_t vacc_lo = vaddq_s32(vaccX0_lo, vaccX1_lo);
      int32x4_t vacc_hi = vaddq_s32(vaccX0_hi, vaccX1_hi);

      vacc_lo = vqrdmulhq_s32(vacc_lo, vmultiplier);
      vacc_hi = vqrdmulhq_s32(vacc_hi, vmultiplier);

      const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
      vacc_lo = vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
      vacc_hi = vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

      vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
      vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

#if XNN_ARCH_ARM64
      const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
#else
      const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), voutput_zero_point);
#endif
      uint8x8_t vout = vqmovun_s16(vacc);
      vout = vmax_u8(vout, voutput_min);
      vout = vmin_u8(vout, voutput_max);

      if (c & 4) {
        vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0); output += 4;
        vout = vext_u8(vout, vout, 4);
      }
      if (c & 2) {
        vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0); output += 2;
        vout = vext_u8(vout, vout, 2);
      }
      if (c & 1) {
        vst1_lane_u8(__builtin_assume_aligned(output, 1), vout, 0); output++;
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
