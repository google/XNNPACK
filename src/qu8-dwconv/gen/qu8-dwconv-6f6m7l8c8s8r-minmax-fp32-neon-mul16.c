// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/multipass-neon-mul16.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_qu8_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__neon_mul16(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 6);

  const uint8x8_t vkernel_zero_point = vld1_dup_u8(params->fp32_neon.kernel_zero_point);
  const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neon.scale);
  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->fp32_neon.magic_bias);
  const int32x4_t vmagic_bias_less_output_zero_point = vld1q_dup_s32(&params->fp32_neon.magic_bias_less_output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&params->fp32_neon.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&params->fp32_neon.output_max);

  do {
    const void* w = weights;

    // First pass to process 6 inputs.
    {
      int32_t* b = buffer;
      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = input[5];
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      input += 6;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 8; c -= 8) {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

        const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));

        const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));

        const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));

        const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));

        const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));

        const int16x8_t vi5x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));

        vst1q_s32(b, vacc0123); b += 4;
        vst1q_s32(b, vacc4567); b += 4;
      }

      assert(c == 0);
    }

    // Middle pass to process 6 inputs in each iteration.
    for (size_t ks = kernel_size - 6; ks > 7; ks -= 6) {
      int32_t* b = buffer;

      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = input[5];
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      input += 6;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 8; c -= 8) {
        int32x4_t vacc0123 = vld1q_s32(b);
        int32x4_t vacc4567 = vld1q_s32(b + 4);

        const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));

        const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));

        const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));

        const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));

        const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));

        const int16x8_t vi5x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));

        vst1q_s32(b, vacc0123); b += 4;
        vst1q_s32(b, vacc4567); b += 4;
      }

      assert(c == 0);
    }

    // Last pass to process up to 7 inputs.
    {
      const int32_t* b = buffer;

      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = input[5];
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint8_t* i6 = input[6];
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }

      size_t c = channels;
      for (; c >= 8; c -= 8) {
        int32x4_t vacc0123 = vld1q_s32(b); b += 4;
        int32x4_t vacc4567 = vld1q_s32(b); b += 4;

        const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));

        const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));

        const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));

        const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));

        const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));

        const int16x8_t vi5x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));

        const int16x8_t vi6x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

        vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale);

        vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
        vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));

        vacc0123 = vqsubq_s32(vacc0123, vmagic_bias_less_output_zero_point);
        vacc4567 = vqsubq_s32(vacc4567, vmagic_bias_less_output_zero_point);

#if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);


        uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
#else  // !XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));


        uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
#endif  // !XNN_ARCH_ARM64

        vout01234567 = vmax_u8(vout01234567, voutput_min);

        vout01234567 = vmin_u8(vout01234567, voutput_max);

        vst1_u8(output, vout01234567); output += 8;
      }

      if XNN_UNLIKELY(c != 0) {
        {
          int32x4_t vacc0123 = vld1q_s32(b); b += 4;
          int32x4_t vacc4567 = vld1q_s32(b); b += 4;

          const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0)));
          const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));

          const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1)));
          const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));

          const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2)));
          const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));

          const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3)));
          const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));

          const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4)));
          const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));

          const int16x8_t vi5x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5)));
          const int16x8_t vk5x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));

          const int16x8_t vi6x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6)));
          const int16x8_t vk6x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));

          float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
          float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

          vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
          vfpacc4567 = vmulq_f32(vfpacc4567, vscale);

          vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
          vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));

          vacc0123 = vqsubq_s32(vacc0123, vmagic_bias_less_output_zero_point);
          vacc4567 = vqsubq_s32(vacc4567, vmagic_bias_less_output_zero_point);

  #if XNN_ARCH_ARM64
          int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
  #else
          int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
  #endif

          uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
          vout01234567 = vmax_u8(vout01234567, voutput_min);
          vout01234567 = vmin_u8(vout01234567, voutput_max);

          if (c & 4) {
            vst1_lane_u32((void*) output, vreinterpret_u32_u8(vout01234567), 0); output += 4;
            vout01234567 = vext_u8(vout01234567, vout01234567, 4);
          }
          if (c & 2) {
            vst1_lane_u16((void*) output, vreinterpret_u16_u8(vout01234567), 0); output += 2;
            vout01234567 = vext_u8(vout01234567, vout01234567, 2);
          }
          if (c & 1) {
            vst1_lane_u8(output, vout01234567, 0); output += 1;
          }
        }
      }
    }

    input = (const uint8_t**) ((uintptr_t) input + input_stride);
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
