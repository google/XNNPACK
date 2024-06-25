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


void xnn_qu8_dwconv_minmax_fp32_ukernel_8f8m9l32c8s8r__neon_mul16(
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
  assert(kernel_size > 8);

  const uint8x8_t vkernel_zero_point = vld1_dup_u8(params->fp32_neon.kernel_zero_point);
  const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neon.scale);
  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->fp32_neon.magic_bias);
  const int32x4_t vmagic_bias_less_output_zero_point = vld1q_dup_s32(&params->fp32_neon.magic_bias_less_output_zero_point);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->fp32_neon.output_min);
  const uint8x16_t voutput_max = vld1q_dup_u8(&params->fp32_neon.output_max);

  do {
    const void* w = weights;

    // First pass to process 8 inputs.
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
      const uint8_t* i6 = input[6];
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = input[7];
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }
      input += 8;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 32; c -= 32) {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vaccCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vaccGHIJ = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vaccKLMN = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vaccOPQR = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vaccSTUV = vld1q_s32(w); w = (const int32_t*) w + 4;

        const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi0x89ABCDEF), vget_low_s16(vk0x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi0x89ABCDEF), vget_high_s16(vk0x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi0xGHIJKLMN), vget_low_s16(vk0xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi0xGHIJKLMN), vget_high_s16(vk0xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi0xOPQRSTUV), vget_low_s16(vk0xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi0xOPQRSTUV), vget_high_s16(vk0xOPQRSTUV));

        const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi1x89ABCDEF), vget_low_s16(vk1x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi1x89ABCDEF), vget_high_s16(vk1x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi1xGHIJKLMN), vget_low_s16(vk1xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi1xGHIJKLMN), vget_high_s16(vk1xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi1xOPQRSTUV), vget_low_s16(vk1xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi1xOPQRSTUV), vget_high_s16(vk1xOPQRSTUV));

        const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi2x89ABCDEF), vget_low_s16(vk2x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi2x89ABCDEF), vget_high_s16(vk2x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi2xGHIJKLMN), vget_low_s16(vk2xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi2xGHIJKLMN), vget_high_s16(vk2xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi2xOPQRSTUV), vget_low_s16(vk2xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi2xOPQRSTUV), vget_high_s16(vk2xOPQRSTUV));

        const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi3x89ABCDEF), vget_low_s16(vk3x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi3x89ABCDEF), vget_high_s16(vk3x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi3xGHIJKLMN), vget_low_s16(vk3xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi3xGHIJKLMN), vget_high_s16(vk3xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi3xOPQRSTUV), vget_low_s16(vk3xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi3xOPQRSTUV), vget_high_s16(vk3xOPQRSTUV));

        const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi4x89ABCDEF), vget_low_s16(vk4x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi4x89ABCDEF), vget_high_s16(vk4x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi4xGHIJKLMN), vget_low_s16(vk4xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi4xGHIJKLMN), vget_high_s16(vk4xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi4xOPQRSTUV), vget_low_s16(vk4xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi4xOPQRSTUV), vget_high_s16(vk4xOPQRSTUV));

        const int16x8_t vi5x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi5x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi5xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi5xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi5x89ABCDEF), vget_low_s16(vk5x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi5x89ABCDEF), vget_high_s16(vk5x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi5xGHIJKLMN), vget_low_s16(vk5xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi5xGHIJKLMN), vget_high_s16(vk5xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi5xOPQRSTUV), vget_low_s16(vk5xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi5xOPQRSTUV), vget_high_s16(vk5xOPQRSTUV));

        const int16x8_t vi6x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi6x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi6xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi6xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi6x89ABCDEF), vget_low_s16(vk6x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi6x89ABCDEF), vget_high_s16(vk6x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi6xGHIJKLMN), vget_low_s16(vk6xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi6xGHIJKLMN), vget_high_s16(vk6xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi6xOPQRSTUV), vget_low_s16(vk6xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi6xOPQRSTUV), vget_high_s16(vk6xOPQRSTUV));

        const int16x8_t vi7x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi7x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi7xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi7xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi7x89ABCDEF), vget_low_s16(vk7x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi7x89ABCDEF), vget_high_s16(vk7x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi7xGHIJKLMN), vget_low_s16(vk7xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi7xGHIJKLMN), vget_high_s16(vk7xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi7xOPQRSTUV), vget_low_s16(vk7xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi7xOPQRSTUV), vget_high_s16(vk7xOPQRSTUV));

        vst1q_s32(b, vacc0123); b += 4;
        vst1q_s32(b, vacc4567); b += 4;
        vst1q_s32(b, vacc89AB); b += 4;
        vst1q_s32(b, vaccCDEF); b += 4;
        vst1q_s32(b, vaccGHIJ); b += 4;
        vst1q_s32(b, vaccKLMN); b += 4;
        vst1q_s32(b, vaccOPQR); b += 4;
        vst1q_s32(b, vaccSTUV); b += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
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

          const int16x8_t vi6x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
          const int16x8_t vk6x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));

          const int16x8_t vi7x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
          const int16x8_t vk7x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));

          vst1q_s32(b, vacc0123); b += 4;
          vst1q_s32(b, vacc4567); b += 4;
          c -= 8;
        } while (c != 0);
      }
    }

    // Middle pass to process 8 inputs in each iteration.
    for (size_t ks = kernel_size - 8; ks > 9; ks -= 8) {
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
      const uint8_t* i6 = input[6];
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = input[7];
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }
      input += 8;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 32; c -= 32) {
        int32x4_t vacc0123 = vld1q_s32(b);
        int32x4_t vacc4567 = vld1q_s32(b + 4);
        int32x4_t vacc89AB = vld1q_s32(b + 8);
        int32x4_t vaccCDEF = vld1q_s32(b + 12);
        int32x4_t vaccGHIJ = vld1q_s32(b + 16);
        int32x4_t vaccKLMN = vld1q_s32(b + 20);
        int32x4_t vaccOPQR = vld1q_s32(b + 24);
        int32x4_t vaccSTUV = vld1q_s32(b + 28);

        const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi0x89ABCDEF), vget_low_s16(vk0x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi0x89ABCDEF), vget_high_s16(vk0x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi0xGHIJKLMN), vget_low_s16(vk0xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi0xGHIJKLMN), vget_high_s16(vk0xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi0xOPQRSTUV), vget_low_s16(vk0xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi0xOPQRSTUV), vget_high_s16(vk0xOPQRSTUV));

        const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi1x89ABCDEF), vget_low_s16(vk1x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi1x89ABCDEF), vget_high_s16(vk1x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi1xGHIJKLMN), vget_low_s16(vk1xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi1xGHIJKLMN), vget_high_s16(vk1xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi1xOPQRSTUV), vget_low_s16(vk1xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi1xOPQRSTUV), vget_high_s16(vk1xOPQRSTUV));

        const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi2x89ABCDEF), vget_low_s16(vk2x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi2x89ABCDEF), vget_high_s16(vk2x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi2xGHIJKLMN), vget_low_s16(vk2xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi2xGHIJKLMN), vget_high_s16(vk2xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi2xOPQRSTUV), vget_low_s16(vk2xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi2xOPQRSTUV), vget_high_s16(vk2xOPQRSTUV));

        const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi3x89ABCDEF), vget_low_s16(vk3x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi3x89ABCDEF), vget_high_s16(vk3x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi3xGHIJKLMN), vget_low_s16(vk3xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi3xGHIJKLMN), vget_high_s16(vk3xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi3xOPQRSTUV), vget_low_s16(vk3xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi3xOPQRSTUV), vget_high_s16(vk3xOPQRSTUV));

        const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi4x89ABCDEF), vget_low_s16(vk4x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi4x89ABCDEF), vget_high_s16(vk4x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi4xGHIJKLMN), vget_low_s16(vk4xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi4xGHIJKLMN), vget_high_s16(vk4xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi4xOPQRSTUV), vget_low_s16(vk4xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi4xOPQRSTUV), vget_high_s16(vk4xOPQRSTUV));

        const int16x8_t vi5x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi5x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi5xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi5xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi5x89ABCDEF), vget_low_s16(vk5x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi5x89ABCDEF), vget_high_s16(vk5x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi5xGHIJKLMN), vget_low_s16(vk5xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi5xGHIJKLMN), vget_high_s16(vk5xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi5xOPQRSTUV), vget_low_s16(vk5xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi5xOPQRSTUV), vget_high_s16(vk5xOPQRSTUV));

        const int16x8_t vi6x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi6x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi6xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi6xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi6x89ABCDEF), vget_low_s16(vk6x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi6x89ABCDEF), vget_high_s16(vk6x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi6xGHIJKLMN), vget_low_s16(vk6xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi6xGHIJKLMN), vget_high_s16(vk6xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi6xOPQRSTUV), vget_low_s16(vk6xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi6xOPQRSTUV), vget_high_s16(vk6xOPQRSTUV));

        const int16x8_t vi7x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi7x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi7xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi7xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi7x89ABCDEF), vget_low_s16(vk7x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi7x89ABCDEF), vget_high_s16(vk7x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi7xGHIJKLMN), vget_low_s16(vk7xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi7xGHIJKLMN), vget_high_s16(vk7xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi7xOPQRSTUV), vget_low_s16(vk7xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi7xOPQRSTUV), vget_high_s16(vk7xOPQRSTUV));

        vst1q_s32(b, vacc0123); b += 4;
        vst1q_s32(b, vacc4567); b += 4;
        vst1q_s32(b, vacc89AB); b += 4;
        vst1q_s32(b, vaccCDEF); b += 4;
        vst1q_s32(b, vaccGHIJ); b += 4;
        vst1q_s32(b, vaccKLMN); b += 4;
        vst1q_s32(b, vaccOPQR); b += 4;
        vst1q_s32(b, vaccSTUV); b += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
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

          const int16x8_t vi6x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
          const int16x8_t vk6x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));

          const int16x8_t vi7x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
          const int16x8_t vk7x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));

          vst1q_s32(b, vacc0123); b += 4;
          vst1q_s32(b, vacc4567); b += 4;
          c -= 8;
        } while (c != 0);
      }
    }

    // Last pass to process up to 9 inputs.
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
      const uint8_t* i7 = input[7];
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint8_t* i8 = input[8];
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
      }

      size_t c = channels;
      for (; c >= 32; c -= 32) {
        int32x4_t vacc0123 = vld1q_s32(b); b += 4;
        int32x4_t vacc4567 = vld1q_s32(b); b += 4;
        int32x4_t vacc89AB = vld1q_s32(b); b += 4;
        int32x4_t vaccCDEF = vld1q_s32(b); b += 4;
        int32x4_t vaccGHIJ = vld1q_s32(b); b += 4;
        int32x4_t vaccKLMN = vld1q_s32(b); b += 4;
        int32x4_t vaccOPQR = vld1q_s32(b); b += 4;
        int32x4_t vaccSTUV = vld1q_s32(b); b += 4;

        const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi0x89ABCDEF), vget_low_s16(vk0x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi0x89ABCDEF), vget_high_s16(vk0x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi0xGHIJKLMN), vget_low_s16(vk0xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi0xGHIJKLMN), vget_high_s16(vk0xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi0xOPQRSTUV), vget_low_s16(vk0xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi0xOPQRSTUV), vget_high_s16(vk0xOPQRSTUV));

        const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi1x89ABCDEF), vget_low_s16(vk1x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi1x89ABCDEF), vget_high_s16(vk1x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi1xGHIJKLMN), vget_low_s16(vk1xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi1xGHIJKLMN), vget_high_s16(vk1xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi1xOPQRSTUV), vget_low_s16(vk1xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi1xOPQRSTUV), vget_high_s16(vk1xOPQRSTUV));

        const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi2x89ABCDEF), vget_low_s16(vk2x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi2x89ABCDEF), vget_high_s16(vk2x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi2xGHIJKLMN), vget_low_s16(vk2xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi2xGHIJKLMN), vget_high_s16(vk2xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi2xOPQRSTUV), vget_low_s16(vk2xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi2xOPQRSTUV), vget_high_s16(vk2xOPQRSTUV));

        const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi3x89ABCDEF), vget_low_s16(vk3x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi3x89ABCDEF), vget_high_s16(vk3x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi3xGHIJKLMN), vget_low_s16(vk3xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi3xGHIJKLMN), vget_high_s16(vk3xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi3xOPQRSTUV), vget_low_s16(vk3xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi3xOPQRSTUV), vget_high_s16(vk3xOPQRSTUV));

        const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi4x89ABCDEF), vget_low_s16(vk4x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi4x89ABCDEF), vget_high_s16(vk4x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi4xGHIJKLMN), vget_low_s16(vk4xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi4xGHIJKLMN), vget_high_s16(vk4xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi4xOPQRSTUV), vget_low_s16(vk4xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi4xOPQRSTUV), vget_high_s16(vk4xOPQRSTUV));

        const int16x8_t vi5x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi5x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi5xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi5xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
        const int16x8_t vk5xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi5x89ABCDEF), vget_low_s16(vk5x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi5x89ABCDEF), vget_high_s16(vk5x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi5xGHIJKLMN), vget_low_s16(vk5xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi5xGHIJKLMN), vget_high_s16(vk5xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi5xOPQRSTUV), vget_low_s16(vk5xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi5xOPQRSTUV), vget_high_s16(vk5xOPQRSTUV));

        const int16x8_t vi6x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi6x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi6xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi6xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
        const int16x8_t vk6xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi6x89ABCDEF), vget_low_s16(vk6x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi6x89ABCDEF), vget_high_s16(vk6x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi6xGHIJKLMN), vget_low_s16(vk6xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi6xGHIJKLMN), vget_high_s16(vk6xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi6xOPQRSTUV), vget_low_s16(vk6xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi6xOPQRSTUV), vget_high_s16(vk6xOPQRSTUV));

        const int16x8_t vi7x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi7x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi7xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi7xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
        const int16x8_t vk7xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi7x89ABCDEF), vget_low_s16(vk7x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi7x89ABCDEF), vget_high_s16(vk7x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi7xGHIJKLMN), vget_low_s16(vk7xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi7xGHIJKLMN), vget_high_s16(vk7xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi7xOPQRSTUV), vget_low_s16(vk7xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi7xOPQRSTUV), vget_high_s16(vk7xOPQRSTUV));

        const int16x8_t vi8x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i8))); i8 += 8;
        const int16x8_t vk8x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi8x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i8))); i8 += 8;
        const int16x8_t vk8x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi8xGHIJKLMN = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i8))); i8 += 8;
        const int16x8_t vk8xGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi8xOPQRSTUV = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i8))); i8 += 8;
        const int16x8_t vk8xOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi8x01234567), vget_low_s16(vk8x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi8x01234567), vget_high_s16(vk8x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi8x89ABCDEF), vget_low_s16(vk8x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi8x89ABCDEF), vget_high_s16(vk8x89ABCDEF));
        vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi8xGHIJKLMN), vget_low_s16(vk8xGHIJKLMN));
        vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi8xGHIJKLMN), vget_high_s16(vk8xGHIJKLMN));
        vaccOPQR = vmlal_s16(vaccOPQR, vget_low_s16(vi8xOPQRSTUV), vget_low_s16(vk8xOPQRSTUV));
        vaccSTUV = vmlal_s16(vaccSTUV, vget_high_s16(vi8xOPQRSTUV), vget_high_s16(vk8xOPQRSTUV));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
        float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
        float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);
        float32x4_t vfpaccGHIJ = vcvtq_f32_s32(vaccGHIJ);
        float32x4_t vfpaccKLMN = vcvtq_f32_s32(vaccKLMN);
        float32x4_t vfpaccOPQR = vcvtq_f32_s32(vaccOPQR);
        float32x4_t vfpaccSTUV = vcvtq_f32_s32(vaccSTUV);

        vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale);
        vfpacc89AB = vmulq_f32(vfpacc89AB, vscale);
        vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscale);
        vfpaccGHIJ = vmulq_f32(vfpaccGHIJ, vscale);
        vfpaccKLMN = vmulq_f32(vfpaccKLMN, vscale);
        vfpaccOPQR = vmulq_f32(vfpaccOPQR, vscale);
        vfpaccSTUV = vmulq_f32(vfpaccSTUV, vscale);

        vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
        vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));
        vacc89AB = vreinterpretq_s32_f32(vaddq_f32(vfpacc89AB, vmagic_bias));
        vaccCDEF = vreinterpretq_s32_f32(vaddq_f32(vfpaccCDEF, vmagic_bias));
        vaccGHIJ = vreinterpretq_s32_f32(vaddq_f32(vfpaccGHIJ, vmagic_bias));
        vaccKLMN = vreinterpretq_s32_f32(vaddq_f32(vfpaccKLMN, vmagic_bias));
        vaccOPQR = vreinterpretq_s32_f32(vaddq_f32(vfpaccOPQR, vmagic_bias));
        vaccSTUV = vreinterpretq_s32_f32(vaddq_f32(vfpaccSTUV, vmagic_bias));

        vacc0123 = vqsubq_s32(vacc0123, vmagic_bias_less_output_zero_point);
        vacc4567 = vqsubq_s32(vacc4567, vmagic_bias_less_output_zero_point);
        vacc89AB = vqsubq_s32(vacc89AB, vmagic_bias_less_output_zero_point);
        vaccCDEF = vqsubq_s32(vaccCDEF, vmagic_bias_less_output_zero_point);
        vaccGHIJ = vqsubq_s32(vaccGHIJ, vmagic_bias_less_output_zero_point);
        vaccKLMN = vqsubq_s32(vaccKLMN, vmagic_bias_less_output_zero_point);
        vaccOPQR = vqsubq_s32(vaccOPQR, vmagic_bias_less_output_zero_point);
        vaccSTUV = vqsubq_s32(vaccSTUV, vmagic_bias_less_output_zero_point);

#if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
        int16x8_t vacc89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF);
        int16x8_t vaccGHIJKLMN = vqmovn_high_s32(vqmovn_s32(vaccGHIJ), vaccKLMN);
        int16x8_t vaccOPQRSTUV = vqmovn_high_s32(vqmovn_s32(vaccOPQR), vaccSTUV);


        uint8x16_t vout0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc01234567), vacc89ABCDEF);
        uint8x16_t voutGHIJKLMNOPQRSTUV = vqmovun_high_s16(vqmovun_s16(vaccGHIJKLMN), vaccOPQRSTUV);
#else  // !XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
        int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));
        int16x8_t vaccGHIJKLMN = vcombine_s16(vqmovn_s32(vaccGHIJ), vqmovn_s32(vaccKLMN));
        int16x8_t vaccOPQRSTUV = vcombine_s16(vqmovn_s32(vaccOPQR), vqmovn_s32(vaccSTUV));


        uint8x16_t vout0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc01234567), vqmovun_s16(vacc89ABCDEF));
        uint8x16_t voutGHIJKLMNOPQRSTUV = vcombine_u8(vqmovun_s16(vaccGHIJKLMN), vqmovun_s16(vaccOPQRSTUV));
#endif  // !XNN_ARCH_ARM64

        vout0123456789ABCDEF = vmaxq_u8(vout0123456789ABCDEF, voutput_min);
        voutGHIJKLMNOPQRSTUV = vmaxq_u8(voutGHIJKLMNOPQRSTUV, voutput_min);

        vout0123456789ABCDEF = vminq_u8(vout0123456789ABCDEF, voutput_max);
        voutGHIJKLMNOPQRSTUV = vminq_u8(voutGHIJKLMNOPQRSTUV, voutput_max);

        vst1q_u8(output, vout0123456789ABCDEF); output += 16;
        vst1q_u8(output, voutGHIJKLMNOPQRSTUV); output += 16;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          int32x4_t vacc0123 = vld1q_s32(b); b += 4;
          int32x4_t vacc4567 = vld1q_s32(b); b += 4;

          const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
          const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));

          const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
          const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));

          const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
          const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));

          const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
          const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));

          const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
          const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));

          const int16x8_t vi5x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
          const int16x8_t vk5x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));

          const int16x8_t vi6x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
          const int16x8_t vk6x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));

          const int16x8_t vi7x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
          const int16x8_t vk7x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));

          const int16x8_t vi8x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i8))); i8 += 8;
          const int16x8_t vk8x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi8x01234567), vget_low_s16(vk8x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi8x01234567), vget_high_s16(vk8x01234567));

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
          vout01234567 = vmax_u8(vout01234567, vget_low_u8(voutput_min));
          vout01234567 = vmin_u8(vout01234567, vget_low_u8(voutput_max));

          if XNN_LIKELY(c >= 8) {
            vst1_u8(output, vout01234567); output += 8;
            c -= 8;
          } else {
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
            c = 0;
          }
        } while (c != 0);
      }
    }

    input = (const uint8_t**) ((uintptr_t) input + input_stride);
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
