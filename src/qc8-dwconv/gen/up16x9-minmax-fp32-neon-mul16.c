// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-neon-mul16.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>


void xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neon_mul16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  const float32x4_t voutput_min_less_zero_point = vld1q_dup_f32(&params->neon_fp32.output_min_less_zero_point);
  const float32x4_t voutput_max_less_zero_point = vld1q_dup_f32(&params->neon_fp32.output_max_less_zero_point);
  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->neon_fp32.magic_bias);
  const int32x4_t vmagic_bias_less_zero_point = vld1q_dup_s32(&params->neon_fp32.magic_bias_less_zero_point);
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      int32x4_t vacc0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vacc4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vacc89AB = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vaccCDEF = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);


      const int16x8_t vi0x01234567 = vmovl_s8(vld1_s8(i0)); i0 += 8;
      const int16x8_t vk0x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi0x89ABCDEF = vmovl_s8(vld1_s8(i0)); i0 += 8;
      const int16x8_t vk0x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi0x89ABCDEF), vget_low_s16(vk0x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi0x89ABCDEF), vget_high_s16(vk0x89ABCDEF));

      const int16x8_t vi1x01234567 = vmovl_s8(vld1_s8(i1)); i1 += 8;
      const int16x8_t vk1x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi1x89ABCDEF = vmovl_s8(vld1_s8(i1)); i1 += 8;
      const int16x8_t vk1x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi1x89ABCDEF), vget_low_s16(vk1x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi1x89ABCDEF), vget_high_s16(vk1x89ABCDEF));

      const int16x8_t vi2x01234567 = vmovl_s8(vld1_s8(i2)); i2 += 8;
      const int16x8_t vk2x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi2x89ABCDEF = vmovl_s8(vld1_s8(i2)); i2 += 8;
      const int16x8_t vk2x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi2x89ABCDEF), vget_low_s16(vk2x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi2x89ABCDEF), vget_high_s16(vk2x89ABCDEF));

      const int16x8_t vi3x01234567 = vmovl_s8(vld1_s8(i3)); i3 += 8;
      const int16x8_t vk3x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi3x89ABCDEF = vmovl_s8(vld1_s8(i3)); i3 += 8;
      const int16x8_t vk3x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi3x89ABCDEF), vget_low_s16(vk3x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi3x89ABCDEF), vget_high_s16(vk3x89ABCDEF));

      const int16x8_t vi4x01234567 = vmovl_s8(vld1_s8(i4)); i4 += 8;
      const int16x8_t vk4x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi4x89ABCDEF = vmovl_s8(vld1_s8(i4)); i4 += 8;
      const int16x8_t vk4x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi4x89ABCDEF), vget_low_s16(vk4x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi4x89ABCDEF), vget_high_s16(vk4x89ABCDEF));

      const int16x8_t vi5x01234567 = vmovl_s8(vld1_s8(i5)); i5 += 8;
      const int16x8_t vk5x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi5x89ABCDEF = vmovl_s8(vld1_s8(i5)); i5 += 8;
      const int16x8_t vk5x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi5x89ABCDEF), vget_low_s16(vk5x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi5x89ABCDEF), vget_high_s16(vk5x89ABCDEF));

      const int16x8_t vi6x01234567 = vmovl_s8(vld1_s8(i6)); i6 += 8;
      const int16x8_t vk6x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi6x89ABCDEF = vmovl_s8(vld1_s8(i6)); i6 += 8;
      const int16x8_t vk6x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi6x89ABCDEF), vget_low_s16(vk6x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi6x89ABCDEF), vget_high_s16(vk6x89ABCDEF));

      const int16x8_t vi7x01234567 = vmovl_s8(vld1_s8(i7)); i7 += 8;
      const int16x8_t vk7x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi7x89ABCDEF = vmovl_s8(vld1_s8(i7)); i7 += 8;
      const int16x8_t vk7x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi7x89ABCDEF), vget_low_s16(vk7x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi7x89ABCDEF), vget_high_s16(vk7x89ABCDEF));

      const int16x8_t vi8x01234567 = vmovl_s8(vld1_s8(i8)); i8 += 8;
      const int16x8_t vk8x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi8x89ABCDEF = vmovl_s8(vld1_s8(i8)); i8 += 8;
      const int16x8_t vk8x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi8x01234567), vget_low_s16(vk8x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi8x01234567), vget_high_s16(vk8x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi8x89ABCDEF), vget_low_s16(vk8x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi8x89ABCDEF), vget_high_s16(vk8x89ABCDEF));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
      float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
      float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);

      const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const void*) ((const float*) w + 4);
      const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const void*) ((const float*) w + 4);
      const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const void*) ((const float*) w + 4);
      const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const void*) ((const float*) w + 4);

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);
      vfpacc89AB = vmulq_f32(vfpacc89AB, vscale89AB);
      vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscaleCDEF);

      vfpacc0123 = vmaxq_f32(vfpacc0123, voutput_min_less_zero_point);
      vfpacc4567 = vmaxq_f32(vfpacc4567, voutput_min_less_zero_point);
      vfpacc89AB = vmaxq_f32(vfpacc89AB, voutput_min_less_zero_point);
      vfpaccCDEF = vmaxq_f32(vfpaccCDEF, voutput_min_less_zero_point);

      vfpacc0123 = vminq_f32(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = vminq_f32(vfpacc4567, voutput_max_less_zero_point);
      vfpacc89AB = vminq_f32(vfpacc89AB, voutput_max_less_zero_point);
      vfpaccCDEF = vminq_f32(vfpaccCDEF, voutput_max_less_zero_point);

      vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
      vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));
      vacc89AB = vreinterpretq_s32_f32(vaddq_f32(vfpacc89AB, vmagic_bias));
      vaccCDEF = vreinterpretq_s32_f32(vaddq_f32(vfpaccCDEF, vmagic_bias));

      vacc0123 = vsubq_s32(vacc0123, vmagic_bias_less_zero_point);
      vacc4567 = vsubq_s32(vacc4567, vmagic_bias_less_zero_point);
      vacc89AB = vsubq_s32(vacc89AB, vmagic_bias_less_zero_point);
      vaccCDEF = vsubq_s32(vaccCDEF, vmagic_bias_less_zero_point);

#if XNN_ARCH_ARM64
      const int16x8_t vacc01234567 = vuzp1q_s16(vreinterpretq_s16_s32(vacc0123), vreinterpretq_s16_s32(vacc4567));
      const int16x8_t vacc89ABCDEF = vuzp1q_s16(vreinterpretq_s16_s32(vacc89AB), vreinterpretq_s16_s32(vaccCDEF));

      int8x16_t vout0123456789ABCDEF = vuzp1q_s8(vreinterpretq_s8_s16(vacc01234567), vreinterpretq_s8_s16(vacc89ABCDEF));
#else
      const int16x8_t vacc01234567 = vcombine_s16(vmovn_s32(vacc0123), vmovn_s32(vacc4567));
      const int16x8_t vacc89ABCDEF = vcombine_s16(vmovn_s32(vacc89AB), vmovn_s32(vaccCDEF));

      int8x16_t vout0123456789ABCDEF = vcombine_s8(vmovn_s16(vacc01234567), vmovn_s16(vacc89ABCDEF));
#endif


      vst1q_s8(output, vout0123456789ABCDEF); output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
        int32x4_t vacc4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);

        const int16x8_t vi0x01234567 = vmovl_s8(vld1_s8(i0)); i0 += 8;
        const int16x8_t vk0x01234567 = vmovl_s8(vld1_s8(k)); k += 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));
        const int16x8_t vi1x01234567 = vmovl_s8(vld1_s8(i1)); i1 += 8;
        const int16x8_t vk1x01234567 = vmovl_s8(vld1_s8((const void*) (k + 8)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));
        const int16x8_t vi2x01234567 = vmovl_s8(vld1_s8(i2)); i2 += 8;
        const int16x8_t vk2x01234567 = vmovl_s8(vld1_s8((const void*) (k + 24)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));
        const int16x8_t vi3x01234567 = vmovl_s8(vld1_s8(i3)); i3 += 8;
        const int16x8_t vk3x01234567 = vmovl_s8(vld1_s8((const void*) (k + 40)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));
        const int16x8_t vi4x01234567 = vmovl_s8(vld1_s8(i4)); i4 += 8;
        const int16x8_t vk4x01234567 = vmovl_s8(vld1_s8((const void*) (k + 56)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));
        const int16x8_t vi5x01234567 = vmovl_s8(vld1_s8(i5)); i5 += 8;
        const int16x8_t vk5x01234567 = vmovl_s8(vld1_s8((const void*) (k + 72)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));
        const int16x8_t vi6x01234567 = vmovl_s8(vld1_s8(i6)); i6 += 8;
        const int16x8_t vk6x01234567 = vmovl_s8(vld1_s8((const void*) (k + 88)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));
        const int16x8_t vi7x01234567 = vmovl_s8(vld1_s8(i7)); i7 += 8;
        const int16x8_t vk7x01234567 = vmovl_s8(vld1_s8((const void*) (k + 104)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));
        const int16x8_t vi8x01234567 = vmovl_s8(vld1_s8(i8)); i8 += 8;
        const int16x8_t vk8x01234567 = vmovl_s8(vld1_s8((const void*) (k + 120)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi8x01234567), vget_low_s16(vk8x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi8x01234567), vget_high_s16(vk8x01234567));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

        const float32x4_t vscale0123 = vld1q_f32((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t)));
        const float32x4_t vscale4567 = vld1q_f32((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t) + 4 * sizeof(float)));
        vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);

        vfpacc0123 = vmaxq_f32(vfpacc0123, voutput_min_less_zero_point);
        vfpacc4567 = vmaxq_f32(vfpacc4567, voutput_min_less_zero_point);

        vfpacc0123 = vminq_f32(vfpacc0123, voutput_max_less_zero_point);
        vfpacc4567 = vminq_f32(vfpacc4567, voutput_max_less_zero_point);

        vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
        vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));

        vacc0123 = vsubq_s32(vacc0123, vmagic_bias_less_zero_point);
        vacc4567 = vsubq_s32(vacc4567, vmagic_bias_less_zero_point);

#if XNN_ARCH_ARM64
        const int16x8_t vacc01234567 = vuzp1q_s16(vreinterpretq_s16_s32(vacc0123), vreinterpretq_s16_s32(vacc4567));
        int8x8_t vout01234567 = vmovn_s16(vacc01234567);
#else
        const int16x8_t vacc01234567 = vcombine_s16(vmovn_s32(vacc0123), vmovn_s32(vacc4567));
        int8x8_t vout01234567 = vmovn_s16(vacc01234567);
#endif


        if XNN_LIKELY(c >= 8) {
          vst1_s8(output, vout01234567); output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_s8(vout01234567), 0); output += 4;
            vout01234567 = vext_s8(vout01234567, vout01234567, 4);
          }
          if (c & 2) {
            vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_s8(vout01234567), 0); output += 2;
            vout01234567 = vext_s8(vout01234567, vout01234567, 2);
          }
          if (c & 1) {
            vst1_lane_s8(output, vout01234567, 0); output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
