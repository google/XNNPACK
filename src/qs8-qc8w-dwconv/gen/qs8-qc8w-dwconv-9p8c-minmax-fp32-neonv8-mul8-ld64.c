// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-neon-mul8.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__neonv8_mul8_ld64(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
  const int8x8_t voutput_min = vld1_dup_s8(&params->fp32_neonv8.output_min);
  const int8x8_t voutput_max = vld1_dup_s8(&params->fp32_neonv8.output_max);
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
    for (; c >= 8; c -= 8) {
      int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

      const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
      const int8x8_t vk0x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod01234567 = vmull_s8(vi0x01234567, vk0x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
      const int8x8_t vk1x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi1x01234567, vk1x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
      const int8x8_t vk2x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi2x01234567, vk2x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi3x01234567 = vld1_s8(i3); i3 += 8;
      const int8x8_t vk3x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi3x01234567, vk3x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi4x01234567 = vld1_s8(i4); i4 += 8;
      const int8x8_t vk4x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi4x01234567, vk4x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi5x01234567 = vld1_s8(i5); i5 += 8;
      const int8x8_t vk5x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi5x01234567, vk5x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi6x01234567 = vld1_s8(i6); i6 += 8;
      const int8x8_t vk6x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi6x01234567, vk6x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi7x01234567 = vld1_s8(i7); i7 += 8;
      const int8x8_t vk7x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi7x01234567, vk7x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi8x01234567 = vld1_s8(i8); i8 += 8;
      const int8x8_t vk8x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi8x01234567, vk8x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

      const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);

      vacc0123 = vcvtnq_s32_f32(vfpacc0123);
      vacc4567 = vcvtnq_s32_f32(vfpacc4567);

#if XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

      int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
#else  // !XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

      int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
#endif  // !XNN_ARCH_ARM64

      vout01234567 = vmax_s8(vout01234567, voutput_min);

      vout01234567 = vmin_s8(vout01234567, voutput_max);

      vst1_s8(output, vout01234567); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

        const int8x8_t vi0x01234567 = vld1_s8(i0);
        const int8x8_t vk0x01234567 = vld1_s8(w);

        int16x8_t vprod01234567 = vmull_s8(vi0x01234567, vk0x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi1x01234567 = vld1_s8(i1);
        const int8x8_t vk1x01234567 = vld1_s8((const void*) ((const int8_t*) w + 8));

        vprod01234567 = vmull_s8(vi1x01234567, vk1x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi2x01234567 = vld1_s8(i2);
        const int8x8_t vk2x01234567 = vld1_s8((const void*) ((const int8_t*) w + 16));

        vprod01234567 = vmull_s8(vi2x01234567, vk2x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi3x01234567 = vld1_s8(i3);
        const int8x8_t vk3x01234567 = vld1_s8((const void*) ((const int8_t*) w + 24));

        vprod01234567 = vmull_s8(vi3x01234567, vk3x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi4x01234567 = vld1_s8(i4);
        const int8x8_t vk4x01234567 = vld1_s8((const void*) ((const int8_t*) w + 32));

        vprod01234567 = vmull_s8(vi4x01234567, vk4x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi5x01234567 = vld1_s8(i5);
        const int8x8_t vk5x01234567 = vld1_s8((const void*) ((const int8_t*) w + 40));

        vprod01234567 = vmull_s8(vi5x01234567, vk5x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi6x01234567 = vld1_s8(i6);
        const int8x8_t vk6x01234567 = vld1_s8((const void*) ((const int8_t*) w + 48));

        vprod01234567 = vmull_s8(vi6x01234567, vk6x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi7x01234567 = vld1_s8(i7);
        const int8x8_t vk7x01234567 = vld1_s8((const void*) ((const int8_t*) w + 56));

        vprod01234567 = vmull_s8(vi7x01234567, vk7x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi8x01234567 = vld1_s8(i8);
        const int8x8_t vk8x01234567 = vld1_s8((const void*) ((const int8_t*) w + 64));

        vprod01234567 = vmull_s8(vi8x01234567, vk8x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

        const float32x4_t vscale0123 = vld1q_f32((const float*) ((uintptr_t) w + 0 * sizeof(int32_t) + 72 * sizeof(int8_t)));
        const float32x4_t vscale4567 = vld1q_f32((const float*) ((uintptr_t) w + 0 * sizeof(int32_t) + 72 * sizeof(int8_t) + 4 * sizeof(float)));
        vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);

        vacc0123 = vcvtnq_s32_f32(vfpacc0123);
        vacc4567 = vcvtnq_s32_f32(vfpacc4567);

#if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
#else
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
#endif
        vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

        int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
        vout01234567 = vmax_s8(vout01234567, voutput_min);
        vout01234567 = vmin_s8(vout01234567, voutput_max);

        if (c & 4) {
          vst1_lane_u32((void*) output, vreinterpret_u32_s8(vout01234567), 0); output += 4;
          vout01234567 = vext_s8(vout01234567, vout01234567, 4);
        }
        if (c & 2) {
          vst1_lane_u16((void*) output, vreinterpret_u16_s8(vout01234567), 0); output += 2;
          vout01234567 = vext_s8(vout01234567, vout01234567, 2);
        }
        if (c & 1) {
          vst1_lane_s8(output, vout01234567, 0); output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
