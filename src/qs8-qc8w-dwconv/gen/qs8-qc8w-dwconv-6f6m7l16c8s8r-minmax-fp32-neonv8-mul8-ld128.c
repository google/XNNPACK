// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/multipass-neon-mul8.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__neonv8_mul8_ld128(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 6);

  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
  const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);
  do {
    const void* w = weights;

    // First pass to process 6 inputs.
    {
      int32_t* b = buffer;
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
      input += 6;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 16; c -= 16) {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vaccCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;

        const int8x16_t vi0x0123456789ABCDEF = vld1q_s8(i0); i0 += 16;
        const int8x16_t vk0x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        int16x8_t vprod01234567 = vmull_s8(vget_low_s8(vi0x0123456789ABCDEF), vget_low_s8(vk0x0123456789ABCDEF));
        int16x8_t vprod89ABCDEF = vmull_s8(vget_high_s8(vi0x0123456789ABCDEF), vget_high_s8(vk0x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi1x0123456789ABCDEF = vld1q_s8(i1); i1 += 16;
        const int8x16_t vk1x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi1x0123456789ABCDEF), vget_low_s8(vk1x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi1x0123456789ABCDEF), vget_high_s8(vk1x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi2x0123456789ABCDEF = vld1q_s8(i2); i2 += 16;
        const int8x16_t vk2x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi2x0123456789ABCDEF), vget_low_s8(vk2x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi2x0123456789ABCDEF), vget_high_s8(vk2x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi3x0123456789ABCDEF = vld1q_s8(i3); i3 += 16;
        const int8x16_t vk3x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi3x0123456789ABCDEF), vget_low_s8(vk3x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi3x0123456789ABCDEF), vget_high_s8(vk3x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi4x0123456789ABCDEF = vld1q_s8(i4); i4 += 16;
        const int8x16_t vk4x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi4x0123456789ABCDEF), vget_low_s8(vk4x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi4x0123456789ABCDEF), vget_high_s8(vk4x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi5x0123456789ABCDEF = vld1q_s8(i5); i5 += 16;
        const int8x16_t vk5x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi5x0123456789ABCDEF), vget_low_s8(vk5x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi5x0123456789ABCDEF), vget_high_s8(vk5x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        vst1q_s32(b, vacc0123); b += 4;
        vst1q_s32(b, vacc4567); b += 4;
        vst1q_s32(b, vacc89AB); b += 4;
        vst1q_s32(b, vaccCDEF); b += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
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

          vst1q_s32(b, vacc0123); b += 4;
          vst1q_s32(b, vacc4567); b += 4;
          c -= 8;
        } while (c != 0);
      }
    }

    // Middle pass to process 6 inputs in each iteration.
    for (size_t ks = kernel_size - 6; ks > 7; ks -= 6) {
      int32_t* b = buffer;

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
      input += 6;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 16; c -= 16) {
        int32x4_t vacc0123 = vld1q_s32(b);
        int32x4_t vacc4567 = vld1q_s32(b + 4);
        int32x4_t vacc89AB = vld1q_s32(b + 8);
        int32x4_t vaccCDEF = vld1q_s32(b + 12);

        const int8x16_t vi0x0123456789ABCDEF = vld1q_s8(i0); i0 += 16;
        const int8x16_t vk0x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        int16x8_t vprod01234567 = vmull_s8(vget_low_s8(vi0x0123456789ABCDEF), vget_low_s8(vk0x0123456789ABCDEF));
        int16x8_t vprod89ABCDEF = vmull_s8(vget_high_s8(vi0x0123456789ABCDEF), vget_high_s8(vk0x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi1x0123456789ABCDEF = vld1q_s8(i1); i1 += 16;
        const int8x16_t vk1x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi1x0123456789ABCDEF), vget_low_s8(vk1x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi1x0123456789ABCDEF), vget_high_s8(vk1x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi2x0123456789ABCDEF = vld1q_s8(i2); i2 += 16;
        const int8x16_t vk2x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi2x0123456789ABCDEF), vget_low_s8(vk2x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi2x0123456789ABCDEF), vget_high_s8(vk2x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi3x0123456789ABCDEF = vld1q_s8(i3); i3 += 16;
        const int8x16_t vk3x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi3x0123456789ABCDEF), vget_low_s8(vk3x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi3x0123456789ABCDEF), vget_high_s8(vk3x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi4x0123456789ABCDEF = vld1q_s8(i4); i4 += 16;
        const int8x16_t vk4x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi4x0123456789ABCDEF), vget_low_s8(vk4x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi4x0123456789ABCDEF), vget_high_s8(vk4x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi5x0123456789ABCDEF = vld1q_s8(i5); i5 += 16;
        const int8x16_t vk5x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi5x0123456789ABCDEF), vget_low_s8(vk5x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi5x0123456789ABCDEF), vget_high_s8(vk5x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        vst1q_s32(b, vacc0123); b += 4;
        vst1q_s32(b, vacc4567); b += 4;
        vst1q_s32(b, vacc89AB); b += 4;
        vst1q_s32(b, vaccCDEF); b += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          int32x4_t vacc0123 = vld1q_s32(b);
          int32x4_t vacc4567 = vld1q_s32(b + 4);

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

          vst1q_s32(b, vacc0123); b += 4;
          vst1q_s32(b, vacc4567); b += 4;
          c -= 8;
        } while (c != 0);
      }
    }

    // Last pass to process up to 7 inputs.
    {
      const int32_t* b = buffer;
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

      size_t c = channels;
      for (; c >= 16; c -= 16) {
        int32x4_t vacc0123 = vld1q_s32(b); b += 4;
        int32x4_t vacc4567 = vld1q_s32(b); b += 4;
        int32x4_t vacc89AB = vld1q_s32(b); b += 4;
        int32x4_t vaccCDEF = vld1q_s32(b); b += 4;

        const int8x16_t vi0x0123456789ABCDEF = vld1q_s8(i0); i0 += 16;
        const int8x16_t vk0x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        int16x8_t vprod01234567 = vmull_s8(vget_low_s8(vi0x0123456789ABCDEF), vget_low_s8(vk0x0123456789ABCDEF));
        int16x8_t vprod89ABCDEF = vmull_s8(vget_high_s8(vi0x0123456789ABCDEF), vget_high_s8(vk0x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi1x0123456789ABCDEF = vld1q_s8(i1); i1 += 16;
        const int8x16_t vk1x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi1x0123456789ABCDEF), vget_low_s8(vk1x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi1x0123456789ABCDEF), vget_high_s8(vk1x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi2x0123456789ABCDEF = vld1q_s8(i2); i2 += 16;
        const int8x16_t vk2x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi2x0123456789ABCDEF), vget_low_s8(vk2x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi2x0123456789ABCDEF), vget_high_s8(vk2x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi3x0123456789ABCDEF = vld1q_s8(i3); i3 += 16;
        const int8x16_t vk3x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi3x0123456789ABCDEF), vget_low_s8(vk3x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi3x0123456789ABCDEF), vget_high_s8(vk3x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi4x0123456789ABCDEF = vld1q_s8(i4); i4 += 16;
        const int8x16_t vk4x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi4x0123456789ABCDEF), vget_low_s8(vk4x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi4x0123456789ABCDEF), vget_high_s8(vk4x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi5x0123456789ABCDEF = vld1q_s8(i5); i5 += 16;
        const int8x16_t vk5x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi5x0123456789ABCDEF), vget_low_s8(vk5x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi5x0123456789ABCDEF), vget_high_s8(vk5x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        const int8x16_t vi6x0123456789ABCDEF = vld1q_s8(i6); i6 += 16;
        const int8x16_t vk6x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        vprod01234567 = vmull_s8(vget_low_s8(vi6x0123456789ABCDEF), vget_low_s8(vk6x0123456789ABCDEF));
        vprod89ABCDEF = vmull_s8(vget_high_s8(vi6x0123456789ABCDEF), vget_high_s8(vk6x0123456789ABCDEF));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
        float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
        float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);

        const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
        const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
        const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
        const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;

        vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);
        vfpacc89AB = vmulq_f32(vfpacc89AB, vscale89AB);
        vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscaleCDEF);

        vacc0123 = vcvtnq_s32_f32(vfpacc0123);
        vacc4567 = vcvtnq_s32_f32(vfpacc4567);
        vacc89AB = vcvtnq_s32_f32(vfpacc89AB);
        vaccCDEF = vcvtnq_s32_f32(vfpaccCDEF);

#if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
        int16x8_t vacc89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF);

        vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
        vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

        int8x16_t vout0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc01234567), vacc89ABCDEF);
#else  // !XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
        int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));

        vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
        vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

        int8x16_t vout0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc01234567), vqmovn_s16(vacc89ABCDEF));
#endif  // !XNN_ARCH_ARM64

        vout0123456789ABCDEF = vmaxq_s8(vout0123456789ABCDEF, voutput_min);

        vout0123456789ABCDEF = vminq_s8(vout0123456789ABCDEF, voutput_max);

        vst1q_s8(output, vout0123456789ABCDEF); output += 16;
      }


      if XNN_UNLIKELY(c != 0) {
        do {
          int32x4_t vacc0123 = vld1q_s32(b); b += 4;
          int32x4_t vacc4567 = vld1q_s32(b); b += 4;

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
#else
          int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
#endif
          vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

          int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
          vout01234567 = vmax_s8(vout01234567, vget_low_s8(voutput_min));
          vout01234567 = vmin_s8(vout01234567, vget_low_s8(voutput_max));

          if XNN_LIKELY(c >= 8) {
            vst1_s8(output, vout01234567); output += 8;
            c -= 8;
          } else {
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
            c = 0;
          }

        } while (c != 0);
      }
    }

    input = (const int8_t**) ((uintptr_t) input + input_stride);
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
