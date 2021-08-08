// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/up-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>


void xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output_ptr,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const struct xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  __fp16* output = (__fp16*) output_ptr;
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->max));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->min));
  do {
    const __fp16* i0 = (const __fp16*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const __fp16*) zero) {
      i0 = (const __fp16*) ((uintptr_t) i0 + input_offset);
    }
    const __fp16* i1 = (const __fp16*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const __fp16*) zero) {
      i1 = (const __fp16*) ((uintptr_t) i1 + input_offset);
    }
    const __fp16* i2 = (const __fp16*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const __fp16*) zero) {
      i2 = (const __fp16*) ((uintptr_t) i2 + input_offset);
    }
    const __fp16* i3 = (const __fp16*) input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const __fp16*) zero) {
      i3 = (const __fp16*) ((uintptr_t) i3 + input_offset);
    }

    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const __fp16* w = (const __fp16*) weights;
    for (; c >= 8; c -= 8) {
      float16x8_t vacc01234567p0 = vld1q_f16(w); w += 8;


      const float16x8_t vi0x01234567 = vld1q_f16(i0); i0 += 8;
      const float16x8_t vk0x01234567 = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vld1q_f16(i1); i1 += 8;
      const float16x8_t vk1x01234567 = vld1q_f16(w); w += 8;
      float16x8_t vacc01234567p1 = vmulq_f16(vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vld1q_f16(i2); i2 += 8;
      const float16x8_t vk2x01234567 = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vld1q_f16(i3); i3 += 8;
      const float16x8_t vk3x01234567 = vld1q_f16(w); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi3x01234567, vk3x01234567);

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = vaddq_f16(vacc01234567p0, vacc01234567p1);

      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      vst1q_f16(output, vacc01234567); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      float16x8_t vacc01234567p0 = vld1q_f16(w); w += 8;


      const float16x8_t vi0x01234567 = vld1q_f16(i0);
      const float16x8_t vk0x01234567 = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vld1q_f16(i1);
      const float16x8_t vk1x01234567 = vld1q_f16(w); w += 8;
      float16x8_t vacc01234567p1 = vmulq_f16(vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vld1q_f16(i2);
      const float16x8_t vk2x01234567 = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vld1q_f16(i3);
      const float16x8_t vk3x01234567 = vld1q_f16(w); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi3x01234567, vk3x01234567);

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = vaddq_f16(vacc01234567p0, vacc01234567p1);

      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      float16x4_t vacc0123 = vget_low_f16(vacc01234567);
      if (c & 4) {
        vst1_f16(output, vacc0123); output += 4;
        vacc0123 = vget_high_f16(vacc01234567);
      }
      if (c & 2) {
        vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_f16(vacc0123), 0); output += 2;
        vacc0123 = vext_f16(vacc0123, vacc0123, 2);
      }
      if (c & 1) {
        vst1_lane_f16(output, vacc0123, 0); output += 1;
      }
    }

    output = (__fp16*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
