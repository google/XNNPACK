// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/maxpool.h>


void xnn_f16_maxpool_minmax_ukernel_9p8x__neonfp16arith_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neon.min));
  const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neon.max));
  do {
    __fp16* o = output;
    {
      const __fp16* i0 = *input++;
      const __fp16* i1 = *input++;
      const __fp16* i2 = *input++;
      const __fp16* i3 = *input++;
      const __fp16* i4 = *input++;
      const __fp16* i5 = *input++;
      const __fp16* i6 = *input++;
      const __fp16* i7 = *input++;
      const __fp16* i8 = *input++;
      i0 = (const __fp16*) ((uintptr_t) i0 + input_offset);
      i1 = (const __fp16*) ((uintptr_t) i1 + input_offset);
      i2 = (const __fp16*) ((uintptr_t) i2 + input_offset);
      i3 = (const __fp16*) ((uintptr_t) i3 + input_offset);
      i4 = (const __fp16*) ((uintptr_t) i4 + input_offset);
      i5 = (const __fp16*) ((uintptr_t) i5 + input_offset);
      i6 = (const __fp16*) ((uintptr_t) i6 + input_offset);
      i7 = (const __fp16*) ((uintptr_t) i7 + input_offset);
      i8 = (const __fp16*) ((uintptr_t) i8 + input_offset);
      if (kernel_elements < 2) {
        i1 = i0;
      }
      if (kernel_elements <= 2) {
        i2 = i0;
      }
      if (kernel_elements < 4) {
        i3 = i0;
      }
      if (kernel_elements <= 4) {
        i4 = i0;
      }
      if (kernel_elements < 6) {
        i5 = i0;
      }
      if (kernel_elements <= 6) {
        i6 = i0;
      }
      if (kernel_elements < 8) {
        i7 = i0;
      }
      if (kernel_elements <= 8) {
        i8 = i0;
      }

      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const float16x8_t vi0 = vld1q_f16(i0); i0 += 8;
        const float16x8_t vi1 = vld1q_f16(i1); i1 += 8;
        const float16x8_t vi2 = vld1q_f16(i2); i2 += 8;
        const float16x8_t vi3 = vld1q_f16(i3); i3 += 8;
        const float16x8_t vi4 = vld1q_f16(i4); i4 += 8;
        const float16x8_t vi5 = vld1q_f16(i5); i5 += 8;
        const float16x8_t vi6 = vld1q_f16(i6); i6 += 8;
        const float16x8_t vi7 = vld1q_f16(i7); i7 += 8;
        const float16x8_t vi8 = vld1q_f16(i8); i8 += 8;

        const float16x8_t vmax018 = vmaxq_f16(vmaxq_f16(vi0, vi1), vi8);
        const float16x8_t vmax23 = vmaxq_f16(vi2, vi3);
        const float16x8_t vmax45 = vmaxq_f16(vi4, vi5);
        const float16x8_t vmax67 = vmaxq_f16(vi6, vi7);

        const float16x8_t vmax2345 = vmaxq_f16(vmax23, vmax45);
        const float16x8_t vmax01678 = vmaxq_f16(vmax018, vmax67);
        const float16x8_t vmax = vmaxq_f16(vmax2345, vmax01678);
        const float16x8_t vout = vmaxq_f16(vminq_f16(vmax, voutput_max), voutput_min);

        vst1q_f16(o, vout); o += 8;
      }
      if (c != 0) {
        const float16x8_t vi0 = vld1q_f16(i0); i0 += 8;
        const float16x8_t vi1 = vld1q_f16(i1); i1 += 8;
        const float16x8_t vi2 = vld1q_f16(i2); i2 += 8;
        const float16x8_t vi3 = vld1q_f16(i3); i3 += 8;
        const float16x8_t vi4 = vld1q_f16(i4); i4 += 8;
        const float16x8_t vi5 = vld1q_f16(i5); i5 += 8;
        const float16x8_t vi6 = vld1q_f16(i6); i6 += 8;
        const float16x8_t vi7 = vld1q_f16(i7); i7 += 8;
        const float16x8_t vi8 = vld1q_f16(i8); i8 += 8;

        const float16x8_t vmax018 = vmaxq_f16(vmaxq_f16(vi0, vi1), vi8);
        const float16x8_t vmax23 = vmaxq_f16(vi2, vi3);
        const float16x8_t vmax45 = vmaxq_f16(vi4, vi5);
        const float16x8_t vmax67 = vmaxq_f16(vi6, vi7);

        const float16x8_t vmax2345 = vmaxq_f16(vmax23, vmax45);
        const float16x8_t vmax01678 = vmaxq_f16(vmax018, vmax67);
        const float16x8_t vmax = vmaxq_f16(vmax2345, vmax01678);
        float16x8_t vout = vmaxq_f16(vminq_f16(vmax, voutput_max), voutput_min);

        float16x4_t vout_lo = vget_low_f16(vout);
        if (c & 4) {
          vst1_f16(o, vout_lo); o += 4;
          vout_lo = vget_high_f16(vout);
        }
        if (c & 2) {
          vst1_lane_u32((void*) o, vreinterpret_u32_f16(vout_lo), 0); o += 2;
          vout_lo = vext_f16(vout_lo, vout_lo, 2);
        }
        if (c & 1) {
          vst1_lane_f16(o, vout_lo, 0); o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const __fp16* i0 = *input++;
      const __fp16* i1 = *input++;
      const __fp16* i2 = *input++;
      const __fp16* i3 = *input++;
      const __fp16* i4 = *input++;
      const __fp16* i5 = *input++;
      const __fp16* i6 = *input++;
      const __fp16* i7 = *input++;
      i0 = (const __fp16*) ((uintptr_t) i0 + input_offset);
      i1 = (const __fp16*) ((uintptr_t) i1 + input_offset);
      i2 = (const __fp16*) ((uintptr_t) i2 + input_offset);
      i3 = (const __fp16*) ((uintptr_t) i3 + input_offset);
      i4 = (const __fp16*) ((uintptr_t) i4 + input_offset);
      i5 = (const __fp16*) ((uintptr_t) i5 + input_offset);
      i6 = (const __fp16*) ((uintptr_t) i6 + input_offset);
      i7 = (const __fp16*) ((uintptr_t) i7 + input_offset);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k < 8) {
        i7 = i0;
      }

      o = output;
      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const float16x8_t vi0 = vld1q_f16(i0); i0 += 8;
        const float16x8_t vi1 = vld1q_f16(i1); i1 += 8;
        const float16x8_t vi2 = vld1q_f16(i2); i2 += 8;
        const float16x8_t vi3 = vld1q_f16(i3); i3 += 8;
        const float16x8_t vi4 = vld1q_f16(i4); i4 += 8;
        const float16x8_t vi5 = vld1q_f16(i5); i5 += 8;
        const float16x8_t vi6 = vld1q_f16(i6); i6 += 8;
        const float16x8_t vi7 = vld1q_f16(i7); i7 += 8;
        const float16x8_t vo = vld1q_f16(o);

        const float16x8_t vmax01 = vmaxq_f16(vmaxq_f16(vi0, vi1), vo);
        const float16x8_t vmax23 = vmaxq_f16(vi2, vi3);
        const float16x8_t vmax45 = vmaxq_f16(vi4, vi5);
        const float16x8_t vmax67 = vmaxq_f16(vi6, vi7);

        const float16x8_t vmax2345 = vmaxq_f16(vmax23, vmax45);
        const float16x8_t vmax0167 = vmaxq_f16(vmax01, vmax67);
        const float16x8_t vmax = vmaxq_f16(vmax2345, vmax0167);
        const float16x8_t vout = vmaxq_f16(vminq_f16(vmax, voutput_max), voutput_min);

        vst1q_f16(o, vout); o += 8;
      }
      if (c != 0) {
        const float16x8_t vi0 = vld1q_f16(i0);
        const float16x8_t vi1 = vld1q_f16(i1);
        const float16x8_t vi2 = vld1q_f16(i2);
        const float16x8_t vi3 = vld1q_f16(i3);
        const float16x8_t vi4 = vld1q_f16(i4);
        const float16x8_t vi5 = vld1q_f16(i5);
        const float16x8_t vi6 = vld1q_f16(i6);
        const float16x8_t vi7 = vld1q_f16(i7);
        const float16x8_t vo = vld1q_f16(o);

        const float16x8_t vmax01 = vmaxq_f16(vmaxq_f16(vi0, vi1), vo);
        const float16x8_t vmax23 = vmaxq_f16(vi2, vi3);
        const float16x8_t vmax45 = vmaxq_f16(vi4, vi5);
        const float16x8_t vmax67 = vmaxq_f16(vi6, vi7);

        const float16x8_t vmax2345 = vmaxq_f16(vmax23, vmax45);
        const float16x8_t vmax0167 = vmaxq_f16(vmax01, vmax67);
        const float16x8_t vmax = vmaxq_f16(vmax2345, vmax0167);
        float16x8_t vout = vmaxq_f16(vminq_f16(vmax, voutput_max), voutput_min);

        float16x4_t vout_lo = vget_low_f16(vout);
        if (c & 4) {
          vst1_f16(o, vout_lo); o += 4;
          vout_lo = vget_high_f16(vout);
        }
        if (c & 2) {
          vst1_lane_u32((void*) o, vreinterpret_u32_f16(vout_lo), 0); o += 2;
          vout_lo = vext_f16(vout_lo, vout_lo, 2);
        }
        if (c & 1) {
          vst1_lane_f16(o, vout_lo, 0); o += 1;
        }
      }
    }
    input = (const void**) ((uintptr_t) input + input_increment);
    output = (__fp16*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}
