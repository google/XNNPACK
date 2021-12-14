// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/maxpool.h>


void xnn_s8_maxpool_minmax_ukernel_9p8x__neon_c16(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const int8x16_t voutput_max = vld1q_dup_s8(&params->neon.max);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->neon.min);
  do {
    int8_t* o = output;
    {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      const int8_t* i8 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
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
      for (; c >= 16; c -= 16) {
        const int8x16_t vi0 = vld1q_s8(i0); i0 += 16;
        const int8x16_t vi1 = vld1q_s8(i1); i1 += 16;
        const int8x16_t vi2 = vld1q_s8(i2); i2 += 16;
        const int8x16_t vi3 = vld1q_s8(i3); i3 += 16;
        const int8x16_t vi4 = vld1q_s8(i4); i4 += 16;
        const int8x16_t vi5 = vld1q_s8(i5); i5 += 16;
        const int8x16_t vi6 = vld1q_s8(i6); i6 += 16;
        const int8x16_t vi7 = vld1q_s8(i7); i7 += 16;
        const int8x16_t vi8 = vld1q_s8(i8); i8 += 16;

        const int8x16_t vmax018 = vmaxq_s8(vmaxq_s8(vi0, vi1), vi8);
        const int8x16_t vmax23 = vmaxq_s8(vi2, vi3);
        const int8x16_t vmax45 = vmaxq_s8(vi4, vi5);
        const int8x16_t vmax67 = vmaxq_s8(vi6, vi7);

        const int8x16_t vmax2345 = vmaxq_s8(vmax23, vmax45);
        const int8x16_t vmax01678 = vmaxq_s8(vmax018, vmax67);
        int8x16_t vout = vmaxq_s8(vmax2345, vmax01678);
        vout = vmaxq_s8(vout, voutput_min);
        vout = vminq_s8(vout, voutput_max);

        vst1q_s8(o, vout); o += 16;
      }
      if (c != 0) {
        const int8x16_t vi0 = vld1q_s8(i0);
        const int8x16_t vi1 = vld1q_s8(i1);
        const int8x16_t vi2 = vld1q_s8(i2);
        const int8x16_t vi3 = vld1q_s8(i3);
        const int8x16_t vi4 = vld1q_s8(i4);
        const int8x16_t vi5 = vld1q_s8(i5);
        const int8x16_t vi6 = vld1q_s8(i6);
        const int8x16_t vi7 = vld1q_s8(i7);
        const int8x16_t vi8 = vld1q_s8(i8);

        const int8x16_t vmax018 = vmaxq_s8(vmaxq_s8(vi0, vi1), vi8);
        const int8x16_t vmax23 = vmaxq_s8(vi2, vi3);
        const int8x16_t vmax45 = vmaxq_s8(vi4, vi5);
        const int8x16_t vmax67 = vmaxq_s8(vi6, vi7);

        const int8x16_t vmax2345 = vmaxq_s8(vmax23, vmax45);
        const int8x16_t vmax01678 = vmaxq_s8(vmax018, vmax67);
        int8x16_t vout = vmaxq_s8(vmax2345, vmax01678);
        vout = vmaxq_s8(vout, voutput_min);
        vout = vminq_s8(vout, voutput_max);

        int8x8_t vout_lo = vget_low_s8(vout);
        if (c & 8) {
          vst1_s8(o, vout_lo); o += 8;
          vout_lo = vget_high_s8(vout);
        }
        if (c & 4) {
          vst1_lane_u32((void*) o, vreinterpret_u32_s8(vout_lo), 0); o += 4;
          vout_lo = vext_s8(vout_lo, vout_lo, 4);
        }
        if (c & 2) {
          vst1_lane_u16((void*) o, vreinterpret_u16_s8(vout_lo), 0); o += 2;
          vout_lo = vext_s8(vout_lo, vout_lo, 2);
        }
        if (c & 1) {
          vst1_lane_s8(o, vout_lo, 0); o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
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
      for (; c >= 16; c -= 16) {
        const int8x16_t vi0 = vld1q_s8(i0); i0 += 16;
        const int8x16_t vi1 = vld1q_s8(i1); i1 += 16;
        const int8x16_t vi2 = vld1q_s8(i2); i2 += 16;
        const int8x16_t vi3 = vld1q_s8(i3); i3 += 16;
        const int8x16_t vi4 = vld1q_s8(i4); i4 += 16;
        const int8x16_t vi5 = vld1q_s8(i5); i5 += 16;
        const int8x16_t vi6 = vld1q_s8(i6); i6 += 16;
        const int8x16_t vi7 = vld1q_s8(i7); i7 += 16;
        const int8x16_t vo = vld1q_s8(o);

        const int8x16_t vmax01 = vmaxq_s8(vmaxq_s8(vi0, vi1), vo);
        const int8x16_t vmax23 = vmaxq_s8(vi2, vi3);
        const int8x16_t vmax45 = vmaxq_s8(vi4, vi5);
        const int8x16_t vmax67 = vmaxq_s8(vi6, vi7);

        const int8x16_t vmax2345 = vmaxq_s8(vmax23, vmax45);
        const int8x16_t vmax0167 = vmaxq_s8(vmax01, vmax67);
        int8x16_t vout = vmaxq_s8(vmax2345, vmax0167);
        vout = vmaxq_s8(vout, voutput_min);
        vout = vminq_s8(vout, voutput_max);

        vst1q_s8(o, vout); o += 16;
      }
      if (c != 0) {
        const int8x16_t vi0 = vld1q_s8(i0);
        const int8x16_t vi1 = vld1q_s8(i1);
        const int8x16_t vi2 = vld1q_s8(i2);
        const int8x16_t vi3 = vld1q_s8(i3);
        const int8x16_t vi4 = vld1q_s8(i4);
        const int8x16_t vi5 = vld1q_s8(i5);
        const int8x16_t vi6 = vld1q_s8(i6);
        const int8x16_t vi7 = vld1q_s8(i7);
        const int8x16_t vo = vld1q_s8(o);

        const int8x16_t vmax01 = vmaxq_s8(vmaxq_s8(vi0, vi1), vo);
        const int8x16_t vmax23 = vmaxq_s8(vi2, vi3);
        const int8x16_t vmax45 = vmaxq_s8(vi4, vi5);
        const int8x16_t vmax67 = vmaxq_s8(vi6, vi7);

        const int8x16_t vmax2345 = vmaxq_s8(vmax23, vmax45);
        const int8x16_t vmax0167 = vmaxq_s8(vmax01, vmax67);
        int8x16_t vout = vmaxq_s8(vmax2345, vmax0167);
        vout = vmaxq_s8(vout, voutput_min);
        vout = vminq_s8(vout, voutput_max);

        int8x8_t vout_lo = vget_low_s8(vout);
        if (c & 8) {
          vst1_s8(o, vout_lo); o += 8;
          vout_lo = vget_high_s8(vout);
        }
        if (c & 4) {
          vst1_lane_u32((void*) o, vreinterpret_u32_s8(vout_lo), 0); o += 4;
          vout_lo = vext_s8(vout_lo, vout_lo, 4);
        }
        if (c & 2) {
          vst1_lane_u16((void*) o, vreinterpret_u16_s8(vout_lo), 0); o += 2;
          vout_lo = vext_s8(vout_lo, vout_lo, 2);
        }
        if (c & 1) {
          vst1_lane_s8(o, vout_lo, 0); o += 1;
        }
      }
    }
    input = (const int8_t**) ((uintptr_t) input + input_increment);
    output = (int8_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}
