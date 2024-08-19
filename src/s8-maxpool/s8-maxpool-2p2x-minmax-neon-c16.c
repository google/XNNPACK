// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/maxpool.h"


void xnn_s8_maxpool_minmax_ukernel_2p2x__neon_c16(
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

  const int8x16_t voutput_max = vld1q_dup_s8(&params->scalar.max);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->scalar.min);
  do {
    int8_t* o = output;
    {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      if (kernel_elements < 2) {
        i1 = i0;
      }

      size_t c = channels;
      for (; c >= 16; c -= 16) {
        const int8x16_t vi0 = vld1q_s8(i0); i0 += 16;
        const int8x16_t vi1 = vld1q_s8(i1); i1 += 16;

        int8x16_t vout = vmaxq_s8(vi0, vi1);
        vout = vmaxq_s8(vout, voutput_min);
        vout = vminq_s8(vout, voutput_max);

        vst1q_s8(o, vout); o += 16;
      }
      if (c != 0) {
        const int8x16_t vi0 = vld1q_s8(i0);
        const int8x16_t vi1 = vld1q_s8(i1);

        int8x16_t vout = vmaxq_s8(vi0, vi1);
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

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 2; k > 0; k -= 2) {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      if (k < 2) {
        i1 = i0;
      }

      o = output;
      size_t c = channels;
      for (; c >= 16; c -= 16) {
        const int8x16_t vi0 = vld1q_s8(i0); i0 += 16;
        const int8x16_t vi1 = vld1q_s8(i1); i1 += 16;
        const int8x16_t vo = vld1q_s8(o);

        const int8x16_t vmax01 = vmaxq_s8(vi0, vi1);
        int8x16_t vout = vmaxq_s8(vo, vmax01);
        vout = vmaxq_s8(vout, voutput_min);
        vout = vminq_s8(vout, voutput_max);

        vst1q_s8(o, vout); o += 16;
      }
      if (c != 0) {
        const int8x16_t vi0 = vld1q_s8(i0);
        const int8x16_t vi1 = vld1q_s8(i1);
        const int8x16_t vo = vld1q_s8(o);

        const int8x16_t vmax01 = vmaxq_s8(vi0, vi1);
        int8x16_t vout = vmaxq_s8(vo, vmax01);
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
