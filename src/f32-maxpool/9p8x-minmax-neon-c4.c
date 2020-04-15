// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/maxpool.h>


XNN_DISABLE_TSAN void xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
  const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
  do {
    float* o = output;
    {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
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
      for (; c >= 4; c -= 4) {
        const float32x4_t vi0 = vld1q_f32(i0);
        i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1);
        i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2);
        i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3);
        i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4);
        i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5);
        i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6);
        i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7);
        i7 += 4;
        const float32x4_t vi8 = vld1q_f32(i8);
        i8 += 4;

        const float32x4_t vmax018 = vmaxq_f32(vmaxq_f32(vi0, vi1), vi8);
        const float32x4_t vmax23 = vmaxq_f32(vi2, vi3);
        const float32x4_t vmax45 = vmaxq_f32(vi4, vi5);
        const float32x4_t vmax67 = vmaxq_f32(vi6, vi7);

        const float32x4_t vmax2345 = vmaxq_f32(vmax23, vmax45);
        const float32x4_t vmax01678 = vmaxq_f32(vmax018, vmax67);
        const float32x4_t vmax = vmaxq_f32(vmax2345, vmax01678);
        const float32x4_t vout = vmaxq_f32(vminq_f32(vmax, voutput_max), voutput_min);

        vst1q_f32(o, vout);
        o += 4;
      }
      if (c != 0) {
        const float32x4_t vi0 = vld1q_f32(i0);
        i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1);
        i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2);
        i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3);
        i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4);
        i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5);
        i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6);
        i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7);
        i7 += 4;
        const float32x4_t vi8 = vld1q_f32(i8);
        i8 += 4;

        const float32x4_t vmax018 = vmaxq_f32(vmaxq_f32(vi0, vi1), vi8);
        const float32x4_t vmax23 = vmaxq_f32(vi2, vi3);
        const float32x4_t vmax45 = vmaxq_f32(vi4, vi5);
        const float32x4_t vmax67 = vmaxq_f32(vi6, vi7);

        const float32x4_t vmax2345 = vmaxq_f32(vmax23, vmax45);
        const float32x4_t vmax01678 = vmaxq_f32(vmax018, vmax67);
        const float32x4_t vmax = vmaxq_f32(vmax2345, vmax01678);
        float32x4_t vout = vmaxq_f32(vminq_f32(vmax, voutput_max), voutput_min);

        float32x2_t vout_lo = vget_low_f32(vout);
        if (c & 2) {
          vst1_f32(o, vout_lo);
          vout_lo = vget_high_f32(vout);
          o += 2;
        }
        if (c & 1) {
          vst1_lane_f32(o, vout_lo, 0);
          o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
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
      for (; c >= 4; c -= 4) {
        const float32x4_t vi0 = vld1q_f32(i0);
        i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1);
        i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2);
        i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3);
        i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4);
        i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5);
        i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6);
        i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7);
        i7 += 4;
        const float32x4_t vo = vld1q_f32(o);

        const float32x4_t vmax01 = vmaxq_f32(vmaxq_f32(vi0, vi1), vo);
        const float32x4_t vmax23 = vmaxq_f32(vi2, vi3);
        const float32x4_t vmax45 = vmaxq_f32(vi4, vi5);
        const float32x4_t vmax67 = vmaxq_f32(vi6, vi7);

        const float32x4_t vmax2345 = vmaxq_f32(vmax23, vmax45);
        const float32x4_t vmax0167 = vmaxq_f32(vmax01, vmax67);
        const float32x4_t vmax = vmaxq_f32(vmax2345, vmax0167);
        const float32x4_t vout = vmaxq_f32(vminq_f32(vmax, voutput_max), voutput_min);

        vst1q_f32(o, vout);
        o += 4;
      }
      if (c != 0) {
        const float32x4_t vi0 = vld1q_f32(i0);
        const float32x4_t vi1 = vld1q_f32(i1);
        const float32x4_t vi2 = vld1q_f32(i2);
        const float32x4_t vi3 = vld1q_f32(i3);
        const float32x4_t vi4 = vld1q_f32(i4);
        const float32x4_t vi5 = vld1q_f32(i5);
        const float32x4_t vi6 = vld1q_f32(i6);
        const float32x4_t vi7 = vld1q_f32(i7);
        const float32x4_t vo = vld1q_f32(o);

        const float32x4_t vmax01 = vmaxq_f32(vmaxq_f32(vi0, vi1), vo);
        const float32x4_t vmax23 = vmaxq_f32(vi2, vi3);
        const float32x4_t vmax45 = vmaxq_f32(vi4, vi5);
        const float32x4_t vmax67 = vmaxq_f32(vi6, vi7);

        const float32x4_t vmax2345 = vmaxq_f32(vmax23, vmax45);
        const float32x4_t vmax0167 = vmaxq_f32(vmax01, vmax67);
        const float32x4_t vmax = vmaxq_f32(vmax2345, vmax0167);
        float32x4_t vout = vmaxq_f32(vminq_f32(vmax, voutput_max), voutput_min);

        float32x2_t vout_lo = vget_low_f32(vout);
        if (c & 2) {
          vst1_f32(o, vout_lo);
          vout_lo = vget_high_f32(vout);
          o += 2;
        }
        if (c & 1) {
          vst1_lane_f32(o, vout_lo, 0);
          o += 1;
        }
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}
