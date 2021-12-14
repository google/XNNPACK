// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/avgpool.h>


void xnn_f32_avgpool_minmax_ukernel_9p8x__neon_c4(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const float32x4_t vscale = vld1q_dup_f32(&params->scalar.scale);
  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);

  do {
    {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      const float* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const float*) ((uintptr_t) i8 + input_offset);
      }

      float* b = buffer;
      for (size_t c = 0; c < channels; c += 4) {
        const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7); i7 += 4;
        const float32x4_t vi8 = vld1q_f32(i8); i8 += 4;

        const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
        const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
        const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
        const float32x4_t vsum67 = vaddq_f32(vi6, vi7);
        const float32x4_t vsum018 = vaddq_f32(vsum01, vi8);
        const float32x4_t vsum2345 = vaddq_f32(vsum23, vsum45);
        const float32x4_t vsum01678 = vaddq_f32(vsum018, vsum67);
        const float32x4_t vsum = vaddq_f32(vsum2345, vsum01678);

        vst1q_f32(b, vsum); b += 4;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      float* b = buffer;
      for (size_t c = 0; c < channels; c += 4) {
        const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7); i7 += 4;
        const float32x4_t vacc = vld1q_f32(b);

        const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
        const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
        const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
        const float32x4_t vsum67 = vaddq_f32(vi6, vi7);
        const float32x4_t vsum01a = vaddq_f32(vsum01, vacc);
        const float32x4_t vsum2345 = vaddq_f32(vsum23, vsum45);
        const float32x4_t vsum0167a = vaddq_f32(vsum01a, vsum67);
        const float32x4_t vsum = vaddq_f32(vsum2345, vsum0167a);

        vst1q_f32(b, vsum); b += 4;
      }
    }

    assert(k >= 1);
    {
      const float* i0 = input[0];
      assert(i0 != NULL);
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      input = (const float**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      size_t c = channels;
      float* b = buffer;
      while (c >= 4) {
        const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7); i7 += 4;
        const float32x4_t vacc = vld1q_f32(b); b += 4;

        const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
        const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
        const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
        const float32x4_t vsum67 = vaddq_f32(vi6, vi7);
        const float32x4_t vsum01a = vaddq_f32(vsum01, vacc);
        const float32x4_t vsum2345 = vaddq_f32(vsum23, vsum45);
        const float32x4_t vsum0167a = vaddq_f32(vsum01a, vsum67);
        const float32x4_t vsum = vaddq_f32(vsum2345, vsum0167a);

        float32x4_t vout = vmulq_f32(vsum, vscale);
        vout = vmaxq_f32(vout, vmin);
        vout = vminq_f32(vout, vmax);

        vst1q_f32(output, vout); output += 4;

        c -= 4;
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
        const float32x4_t vacc = vld1q_f32(b);

        const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
        const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
        const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
        const float32x4_t vsum67 = vaddq_f32(vi6, vi7);
        const float32x4_t vsum01a = vaddq_f32(vsum01, vacc);
        const float32x4_t vsum2345 = vaddq_f32(vsum23, vsum45);
        const float32x4_t vsum0167a = vaddq_f32(vsum01a, vsum67);
        const float32x4_t vsum = vaddq_f32(vsum2345, vsum0167a);

        float32x4_t vout = vmulq_f32(vsum, vscale);
        vout = vmaxq_f32(vout, vmin);
        vout = vminq_f32(vout, vmax);

        float32x2_t vout_lo = vget_low_f32(vout);
        if (c & 2) {
          vst1_f32(output, vout_lo); output += 2;
          vout_lo = vget_high_f32(vout);
        }
        if (c & 1) {
          vst1_lane_f32(output, vout_lo, 0); output += 1;
        }
      }
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
