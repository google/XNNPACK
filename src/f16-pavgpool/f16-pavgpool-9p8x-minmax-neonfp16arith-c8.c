// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/pavgpool.h"


void xnn_f16_pavgpool_minmax_ukernel_9p8x__neonfp16arith_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    const void* multiplier,
    void* buffer,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  do {
    {
      const uint16_t* i0 = (const uint16_t*) *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = (const uint16_t*) *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = (const uint16_t*) *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = (const uint16_t*) *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = (const uint16_t*) *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = (const uint16_t*) *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = (const uint16_t*) *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = (const uint16_t*) *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint16_t* i8 = (const uint16_t*) *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
      }

      uint16_t* b = (uint16_t*) buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vi8 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum018 = vaddq_f16(vsum01, vi8);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum01678 = vaddq_f16(vsum018, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum01678);

        vst1q_u16(b, vreinterpretq_u16_f16(vsum)); b += 8;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const uint16_t* i0 = (const uint16_t*) *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = (const uint16_t*) *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = (const uint16_t*) *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = (const uint16_t*) *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = (const uint16_t*) *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = (const uint16_t*) *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = (const uint16_t*) *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = (const uint16_t*) *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }

      uint16_t* b = (uint16_t*) buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(b));

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum01a = vaddq_f16(vsum01, vacc);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum0167a = vaddq_f16(vsum01a, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum0167a);

        vst1q_u16(b, vreinterpretq_u16_f16(vsum)); b += 8;
      }
    }

    {
      const uint16_t* i0 = (const uint16_t*) input[0];
      assert(i0 != NULL);
      const uint16_t* i1 = (const uint16_t*) input[1];
      const uint16_t* i2 = (const uint16_t*) input[2];
      const uint16_t* i3 = (const uint16_t*) input[3];
      const uint16_t* i4 = (const uint16_t*) input[4];
      const uint16_t* i5 = (const uint16_t*) input[5];
      const uint16_t* i6 = (const uint16_t*) input[6];
      const uint16_t* i7 = (const uint16_t*) input[7];
      input = (const void**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = (const uint16_t*) zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = (const uint16_t*) zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = (const uint16_t*) zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = (const uint16_t*) zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = (const uint16_t*) zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = (const uint16_t*) zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = (const uint16_t*) zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }

      const float16x8_t vmultiplier = vreinterpretq_f16_u16(vld1q_dup_u16(multiplier)); multiplier = (const uint16_t*) multiplier + 1;

      size_t c = channels;
      const uint16_t* b = (const uint16_t*) buffer;
      while (c >= 8) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
        const float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum01a = vaddq_f16(vsum01, vacc);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum0167a = vaddq_f16(vsum01a, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum0167a);

        float16x8_t vout = vmulq_f16(vsum, vmultiplier);
        vout = vmaxq_f16(vout, voutput_min);
        vout = vminq_f16(vout, voutput_max);

        vst1q_u16(output, vreinterpretq_u16_f16(vout)); output = (uint16_t*) output + 8;

        c -= 8;
      }
      if (c != 0) {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i0));
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i1));
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i2));
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i3));
        const float16x8_t vi4 = vreinterpretq_f16_u16(vld1q_u16(i4));
        const float16x8_t vi5 = vreinterpretq_f16_u16(vld1q_u16(i5));
        const float16x8_t vi6 = vreinterpretq_f16_u16(vld1q_u16(i6));
        const float16x8_t vi7 = vreinterpretq_f16_u16(vld1q_u16(i7));
        const float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(b));

        const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
        const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
        const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
        const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
        const float16x8_t vsum01a = vaddq_f16(vsum01, vacc);
        const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
        const float16x8_t vsum0167a = vaddq_f16(vsum01a, vsum67);
        const float16x8_t vsum = vaddq_f16(vsum2345, vsum0167a);

        float16x8_t vout = vmulq_f16(vsum, vmultiplier);
        vout = vmaxq_f16(vout, voutput_min);
        vout = vminq_f16(vout, voutput_max);

        float16x4_t vout_lo = vget_low_f16(vout);
        if (c & 4) {
          vst1_u16(output, vreinterpret_u16_f16(vout_lo)); output = (uint16_t*) output + 4;
          vout_lo = vget_high_f16(vout);
        }
        if (c & 2) {
          vst1_lane_u32(output, vreinterpret_u32_f16(vout_lo), 0); output = (uint16_t*) output + 2;
          vout_lo = vext_f16(vout_lo, vout_lo, 2);
        }
        if (c & 1) {
          vst1_lane_u16(output, vreinterpret_u16_f16(vout_lo), 0); output = (uint16_t*) output + 1;
        }
      }
    }
    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
