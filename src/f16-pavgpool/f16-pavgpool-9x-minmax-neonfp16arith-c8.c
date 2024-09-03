// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/pavgpool.h"


void xnn_f16_pavgpool_minmax_ukernel_9x__neonfp16arith_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const xnn_float16** input,
    size_t input_offset,
    const xnn_float16* zero,
    const xnn_float16* multiplier,
    xnn_float16* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->scalar.min));
  const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->scalar.max));

  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    const uint16_t* i1 = (const uint16_t*) input[1];
    const uint16_t* i2 = (const uint16_t*) input[2];
    const uint16_t* i3 = (const uint16_t*) input[3];
    const uint16_t* i4 = (const uint16_t*) input[4];
    const uint16_t* i5 = (const uint16_t*) input[5];
    const uint16_t* i6 = (const uint16_t*) input[6];
    const uint16_t* i7 = (const uint16_t*) input[7];
    const uint16_t* i8 = (const uint16_t*) input[8];
    input = (const xnn_float16**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = (const uint16_t*) zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = (const uint16_t*) zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = (const uint16_t*) zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = (const uint16_t*) zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = (const uint16_t*) zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = (const uint16_t*) zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = (const uint16_t*) zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = (const uint16_t*) zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != (const uint16_t*) zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }

    const float16x8_t vmultiplier = vreinterpretq_f16_u16(vld1q_dup_u16(multiplier)); multiplier = (const xnn_float16*) multiplier + 1;

    size_t c = channels;
    while (c >= 8) {
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

      float16x8_t vout = vmulq_f16(vsum, vmultiplier);
      vout = vmaxq_f16(vout, voutput_min);
      vout = vminq_f16(vout, voutput_max);

      vst1q_u16((uint16_t*) output, vreinterpretq_u16_f16(vout)); output = (xnn_float16*) output + 8;

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
      const float16x8_t vi8 = vreinterpretq_f16_u16(vld1q_u16(i8));

      const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
      const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
      const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
      const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
      const float16x8_t vsum018 = vaddq_f16(vsum01, vi8);
      const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
      const float16x8_t vsum01678 = vaddq_f16(vsum018, vsum67);
      const float16x8_t vsum = vaddq_f16(vsum2345, vsum01678);

      float16x8_t vout = vmulq_f16(vsum, vmultiplier);
      vout = vmaxq_f16(vout, voutput_min);
      vout = vminq_f16(vout, voutput_max);

      float16x4_t vout_lo = vget_low_f16(vout);
      if (c & 4) {
        vst1_u16((uint16_t*) output, vreinterpret_u16_f16(vout_lo)); output = (xnn_float16*) output + 4;
        vout_lo = vget_high_f16(vout);
      }
      if (c & 2) {
        vst1_lane_u32((void*) output, vreinterpret_u32_f16(vout_lo), 0); output = (xnn_float16*) output + 2;
        vout_lo = vext_f16(vout_lo, vout_lo, 2);
      }
      if (c & 1) {
        vst1_lane_u16((uint16_t*) output, vreinterpret_u16_f16(vout_lo), 0); output = (xnn_float16*) output + 1;
      }
    }
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
