// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-dwconv/unipass-neonbf16.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/microparams.h"


// Packed weights layout, per block of 8 channels:
//   float bias[8];
//   uint16_t kernel[9][8];  // bf16
//
// BFMLALB/BFMLALT multiply-accumulate the even/odd bf16 lanes into fp32, so
// every group of 8 channels is accumulated as its even and odd channels and
// re-interleaved before the store. This needs the Arm BF16 extension (e.g.
// Apple M2 and later).
void xnn_bf16_f32_dwconv_minmax_ukernel_9p8c__neonbf16(
    size_t channels,
    size_t output_width,
    const xnn_bfloat16** input,
    const void* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const xnn_bfloat16* zero,
    const struct xnn_f32_minmax_params* restrict params) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
  const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = (const uint16_t*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = (const uint16_t*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = (const uint16_t*) input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = (const uint16_t*) input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = (const uint16_t*) input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = (const uint16_t*) input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = (const uint16_t*) input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = (const uint16_t*) input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != (const uint16_t*) zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const xnn_bfloat16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* wb = (const float*) weights;
    for (; c >= 8; c -= 8) {
      const uint16_t* wk = (const uint16_t*) (wb + 8);
      const float32x4x2_t vbias0 = vuzpq_f32(vld1q_f32(wb + 0), vld1q_f32(wb + 4));
      float32x4_t vacc0ep0 = vbias0.val[0];
      float32x4_t vacc0op0 = vbias0.val[1];


      const bfloat16x8_t vi0x0 = vreinterpretq_bf16_u16(vld1q_u16(i0 + 0));
      i0 += 8;

      const bfloat16x8_t vk0x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 0));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi0x0, vk0x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi0x0, vk0x0);

      const bfloat16x8_t vi1x0 = vreinterpretq_bf16_u16(vld1q_u16(i1 + 0));
      i1 += 8;

      const bfloat16x8_t vk1x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 8));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi1x0, vk1x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi1x0, vk1x0);

      const bfloat16x8_t vi2x0 = vreinterpretq_bf16_u16(vld1q_u16(i2 + 0));
      i2 += 8;

      const bfloat16x8_t vk2x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 16));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi2x0, vk2x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi2x0, vk2x0);

      const bfloat16x8_t vi3x0 = vreinterpretq_bf16_u16(vld1q_u16(i3 + 0));
      i3 += 8;

      const bfloat16x8_t vk3x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 24));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi3x0, vk3x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi3x0, vk3x0);

      const bfloat16x8_t vi4x0 = vreinterpretq_bf16_u16(vld1q_u16(i4 + 0));
      i4 += 8;

      const bfloat16x8_t vk4x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 32));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi4x0, vk4x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi4x0, vk4x0);

      const bfloat16x8_t vi5x0 = vreinterpretq_bf16_u16(vld1q_u16(i5 + 0));
      i5 += 8;

      const bfloat16x8_t vk5x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 40));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi5x0, vk5x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi5x0, vk5x0);

      const bfloat16x8_t vi6x0 = vreinterpretq_bf16_u16(vld1q_u16(i6 + 0));
      i6 += 8;

      const bfloat16x8_t vk6x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 48));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi6x0, vk6x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi6x0, vk6x0);

      const bfloat16x8_t vi7x0 = vreinterpretq_bf16_u16(vld1q_u16(i7 + 0));
      i7 += 8;

      const bfloat16x8_t vk7x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 56));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi7x0, vk7x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi7x0, vk7x0);

      const bfloat16x8_t vi8x0 = vreinterpretq_bf16_u16(vld1q_u16(i8 + 0));
      i8 += 8;

      const bfloat16x8_t vk8x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 64));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi8x0, vk8x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi8x0, vk8x0);

      wb = (const float*) (wk + 72);


      const float32x4x2_t vacc0 = vzipq_f32(
          vminq_f32(vmaxq_f32(vacc0ep0, vmin), vmax),
          vminq_f32(vmaxq_f32(vacc0op0, vmin), vmax));

      vst1q_f32(output + 0, vacc0.val[0]);
      vst1q_f32(output + 4, vacc0.val[1]);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      // Reads past the last channel are covered by XNN_EXTRA_BYTES on the
      // inputs and by the channel tile padding of the packed weights.
      const uint16_t* wk = (const uint16_t*) (wb + 8);
      do {
        const float32x4x2_t vbias = vuzpq_f32(vld1q_f32(wb), vld1q_f32(wb + 4));
        float32x4_t vaccep0 = vbias.val[0];
        float32x4_t vaccop0 = vbias.val[1];

        const bfloat16x8_t vi0 = vreinterpretq_bf16_u16(vld1q_u16(i0));
        i0 += 8;
        const bfloat16x8_t vk0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 0));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi0, vk0);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi0, vk0);

        const bfloat16x8_t vi1 = vreinterpretq_bf16_u16(vld1q_u16(i1));
        i1 += 8;
        const bfloat16x8_t vk1 = vreinterpretq_bf16_u16(vld1q_u16(wk + 8));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi1, vk1);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi1, vk1);

        const bfloat16x8_t vi2 = vreinterpretq_bf16_u16(vld1q_u16(i2));
        i2 += 8;
        const bfloat16x8_t vk2 = vreinterpretq_bf16_u16(vld1q_u16(wk + 16));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi2, vk2);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi2, vk2);

        const bfloat16x8_t vi3 = vreinterpretq_bf16_u16(vld1q_u16(i3));
        i3 += 8;
        const bfloat16x8_t vk3 = vreinterpretq_bf16_u16(vld1q_u16(wk + 24));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi3, vk3);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi3, vk3);

        const bfloat16x8_t vi4 = vreinterpretq_bf16_u16(vld1q_u16(i4));
        i4 += 8;
        const bfloat16x8_t vk4 = vreinterpretq_bf16_u16(vld1q_u16(wk + 32));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi4, vk4);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi4, vk4);

        const bfloat16x8_t vi5 = vreinterpretq_bf16_u16(vld1q_u16(i5));
        i5 += 8;
        const bfloat16x8_t vk5 = vreinterpretq_bf16_u16(vld1q_u16(wk + 40));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi5, vk5);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi5, vk5);

        const bfloat16x8_t vi6 = vreinterpretq_bf16_u16(vld1q_u16(i6));
        i6 += 8;
        const bfloat16x8_t vk6 = vreinterpretq_bf16_u16(vld1q_u16(wk + 48));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi6, vk6);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi6, vk6);

        const bfloat16x8_t vi7 = vreinterpretq_bf16_u16(vld1q_u16(i7));
        i7 += 8;
        const bfloat16x8_t vk7 = vreinterpretq_bf16_u16(vld1q_u16(wk + 56));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi7, vk7);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi7, vk7);

        const bfloat16x8_t vi8 = vreinterpretq_bf16_u16(vld1q_u16(i8));
        i8 += 8;
        const bfloat16x8_t vk8 = vreinterpretq_bf16_u16(vld1q_u16(wk + 64));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi8, vk8);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi8, vk8);
        wb += 8;
        wk += 8;


        const float32x4x2_t vacc = vzipq_f32(
            vminq_f32(vmaxq_f32(vaccep0, vmin), vmax),
            vminq_f32(vmaxq_f32(vaccop0, vmin), vmax));
        float32x4_t vacc0 = vacc.val[0];

        if XNN_LIKELY(c >= 8) {
          vst1q_f32(output, vacc0);
          vst1q_f32(output + 4, vacc.val[1]);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            vst1q_f32(output, vacc0);
            vacc0 = vacc.val[1];
            output += 4;
          }
          float32x2_t vacc0_lo = vget_low_f32(vacc0);
          if (c & 2) {
            vst1_f32(output, vacc0_lo);
            vacc0_lo = vget_high_f32(vacc0);
            output += 2;
          }
          if (c & 1) {
            vst1_lane_f32(output, vacc0_lo, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
