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


// Packed weights layout, per block of 16 channels:
//   float bias[16];
//   uint16_t kernel[3][16];  // bf16
//
// BFMLALB/BFMLALT multiply-accumulate the even/odd bf16 lanes into fp32, so
// every group of 8 channels is accumulated as its even and odd channels and
// re-interleaved before the store. This needs the Arm BF16 extension (e.g.
// Apple M2 and later).
void xnn_bf16_f32_dwconv_minmax_ukernel_3p16c__neonbf16(
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
    input = (const xnn_bfloat16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* wb = (const float*) weights;
    for (; c >= 16; c -= 16) {
      const uint16_t* wk = (const uint16_t*) (wb + 16);
      const float32x4x2_t vbias0 = vuzpq_f32(vld1q_f32(wb + 0), vld1q_f32(wb + 4));
      float32x4_t vacc0ep0 = vbias0.val[0];
      float32x4_t vacc0op0 = vbias0.val[1];
      const float32x4x2_t vbias8 = vuzpq_f32(vld1q_f32(wb + 8), vld1q_f32(wb + 12));
      float32x4_t vacc8ep0 = vbias8.val[0];
      float32x4_t vacc8op0 = vbias8.val[1];


      const bfloat16x8_t vi0x0 = vreinterpretq_bf16_u16(vld1q_u16(i0 + 0));
      const bfloat16x8_t vi0x8 = vreinterpretq_bf16_u16(vld1q_u16(i0 + 8));
      i0 += 16;

      const bfloat16x8_t vk0x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 0));
      const bfloat16x8_t vk0x8 = vreinterpretq_bf16_u16(vld1q_u16(wk + 8));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi0x0, vk0x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi0x0, vk0x0);
      vacc8ep0 = vbfmlalbq_f32(vacc8ep0, vi0x8, vk0x8);
      vacc8op0 = vbfmlaltq_f32(vacc8op0, vi0x8, vk0x8);

      const bfloat16x8_t vi1x0 = vreinterpretq_bf16_u16(vld1q_u16(i1 + 0));
      const bfloat16x8_t vi1x8 = vreinterpretq_bf16_u16(vld1q_u16(i1 + 8));
      i1 += 16;

      const bfloat16x8_t vk1x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 16));
      const bfloat16x8_t vk1x8 = vreinterpretq_bf16_u16(vld1q_u16(wk + 24));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi1x0, vk1x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi1x0, vk1x0);
      vacc8ep0 = vbfmlalbq_f32(vacc8ep0, vi1x8, vk1x8);
      vacc8op0 = vbfmlaltq_f32(vacc8op0, vi1x8, vk1x8);

      const bfloat16x8_t vi2x0 = vreinterpretq_bf16_u16(vld1q_u16(i2 + 0));
      const bfloat16x8_t vi2x8 = vreinterpretq_bf16_u16(vld1q_u16(i2 + 8));
      i2 += 16;

      const bfloat16x8_t vk2x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 32));
      const bfloat16x8_t vk2x8 = vreinterpretq_bf16_u16(vld1q_u16(wk + 40));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi2x0, vk2x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi2x0, vk2x0);
      vacc8ep0 = vbfmlalbq_f32(vacc8ep0, vi2x8, vk2x8);
      vacc8op0 = vbfmlaltq_f32(vacc8op0, vi2x8, vk2x8);

      wb = (const float*) (wk + 48);


      const float32x4x2_t vacc0 = vzipq_f32(
          vminq_f32(vmaxq_f32(vacc0ep0, vmin), vmax),
          vminq_f32(vmaxq_f32(vacc0op0, vmin), vmax));
      const float32x4x2_t vacc8 = vzipq_f32(
          vminq_f32(vmaxq_f32(vacc8ep0, vmin), vmax),
          vminq_f32(vmaxq_f32(vacc8op0, vmin), vmax));

      vst1q_f32(output + 0, vacc0.val[0]);
      vst1q_f32(output + 4, vacc0.val[1]);
      vst1q_f32(output + 8, vacc8.val[0]);
      vst1q_f32(output + 12, vacc8.val[1]);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      // Reads past the last channel are covered by XNN_EXTRA_BYTES on the
      // inputs and by the channel tile padding of the packed weights.
      const uint16_t* wk = (const uint16_t*) (wb + 16);
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
        const bfloat16x8_t vk1 = vreinterpretq_bf16_u16(vld1q_u16(wk + 16));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi1, vk1);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi1, vk1);

        const bfloat16x8_t vi2 = vreinterpretq_bf16_u16(vld1q_u16(i2));
        i2 += 8;
        const bfloat16x8_t vk2 = vreinterpretq_bf16_u16(vld1q_u16(wk + 32));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi2, vk2);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi2, vk2);
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
