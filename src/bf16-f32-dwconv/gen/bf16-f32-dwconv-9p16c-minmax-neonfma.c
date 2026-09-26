// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-dwconv/unipass-neon.c.in
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


// Widens 8 bf16 values to two vectors of 4 fp32 values.
static XNN_INLINE float32x4_t bf16_lo_to_f32(uint16x8_t v) {
  return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16));
}

static XNN_INLINE float32x4_t bf16_hi_to_f32(uint16x8_t v) {
  return vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v), 16));
}

// Packed weights layout, per block of 16 channels:
//   float bias[16];
//   uint16_t kernel[9][16];  // bf16
void xnn_bf16_f32_dwconv_minmax_ukernel_9p16c__neonfma(
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
    for (; c >= 16; c -= 16) {
      const uint16_t* wk = (const uint16_t*) (wb + 16);
      float32x4_t vacc0p0 = vld1q_f32(wb + 0);
      float32x4_t vacc4p0 = vld1q_f32(wb + 4);
      float32x4_t vacc8p0 = vld1q_f32(wb + 8);
      float32x4_t vacc12p0 = vld1q_f32(wb + 12);


      const uint16x8_t vi0x0 = vld1q_u16(i0 + 0);
      const uint16x8_t vi0x8 = vld1q_u16(i0 + 8);
      i0 += 16;

      const uint16x8_t vk0x0 = vld1q_u16(wk + 0);
      const uint16x8_t vk0x8 = vld1q_u16(wk + 8);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi0x0), bf16_lo_to_f32(vk0x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi0x0), bf16_hi_to_f32(vk0x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi0x8), bf16_lo_to_f32(vk0x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi0x8), bf16_hi_to_f32(vk0x8));

      const uint16x8_t vi1x0 = vld1q_u16(i1 + 0);
      const uint16x8_t vi1x8 = vld1q_u16(i1 + 8);
      i1 += 16;

      const uint16x8_t vk1x0 = vld1q_u16(wk + 16);
      const uint16x8_t vk1x8 = vld1q_u16(wk + 24);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi1x0), bf16_lo_to_f32(vk1x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi1x0), bf16_hi_to_f32(vk1x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi1x8), bf16_lo_to_f32(vk1x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi1x8), bf16_hi_to_f32(vk1x8));

      const uint16x8_t vi2x0 = vld1q_u16(i2 + 0);
      const uint16x8_t vi2x8 = vld1q_u16(i2 + 8);
      i2 += 16;

      const uint16x8_t vk2x0 = vld1q_u16(wk + 32);
      const uint16x8_t vk2x8 = vld1q_u16(wk + 40);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi2x0), bf16_lo_to_f32(vk2x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi2x0), bf16_hi_to_f32(vk2x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi2x8), bf16_lo_to_f32(vk2x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi2x8), bf16_hi_to_f32(vk2x8));

      const uint16x8_t vi3x0 = vld1q_u16(i3 + 0);
      const uint16x8_t vi3x8 = vld1q_u16(i3 + 8);
      i3 += 16;

      const uint16x8_t vk3x0 = vld1q_u16(wk + 48);
      const uint16x8_t vk3x8 = vld1q_u16(wk + 56);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi3x0), bf16_lo_to_f32(vk3x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi3x0), bf16_hi_to_f32(vk3x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi3x8), bf16_lo_to_f32(vk3x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi3x8), bf16_hi_to_f32(vk3x8));

      const uint16x8_t vi4x0 = vld1q_u16(i4 + 0);
      const uint16x8_t vi4x8 = vld1q_u16(i4 + 8);
      i4 += 16;

      const uint16x8_t vk4x0 = vld1q_u16(wk + 64);
      const uint16x8_t vk4x8 = vld1q_u16(wk + 72);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi4x0), bf16_lo_to_f32(vk4x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi4x0), bf16_hi_to_f32(vk4x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi4x8), bf16_lo_to_f32(vk4x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi4x8), bf16_hi_to_f32(vk4x8));

      const uint16x8_t vi5x0 = vld1q_u16(i5 + 0);
      const uint16x8_t vi5x8 = vld1q_u16(i5 + 8);
      i5 += 16;

      const uint16x8_t vk5x0 = vld1q_u16(wk + 80);
      const uint16x8_t vk5x8 = vld1q_u16(wk + 88);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi5x0), bf16_lo_to_f32(vk5x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi5x0), bf16_hi_to_f32(vk5x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi5x8), bf16_lo_to_f32(vk5x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi5x8), bf16_hi_to_f32(vk5x8));

      const uint16x8_t vi6x0 = vld1q_u16(i6 + 0);
      const uint16x8_t vi6x8 = vld1q_u16(i6 + 8);
      i6 += 16;

      const uint16x8_t vk6x0 = vld1q_u16(wk + 96);
      const uint16x8_t vk6x8 = vld1q_u16(wk + 104);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi6x0), bf16_lo_to_f32(vk6x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi6x0), bf16_hi_to_f32(vk6x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi6x8), bf16_lo_to_f32(vk6x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi6x8), bf16_hi_to_f32(vk6x8));

      const uint16x8_t vi7x0 = vld1q_u16(i7 + 0);
      const uint16x8_t vi7x8 = vld1q_u16(i7 + 8);
      i7 += 16;

      const uint16x8_t vk7x0 = vld1q_u16(wk + 112);
      const uint16x8_t vk7x8 = vld1q_u16(wk + 120);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi7x0), bf16_lo_to_f32(vk7x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi7x0), bf16_hi_to_f32(vk7x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi7x8), bf16_lo_to_f32(vk7x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi7x8), bf16_hi_to_f32(vk7x8));

      const uint16x8_t vi8x0 = vld1q_u16(i8 + 0);
      const uint16x8_t vi8x8 = vld1q_u16(i8 + 8);
      i8 += 16;

      const uint16x8_t vk8x0 = vld1q_u16(wk + 128);
      const uint16x8_t vk8x8 = vld1q_u16(wk + 136);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi8x0), bf16_lo_to_f32(vk8x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi8x0), bf16_hi_to_f32(vk8x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi8x8), bf16_lo_to_f32(vk8x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi8x8), bf16_hi_to_f32(vk8x8));

      wb = (const float*) (wk + 144);


      float32x4_t vacc0 = vmaxq_f32(vacc0p0, vmin);
      float32x4_t vacc4 = vmaxq_f32(vacc4p0, vmin);
      float32x4_t vacc8 = vmaxq_f32(vacc8p0, vmin);
      float32x4_t vacc12 = vmaxq_f32(vacc12p0, vmin);
      vacc0 = vminq_f32(vacc0, vmax);
      vacc4 = vminq_f32(vacc4, vmax);
      vacc8 = vminq_f32(vacc8, vmax);
      vacc12 = vminq_f32(vacc12, vmax);

      vst1q_f32(output + 0, vacc0);
      vst1q_f32(output + 4, vacc4);
      vst1q_f32(output + 8, vacc8);
      vst1q_f32(output + 12, vacc12);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      // Reads past the last channel are covered by XNN_EXTRA_BYTES on the
      // inputs and by the channel tile padding of the packed weights.
      const uint16_t* wk = (const uint16_t*) (wb + 16);
      do {
        float32x4_t vacc0p0 = vld1q_f32(wb);
        float32x4_t vacc4p0 = vld1q_f32(wb + 4);

        const uint16x8_t vi0x0 = vld1q_u16(i0);
        i0 += 8;
        const uint16x8_t vk0x0 = vld1q_u16(wk + 0);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi0x0), bf16_lo_to_f32(vk0x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi0x0), bf16_hi_to_f32(vk0x0));

        const uint16x8_t vi1x0 = vld1q_u16(i1);
        i1 += 8;
        const uint16x8_t vk1x0 = vld1q_u16(wk + 16);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi1x0), bf16_lo_to_f32(vk1x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi1x0), bf16_hi_to_f32(vk1x0));

        const uint16x8_t vi2x0 = vld1q_u16(i2);
        i2 += 8;
        const uint16x8_t vk2x0 = vld1q_u16(wk + 32);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi2x0), bf16_lo_to_f32(vk2x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi2x0), bf16_hi_to_f32(vk2x0));

        const uint16x8_t vi3x0 = vld1q_u16(i3);
        i3 += 8;
        const uint16x8_t vk3x0 = vld1q_u16(wk + 48);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi3x0), bf16_lo_to_f32(vk3x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi3x0), bf16_hi_to_f32(vk3x0));

        const uint16x8_t vi4x0 = vld1q_u16(i4);
        i4 += 8;
        const uint16x8_t vk4x0 = vld1q_u16(wk + 64);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi4x0), bf16_lo_to_f32(vk4x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi4x0), bf16_hi_to_f32(vk4x0));

        const uint16x8_t vi5x0 = vld1q_u16(i5);
        i5 += 8;
        const uint16x8_t vk5x0 = vld1q_u16(wk + 80);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi5x0), bf16_lo_to_f32(vk5x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi5x0), bf16_hi_to_f32(vk5x0));

        const uint16x8_t vi6x0 = vld1q_u16(i6);
        i6 += 8;
        const uint16x8_t vk6x0 = vld1q_u16(wk + 96);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi6x0), bf16_lo_to_f32(vk6x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi6x0), bf16_hi_to_f32(vk6x0));

        const uint16x8_t vi7x0 = vld1q_u16(i7);
        i7 += 8;
        const uint16x8_t vk7x0 = vld1q_u16(wk + 112);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi7x0), bf16_lo_to_f32(vk7x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi7x0), bf16_hi_to_f32(vk7x0));

        const uint16x8_t vi8x0 = vld1q_u16(i8);
        i8 += 8;
        const uint16x8_t vk8x0 = vld1q_u16(wk + 128);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi8x0), bf16_lo_to_f32(vk8x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi8x0), bf16_hi_to_f32(vk8x0));
        wb += 8;
        wk += 8;


        float32x4_t vacc0 = vminq_f32(vmaxq_f32(vacc0p0, vmin), vmax);
        float32x4_t vacc4 = vminq_f32(vmaxq_f32(vacc4p0, vmin), vmax);

        if XNN_LIKELY(c >= 8) {
          vst1q_f32(output, vacc0);
          vst1q_f32(output + 4, vacc4);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            vst1q_f32(output, vacc0);
            vacc0 = vacc4;
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
