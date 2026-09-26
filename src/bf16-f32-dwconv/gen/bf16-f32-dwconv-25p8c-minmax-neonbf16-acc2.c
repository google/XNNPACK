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
//   uint16_t kernel[25][8];  // bf16
//
// BFMLALB/BFMLALT multiply-accumulate the even/odd bf16 lanes into fp32, so
// every group of 8 channels is accumulated as its even and odd channels and
// re-interleaved before the store. This needs the Arm BF16 extension (e.g.
// Apple M2 and later).
void xnn_bf16_f32_dwconv_minmax_ukernel_25p8c__neonbf16_acc2(
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
    const uint16_t* i9 = (const uint16_t*) input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != (const uint16_t*) zero) {
      i9 = (const uint16_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint16_t* i10 = (const uint16_t*) input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != (const uint16_t*) zero) {
      i10 = (const uint16_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint16_t* i11 = (const uint16_t*) input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != (const uint16_t*) zero) {
      i11 = (const uint16_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint16_t* i12 = (const uint16_t*) input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != (const uint16_t*) zero) {
      i12 = (const uint16_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint16_t* i13 = (const uint16_t*) input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != (const uint16_t*) zero) {
      i13 = (const uint16_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint16_t* i14 = (const uint16_t*) input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != (const uint16_t*) zero) {
      i14 = (const uint16_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint16_t* i15 = (const uint16_t*) input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != (const uint16_t*) zero) {
      i15 = (const uint16_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint16_t* i16 = (const uint16_t*) input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != (const uint16_t*) zero) {
      i16 = (const uint16_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint16_t* i17 = (const uint16_t*) input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != (const uint16_t*) zero) {
      i17 = (const uint16_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint16_t* i18 = (const uint16_t*) input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != (const uint16_t*) zero) {
      i18 = (const uint16_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint16_t* i19 = (const uint16_t*) input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != (const uint16_t*) zero) {
      i19 = (const uint16_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint16_t* i20 = (const uint16_t*) input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != (const uint16_t*) zero) {
      i20 = (const uint16_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint16_t* i21 = (const uint16_t*) input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != (const uint16_t*) zero) {
      i21 = (const uint16_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint16_t* i22 = (const uint16_t*) input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != (const uint16_t*) zero) {
      i22 = (const uint16_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint16_t* i23 = (const uint16_t*) input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != (const uint16_t*) zero) {
      i23 = (const uint16_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint16_t* i24 = (const uint16_t*) input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != (const uint16_t*) zero) {
      i24 = (const uint16_t*) ((uintptr_t) i24 + input_offset);
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
      float32x4_t vacc0ep1 = vbfmlalbq_f32(vdupq_n_f32(0.0f), vi1x0, vk1x0);
      float32x4_t vacc0op1 = vbfmlaltq_f32(vdupq_n_f32(0.0f), vi1x0, vk1x0);

      const bfloat16x8_t vi2x0 = vreinterpretq_bf16_u16(vld1q_u16(i2 + 0));
      i2 += 8;

      const bfloat16x8_t vk2x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 16));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi2x0, vk2x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi2x0, vk2x0);

      const bfloat16x8_t vi3x0 = vreinterpretq_bf16_u16(vld1q_u16(i3 + 0));
      i3 += 8;

      const bfloat16x8_t vk3x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 24));
      vacc0ep1 = vbfmlalbq_f32(vacc0ep1, vi3x0, vk3x0);
      vacc0op1 = vbfmlaltq_f32(vacc0op1, vi3x0, vk3x0);

      const bfloat16x8_t vi4x0 = vreinterpretq_bf16_u16(vld1q_u16(i4 + 0));
      i4 += 8;

      const bfloat16x8_t vk4x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 32));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi4x0, vk4x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi4x0, vk4x0);

      const bfloat16x8_t vi5x0 = vreinterpretq_bf16_u16(vld1q_u16(i5 + 0));
      i5 += 8;

      const bfloat16x8_t vk5x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 40));
      vacc0ep1 = vbfmlalbq_f32(vacc0ep1, vi5x0, vk5x0);
      vacc0op1 = vbfmlaltq_f32(vacc0op1, vi5x0, vk5x0);

      const bfloat16x8_t vi6x0 = vreinterpretq_bf16_u16(vld1q_u16(i6 + 0));
      i6 += 8;

      const bfloat16x8_t vk6x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 48));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi6x0, vk6x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi6x0, vk6x0);

      const bfloat16x8_t vi7x0 = vreinterpretq_bf16_u16(vld1q_u16(i7 + 0));
      i7 += 8;

      const bfloat16x8_t vk7x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 56));
      vacc0ep1 = vbfmlalbq_f32(vacc0ep1, vi7x0, vk7x0);
      vacc0op1 = vbfmlaltq_f32(vacc0op1, vi7x0, vk7x0);

      const bfloat16x8_t vi8x0 = vreinterpretq_bf16_u16(vld1q_u16(i8 + 0));
      i8 += 8;

      const bfloat16x8_t vk8x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 64));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi8x0, vk8x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi8x0, vk8x0);

      const bfloat16x8_t vi9x0 = vreinterpretq_bf16_u16(vld1q_u16(i9 + 0));
      i9 += 8;

      const bfloat16x8_t vk9x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 72));
      vacc0ep1 = vbfmlalbq_f32(vacc0ep1, vi9x0, vk9x0);
      vacc0op1 = vbfmlaltq_f32(vacc0op1, vi9x0, vk9x0);

      const bfloat16x8_t vi10x0 = vreinterpretq_bf16_u16(vld1q_u16(i10 + 0));
      i10 += 8;

      const bfloat16x8_t vk10x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 80));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi10x0, vk10x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi10x0, vk10x0);

      const bfloat16x8_t vi11x0 = vreinterpretq_bf16_u16(vld1q_u16(i11 + 0));
      i11 += 8;

      const bfloat16x8_t vk11x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 88));
      vacc0ep1 = vbfmlalbq_f32(vacc0ep1, vi11x0, vk11x0);
      vacc0op1 = vbfmlaltq_f32(vacc0op1, vi11x0, vk11x0);

      const bfloat16x8_t vi12x0 = vreinterpretq_bf16_u16(vld1q_u16(i12 + 0));
      i12 += 8;

      const bfloat16x8_t vk12x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 96));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi12x0, vk12x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi12x0, vk12x0);

      const bfloat16x8_t vi13x0 = vreinterpretq_bf16_u16(vld1q_u16(i13 + 0));
      i13 += 8;

      const bfloat16x8_t vk13x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 104));
      vacc0ep1 = vbfmlalbq_f32(vacc0ep1, vi13x0, vk13x0);
      vacc0op1 = vbfmlaltq_f32(vacc0op1, vi13x0, vk13x0);

      const bfloat16x8_t vi14x0 = vreinterpretq_bf16_u16(vld1q_u16(i14 + 0));
      i14 += 8;

      const bfloat16x8_t vk14x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 112));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi14x0, vk14x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi14x0, vk14x0);

      const bfloat16x8_t vi15x0 = vreinterpretq_bf16_u16(vld1q_u16(i15 + 0));
      i15 += 8;

      const bfloat16x8_t vk15x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 120));
      vacc0ep1 = vbfmlalbq_f32(vacc0ep1, vi15x0, vk15x0);
      vacc0op1 = vbfmlaltq_f32(vacc0op1, vi15x0, vk15x0);

      const bfloat16x8_t vi16x0 = vreinterpretq_bf16_u16(vld1q_u16(i16 + 0));
      i16 += 8;

      const bfloat16x8_t vk16x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 128));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi16x0, vk16x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi16x0, vk16x0);

      const bfloat16x8_t vi17x0 = vreinterpretq_bf16_u16(vld1q_u16(i17 + 0));
      i17 += 8;

      const bfloat16x8_t vk17x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 136));
      vacc0ep1 = vbfmlalbq_f32(vacc0ep1, vi17x0, vk17x0);
      vacc0op1 = vbfmlaltq_f32(vacc0op1, vi17x0, vk17x0);

      const bfloat16x8_t vi18x0 = vreinterpretq_bf16_u16(vld1q_u16(i18 + 0));
      i18 += 8;

      const bfloat16x8_t vk18x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 144));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi18x0, vk18x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi18x0, vk18x0);

      const bfloat16x8_t vi19x0 = vreinterpretq_bf16_u16(vld1q_u16(i19 + 0));
      i19 += 8;

      const bfloat16x8_t vk19x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 152));
      vacc0ep1 = vbfmlalbq_f32(vacc0ep1, vi19x0, vk19x0);
      vacc0op1 = vbfmlaltq_f32(vacc0op1, vi19x0, vk19x0);

      const bfloat16x8_t vi20x0 = vreinterpretq_bf16_u16(vld1q_u16(i20 + 0));
      i20 += 8;

      const bfloat16x8_t vk20x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 160));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi20x0, vk20x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi20x0, vk20x0);

      const bfloat16x8_t vi21x0 = vreinterpretq_bf16_u16(vld1q_u16(i21 + 0));
      i21 += 8;

      const bfloat16x8_t vk21x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 168));
      vacc0ep1 = vbfmlalbq_f32(vacc0ep1, vi21x0, vk21x0);
      vacc0op1 = vbfmlaltq_f32(vacc0op1, vi21x0, vk21x0);

      const bfloat16x8_t vi22x0 = vreinterpretq_bf16_u16(vld1q_u16(i22 + 0));
      i22 += 8;

      const bfloat16x8_t vk22x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 176));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi22x0, vk22x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi22x0, vk22x0);

      const bfloat16x8_t vi23x0 = vreinterpretq_bf16_u16(vld1q_u16(i23 + 0));
      i23 += 8;

      const bfloat16x8_t vk23x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 184));
      vacc0ep1 = vbfmlalbq_f32(vacc0ep1, vi23x0, vk23x0);
      vacc0op1 = vbfmlaltq_f32(vacc0op1, vi23x0, vk23x0);

      const bfloat16x8_t vi24x0 = vreinterpretq_bf16_u16(vld1q_u16(i24 + 0));
      i24 += 8;

      const bfloat16x8_t vk24x0 = vreinterpretq_bf16_u16(vld1q_u16(wk + 192));
      vacc0ep0 = vbfmlalbq_f32(vacc0ep0, vi24x0, vk24x0);
      vacc0op0 = vbfmlaltq_f32(vacc0op0, vi24x0, vk24x0);

      wb = (const float*) (wk + 200);

      vacc0ep0 = vaddq_f32(vacc0ep0, vacc0ep1);
      vacc0op0 = vaddq_f32(vacc0op0, vacc0op1);

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
        float32x4_t vaccep1 = vbfmlalbq_f32(vdupq_n_f32(0.0f), vi1, vk1);
        float32x4_t vaccop1 = vbfmlaltq_f32(vdupq_n_f32(0.0f), vi1, vk1);

        const bfloat16x8_t vi2 = vreinterpretq_bf16_u16(vld1q_u16(i2));
        i2 += 8;
        const bfloat16x8_t vk2 = vreinterpretq_bf16_u16(vld1q_u16(wk + 16));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi2, vk2);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi2, vk2);

        const bfloat16x8_t vi3 = vreinterpretq_bf16_u16(vld1q_u16(i3));
        i3 += 8;
        const bfloat16x8_t vk3 = vreinterpretq_bf16_u16(vld1q_u16(wk + 24));
        vaccep1 = vbfmlalbq_f32(vaccep1, vi3, vk3);
        vaccop1 = vbfmlaltq_f32(vaccop1, vi3, vk3);

        const bfloat16x8_t vi4 = vreinterpretq_bf16_u16(vld1q_u16(i4));
        i4 += 8;
        const bfloat16x8_t vk4 = vreinterpretq_bf16_u16(vld1q_u16(wk + 32));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi4, vk4);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi4, vk4);

        const bfloat16x8_t vi5 = vreinterpretq_bf16_u16(vld1q_u16(i5));
        i5 += 8;
        const bfloat16x8_t vk5 = vreinterpretq_bf16_u16(vld1q_u16(wk + 40));
        vaccep1 = vbfmlalbq_f32(vaccep1, vi5, vk5);
        vaccop1 = vbfmlaltq_f32(vaccop1, vi5, vk5);

        const bfloat16x8_t vi6 = vreinterpretq_bf16_u16(vld1q_u16(i6));
        i6 += 8;
        const bfloat16x8_t vk6 = vreinterpretq_bf16_u16(vld1q_u16(wk + 48));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi6, vk6);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi6, vk6);

        const bfloat16x8_t vi7 = vreinterpretq_bf16_u16(vld1q_u16(i7));
        i7 += 8;
        const bfloat16x8_t vk7 = vreinterpretq_bf16_u16(vld1q_u16(wk + 56));
        vaccep1 = vbfmlalbq_f32(vaccep1, vi7, vk7);
        vaccop1 = vbfmlaltq_f32(vaccop1, vi7, vk7);

        const bfloat16x8_t vi8 = vreinterpretq_bf16_u16(vld1q_u16(i8));
        i8 += 8;
        const bfloat16x8_t vk8 = vreinterpretq_bf16_u16(vld1q_u16(wk + 64));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi8, vk8);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi8, vk8);

        const bfloat16x8_t vi9 = vreinterpretq_bf16_u16(vld1q_u16(i9));
        i9 += 8;
        const bfloat16x8_t vk9 = vreinterpretq_bf16_u16(vld1q_u16(wk + 72));
        vaccep1 = vbfmlalbq_f32(vaccep1, vi9, vk9);
        vaccop1 = vbfmlaltq_f32(vaccop1, vi9, vk9);

        const bfloat16x8_t vi10 = vreinterpretq_bf16_u16(vld1q_u16(i10));
        i10 += 8;
        const bfloat16x8_t vk10 = vreinterpretq_bf16_u16(vld1q_u16(wk + 80));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi10, vk10);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi10, vk10);

        const bfloat16x8_t vi11 = vreinterpretq_bf16_u16(vld1q_u16(i11));
        i11 += 8;
        const bfloat16x8_t vk11 = vreinterpretq_bf16_u16(vld1q_u16(wk + 88));
        vaccep1 = vbfmlalbq_f32(vaccep1, vi11, vk11);
        vaccop1 = vbfmlaltq_f32(vaccop1, vi11, vk11);

        const bfloat16x8_t vi12 = vreinterpretq_bf16_u16(vld1q_u16(i12));
        i12 += 8;
        const bfloat16x8_t vk12 = vreinterpretq_bf16_u16(vld1q_u16(wk + 96));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi12, vk12);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi12, vk12);

        const bfloat16x8_t vi13 = vreinterpretq_bf16_u16(vld1q_u16(i13));
        i13 += 8;
        const bfloat16x8_t vk13 = vreinterpretq_bf16_u16(vld1q_u16(wk + 104));
        vaccep1 = vbfmlalbq_f32(vaccep1, vi13, vk13);
        vaccop1 = vbfmlaltq_f32(vaccop1, vi13, vk13);

        const bfloat16x8_t vi14 = vreinterpretq_bf16_u16(vld1q_u16(i14));
        i14 += 8;
        const bfloat16x8_t vk14 = vreinterpretq_bf16_u16(vld1q_u16(wk + 112));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi14, vk14);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi14, vk14);

        const bfloat16x8_t vi15 = vreinterpretq_bf16_u16(vld1q_u16(i15));
        i15 += 8;
        const bfloat16x8_t vk15 = vreinterpretq_bf16_u16(vld1q_u16(wk + 120));
        vaccep1 = vbfmlalbq_f32(vaccep1, vi15, vk15);
        vaccop1 = vbfmlaltq_f32(vaccop1, vi15, vk15);

        const bfloat16x8_t vi16 = vreinterpretq_bf16_u16(vld1q_u16(i16));
        i16 += 8;
        const bfloat16x8_t vk16 = vreinterpretq_bf16_u16(vld1q_u16(wk + 128));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi16, vk16);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi16, vk16);

        const bfloat16x8_t vi17 = vreinterpretq_bf16_u16(vld1q_u16(i17));
        i17 += 8;
        const bfloat16x8_t vk17 = vreinterpretq_bf16_u16(vld1q_u16(wk + 136));
        vaccep1 = vbfmlalbq_f32(vaccep1, vi17, vk17);
        vaccop1 = vbfmlaltq_f32(vaccop1, vi17, vk17);

        const bfloat16x8_t vi18 = vreinterpretq_bf16_u16(vld1q_u16(i18));
        i18 += 8;
        const bfloat16x8_t vk18 = vreinterpretq_bf16_u16(vld1q_u16(wk + 144));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi18, vk18);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi18, vk18);

        const bfloat16x8_t vi19 = vreinterpretq_bf16_u16(vld1q_u16(i19));
        i19 += 8;
        const bfloat16x8_t vk19 = vreinterpretq_bf16_u16(vld1q_u16(wk + 152));
        vaccep1 = vbfmlalbq_f32(vaccep1, vi19, vk19);
        vaccop1 = vbfmlaltq_f32(vaccop1, vi19, vk19);

        const bfloat16x8_t vi20 = vreinterpretq_bf16_u16(vld1q_u16(i20));
        i20 += 8;
        const bfloat16x8_t vk20 = vreinterpretq_bf16_u16(vld1q_u16(wk + 160));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi20, vk20);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi20, vk20);

        const bfloat16x8_t vi21 = vreinterpretq_bf16_u16(vld1q_u16(i21));
        i21 += 8;
        const bfloat16x8_t vk21 = vreinterpretq_bf16_u16(vld1q_u16(wk + 168));
        vaccep1 = vbfmlalbq_f32(vaccep1, vi21, vk21);
        vaccop1 = vbfmlaltq_f32(vaccop1, vi21, vk21);

        const bfloat16x8_t vi22 = vreinterpretq_bf16_u16(vld1q_u16(i22));
        i22 += 8;
        const bfloat16x8_t vk22 = vreinterpretq_bf16_u16(vld1q_u16(wk + 176));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi22, vk22);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi22, vk22);

        const bfloat16x8_t vi23 = vreinterpretq_bf16_u16(vld1q_u16(i23));
        i23 += 8;
        const bfloat16x8_t vk23 = vreinterpretq_bf16_u16(vld1q_u16(wk + 184));
        vaccep1 = vbfmlalbq_f32(vaccep1, vi23, vk23);
        vaccop1 = vbfmlaltq_f32(vaccop1, vi23, vk23);

        const bfloat16x8_t vi24 = vreinterpretq_bf16_u16(vld1q_u16(i24));
        i24 += 8;
        const bfloat16x8_t vk24 = vreinterpretq_bf16_u16(vld1q_u16(wk + 192));
        vaccep0 = vbfmlalbq_f32(vaccep0, vi24, vk24);
        vaccop0 = vbfmlaltq_f32(vaccop0, vi24, vk24);
        wb += 8;
        wk += 8;

        vaccep0 = vaddq_f32(vaccep0, vaccep1);
        vaccop0 = vaddq_f32(vaccop0, vaccop1);

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
