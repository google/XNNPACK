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
//   uint16_t kernel[25][16];  // bf16
void xnn_bf16_f32_dwconv_minmax_ukernel_25p16c__neonfma_acc2(
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
      float32x4_t vacc0p1 = vmulq_f32(bf16_lo_to_f32(vi1x0), bf16_lo_to_f32(vk1x0));
      float32x4_t vacc4p1 = vmulq_f32(bf16_hi_to_f32(vi1x0), bf16_hi_to_f32(vk1x0));
      float32x4_t vacc8p1 = vmulq_f32(bf16_lo_to_f32(vi1x8), bf16_lo_to_f32(vk1x8));
      float32x4_t vacc12p1 = vmulq_f32(bf16_hi_to_f32(vi1x8), bf16_hi_to_f32(vk1x8));

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
      vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi3x0), bf16_lo_to_f32(vk3x0));
      vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi3x0), bf16_hi_to_f32(vk3x0));
      vacc8p1 = vfmaq_f32(vacc8p1, bf16_lo_to_f32(vi3x8), bf16_lo_to_f32(vk3x8));
      vacc12p1 = vfmaq_f32(vacc12p1, bf16_hi_to_f32(vi3x8), bf16_hi_to_f32(vk3x8));

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
      vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi5x0), bf16_lo_to_f32(vk5x0));
      vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi5x0), bf16_hi_to_f32(vk5x0));
      vacc8p1 = vfmaq_f32(vacc8p1, bf16_lo_to_f32(vi5x8), bf16_lo_to_f32(vk5x8));
      vacc12p1 = vfmaq_f32(vacc12p1, bf16_hi_to_f32(vi5x8), bf16_hi_to_f32(vk5x8));

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
      vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi7x0), bf16_lo_to_f32(vk7x0));
      vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi7x0), bf16_hi_to_f32(vk7x0));
      vacc8p1 = vfmaq_f32(vacc8p1, bf16_lo_to_f32(vi7x8), bf16_lo_to_f32(vk7x8));
      vacc12p1 = vfmaq_f32(vacc12p1, bf16_hi_to_f32(vi7x8), bf16_hi_to_f32(vk7x8));

      const uint16x8_t vi8x0 = vld1q_u16(i8 + 0);
      const uint16x8_t vi8x8 = vld1q_u16(i8 + 8);
      i8 += 16;

      const uint16x8_t vk8x0 = vld1q_u16(wk + 128);
      const uint16x8_t vk8x8 = vld1q_u16(wk + 136);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi8x0), bf16_lo_to_f32(vk8x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi8x0), bf16_hi_to_f32(vk8x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi8x8), bf16_lo_to_f32(vk8x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi8x8), bf16_hi_to_f32(vk8x8));

      const uint16x8_t vi9x0 = vld1q_u16(i9 + 0);
      const uint16x8_t vi9x8 = vld1q_u16(i9 + 8);
      i9 += 16;

      const uint16x8_t vk9x0 = vld1q_u16(wk + 144);
      const uint16x8_t vk9x8 = vld1q_u16(wk + 152);
      vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi9x0), bf16_lo_to_f32(vk9x0));
      vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi9x0), bf16_hi_to_f32(vk9x0));
      vacc8p1 = vfmaq_f32(vacc8p1, bf16_lo_to_f32(vi9x8), bf16_lo_to_f32(vk9x8));
      vacc12p1 = vfmaq_f32(vacc12p1, bf16_hi_to_f32(vi9x8), bf16_hi_to_f32(vk9x8));

      const uint16x8_t vi10x0 = vld1q_u16(i10 + 0);
      const uint16x8_t vi10x8 = vld1q_u16(i10 + 8);
      i10 += 16;

      const uint16x8_t vk10x0 = vld1q_u16(wk + 160);
      const uint16x8_t vk10x8 = vld1q_u16(wk + 168);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi10x0), bf16_lo_to_f32(vk10x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi10x0), bf16_hi_to_f32(vk10x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi10x8), bf16_lo_to_f32(vk10x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi10x8), bf16_hi_to_f32(vk10x8));

      const uint16x8_t vi11x0 = vld1q_u16(i11 + 0);
      const uint16x8_t vi11x8 = vld1q_u16(i11 + 8);
      i11 += 16;

      const uint16x8_t vk11x0 = vld1q_u16(wk + 176);
      const uint16x8_t vk11x8 = vld1q_u16(wk + 184);
      vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi11x0), bf16_lo_to_f32(vk11x0));
      vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi11x0), bf16_hi_to_f32(vk11x0));
      vacc8p1 = vfmaq_f32(vacc8p1, bf16_lo_to_f32(vi11x8), bf16_lo_to_f32(vk11x8));
      vacc12p1 = vfmaq_f32(vacc12p1, bf16_hi_to_f32(vi11x8), bf16_hi_to_f32(vk11x8));

      const uint16x8_t vi12x0 = vld1q_u16(i12 + 0);
      const uint16x8_t vi12x8 = vld1q_u16(i12 + 8);
      i12 += 16;

      const uint16x8_t vk12x0 = vld1q_u16(wk + 192);
      const uint16x8_t vk12x8 = vld1q_u16(wk + 200);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi12x0), bf16_lo_to_f32(vk12x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi12x0), bf16_hi_to_f32(vk12x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi12x8), bf16_lo_to_f32(vk12x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi12x8), bf16_hi_to_f32(vk12x8));

      const uint16x8_t vi13x0 = vld1q_u16(i13 + 0);
      const uint16x8_t vi13x8 = vld1q_u16(i13 + 8);
      i13 += 16;

      const uint16x8_t vk13x0 = vld1q_u16(wk + 208);
      const uint16x8_t vk13x8 = vld1q_u16(wk + 216);
      vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi13x0), bf16_lo_to_f32(vk13x0));
      vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi13x0), bf16_hi_to_f32(vk13x0));
      vacc8p1 = vfmaq_f32(vacc8p1, bf16_lo_to_f32(vi13x8), bf16_lo_to_f32(vk13x8));
      vacc12p1 = vfmaq_f32(vacc12p1, bf16_hi_to_f32(vi13x8), bf16_hi_to_f32(vk13x8));

      const uint16x8_t vi14x0 = vld1q_u16(i14 + 0);
      const uint16x8_t vi14x8 = vld1q_u16(i14 + 8);
      i14 += 16;

      const uint16x8_t vk14x0 = vld1q_u16(wk + 224);
      const uint16x8_t vk14x8 = vld1q_u16(wk + 232);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi14x0), bf16_lo_to_f32(vk14x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi14x0), bf16_hi_to_f32(vk14x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi14x8), bf16_lo_to_f32(vk14x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi14x8), bf16_hi_to_f32(vk14x8));

      const uint16x8_t vi15x0 = vld1q_u16(i15 + 0);
      const uint16x8_t vi15x8 = vld1q_u16(i15 + 8);
      i15 += 16;

      const uint16x8_t vk15x0 = vld1q_u16(wk + 240);
      const uint16x8_t vk15x8 = vld1q_u16(wk + 248);
      vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi15x0), bf16_lo_to_f32(vk15x0));
      vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi15x0), bf16_hi_to_f32(vk15x0));
      vacc8p1 = vfmaq_f32(vacc8p1, bf16_lo_to_f32(vi15x8), bf16_lo_to_f32(vk15x8));
      vacc12p1 = vfmaq_f32(vacc12p1, bf16_hi_to_f32(vi15x8), bf16_hi_to_f32(vk15x8));

      const uint16x8_t vi16x0 = vld1q_u16(i16 + 0);
      const uint16x8_t vi16x8 = vld1q_u16(i16 + 8);
      i16 += 16;

      const uint16x8_t vk16x0 = vld1q_u16(wk + 256);
      const uint16x8_t vk16x8 = vld1q_u16(wk + 264);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi16x0), bf16_lo_to_f32(vk16x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi16x0), bf16_hi_to_f32(vk16x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi16x8), bf16_lo_to_f32(vk16x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi16x8), bf16_hi_to_f32(vk16x8));

      const uint16x8_t vi17x0 = vld1q_u16(i17 + 0);
      const uint16x8_t vi17x8 = vld1q_u16(i17 + 8);
      i17 += 16;

      const uint16x8_t vk17x0 = vld1q_u16(wk + 272);
      const uint16x8_t vk17x8 = vld1q_u16(wk + 280);
      vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi17x0), bf16_lo_to_f32(vk17x0));
      vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi17x0), bf16_hi_to_f32(vk17x0));
      vacc8p1 = vfmaq_f32(vacc8p1, bf16_lo_to_f32(vi17x8), bf16_lo_to_f32(vk17x8));
      vacc12p1 = vfmaq_f32(vacc12p1, bf16_hi_to_f32(vi17x8), bf16_hi_to_f32(vk17x8));

      const uint16x8_t vi18x0 = vld1q_u16(i18 + 0);
      const uint16x8_t vi18x8 = vld1q_u16(i18 + 8);
      i18 += 16;

      const uint16x8_t vk18x0 = vld1q_u16(wk + 288);
      const uint16x8_t vk18x8 = vld1q_u16(wk + 296);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi18x0), bf16_lo_to_f32(vk18x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi18x0), bf16_hi_to_f32(vk18x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi18x8), bf16_lo_to_f32(vk18x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi18x8), bf16_hi_to_f32(vk18x8));

      const uint16x8_t vi19x0 = vld1q_u16(i19 + 0);
      const uint16x8_t vi19x8 = vld1q_u16(i19 + 8);
      i19 += 16;

      const uint16x8_t vk19x0 = vld1q_u16(wk + 304);
      const uint16x8_t vk19x8 = vld1q_u16(wk + 312);
      vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi19x0), bf16_lo_to_f32(vk19x0));
      vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi19x0), bf16_hi_to_f32(vk19x0));
      vacc8p1 = vfmaq_f32(vacc8p1, bf16_lo_to_f32(vi19x8), bf16_lo_to_f32(vk19x8));
      vacc12p1 = vfmaq_f32(vacc12p1, bf16_hi_to_f32(vi19x8), bf16_hi_to_f32(vk19x8));

      const uint16x8_t vi20x0 = vld1q_u16(i20 + 0);
      const uint16x8_t vi20x8 = vld1q_u16(i20 + 8);
      i20 += 16;

      const uint16x8_t vk20x0 = vld1q_u16(wk + 320);
      const uint16x8_t vk20x8 = vld1q_u16(wk + 328);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi20x0), bf16_lo_to_f32(vk20x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi20x0), bf16_hi_to_f32(vk20x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi20x8), bf16_lo_to_f32(vk20x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi20x8), bf16_hi_to_f32(vk20x8));

      const uint16x8_t vi21x0 = vld1q_u16(i21 + 0);
      const uint16x8_t vi21x8 = vld1q_u16(i21 + 8);
      i21 += 16;

      const uint16x8_t vk21x0 = vld1q_u16(wk + 336);
      const uint16x8_t vk21x8 = vld1q_u16(wk + 344);
      vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi21x0), bf16_lo_to_f32(vk21x0));
      vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi21x0), bf16_hi_to_f32(vk21x0));
      vacc8p1 = vfmaq_f32(vacc8p1, bf16_lo_to_f32(vi21x8), bf16_lo_to_f32(vk21x8));
      vacc12p1 = vfmaq_f32(vacc12p1, bf16_hi_to_f32(vi21x8), bf16_hi_to_f32(vk21x8));

      const uint16x8_t vi22x0 = vld1q_u16(i22 + 0);
      const uint16x8_t vi22x8 = vld1q_u16(i22 + 8);
      i22 += 16;

      const uint16x8_t vk22x0 = vld1q_u16(wk + 352);
      const uint16x8_t vk22x8 = vld1q_u16(wk + 360);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi22x0), bf16_lo_to_f32(vk22x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi22x0), bf16_hi_to_f32(vk22x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi22x8), bf16_lo_to_f32(vk22x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi22x8), bf16_hi_to_f32(vk22x8));

      const uint16x8_t vi23x0 = vld1q_u16(i23 + 0);
      const uint16x8_t vi23x8 = vld1q_u16(i23 + 8);
      i23 += 16;

      const uint16x8_t vk23x0 = vld1q_u16(wk + 368);
      const uint16x8_t vk23x8 = vld1q_u16(wk + 376);
      vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi23x0), bf16_lo_to_f32(vk23x0));
      vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi23x0), bf16_hi_to_f32(vk23x0));
      vacc8p1 = vfmaq_f32(vacc8p1, bf16_lo_to_f32(vi23x8), bf16_lo_to_f32(vk23x8));
      vacc12p1 = vfmaq_f32(vacc12p1, bf16_hi_to_f32(vi23x8), bf16_hi_to_f32(vk23x8));

      const uint16x8_t vi24x0 = vld1q_u16(i24 + 0);
      const uint16x8_t vi24x8 = vld1q_u16(i24 + 8);
      i24 += 16;

      const uint16x8_t vk24x0 = vld1q_u16(wk + 384);
      const uint16x8_t vk24x8 = vld1q_u16(wk + 392);
      vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi24x0), bf16_lo_to_f32(vk24x0));
      vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi24x0), bf16_hi_to_f32(vk24x0));
      vacc8p0 = vfmaq_f32(vacc8p0, bf16_lo_to_f32(vi24x8), bf16_lo_to_f32(vk24x8));
      vacc12p0 = vfmaq_f32(vacc12p0, bf16_hi_to_f32(vi24x8), bf16_hi_to_f32(vk24x8));

      wb = (const float*) (wk + 400);

      vacc0p0 = vaddq_f32(vacc0p0, vacc0p1);
      vacc4p0 = vaddq_f32(vacc4p0, vacc4p1);
      vacc8p0 = vaddq_f32(vacc8p0, vacc8p1);
      vacc12p0 = vaddq_f32(vacc12p0, vacc12p1);

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
        float32x4_t vacc0p1 = vmulq_f32(bf16_lo_to_f32(vi1x0), bf16_lo_to_f32(vk1x0));
        float32x4_t vacc4p1 = vmulq_f32(bf16_hi_to_f32(vi1x0), bf16_hi_to_f32(vk1x0));

        const uint16x8_t vi2x0 = vld1q_u16(i2);
        i2 += 8;
        const uint16x8_t vk2x0 = vld1q_u16(wk + 32);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi2x0), bf16_lo_to_f32(vk2x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi2x0), bf16_hi_to_f32(vk2x0));

        const uint16x8_t vi3x0 = vld1q_u16(i3);
        i3 += 8;
        const uint16x8_t vk3x0 = vld1q_u16(wk + 48);
        vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi3x0), bf16_lo_to_f32(vk3x0));
        vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi3x0), bf16_hi_to_f32(vk3x0));

        const uint16x8_t vi4x0 = vld1q_u16(i4);
        i4 += 8;
        const uint16x8_t vk4x0 = vld1q_u16(wk + 64);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi4x0), bf16_lo_to_f32(vk4x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi4x0), bf16_hi_to_f32(vk4x0));

        const uint16x8_t vi5x0 = vld1q_u16(i5);
        i5 += 8;
        const uint16x8_t vk5x0 = vld1q_u16(wk + 80);
        vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi5x0), bf16_lo_to_f32(vk5x0));
        vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi5x0), bf16_hi_to_f32(vk5x0));

        const uint16x8_t vi6x0 = vld1q_u16(i6);
        i6 += 8;
        const uint16x8_t vk6x0 = vld1q_u16(wk + 96);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi6x0), bf16_lo_to_f32(vk6x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi6x0), bf16_hi_to_f32(vk6x0));

        const uint16x8_t vi7x0 = vld1q_u16(i7);
        i7 += 8;
        const uint16x8_t vk7x0 = vld1q_u16(wk + 112);
        vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi7x0), bf16_lo_to_f32(vk7x0));
        vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi7x0), bf16_hi_to_f32(vk7x0));

        const uint16x8_t vi8x0 = vld1q_u16(i8);
        i8 += 8;
        const uint16x8_t vk8x0 = vld1q_u16(wk + 128);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi8x0), bf16_lo_to_f32(vk8x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi8x0), bf16_hi_to_f32(vk8x0));

        const uint16x8_t vi9x0 = vld1q_u16(i9);
        i9 += 8;
        const uint16x8_t vk9x0 = vld1q_u16(wk + 144);
        vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi9x0), bf16_lo_to_f32(vk9x0));
        vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi9x0), bf16_hi_to_f32(vk9x0));

        const uint16x8_t vi10x0 = vld1q_u16(i10);
        i10 += 8;
        const uint16x8_t vk10x0 = vld1q_u16(wk + 160);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi10x0), bf16_lo_to_f32(vk10x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi10x0), bf16_hi_to_f32(vk10x0));

        const uint16x8_t vi11x0 = vld1q_u16(i11);
        i11 += 8;
        const uint16x8_t vk11x0 = vld1q_u16(wk + 176);
        vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi11x0), bf16_lo_to_f32(vk11x0));
        vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi11x0), bf16_hi_to_f32(vk11x0));

        const uint16x8_t vi12x0 = vld1q_u16(i12);
        i12 += 8;
        const uint16x8_t vk12x0 = vld1q_u16(wk + 192);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi12x0), bf16_lo_to_f32(vk12x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi12x0), bf16_hi_to_f32(vk12x0));

        const uint16x8_t vi13x0 = vld1q_u16(i13);
        i13 += 8;
        const uint16x8_t vk13x0 = vld1q_u16(wk + 208);
        vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi13x0), bf16_lo_to_f32(vk13x0));
        vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi13x0), bf16_hi_to_f32(vk13x0));

        const uint16x8_t vi14x0 = vld1q_u16(i14);
        i14 += 8;
        const uint16x8_t vk14x0 = vld1q_u16(wk + 224);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi14x0), bf16_lo_to_f32(vk14x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi14x0), bf16_hi_to_f32(vk14x0));

        const uint16x8_t vi15x0 = vld1q_u16(i15);
        i15 += 8;
        const uint16x8_t vk15x0 = vld1q_u16(wk + 240);
        vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi15x0), bf16_lo_to_f32(vk15x0));
        vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi15x0), bf16_hi_to_f32(vk15x0));

        const uint16x8_t vi16x0 = vld1q_u16(i16);
        i16 += 8;
        const uint16x8_t vk16x0 = vld1q_u16(wk + 256);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi16x0), bf16_lo_to_f32(vk16x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi16x0), bf16_hi_to_f32(vk16x0));

        const uint16x8_t vi17x0 = vld1q_u16(i17);
        i17 += 8;
        const uint16x8_t vk17x0 = vld1q_u16(wk + 272);
        vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi17x0), bf16_lo_to_f32(vk17x0));
        vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi17x0), bf16_hi_to_f32(vk17x0));

        const uint16x8_t vi18x0 = vld1q_u16(i18);
        i18 += 8;
        const uint16x8_t vk18x0 = vld1q_u16(wk + 288);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi18x0), bf16_lo_to_f32(vk18x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi18x0), bf16_hi_to_f32(vk18x0));

        const uint16x8_t vi19x0 = vld1q_u16(i19);
        i19 += 8;
        const uint16x8_t vk19x0 = vld1q_u16(wk + 304);
        vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi19x0), bf16_lo_to_f32(vk19x0));
        vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi19x0), bf16_hi_to_f32(vk19x0));

        const uint16x8_t vi20x0 = vld1q_u16(i20);
        i20 += 8;
        const uint16x8_t vk20x0 = vld1q_u16(wk + 320);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi20x0), bf16_lo_to_f32(vk20x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi20x0), bf16_hi_to_f32(vk20x0));

        const uint16x8_t vi21x0 = vld1q_u16(i21);
        i21 += 8;
        const uint16x8_t vk21x0 = vld1q_u16(wk + 336);
        vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi21x0), bf16_lo_to_f32(vk21x0));
        vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi21x0), bf16_hi_to_f32(vk21x0));

        const uint16x8_t vi22x0 = vld1q_u16(i22);
        i22 += 8;
        const uint16x8_t vk22x0 = vld1q_u16(wk + 352);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi22x0), bf16_lo_to_f32(vk22x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi22x0), bf16_hi_to_f32(vk22x0));

        const uint16x8_t vi23x0 = vld1q_u16(i23);
        i23 += 8;
        const uint16x8_t vk23x0 = vld1q_u16(wk + 368);
        vacc0p1 = vfmaq_f32(vacc0p1, bf16_lo_to_f32(vi23x0), bf16_lo_to_f32(vk23x0));
        vacc4p1 = vfmaq_f32(vacc4p1, bf16_hi_to_f32(vi23x0), bf16_hi_to_f32(vk23x0));

        const uint16x8_t vi24x0 = vld1q_u16(i24);
        i24 += 8;
        const uint16x8_t vk24x0 = vld1q_u16(wk + 384);
        vacc0p0 = vfmaq_f32(vacc0p0, bf16_lo_to_f32(vi24x0), bf16_lo_to_f32(vk24x0));
        vacc4p0 = vfmaq_f32(vacc4p0, bf16_hi_to_f32(vi24x0), bf16_hi_to_f32(vk24x0));
        wb += 8;
        wk += 8;

        vacc0p0 = vaddq_f32(vacc0p0, vacc0p1);
        vacc4p0 = vaddq_f32(vacc4p0, vacc4p1);

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
