// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-dwconv/unipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


// Packed weights layout, per block of 2 channels:
//   float bias[2];
//   uint16_t kernel[3][2];  // bf16
void xnn_bf16_f32_dwconv_minmax_ukernel_3p2c__scalar(
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
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
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
    for (; c >= 2; c -= 2) {
      const uint16_t* wk = (const uint16_t*) (wb + 2);
      float vacc0p0 = wb[0];
      float vacc1p0 = wb[1];


      const float vi0x0 = math_cvt_fp32_bf16(i0[0]);
      const float vi0x1 = math_cvt_fp32_bf16(i0[1]);
      i0 += 2;

      const float vk0x0 = math_cvt_fp32_bf16(wk[0]);
      vacc0p0 = math_muladd_f32(vi0x0, vk0x0, vacc0p0);
      const float vk0x1 = math_cvt_fp32_bf16(wk[1]);
      vacc1p0 = math_muladd_f32(vi0x1, vk0x1, vacc1p0);

      const float vi1x0 = math_cvt_fp32_bf16(i1[0]);
      const float vi1x1 = math_cvt_fp32_bf16(i1[1]);
      i1 += 2;

      const float vk1x0 = math_cvt_fp32_bf16(wk[2]);
      vacc0p0 = math_muladd_f32(vi1x0, vk1x0, vacc0p0);
      const float vk1x1 = math_cvt_fp32_bf16(wk[3]);
      vacc1p0 = math_muladd_f32(vi1x1, vk1x1, vacc1p0);

      const float vi2x0 = math_cvt_fp32_bf16(i2[0]);
      const float vi2x1 = math_cvt_fp32_bf16(i2[1]);
      i2 += 2;

      const float vk2x0 = math_cvt_fp32_bf16(wk[4]);
      vacc0p0 = math_muladd_f32(vi2x0, vk2x0, vacc0p0);
      const float vk2x1 = math_cvt_fp32_bf16(wk[5]);
      vacc1p0 = math_muladd_f32(vi2x1, vk2x1, vacc1p0);

      wb = (const float*) (wk + 6);


      float vacc0 = math_max_f32(vacc0p0, vmin);
      float vacc1 = math_max_f32(vacc1p0, vmin);

      vacc0 = math_min_f32(vacc0, vmax);
      vacc1 = math_min_f32(vacc1, vmax);

      output[0] = vacc0;
      output[1] = vacc1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint16_t* wk = (const uint16_t*) (wb + 2);
      do {
        float vacc0p0 = *wb++;

        const float vi0 = math_cvt_fp32_bf16(*i0++);
        const float vk0 = math_cvt_fp32_bf16(wk[0]);
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = math_cvt_fp32_bf16(*i1++);
        const float vk1 = math_cvt_fp32_bf16(wk[2]);
        vacc0p0 = math_muladd_f32(vi1, vk1, vacc0p0);
        const float vi2 = math_cvt_fp32_bf16(*i2++);
        const float vk2 = math_cvt_fp32_bf16(wk[4]);
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);
        wk += 1;


        float vacc0 = math_max_f32(vacc0p0, vmin);
        vacc0 = math_min_f32(vacc0, vmax);
        *output++ = vacc0;
      } while (--c != 0);
    }

    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
