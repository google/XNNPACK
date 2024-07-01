// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/multipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qu8_dwconv_minmax_fp32_ukernel_5f5m5l2c1s1r__scalar_fmagic(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  const float vscale = params->fp32_scalar_fmagic.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
  const int32_t vkernel_zero_point = params->fp32_scalar_fmagic.kernel_zero_point;
  do {
    const void* w = weights;

    // First pass to process 5 inputs.
    {
      int32_t* b = buffer;
      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      size_t c = channels;
      for (; c >= 2; c -= 2) {
        int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
        int32_t vacc1 = unaligned_indexed_load_s32(w, 1);

        const int32_t vi0x0 = (int32_t) (uint32_t) i0[0];
        const int32_t vi0x1 = (int32_t) (uint32_t) i0[1];
        i0 += 2;

        const int32_t vk0x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0] - vkernel_zero_point;
        const int32_t vk0x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1] - vkernel_zero_point;

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;

        const int32_t vi1x0 = (int32_t) (uint32_t) i1[0];
        const int32_t vi1x1 = (int32_t) (uint32_t) i1[1];
        i1 += 2;

        const int32_t vk1x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2] - vkernel_zero_point;
        const int32_t vk1x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3] - vkernel_zero_point;

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;

        const int32_t vi2x0 = (int32_t) (uint32_t) i2[0];
        const int32_t vi2x1 = (int32_t) (uint32_t) i2[1];
        i2 += 2;

        const int32_t vk2x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4] - vkernel_zero_point;
        const int32_t vk2x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5] - vkernel_zero_point;

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;

        const int32_t vi3x0 = (int32_t) (uint32_t) i3[0];
        const int32_t vi3x1 = (int32_t) (uint32_t) i3[1];
        i3 += 2;

        const int32_t vk3x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6] - vkernel_zero_point;
        const int32_t vk3x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7] - vkernel_zero_point;

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;

        const int32_t vi4x0 = (int32_t) (uint32_t) i4[0];
        const int32_t vi4x1 = (int32_t) (uint32_t) i4[1];
        i4 += 2;

        const int32_t vk4x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8] - vkernel_zero_point;
        const int32_t vk4x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9] - vkernel_zero_point;

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;

        w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 10 * sizeof(uint8_t));
        b[0] = vacc0;
        b[1] = vacc1;
        b += 2;
      }
      if XNN_UNLIKELY(c != 0) {
        int32_t vacc = unaligned_load_s32(w);

        const int32_t vi0 = (int32_t) (uint32_t) *i0;
        const int32_t vk0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[0] - vkernel_zero_point;
        vacc += vi0 * vk0;
        const int32_t vi1 = (int32_t) (uint32_t) *i1;
        const int32_t vk1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[1] - vkernel_zero_point;
        vacc += vi1 * vk1;
        const int32_t vi2 = (int32_t) (uint32_t) *i2;
        const int32_t vk2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[2] - vkernel_zero_point;
        vacc += vi2 * vk2;
        const int32_t vi3 = (int32_t) (uint32_t) *i3;
        const int32_t vk3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[3] - vkernel_zero_point;
        vacc += vi3 * vk3;
        const int32_t vi4 = (int32_t) (uint32_t) *i4;
        const int32_t vk4 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[4] - vkernel_zero_point;
        vacc += vi4 * vk4;

        w = (const void*) ((uintptr_t) w + 1 * sizeof(int32_t) + 5 * sizeof(uint8_t));
        *b++ = vacc;
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
      int32_t* b = buffer;
      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      size_t c = channels;
      for (; c >= 2; c -= 2) {
        int32_t vacc0 = b[0];
        int32_t vacc1 = b[1];

        const int32_t vi0x0 = (int32_t) (uint32_t) i0[0];
        const int32_t vi0x1 = (int32_t) (uint32_t) i0[1];
        i0 += 2;

        const int32_t vk0x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vkernel_zero_point;
        const int32_t vk0x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vkernel_zero_point;

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;

        const int32_t vi1x0 = (int32_t) (uint32_t) i1[0];
        const int32_t vi1x1 = (int32_t) (uint32_t) i1[1];
        i1 += 2;

        const int32_t vk1x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vkernel_zero_point;
        const int32_t vk1x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vkernel_zero_point;

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;

        const int32_t vi2x0 = (int32_t) (uint32_t) i2[0];
        const int32_t vi2x1 = (int32_t) (uint32_t) i2[1];
        i2 += 2;

        const int32_t vk2x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[4] - vkernel_zero_point;
        const int32_t vk2x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[5] - vkernel_zero_point;

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;

        const int32_t vi3x0 = (int32_t) (uint32_t) i3[0];
        const int32_t vi3x1 = (int32_t) (uint32_t) i3[1];
        i3 += 2;

        const int32_t vk3x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[6] - vkernel_zero_point;
        const int32_t vk3x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[7] - vkernel_zero_point;

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;

        const int32_t vi4x0 = (int32_t) (uint32_t) i4[0];
        const int32_t vi4x1 = (int32_t) (uint32_t) i4[1];
        i4 += 2;

        const int32_t vk4x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[8] - vkernel_zero_point;
        const int32_t vk4x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[9] - vkernel_zero_point;

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;

        w = (const void*) ((uintptr_t) w + 10 * sizeof(uint8_t));
        b[0] = vacc0;
        b[1] = vacc1;
        b += 2;
      }
      if XNN_UNLIKELY(c != 0) {
        int32_t vacc = b[0];

        const int32_t vi0 = (int32_t) (uint32_t) *i0;
        const int32_t vk0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vkernel_zero_point;
        vacc += vi0 * vk0;
        const int32_t vi1 = (int32_t) (uint32_t) *i1;
        const int32_t vk1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vkernel_zero_point;
        vacc += vi1 * vk1;
        const int32_t vi2 = (int32_t) (uint32_t) *i2;
        const int32_t vk2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vkernel_zero_point;
        vacc += vi2 * vk2;
        const int32_t vi3 = (int32_t) (uint32_t) *i3;
        const int32_t vk3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vkernel_zero_point;
        vacc += vi3 * vk3;
        const int32_t vi4 = (int32_t) (uint32_t) *i4;
        const int32_t vk4 = (int32_t) (uint32_t) ((const uint8_t*) w)[4] - vkernel_zero_point;
        vacc += vi4 * vk4;
        w = (const void*) ((uintptr_t) w + 5 * sizeof(uint8_t));
        *b++ = vacc;
      }
    }

    // Last pass to process up to 5 inputs.
    {
      const int32_t* b = buffer;
      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }

      size_t c = channels;
      for (; c >= 2; c -= 2) {
        int32_t vacc0 = b[0];
        int32_t vacc1 = b[1];
        b += 2;

        const int32_t vi0x0 = (int32_t) (uint32_t) i0[0];
        const int32_t vi0x1 = (int32_t) (uint32_t) i0[1];
        i0 += 2;

        const int32_t vk0x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vkernel_zero_point;
        const int32_t vk0x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vkernel_zero_point;

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;

        const int32_t vi1x0 = (int32_t) (uint32_t) i1[0];
        const int32_t vi1x1 = (int32_t) (uint32_t) i1[1];
        i1 += 2;

        const int32_t vk1x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vkernel_zero_point;
        const int32_t vk1x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vkernel_zero_point;

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;

        const int32_t vi2x0 = (int32_t) (uint32_t) i2[0];
        const int32_t vi2x1 = (int32_t) (uint32_t) i2[1];
        i2 += 2;

        const int32_t vk2x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[4] - vkernel_zero_point;
        const int32_t vk2x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[5] - vkernel_zero_point;

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;

        const int32_t vi3x0 = (int32_t) (uint32_t) i3[0];
        const int32_t vi3x1 = (int32_t) (uint32_t) i3[1];
        i3 += 2;

        const int32_t vk3x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[6] - vkernel_zero_point;
        const int32_t vk3x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[7] - vkernel_zero_point;

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;

        const int32_t vi4x0 = (int32_t) (uint32_t) i4[0];
        const int32_t vi4x1 = (int32_t) (uint32_t) i4[1];
        i4 += 2;

        const int32_t vk4x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[8] - vkernel_zero_point;
        const int32_t vk4x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[9] - vkernel_zero_point;

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;

        w = (const void*) ((uintptr_t) w + 10 * sizeof(uint8_t));

        float vfpacc0 = (float) vacc0;
        float vfpacc1 = (float) vacc1;

        vfpacc0 *= vscale;
        vfpacc1 *= vscale;

        vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
        vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);

        vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
        vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);

        vfpacc0 += vmagic_bias;
        vfpacc1 += vmagic_bias;

        int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
        int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;

        output[0] = (uint8_t) vout0;
        output[1] = (uint8_t) vout1;
        output += 2;
      }
      if XNN_UNLIKELY(c != 0) {
        int32_t vacc = b[0];

        const int32_t vi0 = (int32_t) (uint32_t) *i0;
        const int32_t vk0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vkernel_zero_point;
        vacc += vi0 * vk0;
        const int32_t vi1 = (int32_t) (uint32_t) *i1;
        const int32_t vk1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vkernel_zero_point;
        vacc += vi1 * vk1;
        const int32_t vi2 = (int32_t) (uint32_t) *i2;
        const int32_t vk2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vkernel_zero_point;
        vacc += vi2 * vk2;
        const int32_t vi3 = (int32_t) (uint32_t) *i3;
        const int32_t vk3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vkernel_zero_point;
        vacc += vi3 * vk3;
        const int32_t vi4 = (int32_t) (uint32_t) *i4;
        const int32_t vk4 = (int32_t) (uint32_t) ((const uint8_t*) w)[4] - vkernel_zero_point;
        vacc += vi4 * vk4;

        float vfpacc = (float) vacc * vscale;

        vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
        vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
        vfpacc += vmagic_bias;
        int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;

        *output++ = (uint8_t) vout;
      }
    }

    input = (const uint8_t**) ((uintptr_t) input + input_stride);
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
