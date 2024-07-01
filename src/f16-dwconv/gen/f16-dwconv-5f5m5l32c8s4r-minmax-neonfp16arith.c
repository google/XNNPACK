// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/multipass-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__neonfp16arith(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    size_t kernel_size,
    void* buffer,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  do {
    const uint16_t* w = weights;

    // First pass to process 5 inputs.
    {
      uint16_t* b = buffer;
      const uint16_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 32; c -= 32) {
        float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vacc89ABCDEFp0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vaccGHIJKLMNp0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vaccOPQRSTUVp0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;

        const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi0xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk0xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi0x89ABCDEF, vk0x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi0xGHIJKLMN, vk0xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi0xOPQRSTUV, vk0xOPQRSTUV);

        const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi1xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk1xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi1x89ABCDEF, vk1x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi1xGHIJKLMN, vk1xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi1xOPQRSTUV, vk1xOPQRSTUV);

        const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi2xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk2xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi2x89ABCDEF, vk2x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi2xGHIJKLMN, vk2xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi2xOPQRSTUV, vk2xOPQRSTUV);

        const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi3xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk3xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi3x89ABCDEF, vk3x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi3xGHIJKLMN, vk3xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi3xOPQRSTUV, vk3xOPQRSTUV);

        const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi4xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi4xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk4xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk4xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi4x89ABCDEF, vk4x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi4xGHIJKLMN, vk4xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi4xOPQRSTUV, vk4xOPQRSTUV);


        vst1q_u16(b, vreinterpretq_u16_f16(vacc01234567p0)); b += 8;
        vst1q_u16(b, vreinterpretq_u16_f16(vacc89ABCDEFp0)); b += 8;
        vst1q_u16(b, vreinterpretq_u16_f16(vaccGHIJKLMNp0)); b += 8;
        vst1q_u16(b, vreinterpretq_u16_f16(vaccOPQRSTUVp0)); b += 8;
      }

      for (; c >= 8; c -= 8) {
        float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


        const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

        const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

        const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

        const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

        const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);


        vst1q_u16(b, vreinterpretq_u16_f16(vacc01234567p0)); b += 8;
      }

      if (c != 0) {
        float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;

        const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

        const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

        const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

        const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

        const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);


        vst1q_u16(b, vreinterpretq_u16_f16(vacc01234567p0)); b += 8;
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
      uint16_t* b = buffer;
      const uint16_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 32; c -= 32) {
        float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(b));
        float16x8_t vacc89ABCDEFp0 = vreinterpretq_f16_u16(vld1q_u16(b + 8));
        float16x8_t vaccGHIJKLMNp0 = vreinterpretq_f16_u16(vld1q_u16(b + 16));
        float16x8_t vaccOPQRSTUVp0 = vreinterpretq_f16_u16(vld1q_u16(b + 24));

        const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi0xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk0xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi0x89ABCDEF, vk0x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi0xGHIJKLMN, vk0xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi0xOPQRSTUV, vk0xOPQRSTUV);

        const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi1xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk1xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi1x89ABCDEF, vk1x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi1xGHIJKLMN, vk1xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi1xOPQRSTUV, vk1xOPQRSTUV);

        const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi2xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk2xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi2x89ABCDEF, vk2x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi2xGHIJKLMN, vk2xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi2xOPQRSTUV, vk2xOPQRSTUV);

        const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi3xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk3xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi3x89ABCDEF, vk3x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi3xGHIJKLMN, vk3xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi3xOPQRSTUV, vk3xOPQRSTUV);

        const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi4xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi4xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk4xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        const float16x8_t vk4xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi4x89ABCDEF, vk4x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi4xGHIJKLMN, vk4xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi4xOPQRSTUV, vk4xOPQRSTUV);


        vst1q_u16(b, vreinterpretq_u16_f16(vacc01234567p0)); b += 8;
        vst1q_u16(b, vreinterpretq_u16_f16(vacc89ABCDEFp0)); b += 8;
        vst1q_u16(b, vreinterpretq_u16_f16(vaccGHIJKLMNp0)); b += 8;
        vst1q_u16(b, vreinterpretq_u16_f16(vaccOPQRSTUVp0)); b += 8;
      }

      for (; c >= 8; c -= 8) {
        float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(b));


        const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

        const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

        const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

        const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

        const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);


        vst1q_u16(b, vreinterpretq_u16_f16(vacc01234567p0)); b += 8;
      }

      if (c != 0) {
        float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(b));

        const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

        const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

        const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

        const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

        const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);


        vst1q_u16(b, vreinterpretq_u16_f16(vacc01234567p0)); b += 8;
      }
    }

    // Last pass to process up to 5 inputs.
    {
      uint16_t* b = buffer;
      const uint16_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }

      size_t c = channels;
      for (; c >= 32; c -= 32) {
        float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;
        float16x8_t vacc89ABCDEFp0 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;
        float16x8_t vaccGHIJKLMNp0 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;
        float16x8_t vaccOPQRSTUVp0 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

        const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        const float16x8_t vi0xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk0xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi0x89ABCDEF, vk0x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi0xGHIJKLMN, vk0xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi0xOPQRSTUV, vk0xOPQRSTUV);

        const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        const float16x8_t vi1xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk1xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi1x89ABCDEF, vk1x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi1xGHIJKLMN, vk1xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi1xOPQRSTUV, vk1xOPQRSTUV);

        const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        const float16x8_t vi2xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk2xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi2x89ABCDEF, vk2x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi2xGHIJKLMN, vk2xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi2xOPQRSTUV, vk2xOPQRSTUV);

        const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        const float16x8_t vi3xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk3xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi3x89ABCDEF, vk3x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi3xGHIJKLMN, vk3xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi3xOPQRSTUV, vk3xOPQRSTUV);

        const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi4xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        const float16x8_t vi4xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk4xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        float16x8_t vk4xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);
        vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi4x89ABCDEF, vk4x89ABCDEF);
        vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi4xGHIJKLMN, vk4xGHIJKLMN);
        vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi4xOPQRSTUV, vk4xOPQRSTUV);


        float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
        float16x8_t vacc89ABCDEF = vmaxq_f16(vacc89ABCDEFp0, vmin);
        float16x8_t vaccGHIJKLMN = vmaxq_f16(vaccGHIJKLMNp0, vmin);
        float16x8_t vaccOPQRSTUV = vmaxq_f16(vaccOPQRSTUVp0, vmin);

        vacc01234567 = vminq_f16(vacc01234567, vmax);
        vacc89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);
        vaccGHIJKLMN = vminq_f16(vaccGHIJKLMN, vmax);
        vaccOPQRSTUV = vminq_f16(vaccOPQRSTUV, vmax);

        vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output = (uint16_t*) output + 8;
        vst1q_u16(output, vreinterpretq_u16_f16(vacc89ABCDEF)); output = (uint16_t*) output + 8;
        vst1q_u16(output, vreinterpretq_u16_f16(vaccGHIJKLMN)); output = (uint16_t*) output + 8;
        vst1q_u16(output, vreinterpretq_u16_f16(vaccOPQRSTUV)); output = (uint16_t*) output + 8;
      }


      for (; c >= 8; c -= 8) {
        float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

        const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
        float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

        const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
        float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

        const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
        float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

        const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
        float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

        const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
        float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);


        float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);

        vacc01234567 = vminq_f16(vacc01234567, vmax);

        vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output = (uint16_t*) output + 8;
      }

      if XNN_UNLIKELY(c != 0) {
        float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(b));

        const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0));
        float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

        const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1));
        float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

        const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2));
        float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

        const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3));
        float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

        const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4));
        float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
        vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);


        float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
        vacc01234567 = vminq_f16(vacc01234567, vmax);

        float16x4_t vacc0123 = vget_low_f16(vacc01234567);
        if (c & 4) {
          vst1_u16(output, vreinterpret_u16_f16(vacc0123)); output = (uint16_t*) output + 4;
          vacc0123 = vget_high_f16(vacc01234567);
        }
        if (c & 2) {
          vst1_lane_u32((void*) output, vreinterpret_u32_f16(vacc0123), 0); output = (uint16_t*) output + 2;
          vacc0123 = vext_f16(vacc0123, vacc0123, 2);
        }
        if (c & 1) {
          vst1_lane_u16(output, vreinterpret_u16_f16(vacc0123), 0); output = (uint16_t*) output + 1;
        }
      }

    }
    input = (const void**) (const uint16_t**) ((uintptr_t) input + input_stride);
    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
