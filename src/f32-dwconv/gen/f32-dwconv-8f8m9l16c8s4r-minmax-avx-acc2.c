// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/multipass-avx.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 8);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vmin = _mm256_set1_ps(params->scalar.min);
  const __m256 vmax = _mm256_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* w = weights;

    // First pass to process 8 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = input[5];
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = input[6];
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = input[7];
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      input += 8;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);
      for (; c >= 16; c -= 16) {
        __m256 vacc01234567p0 = _mm256_load_ps(w);
        __m256 vacc89ABCDEFp0 = _mm256_load_ps(w + 8);


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
        i0 += 16;

        const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
        const __m256 vk0x89ABCDEF = _mm256_load_ps(w + 24);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi0x89ABCDEF, vk0x89ABCDEF));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
        i1 += 16;

        const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
        const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);
        __m256 vacc89ABCDEFp1 = _mm256_mul_ps(vi1x89ABCDEF, vk1x89ABCDEF);

        const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
        const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
        i2 += 16;

        const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
        const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 56);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi2x89ABCDEF, vk2x89ABCDEF));

        const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
        const __m256 vi3x89ABCDEF = _mm256_loadu_ps(i3 + 8);
        i3 += 16;

        const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
        const __m256 vk3x89ABCDEF = _mm256_load_ps(w + 72);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi3x01234567, vk3x01234567));
        vacc89ABCDEFp1 = _mm256_add_ps(vacc89ABCDEFp1, _mm256_mul_ps(vi3x89ABCDEF, vk3x89ABCDEF));

        const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
        const __m256 vi4x89ABCDEF = _mm256_loadu_ps(i4 + 8);
        i4 += 16;

        const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
        const __m256 vk4x89ABCDEF = _mm256_load_ps(w + 88);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi4x89ABCDEF, vk4x89ABCDEF));

        const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
        const __m256 vi5x89ABCDEF = _mm256_loadu_ps(i5 + 8);
        i5 += 16;

        const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
        const __m256 vk5x89ABCDEF = _mm256_load_ps(w + 104);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi5x01234567, vk5x01234567));
        vacc89ABCDEFp1 = _mm256_add_ps(vacc89ABCDEFp1, _mm256_mul_ps(vi5x89ABCDEF, vk5x89ABCDEF));

        const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
        const __m256 vi6x89ABCDEF = _mm256_loadu_ps(i6 + 8);
        i6 += 16;

        const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
        const __m256 vk6x89ABCDEF = _mm256_load_ps(w + 120);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi6x89ABCDEF, vk6x89ABCDEF));

        const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
        const __m256 vi7x89ABCDEF = _mm256_loadu_ps(i7 + 8);
        i7 += 16;

        const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
        const __m256 vk7x89ABCDEF = _mm256_load_ps(w + 136);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi7x01234567, vk7x01234567));
        vacc89ABCDEFp1 = _mm256_add_ps(vacc89ABCDEFp1, _mm256_mul_ps(vi7x89ABCDEF, vk7x89ABCDEF));

        w += 144;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, vacc89ABCDEFp1);

        _mm256_store_ps(b, vacc01234567p0);
        _mm256_store_ps(b + 8, vacc89ABCDEFp0);
        b += 16;
      }

      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_load_ps(w);


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        i0 += 8;

        const __m256 vk0x01234567 = _mm256_load_ps(w + 8);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        i1 += 8;

        const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
        i2 += 8;

        const __m256 vk2x01234567 = _mm256_load_ps(w + 24);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

        const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
        i3 += 8;

        const __m256 vk3x01234567 = _mm256_load_ps(w + 32);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi3x01234567, vk3x01234567));

        const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
        i4 += 8;

        const __m256 vk4x01234567 = _mm256_load_ps(w + 40);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));

        const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
        i5 += 8;

        const __m256 vk5x01234567 = _mm256_load_ps(w + 48);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi5x01234567, vk5x01234567));

        const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
        i6 += 8;

        const __m256 vk6x01234567 = _mm256_load_ps(w + 56);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));

        const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
        i7 += 8;

        const __m256 vk7x01234567 = _mm256_load_ps(w + 64);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi7x01234567, vk7x01234567));

        w += 72;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        _mm256_store_ps(b, vacc01234567p0);
        b += 8;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);
        __m256 vacc01234567p0 = _mm256_load_ps(w);


        const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);

        const __m256 vk0x01234567 = _mm256_load_ps(w + 8);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);

        const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);

        const __m256 vk2x01234567 = _mm256_load_ps(w + 24);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

        const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);

        const __m256 vk3x01234567 = _mm256_load_ps(w + 32);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi3x01234567, vk3x01234567));

        const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);

        const __m256 vk4x01234567 = _mm256_load_ps(w + 40);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));

        const __m256 vi5x01234567 = _mm256_maskload_ps(i5, vmask);

        const __m256 vk5x01234567 = _mm256_load_ps(w + 48);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi5x01234567, vk5x01234567));

        const __m256 vi6x01234567 = _mm256_maskload_ps(i6, vmask);

        const __m256 vk6x01234567 = _mm256_load_ps(w + 56);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));

        const __m256 vi7x01234567 = _mm256_maskload_ps(i7, vmask);

        const __m256 vk7x01234567 = _mm256_load_ps(w + 64);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi7x01234567, vk7x01234567));

        w += 72;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        _mm256_store_ps(b, vacc01234567p0);
      }
    }

    // Middle pass to process 8 inputs in each iteration.
    for (size_t ks = kernel_size - 8; ks > 9; ks -= 8) {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = input[5];
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = input[6];
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = input[7];
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      input += 8;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 16; c -= 16) {
        __m256 vacc01234567p0 = _mm256_load_ps(b);
        __m256 vacc89ABCDEFp0 = _mm256_load_ps(b + 8);


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
        i0 += 16;

        const __m256 vk0x01234567 = _mm256_load_ps(w);
        const __m256 vk0x89ABCDEF = _mm256_load_ps(w + 8);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi0x89ABCDEF, vk0x89ABCDEF));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
        i1 += 16;

        const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
        const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 24);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);
        __m256 vacc89ABCDEFp1 = _mm256_mul_ps(vi1x89ABCDEF, vk1x89ABCDEF);

        const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
        const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
        i2 += 16;

        const __m256 vk2x01234567 = _mm256_load_ps(w + 32);
        const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 40);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi2x89ABCDEF, vk2x89ABCDEF));

        const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
        const __m256 vi3x89ABCDEF = _mm256_loadu_ps(i3 + 8);
        i3 += 16;

        const __m256 vk3x01234567 = _mm256_load_ps(w + 48);
        const __m256 vk3x89ABCDEF = _mm256_load_ps(w + 56);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi3x01234567, vk3x01234567));
        vacc89ABCDEFp1 = _mm256_add_ps(vacc89ABCDEFp1, _mm256_mul_ps(vi3x89ABCDEF, vk3x89ABCDEF));

        const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
        const __m256 vi4x89ABCDEF = _mm256_loadu_ps(i4 + 8);
        i4 += 16;

        const __m256 vk4x01234567 = _mm256_load_ps(w + 64);
        const __m256 vk4x89ABCDEF = _mm256_load_ps(w + 72);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi4x89ABCDEF, vk4x89ABCDEF));

        const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
        const __m256 vi5x89ABCDEF = _mm256_loadu_ps(i5 + 8);
        i5 += 16;

        const __m256 vk5x01234567 = _mm256_load_ps(w + 80);
        const __m256 vk5x89ABCDEF = _mm256_load_ps(w + 88);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi5x01234567, vk5x01234567));
        vacc89ABCDEFp1 = _mm256_add_ps(vacc89ABCDEFp1, _mm256_mul_ps(vi5x89ABCDEF, vk5x89ABCDEF));

        const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
        const __m256 vi6x89ABCDEF = _mm256_loadu_ps(i6 + 8);
        i6 += 16;

        const __m256 vk6x01234567 = _mm256_load_ps(w + 96);
        const __m256 vk6x89ABCDEF = _mm256_load_ps(w + 104);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi6x89ABCDEF, vk6x89ABCDEF));

        const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
        const __m256 vi7x89ABCDEF = _mm256_loadu_ps(i7 + 8);
        i7 += 16;

        const __m256 vk7x01234567 = _mm256_load_ps(w + 112);
        const __m256 vk7x89ABCDEF = _mm256_load_ps(w + 120);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi7x01234567, vk7x01234567));
        vacc89ABCDEFp1 = _mm256_add_ps(vacc89ABCDEFp1, _mm256_mul_ps(vi7x89ABCDEF, vk7x89ABCDEF));

        w += 128;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, vacc89ABCDEFp1);

        _mm256_store_ps(b, vacc01234567p0);
        _mm256_store_ps(b + 8, vacc89ABCDEFp0);
        b += 16;
      }

      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_load_ps(b);


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        i0 += 8;

        const __m256 vk0x01234567 = _mm256_load_ps(w);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        i1 += 8;

        const __m256 vk1x01234567 = _mm256_load_ps(w + 8);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
        i2 += 8;

        const __m256 vk2x01234567 = _mm256_load_ps(w + 16);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

        const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
        i3 += 8;

        const __m256 vk3x01234567 = _mm256_load_ps(w + 24);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi3x01234567, vk3x01234567));

        const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
        i4 += 8;

        const __m256 vk4x01234567 = _mm256_load_ps(w + 32);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));

        const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
        i5 += 8;

        const __m256 vk5x01234567 = _mm256_load_ps(w + 40);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi5x01234567, vk5x01234567));

        const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
        i6 += 8;

        const __m256 vk6x01234567 = _mm256_load_ps(w + 48);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));

        const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
        i7 += 8;

        const __m256 vk7x01234567 = _mm256_load_ps(w + 56);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi7x01234567, vk7x01234567));

        w += 64;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        _mm256_store_ps(b, vacc01234567p0);
        b += 8;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);
        __m256 vacc01234567p0 = _mm256_load_ps(b);


        const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);

        const __m256 vk0x01234567 = _mm256_load_ps(w);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);

        const __m256 vk1x01234567 = _mm256_load_ps(w + 8);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);

        const __m256 vk2x01234567 = _mm256_load_ps(w + 16);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

        const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);

        const __m256 vk3x01234567 = _mm256_load_ps(w + 24);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi3x01234567, vk3x01234567));

        const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);

        const __m256 vk4x01234567 = _mm256_load_ps(w + 32);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));

        const __m256 vi5x01234567 = _mm256_maskload_ps(i5, vmask);

        const __m256 vk5x01234567 = _mm256_load_ps(w + 40);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi5x01234567, vk5x01234567));

        const __m256 vi6x01234567 = _mm256_maskload_ps(i6, vmask);

        const __m256 vk6x01234567 = _mm256_load_ps(w + 48);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));

        const __m256 vi7x01234567 = _mm256_maskload_ps(i7, vmask);

        const __m256 vk7x01234567 = _mm256_load_ps(w + 56);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi7x01234567, vk7x01234567));

        w += 64;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        _mm256_store_ps(b, vacc01234567p0);
      }
    }

    // Last pass to process up to 9 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = input[5];
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = input[6];
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = input[7];
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      const float* i8 = input[8];
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const float*) ((uintptr_t) i8 + input_offset);
      }

      size_t c = channels;
      for (; c >= 16; c -= 16) {
        __m256 vacc01234567p0 = _mm256_load_ps(b);
        __m256 vacc89ABCDEFp0 = _mm256_load_ps(b + 8);
        b += 16;


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
        i0 += 16;

        __m256 vk0x01234567 = _mm256_load_ps(w);
        __m256 vk0x89ABCDEF = _mm256_load_ps(w + 8);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi0x89ABCDEF, vk0x89ABCDEF));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
        i1 += 16;

        __m256 vk1x01234567 = _mm256_load_ps(w + 16);
        __m256 vk1x89ABCDEF = _mm256_load_ps(w + 24);

        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);
        __m256 vacc89ABCDEFp1 = _mm256_mul_ps(vi1x89ABCDEF, vk1x89ABCDEF);

        const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
        const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
        i2 += 16;

        __m256 vk2x01234567 = _mm256_load_ps(w + 32);
        __m256 vk2x89ABCDEF = _mm256_load_ps(w + 40);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi2x89ABCDEF, vk2x89ABCDEF));

        const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
        const __m256 vi3x89ABCDEF = _mm256_loadu_ps(i3 + 8);
        i3 += 16;

        __m256 vk3x01234567 = _mm256_load_ps(w + 48);
        __m256 vk3x89ABCDEF = _mm256_load_ps(w + 56);

        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi3x01234567, vk3x01234567));
        vacc89ABCDEFp1 = _mm256_add_ps(vacc89ABCDEFp1, _mm256_mul_ps(vi3x89ABCDEF, vk3x89ABCDEF));

        const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
        const __m256 vi4x89ABCDEF = _mm256_loadu_ps(i4 + 8);
        i4 += 16;

        __m256 vk4x01234567 = _mm256_load_ps(w + 64);
        __m256 vk4x89ABCDEF = _mm256_load_ps(w + 72);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi4x89ABCDEF, vk4x89ABCDEF));

        const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
        const __m256 vi5x89ABCDEF = _mm256_loadu_ps(i5 + 8);
        i5 += 16;

        __m256 vk5x01234567 = _mm256_load_ps(w + 80);
        __m256 vk5x89ABCDEF = _mm256_load_ps(w + 88);

        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi5x01234567, vk5x01234567));
        vacc89ABCDEFp1 = _mm256_add_ps(vacc89ABCDEFp1, _mm256_mul_ps(vi5x89ABCDEF, vk5x89ABCDEF));

        const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
        const __m256 vi6x89ABCDEF = _mm256_loadu_ps(i6 + 8);
        i6 += 16;

        __m256 vk6x01234567 = _mm256_load_ps(w + 96);
        __m256 vk6x89ABCDEF = _mm256_load_ps(w + 104);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi6x89ABCDEF, vk6x89ABCDEF));

        const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
        const __m256 vi7x89ABCDEF = _mm256_loadu_ps(i7 + 8);
        i7 += 16;

        __m256 vk7x01234567 = _mm256_load_ps(w + 112);
        __m256 vk7x89ABCDEF = _mm256_load_ps(w + 120);

        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi7x01234567, vk7x01234567));
        vacc89ABCDEFp1 = _mm256_add_ps(vacc89ABCDEFp1, _mm256_mul_ps(vi7x89ABCDEF, vk7x89ABCDEF));

        const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
        const __m256 vi8x89ABCDEF = _mm256_loadu_ps(i8 + 8);
        i8 += 16;

        __m256 vk8x01234567 = _mm256_load_ps(w + 128);
        __m256 vk8x89ABCDEF = _mm256_load_ps(w + 136);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi8x01234567, vk8x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi8x89ABCDEF, vk8x89ABCDEF));

        w += 144;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, vacc89ABCDEFp1);

        __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
        __m256 vacc89ABCDEF = _mm256_max_ps(vmin, vacc89ABCDEFp0);

        vacc01234567 = _mm256_min_ps(vmax, vacc01234567);
        vacc89ABCDEF = _mm256_min_ps(vmax, vacc89ABCDEF);

        _mm256_storeu_ps(output, vacc01234567);
        _mm256_storeu_ps(output + 8, vacc89ABCDEF);
        output += 16;
      }


      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_load_ps(b);
        b += 8;


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        i0 += 8;

        __m256 vk0x01234567 = _mm256_load_ps(w);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        i1 += 8;

        __m256 vk1x01234567 = _mm256_load_ps(w + 8);

        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
        i2 += 8;

        __m256 vk2x01234567 = _mm256_load_ps(w + 16);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

        const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
        i3 += 8;

        __m256 vk3x01234567 = _mm256_load_ps(w + 24);

        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi3x01234567, vk3x01234567));

        const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
        i4 += 8;

        __m256 vk4x01234567 = _mm256_load_ps(w + 32);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));

        const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
        i5 += 8;

        __m256 vk5x01234567 = _mm256_load_ps(w + 40);

        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi5x01234567, vk5x01234567));

        const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
        i6 += 8;

        __m256 vk6x01234567 = _mm256_load_ps(w + 48);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));

        const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
        i7 += 8;

        __m256 vk7x01234567 = _mm256_load_ps(w + 56);

        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi7x01234567, vk7x01234567));

        const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
        i8 += 8;

        __m256 vk8x01234567 = _mm256_load_ps(w + 64);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi8x01234567, vk8x01234567));

        w += 72;


        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);

        vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

        _mm256_storeu_ps(output, vacc01234567);
        output += 8;
      }

      if XNN_UNLIKELY(c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        __m256 vacc01234567p0 = _mm256_load_ps(b);
        const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);

        const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
        __m256 vk0x01234567 = _mm256_load_ps(w);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
        __m256 vk1x01234567 = _mm256_load_ps(w + 8);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
        __m256 vk2x01234567 = _mm256_load_ps(w + 16);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

        const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
        __m256 vk3x01234567 = _mm256_load_ps(w + 24);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi3x01234567, vk3x01234567));

        const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);
        __m256 vk4x01234567 = _mm256_load_ps(w + 32);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi4x01234567, vk4x01234567));

        const __m256 vi5x01234567 = _mm256_maskload_ps(i5, vmask);
        __m256 vk5x01234567 = _mm256_load_ps(w + 40);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi5x01234567, vk5x01234567));

        const __m256 vi6x01234567 = _mm256_maskload_ps(i6, vmask);
        __m256 vk6x01234567 = _mm256_load_ps(w + 48);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi6x01234567, vk6x01234567));

        const __m256 vi7x01234567 = _mm256_maskload_ps(i7, vmask);
        __m256 vk7x01234567 = _mm256_load_ps(w + 56);
        vacc01234567p1 = _mm256_add_ps(vacc01234567p1, _mm256_mul_ps(vi7x01234567, vk7x01234567));

        const __m256 vi8x01234567 = _mm256_maskload_ps(i8, vmask);
        __m256 vk8x01234567 = _mm256_load_ps(w + 64);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi8x01234567, vk8x01234567));

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
        vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

        __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
        if (c & 4) {
          _mm_storeu_ps(output, vacc0123);
          vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
          output += 4;
        }
        if (c & 2) {
          _mm_storel_pi((__m64*) output, vacc0123);
          vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
          output += 2;
        }
        if (c & 1) {
          _mm_store_ss(output, vacc0123);
          output += 1;
        }
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
