// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/unipass-avx.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_4p16c__avx(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vmin = _mm256_set1_ps(params->scalar.min);
  const __m256 vmax = _mm256_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
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
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
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
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi1x89ABCDEF, vk1x89ABCDEF));

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
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi3x01234567, vk3x01234567));
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi3x89ABCDEF, vk3x89ABCDEF));

      w += 80;


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      __m256 vacc89ABCDEF = _mm256_max_ps(vmin, vacc89ABCDEFp0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);
      vacc89ABCDEF = _mm256_min_ps(vmax, vacc89ABCDEF);

      _mm256_storeu_ps(output, vacc01234567);
      _mm256_storeu_ps(output + 8, vacc89ABCDEF);
      output += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi3x01234567, vk3x01234567));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

      _mm256_storeu_ps(output, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);
      const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);

      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi1x01234567, vk1x01234567));

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi2x01234567, vk2x01234567));

      const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi3x01234567, vk3x01234567));


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

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
