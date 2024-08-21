// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/unipass-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_4p8c__sse(
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

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
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
    for (; c >= 8; c -= 8) {
      __m128 vacc0123p0 = _mm_load_ps(w);
      __m128 vacc4567p0 = _mm_load_ps(w + 4);


      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
      i0 += 8;

      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      const __m128 vk0x4567 = _mm_load_ps(w + 12);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
      i1 += 8;

      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      const __m128 vk1x4567 = _mm_load_ps(w + 20);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      const __m128 vi2x4567 = _mm_loadu_ps(i2 + 4);
      i2 += 8;

      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      const __m128 vk2x4567 = _mm_load_ps(w + 28);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi2x4567, vk2x4567));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      const __m128 vi3x4567 = _mm_loadu_ps(i3 + 4);
      i3 += 8;

      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      const __m128 vk3x4567 = _mm_load_ps(w + 36);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi3x4567, vk3x4567));

      w += 40;


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      __m128 vacc4567 = _mm_max_ps(vacc4567p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);
      vacc4567 = _mm_min_ps(vacc4567, vmax);

      _mm_storeu_ps(output, vacc0123);
      _mm_storeu_ps(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      __m128 vacc0123p0 = _mm_load_ps(w);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 += 4;

      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 += 4;

      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      i2 += 4;

      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      i3 += 4;

      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

      w += 4;


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);

      _mm_storeu_ps(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128 vacc0123p0 = _mm_load_ps(w);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);

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
