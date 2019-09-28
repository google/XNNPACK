// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/up-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/dwconv.h>


void xnn_f32_dwconv_ukernel_up4x4__sse(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m128 vmax = _mm_load_ps(params->sse.max);
  const __m128 vmin = _mm_load_ps(params->sse.min);
  do {
    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 4; c -= 4) {
      __m128 vacc0 = _mm_load_ps(w);

      const __m128 vi0 = _mm_loadu_ps(i0);
      const __m128 vk0 = _mm_load_ps(w + 4);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi0, vk0));
      i0 += 4;

      const __m128 vi1 = _mm_loadu_ps(i1);
      const __m128 vk1 = _mm_load_ps(w + 8);
      __m128 vacc1 = _mm_mul_ps(vi1, vk1);
      i1 += 4;

      const __m128 vi2 = _mm_loadu_ps(i2);
      const __m128 vk2 = _mm_load_ps(w + 12);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi2, vk2));
      i2 += 4;

      const __m128 vi3 = _mm_loadu_ps(i3);
      const __m128 vk3 = _mm_load_ps(w + 16);
      vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(vi3, vk3));
      i3 += 4;

      w += 20;

      vacc0 = _mm_add_ps(vacc0, vacc1);

      vacc0 = _mm_max_ps(vacc0, vmin);
      vacc0 = _mm_min_ps(vacc0, vmax);

      _mm_storeu_ps(output, vacc0);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128 vacc = _mm_load_ps(w);

      const __m128 vi0 = _mm_loadu_ps(i0);
      const __m128 vk0 = _mm_load_ps(w + 4);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi0, vk0));

      const __m128 vi1 = _mm_loadu_ps(i1);
      const __m128 vk1 = _mm_load_ps(w + 8);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi1, vk1));

      const __m128 vi2 = _mm_loadu_ps(i2);
      const __m128 vk2 = _mm_load_ps(w + 12);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi2, vk2));

      const __m128 vi3 = _mm_loadu_ps(i3);
      const __m128 vk3 = _mm_load_ps(w + 16);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi3, vk3));

      w += 20;

      vacc = _mm_max_ps(vacc, vmin);
      vacc = _mm_min_ps(vacc, vmax);

      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc);
        vacc = _mm_movehl_ps(vacc, vacc);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
