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


void xnn_f32_dwconv_ukernel_up4x25__sse(
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
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    const float* i9 = input[9];
    const float* i10 = input[10];
    const float* i11 = input[11];
    const float* i12 = input[12];
    const float* i13 = input[13];
    const float* i14 = input[14];
    const float* i15 = input[15];
    const float* i16 = input[16];
    const float* i17 = input[17];
    const float* i18 = input[18];
    const float* i19 = input[19];
    const float* i20 = input[20];
    const float* i21 = input[21];
    const float* i22 = input[22];
    const float* i23 = input[23];
    const float* i24 = input[24];
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

      const __m128 vi4 = _mm_loadu_ps(i4);
      const __m128 vk4 = _mm_load_ps(w + 20);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi4, vk4));
      i4 += 4;

      const __m128 vi5 = _mm_loadu_ps(i5);
      const __m128 vk5 = _mm_load_ps(w + 24);
      vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(vi5, vk5));
      i5 += 4;

      const __m128 vi6 = _mm_loadu_ps(i6);
      const __m128 vk6 = _mm_load_ps(w + 28);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi6, vk6));
      i6 += 4;

      const __m128 vi7 = _mm_loadu_ps(i7);
      const __m128 vk7 = _mm_load_ps(w + 32);
      vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(vi7, vk7));
      i7 += 4;

      const __m128 vi8 = _mm_loadu_ps(i8);
      const __m128 vk8 = _mm_load_ps(w + 36);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi8, vk8));
      i8 += 4;

      const __m128 vi9 = _mm_loadu_ps(i9);
      const __m128 vk9 = _mm_load_ps(w + 40);
      vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(vi9, vk9));
      i9 += 4;

      const __m128 vi10 = _mm_loadu_ps(i10);
      const __m128 vk10 = _mm_load_ps(w + 44);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi10, vk10));
      i10 += 4;

      const __m128 vi11 = _mm_loadu_ps(i11);
      const __m128 vk11 = _mm_load_ps(w + 48);
      vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(vi11, vk11));
      i11 += 4;

      const __m128 vi12 = _mm_loadu_ps(i12);
      const __m128 vk12 = _mm_load_ps(w + 52);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi12, vk12));
      i12 += 4;

      const __m128 vi13 = _mm_loadu_ps(i13);
      const __m128 vk13 = _mm_load_ps(w + 56);
      vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(vi13, vk13));
      i13 += 4;

      const __m128 vi14 = _mm_loadu_ps(i14);
      const __m128 vk14 = _mm_load_ps(w + 60);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi14, vk14));
      i14 += 4;

      const __m128 vi15 = _mm_loadu_ps(i15);
      const __m128 vk15 = _mm_load_ps(w + 64);
      vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(vi15, vk15));
      i15 += 4;

      const __m128 vi16 = _mm_loadu_ps(i16);
      const __m128 vk16 = _mm_load_ps(w + 68);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi16, vk16));
      i16 += 4;

      const __m128 vi17 = _mm_loadu_ps(i17);
      const __m128 vk17 = _mm_load_ps(w + 72);
      vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(vi17, vk17));
      i17 += 4;

      const __m128 vi18 = _mm_loadu_ps(i18);
      const __m128 vk18 = _mm_load_ps(w + 76);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi18, vk18));
      i18 += 4;

      const __m128 vi19 = _mm_loadu_ps(i19);
      const __m128 vk19 = _mm_load_ps(w + 80);
      vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(vi19, vk19));
      i19 += 4;

      const __m128 vi20 = _mm_loadu_ps(i20);
      const __m128 vk20 = _mm_load_ps(w + 84);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi20, vk20));
      i20 += 4;

      const __m128 vi21 = _mm_loadu_ps(i21);
      const __m128 vk21 = _mm_load_ps(w + 88);
      vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(vi21, vk21));
      i21 += 4;

      const __m128 vi22 = _mm_loadu_ps(i22);
      const __m128 vk22 = _mm_load_ps(w + 92);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi22, vk22));
      i22 += 4;

      const __m128 vi23 = _mm_loadu_ps(i23);
      const __m128 vk23 = _mm_load_ps(w + 96);
      vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(vi23, vk23));
      i23 += 4;

      const __m128 vi24 = _mm_loadu_ps(i24);
      const __m128 vk24 = _mm_load_ps(w + 100);
      vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(vi24, vk24));
      i24 += 4;

      w += 104;

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

      const __m128 vi4 = _mm_loadu_ps(i4);
      const __m128 vk4 = _mm_load_ps(w + 20);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi4, vk4));

      const __m128 vi5 = _mm_loadu_ps(i5);
      const __m128 vk5 = _mm_load_ps(w + 24);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi5, vk5));

      const __m128 vi6 = _mm_loadu_ps(i6);
      const __m128 vk6 = _mm_load_ps(w + 28);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi6, vk6));

      const __m128 vi7 = _mm_loadu_ps(i7);
      const __m128 vk7 = _mm_load_ps(w + 32);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi7, vk7));

      const __m128 vi8 = _mm_loadu_ps(i8);
      const __m128 vk8 = _mm_load_ps(w + 36);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi8, vk8));

      const __m128 vi9 = _mm_loadu_ps(i9);
      const __m128 vk9 = _mm_load_ps(w + 40);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi9, vk9));

      const __m128 vi10 = _mm_loadu_ps(i10);
      const __m128 vk10 = _mm_load_ps(w + 44);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi10, vk10));

      const __m128 vi11 = _mm_loadu_ps(i11);
      const __m128 vk11 = _mm_load_ps(w + 48);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi11, vk11));

      const __m128 vi12 = _mm_loadu_ps(i12);
      const __m128 vk12 = _mm_load_ps(w + 52);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi12, vk12));

      const __m128 vi13 = _mm_loadu_ps(i13);
      const __m128 vk13 = _mm_load_ps(w + 56);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi13, vk13));

      const __m128 vi14 = _mm_loadu_ps(i14);
      const __m128 vk14 = _mm_load_ps(w + 60);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi14, vk14));

      const __m128 vi15 = _mm_loadu_ps(i15);
      const __m128 vk15 = _mm_load_ps(w + 64);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi15, vk15));

      const __m128 vi16 = _mm_loadu_ps(i16);
      const __m128 vk16 = _mm_load_ps(w + 68);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi16, vk16));

      const __m128 vi17 = _mm_loadu_ps(i17);
      const __m128 vk17 = _mm_load_ps(w + 72);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi17, vk17));

      const __m128 vi18 = _mm_loadu_ps(i18);
      const __m128 vk18 = _mm_load_ps(w + 76);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi18, vk18));

      const __m128 vi19 = _mm_loadu_ps(i19);
      const __m128 vk19 = _mm_load_ps(w + 80);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi19, vk19));

      const __m128 vi20 = _mm_loadu_ps(i20);
      const __m128 vk20 = _mm_load_ps(w + 84);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi20, vk20));

      const __m128 vi21 = _mm_loadu_ps(i21);
      const __m128 vk21 = _mm_load_ps(w + 88);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi21, vk21));

      const __m128 vi22 = _mm_loadu_ps(i22);
      const __m128 vk22 = _mm_load_ps(w + 92);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi22, vk22));

      const __m128 vi23 = _mm_loadu_ps(i23);
      const __m128 vk23 = _mm_load_ps(w + 96);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi23, vk23));

      const __m128 vi24 = _mm_loadu_ps(i24);
      const __m128 vk24 = _mm_load_ps(w + 100);
      vacc = _mm_add_ps(vacc, _mm_mul_ps(vi24, vk24));

      w += 104;

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
