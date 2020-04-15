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


void xnn_f32_dwconv_minmax_ukernel_up4x25__sse(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m128 vmax = _mm_load_ps(params->sse.max);
  const __m128 vmin = _mm_load_ps(params->sse.min);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    const float* i1 = input[1];
    assert(i1 != NULL);
    const float* i2 = input[2];
    assert(i2 != NULL);
    const float* i3 = input[3];
    assert(i3 != NULL);
    const float* i4 = input[4];
    assert(i4 != NULL);
    const float* i5 = input[5];
    assert(i5 != NULL);
    const float* i6 = input[6];
    assert(i6 != NULL);
    const float* i7 = input[7];
    assert(i7 != NULL);
    const float* i8 = input[8];
    assert(i8 != NULL);
    const float* i9 = input[9];
    assert(i9 != NULL);
    const float* i10 = input[10];
    assert(i10 != NULL);
    const float* i11 = input[11];
    assert(i11 != NULL);
    const float* i12 = input[12];
    assert(i12 != NULL);
    const float* i13 = input[13];
    assert(i13 != NULL);
    const float* i14 = input[14];
    assert(i14 != NULL);
    const float* i15 = input[15];
    assert(i15 != NULL);
    const float* i16 = input[16];
    assert(i16 != NULL);
    const float* i17 = input[17];
    assert(i17 != NULL);
    const float* i18 = input[18];
    assert(i18 != NULL);
    const float* i19 = input[19];
    assert(i19 != NULL);
    const float* i20 = input[20];
    assert(i20 != NULL);
    const float* i21 = input[21];
    assert(i21 != NULL);
    const float* i22 = input[22];
    assert(i22 != NULL);
    const float* i23 = input[23];
    assert(i23 != NULL);
    const float* i24 = input[24];
    assert(i24 != NULL);
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 4; c -= 4) {
      __m128 vacc0123p0 = _mm_load_ps(w);


      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 += 4;

      const __m128 vk0x0123 = _mm_load_ps(w + 4);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 += 4;

      const __m128 vk1x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      i2 += 4;

      const __m128 vk2x0123 = _mm_load_ps(w + 12);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      i3 += 4;

      const __m128 vk3x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

      const __m128 vi4x0123 = _mm_loadu_ps(i4);
      i4 += 4;

      const __m128 vk4x0123 = _mm_load_ps(w + 20);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

      const __m128 vi5x0123 = _mm_loadu_ps(i5);
      i5 += 4;

      const __m128 vk5x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

      const __m128 vi6x0123 = _mm_loadu_ps(i6);
      i6 += 4;

      const __m128 vk6x0123 = _mm_load_ps(w + 28);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

      const __m128 vi7x0123 = _mm_loadu_ps(i7);
      i7 += 4;

      const __m128 vk7x0123 = _mm_load_ps(w + 32);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

      const __m128 vi8x0123 = _mm_loadu_ps(i8);
      i8 += 4;

      const __m128 vk8x0123 = _mm_load_ps(w + 36);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));

      const __m128 vi9x0123 = _mm_loadu_ps(i9);
      i9 += 4;

      const __m128 vk9x0123 = _mm_load_ps(w + 40);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi9x0123, vk9x0123));

      const __m128 vi10x0123 = _mm_loadu_ps(i10);
      i10 += 4;

      const __m128 vk10x0123 = _mm_load_ps(w + 44);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi10x0123, vk10x0123));

      const __m128 vi11x0123 = _mm_loadu_ps(i11);
      i11 += 4;

      const __m128 vk11x0123 = _mm_load_ps(w + 48);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi11x0123, vk11x0123));

      const __m128 vi12x0123 = _mm_loadu_ps(i12);
      i12 += 4;

      const __m128 vk12x0123 = _mm_load_ps(w + 52);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi12x0123, vk12x0123));

      const __m128 vi13x0123 = _mm_loadu_ps(i13);
      i13 += 4;

      const __m128 vk13x0123 = _mm_load_ps(w + 56);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi13x0123, vk13x0123));

      const __m128 vi14x0123 = _mm_loadu_ps(i14);
      i14 += 4;

      const __m128 vk14x0123 = _mm_load_ps(w + 60);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi14x0123, vk14x0123));

      const __m128 vi15x0123 = _mm_loadu_ps(i15);
      i15 += 4;

      const __m128 vk15x0123 = _mm_load_ps(w + 64);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi15x0123, vk15x0123));

      const __m128 vi16x0123 = _mm_loadu_ps(i16);
      i16 += 4;

      const __m128 vk16x0123 = _mm_load_ps(w + 68);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi16x0123, vk16x0123));

      const __m128 vi17x0123 = _mm_loadu_ps(i17);
      i17 += 4;

      const __m128 vk17x0123 = _mm_load_ps(w + 72);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi17x0123, vk17x0123));

      const __m128 vi18x0123 = _mm_loadu_ps(i18);
      i18 += 4;

      const __m128 vk18x0123 = _mm_load_ps(w + 76);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi18x0123, vk18x0123));

      const __m128 vi19x0123 = _mm_loadu_ps(i19);
      i19 += 4;

      const __m128 vk19x0123 = _mm_load_ps(w + 80);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi19x0123, vk19x0123));

      const __m128 vi20x0123 = _mm_loadu_ps(i20);
      i20 += 4;

      const __m128 vk20x0123 = _mm_load_ps(w + 84);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi20x0123, vk20x0123));

      const __m128 vi21x0123 = _mm_loadu_ps(i21);
      i21 += 4;

      const __m128 vk21x0123 = _mm_load_ps(w + 88);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi21x0123, vk21x0123));

      const __m128 vi22x0123 = _mm_loadu_ps(i22);
      i22 += 4;

      const __m128 vk22x0123 = _mm_load_ps(w + 92);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi22x0123, vk22x0123));

      const __m128 vi23x0123 = _mm_loadu_ps(i23);
      i23 += 4;

      const __m128 vk23x0123 = _mm_load_ps(w + 96);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi23x0123, vk23x0123));

      const __m128 vi24x0123 = _mm_loadu_ps(i24);
      i24 += 4;

      const __m128 vk24x0123 = _mm_load_ps(w + 100);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi24x0123, vk24x0123));

      w += 104;


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);

      _mm_storeu_ps(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128 vacc0123p0 = _mm_load_ps(w);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vk0x0123 = _mm_load_ps(w + 4);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vk1x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      const __m128 vk2x0123 = _mm_load_ps(w + 12);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      const __m128 vk3x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

      const __m128 vi4x0123 = _mm_loadu_ps(i4);
      const __m128 vk4x0123 = _mm_load_ps(w + 20);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

      const __m128 vi5x0123 = _mm_loadu_ps(i5);
      const __m128 vk5x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

      const __m128 vi6x0123 = _mm_loadu_ps(i6);
      const __m128 vk6x0123 = _mm_load_ps(w + 28);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

      const __m128 vi7x0123 = _mm_loadu_ps(i7);
      const __m128 vk7x0123 = _mm_load_ps(w + 32);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

      const __m128 vi8x0123 = _mm_loadu_ps(i8);
      const __m128 vk8x0123 = _mm_load_ps(w + 36);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));

      const __m128 vi9x0123 = _mm_loadu_ps(i9);
      const __m128 vk9x0123 = _mm_load_ps(w + 40);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi9x0123, vk9x0123));

      const __m128 vi10x0123 = _mm_loadu_ps(i10);
      const __m128 vk10x0123 = _mm_load_ps(w + 44);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi10x0123, vk10x0123));

      const __m128 vi11x0123 = _mm_loadu_ps(i11);
      const __m128 vk11x0123 = _mm_load_ps(w + 48);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi11x0123, vk11x0123));

      const __m128 vi12x0123 = _mm_loadu_ps(i12);
      const __m128 vk12x0123 = _mm_load_ps(w + 52);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi12x0123, vk12x0123));

      const __m128 vi13x0123 = _mm_loadu_ps(i13);
      const __m128 vk13x0123 = _mm_load_ps(w + 56);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi13x0123, vk13x0123));

      const __m128 vi14x0123 = _mm_loadu_ps(i14);
      const __m128 vk14x0123 = _mm_load_ps(w + 60);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi14x0123, vk14x0123));

      const __m128 vi15x0123 = _mm_loadu_ps(i15);
      const __m128 vk15x0123 = _mm_load_ps(w + 64);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi15x0123, vk15x0123));

      const __m128 vi16x0123 = _mm_loadu_ps(i16);
      const __m128 vk16x0123 = _mm_load_ps(w + 68);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi16x0123, vk16x0123));

      const __m128 vi17x0123 = _mm_loadu_ps(i17);
      const __m128 vk17x0123 = _mm_load_ps(w + 72);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi17x0123, vk17x0123));

      const __m128 vi18x0123 = _mm_loadu_ps(i18);
      const __m128 vk18x0123 = _mm_load_ps(w + 76);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi18x0123, vk18x0123));

      const __m128 vi19x0123 = _mm_loadu_ps(i19);
      const __m128 vk19x0123 = _mm_load_ps(w + 80);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi19x0123, vk19x0123));

      const __m128 vi20x0123 = _mm_loadu_ps(i20);
      const __m128 vk20x0123 = _mm_load_ps(w + 84);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi20x0123, vk20x0123));

      const __m128 vi21x0123 = _mm_loadu_ps(i21);
      const __m128 vk21x0123 = _mm_load_ps(w + 88);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi21x0123, vk21x0123));

      const __m128 vi22x0123 = _mm_loadu_ps(i22);
      const __m128 vk22x0123 = _mm_load_ps(w + 92);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi22x0123, vk22x0123));

      const __m128 vi23x0123 = _mm_loadu_ps(i23);
      const __m128 vk23x0123 = _mm_load_ps(w + 96);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi23x0123, vk23x0123));

      const __m128 vi24x0123 = _mm_loadu_ps(i24);
      const __m128 vk24x0123 = _mm_load_ps(w + 100);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi24x0123, vk24x0123));


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
