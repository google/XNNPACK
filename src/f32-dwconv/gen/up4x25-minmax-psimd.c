// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/up-psimd.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/dwconv.h>


void xnn_f32_dwconv_minmax_ukernel_up4x25__psimd(
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

  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);
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
      psimd_f32 vacc0123p0 = psimd_load_f32(w);


      const psimd_f32 vi0x0123 = psimd_load_f32(i0);
      i0 += 4;

      const psimd_f32 vk0x0123 = psimd_load_f32(w + 4);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi0x0123, vk0x0123);

      const psimd_f32 vi1x0123 = psimd_load_f32(i1);
      i1 += 4;

      const psimd_f32 vk1x0123 = psimd_load_f32(w + 8);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi1x0123, vk1x0123);

      const psimd_f32 vi2x0123 = psimd_load_f32(i2);
      i2 += 4;

      const psimd_f32 vk2x0123 = psimd_load_f32(w + 12);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi2x0123, vk2x0123);

      const psimd_f32 vi3x0123 = psimd_load_f32(i3);
      i3 += 4;

      const psimd_f32 vk3x0123 = psimd_load_f32(w + 16);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi3x0123, vk3x0123);

      const psimd_f32 vi4x0123 = psimd_load_f32(i4);
      i4 += 4;

      const psimd_f32 vk4x0123 = psimd_load_f32(w + 20);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi4x0123, vk4x0123);

      const psimd_f32 vi5x0123 = psimd_load_f32(i5);
      i5 += 4;

      const psimd_f32 vk5x0123 = psimd_load_f32(w + 24);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi5x0123, vk5x0123);

      const psimd_f32 vi6x0123 = psimd_load_f32(i6);
      i6 += 4;

      const psimd_f32 vk6x0123 = psimd_load_f32(w + 28);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi6x0123, vk6x0123);

      const psimd_f32 vi7x0123 = psimd_load_f32(i7);
      i7 += 4;

      const psimd_f32 vk7x0123 = psimd_load_f32(w + 32);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi7x0123, vk7x0123);

      const psimd_f32 vi8x0123 = psimd_load_f32(i8);
      i8 += 4;

      const psimd_f32 vk8x0123 = psimd_load_f32(w + 36);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi8x0123, vk8x0123);

      const psimd_f32 vi9x0123 = psimd_load_f32(i9);
      i9 += 4;

      const psimd_f32 vk9x0123 = psimd_load_f32(w + 40);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi9x0123, vk9x0123);

      const psimd_f32 vi10x0123 = psimd_load_f32(i10);
      i10 += 4;

      const psimd_f32 vk10x0123 = psimd_load_f32(w + 44);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi10x0123, vk10x0123);

      const psimd_f32 vi11x0123 = psimd_load_f32(i11);
      i11 += 4;

      const psimd_f32 vk11x0123 = psimd_load_f32(w + 48);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi11x0123, vk11x0123);

      const psimd_f32 vi12x0123 = psimd_load_f32(i12);
      i12 += 4;

      const psimd_f32 vk12x0123 = psimd_load_f32(w + 52);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi12x0123, vk12x0123);

      const psimd_f32 vi13x0123 = psimd_load_f32(i13);
      i13 += 4;

      const psimd_f32 vk13x0123 = psimd_load_f32(w + 56);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi13x0123, vk13x0123);

      const psimd_f32 vi14x0123 = psimd_load_f32(i14);
      i14 += 4;

      const psimd_f32 vk14x0123 = psimd_load_f32(w + 60);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi14x0123, vk14x0123);

      const psimd_f32 vi15x0123 = psimd_load_f32(i15);
      i15 += 4;

      const psimd_f32 vk15x0123 = psimd_load_f32(w + 64);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi15x0123, vk15x0123);

      const psimd_f32 vi16x0123 = psimd_load_f32(i16);
      i16 += 4;

      const psimd_f32 vk16x0123 = psimd_load_f32(w + 68);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi16x0123, vk16x0123);

      const psimd_f32 vi17x0123 = psimd_load_f32(i17);
      i17 += 4;

      const psimd_f32 vk17x0123 = psimd_load_f32(w + 72);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi17x0123, vk17x0123);

      const psimd_f32 vi18x0123 = psimd_load_f32(i18);
      i18 += 4;

      const psimd_f32 vk18x0123 = psimd_load_f32(w + 76);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi18x0123, vk18x0123);

      const psimd_f32 vi19x0123 = psimd_load_f32(i19);
      i19 += 4;

      const psimd_f32 vk19x0123 = psimd_load_f32(w + 80);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi19x0123, vk19x0123);

      const psimd_f32 vi20x0123 = psimd_load_f32(i20);
      i20 += 4;

      const psimd_f32 vk20x0123 = psimd_load_f32(w + 84);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi20x0123, vk20x0123);

      const psimd_f32 vi21x0123 = psimd_load_f32(i21);
      i21 += 4;

      const psimd_f32 vk21x0123 = psimd_load_f32(w + 88);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi21x0123, vk21x0123);

      const psimd_f32 vi22x0123 = psimd_load_f32(i22);
      i22 += 4;

      const psimd_f32 vk22x0123 = psimd_load_f32(w + 92);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi22x0123, vk22x0123);

      const psimd_f32 vi23x0123 = psimd_load_f32(i23);
      i23 += 4;

      const psimd_f32 vk23x0123 = psimd_load_f32(w + 96);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi23x0123, vk23x0123);

      const psimd_f32 vi24x0123 = psimd_load_f32(i24);
      i24 += 4;

      const psimd_f32 vk24x0123 = psimd_load_f32(w + 100);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi24x0123, vk24x0123);

      w += 104;


      psimd_f32 vacc0123 = psimd_max_f32(vacc0123p0, vmin);
      vacc0123 = psimd_min_f32(vacc0123, vmax);

      psimd_store_f32(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      psimd_f32 vacc0123p0 = psimd_load_f32(w);

      const psimd_f32 vi0x0123 = psimd_load_f32(i0);
      const psimd_f32 vk0x0123 = psimd_load_f32(w + 4);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi0x0123, vk0x0123);

      const psimd_f32 vi1x0123 = psimd_load_f32(i1);
      const psimd_f32 vk1x0123 = psimd_load_f32(w + 8);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi1x0123, vk1x0123);

      const psimd_f32 vi2x0123 = psimd_load_f32(i2);
      const psimd_f32 vk2x0123 = psimd_load_f32(w + 12);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi2x0123, vk2x0123);

      const psimd_f32 vi3x0123 = psimd_load_f32(i3);
      const psimd_f32 vk3x0123 = psimd_load_f32(w + 16);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi3x0123, vk3x0123);

      const psimd_f32 vi4x0123 = psimd_load_f32(i4);
      const psimd_f32 vk4x0123 = psimd_load_f32(w + 20);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi4x0123, vk4x0123);

      const psimd_f32 vi5x0123 = psimd_load_f32(i5);
      const psimd_f32 vk5x0123 = psimd_load_f32(w + 24);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi5x0123, vk5x0123);

      const psimd_f32 vi6x0123 = psimd_load_f32(i6);
      const psimd_f32 vk6x0123 = psimd_load_f32(w + 28);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi6x0123, vk6x0123);

      const psimd_f32 vi7x0123 = psimd_load_f32(i7);
      const psimd_f32 vk7x0123 = psimd_load_f32(w + 32);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi7x0123, vk7x0123);

      const psimd_f32 vi8x0123 = psimd_load_f32(i8);
      const psimd_f32 vk8x0123 = psimd_load_f32(w + 36);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi8x0123, vk8x0123);

      const psimd_f32 vi9x0123 = psimd_load_f32(i9);
      const psimd_f32 vk9x0123 = psimd_load_f32(w + 40);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi9x0123, vk9x0123);

      const psimd_f32 vi10x0123 = psimd_load_f32(i10);
      const psimd_f32 vk10x0123 = psimd_load_f32(w + 44);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi10x0123, vk10x0123);

      const psimd_f32 vi11x0123 = psimd_load_f32(i11);
      const psimd_f32 vk11x0123 = psimd_load_f32(w + 48);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi11x0123, vk11x0123);

      const psimd_f32 vi12x0123 = psimd_load_f32(i12);
      const psimd_f32 vk12x0123 = psimd_load_f32(w + 52);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi12x0123, vk12x0123);

      const psimd_f32 vi13x0123 = psimd_load_f32(i13);
      const psimd_f32 vk13x0123 = psimd_load_f32(w + 56);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi13x0123, vk13x0123);

      const psimd_f32 vi14x0123 = psimd_load_f32(i14);
      const psimd_f32 vk14x0123 = psimd_load_f32(w + 60);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi14x0123, vk14x0123);

      const psimd_f32 vi15x0123 = psimd_load_f32(i15);
      const psimd_f32 vk15x0123 = psimd_load_f32(w + 64);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi15x0123, vk15x0123);

      const psimd_f32 vi16x0123 = psimd_load_f32(i16);
      const psimd_f32 vk16x0123 = psimd_load_f32(w + 68);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi16x0123, vk16x0123);

      const psimd_f32 vi17x0123 = psimd_load_f32(i17);
      const psimd_f32 vk17x0123 = psimd_load_f32(w + 72);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi17x0123, vk17x0123);

      const psimd_f32 vi18x0123 = psimd_load_f32(i18);
      const psimd_f32 vk18x0123 = psimd_load_f32(w + 76);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi18x0123, vk18x0123);

      const psimd_f32 vi19x0123 = psimd_load_f32(i19);
      const psimd_f32 vk19x0123 = psimd_load_f32(w + 80);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi19x0123, vk19x0123);

      const psimd_f32 vi20x0123 = psimd_load_f32(i20);
      const psimd_f32 vk20x0123 = psimd_load_f32(w + 84);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi20x0123, vk20x0123);

      const psimd_f32 vi21x0123 = psimd_load_f32(i21);
      const psimd_f32 vk21x0123 = psimd_load_f32(w + 88);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi21x0123, vk21x0123);

      const psimd_f32 vi22x0123 = psimd_load_f32(i22);
      const psimd_f32 vk22x0123 = psimd_load_f32(w + 92);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi22x0123, vk22x0123);

      const psimd_f32 vi23x0123 = psimd_load_f32(i23);
      const psimd_f32 vk23x0123 = psimd_load_f32(w + 96);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi23x0123, vk23x0123);

      const psimd_f32 vi24x0123 = psimd_load_f32(i24);
      const psimd_f32 vk24x0123 = psimd_load_f32(w + 100);
      vacc0123p0 = psimd_qfma_f32(vacc0123p0, vi24x0123, vk24x0123);


      psimd_f32 vacc0123 = psimd_max_f32(vacc0123p0, vmin);
      vacc0123 = psimd_min_f32(vacc0123, vmax);

      if (c & 2) {
        psimd_store2_f32(output, vacc0123);
        vacc0123 = psimd_concat_hi_f32(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        psimd_store1_f32(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
