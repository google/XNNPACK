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


void xnn_f32_dwconv_ukernel_up4x25__psimd(
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

  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);
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
      psimd_f32 vacc0 = psimd_load_f32(w);

      const psimd_f32 vi0 = psimd_load_f32(i0);
      const psimd_f32 vk0 = psimd_load_f32(w + 4);
      vacc0 = psimd_qfma_f32(vacc0, vi0, vk0);
      i0 += 4;

      const psimd_f32 vi1 = psimd_load_f32(i1);
      const psimd_f32 vk1 = psimd_load_f32(w + 8);
      psimd_f32 vacc1 = psimd_mul_f32(vi1, vk1);
      i1 += 4;

      const psimd_f32 vi2 = psimd_load_f32(i2);
      const psimd_f32 vk2 = psimd_load_f32(w + 12);
      vacc0 = psimd_qfma_f32(vacc0, vi2, vk2);
      i2 += 4;

      const psimd_f32 vi3 = psimd_load_f32(i3);
      const psimd_f32 vk3 = psimd_load_f32(w + 16);
      vacc1 = psimd_qfma_f32(vacc1, vi3, vk3);
      i3 += 4;

      const psimd_f32 vi4 = psimd_load_f32(i4);
      const psimd_f32 vk4 = psimd_load_f32(w + 20);
      vacc0 = psimd_qfma_f32(vacc0, vi4, vk4);
      i4 += 4;

      const psimd_f32 vi5 = psimd_load_f32(i5);
      const psimd_f32 vk5 = psimd_load_f32(w + 24);
      vacc1 = psimd_qfma_f32(vacc1, vi5, vk5);
      i5 += 4;

      const psimd_f32 vi6 = psimd_load_f32(i6);
      const psimd_f32 vk6 = psimd_load_f32(w + 28);
      vacc0 = psimd_qfma_f32(vacc0, vi6, vk6);
      i6 += 4;

      const psimd_f32 vi7 = psimd_load_f32(i7);
      const psimd_f32 vk7 = psimd_load_f32(w + 32);
      vacc1 = psimd_qfma_f32(vacc1, vi7, vk7);
      i7 += 4;

      const psimd_f32 vi8 = psimd_load_f32(i8);
      const psimd_f32 vk8 = psimd_load_f32(w + 36);
      vacc0 = psimd_qfma_f32(vacc0, vi8, vk8);
      i8 += 4;

      const psimd_f32 vi9 = psimd_load_f32(i9);
      const psimd_f32 vk9 = psimd_load_f32(w + 40);
      vacc1 = psimd_qfma_f32(vacc1, vi9, vk9);
      i9 += 4;

      const psimd_f32 vi10 = psimd_load_f32(i10);
      const psimd_f32 vk10 = psimd_load_f32(w + 44);
      vacc0 = psimd_qfma_f32(vacc0, vi10, vk10);
      i10 += 4;

      const psimd_f32 vi11 = psimd_load_f32(i11);
      const psimd_f32 vk11 = psimd_load_f32(w + 48);
      vacc1 = psimd_qfma_f32(vacc1, vi11, vk11);
      i11 += 4;

      const psimd_f32 vi12 = psimd_load_f32(i12);
      const psimd_f32 vk12 = psimd_load_f32(w + 52);
      vacc0 = psimd_qfma_f32(vacc0, vi12, vk12);
      i12 += 4;

      const psimd_f32 vi13 = psimd_load_f32(i13);
      const psimd_f32 vk13 = psimd_load_f32(w + 56);
      vacc1 = psimd_qfma_f32(vacc1, vi13, vk13);
      i13 += 4;

      const psimd_f32 vi14 = psimd_load_f32(i14);
      const psimd_f32 vk14 = psimd_load_f32(w + 60);
      vacc0 = psimd_qfma_f32(vacc0, vi14, vk14);
      i14 += 4;

      const psimd_f32 vi15 = psimd_load_f32(i15);
      const psimd_f32 vk15 = psimd_load_f32(w + 64);
      vacc1 = psimd_qfma_f32(vacc1, vi15, vk15);
      i15 += 4;

      const psimd_f32 vi16 = psimd_load_f32(i16);
      const psimd_f32 vk16 = psimd_load_f32(w + 68);
      vacc0 = psimd_qfma_f32(vacc0, vi16, vk16);
      i16 += 4;

      const psimd_f32 vi17 = psimd_load_f32(i17);
      const psimd_f32 vk17 = psimd_load_f32(w + 72);
      vacc1 = psimd_qfma_f32(vacc1, vi17, vk17);
      i17 += 4;

      const psimd_f32 vi18 = psimd_load_f32(i18);
      const psimd_f32 vk18 = psimd_load_f32(w + 76);
      vacc0 = psimd_qfma_f32(vacc0, vi18, vk18);
      i18 += 4;

      const psimd_f32 vi19 = psimd_load_f32(i19);
      const psimd_f32 vk19 = psimd_load_f32(w + 80);
      vacc1 = psimd_qfma_f32(vacc1, vi19, vk19);
      i19 += 4;

      const psimd_f32 vi20 = psimd_load_f32(i20);
      const psimd_f32 vk20 = psimd_load_f32(w + 84);
      vacc0 = psimd_qfma_f32(vacc0, vi20, vk20);
      i20 += 4;

      const psimd_f32 vi21 = psimd_load_f32(i21);
      const psimd_f32 vk21 = psimd_load_f32(w + 88);
      vacc1 = psimd_qfma_f32(vacc1, vi21, vk21);
      i21 += 4;

      const psimd_f32 vi22 = psimd_load_f32(i22);
      const psimd_f32 vk22 = psimd_load_f32(w + 92);
      vacc0 = psimd_qfma_f32(vacc0, vi22, vk22);
      i22 += 4;

      const psimd_f32 vi23 = psimd_load_f32(i23);
      const psimd_f32 vk23 = psimd_load_f32(w + 96);
      vacc1 = psimd_qfma_f32(vacc1, vi23, vk23);
      i23 += 4;

      const psimd_f32 vi24 = psimd_load_f32(i24);
      const psimd_f32 vk24 = psimd_load_f32(w + 100);
      vacc0 = psimd_qfma_f32(vacc0, vi24, vk24);
      i24 += 4;

      w += 104;

      vacc0 = psimd_add_f32(vacc0, vacc1);

      vacc0 = psimd_max_f32(vacc0, vmin);
      vacc0 = psimd_min_f32(vacc0, vmax);

      psimd_store_f32(output, vacc0);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      psimd_f32 vacc = psimd_load_f32(w);

      const psimd_f32 vi0 = psimd_load_f32(i0);
      const psimd_f32 vk0 = psimd_load_f32(w + 4);
      vacc = psimd_qfma_f32(vacc, vi0, vk0);

      const psimd_f32 vi1 = psimd_load_f32(i1);
      const psimd_f32 vk1 = psimd_load_f32(w + 8);
      vacc = psimd_qfma_f32(vacc, vi1, vk1);

      const psimd_f32 vi2 = psimd_load_f32(i2);
      const psimd_f32 vk2 = psimd_load_f32(w + 12);
      vacc = psimd_qfma_f32(vacc, vi2, vk2);

      const psimd_f32 vi3 = psimd_load_f32(i3);
      const psimd_f32 vk3 = psimd_load_f32(w + 16);
      vacc = psimd_qfma_f32(vacc, vi3, vk3);

      const psimd_f32 vi4 = psimd_load_f32(i4);
      const psimd_f32 vk4 = psimd_load_f32(w + 20);
      vacc = psimd_qfma_f32(vacc, vi4, vk4);

      const psimd_f32 vi5 = psimd_load_f32(i5);
      const psimd_f32 vk5 = psimd_load_f32(w + 24);
      vacc = psimd_qfma_f32(vacc, vi5, vk5);

      const psimd_f32 vi6 = psimd_load_f32(i6);
      const psimd_f32 vk6 = psimd_load_f32(w + 28);
      vacc = psimd_qfma_f32(vacc, vi6, vk6);

      const psimd_f32 vi7 = psimd_load_f32(i7);
      const psimd_f32 vk7 = psimd_load_f32(w + 32);
      vacc = psimd_qfma_f32(vacc, vi7, vk7);

      const psimd_f32 vi8 = psimd_load_f32(i8);
      const psimd_f32 vk8 = psimd_load_f32(w + 36);
      vacc = psimd_qfma_f32(vacc, vi8, vk8);

      const psimd_f32 vi9 = psimd_load_f32(i9);
      const psimd_f32 vk9 = psimd_load_f32(w + 40);
      vacc = psimd_qfma_f32(vacc, vi9, vk9);

      const psimd_f32 vi10 = psimd_load_f32(i10);
      const psimd_f32 vk10 = psimd_load_f32(w + 44);
      vacc = psimd_qfma_f32(vacc, vi10, vk10);

      const psimd_f32 vi11 = psimd_load_f32(i11);
      const psimd_f32 vk11 = psimd_load_f32(w + 48);
      vacc = psimd_qfma_f32(vacc, vi11, vk11);

      const psimd_f32 vi12 = psimd_load_f32(i12);
      const psimd_f32 vk12 = psimd_load_f32(w + 52);
      vacc = psimd_qfma_f32(vacc, vi12, vk12);

      const psimd_f32 vi13 = psimd_load_f32(i13);
      const psimd_f32 vk13 = psimd_load_f32(w + 56);
      vacc = psimd_qfma_f32(vacc, vi13, vk13);

      const psimd_f32 vi14 = psimd_load_f32(i14);
      const psimd_f32 vk14 = psimd_load_f32(w + 60);
      vacc = psimd_qfma_f32(vacc, vi14, vk14);

      const psimd_f32 vi15 = psimd_load_f32(i15);
      const psimd_f32 vk15 = psimd_load_f32(w + 64);
      vacc = psimd_qfma_f32(vacc, vi15, vk15);

      const psimd_f32 vi16 = psimd_load_f32(i16);
      const psimd_f32 vk16 = psimd_load_f32(w + 68);
      vacc = psimd_qfma_f32(vacc, vi16, vk16);

      const psimd_f32 vi17 = psimd_load_f32(i17);
      const psimd_f32 vk17 = psimd_load_f32(w + 72);
      vacc = psimd_qfma_f32(vacc, vi17, vk17);

      const psimd_f32 vi18 = psimd_load_f32(i18);
      const psimd_f32 vk18 = psimd_load_f32(w + 76);
      vacc = psimd_qfma_f32(vacc, vi18, vk18);

      const psimd_f32 vi19 = psimd_load_f32(i19);
      const psimd_f32 vk19 = psimd_load_f32(w + 80);
      vacc = psimd_qfma_f32(vacc, vi19, vk19);

      const psimd_f32 vi20 = psimd_load_f32(i20);
      const psimd_f32 vk20 = psimd_load_f32(w + 84);
      vacc = psimd_qfma_f32(vacc, vi20, vk20);

      const psimd_f32 vi21 = psimd_load_f32(i21);
      const psimd_f32 vk21 = psimd_load_f32(w + 88);
      vacc = psimd_qfma_f32(vacc, vi21, vk21);

      const psimd_f32 vi22 = psimd_load_f32(i22);
      const psimd_f32 vk22 = psimd_load_f32(w + 92);
      vacc = psimd_qfma_f32(vacc, vi22, vk22);

      const psimd_f32 vi23 = psimd_load_f32(i23);
      const psimd_f32 vk23 = psimd_load_f32(w + 96);
      vacc = psimd_qfma_f32(vacc, vi23, vk23);

      const psimd_f32 vi24 = psimd_load_f32(i24);
      const psimd_f32 vk24 = psimd_load_f32(w + 100);
      vacc = psimd_qfma_f32(vacc, vi24, vk24);

      w += 104;

      vacc = psimd_max_f32(vacc, vmin);
      vacc = psimd_min_f32(vacc, vmax);

      if (c & 2) {
        psimd_store2_f32(output, vacc);
        vacc = psimd_concat_hi_f32(vacc, vacc);
        output += 2;
      }
      if (c & 1) {
        psimd_store1_f32(output, vacc);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
