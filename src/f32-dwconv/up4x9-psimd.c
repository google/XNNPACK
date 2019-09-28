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


void xnn_f32_dwconv_ukernel_up4x9__psimd(
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

      w += 40;

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

      w += 40;

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
