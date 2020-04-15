// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/avgpool.h>


void xnn_f32_avgpool_minmax_ukernel_9x__psimd_c4(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const psimd_f32 vscale = psimd_load_splat_f32(&params->scalar.scale);
  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);
  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    input = (const float**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }

    size_t c = channels;
    while (c >= 4) {
      const psimd_f32 vi0 = psimd_load_f32(i0);
      i0 += 4;
      const psimd_f32 vi1 = psimd_load_f32(i1);
      i1 += 4;
      const psimd_f32 vi2 = psimd_load_f32(i2);
      i2 += 4;
      const psimd_f32 vi3 = psimd_load_f32(i3);
      i3 += 4;
      const psimd_f32 vi4 = psimd_load_f32(i4);
      i4 += 4;
      const psimd_f32 vi5 = psimd_load_f32(i5);
      i5 += 4;
      const psimd_f32 vi6 = psimd_load_f32(i6);
      i6 += 4;
      const psimd_f32 vi7 = psimd_load_f32(i7);
      i7 += 4;
      const psimd_f32 vi8 = psimd_load_f32(i8);
      i8 += 4;

      const psimd_f32 vsum018 = psimd_add_f32(psimd_add_f32(vi0, vi1), vi8);
      const psimd_f32 vsum23 = psimd_add_f32(vi2, vi3);
      const psimd_f32 vsum45 = psimd_add_f32(vi4, vi5);
      const psimd_f32 vsum67 = psimd_add_f32(vi6, vi7);

      const psimd_f32 vsum2345 = psimd_add_f32(vsum23, vsum45);
      const psimd_f32 vsum01678 = psimd_add_f32(vsum018, vsum67);
      const psimd_f32 vsum = psimd_add_f32(vsum2345, vsum01678);

      psimd_f32 vout = psimd_mul_f32(vsum, vscale);
      vout = psimd_max_f32(vout, vmin);
      vout = psimd_min_f32(vout, vmax);

      psimd_store_f32(output, vout);
      output += 4;

      c -= 4;
    }
    if (c != 0) {
      const psimd_f32 vi0 = psimd_load_f32(i0);
      const psimd_f32 vi1 = psimd_load_f32(i1);
      const psimd_f32 vi2 = psimd_load_f32(i2);
      const psimd_f32 vi3 = psimd_load_f32(i3);
      const psimd_f32 vi4 = psimd_load_f32(i4);
      const psimd_f32 vi5 = psimd_load_f32(i5);
      const psimd_f32 vi6 = psimd_load_f32(i6);
      const psimd_f32 vi7 = psimd_load_f32(i7);
      const psimd_f32 vi8 = psimd_load_f32(i8);

      const psimd_f32 vsum01 = psimd_add_f32(vi0, vi1);
      const psimd_f32 vsum23 = psimd_add_f32(vi2, vi3);
      const psimd_f32 vsum45 = psimd_add_f32(vi4, vi5);
      const psimd_f32 vsum67 = psimd_add_f32(vi6, vi7);
      const psimd_f32 vsum018 = psimd_add_f32(vsum01, vi8);
      const psimd_f32 vsum2345 = psimd_add_f32(vsum23, vsum45);
      const psimd_f32 vsum01678 = psimd_add_f32(vsum018, vsum67);
      const psimd_f32 vsum = psimd_add_f32(vsum2345, vsum01678);

      psimd_f32 vout = psimd_mul_f32(vsum, vscale);
      vout = psimd_max_f32(vout, vmin);
      vout = psimd_min_f32(vout, vmax);

      if (c & 2) {
        psimd_store2_f32(output, vout);
        output += 2;
        vout = psimd_concat_hi_f32(vout, vout);
      }
      if (c & 1) {
        psimd_store1_f32(output, vout);
        output += 1;
      }
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
