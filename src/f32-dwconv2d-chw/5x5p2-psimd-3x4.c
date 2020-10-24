// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>



PSIMD_INTRINSIC psimd_f32 extq1_f32(psimd_f32 a, psimd_f32 b) {
  #if defined(__clang__)
    return __builtin_shufflevector(a, b, 1, 2, 3, 4);
  #else
    return __builtin_shuffle(a, b, (psimd_s32){1, 2, 3, 4});
  #endif  // defined(__clang__)
}

PSIMD_INTRINSIC psimd_f32 extq2_f32(psimd_f32 a, psimd_f32 b) {
  #if defined(__clang__)
    return __builtin_shufflevector(a, b, 2, 3, 4, 5);
  #else
    return __builtin_shuffle(a, b, (psimd_s32){2, 3, 4, 5});
  #endif  // defined(__clang__)
}

PSIMD_INTRINSIC psimd_f32 extq3_f32(psimd_f32 a, psimd_f32 b) {
  #if defined(__clang__)
    return __builtin_shufflevector(a, b, 3, 4, 5, 6);
  #else
    return __builtin_shuffle(a, b, (psimd_s32){3, 4, 5, 6});
  #endif  // defined(__clang__)
}

void xnn_f32_dwconv2d_chw_ukernel_5x5p2__psimd_3x4(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float *zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 2);

  const psimd_s32 vmask = psimd_load_s32(params->scalar.mask);
  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);

  const psimd_f32 vbias = psimd_load_splat_f32(weights);
  const psimd_f32 vk00 = psimd_load_splat_f32(weights + 1);
  const psimd_f32 vk01 = psimd_load_splat_f32(weights + 2);
  const psimd_f32 vk02 = psimd_load_splat_f32(weights + 3);
  const psimd_f32 vk03 = psimd_load_splat_f32(weights + 4);
  const psimd_f32 vk04 = psimd_load_splat_f32(weights + 5);
  const psimd_f32 vk10 = psimd_load_splat_f32(weights + 6);
  const psimd_f32 vk11 = psimd_load_splat_f32(weights + 7);
  const psimd_f32 vk12 = psimd_load_splat_f32(weights + 8);
  const psimd_f32 vk13 = psimd_load_splat_f32(weights + 9);
  const psimd_f32 vk14 = psimd_load_splat_f32(weights + 10);
  const psimd_f32 vk20 = psimd_load_splat_f32(weights + 11);
  const psimd_f32 vk21 = psimd_load_splat_f32(weights + 12);
  const psimd_f32 vk22 = psimd_load_splat_f32(weights + 13);
  const psimd_f32 vk23 = psimd_load_splat_f32(weights + 14);
  const psimd_f32 vk24 = psimd_load_splat_f32(weights + 15);
  const psimd_f32 vk30 = psimd_load_splat_f32(weights + 16);
  const psimd_f32 vk31 = psimd_load_splat_f32(weights + 17);
  const psimd_f32 vk32 = psimd_load_splat_f32(weights + 18);
  const psimd_f32 vk33 = psimd_load_splat_f32(weights + 19);
  const psimd_f32 vk34 = psimd_load_splat_f32(weights + 20);
  const psimd_f32 vk40 = psimd_load_splat_f32(weights + 21);
  const psimd_f32 vk41 = psimd_load_splat_f32(weights + 22);
  const psimd_f32 vk42 = psimd_load_splat_f32(weights + 23);
  const psimd_f32 vk43 = psimd_load_splat_f32(weights + 24);
  const psimd_f32 vk44 = psimd_load_splat_f32(weights + 25);

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = zero;
  const float* i2 = input;
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height <= 2) {
      i4 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i5 = zero;
    }
    if XNN_UNPREDICTABLE(output_height <= 4) {
      i6 = zero;
    }

    psimd_f32 vi0x0123 = psimd_zero_f32();
    psimd_f32 vi1x0123 = psimd_zero_f32();
    psimd_f32 vi2x0123 = psimd_zero_f32();
    psimd_f32 vi3x0123 = psimd_zero_f32();
    psimd_f32 vi4x0123 = psimd_zero_f32();
    psimd_f32 vi5x0123 = psimd_zero_f32();
    psimd_f32 vi6x0123 = psimd_zero_f32();
    psimd_f32 vi0x4567 = psimd_load_f32(i0); i0 += 4;
    psimd_f32 vi1x4567 = psimd_load_f32(i1); i1 += 4;
    psimd_f32 vi2x4567 = psimd_load_f32(i2); i2 += 4;
    psimd_f32 vi3x4567 = psimd_load_f32(i3); i3 += 4;
    psimd_f32 vi4x4567 = psimd_load_f32(i4); i4 += 4;
    psimd_f32 vi5x4567 = psimd_load_f32(i5); i5 += 4;
    psimd_f32 vi6x4567 = psimd_load_f32(i6); i6 += 4;

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 4 * sizeof(float)) {
      psimd_f32 vo4567p0 = vbias;
      psimd_f32 vo4567p1 = vbias;
      psimd_f32 vo4567p2 = vbias;

      const psimd_f32 vi0x89AB = psimd_load_f32(i0); i0 += 4;
      const psimd_f32 vi1x89AB = psimd_load_f32(i1); i1 += 4;
      const psimd_f32 vi2x89AB = psimd_load_f32(i2); i2 += 4;
      const psimd_f32 vi3x89AB = psimd_load_f32(i3); i3 += 4;
      const psimd_f32 vi4x89AB = psimd_load_f32(i4); i4 += 4;
      const psimd_f32 vi5x89AB = psimd_load_f32(i5); i5 += 4;
      const psimd_f32 vi6x89AB = psimd_load_f32(i6); i6 += 4;

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x4567, vk02);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x4567, vk02);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x4567, vk02);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x4567, vk12);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x4567, vk12);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x4567, vk12);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x4567, vk22);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x4567, vk22);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x4567, vk22);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x4567, vk32);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x4567, vk32);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x4567, vk32);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x4567, vk42);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x4567, vk42);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x4567, vk42);

      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x3456 = extq3_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x3456 = extq3_f32(vi6x0123, vi6x4567);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x3456, vk01);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x3456, vk01);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x3456, vk01);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x3456, vk11);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x3456, vk11);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x3456, vk11);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x3456, vk21);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x3456, vk21);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x3456, vk21);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x3456, vk31);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x3456, vk31);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x3456, vk31);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x3456, vk41);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x3456, vk41);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x3456, vk41);

      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x2345 = extq2_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x2345 = extq2_f32(vi6x0123, vi6x4567);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x2345, vk00);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x2345, vk00);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x2345, vk00);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x2345, vk10);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x2345, vk10);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x2345, vk10);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x2345, vk20);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x2345, vk20);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x2345, vk20);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x2345, vk30);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x2345, vk30);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x2345, vk30);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x2345, vk40);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x2345, vk40);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x2345, vk40);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;
      vi6x0123 = vi6x4567;

      const psimd_f32 vi0x5678 = extq1_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x5678 = extq1_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x5678 = extq1_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x5678 = extq1_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x5678 = extq1_f32(vi4x4567, vi4x89AB);
      const psimd_f32 vi5x5678 = extq1_f32(vi5x4567, vi5x89AB);
      const psimd_f32 vi6x5678 = extq1_f32(vi6x4567, vi6x89AB);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x5678, vk03);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x5678, vk03);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x5678, vk03);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x5678, vk13);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x5678, vk13);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x5678, vk13);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x5678, vk23);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x5678, vk23);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x5678, vk23);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x5678, vk33);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x5678, vk33);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x5678, vk33);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x5678, vk43);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x5678, vk43);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x5678, vk43);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vi4x89AB);
      const psimd_f32 vi5x6789 = extq2_f32(vi5x4567, vi5x89AB);
      const psimd_f32 vi6x6789 = extq2_f32(vi6x4567, vi6x89AB);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x6789, vk04);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x6789, vk04);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x6789, vk04);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x6789, vk14);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x6789, vk14);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x6789, vk14);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x6789, vk24);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x6789, vk24);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x6789, vk24);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x6789, vk34);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x6789, vk34);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x6789, vk34);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x6789, vk44);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x6789, vk44);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x6789, vk44);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;
      vi6x4567 = vi6x89AB;

      psimd_f32 vo0 = vo4567p0;
      psimd_f32 vo1 = vo4567p1;
      psimd_f32 vo2 = vo4567p2;

      vo0 = psimd_max_f32(vo0, vmin);
      vo1 = psimd_max_f32(vo1, vmin);
      vo2 = psimd_max_f32(vo2, vmin);

      vo0 = psimd_min_f32(vo0, vmax);
      vo1 = psimd_min_f32(vo1, vmax);
      vo2 = psimd_min_f32(vo2, vmax);

      psimd_store_f32(o2, vo2); o2 += 4;
      psimd_store_f32(o1, vo1); o1 += 4;
      psimd_store_f32(o0, vo0); o0 += 4;
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(w > 4 * sizeof(float)) {
      psimd_f32 vo4567p0 = vbias;
      psimd_f32 vo4567p1 = vbias;
      psimd_f32 vo4567p2 = vbias;

      psimd_f32 vi0x89AB = psimd_load_f32(i0); i0 += 4;
      psimd_f32 vi1x89AB = psimd_load_f32(i1); i1 += 4;
      psimd_f32 vi2x89AB = psimd_load_f32(i2); i2 += 4;
      psimd_f32 vi3x89AB = psimd_load_f32(i3); i3 += 4;
      psimd_f32 vi4x89AB = psimd_load_f32(i4); i4 += 4;
      psimd_f32 vi5x89AB = psimd_load_f32(i5); i5 += 4;
      psimd_f32 vi6x89AB = psimd_load_f32(i6); i6 += 4;

      vi0x89AB = psimd_andmask_f32(vmask, vi0x89AB);
      vi1x89AB = psimd_andmask_f32(vmask, vi1x89AB);
      vi2x89AB = psimd_andmask_f32(vmask, vi2x89AB);
      vi3x89AB = psimd_andmask_f32(vmask, vi3x89AB);
      vi4x89AB = psimd_andmask_f32(vmask, vi4x89AB);
      vi5x89AB = psimd_andmask_f32(vmask, vi5x89AB);
      vi6x89AB = psimd_andmask_f32(vmask, vi6x89AB);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x4567, vk02);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x4567, vk02);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x4567, vk02);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x4567, vk12);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x4567, vk12);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x4567, vk12);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x4567, vk22);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x4567, vk22);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x4567, vk22);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x4567, vk32);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x4567, vk32);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x4567, vk32);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x4567, vk42);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x4567, vk42);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x4567, vk42);

      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x3456 = extq3_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x3456 = extq3_f32(vi6x0123, vi6x4567);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x3456, vk01);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x3456, vk01);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x3456, vk01);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x3456, vk11);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x3456, vk11);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x3456, vk11);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x3456, vk21);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x3456, vk21);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x3456, vk21);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x3456, vk31);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x3456, vk31);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x3456, vk31);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x3456, vk41);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x3456, vk41);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x3456, vk41);

      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x2345 = extq2_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x2345 = extq2_f32(vi6x0123, vi6x4567);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x2345, vk00);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x2345, vk00);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x2345, vk00);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x2345, vk10);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x2345, vk10);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x2345, vk10);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x2345, vk20);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x2345, vk20);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x2345, vk20);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x2345, vk30);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x2345, vk30);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x2345, vk30);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x2345, vk40);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x2345, vk40);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x2345, vk40);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;
      vi6x0123 = vi6x4567;

      const psimd_f32 vi0x5678 = extq1_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x5678 = extq1_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x5678 = extq1_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x5678 = extq1_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x5678 = extq1_f32(vi4x4567, vi4x89AB);
      const psimd_f32 vi5x5678 = extq1_f32(vi5x4567, vi5x89AB);
      const psimd_f32 vi6x5678 = extq1_f32(vi6x4567, vi6x89AB);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x5678, vk03);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x5678, vk03);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x5678, vk03);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x5678, vk13);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x5678, vk13);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x5678, vk13);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x5678, vk23);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x5678, vk23);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x5678, vk23);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x5678, vk33);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x5678, vk33);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x5678, vk33);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x5678, vk43);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x5678, vk43);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x5678, vk43);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vi4x89AB);
      const psimd_f32 vi5x6789 = extq2_f32(vi5x4567, vi5x89AB);
      const psimd_f32 vi6x6789 = extq2_f32(vi6x4567, vi6x89AB);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x6789, vk04);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x6789, vk04);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x6789, vk04);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x6789, vk14);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x6789, vk14);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x6789, vk14);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x6789, vk24);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x6789, vk24);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x6789, vk24);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x6789, vk34);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x6789, vk34);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x6789, vk34);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x6789, vk44);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x6789, vk44);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x6789, vk44);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;
      vi6x4567 = vi6x89AB;

      psimd_f32 vo0 = vo4567p0;
      psimd_f32 vo1 = vo4567p1;
      psimd_f32 vo2 = vo4567p2;

      vo0 = psimd_max_f32(vo0, vmin);
      vo1 = psimd_max_f32(vo1, vmin);
      vo2 = psimd_max_f32(vo2, vmin);

      vo0 = psimd_min_f32(vo0, vmax);
      vo1 = psimd_min_f32(vo1, vmax);
      vo2 = psimd_min_f32(vo2, vmax);

      psimd_store_f32(o2, vo2); o2 += 4;
      psimd_store_f32(o1, vo1); o1 += 4;
      psimd_store_f32(o0, vo0); o0 += 4;
      w -= 4 * sizeof(float);
    }
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      psimd_f32 vo4567p0 = vbias;
      psimd_f32 vo4567p1 = vbias;
      psimd_f32 vo4567p2 = vbias;

      // This might have already happened if there are more than 4 pixels, but we can't count on it.
      vi0x4567 = psimd_andmask_f32(vmask, vi0x4567);
      vi1x4567 = psimd_andmask_f32(vmask, vi1x4567);
      vi2x4567 = psimd_andmask_f32(vmask, vi2x4567);
      vi3x4567 = psimd_andmask_f32(vmask, vi3x4567);
      vi4x4567 = psimd_andmask_f32(vmask, vi4x4567);
      vi5x4567 = psimd_andmask_f32(vmask, vi5x4567);
      vi6x4567 = psimd_andmask_f32(vmask, vi6x4567);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x4567, vk02);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x4567, vk02);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x4567, vk02);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x4567, vk12);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x4567, vk12);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x4567, vk12);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x4567, vk22);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x4567, vk22);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x4567, vk22);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x4567, vk32);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x4567, vk32);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x4567, vk32);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x4567, vk42);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x4567, vk42);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x4567, vk42);

      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x3456 = extq3_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x3456 = extq3_f32(vi6x0123, vi6x4567);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x3456, vk01);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x3456, vk01);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x3456, vk01);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x3456, vk11);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x3456, vk11);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x3456, vk11);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x3456, vk21);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x3456, vk21);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x3456, vk21);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x3456, vk31);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x3456, vk31);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x3456, vk31);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x3456, vk41);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x3456, vk41);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x3456, vk41);

      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x2345 = extq2_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x2345 = extq2_f32(vi6x0123, vi6x4567);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x2345, vk00);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x2345, vk00);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x2345, vk00);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x2345, vk10);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x2345, vk10);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x2345, vk10);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x2345, vk20);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x2345, vk20);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x2345, vk20);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x2345, vk30);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x2345, vk30);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x2345, vk30);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x2345, vk40);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x2345, vk40);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x2345, vk40);

      const psimd_f32 vzero = psimd_zero_f32();
      const psimd_f32 vi0x5678 = extq1_f32(vi0x4567, vzero);
      const psimd_f32 vi1x5678 = extq1_f32(vi1x4567, vzero);
      const psimd_f32 vi2x5678 = extq1_f32(vi2x4567, vzero);
      const psimd_f32 vi3x5678 = extq1_f32(vi3x4567, vzero);
      const psimd_f32 vi4x5678 = extq1_f32(vi4x4567, vzero);
      const psimd_f32 vi5x5678 = extq1_f32(vi5x4567, vzero);
      const psimd_f32 vi6x5678 = extq1_f32(vi6x4567, vzero);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x5678, vk03);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x5678, vk03);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x5678, vk03);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x5678, vk13);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x5678, vk13);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x5678, vk13);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x5678, vk23);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x5678, vk23);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x5678, vk23);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x5678, vk33);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x5678, vk33);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x5678, vk33);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x5678, vk43);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x5678, vk43);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x5678, vk43);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vzero);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vzero);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vzero);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vzero);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vzero);
      const psimd_f32 vi5x6789 = extq2_f32(vi5x4567, vzero);
      const psimd_f32 vi6x6789 = extq2_f32(vi6x4567, vzero);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi0x6789, vk04);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi1x6789, vk04);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi2x6789, vk04);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi1x6789, vk14);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi2x6789, vk14);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi3x6789, vk14);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi2x6789, vk24);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi3x6789, vk24);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi4x6789, vk24);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi3x6789, vk34);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi4x6789, vk34);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi5x6789, vk34);

      vo4567p0 = psimd_qfma_f32(vo4567p0, vi4x6789, vk44);
      vo4567p1 = psimd_qfma_f32(vo4567p1, vi5x6789, vk44);
      vo4567p2 = psimd_qfma_f32(vo4567p2, vi6x6789, vk44);

      psimd_f32 vo0 = vo4567p0;
      psimd_f32 vo1 = vo4567p1;
      psimd_f32 vo2 = vo4567p2;

      vo0 = psimd_max_f32(vo0, vmin);
      vo1 = psimd_max_f32(vo1, vmin);
      vo2 = psimd_max_f32(vo2, vmin);

      vo0 = psimd_min_f32(vo0, vmax);
      vo1 = psimd_min_f32(vo1, vmax);
      vo2 = psimd_min_f32(vo2, vmax);

      if XNN_LIKELY(w & (4 * sizeof(float))) {
        psimd_store_f32(o2, vo2);
        o2 += 4;
        psimd_store_f32(o1, vo1);
        o1 += 4;
        psimd_store_f32(o0, vo0);
        o0 += 4;
      } else {
        if (w & (2 * sizeof(float))) {
          psimd_store2_f32(o2, vo2);
          o2 += 2;
          psimd_store2_f32(o1, vo1);
          o1 += 2;
          psimd_store2_f32(o0, vo0);
          o0 += 2;

          vo0 = psimd_splat2_f32(vo0);
          vo1 = psimd_splat2_f32(vo1);
          vo2 = psimd_splat2_f32(vo2);
        }
        if (w & (1 * sizeof(float))) {
          psimd_store1_f32(o2, vo2);
          o2 += 1;
          psimd_store1_f32(o1, vo1);
          o1 += 1;
          psimd_store1_f32(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i3 - input_decrement);
    i1 = (const float*) ((uintptr_t) i4 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);

    o0 = o2;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);

    output_height = doz(output_height, 3);
  } while (output_height != 0);
}
