// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>

PSIMD_INTRINSIC psimd_f32 concat_2456_f32(psimd_f32 a, psimd_f32 b) {
  #if defined(__clang__)
    return __builtin_shufflevector(a, b, 2, 4, 5, 6);
  #else
    return __builtin_shuffle(a, b, (psimd_s32){2, 4, 5, 6});
  #endif  // defined(__clang__)
}

PSIMD_INTRINSIC psimd_f32 extq1_f32(psimd_f32 a, psimd_f32 b) {
  #if defined(__clang__)
    return __builtin_shufflevector(a, b, 1, 2, 3, 4);
  #else
    return __builtin_shuffle(a, b, (psimd_s32){1, 2, 3, 4});
  #endif  // defined(__clang__)
}

PSIMD_INTRINSIC psimd_f32 extq3_f32(psimd_f32 a, psimd_f32 b) {
  #if defined(__clang__)
    return __builtin_shufflevector(a, b, 3, 4, 5, 6);
  #else
    return __builtin_shuffle(a, b, (psimd_s32){3, 4, 5, 6});
  #endif  // defined(__clang__)
}

void xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__psimd_1x4_acc2(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const psimd_s32 vmask_even = psimd_load_s32(params->scalar.mask_even);
  const psimd_s32 vmask_odd = psimd_load_s32(params->scalar.mask_odd);
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

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_down_po2(input_width - 1 * sizeof(float), 4 * sizeof(float)) + 4 * sizeof(float);

  const float* i0 = zero;
  const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height <= 6) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
    }

    psimd_f32 vi0x0123 = psimd_zero_f32();
    psimd_f32 vi1x0123 = psimd_zero_f32();
    psimd_f32 vi2x0123 = psimd_zero_f32();
    psimd_f32 vi3x0123 = psimd_zero_f32();
    psimd_f32 vi4x0123 = psimd_zero_f32();
    psimd_f32 vi0x4567 = psimd_load_f32(i0);
    i0 += 4;
    psimd_f32 vi1x4567 = psimd_load_f32(i1);
    i1 += 4;
    psimd_f32 vi2x4567 = psimd_load_f32(i2);
    i2 += 4;
    psimd_f32 vi3x4567 = psimd_load_f32(i3);
    i3 += 4;
    psimd_f32 vi4x4567 = psimd_load_f32(i4);
    i4 += 4;

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 8 * sizeof(float)) {
      psimd_f32 vo468Ap0 = vbias;

      const psimd_f32 vi0x89AB = psimd_load_f32(i0);
      const psimd_f32 vi1x89AB = psimd_load_f32(i1);
      const psimd_f32 vi2x89AB = psimd_load_f32(i2);
      const psimd_f32 vi3x89AB = psimd_load_f32(i3);
      const psimd_f32 vi4x89AB = psimd_load_f32(i4);

      const psimd_f32 vi0xCDEF = psimd_load_f32(i0 + 4);
      i0 += 8;
      const psimd_f32 vi1xCDEF = psimd_load_f32(i1 + 4);
      i1 += 8;
      const psimd_f32 vi2xCDEF = psimd_load_f32(i2 + 4);
      i2 += 8;
      const psimd_f32 vi3xCDEF = psimd_load_f32(i3 + 4);
      i3 += 8;
      const psimd_f32 vi4xCDEF = psimd_load_f32(i4 + 4);
      i4 += 8;

      const psimd_f32 vi0x468A = psimd_concat_even_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi0x579B = psimd_concat_odd_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x468A = psimd_concat_even_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi1x579B = psimd_concat_odd_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x468A = psimd_concat_even_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi2x579B = psimd_concat_odd_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x468A = psimd_concat_even_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi3x579B = psimd_concat_odd_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x468A = psimd_concat_even_f32(vi4x4567, vi4x89AB);
      const psimd_f32 vi4x579B = psimd_concat_odd_f32(vi4x4567, vi4x89AB);

      // middle tap
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi0x468A, vk02);
      psimd_f32 vo468Ap1 = psimd_mul_f32(vi1x468A, vk12);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi2x468A, vk22);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi3x468A, vk32);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi4x468A, vk42);

      // one left
      const psimd_f32 vi0x3579 = extq3_f32(vi0x0123, vi0x579B);
      const psimd_f32 vi1x3579 = extq3_f32(vi1x0123, vi1x579B);
      const psimd_f32 vi2x3579 = extq3_f32(vi2x0123, vi2x579B);
      const psimd_f32 vi3x3579 = extq3_f32(vi3x0123, vi3x579B);
      const psimd_f32 vi4x3579 = extq3_f32(vi4x0123, vi4x579B);

      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi0x3579, vk01);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi1x3579, vk11);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi2x3579, vk21);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi3x3579, vk31);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi4x3579, vk41);

      // two left
      const psimd_f32 vi0x2468 = concat_2456_f32(vi0x0123, vi0x468A);
      const psimd_f32 vi1x2468 = concat_2456_f32(vi1x0123, vi1x468A);
      const psimd_f32 vi2x2468 = concat_2456_f32(vi2x0123, vi2x468A);
      const psimd_f32 vi3x2468 = concat_2456_f32(vi3x0123, vi3x468A);
      const psimd_f32 vi4x2468 = concat_2456_f32(vi4x0123, vi4x468A);

      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi0x2468, vk00);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi1x2468, vk10);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi2x2468, vk20);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi3x2468, vk30);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi4x2468, vk40);

      vi0x0123 = vi0x89AB;
      vi1x0123 = vi1x89AB;
      vi2x0123 = vi2x89AB;
      vi3x0123 = vi3x89AB;
      vi4x0123 = vi4x89AB;

      // one right
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi0x579B, vk03);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi1x579B, vk13);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi2x579B, vk23);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi3x579B, vk33);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi4x579B, vk43);

      // two right
      const psimd_f32 vi0x68AC = extq1_f32(vi0x468A, vi0xCDEF);
      const psimd_f32 vi1x68AC = extq1_f32(vi1x468A, vi1xCDEF);
      const psimd_f32 vi2x68AC = extq1_f32(vi2x468A, vi2xCDEF);
      const psimd_f32 vi3x68AC = extq1_f32(vi3x468A, vi3xCDEF);
      const psimd_f32 vi4x68AC = extq1_f32(vi4x468A, vi4xCDEF);

      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi0x68AC, vk04);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi1x68AC, vk14);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi2x68AC, vk24);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi3x68AC, vk34);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi4x68AC, vk44);

      vi0x4567 = vi0xCDEF;
      vi1x4567 = vi1xCDEF;
      vi2x4567 = vi2xCDEF;
      vi3x4567 = vi3xCDEF;
      vi4x4567 = vi4xCDEF;

      psimd_f32 vo0 = psimd_add_f32(vo468Ap0, vo468Ap1);

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);

      size_t w_tmp = (w + 1 * sizeof(float)) / (2 * sizeof(float));
      if XNN_LIKELY(w_tmp >= 4) {
        psimd_store_f32(output, vo0);
        output += 4;
      } else {
        if (w_tmp & 2) {
          psimd_store2_f32(output, vo0);
          output += 2;
          vo0 = psimd_splat2_f32(vo0);
        }
        if (w_tmp & 1) {
          psimd_store1_f32(output, vo0);
          output += 1;
        }
      }
    }

    {
      psimd_f32 vo468Ap0 = vbias;

      psimd_f32 vi0x89AB = psimd_zero_f32();
      psimd_f32 vi1x89AB = psimd_zero_f32();
      psimd_f32 vi2x89AB = psimd_zero_f32();
      psimd_f32 vi3x89AB = psimd_zero_f32();
      psimd_f32 vi4x89AB = psimd_zero_f32();
      if XNN_LIKELY(w > 4 * sizeof(float)) {
        vi0x89AB = psimd_load_f32(i0);
        i0 += 4;
        vi1x89AB = psimd_load_f32(i1);
        i1 += 4;
        vi2x89AB = psimd_load_f32(i2);
        i2 += 4;
        vi3x89AB = psimd_load_f32(i3);
        i3 += 4;
        vi4x89AB = psimd_load_f32(i4);
        i4 += 4;
      }

      psimd_f32 vi0xCDEF = psimd_zero_f32();
      psimd_f32 vi1xCDEF = psimd_zero_f32();
      psimd_f32 vi2xCDEF = psimd_zero_f32();
      psimd_f32 vi3xCDEF = psimd_zero_f32();
      psimd_f32 vi4xCDEF = psimd_zero_f32();
      if XNN_LIKELY(w > 8 * sizeof(float)) {
        vi0xCDEF = psimd_load_f32(i0);
        i0 += 4;
        vi1xCDEF = psimd_load_f32(i1);
        i1 += 4;
        vi2xCDEF = psimd_load_f32(i2);
        i2 += 4;
        vi3xCDEF = psimd_load_f32(i3);
        i3 += 4;
        vi4xCDEF = psimd_load_f32(i4);
        i4 += 4;
      }

      psimd_f32 vi0x468A = psimd_concat_even_f32(vi0x4567, vi0x89AB);
      psimd_f32 vi0x579B = psimd_concat_odd_f32(vi0x4567, vi0x89AB);
      psimd_f32 vi1x468A = psimd_concat_even_f32(vi1x4567, vi1x89AB);
      psimd_f32 vi1x579B = psimd_concat_odd_f32(vi1x4567, vi1x89AB);
      psimd_f32 vi2x468A = psimd_concat_even_f32(vi2x4567, vi2x89AB);
      psimd_f32 vi2x579B = psimd_concat_odd_f32(vi2x4567, vi2x89AB);
      psimd_f32 vi3x468A = psimd_concat_even_f32(vi3x4567, vi3x89AB);
      psimd_f32 vi3x579B = psimd_concat_odd_f32(vi3x4567, vi3x89AB);
      psimd_f32 vi4x468A = psimd_concat_even_f32(vi4x4567, vi4x89AB);
      psimd_f32 vi4x579B = psimd_concat_odd_f32(vi4x4567, vi4x89AB);

      vi0x468A = psimd_andmask_f32(vmask_even, vi0x468A);
      vi1x468A = psimd_andmask_f32(vmask_even, vi1x468A);
      vi2x468A = psimd_andmask_f32(vmask_even, vi2x468A);
      vi3x468A = psimd_andmask_f32(vmask_even, vi3x468A);
      vi4x468A = psimd_andmask_f32(vmask_even, vi4x468A);

      vi0x579B = psimd_andmask_f32(vmask_odd, vi0x579B);
      vi1x579B = psimd_andmask_f32(vmask_odd, vi1x579B);
      vi2x579B = psimd_andmask_f32(vmask_odd, vi2x579B);
      vi3x579B = psimd_andmask_f32(vmask_odd, vi3x579B);
      vi4x579B = psimd_andmask_f32(vmask_odd, vi4x579B);

      // middle tap
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi0x468A, vk02);
      psimd_f32 vo468Ap1 = psimd_mul_f32(vi1x468A, vk12);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi2x468A, vk22);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi3x468A, vk32);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi4x468A, vk42);

      // one left
      const psimd_f32 vi0x3579 = extq3_f32(vi0x0123, vi0x579B);
      const psimd_f32 vi1x3579 = extq3_f32(vi1x0123, vi1x579B);
      const psimd_f32 vi2x3579 = extq3_f32(vi2x0123, vi2x579B);
      const psimd_f32 vi3x3579 = extq3_f32(vi3x0123, vi3x579B);
      const psimd_f32 vi4x3579 = extq3_f32(vi4x0123, vi4x579B);

      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi0x3579, vk01);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi1x3579, vk11);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi2x3579, vk21);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi3x3579, vk31);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi4x3579, vk41);

      // two left
      const psimd_f32 vi0x2468 = concat_2456_f32(vi0x0123, vi0x468A);
      const psimd_f32 vi1x2468 = concat_2456_f32(vi1x0123, vi1x468A);
      const psimd_f32 vi2x2468 = concat_2456_f32(vi2x0123, vi2x468A);
      const psimd_f32 vi3x2468 = concat_2456_f32(vi3x0123, vi3x468A);
      const psimd_f32 vi4x2468 = concat_2456_f32(vi4x0123, vi4x468A);

      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi0x2468, vk00);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi1x2468, vk10);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi2x2468, vk20);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi3x2468, vk30);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi4x2468, vk40);

      vi0x0123 = vi0x89AB;
      vi1x0123 = vi1x89AB;
      vi2x0123 = vi2x89AB;
      vi3x0123 = vi3x89AB;
      vi4x0123 = vi4x89AB;

      // one right
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi0x579B, vk03);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi1x579B, vk13);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi2x579B, vk23);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi3x579B, vk33);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi4x579B, vk43);

      // two right
      const psimd_f32 vi0x68AC = extq1_f32(vi0x468A, vi0xCDEF);
      const psimd_f32 vi1x68AC = extq1_f32(vi1x468A, vi1xCDEF);
      const psimd_f32 vi2x68AC = extq1_f32(vi2x468A, vi2xCDEF);
      const psimd_f32 vi3x68AC = extq1_f32(vi3x468A, vi3xCDEF);
      const psimd_f32 vi4x68AC = extq1_f32(vi4x468A, vi4xCDEF);

      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi0x68AC, vk04);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi1x68AC, vk14);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi2x68AC, vk24);
      vo468Ap1 = psimd_qfma_f32(vo468Ap1, vi3x68AC, vk34);
      vo468Ap0 = psimd_qfma_f32(vo468Ap0, vi4x68AC, vk44);

      vi0x4567 = vi0xCDEF;
      vi1x4567 = vi1xCDEF;
      vi2x4567 = vi2xCDEF;
      vi3x4567 = vi3xCDEF;
      vi4x4567 = vi4xCDEF;

      psimd_f32 vo0 = psimd_add_f32(vo468Ap0, vo468Ap1);

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);

      size_t w_tmp = (w + 1 * sizeof(float)) / (2 * sizeof(float));
      if XNN_LIKELY(w_tmp >= 4) {
        psimd_store_f32(output, vo0);
        output += 4;
      } else {
        if (w_tmp & 2) {
          psimd_store2_f32(output, vo0);
          output += 2;
          vo0 = psimd_splat2_f32(vo0);
        }
        if (w_tmp & 1) {
          psimd_store1_f32(output, vo0);
          output += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_decrement);
    i1 = (const float*) ((uintptr_t) i3 - input_decrement);
    i2 = (const float*) ((uintptr_t) i4 - input_decrement);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);

    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}
