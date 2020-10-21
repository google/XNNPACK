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

PSIMD_INTRINSIC psimd_f32 vmulq_lane0_f32(psimd_f32 a, psimd_f32 b) {
  return psimd_mul_f32(a, psimd_splat0_f32(b));
}

PSIMD_INTRINSIC psimd_f32 vfmaq_lane0_f32(psimd_f32 a, psimd_f32 b, psimd_f32 c) {
  return psimd_qfma_f32(a, b, psimd_splat0_f32(c));
}

PSIMD_INTRINSIC psimd_f32 vfmaq_lane1_f32(psimd_f32 a, psimd_f32 b, psimd_f32 c) {
  return psimd_qfma_f32(a, b, psimd_splat1_f32(c));
}

PSIMD_INTRINSIC psimd_f32 vfmaq_lane2_f32(psimd_f32 a, psimd_f32 b, psimd_f32 c) {
  return psimd_qfma_f32(a, b, psimd_splat2_f32(c));
}

PSIMD_INTRINSIC psimd_f32 vfmaq_lane3_f32(psimd_f32 a, psimd_f32 b, psimd_f32 c) {
  return psimd_qfma_f32(a, b, psimd_splat3_f32(c));
}


void xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_width != 0);
  assert(input_height != 0);
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const psimd_s32 vmask_even = psimd_load_s32(params->scalar.mask_even);
  const psimd_s32 vmask_odd = psimd_load_s32(params->scalar.mask_odd);
  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);

  const psimd_f32 vw0123 = psimd_load_f32(weights);
  const psimd_f32 vw4567 = psimd_load_f32(weights + 4);
  const psimd_f32 vw89AB = psimd_load_f32(weights + 8);
  const psimd_f32 vwCDEF = psimd_load_f32(weights + 12);
  const psimd_f32 vwGHIJ = psimd_load_f32(weights + 16);
  const psimd_f32 vwKLMN = psimd_load_f32(weights + 20);
  const psimd_f32 vwOP   = psimd_load2_f32( weights + 24);

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_width_stride = input_width * sizeof(float);
  const size_t input_decrement = (round_down_po2(input_width - 1, 4) + 4) * sizeof(float);

  const float* i0 = zero;
  const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width_stride));
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width_stride);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width_stride);

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
    for (; w > 8; w -= 8) {
      psimd_f32 vo468Ap00 = psimd_splat0_f32(vw0123);

      psimd_f32 vi0x89AB;
      psimd_f32 vi1x89AB;
      psimd_f32 vi2x89AB;
      psimd_f32 vi3x89AB;
      psimd_f32 vi4x89AB;

      vi0x89AB = psimd_load_f32(i0);
      vi1x89AB = psimd_load_f32(i1);
      vi2x89AB = psimd_load_f32(i2);
      vi3x89AB = psimd_load_f32(i3);
      vi4x89AB = psimd_load_f32(i4);

      psimd_f32 vi0xCDEF;
      psimd_f32 vi1xCDEF;
      psimd_f32 vi2xCDEF;
      psimd_f32 vi3xCDEF;
      psimd_f32 vi4xCDEF;

      vi0xCDEF = psimd_load_f32(i0 + 4);
      i0 += 8;
      vi1xCDEF = psimd_load_f32(i1 + 4);
      i1 += 8;
      vi2xCDEF = psimd_load_f32(i2 + 4);
      i2 += 8;
      vi3xCDEF = psimd_load_f32(i3 + 4);
      i3 += 8;
      vi4xCDEF = psimd_load_f32(i4 + 4);
      i4 += 8;

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

      // middle tap
      vo468Ap00 = vfmaq_lane3_f32(vo468Ap00, vi0x468A, vw0123);
      psimd_f32 vo468Ap01 = vmulq_lane0_f32(vi1x468A, vw89AB);
      vo468Ap00 = vfmaq_lane1_f32(vo468Ap00, vi2x468A, vwCDEF);
      vo468Ap01 = vfmaq_lane2_f32(vo468Ap01, vi3x468A, vwGHIJ);
      vo468Ap00 = vfmaq_lane3_f32(vo468Ap00, vi4x468A, vwKLMN);

      // one left
      const psimd_f32 vi0x3579 = extq3_f32(vi0x0123, vi0x579B);
      const psimd_f32 vi1x3579 = extq3_f32(vi1x0123, vi1x579B);
      const psimd_f32 vi2x3579 = extq3_f32(vi2x0123, vi2x579B);
      const psimd_f32 vi3x3579 = extq3_f32(vi3x0123, vi3x579B);
      const psimd_f32 vi4x3579 = extq3_f32(vi4x0123, vi4x579B);

      vo468Ap00 = vfmaq_lane2_f32(vo468Ap00, vi0x3579, vw0123);
      vo468Ap01 = vfmaq_lane3_f32(vo468Ap01, vi1x3579, vw4567);
      vo468Ap00 = vfmaq_lane0_f32(vo468Ap00, vi2x3579, vwCDEF);
      vo468Ap01 = vfmaq_lane1_f32(vo468Ap01, vi3x3579, vwGHIJ);
      vo468Ap00 = vfmaq_lane2_f32(vo468Ap00, vi4x3579, vwKLMN);

      // two left
      const psimd_f32 vi0x2468 = concat_2456_f32(vi0x0123, vi0x468A);
      const psimd_f32 vi1x2468 = concat_2456_f32(vi1x0123, vi1x468A);
      const psimd_f32 vi2x2468 = concat_2456_f32(vi2x0123, vi2x468A);
      const psimd_f32 vi3x2468 = concat_2456_f32(vi3x0123, vi3x468A);
      const psimd_f32 vi4x2468 = concat_2456_f32(vi4x0123, vi4x468A);

      vo468Ap00 = vfmaq_lane1_f32(vo468Ap00, vi0x2468, vw0123);
      vo468Ap01 = vfmaq_lane2_f32(vo468Ap01, vi1x2468, vw4567);
      vo468Ap00 = vfmaq_lane3_f32(vo468Ap00, vi2x2468, vw89AB);
      vo468Ap01 = vfmaq_lane0_f32(vo468Ap01, vi3x2468, vwGHIJ);
      vo468Ap00 = vfmaq_lane1_f32(vo468Ap00, vi4x2468, vwKLMN);

      vi0x0123 = vi0x89AB;
      vi1x0123 = vi1x89AB;
      vi2x0123 = vi2x89AB;
      vi3x0123 = vi3x89AB;
      vi4x0123 = vi4x89AB;

      // one right
      vo468Ap00 = vfmaq_lane0_f32(vo468Ap00, vi0x579B, vw4567);
      vo468Ap01 = vfmaq_lane1_f32(vo468Ap01, vi1x579B, vw89AB);
      vo468Ap00 = vfmaq_lane2_f32(vo468Ap00, vi2x579B, vwCDEF);
      vo468Ap01 = vfmaq_lane3_f32(vo468Ap01, vi3x579B, vwGHIJ);
      vo468Ap00 = vfmaq_lane0_f32( vo468Ap00, vi4x579B, vwOP);

      // two right
      const psimd_f32 vi0x68AC = extq1_f32(vi0x468A, vi0xCDEF);
      const psimd_f32 vi1x68AC = extq1_f32(vi1x468A, vi1xCDEF);
      const psimd_f32 vi2x68AC = extq1_f32(vi2x468A, vi2xCDEF);
      const psimd_f32 vi3x68AC = extq1_f32(vi3x468A, vi3xCDEF);
      const psimd_f32 vi4x68AC = extq1_f32(vi4x468A, vi4xCDEF);

      vo468Ap00 = vfmaq_lane1_f32(vo468Ap00, vi0x68AC, vw4567);
      vo468Ap01 = vfmaq_lane2_f32(vo468Ap01, vi1x68AC, vw89AB);
      vo468Ap00 = vfmaq_lane3_f32(vo468Ap00, vi2x68AC, vwCDEF);
      vo468Ap01 = vfmaq_lane0_f32(vo468Ap01, vi3x68AC, vwKLMN);
      vo468Ap00 = vfmaq_lane1_f32( vo468Ap00, vi4x68AC, vwOP);

      vi0x4567 = vi0xCDEF;
      vi1x4567 = vi1xCDEF;
      vi2x4567 = vi2xCDEF;
      vi3x4567 = vi3xCDEF;
      vi4x4567 = vi4xCDEF;

      psimd_f32 vo0 = psimd_add_f32(vo468Ap00, vo468Ap01);

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);

      size_t w_tmp = (w + 1) / 2;
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
      psimd_f32 vo468Ap00 = psimd_splat0_f32(vw0123);

      psimd_f32 vi0x89AB;
      psimd_f32 vi1x89AB;
      psimd_f32 vi2x89AB;
      psimd_f32 vi3x89AB;
      psimd_f32 vi4x89AB;

      if XNN_LIKELY(w > 4) {
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
      } else {
        vi0x89AB = psimd_zero_f32();
        vi1x89AB = psimd_zero_f32();
        vi2x89AB = psimd_zero_f32();
        vi3x89AB = psimd_zero_f32();
        vi4x89AB = psimd_zero_f32();
      }

      psimd_f32 vi0xCDEF;
      psimd_f32 vi1xCDEF;
      psimd_f32 vi2xCDEF;
      psimd_f32 vi3xCDEF;
      psimd_f32 vi4xCDEF;

      if XNN_LIKELY(w > 8) {
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
      } else {
        vi0xCDEF = psimd_zero_f32();
        vi1xCDEF = psimd_zero_f32();
        vi2xCDEF = psimd_zero_f32();
        vi3xCDEF = psimd_zero_f32();
        vi4xCDEF = psimd_zero_f32();
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
      vo468Ap00 = vfmaq_lane3_f32(vo468Ap00, vi0x468A, vw0123);
      psimd_f32 vo468Ap01 = vmulq_lane0_f32(vi1x468A, vw89AB);
      vo468Ap00 = vfmaq_lane1_f32(vo468Ap00, vi2x468A, vwCDEF);
      vo468Ap01 = vfmaq_lane2_f32(vo468Ap01, vi3x468A, vwGHIJ);
      vo468Ap00 = vfmaq_lane3_f32(vo468Ap00, vi4x468A, vwKLMN);

      // one left
      const psimd_f32 vi0x3579 = extq3_f32(vi0x0123, vi0x579B);
      const psimd_f32 vi1x3579 = extq3_f32(vi1x0123, vi1x579B);
      const psimd_f32 vi2x3579 = extq3_f32(vi2x0123, vi2x579B);
      const psimd_f32 vi3x3579 = extq3_f32(vi3x0123, vi3x579B);
      const psimd_f32 vi4x3579 = extq3_f32(vi4x0123, vi4x579B);

      vo468Ap00 = vfmaq_lane2_f32(vo468Ap00, vi0x3579, vw0123);
      vo468Ap01 = vfmaq_lane3_f32(vo468Ap01, vi1x3579, vw4567);
      vo468Ap00 = vfmaq_lane0_f32(vo468Ap00, vi2x3579, vwCDEF);
      vo468Ap01 = vfmaq_lane1_f32(vo468Ap01, vi3x3579, vwGHIJ);
      vo468Ap00 = vfmaq_lane2_f32(vo468Ap00, vi4x3579, vwKLMN);

      // two left
      const psimd_f32 vi0x2468 = concat_2456_f32(vi0x0123, vi0x468A);
      const psimd_f32 vi1x2468 = concat_2456_f32(vi1x0123, vi1x468A);
      const psimd_f32 vi2x2468 = concat_2456_f32(vi2x0123, vi2x468A);
      const psimd_f32 vi3x2468 = concat_2456_f32(vi3x0123, vi3x468A);
      const psimd_f32 vi4x2468 = concat_2456_f32(vi4x0123, vi4x468A);

      vo468Ap00 = vfmaq_lane1_f32(vo468Ap00, vi0x2468, vw0123);
      vo468Ap01 = vfmaq_lane2_f32(vo468Ap01, vi1x2468, vw4567);
      vo468Ap00 = vfmaq_lane3_f32(vo468Ap00, vi2x2468, vw89AB);
      vo468Ap01 = vfmaq_lane0_f32(vo468Ap01, vi3x2468, vwGHIJ);
      vo468Ap00 = vfmaq_lane1_f32(vo468Ap00, vi4x2468, vwKLMN);

      vi0x0123 = vi0x89AB;
      vi1x0123 = vi1x89AB;
      vi2x0123 = vi2x89AB;
      vi3x0123 = vi3x89AB;
      vi4x0123 = vi4x89AB;

      // one right
      vo468Ap00 = vfmaq_lane0_f32(vo468Ap00, vi0x579B, vw4567);
      vo468Ap01 = vfmaq_lane1_f32(vo468Ap01, vi1x579B, vw89AB);
      vo468Ap00 = vfmaq_lane2_f32(vo468Ap00, vi2x579B, vwCDEF);
      vo468Ap01 = vfmaq_lane3_f32(vo468Ap01, vi3x579B, vwGHIJ);
      vo468Ap00 = vfmaq_lane0_f32( vo468Ap00, vi4x579B, vwOP);

      // two right
      const psimd_f32 vi0x68AC = extq1_f32(vi0x468A, vi0xCDEF);
      const psimd_f32 vi1x68AC = extq1_f32(vi1x468A, vi1xCDEF);
      const psimd_f32 vi2x68AC = extq1_f32(vi2x468A, vi2xCDEF);
      const psimd_f32 vi3x68AC = extq1_f32(vi3x468A, vi3xCDEF);
      const psimd_f32 vi4x68AC = extq1_f32(vi4x468A, vi4xCDEF);

      vo468Ap00 = vfmaq_lane1_f32(vo468Ap00, vi0x68AC, vw4567);
      vo468Ap01 = vfmaq_lane2_f32(vo468Ap01, vi1x68AC, vw89AB);
      vo468Ap00 = vfmaq_lane3_f32(vo468Ap00, vi2x68AC, vwCDEF);
      vo468Ap01 = vfmaq_lane0_f32(vo468Ap01, vi3x68AC, vwKLMN);
      vo468Ap00 = vfmaq_lane1_f32( vo468Ap00, vi4x68AC, vwOP);

      vi0x4567 = vi0xCDEF;
      vi1x4567 = vi1xCDEF;
      vi2x4567 = vi2xCDEF;
      vi3x4567 = vi3xCDEF;
      vi4x4567 = vi4xCDEF;

      psimd_f32 vo0 = psimd_add_f32(vo468Ap00, vo468Ap01);

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);

      size_t w_tmp = (w + 1) / 2;
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
    i3 = (const float*) ((uintptr_t) i2 + input_width_stride);
    i4 = (const float*) ((uintptr_t) i3 + input_width_stride);

    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}
