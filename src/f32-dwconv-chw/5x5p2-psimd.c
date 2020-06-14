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

void xnn_f32_dwconv_chw_ukernel_5x5p2__psimd(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float *zero,
    float* output,
    uint32_t padding_top,
    size_t input_tuple_stride,
    size_t output_tuple_stride,
    size_t input_width_stride,
    size_t output_width_stride,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_width != 0);
  assert(input_height != 0);
  assert(padding_top == 2);

  const size_t padded_input_height = input_height + padding_top + 2 /* padding_bottom */;
  const size_t output_height = padded_input_height - 5 + 1;

  const psimd_s32 vmask = psimd_load_s32(params->scalar.mask);
  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);

  const size_t input_width_increment_single = input_width_stride - round_up_po2(input_width, 4) / 4 * input_tuple_stride;
  const size_t input_width_increment_triple = 3  * input_width_stride - round_up_po2(input_width, 4) / 4 * input_tuple_stride;
  const size_t output_width_increment_single = output_width_stride - (input_width - 1) / 4 * output_tuple_stride;

  const float* i0 = zero;
  const float* i1 = zero;;
  const float* i2 = input;
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width_stride);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width_stride);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width_stride);
  if (input_height <= 4) {
    i6 = zero;
  }
  if (input_height <= 3) {
    i5 = zero;
  }
  if (input_height <= 2) {
    i4 = zero;
  }
  if (input_height <= 1) {
    i3 = zero;
  }

  float* output0 = output;
  float* output1 = (float *)((uintptr_t) output0 + output_width_stride);
  float* output2 = (float *)((uintptr_t) output1 + output_width_stride);

  const psimd_f32 vw0123 = psimd_load_f32(weights);
  const psimd_f32 vw4567 = psimd_load_f32(weights + 4);
  const psimd_f32 vw89AB = psimd_load_f32(weights + 8);
  const psimd_f32 vwCDEF = psimd_load_f32(weights + 12);
  const psimd_f32 vwGHIJ = psimd_load_f32(weights + 16);
  const psimd_f32 vwKLMN = psimd_load_f32(weights + 20);
  const psimd_f32 vwOP   = psimd_load2_f32(weights + 24);

  size_t m = output_height;
  while (m >= 3) {
    psimd_f32 vi0x0123 = psimd_zero_f32();
    psimd_f32 vi1x0123 = psimd_zero_f32();
    psimd_f32 vi2x0123 = psimd_zero_f32();
    psimd_f32 vi3x0123 = psimd_zero_f32();
    psimd_f32 vi4x0123 = psimd_zero_f32();
    psimd_f32 vi5x0123 = psimd_zero_f32();
    psimd_f32 vi6x0123 = psimd_zero_f32();
    psimd_f32 vi0x4567 = psimd_load_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
    psimd_f32 vi1x4567 = psimd_load_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
    psimd_f32 vi2x4567 = psimd_load_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
    psimd_f32 vi3x4567 = psimd_load_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
    psimd_f32 vi4x4567 = psimd_load_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
    psimd_f32 vi5x4567 = psimd_load_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);
    psimd_f32 vi6x4567 = psimd_load_f32(i6); i6 = (const float*) ((uintptr_t) i6 + input_tuple_stride);

    size_t k = input_width;
    for (; k > 8; k -= 4) {
      psimd_f32 vo4567p00 = psimd_splat0_f32(vw0123);
      psimd_f32 vo4567p10 = psimd_splat0_f32(vw0123);
      psimd_f32 vo4567p20 = psimd_splat0_f32(vw0123);

      const psimd_f32 vi0x89AB = psimd_load_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const psimd_f32 vi1x89AB = psimd_load_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const psimd_f32 vi2x89AB = psimd_load_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      const psimd_f32 vi3x89AB = psimd_load_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      const psimd_f32 vi4x89AB = psimd_load_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
      const psimd_f32 vi5x89AB = psimd_load_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);
      const psimd_f32 vi6x89AB = psimd_load_f32(i6); i6 = (const float*) ((uintptr_t) i6 + input_tuple_stride);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi0x4567, vw0123);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi1x4567, vw0123);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi2x4567, vw0123);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi1x4567, vw89AB);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi2x4567, vw89AB);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi3x4567, vw89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi2x4567, vwCDEF);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi3x4567, vwCDEF);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi4x4567, vwCDEF);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi3x4567, vwGHIJ);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi4x4567, vwGHIJ);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi5x4567, vwGHIJ);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi4x4567, vwKLMN);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi5x4567, vwKLMN);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi6x4567, vwKLMN);


      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x3456 = extq3_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x3456 = extq3_f32(vi6x0123, vi6x4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi0x3456, vw0123);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi1x3456, vw0123);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi2x3456, vw0123);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi1x3456, vw4567);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi2x3456, vw4567);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi3x3456, vw4567);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi2x3456, vwCDEF);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi3x3456, vwCDEF);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi4x3456, vwCDEF);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi3x3456, vwGHIJ);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi4x3456, vwGHIJ);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi5x3456, vwGHIJ);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi4x3456, vwKLMN);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi5x3456, vwKLMN);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi6x3456, vwKLMN);

      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x2345 = extq2_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x2345 = extq2_f32(vi6x0123, vi6x4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x2345, vw0123);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x2345, vw0123);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi2x2345, vw0123);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x2345, vw4567);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x2345, vw4567);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi3x2345, vw4567);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x2345, vw89AB);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x2345, vw89AB);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi4x2345, vw89AB);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x2345, vwGHIJ);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x2345, vwGHIJ);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi5x2345, vwGHIJ);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi4x2345, vwKLMN);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi5x2345, vwKLMN);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi6x2345, vwKLMN);

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

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi0x5678, vw4567);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi1x5678, vw4567);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi2x5678, vw4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi1x5678, vw89AB);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi2x5678, vw89AB);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi3x5678, vw89AB);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi2x5678, vwCDEF);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi3x5678, vwCDEF);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi4x5678, vwCDEF);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi3x5678, vwGHIJ);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi4x5678, vwGHIJ);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi5x5678, vwGHIJ);

      vo4567p00 = vfmaq_lane0_f32( vo4567p00, vi4x5678, vwOP);
      vo4567p10 = vfmaq_lane0_f32( vo4567p10, vi5x5678, vwOP);
      vo4567p20 = vfmaq_lane0_f32( vo4567p20, vi6x5678, vwOP);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vi4x89AB);
      const psimd_f32 vi5x6789 = extq2_f32(vi5x4567, vi5x89AB);
      const psimd_f32 vi6x6789 = extq2_f32(vi6x4567, vi6x89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x6789, vw4567);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x6789, vw4567);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi2x6789, vw4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x6789, vw89AB);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x6789, vw89AB);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi3x6789, vw89AB);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x6789, vwCDEF);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x6789, vwCDEF);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi4x6789, vwCDEF);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x6789, vwKLMN);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x6789, vwKLMN);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi5x6789, vwKLMN);

      vo4567p00 = vfmaq_lane1_f32( vo4567p00, vi4x6789, vwOP);
      vo4567p10 = vfmaq_lane1_f32( vo4567p10, vi5x6789, vwOP);
      vo4567p20 = vfmaq_lane1_f32( vo4567p20, vi6x6789, vwOP);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;
      vi6x4567 = vi6x89AB;

      psimd_f32 vo0 = vo4567p00;
      psimd_f32 vo1 = vo4567p10;
      psimd_f32 vo2 = vo4567p20;

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);
      vo1 = psimd_max_f32(vo1, vmin);
      vo1 = psimd_min_f32(vo1, vmax);
      vo2 = psimd_max_f32(vo2, vmin);
      vo2 = psimd_min_f32(vo2, vmax);

      psimd_store_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
      psimd_store_f32(output1, vo1); output1 = (float*) ((uintptr_t) output1 + output_tuple_stride);
      psimd_store_f32(output2, vo2); output2 = (float*) ((uintptr_t) output2 + output_tuple_stride);
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(k > 4)
    {
      psimd_f32 vo4567p00 = psimd_splat0_f32(vw0123);
      psimd_f32 vo4567p10 = psimd_splat0_f32(vw0123);
      psimd_f32 vo4567p20 = psimd_splat0_f32(vw0123);

      psimd_f32 vi0x89AB = psimd_load_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      psimd_f32 vi1x89AB = psimd_load_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      psimd_f32 vi2x89AB = psimd_load_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      psimd_f32 vi3x89AB = psimd_load_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      psimd_f32 vi4x89AB = psimd_load_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
      psimd_f32 vi5x89AB = psimd_load_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);
      psimd_f32 vi6x89AB = psimd_load_f32(i6); i6 = (const float*) ((uintptr_t) i6 + input_tuple_stride);

      vi0x89AB = psimd_andmask_f32(vmask, vi0x89AB);
      vi1x89AB = psimd_andmask_f32(vmask, vi1x89AB);
      vi2x89AB = psimd_andmask_f32(vmask, vi2x89AB);
      vi3x89AB = psimd_andmask_f32(vmask, vi3x89AB);
      vi4x89AB = psimd_andmask_f32(vmask, vi4x89AB);
      vi5x89AB = psimd_andmask_f32(vmask, vi5x89AB);
      vi6x89AB = psimd_andmask_f32(vmask, vi6x89AB);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi0x4567, vw0123);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi1x4567, vw0123);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi2x4567, vw0123);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi1x4567, vw89AB);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi2x4567, vw89AB);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi3x4567, vw89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi2x4567, vwCDEF);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi3x4567, vwCDEF);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi4x4567, vwCDEF);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi3x4567, vwGHIJ);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi4x4567, vwGHIJ);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi5x4567, vwGHIJ);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi4x4567, vwKLMN);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi5x4567, vwKLMN);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi6x4567, vwKLMN);


      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x3456 = extq3_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x3456 = extq3_f32(vi6x0123, vi6x4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi0x3456, vw0123);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi1x3456, vw0123);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi2x3456, vw0123);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi1x3456, vw4567);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi2x3456, vw4567);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi3x3456, vw4567);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi2x3456, vwCDEF);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi3x3456, vwCDEF);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi4x3456, vwCDEF);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi3x3456, vwGHIJ);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi4x3456, vwGHIJ);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi5x3456, vwGHIJ);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi4x3456, vwKLMN);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi5x3456, vwKLMN);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi6x3456, vwKLMN);


      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x2345 = extq2_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x2345 = extq2_f32(vi6x0123, vi6x4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x2345, vw0123);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x2345, vw0123);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi2x2345, vw0123);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x2345, vw4567);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x2345, vw4567);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi3x2345, vw4567);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x2345, vw89AB);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x2345, vw89AB);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi4x2345, vw89AB);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x2345, vwGHIJ);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x2345, vwGHIJ);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi5x2345, vwGHIJ);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi4x2345, vwKLMN);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi5x2345, vwKLMN);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi6x2345, vwKLMN);


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

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi0x5678, vw4567);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi1x5678, vw4567);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi2x5678, vw4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi1x5678, vw89AB);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi2x5678, vw89AB);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi3x5678, vw89AB);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi2x5678, vwCDEF);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi3x5678, vwCDEF);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi4x5678, vwCDEF);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi3x5678, vwGHIJ);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi4x5678, vwGHIJ);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi5x5678, vwGHIJ);

      vo4567p00 = vfmaq_lane0_f32( vo4567p00, vi4x5678, vwOP);
      vo4567p10 = vfmaq_lane0_f32( vo4567p10, vi5x5678, vwOP);
      vo4567p20 = vfmaq_lane0_f32( vo4567p20, vi6x5678, vwOP);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vi4x89AB);
      const psimd_f32 vi5x6789 = extq2_f32(vi5x4567, vi5x89AB);
      const psimd_f32 vi6x6789 = extq2_f32(vi6x4567, vi6x89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x6789, vw4567);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x6789, vw4567);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi2x6789, vw4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x6789, vw89AB);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x6789, vw89AB);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi3x6789, vw89AB);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x6789, vwCDEF);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x6789, vwCDEF);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi4x6789, vwCDEF);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x6789, vwKLMN);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x6789, vwKLMN);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi5x6789, vwKLMN);

      vo4567p00 = vfmaq_lane1_f32( vo4567p00, vi4x6789, vwOP);
      vo4567p10 = vfmaq_lane1_f32( vo4567p10, vi5x6789, vwOP);
      vo4567p20 = vfmaq_lane1_f32( vo4567p20, vi6x6789, vwOP);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;
      vi6x4567 = vi6x89AB;

      psimd_f32 vo0 = vo4567p00;
      psimd_f32 vo1 = vo4567p10;
      psimd_f32 vo2 = vo4567p20;

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);
      vo1 = psimd_max_f32(vo1, vmin);
      vo1 = psimd_min_f32(vo1, vmax);
      vo2 = psimd_max_f32(vo2, vmin);
      vo2 = psimd_min_f32(vo2, vmax);

      psimd_store_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
      psimd_store_f32(output1, vo1); output1 = (float*) ((uintptr_t) output1 + output_tuple_stride);
      psimd_store_f32(output2, vo2); output2 = (float*) ((uintptr_t) output2 + output_tuple_stride);
      k -= 4;
    }
    assert(k >= 1);
    assert(k <= 4);
    {
      psimd_f32 vo4567p00 = psimd_splat0_f32(vw0123);
      psimd_f32 vo4567p10 = psimd_splat0_f32(vw0123);
      psimd_f32 vo4567p20 = psimd_splat0_f32(vw0123);

      // This might have already happened if there are more than 4 pixels, but
      // we can't count on it.
      vi0x4567 = psimd_andmask_f32(vmask, vi0x4567);
      vi1x4567 = psimd_andmask_f32(vmask, vi1x4567);
      vi2x4567 = psimd_andmask_f32(vmask, vi2x4567);
      vi3x4567 = psimd_andmask_f32(vmask, vi3x4567);
      vi4x4567 = psimd_andmask_f32(vmask, vi4x4567);
      vi5x4567 = psimd_andmask_f32(vmask, vi5x4567);
      vi6x4567 = psimd_andmask_f32(vmask, vi6x4567);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi0x4567, vw0123);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi1x4567, vw0123);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi2x4567, vw0123);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi1x4567, vw89AB);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi2x4567, vw89AB);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi3x4567, vw89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi2x4567, vwCDEF);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi3x4567, vwCDEF);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi4x4567, vwCDEF);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi3x4567, vwGHIJ);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi4x4567, vwGHIJ);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi5x4567, vwGHIJ);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi4x4567, vwKLMN);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi5x4567, vwKLMN);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi6x4567, vwKLMN);


      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x3456 = extq3_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x3456 = extq3_f32(vi6x0123, vi6x4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi0x3456, vw0123);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi1x3456, vw0123);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi2x3456, vw0123);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi1x3456, vw4567);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi2x3456, vw4567);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi3x3456, vw4567);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi2x3456, vwCDEF);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi3x3456, vwCDEF);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi4x3456, vwCDEF);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi3x3456, vwGHIJ);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi4x3456, vwGHIJ);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi5x3456, vwGHIJ);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi4x3456, vwKLMN);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi5x3456, vwKLMN);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi6x3456, vwKLMN);

      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x2345 = extq2_f32(vi5x0123, vi5x4567);
      const psimd_f32 vi6x2345 = extq2_f32(vi6x0123, vi6x4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x2345, vw0123);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x2345, vw0123);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi2x2345, vw0123);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x2345, vw4567);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x2345, vw4567);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi3x2345, vw4567);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x2345, vw89AB);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x2345, vw89AB);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi4x2345, vw89AB);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x2345, vwGHIJ);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x2345, vwGHIJ);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi5x2345, vwGHIJ);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi4x2345, vwKLMN);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi5x2345, vwKLMN);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi6x2345, vwKLMN);


      const psimd_f32 vzero = psimd_zero_f32();
      const psimd_f32 vi0x5678 = extq1_f32(vi0x4567, vzero);
      const psimd_f32 vi1x5678 = extq1_f32(vi1x4567, vzero);
      const psimd_f32 vi2x5678 = extq1_f32(vi2x4567, vzero);
      const psimd_f32 vi3x5678 = extq1_f32(vi3x4567, vzero);
      const psimd_f32 vi4x5678 = extq1_f32(vi4x4567, vzero);
      const psimd_f32 vi5x5678 = extq1_f32(vi5x4567, vzero);
      const psimd_f32 vi6x5678 = extq1_f32(vi6x4567, vzero);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi0x5678, vw4567);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi1x5678, vw4567);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi2x5678, vw4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi1x5678, vw89AB);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi2x5678, vw89AB);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi3x5678, vw89AB);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi2x5678, vwCDEF);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi3x5678, vwCDEF);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi4x5678, vwCDEF);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi3x5678, vwGHIJ);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi4x5678, vwGHIJ);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi5x5678, vwGHIJ);

      vo4567p00 = vfmaq_lane0_f32( vo4567p00, vi4x5678, vwOP);
      vo4567p10 = vfmaq_lane0_f32( vo4567p10, vi5x5678, vwOP);
      vo4567p20 = vfmaq_lane0_f32( vo4567p20, vi6x5678, vwOP);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vzero);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vzero);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vzero);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vzero);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vzero);
      const psimd_f32 vi5x6789 = extq2_f32(vi5x4567, vzero);
      const psimd_f32 vi6x6789 = extq2_f32(vi6x4567, vzero);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x6789, vw4567);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x6789, vw4567);
      vo4567p20 = vfmaq_lane1_f32(vo4567p20, vi2x6789, vw4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x6789, vw89AB);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x6789, vw89AB);
      vo4567p20 = vfmaq_lane2_f32(vo4567p20, vi3x6789, vw89AB);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x6789, vwCDEF);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x6789, vwCDEF);
      vo4567p20 = vfmaq_lane3_f32(vo4567p20, vi4x6789, vwCDEF);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x6789, vwKLMN);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x6789, vwKLMN);
      vo4567p20 = vfmaq_lane0_f32(vo4567p20, vi5x6789, vwKLMN);

      vo4567p00 = vfmaq_lane1_f32( vo4567p00, vi4x6789, vwOP);
      vo4567p10 = vfmaq_lane1_f32( vo4567p10, vi5x6789, vwOP);
      vo4567p20 = vfmaq_lane1_f32( vo4567p20, vi6x6789, vwOP);

      psimd_f32 vo0 = vo4567p00;
      psimd_f32 vo1 = vo4567p10;
      psimd_f32 vo2 = vo4567p20;

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);
      vo1 = psimd_max_f32(vo1, vmin);
      vo1 = psimd_min_f32(vo1, vmax);
      vo2 = psimd_max_f32(vo2, vmin);
      vo2 = psimd_min_f32(vo2, vmax);

      if XNN_LIKELY(k & 4) {
        psimd_store_f32(output0, vo0);
        psimd_store_f32(output1, vo1);
        psimd_store_f32(output2, vo2);
      } else {
        float* output0_lo = output0;
        float* output1_lo = output1;
        float* output2_lo = output2;
        if (k & 2) {
          psimd_store2_f32(output0_lo, vo0); output0_lo += 2;
          psimd_store2_f32(output1_lo, vo1); output1_lo += 2;
          psimd_store2_f32(output2_lo, vo2); output2_lo += 2;
          vo0 = psimd_splat2_f32(vo0);
          vo1 = psimd_splat2_f32(vo1);
          vo2 = psimd_splat2_f32(vo2);
        }
        if (k & 1) {
          psimd_store1_f32(output0_lo, vo0);
          psimd_store1_f32(output1_lo, vo1);
          psimd_store1_f32(output2_lo, vo2);
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 + input_width_increment_single);
    i1 = (const float*) ((uintptr_t) i3 + input_width_increment_single);
    i2 = (const float*) ((uintptr_t) i2 + input_width_increment_triple);
    i3 = (const float*) ((uintptr_t) i3 + input_width_increment_triple);
    i4 = (const float*) ((uintptr_t) i4 + input_width_increment_triple);
    i5 = (const float*) ((uintptr_t) i5 + input_width_increment_triple);
    i6 = (const float*) ((uintptr_t) i6 + input_width_increment_triple);
    output0 = (float*) ((uintptr_t) output2 + output_width_increment_single);
    output1 = (float*) ((uintptr_t) output0 + output_width_stride);
    output2 = (float*) ((uintptr_t) output1 + output_width_stride);
    m -= 3;
    if (m <= 4) {
      i6 = zero;
    }
    if (m <= 3) {
      i5 = zero;
    }
    if (m <= 2) {
      i4 = zero;
    }
    if (m <= 1) {
      i3 = zero;
    }
  }

  if (m == 2) {
    i4 = i5 = zero;
    psimd_f32 vi0x0123 = psimd_zero_f32();
    psimd_f32 vi1x0123 = psimd_zero_f32();
    psimd_f32 vi2x0123 = psimd_zero_f32();
    psimd_f32 vi3x0123 = psimd_zero_f32();
    psimd_f32 vi4x0123 = psimd_zero_f32();
    psimd_f32 vi5x0123 = psimd_zero_f32();
    psimd_f32 vi0x4567 = psimd_load_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
    psimd_f32 vi1x4567 = psimd_load_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
    psimd_f32 vi2x4567 = psimd_load_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
    psimd_f32 vi3x4567 = psimd_load_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
    psimd_f32 vi4x4567 = psimd_load_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
    psimd_f32 vi5x4567 = psimd_load_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);

    size_t k = input_width;
    for (; k > 8; k -= 4) {
      psimd_f32 vo4567p00 = psimd_splat0_f32(vw0123);
      psimd_f32 vo4567p10 = psimd_splat0_f32(vw0123);

      const psimd_f32 vi0x89AB = psimd_load_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const psimd_f32 vi1x89AB = psimd_load_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const psimd_f32 vi2x89AB = psimd_load_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      const psimd_f32 vi3x89AB = psimd_load_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      const psimd_f32 vi4x89AB = psimd_load_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
      const psimd_f32 vi5x89AB = psimd_load_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi0x4567, vw0123);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi1x4567, vw0123);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi1x4567, vw89AB);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi2x4567, vw89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi2x4567, vwCDEF);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi3x4567, vwCDEF);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi3x4567, vwGHIJ);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi4x4567, vwGHIJ);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi4x4567, vwKLMN);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi5x4567, vwKLMN);


      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x3456 = extq3_f32(vi5x0123, vi5x4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi0x3456, vw0123);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi1x3456, vw0123);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi1x3456, vw4567);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi2x3456, vw4567);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi2x3456, vwCDEF);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi3x3456, vwCDEF);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi3x3456, vwGHIJ);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi4x3456, vwGHIJ);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi4x3456, vwKLMN);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi5x3456, vwKLMN);

      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x2345 = extq2_f32(vi5x0123, vi5x4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x2345, vw0123);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x2345, vw0123);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x2345, vw4567);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x2345, vw4567);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x2345, vw89AB);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x2345, vw89AB);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x2345, vwGHIJ);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x2345, vwGHIJ);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi4x2345, vwKLMN);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi5x2345, vwKLMN);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;

      const psimd_f32 vi0x5678 = extq1_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x5678 = extq1_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x5678 = extq1_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x5678 = extq1_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x5678 = extq1_f32(vi4x4567, vi4x89AB);
      const psimd_f32 vi5x5678 = extq1_f32(vi5x4567, vi5x89AB);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi0x5678, vw4567);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi1x5678, vw4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi1x5678, vw89AB);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi2x5678, vw89AB);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi2x5678, vwCDEF);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi3x5678, vwCDEF);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi3x5678, vwGHIJ);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi4x5678, vwGHIJ);

      vo4567p00 = vfmaq_lane0_f32( vo4567p00, vi4x5678, vwOP);
      vo4567p10 = vfmaq_lane0_f32( vo4567p10, vi5x5678, vwOP);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vi4x89AB);
      const psimd_f32 vi5x6789 = extq2_f32(vi5x4567, vi5x89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x6789, vw4567);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x6789, vw4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x6789, vw89AB);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x6789, vw89AB);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x6789, vwCDEF);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x6789, vwCDEF);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x6789, vwKLMN);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x6789, vwKLMN);

      vo4567p00 = vfmaq_lane1_f32( vo4567p00, vi4x6789, vwOP);
      vo4567p10 = vfmaq_lane1_f32( vo4567p10, vi5x6789, vwOP);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;

      psimd_f32 vo0 = vo4567p00;
      psimd_f32 vo1 = vo4567p10;

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);
      vo1 = psimd_max_f32(vo1, vmin);
      vo1 = psimd_min_f32(vo1, vmax);

      psimd_store_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
      psimd_store_f32(output1, vo1); output1 = (float*) ((uintptr_t) output1 + output_tuple_stride);
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(k > 4)
    {
      psimd_f32 vo4567p00 = psimd_splat0_f32(vw0123);
      psimd_f32 vo4567p10 = psimd_splat0_f32(vw0123);

      psimd_f32 vi0x89AB = psimd_load_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      psimd_f32 vi1x89AB = psimd_load_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      psimd_f32 vi2x89AB = psimd_load_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      psimd_f32 vi3x89AB = psimd_load_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      psimd_f32 vi4x89AB = psimd_load_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
      psimd_f32 vi5x89AB = psimd_load_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);

      vi0x89AB = psimd_andmask_f32(vmask, vi0x89AB);
      vi1x89AB = psimd_andmask_f32(vmask, vi1x89AB);
      vi2x89AB = psimd_andmask_f32(vmask, vi2x89AB);
      vi3x89AB = psimd_andmask_f32(vmask, vi3x89AB);
      vi4x89AB = psimd_andmask_f32(vmask, vi4x89AB);
      vi5x89AB = psimd_andmask_f32(vmask, vi5x89AB);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi0x4567, vw0123);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi1x4567, vw0123);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi1x4567, vw89AB);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi2x4567, vw89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi2x4567, vwCDEF);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi3x4567, vwCDEF);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi3x4567, vwGHIJ);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi4x4567, vwGHIJ);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi4x4567, vwKLMN);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi5x4567, vwKLMN);


      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x3456 = extq3_f32(vi5x0123, vi5x4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi0x3456, vw0123);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi1x3456, vw0123);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi1x3456, vw4567);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi2x3456, vw4567);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi2x3456, vwCDEF);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi3x3456, vwCDEF);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi3x3456, vwGHIJ);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi4x3456, vwGHIJ);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi4x3456, vwKLMN);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi5x3456, vwKLMN);


      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x2345 = extq2_f32(vi5x0123, vi5x4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x2345, vw0123);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x2345, vw0123);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x2345, vw4567);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x2345, vw4567);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x2345, vw89AB);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x2345, vw89AB);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x2345, vwGHIJ);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x2345, vwGHIJ);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi4x2345, vwKLMN);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi5x2345, vwKLMN);


      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;

      const psimd_f32 vi0x5678 = extq1_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x5678 = extq1_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x5678 = extq1_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x5678 = extq1_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x5678 = extq1_f32(vi4x4567, vi4x89AB);
      const psimd_f32 vi5x5678 = extq1_f32(vi5x4567, vi5x89AB);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi0x5678, vw4567);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi1x5678, vw4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi1x5678, vw89AB);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi2x5678, vw89AB);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi2x5678, vwCDEF);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi3x5678, vwCDEF);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi3x5678, vwGHIJ);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi4x5678, vwGHIJ);

      vo4567p00 = vfmaq_lane0_f32( vo4567p00, vi4x5678, vwOP);
      vo4567p10 = vfmaq_lane0_f32( vo4567p10, vi5x5678, vwOP);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vi4x89AB);
      const psimd_f32 vi5x6789 = extq2_f32(vi5x4567, vi5x89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x6789, vw4567);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x6789, vw4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x6789, vw89AB);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x6789, vw89AB);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x6789, vwCDEF);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x6789, vwCDEF);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x6789, vwKLMN);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x6789, vwKLMN);

      vo4567p00 = vfmaq_lane1_f32( vo4567p00, vi4x6789, vwOP);
      vo4567p10 = vfmaq_lane1_f32( vo4567p10, vi5x6789, vwOP);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;

      psimd_f32 vo0 = vo4567p00;
      psimd_f32 vo1 = vo4567p10;

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);
      vo1 = psimd_max_f32(vo1, vmin);
      vo1 = psimd_min_f32(vo1, vmax);

      psimd_store_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
      psimd_store_f32(output1, vo1); output1 = (float*) ((uintptr_t) output1 + output_tuple_stride);
      k -= 4;
    }
    assert(k >= 1);
    assert(k <= 4);
    {
      psimd_f32 vo4567p00 = psimd_splat0_f32(vw0123);
      psimd_f32 vo4567p10 = psimd_splat0_f32(vw0123);

      // This might have already happened if there are more than 4 pixels, but
      // we can't count on it.
      vi0x4567 = psimd_andmask_f32(vmask, vi0x4567);
      vi1x4567 = psimd_andmask_f32(vmask, vi1x4567);
      vi2x4567 = psimd_andmask_f32(vmask, vi2x4567);
      vi3x4567 = psimd_andmask_f32(vmask, vi3x4567);
      vi4x4567 = psimd_andmask_f32(vmask, vi4x4567);
      vi5x4567 = psimd_andmask_f32(vmask, vi5x4567);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi0x4567, vw0123);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi1x4567, vw0123);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi1x4567, vw89AB);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi2x4567, vw89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi2x4567, vwCDEF);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi3x4567, vwCDEF);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi3x4567, vwGHIJ);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi4x4567, vwGHIJ);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi4x4567, vwKLMN);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi5x4567, vwKLMN);


      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x3456 = extq3_f32(vi5x0123, vi5x4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi0x3456, vw0123);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi1x3456, vw0123);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi1x3456, vw4567);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi2x3456, vw4567);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi2x3456, vwCDEF);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi3x3456, vwCDEF);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi3x3456, vwGHIJ);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi4x3456, vwGHIJ);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi4x3456, vwKLMN);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi5x3456, vwKLMN);

      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);
      const psimd_f32 vi5x2345 = extq2_f32(vi5x0123, vi5x4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x2345, vw0123);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x2345, vw0123);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x2345, vw4567);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x2345, vw4567);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x2345, vw89AB);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x2345, vw89AB);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x2345, vwGHIJ);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x2345, vwGHIJ);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi4x2345, vwKLMN);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi5x2345, vwKLMN);

      const psimd_f32 vzero = psimd_zero_f32();
      const psimd_f32 vi0x5678 = extq1_f32(vi0x4567, vzero);
      const psimd_f32 vi1x5678 = extq1_f32(vi1x4567, vzero);
      const psimd_f32 vi2x5678 = extq1_f32(vi2x4567, vzero);
      const psimd_f32 vi3x5678 = extq1_f32(vi3x4567, vzero);
      const psimd_f32 vi4x5678 = extq1_f32(vi4x4567, vzero);
      const psimd_f32 vi5x5678 = extq1_f32(vi5x4567, vzero);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi0x5678, vw4567);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi1x5678, vw4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi1x5678, vw89AB);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi2x5678, vw89AB);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi2x5678, vwCDEF);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi3x5678, vwCDEF);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi3x5678, vwGHIJ);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi4x5678, vwGHIJ);

      vo4567p00 = vfmaq_lane0_f32( vo4567p00, vi4x5678, vwOP);
      vo4567p10 = vfmaq_lane0_f32( vo4567p10, vi5x5678, vwOP);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vzero);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vzero);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vzero);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vzero);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vzero);
      const psimd_f32 vi5x6789 = extq2_f32(vi5x4567, vzero);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x6789, vw4567);
      vo4567p10 = vfmaq_lane1_f32(vo4567p10, vi1x6789, vw4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x6789, vw89AB);
      vo4567p10 = vfmaq_lane2_f32(vo4567p10, vi2x6789, vw89AB);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x6789, vwCDEF);
      vo4567p10 = vfmaq_lane3_f32(vo4567p10, vi3x6789, vwCDEF);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x6789, vwKLMN);
      vo4567p10 = vfmaq_lane0_f32(vo4567p10, vi4x6789, vwKLMN);

      vo4567p00 = vfmaq_lane1_f32( vo4567p00, vi4x6789, vwOP);
      vo4567p10 = vfmaq_lane1_f32( vo4567p10, vi5x6789, vwOP);

      psimd_f32 vo0 = vo4567p00;
      psimd_f32 vo1 = vo4567p10;

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);
      vo1 = psimd_max_f32(vo1, vmin);
      vo1 = psimd_min_f32(vo1, vmax);

      if XNN_LIKELY(k & 4) {
        psimd_store_f32(output0, vo0);
        psimd_store_f32(output1, vo1);
      } else {
        float* output0_lo = output0;
        float* output1_lo = output1;
        if (k & 2) {
          psimd_store2_f32(output0_lo, vo0); output0_lo += 2;
          psimd_store2_f32(output1_lo, vo1); output1_lo += 2;
          vo0 = psimd_splat2_f32(vo0);
          vo1 = psimd_splat2_f32(vo1);
        }
        if (k & 1) {
          psimd_store1_f32(output0_lo, vo0);
          psimd_store1_f32(output1_lo, vo1);
        }
      }
    }
  }
  else if (m == 1) {
    i3 = i4 = zero;
    psimd_f32 vi0x0123 = psimd_zero_f32();
    psimd_f32 vi1x0123 = psimd_zero_f32();
    psimd_f32 vi2x0123 = psimd_zero_f32();
    psimd_f32 vi3x0123 = psimd_zero_f32();
    psimd_f32 vi4x0123 = psimd_zero_f32();
    psimd_f32 vi0x4567 = psimd_load_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
    psimd_f32 vi1x4567 = psimd_load_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
    psimd_f32 vi2x4567 = psimd_load_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
    psimd_f32 vi3x4567 = psimd_load_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
    psimd_f32 vi4x4567 = psimd_load_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);

    size_t k = input_width;
    for (; k > 8; k -= 4) {
      psimd_f32 vo4567p00 = psimd_splat0_f32(vw0123);

      const psimd_f32 vi0x89AB = psimd_load_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const psimd_f32 vi1x89AB = psimd_load_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const psimd_f32 vi2x89AB = psimd_load_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      const psimd_f32 vi3x89AB = psimd_load_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      const psimd_f32 vi4x89AB = psimd_load_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi0x4567, vw0123);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi1x4567, vw89AB);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi2x4567, vwCDEF);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi3x4567, vwGHIJ);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi4x4567, vwKLMN);

      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi0x3456, vw0123);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi1x3456, vw4567);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi2x3456, vwCDEF);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi3x3456, vwGHIJ);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi4x3456, vwKLMN);

      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x2345, vw0123);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x2345, vw4567);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x2345, vw89AB);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x2345, vwGHIJ);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi4x2345, vwKLMN);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;

      const psimd_f32 vi0x5678 = extq1_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x5678 = extq1_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x5678 = extq1_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x5678 = extq1_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x5678 = extq1_f32(vi4x4567, vi4x89AB);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi0x5678, vw4567);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi1x5678, vw89AB);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi2x5678, vwCDEF);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi3x5678, vwGHIJ);
      vo4567p00 = vfmaq_lane0_f32( vo4567p00, vi4x5678, vwOP);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vi4x89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x6789, vw4567);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x6789, vw89AB);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x6789, vwCDEF);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x6789, vwKLMN);
      vo4567p00 = vfmaq_lane1_f32( vo4567p00, vi4x6789, vwOP);


      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;

      psimd_f32 vo0 = vo4567p00;

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);

      psimd_store_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(k > 4)
    {
      psimd_f32 vo4567p00 = psimd_splat0_f32(vw0123);

      psimd_f32 vi0x89AB = psimd_load_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      psimd_f32 vi1x89AB = psimd_load_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      psimd_f32 vi2x89AB = psimd_load_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      psimd_f32 vi3x89AB = psimd_load_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      psimd_f32 vi4x89AB = psimd_load_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);

      vi0x89AB = psimd_andmask_f32(vmask, vi0x89AB);
      vi1x89AB = psimd_andmask_f32(vmask, vi1x89AB);
      vi2x89AB = psimd_andmask_f32(vmask, vi2x89AB);
      vi3x89AB = psimd_andmask_f32(vmask, vi3x89AB);
      vi4x89AB = psimd_andmask_f32(vmask, vi4x89AB);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi0x4567, vw0123);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi1x4567, vw89AB);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi2x4567, vwCDEF);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi3x4567, vwGHIJ);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi4x4567, vwKLMN);

      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi0x3456, vw0123);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi1x3456, vw4567);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi2x3456, vwCDEF);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi3x3456, vwGHIJ);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi4x3456, vwKLMN);

      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x2345, vw0123);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x2345, vw4567);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x2345, vw89AB);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x2345, vwGHIJ);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi4x2345, vwKLMN);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;

      const psimd_f32 vi0x5678 = extq1_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x5678 = extq1_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x5678 = extq1_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x5678 = extq1_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x5678 = extq1_f32(vi4x4567, vi4x89AB);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi0x5678, vw4567);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi1x5678, vw89AB);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi2x5678, vwCDEF);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi3x5678, vwGHIJ);
      vo4567p00 = vfmaq_lane0_f32( vo4567p00, vi4x5678, vwOP);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vi0x89AB);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vi1x89AB);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vi2x89AB);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vi3x89AB);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vi4x89AB);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x6789, vw4567);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x6789, vw89AB);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x6789, vwCDEF);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x6789, vwKLMN);
      vo4567p00 = vfmaq_lane1_f32( vo4567p00, vi4x6789, vwOP);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;

      psimd_f32 vo0 = vo4567p00;

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);

      psimd_store_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
      k -= 4;
    }
    assert(k >= 1);
    assert(k <= 4);
    {
      psimd_f32 vo4567p00 = psimd_splat0_f32(vw0123);

      // This might have already happened if there are more than 4 pixels, but
      // we can't count on it.
      vi0x4567 = psimd_andmask_f32(vmask, vi0x4567);
      vi1x4567 = psimd_andmask_f32(vmask, vi1x4567);
      vi2x4567 = psimd_andmask_f32(vmask, vi2x4567);
      vi3x4567 = psimd_andmask_f32(vmask, vi3x4567);
      vi4x4567 = psimd_andmask_f32(vmask, vi4x4567);

      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi0x4567, vw0123);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi1x4567, vw89AB);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi2x4567, vwCDEF);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi3x4567, vwGHIJ);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi4x4567, vwKLMN);


      const psimd_f32 vi0x3456 = extq3_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x3456 = extq3_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x3456 = extq3_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x3456 = extq3_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x3456 = extq3_f32(vi4x0123, vi4x4567);

      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi0x3456, vw0123);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi1x3456, vw4567);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi2x3456, vwCDEF);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi3x3456, vwGHIJ);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi4x3456, vwKLMN);

      const psimd_f32 vi0x2345 = extq2_f32(vi0x0123, vi0x4567);
      const psimd_f32 vi1x2345 = extq2_f32(vi1x0123, vi1x4567);
      const psimd_f32 vi2x2345 = extq2_f32(vi2x0123, vi2x4567);
      const psimd_f32 vi3x2345 = extq2_f32(vi3x0123, vi3x4567);
      const psimd_f32 vi4x2345 = extq2_f32(vi4x0123, vi4x4567);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x2345, vw0123);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x2345, vw4567);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x2345, vw89AB);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x2345, vwGHIJ);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi4x2345, vwKLMN);

      const psimd_f32 vzero = psimd_zero_f32();
      const psimd_f32 vi0x5678 = extq1_f32(vi0x4567, vzero);
      const psimd_f32 vi1x5678 = extq1_f32(vi1x4567, vzero);
      const psimd_f32 vi2x5678 = extq1_f32(vi2x4567, vzero);
      const psimd_f32 vi3x5678 = extq1_f32(vi3x4567, vzero);
      const psimd_f32 vi4x5678 = extq1_f32(vi4x4567, vzero);

      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi0x5678, vw4567);
      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi1x5678, vw89AB);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi2x5678, vwCDEF);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi3x5678, vwGHIJ);
      vo4567p00 = vfmaq_lane0_f32( vo4567p00, vi4x5678, vwOP);

      const psimd_f32 vi0x6789 = extq2_f32(vi0x4567, vzero);
      const psimd_f32 vi1x6789 = extq2_f32(vi1x4567, vzero);
      const psimd_f32 vi2x6789 = extq2_f32(vi2x4567, vzero);
      const psimd_f32 vi3x6789 = extq2_f32(vi3x4567, vzero);
      const psimd_f32 vi4x6789 = extq2_f32(vi4x4567, vzero);

      vo4567p00 = vfmaq_lane1_f32(vo4567p00, vi0x6789, vw4567);
      vo4567p00 = vfmaq_lane2_f32(vo4567p00, vi1x6789, vw89AB);
      vo4567p00 = vfmaq_lane3_f32(vo4567p00, vi2x6789, vwCDEF);
      vo4567p00 = vfmaq_lane0_f32(vo4567p00, vi3x6789, vwKLMN);
      vo4567p00 = vfmaq_lane1_f32( vo4567p00, vi4x6789, vwOP);

      psimd_f32 vo0 = vo4567p00;

      vo0 = psimd_max_f32(vo0, vmin);
      vo0 = psimd_min_f32(vo0, vmax);

      if XNN_LIKELY(k & 4) {
        psimd_store_f32(output0, vo0);
      } else {
        float* output0_lo = output0;
        if (k & 2) {
          psimd_store2_f32(output0_lo, vo0); output0_lo += 2;
          vo0 = psimd_splat2_f32(vo0);
        }
        if (k & 1) {
          psimd_store1_f32(output0_lo, vo0);
        }
      }
    }
  }
}
