// Copyright 2026 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/kai-dwconv.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/kai_common.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p_interface.h"

// KleidiAI's public headers hide SME helpers unless SVE2 is enabled and do not
// declare concrete kernel symbols. This forwarding wrapper must remain usable
// on SME2-only CPUs, so declare the linked symbols directly.
extern uint64_t kai_get_sme_vector_length_u8(void);
extern size_t
kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(void);
extern void kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
    const void* src, const void* rhs_packed, void* dst, size_t in_stride_row,
    size_t in_stride_col, size_t dst_stride_row, size_t dst_stride_col,
    size_t valid_input_rows, size_t valid_dst_rows, size_t pad_left,
    size_t pad_top, float pad_value, float clamp_min, float clamp_max);

void xnn_kai_f32_dwconv_minmax_ukernel_9pvc__neonsme2(
    const void* input, const void* packed_weights, void* output,
    size_t input_height_stride, size_t input_pixel_stride,
    size_t output_height_stride, size_t output_pixel_stride,
    size_t valid_input_rows, size_t valid_output_rows,
    size_t input_padding_left, size_t input_padding_top, float padding_value,
    float output_min, float output_max) {
  kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
      input, packed_weights, output, input_height_stride, input_pixel_stride,
      output_height_stride, output_pixel_stride, valid_input_rows,
      valid_output_rows, input_padding_left, input_padding_top, padding_value,
      output_min, output_max);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>

__arm_locally_streaming void xnn_kai_f32_dwconv_minmax_ukernel_9pvc__neonsme2(
    const void* input, const void* packed_weights, void* output,
    size_t input_height_stride, size_t input_pixel_stride,
    size_t output_height_stride, size_t output_pixel_stride,
    size_t valid_input_rows, size_t valid_output_rows,
    size_t input_padding_left, size_t input_padding_top, float padding_value,
    float output_min, float output_max) {
  const size_t vl = svcntw();
  const size_t num_channels = output_pixel_stride / sizeof(float);
  const size_t output_cols = output_height_stride / output_pixel_stride;
  if (valid_output_rows == 0 || output_cols == 0 || num_channels == 0) {
    return;
  }
  const size_t valid_input_cols = input_height_stride / input_pixel_stride;
  const size_t in_col_stride = input_pixel_stride / sizeof(float);
  const size_t out_col_stride = output_pixel_stride / sizeof(float);
  const float* w_ptr = (const float*)packed_weights;
  const svfloat32_t vmin = svdup_f32(output_min);
  const svfloat32_t vmax = svdup_f32(output_max);
  const svfloat32_t vpad = svdup_f32(padding_value);

  float pad_row[64];
  svst1_f32(svptrue_b32(), pad_row, vpad);

  for (size_t c = 0; c < num_channels; c += vl) {
    const svbool_t pg = svwhilelt_b32((uint64_t)c, (uint64_t)num_channels);
    const svfloat32_t vbias = svld1_f32(pg, w_ptr);
    const svfloat32_t vk0 = svld1_f32(pg, w_ptr + 1 * vl);
    const svfloat32_t vk1 = svld1_f32(pg, w_ptr + 2 * vl);
    const svfloat32_t vk2 = svld1_f32(pg, w_ptr + 3 * vl);
    const svfloat32_t vk3 = svld1_f32(pg, w_ptr + 4 * vl);
    const svfloat32_t vk4 = svld1_f32(pg, w_ptr + 5 * vl);
    const svfloat32_t vk5 = svld1_f32(pg, w_ptr + 6 * vl);
    const svfloat32_t vk6 = svld1_f32(pg, w_ptr + 7 * vl);
    const svfloat32_t vk7 = svld1_f32(pg, w_ptr + 8 * vl);
    const svfloat32_t vk8 = svld1_f32(pg, w_ptr + 9 * vl);
    w_ptr += 10 * vl;

    const float* in_rows[6];
    size_t row_col_strides[6];
    for (size_t vrow = 0; vrow < 6; ++vrow) {
      if (vrow >= input_padding_top &&
          (vrow - input_padding_top) < valid_input_rows) {
        in_rows[vrow] =
            (const float*)((const uint8_t*)input +
                           (vrow - input_padding_top) * input_height_stride) +
            c;
        row_col_strides[vrow] = in_col_stride;
      } else {
        in_rows[vrow] = pad_row;
        row_col_strides[vrow] = 0;
      }
    }

    float* out_r0 = (float*)output + c;
    float* out_r1 = (float*)((uint8_t*)output + 1 * output_height_stride) + c;
    float* out_r2 = (float*)((uint8_t*)output + 2 * output_height_stride) + c;
    float* out_r3 = (float*)((uint8_t*)output + 3 * output_height_stride) + c;

    svfloat32_t c0_r0 = vpad;
    svfloat32_t c0_r1 = vpad;
    svfloat32_t c0_r2 = vpad;
    svfloat32_t c0_r3 = vpad;
    svfloat32_t c0_r4 = vpad;
    svfloat32_t c0_r5 = vpad;
    if (input_padding_left == 0 && valid_input_cols > 0) {
      c0_r0 = svld1_f32(pg, in_rows[0]);
      c0_r1 = svld1_f32(pg, in_rows[1]);
      c0_r2 = svld1_f32(pg, in_rows[2]);
      c0_r3 = svld1_f32(pg, in_rows[3]);
      c0_r4 = svld1_f32(pg, in_rows[4]);
      c0_r5 = svld1_f32(pg, in_rows[5]);
    }

    svfloat32_t c1_r0 = vpad;
    svfloat32_t c1_r1 = vpad;
    svfloat32_t c1_r2 = vpad;
    svfloat32_t c1_r3 = vpad;
    svfloat32_t c1_r4 = vpad;
    svfloat32_t c1_r5 = vpad;
    if (1 >= input_padding_left &&
        (1 - input_padding_left) < valid_input_cols) {
      const size_t icol1 = 1 - input_padding_left;
      c1_r0 = svld1_f32(pg, in_rows[0] + icol1 * row_col_strides[0]);
      c1_r1 = svld1_f32(pg, in_rows[1] + icol1 * row_col_strides[1]);
      c1_r2 = svld1_f32(pg, in_rows[2] + icol1 * row_col_strides[2]);
      c1_r3 = svld1_f32(pg, in_rows[3] + icol1 * row_col_strides[3]);
      c1_r4 = svld1_f32(pg, in_rows[4] + icol1 * row_col_strides[4]);
      c1_r5 = svld1_f32(pg, in_rows[5] + icol1 * row_col_strides[5]);
    }

    size_t ox = 0;
    if (input_padding_left <= 2 && valid_output_rows >= 4) {
      const size_t max_unpadded_vcol2 = input_padding_left + valid_input_cols;
      const size_t ox_fast_end = (max_unpadded_vcol2 > 2)
                                     ? ((max_unpadded_vcol2 - 2 < output_cols)
                                            ? (max_unpadded_vcol2 - 2)
                                            : output_cols)
                                     : 0;
      const size_t icol2_start = 2 - input_padding_left;
      const size_t out_row_stride = output_height_stride / sizeof(float);

      if (input_padding_top == 0 && valid_input_rows >= 6) {
        const size_t in_row_stride = input_height_stride / sizeof(float);
        const float* col_ptr =
            (const float*)input + c + icol2_start * in_col_stride;
        float* out_ptr = out_r0;

        for (; ox + 2 <= ox_fast_end; ox += 2) {
          svfloat32_t acc0_0 = svmla_f32_x(pg, vbias, c0_r0, vk0);
          svfloat32_t acc0_1 = svmla_f32_x(pg, vbias, c0_r1, vk0);
          svfloat32_t acc0_2 = svmla_f32_x(pg, vbias, c0_r2, vk0);
          svfloat32_t acc0_3 = svmla_f32_x(pg, vbias, c0_r3, vk0);

          acc0_0 = svmla_f32_x(pg, acc0_0, c0_r1, vk3);
          acc0_1 = svmla_f32_x(pg, acc0_1, c0_r2, vk3);
          acc0_2 = svmla_f32_x(pg, acc0_2, c0_r3, vk3);
          acc0_3 = svmla_f32_x(pg, acc0_3, c0_r4, vk3);

          acc0_0 = svmla_f32_x(pg, acc0_0, c0_r2, vk6);
          acc0_1 = svmla_f32_x(pg, acc0_1, c0_r3, vk6);
          acc0_2 = svmla_f32_x(pg, acc0_2, c0_r4, vk6);
          acc0_3 = svmla_f32_x(pg, acc0_3, c0_r5, vk6);

          svfloat32_t acc1_0 = svmla_f32_x(pg, vbias, c1_r0, vk0);
          svfloat32_t acc1_1 = svmla_f32_x(pg, vbias, c1_r1, vk0);
          svfloat32_t acc1_2 = svmla_f32_x(pg, vbias, c1_r2, vk0);
          svfloat32_t acc1_3 = svmla_f32_x(pg, vbias, c1_r3, vk0);

          acc0_0 = svmla_f32_x(pg, acc0_0, c1_r0, vk1);
          acc0_1 = svmla_f32_x(pg, acc0_1, c1_r1, vk1);
          acc0_2 = svmla_f32_x(pg, acc0_2, c1_r2, vk1);
          acc0_3 = svmla_f32_x(pg, acc0_3, c1_r3, vk1);

          acc1_0 = svmla_f32_x(pg, acc1_0, c1_r1, vk3);
          acc1_1 = svmla_f32_x(pg, acc1_1, c1_r2, vk3);
          acc1_2 = svmla_f32_x(pg, acc1_2, c1_r3, vk3);
          acc1_3 = svmla_f32_x(pg, acc1_3, c1_r4, vk3);

          acc0_0 = svmla_f32_x(pg, acc0_0, c1_r1, vk4);
          acc0_1 = svmla_f32_x(pg, acc0_1, c1_r2, vk4);
          acc0_2 = svmla_f32_x(pg, acc0_2, c1_r3, vk4);
          acc0_3 = svmla_f32_x(pg, acc0_3, c1_r4, vk4);

          acc1_0 = svmla_f32_x(pg, acc1_0, c1_r2, vk6);
          acc1_1 = svmla_f32_x(pg, acc1_1, c1_r3, vk6);
          acc1_2 = svmla_f32_x(pg, acc1_2, c1_r4, vk6);
          acc1_3 = svmla_f32_x(pg, acc1_3, c1_r5, vk6);

          acc0_0 = svmla_f32_x(pg, acc0_0, c1_r2, vk7);
          acc0_1 = svmla_f32_x(pg, acc0_1, c1_r3, vk7);
          acc0_2 = svmla_f32_x(pg, acc0_2, c1_r4, vk7);
          acc0_3 = svmla_f32_x(pg, acc0_3, c1_r5, vk7);

          const svfloat32_t c2_r0 = svld1_f32(pg, col_ptr);
          const svfloat32_t c2_r1 = svld1_f32(pg, col_ptr + in_row_stride);
          const svfloat32_t c2_r2 = svld1_f32(pg, col_ptr + 2 * in_row_stride);
          const svfloat32_t c2_r3 = svld1_f32(pg, col_ptr + 3 * in_row_stride);
          const svfloat32_t c2_r4 = svld1_f32(pg, col_ptr + 4 * in_row_stride);
          const svfloat32_t c2_r5 = svld1_f32(pg, col_ptr + 5 * in_row_stride);
          col_ptr += in_col_stride;

          acc0_0 = svmla_f32_x(pg, acc0_0, c2_r0, vk2);
          acc0_1 = svmla_f32_x(pg, acc0_1, c2_r1, vk2);
          acc0_2 = svmla_f32_x(pg, acc0_2, c2_r2, vk2);
          acc0_3 = svmla_f32_x(pg, acc0_3, c2_r3, vk2);

          acc1_0 = svmla_f32_x(pg, acc1_0, c2_r0, vk1);
          acc1_1 = svmla_f32_x(pg, acc1_1, c2_r1, vk1);
          acc1_2 = svmla_f32_x(pg, acc1_2, c2_r2, vk1);
          acc1_3 = svmla_f32_x(pg, acc1_3, c2_r3, vk1);

          acc0_0 = svmla_f32_x(pg, acc0_0, c2_r1, vk5);
          acc0_1 = svmla_f32_x(pg, acc0_1, c2_r2, vk5);
          acc0_2 = svmla_f32_x(pg, acc0_2, c2_r3, vk5);
          acc0_3 = svmla_f32_x(pg, acc0_3, c2_r4, vk5);

          acc1_0 = svmla_f32_x(pg, acc1_0, c2_r1, vk4);
          acc1_1 = svmla_f32_x(pg, acc1_1, c2_r2, vk4);
          acc1_2 = svmla_f32_x(pg, acc1_2, c2_r3, vk4);
          acc1_3 = svmla_f32_x(pg, acc1_3, c2_r4, vk4);

          acc0_0 = svmla_f32_x(pg, acc0_0, c2_r2, vk8);
          acc0_1 = svmla_f32_x(pg, acc0_1, c2_r3, vk8);
          acc0_2 = svmla_f32_x(pg, acc0_2, c2_r4, vk8);
          acc0_3 = svmla_f32_x(pg, acc0_3, c2_r5, vk8);

          acc1_0 = svmla_f32_x(pg, acc1_0, c2_r2, vk7);
          acc1_1 = svmla_f32_x(pg, acc1_1, c2_r3, vk7);
          acc1_2 = svmla_f32_x(pg, acc1_2, c2_r4, vk7);
          acc1_3 = svmla_f32_x(pg, acc1_3, c2_r5, vk7);

          acc0_0 = svmax_f32_x(pg, svmin_f32_x(pg, acc0_0, vmax), vmin);
          acc0_1 = svmax_f32_x(pg, svmin_f32_x(pg, acc0_1, vmax), vmin);
          acc0_2 = svmax_f32_x(pg, svmin_f32_x(pg, acc0_2, vmax), vmin);
          acc0_3 = svmax_f32_x(pg, svmin_f32_x(pg, acc0_3, vmax), vmin);

          svst1_f32(pg, out_ptr, acc0_0);
          svst1_f32(pg, out_ptr + out_row_stride, acc0_1);
          svst1_f32(pg, out_ptr + 2 * out_row_stride, acc0_2);
          svst1_f32(pg, out_ptr + 3 * out_row_stride, acc0_3);
          out_ptr += out_col_stride;

          const svfloat32_t c3_r0 = svld1_f32(pg, col_ptr);
          const svfloat32_t c3_r1 = svld1_f32(pg, col_ptr + in_row_stride);
          const svfloat32_t c3_r2 = svld1_f32(pg, col_ptr + 2 * in_row_stride);
          const svfloat32_t c3_r3 = svld1_f32(pg, col_ptr + 3 * in_row_stride);
          const svfloat32_t c3_r4 = svld1_f32(pg, col_ptr + 4 * in_row_stride);
          const svfloat32_t c3_r5 = svld1_f32(pg, col_ptr + 5 * in_row_stride);
          col_ptr += in_col_stride;

          acc1_0 = svmla_f32_x(pg, acc1_0, c3_r0, vk2);
          acc1_1 = svmla_f32_x(pg, acc1_1, c3_r1, vk2);
          acc1_2 = svmla_f32_x(pg, acc1_2, c3_r2, vk2);
          acc1_3 = svmla_f32_x(pg, acc1_3, c3_r3, vk2);

          acc1_0 = svmla_f32_x(pg, acc1_0, c3_r1, vk5);
          acc1_1 = svmla_f32_x(pg, acc1_1, c3_r2, vk5);
          acc1_2 = svmla_f32_x(pg, acc1_2, c3_r3, vk5);
          acc1_3 = svmla_f32_x(pg, acc1_3, c3_r4, vk5);

          acc1_0 = svmla_f32_x(pg, acc1_0, c3_r2, vk8);
          acc1_1 = svmla_f32_x(pg, acc1_1, c3_r3, vk8);
          acc1_2 = svmla_f32_x(pg, acc1_2, c3_r4, vk8);
          acc1_3 = svmla_f32_x(pg, acc1_3, c3_r5, vk8);

          acc1_0 = svmax_f32_x(pg, svmin_f32_x(pg, acc1_0, vmax), vmin);
          acc1_1 = svmax_f32_x(pg, svmin_f32_x(pg, acc1_1, vmax), vmin);
          acc1_2 = svmax_f32_x(pg, svmin_f32_x(pg, acc1_2, vmax), vmin);
          acc1_3 = svmax_f32_x(pg, svmin_f32_x(pg, acc1_3, vmax), vmin);

          svst1_f32(pg, out_ptr, acc1_0);
          svst1_f32(pg, out_ptr + out_row_stride, acc1_1);
          svst1_f32(pg, out_ptr + 2 * out_row_stride, acc1_2);
          svst1_f32(pg, out_ptr + 3 * out_row_stride, acc1_3);
          out_ptr += out_col_stride;

          c0_r0 = c2_r0;
          c0_r1 = c2_r1;
          c0_r2 = c2_r2;
          c0_r3 = c2_r3;
          c0_r4 = c2_r4;
          c0_r5 = c2_r5;

          c1_r0 = c3_r0;
          c1_r1 = c3_r1;
          c1_r2 = c3_r2;
          c1_r3 = c3_r3;
          c1_r4 = c3_r4;
          c1_r5 = c3_r5;
        }

        for (; ox < ox_fast_end; ++ox) {
          svfloat32_t acc0 = svmla_f32_x(pg, vbias, c0_r0, vk0);
          svfloat32_t acc1 = svmla_f32_x(pg, vbias, c0_r1, vk0);
          svfloat32_t acc2 = svmla_f32_x(pg, vbias, c0_r2, vk0);
          svfloat32_t acc3 = svmla_f32_x(pg, vbias, c0_r3, vk0);

          acc0 = svmla_f32_x(pg, acc0, c0_r1, vk3);
          acc1 = svmla_f32_x(pg, acc1, c0_r2, vk3);
          acc2 = svmla_f32_x(pg, acc2, c0_r3, vk3);
          acc3 = svmla_f32_x(pg, acc3, c0_r4, vk3);

          acc0 = svmla_f32_x(pg, acc0, c0_r2, vk6);
          acc1 = svmla_f32_x(pg, acc1, c0_r3, vk6);
          acc2 = svmla_f32_x(pg, acc2, c0_r4, vk6);
          acc3 = svmla_f32_x(pg, acc3, c0_r5, vk6);

          acc0 = svmla_f32_x(pg, acc0, c1_r0, vk1);
          acc1 = svmla_f32_x(pg, acc1, c1_r1, vk1);
          acc2 = svmla_f32_x(pg, acc2, c1_r2, vk1);
          acc3 = svmla_f32_x(pg, acc3, c1_r3, vk1);

          acc0 = svmla_f32_x(pg, acc0, c1_r1, vk4);
          acc1 = svmla_f32_x(pg, acc1, c1_r2, vk4);
          acc2 = svmla_f32_x(pg, acc2, c1_r3, vk4);
          acc3 = svmla_f32_x(pg, acc3, c1_r4, vk4);

          acc0 = svmla_f32_x(pg, acc0, c1_r2, vk7);
          acc1 = svmla_f32_x(pg, acc1, c1_r3, vk7);
          acc2 = svmla_f32_x(pg, acc2, c1_r4, vk7);
          acc3 = svmla_f32_x(pg, acc3, c1_r5, vk7);

          const svfloat32_t c2_r0 = svld1_f32(pg, col_ptr);
          const svfloat32_t c2_r1 = svld1_f32(pg, col_ptr + in_row_stride);
          const svfloat32_t c2_r2 = svld1_f32(pg, col_ptr + 2 * in_row_stride);
          const svfloat32_t c2_r3 = svld1_f32(pg, col_ptr + 3 * in_row_stride);
          const svfloat32_t c2_r4 = svld1_f32(pg, col_ptr + 4 * in_row_stride);
          const svfloat32_t c2_r5 = svld1_f32(pg, col_ptr + 5 * in_row_stride);
          col_ptr += in_col_stride;

          acc0 = svmla_f32_x(pg, acc0, c2_r0, vk2);
          acc1 = svmla_f32_x(pg, acc1, c2_r1, vk2);
          acc2 = svmla_f32_x(pg, acc2, c2_r2, vk2);
          acc3 = svmla_f32_x(pg, acc3, c2_r3, vk2);

          acc0 = svmla_f32_x(pg, acc0, c2_r1, vk5);
          acc1 = svmla_f32_x(pg, acc1, c2_r2, vk5);
          acc2 = svmla_f32_x(pg, acc2, c2_r3, vk5);
          acc3 = svmla_f32_x(pg, acc3, c2_r4, vk5);

          acc0 = svmla_f32_x(pg, acc0, c2_r2, vk8);
          acc1 = svmla_f32_x(pg, acc1, c2_r3, vk8);
          acc2 = svmla_f32_x(pg, acc2, c2_r4, vk8);
          acc3 = svmla_f32_x(pg, acc3, c2_r5, vk8);

          acc0 = svmax_f32_x(pg, svmin_f32_x(pg, acc0, vmax), vmin);
          acc1 = svmax_f32_x(pg, svmin_f32_x(pg, acc1, vmax), vmin);
          acc2 = svmax_f32_x(pg, svmin_f32_x(pg, acc2, vmax), vmin);
          acc3 = svmax_f32_x(pg, svmin_f32_x(pg, acc3, vmax), vmin);

          svst1_f32(pg, out_ptr, acc0);
          svst1_f32(pg, out_ptr + out_row_stride, acc1);
          svst1_f32(pg, out_ptr + 2 * out_row_stride, acc2);
          svst1_f32(pg, out_ptr + 3 * out_row_stride, acc3);
          out_ptr += out_col_stride;

          c0_r0 = c1_r0;
          c0_r1 = c1_r1;
          c0_r2 = c1_r2;
          c0_r3 = c1_r3;
          c0_r4 = c1_r4;
          c0_r5 = c1_r5;

          c1_r0 = c2_r0;
          c1_r1 = c2_r1;
          c1_r2 = c2_r2;
          c1_r3 = c2_r3;
          c1_r4 = c2_r4;
          c1_r5 = c2_r5;
        }

        out_r0 += ox * out_col_stride;
        out_r1 += ox * out_col_stride;
        out_r2 += ox * out_col_stride;
        out_r3 += ox * out_col_stride;
      } else {
        const float* col2_ptr0 = in_rows[0] + icol2_start * row_col_strides[0];
        const float* col2_ptr1 = in_rows[1] + icol2_start * row_col_strides[1];
        const float* col2_ptr2 = in_rows[2] + icol2_start * row_col_strides[2];
        const float* col2_ptr3 = in_rows[3] + icol2_start * row_col_strides[3];
        const float* col2_ptr4 = in_rows[4] + icol2_start * row_col_strides[4];
        const float* col2_ptr5 = in_rows[5] + icol2_start * row_col_strides[5];
        const size_t s0 = row_col_strides[0];
        const size_t s1 = row_col_strides[1];
        const size_t s2 = row_col_strides[2];
        const size_t s3 = row_col_strides[3];
        const size_t s4 = row_col_strides[4];
        const size_t s5 = row_col_strides[5];

        for (; ox < ox_fast_end; ++ox) {
          svfloat32_t acc0 = svmla_f32_x(pg, vbias, c0_r0, vk0);
          svfloat32_t acc1 = svmla_f32_x(pg, vbias, c0_r1, vk0);
          svfloat32_t acc2 = svmla_f32_x(pg, vbias, c0_r2, vk0);
          svfloat32_t acc3 = svmla_f32_x(pg, vbias, c0_r3, vk0);

          acc0 = svmla_f32_x(pg, acc0, c0_r1, vk3);
          acc1 = svmla_f32_x(pg, acc1, c0_r2, vk3);
          acc2 = svmla_f32_x(pg, acc2, c0_r3, vk3);
          acc3 = svmla_f32_x(pg, acc3, c0_r4, vk3);

          acc0 = svmla_f32_x(pg, acc0, c0_r2, vk6);
          acc1 = svmla_f32_x(pg, acc1, c0_r3, vk6);
          acc2 = svmla_f32_x(pg, acc2, c0_r4, vk6);
          acc3 = svmla_f32_x(pg, acc3, c0_r5, vk6);

          acc0 = svmla_f32_x(pg, acc0, c1_r0, vk1);
          acc1 = svmla_f32_x(pg, acc1, c1_r1, vk1);
          acc2 = svmla_f32_x(pg, acc2, c1_r2, vk1);
          acc3 = svmla_f32_x(pg, acc3, c1_r3, vk1);

          acc0 = svmla_f32_x(pg, acc0, c1_r1, vk4);
          acc1 = svmla_f32_x(pg, acc1, c1_r2, vk4);
          acc2 = svmla_f32_x(pg, acc2, c1_r3, vk4);
          acc3 = svmla_f32_x(pg, acc3, c1_r4, vk4);

          acc0 = svmla_f32_x(pg, acc0, c1_r2, vk7);
          acc1 = svmla_f32_x(pg, acc1, c1_r3, vk7);
          acc2 = svmla_f32_x(pg, acc2, c1_r4, vk7);
          acc3 = svmla_f32_x(pg, acc3, c1_r5, vk7);

          const svfloat32_t c2_r0 = svld1_f32(pg, col2_ptr0);
          const svfloat32_t c2_r1 = svld1_f32(pg, col2_ptr1);
          const svfloat32_t c2_r2 = svld1_f32(pg, col2_ptr2);
          const svfloat32_t c2_r3 = svld1_f32(pg, col2_ptr3);
          const svfloat32_t c2_r4 = svld1_f32(pg, col2_ptr4);
          const svfloat32_t c2_r5 = svld1_f32(pg, col2_ptr5);
          col2_ptr0 += s0;
          col2_ptr1 += s1;
          col2_ptr2 += s2;
          col2_ptr3 += s3;
          col2_ptr4 += s4;
          col2_ptr5 += s5;

          acc0 = svmla_f32_x(pg, acc0, c2_r0, vk2);
          acc1 = svmla_f32_x(pg, acc1, c2_r1, vk2);
          acc2 = svmla_f32_x(pg, acc2, c2_r2, vk2);
          acc3 = svmla_f32_x(pg, acc3, c2_r3, vk2);

          acc0 = svmla_f32_x(pg, acc0, c2_r1, vk5);
          acc1 = svmla_f32_x(pg, acc1, c2_r2, vk5);
          acc2 = svmla_f32_x(pg, acc2, c2_r3, vk5);
          acc3 = svmla_f32_x(pg, acc3, c2_r4, vk5);

          acc0 = svmla_f32_x(pg, acc0, c2_r2, vk8);
          acc1 = svmla_f32_x(pg, acc1, c2_r3, vk8);
          acc2 = svmla_f32_x(pg, acc2, c2_r4, vk8);
          acc3 = svmla_f32_x(pg, acc3, c2_r5, vk8);

          acc0 = svmax_f32_x(pg, svmin_f32_x(pg, acc0, vmax), vmin);
          acc1 = svmax_f32_x(pg, svmin_f32_x(pg, acc1, vmax), vmin);
          acc2 = svmax_f32_x(pg, svmin_f32_x(pg, acc2, vmax), vmin);
          acc3 = svmax_f32_x(pg, svmin_f32_x(pg, acc3, vmax), vmin);

          svst1_f32(pg, out_r0, acc0);
          svst1_f32(pg, out_r1, acc1);
          svst1_f32(pg, out_r2, acc2);
          svst1_f32(pg, out_r3, acc3);
          out_r0 += out_col_stride;
          out_r1 += out_col_stride;
          out_r2 += out_col_stride;
          out_r3 += out_col_stride;

          c0_r0 = c1_r0;
          c0_r1 = c1_r1;
          c0_r2 = c1_r2;
          c0_r3 = c1_r3;
          c0_r4 = c1_r4;
          c0_r5 = c1_r5;

          c1_r0 = c2_r0;
          c1_r1 = c2_r1;
          c1_r2 = c2_r2;
          c1_r3 = c2_r3;
          c1_r4 = c2_r4;
          c1_r5 = c2_r5;
        }
      }
    }

    for (; ox < output_cols; ++ox) {
      svfloat32_t acc0 = svmla_f32_x(pg, vbias, c0_r0, vk0);
      svfloat32_t acc1 = svmla_f32_x(pg, vbias, c0_r1, vk0);
      svfloat32_t acc2 = svmla_f32_x(pg, vbias, c0_r2, vk0);
      svfloat32_t acc3 = svmla_f32_x(pg, vbias, c0_r3, vk0);

      acc0 = svmla_f32_x(pg, acc0, c0_r1, vk3);
      acc1 = svmla_f32_x(pg, acc1, c0_r2, vk3);
      acc2 = svmla_f32_x(pg, acc2, c0_r3, vk3);
      acc3 = svmla_f32_x(pg, acc3, c0_r4, vk3);

      acc0 = svmla_f32_x(pg, acc0, c0_r2, vk6);
      acc1 = svmla_f32_x(pg, acc1, c0_r3, vk6);
      acc2 = svmla_f32_x(pg, acc2, c0_r4, vk6);
      acc3 = svmla_f32_x(pg, acc3, c0_r5, vk6);

      const size_t vcol2 = ox + 2;
      svfloat32_t c2_r0 = vpad;
      svfloat32_t c2_r1 = vpad;
      svfloat32_t c2_r2 = vpad;
      svfloat32_t c2_r3 = vpad;
      svfloat32_t c2_r4 = vpad;
      svfloat32_t c2_r5 = vpad;
      if (vcol2 >= input_padding_left &&
          (vcol2 - input_padding_left) < valid_input_cols) {
        const size_t icol2 = vcol2 - input_padding_left;
        c2_r0 = svld1_f32(pg, in_rows[0] + icol2 * row_col_strides[0]);
        c2_r1 = svld1_f32(pg, in_rows[1] + icol2 * row_col_strides[1]);
        c2_r2 = svld1_f32(pg, in_rows[2] + icol2 * row_col_strides[2]);
        c2_r3 = svld1_f32(pg, in_rows[3] + icol2 * row_col_strides[3]);
        c2_r4 = svld1_f32(pg, in_rows[4] + icol2 * row_col_strides[4]);
        c2_r5 = svld1_f32(pg, in_rows[5] + icol2 * row_col_strides[5]);
      }

      acc0 = svmla_f32_x(pg, acc0, c1_r0, vk1);
      acc1 = svmla_f32_x(pg, acc1, c1_r1, vk1);
      acc2 = svmla_f32_x(pg, acc2, c1_r2, vk1);
      acc3 = svmla_f32_x(pg, acc3, c1_r3, vk1);

      acc0 = svmla_f32_x(pg, acc0, c1_r1, vk4);
      acc1 = svmla_f32_x(pg, acc1, c1_r2, vk4);
      acc2 = svmla_f32_x(pg, acc2, c1_r3, vk4);
      acc3 = svmla_f32_x(pg, acc3, c1_r4, vk4);

      acc0 = svmla_f32_x(pg, acc0, c1_r2, vk7);
      acc1 = svmla_f32_x(pg, acc1, c1_r3, vk7);
      acc2 = svmla_f32_x(pg, acc2, c1_r4, vk7);
      acc3 = svmla_f32_x(pg, acc3, c1_r5, vk7);

      acc0 = svmla_f32_x(pg, acc0, c2_r0, vk2);
      acc1 = svmla_f32_x(pg, acc1, c2_r1, vk2);
      acc2 = svmla_f32_x(pg, acc2, c2_r2, vk2);
      acc3 = svmla_f32_x(pg, acc3, c2_r3, vk2);

      acc0 = svmla_f32_x(pg, acc0, c2_r1, vk5);
      acc1 = svmla_f32_x(pg, acc1, c2_r2, vk5);
      acc2 = svmla_f32_x(pg, acc2, c2_r3, vk5);
      acc3 = svmla_f32_x(pg, acc3, c2_r4, vk5);

      acc0 = svmla_f32_x(pg, acc0, c2_r2, vk8);
      acc1 = svmla_f32_x(pg, acc1, c2_r3, vk8);
      acc2 = svmla_f32_x(pg, acc2, c2_r4, vk8);
      acc3 = svmla_f32_x(pg, acc3, c2_r5, vk8);

      acc0 = svmax_f32_x(pg, svmin_f32_x(pg, acc0, vmax), vmin);
      svst1_f32(pg, out_r0, acc0);
      out_r0 += out_col_stride;
      if (valid_output_rows > 1) {
        acc1 = svmax_f32_x(pg, svmin_f32_x(pg, acc1, vmax), vmin);
        svst1_f32(pg, out_r1, acc1);
        out_r1 += out_col_stride;
      }
      if (valid_output_rows > 2) {
        acc2 = svmax_f32_x(pg, svmin_f32_x(pg, acc2, vmax), vmin);
        svst1_f32(pg, out_r2, acc2);
        out_r2 += out_col_stride;
      }
      if (valid_output_rows > 3) {
        acc3 = svmax_f32_x(pg, svmin_f32_x(pg, acc3, vmax), vmin);
        svst1_f32(pg, out_r3, acc3);
        out_r3 += out_col_stride;
      }

      c0_r0 = c1_r0;
      c0_r1 = c1_r1;
      c0_r2 = c1_r2;
      c0_r3 = c1_r3;
      c0_r4 = c1_r4;
      c0_r5 = c1_r5;

      c1_r0 = c2_r0;
      c1_r1 = c2_r1;
      c1_r2 = c2_r2;
      c1_r3 = c2_r3;
      c1_r4 = c2_r4;
      c1_r5 = c2_r5;
    }
  }
}
#endif  // XNN_ENABLE_KLEIDIAI

size_t xnn_f32_dwconv_minmax_ukernel_9pvc__neonsme2_get_channel_tile(void) {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_sme_vector_length_u8() / sizeof(float);
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
  return svcntsw();
#else
  return 1;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_f32_dwconv_minmax_ukernel_9pvc__neonsme2_get_output_height_tile(
    void) {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
  return 4;
#else
  return 1;
#endif  // XNN_ENABLE_KLEIDIAI
}
