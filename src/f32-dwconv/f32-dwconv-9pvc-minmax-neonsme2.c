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

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_f32_dwconv_9pvc_sme2_impl(
    const void* input, const void* packed_weights, void* output,
    size_t input_height_stride, size_t input_pixel_stride,
    size_t output_height_stride, size_t output_pixel_stride,
    size_t valid_input_rows, size_t valid_output_rows, size_t input_padding_left,
    size_t input_padding_top, float padding_value, float output_min,
    float output_max) {
  const size_t vl = svcntw();
  const size_t num_channels = output_pixel_stride / sizeof(float);
  const size_t output_cols = output_height_stride / output_pixel_stride;
  const size_t valid_input_cols = input_height_stride / input_pixel_stride;
  const size_t in_col_stride = input_pixel_stride / sizeof(float);
  const size_t out_col_stride = output_pixel_stride / sizeof(float);
  const float* w_ptr = (const float*)packed_weights;
  const svfloat32_t vmin = svdup_f32(output_min);
  const svfloat32_t vmax = svdup_f32(output_max);
  const svfloat32_t vpad = svdup_f32(padding_value);

  float pad_row[64];
  float sink_row[64];
  svst1_f32(svptrue_b32(), pad_row, vpad);

  const size_t out_st1 = valid_output_rows > 1 ? out_col_stride : 0;
  const size_t out_st2 = valid_output_rows > 2 ? out_col_stride : 0;
  const size_t out_st3 = valid_output_rows > 3 ? out_col_stride : 0;

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
    const svfloat32x4_t vbiasx4 = svcreate4_f32(vbias, vbias, vbias, vbias);

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
    float* out_r1 =
        valid_output_rows > 1
            ? (float*)((uint8_t*)output + 1 * output_height_stride) + c
            : sink_row;
    float* out_r2 =
        valid_output_rows > 2
            ? (float*)((uint8_t*)output + 2 * output_height_stride) + c
            : sink_row;
    float* out_r3 =
        valid_output_rows > 3
            ? (float*)((uint8_t*)output + 3 * output_height_stride) + c
            : sink_row;

    uint32_t slice = 0;
    svwrite_za32_f32_vg1x4(slice + 0, vbiasx4);
    svwrite_za32_f32_vg1x4(slice + 1, vbiasx4);
    svwrite_za32_f32_vg1x4(slice + 2, vbiasx4);

    {
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
      svmla_single_za32_f32_vg1x4(
          slice + 0, svcreate4_f32(c0_r0, c0_r1, c0_r2, c0_r3), vk0);
      svmla_single_za32_f32_vg1x4(
          slice + 0, svcreate4_f32(c0_r1, c0_r2, c0_r3, c0_r4), vk3);
      svmla_single_za32_f32_vg1x4(
          slice + 0, svcreate4_f32(c0_r2, c0_r3, c0_r4, c0_r5), vk6);
    }

    {
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
      const svfloat32x4_t g0 = svcreate4_f32(c1_r0, c1_r1, c1_r2, c1_r3);
      svmla_single_za32_f32_vg1x4(slice + 0, g0, vk1);
      svmla_single_za32_f32_vg1x4(slice + 1, g0, vk0);
      const svfloat32x4_t g1 = svcreate4_f32(c1_r1, c1_r2, c1_r3, c1_r4);
      svmla_single_za32_f32_vg1x4(slice + 0, g1, vk4);
      svmla_single_za32_f32_vg1x4(slice + 1, g1, vk3);
      const svfloat32x4_t g2 = svcreate4_f32(c1_r2, c1_r3, c1_r4, c1_r5);
      svmla_single_za32_f32_vg1x4(slice + 0, g2, vk7);
      svmla_single_za32_f32_vg1x4(slice + 1, g2, vk6);
    }

    size_t ox = 0;

    if (input_padding_left <= 2) {
      const size_t max_vcol = input_padding_left + valid_input_cols;
      const size_t avail_vcols = max_vcol > 2 ? (max_vcol - 2) : 0;
      const size_t fast_cols =
          output_cols < avail_vcols ? output_cols : avail_vcols;

      if (fast_cols > 0) {
        const size_t icol2_start = 2 - input_padding_left;

        if (input_padding_top == 0 && valid_input_rows >= 6) {
          const size_t in_row_stride = input_height_stride / sizeof(float);
          const float* col_ptr =
              (const float*)input + c + icol2_start * in_col_stride;

          svfloat32_t r0 = svld1_f32(pg, col_ptr);
          svfloat32_t r1 = svld1_f32(pg, col_ptr + in_row_stride);
          svfloat32_t r2 = svld1_f32(pg, col_ptr + 2 * in_row_stride);
          svfloat32_t r3 = svld1_f32(pg, col_ptr + 3 * in_row_stride);
          svfloat32_t r4 = svld1_f32(pg, col_ptr + 4 * in_row_stride);
          svfloat32_t r5 = svld1_f32(pg, col_ptr + 5 * in_row_stride);
          col_ptr += in_col_stride;

          for (; ox + 1 < fast_cols; ++ox) {
            const svfloat32x4_t g0 = svcreate4_f32(r0, r1, r2, r3);
            svmla_single_za32_f32_vg1x4(slice + 0, g0, vk2);
            svmla_single_za32_f32_vg1x4(slice + 1, g0, vk1);
            svmla_single_za32_f32_vg1x4(slice + 2, g0, vk0);
            r0 = svld1_f32(pg, col_ptr);

            const svfloat32x4_t g1 = svcreate4_f32(r1, r2, r3, r4);
            svmla_single_za32_f32_vg1x4(slice + 0, g1, vk5);
            svmla_single_za32_f32_vg1x4(slice + 1, g1, vk4);
            svmla_single_za32_f32_vg1x4(slice + 2, g1, vk3);
            r1 = svld1_f32(pg, col_ptr + in_row_stride);

            const svfloat32x4_t g2 = svcreate4_f32(r2, r3, r4, r5);
            svmla_single_za32_f32_vg1x4(slice + 0, g2, vk8);
            svmla_single_za32_f32_vg1x4(slice + 1, g2, vk7);
            svmla_single_za32_f32_vg1x4(slice + 2, g2, vk6);
            r2 = svld1_f32(pg, col_ptr + 2 * in_row_stride);
            r3 = svld1_f32(pg, col_ptr + 3 * in_row_stride);

            const svfloat32x4_t vacc = svread_za32_f32_vg1x4(slice + 0);
            ++slice;
            r4 = svld1_f32(pg, col_ptr + 4 * in_row_stride);
            svwrite_za32_f32_vg1x4(slice + 2, vbiasx4);
            r5 = svld1_f32(pg, col_ptr + 5 * in_row_stride);
            col_ptr += in_col_stride;

            const svfloat32x4_t clamped =
                svclamp_single_f32_x4(vacc, vmin, vmax);
            svst1_f32(pg, out_r0, svget4_f32(clamped, 0));
            out_r0 += out_col_stride;
            svst1_f32(pg, out_r1, svget4_f32(clamped, 1));
            out_r1 += out_st1;
            svst1_f32(pg, out_r2, svget4_f32(clamped, 2));
            out_r2 += out_st2;
            svst1_f32(pg, out_r3, svget4_f32(clamped, 3));
            out_r3 += out_st3;
          }

          {
            const svfloat32x4_t g0 = svcreate4_f32(r0, r1, r2, r3);
            svmla_single_za32_f32_vg1x4(slice + 0, g0, vk2);
            svmla_single_za32_f32_vg1x4(slice + 1, g0, vk1);
            svmla_single_za32_f32_vg1x4(slice + 2, g0, vk0);
            const svfloat32x4_t g1 = svcreate4_f32(r1, r2, r3, r4);
            svmla_single_za32_f32_vg1x4(slice + 0, g1, vk5);
            svmla_single_za32_f32_vg1x4(slice + 1, g1, vk4);
            svmla_single_za32_f32_vg1x4(slice + 2, g1, vk3);
            const svfloat32x4_t g2 = svcreate4_f32(r2, r3, r4, r5);
            svmla_single_za32_f32_vg1x4(slice + 0, g2, vk8);
            svmla_single_za32_f32_vg1x4(slice + 1, g2, vk7);
            svmla_single_za32_f32_vg1x4(slice + 2, g2, vk6);

            const svfloat32x4_t vacc = svread_za32_f32_vg1x4(slice + 0);
            ++slice;
            svwrite_za32_f32_vg1x4(slice + 2, vbiasx4);
            const svfloat32x4_t clamped =
                svclamp_single_f32_x4(vacc, vmin, vmax);
            svst1_f32(pg, out_r0, svget4_f32(clamped, 0));
            out_r0 += out_col_stride;
            svst1_f32(pg, out_r1, svget4_f32(clamped, 1));
            out_r1 += out_st1;
            svst1_f32(pg, out_r2, svget4_f32(clamped, 2));
            out_r2 += out_st2;
            svst1_f32(pg, out_r3, svget4_f32(clamped, 3));
            out_r3 += out_st3;
            ++ox;
          }
        } else {
          const float* col2_ptr0 =
              in_rows[0] + icol2_start * row_col_strides[0];
          const float* col2_ptr1 =
              in_rows[1] + icol2_start * row_col_strides[1];
          const float* col2_ptr2 =
              in_rows[2] + icol2_start * row_col_strides[2];
          const float* col2_ptr3 =
              in_rows[3] + icol2_start * row_col_strides[3];
          const float* col2_ptr4 =
              in_rows[4] + icol2_start * row_col_strides[4];
          const float* col2_ptr5 =
              in_rows[5] + icol2_start * row_col_strides[5];
          const size_t st0 = row_col_strides[0];
          const size_t st1 = row_col_strides[1];
          const size_t st2 = row_col_strides[2];
          const size_t st3 = row_col_strides[3];
          const size_t st4 = row_col_strides[4];
          const size_t st5 = row_col_strides[5];

          svfloat32_t r0 = svld1_f32(pg, col2_ptr0);
          svfloat32_t r1 = svld1_f32(pg, col2_ptr1);
          svfloat32_t r2 = svld1_f32(pg, col2_ptr2);
          svfloat32_t r3 = svld1_f32(pg, col2_ptr3);
          svfloat32_t r4 = svld1_f32(pg, col2_ptr4);
          svfloat32_t r5 = svld1_f32(pg, col2_ptr5);
          col2_ptr0 += st0;
          col2_ptr1 += st1;
          col2_ptr2 += st2;
          col2_ptr3 += st3;
          col2_ptr4 += st4;
          col2_ptr5 += st5;

          for (; ox + 1 < fast_cols; ++ox) {
            const svfloat32x4_t g0 = svcreate4_f32(r0, r1, r2, r3);
            svmla_single_za32_f32_vg1x4(slice + 0, g0, vk2);
            svmla_single_za32_f32_vg1x4(slice + 1, g0, vk1);
            svmla_single_za32_f32_vg1x4(slice + 2, g0, vk0);
            r0 = svld1_f32(pg, col2_ptr0);
            col2_ptr0 += st0;

            const svfloat32x4_t g1 = svcreate4_f32(r1, r2, r3, r4);
            svmla_single_za32_f32_vg1x4(slice + 0, g1, vk5);
            svmla_single_za32_f32_vg1x4(slice + 1, g1, vk4);
            svmla_single_za32_f32_vg1x4(slice + 2, g1, vk3);
            r1 = svld1_f32(pg, col2_ptr1);
            col2_ptr1 += st1;

            const svfloat32x4_t g2 = svcreate4_f32(r2, r3, r4, r5);
            svmla_single_za32_f32_vg1x4(slice + 0, g2, vk8);
            svmla_single_za32_f32_vg1x4(slice + 1, g2, vk7);
            svmla_single_za32_f32_vg1x4(slice + 2, g2, vk6);
            r2 = svld1_f32(pg, col2_ptr2);
            col2_ptr2 += st2;
            r3 = svld1_f32(pg, col2_ptr3);
            col2_ptr3 += st3;

            const svfloat32x4_t vacc = svread_za32_f32_vg1x4(slice + 0);
            ++slice;
            r4 = svld1_f32(pg, col2_ptr4);
            col2_ptr4 += st4;
            svwrite_za32_f32_vg1x4(slice + 2, vbiasx4);
            r5 = svld1_f32(pg, col2_ptr5);
            col2_ptr5 += st5;

            const svfloat32x4_t clamped =
                svclamp_single_f32_x4(vacc, vmin, vmax);
            svst1_f32(pg, out_r0, svget4_f32(clamped, 0));
            out_r0 += out_col_stride;
            svst1_f32(pg, out_r1, svget4_f32(clamped, 1));
            out_r1 += out_st1;
            svst1_f32(pg, out_r2, svget4_f32(clamped, 2));
            out_r2 += out_st2;
            svst1_f32(pg, out_r3, svget4_f32(clamped, 3));
            out_r3 += out_st3;
          }

          {
            const svfloat32x4_t g0 = svcreate4_f32(r0, r1, r2, r3);
            svmla_single_za32_f32_vg1x4(slice + 0, g0, vk2);
            svmla_single_za32_f32_vg1x4(slice + 1, g0, vk1);
            svmla_single_za32_f32_vg1x4(slice + 2, g0, vk0);
            const svfloat32x4_t g1 = svcreate4_f32(r1, r2, r3, r4);
            svmla_single_za32_f32_vg1x4(slice + 0, g1, vk5);
            svmla_single_za32_f32_vg1x4(slice + 1, g1, vk4);
            svmla_single_za32_f32_vg1x4(slice + 2, g1, vk3);
            const svfloat32x4_t g2 = svcreate4_f32(r2, r3, r4, r5);
            svmla_single_za32_f32_vg1x4(slice + 0, g2, vk8);
            svmla_single_za32_f32_vg1x4(slice + 1, g2, vk7);
            svmla_single_za32_f32_vg1x4(slice + 2, g2, vk6);

            const svfloat32x4_t vacc = svread_za32_f32_vg1x4(slice + 0);
            ++slice;
            svwrite_za32_f32_vg1x4(slice + 2, vbiasx4);
            const svfloat32x4_t clamped =
                svclamp_single_f32_x4(vacc, vmin, vmax);
            svst1_f32(pg, out_r0, svget4_f32(clamped, 0));
            out_r0 += out_col_stride;
            svst1_f32(pg, out_r1, svget4_f32(clamped, 1));
            out_r1 += out_st1;
            svst1_f32(pg, out_r2, svget4_f32(clamped, 2));
            out_r2 += out_st2;
            svst1_f32(pg, out_r3, svget4_f32(clamped, 3));
            out_r3 += out_st3;
            ++ox;
          }
        }
      }
    }

    for (; ox < output_cols; ++ox) {
      const size_t vcol2 = ox + 2;
      svfloat32_t r0 = vpad;
      svfloat32_t r1 = vpad;
      svfloat32_t r2 = vpad;
      svfloat32_t r3 = vpad;
      svfloat32_t r4 = vpad;
      svfloat32_t r5 = vpad;
      if (vcol2 >= input_padding_left &&
          (vcol2 - input_padding_left) < valid_input_cols) {
        const size_t icol2 = vcol2 - input_padding_left;
        r0 = svld1_f32(pg, in_rows[0] + icol2 * row_col_strides[0]);
        r1 = svld1_f32(pg, in_rows[1] + icol2 * row_col_strides[1]);
        r2 = svld1_f32(pg, in_rows[2] + icol2 * row_col_strides[2]);
        r3 = svld1_f32(pg, in_rows[3] + icol2 * row_col_strides[3]);
        r4 = svld1_f32(pg, in_rows[4] + icol2 * row_col_strides[4]);
        r5 = svld1_f32(pg, in_rows[5] + icol2 * row_col_strides[5]);
      }

      const svfloat32x4_t g0 = svcreate4_f32(r0, r1, r2, r3);
      const svfloat32x4_t g1 = svcreate4_f32(r1, r2, r3, r4);
      const svfloat32x4_t g2 = svcreate4_f32(r2, r3, r4, r5);
      svmla_single_za32_f32_vg1x4(slice + 0, g0, vk2);
      svmla_single_za32_f32_vg1x4(slice + 0, g1, vk5);
      svmla_single_za32_f32_vg1x4(slice + 0, g2, vk8);

      if (ox + 1 < output_cols) {
        svmla_single_za32_f32_vg1x4(slice + 1, g0, vk1);
        svmla_single_za32_f32_vg1x4(slice + 2, g0, vk0);
        svmla_single_za32_f32_vg1x4(slice + 1, g1, vk4);
        svmla_single_za32_f32_vg1x4(slice + 2, g1, vk3);
        svmla_single_za32_f32_vg1x4(slice + 1, g2, vk7);
        svmla_single_za32_f32_vg1x4(slice + 2, g2, vk6);
      }

      const svfloat32x4_t clamped =
          svclamp_single_f32_x4(svread_za32_f32_vg1x4(slice + 0), vmin, vmax);
      svst1_f32(pg, out_r0, svget4_f32(clamped, 0));
      out_r0 += out_col_stride;
      svst1_f32(pg, out_r1, svget4_f32(clamped, 1));
      out_r1 += out_st1;
      svst1_f32(pg, out_r2, svget4_f32(clamped, 2));
      out_r2 += out_st2;
      svst1_f32(pg, out_r3, svget4_f32(clamped, 3));
      out_r3 += out_st3;

      if (ox + 1 < output_cols) {
        ++slice;
        svwrite_za32_f32_vg1x4(slice + 2, vbiasx4);
      }
    }
  }
}

void xnn_kai_f32_dwconv_minmax_ukernel_9pvc__neonsme2(
    const void* input, const void* packed_weights, void* output,
    size_t input_height_stride, size_t input_pixel_stride,
    size_t output_height_stride, size_t output_pixel_stride,
    size_t valid_input_rows, size_t valid_output_rows,
    size_t input_padding_left, size_t input_padding_top, float padding_value,
    float output_min, float output_max) {
  const size_t num_channels = output_pixel_stride / sizeof(float);
  const size_t output_cols = output_height_stride / output_pixel_stride;
  if (valid_output_rows == 0 || output_cols == 0 || num_channels == 0) {
    return;
  }
  xnn_f32_dwconv_9pvc_sme2_impl(
      input, packed_weights, output, input_height_stride, input_pixel_stride,
      output_height_stride, output_pixel_stride, valid_input_rows,
      valid_output_rows, input_padding_left, input_padding_top, padding_value,
      output_min, output_max);
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
