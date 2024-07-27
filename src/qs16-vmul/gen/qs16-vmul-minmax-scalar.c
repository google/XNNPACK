// Auto-generated file. Do not edit!
//   Template: src/qs16-vmul/qs16-vmul.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/simd/s16-scalar.h"
#include "xnnpack/simd/s32-scalar.h"
#include "xnnpack/simd/f32-scalar.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vbinary.h"


void xnn_qs16_vmul_minmax_ukernel__scalar_u1(
    size_t batch,
    const int16_t* input_a,
    const int16_t* input_b,
    int16_t* output,
    const union xnn_qs16_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s16 == 1);

  xnn_simd_s32_t vzero_point_a = xnn_set1_s32(params->qs16_scalar.a_zero_point);
  xnn_simd_s32_t vzero_point_b = xnn_set1_s32(params->qs16_scalar.b_zero_point);
  xnn_simd_s32_t vzero_point_output = xnn_set1_s32(params->qs16_scalar.output_zero_point);

  xnn_simd_f32_t vscale = xnn_set1_f32(params->qs16_scalar.scale);

  for (; batch >= xnn_simd_bytes_s16; batch -= xnn_simd_bytes_s16) {
    xnn_simd_s16_t vin1 = xnn_loadu_s16(input_a);
    input_a += xnn_simd_size_s16;

    xnn_simd_s16_t vin2 = xnn_loadu_s16(input_b);
    input_b += xnn_simd_size_s16;

    xnn_simd_s32_t vin1_low = xnn_low_cvt_s16_s32(vin1);
    xnn_simd_s32_t vin1_high = xnn_high_cvt_s16_s32(vin1);
    vin1_low = xnn_sub_s32(vin1_low, vzero_point_a);
    vin1_high = xnn_sub_s32(vin1_high, vzero_point_a);

    xnn_simd_s32_t vin2_low = xnn_low_cvt_s16_s32(vin2);
    xnn_simd_s32_t vin2_high = xnn_high_cvt_s16_s32(vin2);
    vin2_low = xnn_sub_s32(vin2_low, vzero_point_b);
    vin2_high = xnn_sub_s32(vin2_high, vzero_point_b);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low, vin2_low);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high, vin2_high);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low), vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high), vscale);

    vy_s32_low = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled), vzero_point_output);
    vy_s32_high = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled), vzero_point_output);

    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low, vy_s32_high);

    xnn_storeu_s16(output, vy);
    output += xnn_simd_size_s16;
  }
}

void xnn_qs16_vmul_minmax_ukernel__scalar_u2(
    size_t batch,
    const int16_t* input_a,
    const int16_t* input_b,
    int16_t* output,
    const union xnn_qs16_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s16 == 1);

  xnn_simd_s32_t vzero_point_a = xnn_set1_s32(params->qs16_scalar.a_zero_point);
  xnn_simd_s32_t vzero_point_b = xnn_set1_s32(params->qs16_scalar.b_zero_point);
  xnn_simd_s32_t vzero_point_output = xnn_set1_s32(params->qs16_scalar.output_zero_point);

  xnn_simd_f32_t vscale = xnn_set1_f32(params->qs16_scalar.scale);

  for (; batch >= 2 * sizeof(int16_t); batch -= 2 * sizeof(int16_t)) {
    xnn_simd_s16_t vin1_0 = xnn_loadu_s16(input_a);
    xnn_simd_s16_t vin1_1 = xnn_loadu_s16(input_a + 1 * xnn_simd_size_s16);
    input_a += 2;

    xnn_simd_s16_t vin2_0 = xnn_loadu_s16(input_b);
    xnn_simd_s16_t vin2_1 = (xnn_loadu_s16(input_b + 1 * xnn_simd_size_s16));
    input_b += 2;

    xnn_simd_s32_t vin1_low_0 = xnn_low_cvt_s16_s32(vin1_0);
    xnn_simd_s32_t vin1_high_0 = xnn_high_cvt_s16_s32(vin1_0);
    vin1_low_0 = xnn_sub_s32(vin1_low_0, vzero_point_a);
    vin1_high_0 = xnn_sub_s32(vin1_high_0, vzero_point_a);

    xnn_simd_s32_t vin2_low_0 = xnn_low_cvt_s16_s32(vin2_0);
    xnn_simd_s32_t vin2_high_0 = xnn_high_cvt_s16_s32(vin2_0);
    vin2_low_0 = xnn_sub_s32(vin2_low_0, vzero_point_b);
    vin2_high_0 = xnn_sub_s32(vin2_high_0, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_0 = xnn_mul_s32(vin1_low_0, vin2_low_0);
    xnn_simd_s32_t vy_s32_high_0 = xnn_mul_s32(vin1_high_0, vin2_high_0);

    xnn_simd_f32_t vy_f32_low_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_0), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_0), vscale);

    vy_s32_low_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_0), vzero_point_output);
    vy_s32_high_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_0), vzero_point_output);

    xnn_simd_s16_t vy_0 = xnn_cvt_s32_s16(vy_s32_low_0, vy_s32_high_0);
    xnn_simd_s32_t vin1_low_1 = xnn_low_cvt_s16_s32(vin1_1);
    xnn_simd_s32_t vin1_high_1 = xnn_high_cvt_s16_s32(vin1_1);
    vin1_low_1 = xnn_sub_s32(vin1_low_1, vzero_point_a);
    vin1_high_1 = xnn_sub_s32(vin1_high_1, vzero_point_a);

    xnn_simd_s32_t vin2_low_1 = xnn_low_cvt_s16_s32(vin2_1);
    xnn_simd_s32_t vin2_high_1 = xnn_high_cvt_s16_s32(vin2_1);
    vin2_low_1 = xnn_sub_s32(vin2_low_1, vzero_point_b);
    vin2_high_1 = xnn_sub_s32(vin2_high_1, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_1 = xnn_mul_s32(vin1_low_1, vin2_low_1);
    xnn_simd_s32_t vy_s32_high_1 = xnn_mul_s32(vin1_high_1, vin2_high_1);

    xnn_simd_f32_t vy_f32_low_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_1), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_1), vscale);

    vy_s32_low_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_1), vzero_point_output);
    vy_s32_high_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_1), vzero_point_output);

    xnn_simd_s16_t vy_1 = xnn_cvt_s32_s16(vy_s32_low_1, vy_s32_high_1);

    xnn_storeu_s16(output, vy_0);
    xnn_storeu_s16(output + 1 * xnn_simd_size_s16, vy_1);
    output += 2;
  }
  for (; batch >= xnn_simd_bytes_s16; batch -= xnn_simd_bytes_s16) {
    xnn_simd_s16_t vin1 = xnn_loadu_s16(input_a);
    input_a += xnn_simd_size_s16;

    xnn_simd_s16_t vin2 = xnn_loadu_s16(input_b);
    input_b += xnn_simd_size_s16;

    xnn_simd_s32_t vin1_low = xnn_low_cvt_s16_s32(vin1);
    xnn_simd_s32_t vin1_high = xnn_high_cvt_s16_s32(vin1);
    vin1_low = xnn_sub_s32(vin1_low, vzero_point_a);
    vin1_high = xnn_sub_s32(vin1_high, vzero_point_a);

    xnn_simd_s32_t vin2_low = xnn_low_cvt_s16_s32(vin2);
    xnn_simd_s32_t vin2_high = xnn_high_cvt_s16_s32(vin2);
    vin2_low = xnn_sub_s32(vin2_low, vzero_point_b);
    vin2_high = xnn_sub_s32(vin2_high, vzero_point_b);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low, vin2_low);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high, vin2_high);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low), vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high), vscale);

    vy_s32_low = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled), vzero_point_output);
    vy_s32_high = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled), vzero_point_output);

    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low, vy_s32_high);

    xnn_storeu_s16(output, vy);
    output += xnn_simd_size_s16;
  }
}

void xnn_qs16_vmul_minmax_ukernel__scalar_u4(
    size_t batch,
    const int16_t* input_a,
    const int16_t* input_b,
    int16_t* output,
    const union xnn_qs16_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s16 == 1);

  xnn_simd_s32_t vzero_point_a = xnn_set1_s32(params->qs16_scalar.a_zero_point);
  xnn_simd_s32_t vzero_point_b = xnn_set1_s32(params->qs16_scalar.b_zero_point);
  xnn_simd_s32_t vzero_point_output = xnn_set1_s32(params->qs16_scalar.output_zero_point);

  xnn_simd_f32_t vscale = xnn_set1_f32(params->qs16_scalar.scale);

  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    xnn_simd_s16_t vin1_0 = xnn_loadu_s16(input_a);
    xnn_simd_s16_t vin1_1 = xnn_loadu_s16(input_a + 1 * xnn_simd_size_s16);
    xnn_simd_s16_t vin1_2 = xnn_loadu_s16(input_a + 2 * xnn_simd_size_s16);
    xnn_simd_s16_t vin1_3 = xnn_loadu_s16(input_a + 3 * xnn_simd_size_s16);
    input_a += 4;

    xnn_simd_s16_t vin2_0 = xnn_loadu_s16(input_b);
    xnn_simd_s16_t vin2_1 = (xnn_loadu_s16(input_b + 1 * xnn_simd_size_s16));
    xnn_simd_s16_t vin2_2 = (xnn_loadu_s16(input_b + 2 * xnn_simd_size_s16));
    xnn_simd_s16_t vin2_3 = (xnn_loadu_s16(input_b + 3 * xnn_simd_size_s16));
    input_b += 4;

    xnn_simd_s32_t vin1_low_0 = xnn_low_cvt_s16_s32(vin1_0);
    xnn_simd_s32_t vin1_high_0 = xnn_high_cvt_s16_s32(vin1_0);
    vin1_low_0 = xnn_sub_s32(vin1_low_0, vzero_point_a);
    vin1_high_0 = xnn_sub_s32(vin1_high_0, vzero_point_a);

    xnn_simd_s32_t vin2_low_0 = xnn_low_cvt_s16_s32(vin2_0);
    xnn_simd_s32_t vin2_high_0 = xnn_high_cvt_s16_s32(vin2_0);
    vin2_low_0 = xnn_sub_s32(vin2_low_0, vzero_point_b);
    vin2_high_0 = xnn_sub_s32(vin2_high_0, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_0 = xnn_mul_s32(vin1_low_0, vin2_low_0);
    xnn_simd_s32_t vy_s32_high_0 = xnn_mul_s32(vin1_high_0, vin2_high_0);

    xnn_simd_f32_t vy_f32_low_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_0), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_0), vscale);

    vy_s32_low_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_0), vzero_point_output);
    vy_s32_high_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_0), vzero_point_output);

    xnn_simd_s16_t vy_0 = xnn_cvt_s32_s16(vy_s32_low_0, vy_s32_high_0);
    xnn_simd_s32_t vin1_low_1 = xnn_low_cvt_s16_s32(vin1_1);
    xnn_simd_s32_t vin1_high_1 = xnn_high_cvt_s16_s32(vin1_1);
    vin1_low_1 = xnn_sub_s32(vin1_low_1, vzero_point_a);
    vin1_high_1 = xnn_sub_s32(vin1_high_1, vzero_point_a);

    xnn_simd_s32_t vin2_low_1 = xnn_low_cvt_s16_s32(vin2_1);
    xnn_simd_s32_t vin2_high_1 = xnn_high_cvt_s16_s32(vin2_1);
    vin2_low_1 = xnn_sub_s32(vin2_low_1, vzero_point_b);
    vin2_high_1 = xnn_sub_s32(vin2_high_1, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_1 = xnn_mul_s32(vin1_low_1, vin2_low_1);
    xnn_simd_s32_t vy_s32_high_1 = xnn_mul_s32(vin1_high_1, vin2_high_1);

    xnn_simd_f32_t vy_f32_low_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_1), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_1), vscale);

    vy_s32_low_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_1), vzero_point_output);
    vy_s32_high_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_1), vzero_point_output);

    xnn_simd_s16_t vy_1 = xnn_cvt_s32_s16(vy_s32_low_1, vy_s32_high_1);
    xnn_simd_s32_t vin1_low_2 = xnn_low_cvt_s16_s32(vin1_2);
    xnn_simd_s32_t vin1_high_2 = xnn_high_cvt_s16_s32(vin1_2);
    vin1_low_2 = xnn_sub_s32(vin1_low_2, vzero_point_a);
    vin1_high_2 = xnn_sub_s32(vin1_high_2, vzero_point_a);

    xnn_simd_s32_t vin2_low_2 = xnn_low_cvt_s16_s32(vin2_2);
    xnn_simd_s32_t vin2_high_2 = xnn_high_cvt_s16_s32(vin2_2);
    vin2_low_2 = xnn_sub_s32(vin2_low_2, vzero_point_b);
    vin2_high_2 = xnn_sub_s32(vin2_high_2, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_2 = xnn_mul_s32(vin1_low_2, vin2_low_2);
    xnn_simd_s32_t vy_s32_high_2 = xnn_mul_s32(vin1_high_2, vin2_high_2);

    xnn_simd_f32_t vy_f32_low_scaled_2 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_2), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_2 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_2), vscale);

    vy_s32_low_2 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_2), vzero_point_output);
    vy_s32_high_2 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_2), vzero_point_output);

    xnn_simd_s16_t vy_2 = xnn_cvt_s32_s16(vy_s32_low_2, vy_s32_high_2);
    xnn_simd_s32_t vin1_low_3 = xnn_low_cvt_s16_s32(vin1_3);
    xnn_simd_s32_t vin1_high_3 = xnn_high_cvt_s16_s32(vin1_3);
    vin1_low_3 = xnn_sub_s32(vin1_low_3, vzero_point_a);
    vin1_high_3 = xnn_sub_s32(vin1_high_3, vzero_point_a);

    xnn_simd_s32_t vin2_low_3 = xnn_low_cvt_s16_s32(vin2_3);
    xnn_simd_s32_t vin2_high_3 = xnn_high_cvt_s16_s32(vin2_3);
    vin2_low_3 = xnn_sub_s32(vin2_low_3, vzero_point_b);
    vin2_high_3 = xnn_sub_s32(vin2_high_3, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_3 = xnn_mul_s32(vin1_low_3, vin2_low_3);
    xnn_simd_s32_t vy_s32_high_3 = xnn_mul_s32(vin1_high_3, vin2_high_3);

    xnn_simd_f32_t vy_f32_low_scaled_3 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_3), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_3 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_3), vscale);

    vy_s32_low_3 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_3), vzero_point_output);
    vy_s32_high_3 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_3), vzero_point_output);

    xnn_simd_s16_t vy_3 = xnn_cvt_s32_s16(vy_s32_low_3, vy_s32_high_3);

    xnn_storeu_s16(output, vy_0);
    xnn_storeu_s16(output + 1 * xnn_simd_size_s16, vy_1);
    xnn_storeu_s16(output + 2 * xnn_simd_size_s16, vy_2);
    xnn_storeu_s16(output + 3 * xnn_simd_size_s16, vy_3);
    output += 4;
  }
  for (; batch >= xnn_simd_bytes_s16; batch -= xnn_simd_bytes_s16) {
    xnn_simd_s16_t vin1 = xnn_loadu_s16(input_a);
    input_a += xnn_simd_size_s16;

    xnn_simd_s16_t vin2 = xnn_loadu_s16(input_b);
    input_b += xnn_simd_size_s16;

    xnn_simd_s32_t vin1_low = xnn_low_cvt_s16_s32(vin1);
    xnn_simd_s32_t vin1_high = xnn_high_cvt_s16_s32(vin1);
    vin1_low = xnn_sub_s32(vin1_low, vzero_point_a);
    vin1_high = xnn_sub_s32(vin1_high, vzero_point_a);

    xnn_simd_s32_t vin2_low = xnn_low_cvt_s16_s32(vin2);
    xnn_simd_s32_t vin2_high = xnn_high_cvt_s16_s32(vin2);
    vin2_low = xnn_sub_s32(vin2_low, vzero_point_b);
    vin2_high = xnn_sub_s32(vin2_high, vzero_point_b);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low, vin2_low);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high, vin2_high);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low), vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high), vscale);

    vy_s32_low = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled), vzero_point_output);
    vy_s32_high = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled), vzero_point_output);

    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low, vy_s32_high);

    xnn_storeu_s16(output, vy);
    output += xnn_simd_size_s16;
  }
}

void xnn_qs16_vmul_minmax_ukernel__scalar_u8(
    size_t batch,
    const int16_t* input_a,
    const int16_t* input_b,
    int16_t* output,
    const union xnn_qs16_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s16 == 1);

  xnn_simd_s32_t vzero_point_a = xnn_set1_s32(params->qs16_scalar.a_zero_point);
  xnn_simd_s32_t vzero_point_b = xnn_set1_s32(params->qs16_scalar.b_zero_point);
  xnn_simd_s32_t vzero_point_output = xnn_set1_s32(params->qs16_scalar.output_zero_point);

  xnn_simd_f32_t vscale = xnn_set1_f32(params->qs16_scalar.scale);

  for (; batch >= 8 * sizeof(int16_t); batch -= 8 * sizeof(int16_t)) {
    xnn_simd_s16_t vin1_0 = xnn_loadu_s16(input_a);
    xnn_simd_s16_t vin1_1 = xnn_loadu_s16(input_a + 1 * xnn_simd_size_s16);
    xnn_simd_s16_t vin1_2 = xnn_loadu_s16(input_a + 2 * xnn_simd_size_s16);
    xnn_simd_s16_t vin1_3 = xnn_loadu_s16(input_a + 3 * xnn_simd_size_s16);
    xnn_simd_s16_t vin1_4 = xnn_loadu_s16(input_a + 4 * xnn_simd_size_s16);
    xnn_simd_s16_t vin1_5 = xnn_loadu_s16(input_a + 5 * xnn_simd_size_s16);
    xnn_simd_s16_t vin1_6 = xnn_loadu_s16(input_a + 6 * xnn_simd_size_s16);
    xnn_simd_s16_t vin1_7 = xnn_loadu_s16(input_a + 7 * xnn_simd_size_s16);
    input_a += 8;

    xnn_simd_s16_t vin2_0 = xnn_loadu_s16(input_b);
    xnn_simd_s16_t vin2_1 = (xnn_loadu_s16(input_b + 1 * xnn_simd_size_s16));
    xnn_simd_s16_t vin2_2 = (xnn_loadu_s16(input_b + 2 * xnn_simd_size_s16));
    xnn_simd_s16_t vin2_3 = (xnn_loadu_s16(input_b + 3 * xnn_simd_size_s16));
    xnn_simd_s16_t vin2_4 = (xnn_loadu_s16(input_b + 4 * xnn_simd_size_s16));
    xnn_simd_s16_t vin2_5 = (xnn_loadu_s16(input_b + 5 * xnn_simd_size_s16));
    xnn_simd_s16_t vin2_6 = (xnn_loadu_s16(input_b + 6 * xnn_simd_size_s16));
    xnn_simd_s16_t vin2_7 = (xnn_loadu_s16(input_b + 7 * xnn_simd_size_s16));
    input_b += 8;

    xnn_simd_s32_t vin1_low_0 = xnn_low_cvt_s16_s32(vin1_0);
    xnn_simd_s32_t vin1_high_0 = xnn_high_cvt_s16_s32(vin1_0);
    vin1_low_0 = xnn_sub_s32(vin1_low_0, vzero_point_a);
    vin1_high_0 = xnn_sub_s32(vin1_high_0, vzero_point_a);

    xnn_simd_s32_t vin2_low_0 = xnn_low_cvt_s16_s32(vin2_0);
    xnn_simd_s32_t vin2_high_0 = xnn_high_cvt_s16_s32(vin2_0);
    vin2_low_0 = xnn_sub_s32(vin2_low_0, vzero_point_b);
    vin2_high_0 = xnn_sub_s32(vin2_high_0, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_0 = xnn_mul_s32(vin1_low_0, vin2_low_0);
    xnn_simd_s32_t vy_s32_high_0 = xnn_mul_s32(vin1_high_0, vin2_high_0);

    xnn_simd_f32_t vy_f32_low_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_0), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_0), vscale);

    vy_s32_low_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_0), vzero_point_output);
    vy_s32_high_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_0), vzero_point_output);

    xnn_simd_s16_t vy_0 = xnn_cvt_s32_s16(vy_s32_low_0, vy_s32_high_0);
    xnn_simd_s32_t vin1_low_1 = xnn_low_cvt_s16_s32(vin1_1);
    xnn_simd_s32_t vin1_high_1 = xnn_high_cvt_s16_s32(vin1_1);
    vin1_low_1 = xnn_sub_s32(vin1_low_1, vzero_point_a);
    vin1_high_1 = xnn_sub_s32(vin1_high_1, vzero_point_a);

    xnn_simd_s32_t vin2_low_1 = xnn_low_cvt_s16_s32(vin2_1);
    xnn_simd_s32_t vin2_high_1 = xnn_high_cvt_s16_s32(vin2_1);
    vin2_low_1 = xnn_sub_s32(vin2_low_1, vzero_point_b);
    vin2_high_1 = xnn_sub_s32(vin2_high_1, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_1 = xnn_mul_s32(vin1_low_1, vin2_low_1);
    xnn_simd_s32_t vy_s32_high_1 = xnn_mul_s32(vin1_high_1, vin2_high_1);

    xnn_simd_f32_t vy_f32_low_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_1), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_1), vscale);

    vy_s32_low_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_1), vzero_point_output);
    vy_s32_high_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_1), vzero_point_output);

    xnn_simd_s16_t vy_1 = xnn_cvt_s32_s16(vy_s32_low_1, vy_s32_high_1);
    xnn_simd_s32_t vin1_low_2 = xnn_low_cvt_s16_s32(vin1_2);
    xnn_simd_s32_t vin1_high_2 = xnn_high_cvt_s16_s32(vin1_2);
    vin1_low_2 = xnn_sub_s32(vin1_low_2, vzero_point_a);
    vin1_high_2 = xnn_sub_s32(vin1_high_2, vzero_point_a);

    xnn_simd_s32_t vin2_low_2 = xnn_low_cvt_s16_s32(vin2_2);
    xnn_simd_s32_t vin2_high_2 = xnn_high_cvt_s16_s32(vin2_2);
    vin2_low_2 = xnn_sub_s32(vin2_low_2, vzero_point_b);
    vin2_high_2 = xnn_sub_s32(vin2_high_2, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_2 = xnn_mul_s32(vin1_low_2, vin2_low_2);
    xnn_simd_s32_t vy_s32_high_2 = xnn_mul_s32(vin1_high_2, vin2_high_2);

    xnn_simd_f32_t vy_f32_low_scaled_2 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_2), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_2 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_2), vscale);

    vy_s32_low_2 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_2), vzero_point_output);
    vy_s32_high_2 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_2), vzero_point_output);

    xnn_simd_s16_t vy_2 = xnn_cvt_s32_s16(vy_s32_low_2, vy_s32_high_2);
    xnn_simd_s32_t vin1_low_3 = xnn_low_cvt_s16_s32(vin1_3);
    xnn_simd_s32_t vin1_high_3 = xnn_high_cvt_s16_s32(vin1_3);
    vin1_low_3 = xnn_sub_s32(vin1_low_3, vzero_point_a);
    vin1_high_3 = xnn_sub_s32(vin1_high_3, vzero_point_a);

    xnn_simd_s32_t vin2_low_3 = xnn_low_cvt_s16_s32(vin2_3);
    xnn_simd_s32_t vin2_high_3 = xnn_high_cvt_s16_s32(vin2_3);
    vin2_low_3 = xnn_sub_s32(vin2_low_3, vzero_point_b);
    vin2_high_3 = xnn_sub_s32(vin2_high_3, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_3 = xnn_mul_s32(vin1_low_3, vin2_low_3);
    xnn_simd_s32_t vy_s32_high_3 = xnn_mul_s32(vin1_high_3, vin2_high_3);

    xnn_simd_f32_t vy_f32_low_scaled_3 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_3), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_3 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_3), vscale);

    vy_s32_low_3 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_3), vzero_point_output);
    vy_s32_high_3 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_3), vzero_point_output);

    xnn_simd_s16_t vy_3 = xnn_cvt_s32_s16(vy_s32_low_3, vy_s32_high_3);
    xnn_simd_s32_t vin1_low_4 = xnn_low_cvt_s16_s32(vin1_4);
    xnn_simd_s32_t vin1_high_4 = xnn_high_cvt_s16_s32(vin1_4);
    vin1_low_4 = xnn_sub_s32(vin1_low_4, vzero_point_a);
    vin1_high_4 = xnn_sub_s32(vin1_high_4, vzero_point_a);

    xnn_simd_s32_t vin2_low_4 = xnn_low_cvt_s16_s32(vin2_4);
    xnn_simd_s32_t vin2_high_4 = xnn_high_cvt_s16_s32(vin2_4);
    vin2_low_4 = xnn_sub_s32(vin2_low_4, vzero_point_b);
    vin2_high_4 = xnn_sub_s32(vin2_high_4, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_4 = xnn_mul_s32(vin1_low_4, vin2_low_4);
    xnn_simd_s32_t vy_s32_high_4 = xnn_mul_s32(vin1_high_4, vin2_high_4);

    xnn_simd_f32_t vy_f32_low_scaled_4 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_4), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_4 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_4), vscale);

    vy_s32_low_4 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_4), vzero_point_output);
    vy_s32_high_4 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_4), vzero_point_output);

    xnn_simd_s16_t vy_4 = xnn_cvt_s32_s16(vy_s32_low_4, vy_s32_high_4);
    xnn_simd_s32_t vin1_low_5 = xnn_low_cvt_s16_s32(vin1_5);
    xnn_simd_s32_t vin1_high_5 = xnn_high_cvt_s16_s32(vin1_5);
    vin1_low_5 = xnn_sub_s32(vin1_low_5, vzero_point_a);
    vin1_high_5 = xnn_sub_s32(vin1_high_5, vzero_point_a);

    xnn_simd_s32_t vin2_low_5 = xnn_low_cvt_s16_s32(vin2_5);
    xnn_simd_s32_t vin2_high_5 = xnn_high_cvt_s16_s32(vin2_5);
    vin2_low_5 = xnn_sub_s32(vin2_low_5, vzero_point_b);
    vin2_high_5 = xnn_sub_s32(vin2_high_5, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_5 = xnn_mul_s32(vin1_low_5, vin2_low_5);
    xnn_simd_s32_t vy_s32_high_5 = xnn_mul_s32(vin1_high_5, vin2_high_5);

    xnn_simd_f32_t vy_f32_low_scaled_5 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_5), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_5 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_5), vscale);

    vy_s32_low_5 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_5), vzero_point_output);
    vy_s32_high_5 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_5), vzero_point_output);

    xnn_simd_s16_t vy_5 = xnn_cvt_s32_s16(vy_s32_low_5, vy_s32_high_5);
    xnn_simd_s32_t vin1_low_6 = xnn_low_cvt_s16_s32(vin1_6);
    xnn_simd_s32_t vin1_high_6 = xnn_high_cvt_s16_s32(vin1_6);
    vin1_low_6 = xnn_sub_s32(vin1_low_6, vzero_point_a);
    vin1_high_6 = xnn_sub_s32(vin1_high_6, vzero_point_a);

    xnn_simd_s32_t vin2_low_6 = xnn_low_cvt_s16_s32(vin2_6);
    xnn_simd_s32_t vin2_high_6 = xnn_high_cvt_s16_s32(vin2_6);
    vin2_low_6 = xnn_sub_s32(vin2_low_6, vzero_point_b);
    vin2_high_6 = xnn_sub_s32(vin2_high_6, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_6 = xnn_mul_s32(vin1_low_6, vin2_low_6);
    xnn_simd_s32_t vy_s32_high_6 = xnn_mul_s32(vin1_high_6, vin2_high_6);

    xnn_simd_f32_t vy_f32_low_scaled_6 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_6), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_6 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_6), vscale);

    vy_s32_low_6 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_6), vzero_point_output);
    vy_s32_high_6 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_6), vzero_point_output);

    xnn_simd_s16_t vy_6 = xnn_cvt_s32_s16(vy_s32_low_6, vy_s32_high_6);
    xnn_simd_s32_t vin1_low_7 = xnn_low_cvt_s16_s32(vin1_7);
    xnn_simd_s32_t vin1_high_7 = xnn_high_cvt_s16_s32(vin1_7);
    vin1_low_7 = xnn_sub_s32(vin1_low_7, vzero_point_a);
    vin1_high_7 = xnn_sub_s32(vin1_high_7, vzero_point_a);

    xnn_simd_s32_t vin2_low_7 = xnn_low_cvt_s16_s32(vin2_7);
    xnn_simd_s32_t vin2_high_7 = xnn_high_cvt_s16_s32(vin2_7);
    vin2_low_7 = xnn_sub_s32(vin2_low_7, vzero_point_b);
    vin2_high_7 = xnn_sub_s32(vin2_high_7, vzero_point_b);

    xnn_simd_s32_t vy_s32_low_7 = xnn_mul_s32(vin1_low_7, vin2_low_7);
    xnn_simd_s32_t vy_s32_high_7 = xnn_mul_s32(vin1_high_7, vin2_high_7);

    xnn_simd_f32_t vy_f32_low_scaled_7 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_7), vscale);
    xnn_simd_f32_t vy_f32_high_scaled_7 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_7), vscale);

    vy_s32_low_7 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_7), vzero_point_output);
    vy_s32_high_7 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_7), vzero_point_output);

    xnn_simd_s16_t vy_7 = xnn_cvt_s32_s16(vy_s32_low_7, vy_s32_high_7);

    xnn_storeu_s16(output, vy_0);
    xnn_storeu_s16(output + 1 * xnn_simd_size_s16, vy_1);
    xnn_storeu_s16(output + 2 * xnn_simd_size_s16, vy_2);
    xnn_storeu_s16(output + 3 * xnn_simd_size_s16, vy_3);
    xnn_storeu_s16(output + 4 * xnn_simd_size_s16, vy_4);
    xnn_storeu_s16(output + 5 * xnn_simd_size_s16, vy_5);
    xnn_storeu_s16(output + 6 * xnn_simd_size_s16, vy_6);
    xnn_storeu_s16(output + 7 * xnn_simd_size_s16, vy_7);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_s16; batch -= xnn_simd_bytes_s16) {
    xnn_simd_s16_t vin1 = xnn_loadu_s16(input_a);
    input_a += xnn_simd_size_s16;

    xnn_simd_s16_t vin2 = xnn_loadu_s16(input_b);
    input_b += xnn_simd_size_s16;

    xnn_simd_s32_t vin1_low = xnn_low_cvt_s16_s32(vin1);
    xnn_simd_s32_t vin1_high = xnn_high_cvt_s16_s32(vin1);
    vin1_low = xnn_sub_s32(vin1_low, vzero_point_a);
    vin1_high = xnn_sub_s32(vin1_high, vzero_point_a);

    xnn_simd_s32_t vin2_low = xnn_low_cvt_s16_s32(vin2);
    xnn_simd_s32_t vin2_high = xnn_high_cvt_s16_s32(vin2);
    vin2_low = xnn_sub_s32(vin2_low, vzero_point_b);
    vin2_high = xnn_sub_s32(vin2_high, vzero_point_b);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low, vin2_low);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high, vin2_high);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low), vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high), vscale);

    vy_s32_low = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled), vzero_point_output);
    vy_s32_high = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled), vzero_point_output);

    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low, vy_s32_high);

    xnn_storeu_s16(output, vy);
    output += xnn_simd_size_s16;
  }
}
