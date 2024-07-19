// Auto-generated file. Do not edit!
//   Template: src/s16-vmul/s16-vmulc.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/simd/s16-avx2.h"
#include "xnnpack/simd/s32-avx2.h"
#include "xnnpack/simd/f32-avx2.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_s16_vmulc_ukernel__avx2_u8(
    size_t batch,
    const int16_t* input_a,
    const int16_t* input_b,
    int16_t* output,
    const union xnn_s16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s16 == 8);

  xnn_simd_s32_t vzero_point_a = xnn_set1_s32(params->fp32_scalar.a_zero_point);
  xnn_simd_s32_t vzero_point_b = xnn_set1_s32(params->fp32_scalar.b_zero_point);
  xnn_simd_s32_t vzero_point_output = xnn_set1_s32(params->fp32_scalar.output_zero_point);

  xnn_simd_f32_t vscale = xnn_set1_f32(params->fp32_scalar.scale);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input_b);
  xnn_simd_s32_t vin2_low = xnn_cvt_s16_s32(xnn_low_s16(vin2));
  xnn_simd_s32_t vin2_high = xnn_cvt_s16_s32(xnn_high_s16(vin2));
  xnn_simd_s32_t vin2_low_sub = xnn_sub_s32(vin2_low,vzero_point_b);
  xnn_simd_s32_t vin2_high_sub = xnn_sub_s32(vin2_high,vzero_point_b);

  for (; batch >= xnn_simd_bytes_s16; batch -= xnn_simd_bytes_s16) {
    xnn_simd_s16_t vin1 = xnn_loadu_s16(input_a);
    input_a += xnn_simd_size_s16;

    xnn_simd_s32_t vin1_low = xnn_cvt_s16_s32(xnn_low_s16(vin1));
    xnn_simd_s32_t vin1_high = xnn_cvt_s16_s32(xnn_high_s16(vin1));
    xnn_simd_s32_t vin1_low_sub = xnn_sub_s32(vin1_low,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub = xnn_sub_s32(vin1_high,vzero_point_a);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low_sub,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high_sub,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low),vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high),vscale);

    xnn_simd_s32_t vy_s32_low_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled),vzero_point_output);

    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low_add, vy_s32_high_add);

    xnn_storeu_s16(output, vy);
    output += xnn_simd_size_s16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s16_t vin1 = xnn_load_tail_s16(input_a, batch >> XNN_LOG2_SIZEOF_INT16_T);

    xnn_simd_s32_t vin1_low = xnn_cvt_s16_s32(xnn_low_s16(vin1));
    xnn_simd_s32_t vin1_high = xnn_cvt_s16_s32(xnn_high_s16(vin1));
    xnn_simd_s32_t vin1_low_sub = xnn_sub_s32(vin1_low,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub = xnn_sub_s32(vin1_high,vzero_point_a);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low_sub,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high_sub,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low),vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high),vscale);

    xnn_simd_s32_t vy_s32_low_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled),vzero_point_output);
    
    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low_add, vy_s32_high_add);

    xnn_store_tail_s16(output, vy, batch >> XNN_LOG2_SIZEOF_INT16_T);
  }
}

void xnn_s16_vmulc_ukernel__avx2_u16(
    size_t batch,
    const int16_t* input_a,
    const int16_t* input_b,
    int16_t* output,
    const union xnn_s16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s16 == 8);

  xnn_simd_s32_t vzero_point_a = xnn_set1_s32(params->fp32_scalar.a_zero_point);
  xnn_simd_s32_t vzero_point_b = xnn_set1_s32(params->fp32_scalar.b_zero_point);
  xnn_simd_s32_t vzero_point_output = xnn_set1_s32(params->fp32_scalar.output_zero_point);

  xnn_simd_f32_t vscale = xnn_set1_f32(params->fp32_scalar.scale);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input_b);
  xnn_simd_s32_t vin2_low = xnn_cvt_s16_s32(xnn_low_s16(vin2));
  xnn_simd_s32_t vin2_high = xnn_cvt_s16_s32(xnn_high_s16(vin2));
  xnn_simd_s32_t vin2_low_sub = xnn_sub_s32(vin2_low,vzero_point_b);
  xnn_simd_s32_t vin2_high_sub = xnn_sub_s32(vin2_high,vzero_point_b);

  for (; batch >= 16 * sizeof(int16_t); batch -= 16 * sizeof(int16_t)) {
    xnn_simd_s16_t vin1_0 = xnn_loadu_s16(input_a);
    xnn_simd_s16_t vin1_1 = xnn_loadu_s16(input_a + 1 * xnn_simd_size_s16);
    input_a += 16;

    xnn_simd_s32_t vin1_low_0 = xnn_cvt_s16_s32(xnn_low_s16(vin1_0));
    xnn_simd_s32_t vin1_high_0 = xnn_cvt_s16_s32(xnn_high_s16(vin1_0));
    xnn_simd_s32_t vin1_low_sub_0 = xnn_sub_s32(vin1_low_0,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub_0 = xnn_sub_s32(vin1_high_0,vzero_point_a);

    xnn_simd_s32_t vy_s32_low_0 = xnn_mul_s32(vin1_low_sub_0,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high_0 = xnn_mul_s32(vin1_high_sub_0,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_0),vscale);
    xnn_simd_f32_t vy_f32_high_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_0),vscale);

    xnn_simd_s32_t vy_s32_low_add_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_0),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_0),vzero_point_output);

    xnn_simd_s16_t vy_0 = xnn_cvt_s32_s16(vy_s32_low_add_0, vy_s32_high_add_0);
    xnn_simd_s32_t vin1_low_1 = xnn_cvt_s16_s32(xnn_low_s16(vin1_1));
    xnn_simd_s32_t vin1_high_1 = xnn_cvt_s16_s32(xnn_high_s16(vin1_1));
    xnn_simd_s32_t vin1_low_sub_1 = xnn_sub_s32(vin1_low_1,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub_1 = xnn_sub_s32(vin1_high_1,vzero_point_a);

    xnn_simd_s32_t vy_s32_low_1 = xnn_mul_s32(vin1_low_sub_1,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high_1 = xnn_mul_s32(vin1_high_sub_1,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_1),vscale);
    xnn_simd_f32_t vy_f32_high_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_1),vscale);

    xnn_simd_s32_t vy_s32_low_add_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_1),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_1),vzero_point_output);

    xnn_simd_s16_t vy_1 = xnn_cvt_s32_s16(vy_s32_low_add_1, vy_s32_high_add_1);

    xnn_storeu_s16(output, vy_0);
    xnn_storeu_s16(output + 1 * xnn_simd_size_s16, vy_1);
    output += 16;
  }
  for (; batch >= xnn_simd_bytes_s16; batch -= xnn_simd_bytes_s16) {
    xnn_simd_s16_t vin1 = xnn_loadu_s16(input_a);
    input_a += xnn_simd_size_s16;

    xnn_simd_s32_t vin1_low = xnn_cvt_s16_s32(xnn_low_s16(vin1));
    xnn_simd_s32_t vin1_high = xnn_cvt_s16_s32(xnn_high_s16(vin1));
    xnn_simd_s32_t vin1_low_sub = xnn_sub_s32(vin1_low,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub = xnn_sub_s32(vin1_high,vzero_point_a);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low_sub,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high_sub,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low),vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high),vscale);

    xnn_simd_s32_t vy_s32_low_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled),vzero_point_output);

    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low_add, vy_s32_high_add);

    xnn_storeu_s16(output, vy);
    output += xnn_simd_size_s16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s16_t vin1 = xnn_load_tail_s16(input_a, batch >> XNN_LOG2_SIZEOF_INT16_T);

    xnn_simd_s32_t vin1_low = xnn_cvt_s16_s32(xnn_low_s16(vin1));
    xnn_simd_s32_t vin1_high = xnn_cvt_s16_s32(xnn_high_s16(vin1));
    xnn_simd_s32_t vin1_low_sub = xnn_sub_s32(vin1_low,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub = xnn_sub_s32(vin1_high,vzero_point_a);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low_sub,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high_sub,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low),vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high),vscale);

    xnn_simd_s32_t vy_s32_low_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled),vzero_point_output);
    
    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low_add, vy_s32_high_add);

    xnn_store_tail_s16(output, vy, batch >> XNN_LOG2_SIZEOF_INT16_T);
  }
}

void xnn_s16_vmulc_ukernel__avx2_u24(
    size_t batch,
    const int16_t* input_a,
    const int16_t* input_b,
    int16_t* output,
    const union xnn_s16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s16 == 8);

  xnn_simd_s32_t vzero_point_a = xnn_set1_s32(params->fp32_scalar.a_zero_point);
  xnn_simd_s32_t vzero_point_b = xnn_set1_s32(params->fp32_scalar.b_zero_point);
  xnn_simd_s32_t vzero_point_output = xnn_set1_s32(params->fp32_scalar.output_zero_point);

  xnn_simd_f32_t vscale = xnn_set1_f32(params->fp32_scalar.scale);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input_b);
  xnn_simd_s32_t vin2_low = xnn_cvt_s16_s32(xnn_low_s16(vin2));
  xnn_simd_s32_t vin2_high = xnn_cvt_s16_s32(xnn_high_s16(vin2));
  xnn_simd_s32_t vin2_low_sub = xnn_sub_s32(vin2_low,vzero_point_b);
  xnn_simd_s32_t vin2_high_sub = xnn_sub_s32(vin2_high,vzero_point_b);

  for (; batch >= 24 * sizeof(int16_t); batch -= 24 * sizeof(int16_t)) {
    xnn_simd_s16_t vin1_0 = xnn_loadu_s16(input_a);
    xnn_simd_s16_t vin1_1 = xnn_loadu_s16(input_a + 1 * xnn_simd_size_s16);
    xnn_simd_s16_t vin1_2 = xnn_loadu_s16(input_a + 2 * xnn_simd_size_s16);
    input_a += 24;

    xnn_simd_s32_t vin1_low_0 = xnn_cvt_s16_s32(xnn_low_s16(vin1_0));
    xnn_simd_s32_t vin1_high_0 = xnn_cvt_s16_s32(xnn_high_s16(vin1_0));
    xnn_simd_s32_t vin1_low_sub_0 = xnn_sub_s32(vin1_low_0,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub_0 = xnn_sub_s32(vin1_high_0,vzero_point_a);

    xnn_simd_s32_t vy_s32_low_0 = xnn_mul_s32(vin1_low_sub_0,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high_0 = xnn_mul_s32(vin1_high_sub_0,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_0),vscale);
    xnn_simd_f32_t vy_f32_high_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_0),vscale);

    xnn_simd_s32_t vy_s32_low_add_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_0),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_0),vzero_point_output);

    xnn_simd_s16_t vy_0 = xnn_cvt_s32_s16(vy_s32_low_add_0, vy_s32_high_add_0);
    xnn_simd_s32_t vin1_low_1 = xnn_cvt_s16_s32(xnn_low_s16(vin1_1));
    xnn_simd_s32_t vin1_high_1 = xnn_cvt_s16_s32(xnn_high_s16(vin1_1));
    xnn_simd_s32_t vin1_low_sub_1 = xnn_sub_s32(vin1_low_1,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub_1 = xnn_sub_s32(vin1_high_1,vzero_point_a);

    xnn_simd_s32_t vy_s32_low_1 = xnn_mul_s32(vin1_low_sub_1,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high_1 = xnn_mul_s32(vin1_high_sub_1,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_1),vscale);
    xnn_simd_f32_t vy_f32_high_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_1),vscale);

    xnn_simd_s32_t vy_s32_low_add_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_1),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_1),vzero_point_output);

    xnn_simd_s16_t vy_1 = xnn_cvt_s32_s16(vy_s32_low_add_1, vy_s32_high_add_1);
    xnn_simd_s32_t vin1_low_2 = xnn_cvt_s16_s32(xnn_low_s16(vin1_2));
    xnn_simd_s32_t vin1_high_2 = xnn_cvt_s16_s32(xnn_high_s16(vin1_2));
    xnn_simd_s32_t vin1_low_sub_2 = xnn_sub_s32(vin1_low_2,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub_2 = xnn_sub_s32(vin1_high_2,vzero_point_a);

    xnn_simd_s32_t vy_s32_low_2 = xnn_mul_s32(vin1_low_sub_2,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high_2 = xnn_mul_s32(vin1_high_sub_2,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled_2 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_2),vscale);
    xnn_simd_f32_t vy_f32_high_scaled_2 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_2),vscale);

    xnn_simd_s32_t vy_s32_low_add_2 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_2),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add_2 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_2),vzero_point_output);

    xnn_simd_s16_t vy_2 = xnn_cvt_s32_s16(vy_s32_low_add_2, vy_s32_high_add_2);

    xnn_storeu_s16(output, vy_0);
    xnn_storeu_s16(output + 1 * xnn_simd_size_s16, vy_1);
    xnn_storeu_s16(output + 2 * xnn_simd_size_s16, vy_2);
    output += 24;
  }
  for (; batch >= xnn_simd_bytes_s16; batch -= xnn_simd_bytes_s16) {
    xnn_simd_s16_t vin1 = xnn_loadu_s16(input_a);
    input_a += xnn_simd_size_s16;

    xnn_simd_s32_t vin1_low = xnn_cvt_s16_s32(xnn_low_s16(vin1));
    xnn_simd_s32_t vin1_high = xnn_cvt_s16_s32(xnn_high_s16(vin1));
    xnn_simd_s32_t vin1_low_sub = xnn_sub_s32(vin1_low,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub = xnn_sub_s32(vin1_high,vzero_point_a);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low_sub,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high_sub,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low),vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high),vscale);

    xnn_simd_s32_t vy_s32_low_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled),vzero_point_output);

    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low_add, vy_s32_high_add);

    xnn_storeu_s16(output, vy);
    output += xnn_simd_size_s16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s16_t vin1 = xnn_load_tail_s16(input_a, batch >> XNN_LOG2_SIZEOF_INT16_T);

    xnn_simd_s32_t vin1_low = xnn_cvt_s16_s32(xnn_low_s16(vin1));
    xnn_simd_s32_t vin1_high = xnn_cvt_s16_s32(xnn_high_s16(vin1));
    xnn_simd_s32_t vin1_low_sub = xnn_sub_s32(vin1_low,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub = xnn_sub_s32(vin1_high,vzero_point_a);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low_sub,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high_sub,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low),vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high),vscale);

    xnn_simd_s32_t vy_s32_low_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled),vzero_point_output);
    
    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low_add, vy_s32_high_add);

    xnn_store_tail_s16(output, vy, batch >> XNN_LOG2_SIZEOF_INT16_T);
  }
}

void xnn_s16_vmulc_ukernel__avx2_u32(
    size_t batch,
    const int16_t* input_a,
    const int16_t* input_b,
    int16_t* output,
    const union xnn_s16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s16 == 8);

  xnn_simd_s32_t vzero_point_a = xnn_set1_s32(params->fp32_scalar.a_zero_point);
  xnn_simd_s32_t vzero_point_b = xnn_set1_s32(params->fp32_scalar.b_zero_point);
  xnn_simd_s32_t vzero_point_output = xnn_set1_s32(params->fp32_scalar.output_zero_point);

  xnn_simd_f32_t vscale = xnn_set1_f32(params->fp32_scalar.scale);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input_b);
  xnn_simd_s32_t vin2_low = xnn_cvt_s16_s32(xnn_low_s16(vin2));
  xnn_simd_s32_t vin2_high = xnn_cvt_s16_s32(xnn_high_s16(vin2));
  xnn_simd_s32_t vin2_low_sub = xnn_sub_s32(vin2_low,vzero_point_b);
  xnn_simd_s32_t vin2_high_sub = xnn_sub_s32(vin2_high,vzero_point_b);

  for (; batch >= 32 * sizeof(int16_t); batch -= 32 * sizeof(int16_t)) {
    xnn_simd_s16_t vin1_0 = xnn_loadu_s16(input_a);
    xnn_simd_s16_t vin1_1 = xnn_loadu_s16(input_a + 1 * xnn_simd_size_s16);
    xnn_simd_s16_t vin1_2 = xnn_loadu_s16(input_a + 2 * xnn_simd_size_s16);
    xnn_simd_s16_t vin1_3 = xnn_loadu_s16(input_a + 3 * xnn_simd_size_s16);
    input_a += 32;

    xnn_simd_s32_t vin1_low_0 = xnn_cvt_s16_s32(xnn_low_s16(vin1_0));
    xnn_simd_s32_t vin1_high_0 = xnn_cvt_s16_s32(xnn_high_s16(vin1_0));
    xnn_simd_s32_t vin1_low_sub_0 = xnn_sub_s32(vin1_low_0,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub_0 = xnn_sub_s32(vin1_high_0,vzero_point_a);

    xnn_simd_s32_t vy_s32_low_0 = xnn_mul_s32(vin1_low_sub_0,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high_0 = xnn_mul_s32(vin1_high_sub_0,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_0),vscale);
    xnn_simd_f32_t vy_f32_high_scaled_0 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_0),vscale);

    xnn_simd_s32_t vy_s32_low_add_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_0),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add_0 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_0),vzero_point_output);

    xnn_simd_s16_t vy_0 = xnn_cvt_s32_s16(vy_s32_low_add_0, vy_s32_high_add_0);
    xnn_simd_s32_t vin1_low_1 = xnn_cvt_s16_s32(xnn_low_s16(vin1_1));
    xnn_simd_s32_t vin1_high_1 = xnn_cvt_s16_s32(xnn_high_s16(vin1_1));
    xnn_simd_s32_t vin1_low_sub_1 = xnn_sub_s32(vin1_low_1,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub_1 = xnn_sub_s32(vin1_high_1,vzero_point_a);

    xnn_simd_s32_t vy_s32_low_1 = xnn_mul_s32(vin1_low_sub_1,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high_1 = xnn_mul_s32(vin1_high_sub_1,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_1),vscale);
    xnn_simd_f32_t vy_f32_high_scaled_1 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_1),vscale);

    xnn_simd_s32_t vy_s32_low_add_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_1),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add_1 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_1),vzero_point_output);

    xnn_simd_s16_t vy_1 = xnn_cvt_s32_s16(vy_s32_low_add_1, vy_s32_high_add_1);
    xnn_simd_s32_t vin1_low_2 = xnn_cvt_s16_s32(xnn_low_s16(vin1_2));
    xnn_simd_s32_t vin1_high_2 = xnn_cvt_s16_s32(xnn_high_s16(vin1_2));
    xnn_simd_s32_t vin1_low_sub_2 = xnn_sub_s32(vin1_low_2,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub_2 = xnn_sub_s32(vin1_high_2,vzero_point_a);

    xnn_simd_s32_t vy_s32_low_2 = xnn_mul_s32(vin1_low_sub_2,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high_2 = xnn_mul_s32(vin1_high_sub_2,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled_2 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_2),vscale);
    xnn_simd_f32_t vy_f32_high_scaled_2 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_2),vscale);

    xnn_simd_s32_t vy_s32_low_add_2 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_2),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add_2 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_2),vzero_point_output);

    xnn_simd_s16_t vy_2 = xnn_cvt_s32_s16(vy_s32_low_add_2, vy_s32_high_add_2);
    xnn_simd_s32_t vin1_low_3 = xnn_cvt_s16_s32(xnn_low_s16(vin1_3));
    xnn_simd_s32_t vin1_high_3 = xnn_cvt_s16_s32(xnn_high_s16(vin1_3));
    xnn_simd_s32_t vin1_low_sub_3 = xnn_sub_s32(vin1_low_3,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub_3 = xnn_sub_s32(vin1_high_3,vzero_point_a);

    xnn_simd_s32_t vy_s32_low_3 = xnn_mul_s32(vin1_low_sub_3,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high_3 = xnn_mul_s32(vin1_high_sub_3,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled_3 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low_3),vscale);
    xnn_simd_f32_t vy_f32_high_scaled_3 = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high_3),vscale);

    xnn_simd_s32_t vy_s32_low_add_3 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled_3),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add_3 = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled_3),vzero_point_output);

    xnn_simd_s16_t vy_3 = xnn_cvt_s32_s16(vy_s32_low_add_3, vy_s32_high_add_3);

    xnn_storeu_s16(output, vy_0);
    xnn_storeu_s16(output + 1 * xnn_simd_size_s16, vy_1);
    xnn_storeu_s16(output + 2 * xnn_simd_size_s16, vy_2);
    xnn_storeu_s16(output + 3 * xnn_simd_size_s16, vy_3);
    output += 32;
  }
  for (; batch >= xnn_simd_bytes_s16; batch -= xnn_simd_bytes_s16) {
    xnn_simd_s16_t vin1 = xnn_loadu_s16(input_a);
    input_a += xnn_simd_size_s16;

    xnn_simd_s32_t vin1_low = xnn_cvt_s16_s32(xnn_low_s16(vin1));
    xnn_simd_s32_t vin1_high = xnn_cvt_s16_s32(xnn_high_s16(vin1));
    xnn_simd_s32_t vin1_low_sub = xnn_sub_s32(vin1_low,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub = xnn_sub_s32(vin1_high,vzero_point_a);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low_sub,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high_sub,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low),vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high),vscale);

    xnn_simd_s32_t vy_s32_low_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled),vzero_point_output);

    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low_add, vy_s32_high_add);

    xnn_storeu_s16(output, vy);
    output += xnn_simd_size_s16;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s16_t vin1 = xnn_load_tail_s16(input_a, batch >> XNN_LOG2_SIZEOF_INT16_T);

    xnn_simd_s32_t vin1_low = xnn_cvt_s16_s32(xnn_low_s16(vin1));
    xnn_simd_s32_t vin1_high = xnn_cvt_s16_s32(xnn_high_s16(vin1));
    xnn_simd_s32_t vin1_low_sub = xnn_sub_s32(vin1_low,vzero_point_a);
    xnn_simd_s32_t vin1_high_sub = xnn_sub_s32(vin1_high,vzero_point_a);

    xnn_simd_s32_t vy_s32_low = xnn_mul_s32(vin1_low_sub,vin2_low_sub);
    xnn_simd_s32_t vy_s32_high = xnn_mul_s32(vin1_high_sub,vin2_high_sub);

    xnn_simd_f32_t vy_f32_low_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_low),vscale);
    xnn_simd_f32_t vy_f32_high_scaled = xnn_mul_f32(xnn_cvt_s32_f32(vy_s32_high),vscale);

    xnn_simd_s32_t vy_s32_low_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_low_scaled),vzero_point_output);
    xnn_simd_s32_t vy_s32_high_add = xnn_add_s32(xnn_cvt_f32_s32(vy_f32_high_scaled),vzero_point_output);
    
    xnn_simd_s16_t vy = xnn_cvt_s32_s16(vy_s32_low_add, vy_s32_high_add);

    xnn_store_tail_s16(output, vy, batch >> XNN_LOG2_SIZEOF_INT16_T);
  }
}
