// Auto-generated file. Do not edit!
//   Template: src/s32-vmul/s32-vmulc.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/simd/s32-scalar.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_s32_vmulc_minmax_ukernel__scalar_u1(
    size_t batch,
    const int32_t* input1,
    const int32_t* input2,
    int32_t* output,
    const union xnn_s32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input1 != NULL);
  assert(input2 != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 1);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input2);
  xnn_simd_s32_t voutput_min = xnn_set1_s32(&params->scalar.min);
  xnn_simd_s32_t voutput_max = xnn_set1_s32(&params->scalar.max);

  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input1);
    input1 += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    vy = xnn_max_s32(vy, voutput_min);
    vy = xnn_min_s32(vy, voutput_max);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
}

void xnn_s32_vmulc_minmax_ukernel__scalar_u2(
    size_t batch,
    const int32_t* input1,
    const int32_t* input2,
    int32_t* output,
    const union xnn_s32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input1 != NULL);
  assert(input2 != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 1);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input2);
  xnn_simd_s32_t voutput_min = xnn_set1_s32(&params->scalar.min);
  xnn_simd_s32_t voutput_max = xnn_set1_s32(&params->scalar.max);

  for (; batch >= 2 * sizeof(int32_t); batch -= 2 * sizeof(int32_t)) {

    xnn_simd_s32_t vin1_0 = (xnn_loadu_s32(input1));
    xnn_simd_s32_t vin1_1 = (xnn_loadu_s32(input1 + 1 * xnn_simd_size_s32));
    input1 += 2;

    xnn_simd_s32_t vy_0 = xnn_mul_s32(vin1_0, vin2);
    vy_0 = xnn_max_s32(vy_0, voutput_min);
    vy_0 = xnn_min_s32(vy_0, voutput_max);
    xnn_simd_s32_t vy_1 = xnn_mul_s32(vin1_1, vin2);
    vy_1 = xnn_max_s32(vy_1, voutput_min);
    vy_1 = xnn_min_s32(vy_1, voutput_max);

    xnn_storeu_s32(output, vy_0);
    xnn_storeu_s32(output + 1 * xnn_simd_size_s32, vy_1);
    output += 2;
  }
  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input1);
    input1 += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    vy = xnn_max_s32(vy, voutput_min);
    vy = xnn_min_s32(vy, voutput_max);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
}

void xnn_s32_vmulc_minmax_ukernel__scalar_u4(
    size_t batch,
    const int32_t* input1,
    const int32_t* input2,
    int32_t* output,
    const union xnn_s32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input1 != NULL);
  assert(input2 != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 1);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input2);
  xnn_simd_s32_t voutput_min = xnn_set1_s32(&params->scalar.min);
  xnn_simd_s32_t voutput_max = xnn_set1_s32(&params->scalar.max);

  for (; batch >= 4 * sizeof(int32_t); batch -= 4 * sizeof(int32_t)) {

    xnn_simd_s32_t vin1_0 = (xnn_loadu_s32(input1));
    xnn_simd_s32_t vin1_1 = (xnn_loadu_s32(input1 + 1 * xnn_simd_size_s32));
    xnn_simd_s32_t vin1_2 = (xnn_loadu_s32(input1 + 2 * xnn_simd_size_s32));
    xnn_simd_s32_t vin1_3 = (xnn_loadu_s32(input1 + 3 * xnn_simd_size_s32));
    input1 += 4;

    xnn_simd_s32_t vy_0 = xnn_mul_s32(vin1_0, vin2);
    vy_0 = xnn_max_s32(vy_0, voutput_min);
    vy_0 = xnn_min_s32(vy_0, voutput_max);
    xnn_simd_s32_t vy_1 = xnn_mul_s32(vin1_1, vin2);
    vy_1 = xnn_max_s32(vy_1, voutput_min);
    vy_1 = xnn_min_s32(vy_1, voutput_max);
    xnn_simd_s32_t vy_2 = xnn_mul_s32(vin1_2, vin2);
    vy_2 = xnn_max_s32(vy_2, voutput_min);
    vy_2 = xnn_min_s32(vy_2, voutput_max);
    xnn_simd_s32_t vy_3 = xnn_mul_s32(vin1_3, vin2);
    vy_3 = xnn_max_s32(vy_3, voutput_min);
    vy_3 = xnn_min_s32(vy_3, voutput_max);

    xnn_storeu_s32(output, vy_0);
    xnn_storeu_s32(output + 1 * xnn_simd_size_s32, vy_1);
    xnn_storeu_s32(output + 2 * xnn_simd_size_s32, vy_2);
    xnn_storeu_s32(output + 3 * xnn_simd_size_s32, vy_3);
    output += 4;
  }
  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input1);
    input1 += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    vy = xnn_max_s32(vy, voutput_min);
    vy = xnn_min_s32(vy, voutput_max);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
}

void xnn_s32_vmulc_minmax_ukernel__scalar_u8(
    size_t batch,
    const int32_t* input1,
    const int32_t* input2,
    int32_t* output,
    const union xnn_s32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input1 != NULL);
  assert(input2 != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 1);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input2);
  xnn_simd_s32_t voutput_min = xnn_set1_s32(&params->scalar.min);
  xnn_simd_s32_t voutput_max = xnn_set1_s32(&params->scalar.max);

  for (; batch >= 8 * sizeof(int32_t); batch -= 8 * sizeof(int32_t)) {

    xnn_simd_s32_t vin1_0 = (xnn_loadu_s32(input1));
    xnn_simd_s32_t vin1_1 = (xnn_loadu_s32(input1 + 1 * xnn_simd_size_s32));
    xnn_simd_s32_t vin1_2 = (xnn_loadu_s32(input1 + 2 * xnn_simd_size_s32));
    xnn_simd_s32_t vin1_3 = (xnn_loadu_s32(input1 + 3 * xnn_simd_size_s32));
    xnn_simd_s32_t vin1_4 = (xnn_loadu_s32(input1 + 4 * xnn_simd_size_s32));
    xnn_simd_s32_t vin1_5 = (xnn_loadu_s32(input1 + 5 * xnn_simd_size_s32));
    xnn_simd_s32_t vin1_6 = (xnn_loadu_s32(input1 + 6 * xnn_simd_size_s32));
    xnn_simd_s32_t vin1_7 = (xnn_loadu_s32(input1 + 7 * xnn_simd_size_s32));
    input1 += 8;

    xnn_simd_s32_t vy_0 = xnn_mul_s32(vin1_0, vin2);
    vy_0 = xnn_max_s32(vy_0, voutput_min);
    vy_0 = xnn_min_s32(vy_0, voutput_max);
    xnn_simd_s32_t vy_1 = xnn_mul_s32(vin1_1, vin2);
    vy_1 = xnn_max_s32(vy_1, voutput_min);
    vy_1 = xnn_min_s32(vy_1, voutput_max);
    xnn_simd_s32_t vy_2 = xnn_mul_s32(vin1_2, vin2);
    vy_2 = xnn_max_s32(vy_2, voutput_min);
    vy_2 = xnn_min_s32(vy_2, voutput_max);
    xnn_simd_s32_t vy_3 = xnn_mul_s32(vin1_3, vin2);
    vy_3 = xnn_max_s32(vy_3, voutput_min);
    vy_3 = xnn_min_s32(vy_3, voutput_max);
    xnn_simd_s32_t vy_4 = xnn_mul_s32(vin1_4, vin2);
    vy_4 = xnn_max_s32(vy_4, voutput_min);
    vy_4 = xnn_min_s32(vy_4, voutput_max);
    xnn_simd_s32_t vy_5 = xnn_mul_s32(vin1_5, vin2);
    vy_5 = xnn_max_s32(vy_5, voutput_min);
    vy_5 = xnn_min_s32(vy_5, voutput_max);
    xnn_simd_s32_t vy_6 = xnn_mul_s32(vin1_6, vin2);
    vy_6 = xnn_max_s32(vy_6, voutput_min);
    vy_6 = xnn_min_s32(vy_6, voutput_max);
    xnn_simd_s32_t vy_7 = xnn_mul_s32(vin1_7, vin2);
    vy_7 = xnn_max_s32(vy_7, voutput_min);
    vy_7 = xnn_min_s32(vy_7, voutput_max);

    xnn_storeu_s32(output, vy_0);
    xnn_storeu_s32(output + 1 * xnn_simd_size_s32, vy_1);
    xnn_storeu_s32(output + 2 * xnn_simd_size_s32, vy_2);
    xnn_storeu_s32(output + 3 * xnn_simd_size_s32, vy_3);
    xnn_storeu_s32(output + 4 * xnn_simd_size_s32, vy_4);
    xnn_storeu_s32(output + 5 * xnn_simd_size_s32, vy_5);
    xnn_storeu_s32(output + 6 * xnn_simd_size_s32, vy_6);
    xnn_storeu_s32(output + 7 * xnn_simd_size_s32, vy_7);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input1);
    input1 += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    vy = xnn_max_s32(vy, voutput_min);
    vy = xnn_min_s32(vy, voutput_max);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
}
