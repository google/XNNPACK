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

#include "xnnpack/simd/s32-wasmsimd.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_s32_vmulc_ukernel__wasmsimd_u4(
    size_t batch,
    const int32_t* input1,
    const int32_t* input2,
    int32_t* output,
    const union xnn_s32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input1 != NULL);
  assert(input2 != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 4);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input2);

  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input1);
    input1 += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s32_t vin1 = (xnn_load_tail_s32(input1, batch >> XNN_LOG2_SIZEOF_INT32_T));

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_store_tail_s32(output, vy, batch >> XNN_LOG2_SIZEOF_INT32_T);
  }
}

void xnn_s32_vmulc_ukernel__wasmsimd_u8(
    size_t batch,
    const int32_t* input1,
    const int32_t* input2,
    int32_t* output,
    const union xnn_s32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input1 != NULL);
  assert(input2 != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 4);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input2);

  for (; batch >= 8 * sizeof(int32_t); batch -= 8 * sizeof(int32_t)) {

    xnn_simd_s32_t vin1_0 = (xnn_loadu_s32(input1));
    xnn_simd_s32_t vin1_1 = (xnn_loadu_s32(input1 + 1 * xnn_simd_size_s32));
    input1 += 8;

    xnn_simd_s32_t vy_0 = xnn_mul_s32(vin1_0, vin2);
    xnn_simd_s32_t vy_1 = xnn_mul_s32(vin1_1, vin2);

    xnn_storeu_s32(output, vy_0);
    xnn_storeu_s32(output + 1 * xnn_simd_size_s32, vy_1);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input1);
    input1 += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s32_t vin1 = (xnn_load_tail_s32(input1, batch >> XNN_LOG2_SIZEOF_INT32_T));

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_store_tail_s32(output, vy, batch >> XNN_LOG2_SIZEOF_INT32_T);
  }
}

void xnn_s32_vmulc_ukernel__wasmsimd_u12(
    size_t batch,
    const int32_t* input1,
    const int32_t* input2,
    int32_t* output,
    const union xnn_s32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input1 != NULL);
  assert(input2 != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 4);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input2);

  for (; batch >= 12 * sizeof(int32_t); batch -= 12 * sizeof(int32_t)) {

    xnn_simd_s32_t vin1_0 = (xnn_loadu_s32(input1));
    xnn_simd_s32_t vin1_1 = (xnn_loadu_s32(input1 + 1 * xnn_simd_size_s32));
    xnn_simd_s32_t vin1_2 = (xnn_loadu_s32(input1 + 2 * xnn_simd_size_s32));
    input1 += 12;

    xnn_simd_s32_t vy_0 = xnn_mul_s32(vin1_0, vin2);
    xnn_simd_s32_t vy_1 = xnn_mul_s32(vin1_1, vin2);
    xnn_simd_s32_t vy_2 = xnn_mul_s32(vin1_2, vin2);

    xnn_storeu_s32(output, vy_0);
    xnn_storeu_s32(output + 1 * xnn_simd_size_s32, vy_1);
    xnn_storeu_s32(output + 2 * xnn_simd_size_s32, vy_2);
    output += 12;
  }
  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input1);
    input1 += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s32_t vin1 = (xnn_load_tail_s32(input1, batch >> XNN_LOG2_SIZEOF_INT32_T));

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_store_tail_s32(output, vy, batch >> XNN_LOG2_SIZEOF_INT32_T);
  }
}

void xnn_s32_vmulc_ukernel__wasmsimd_u16(
    size_t batch,
    const int32_t* input1,
    const int32_t* input2,
    int32_t* output,
    const union xnn_s32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input1 != NULL);
  assert(input2 != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 4);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input2);

  for (; batch >= 16 * sizeof(int32_t); batch -= 16 * sizeof(int32_t)) {

    xnn_simd_s32_t vin1_0 = (xnn_loadu_s32(input1));
    xnn_simd_s32_t vin1_1 = (xnn_loadu_s32(input1 + 1 * xnn_simd_size_s32));
    xnn_simd_s32_t vin1_2 = (xnn_loadu_s32(input1 + 2 * xnn_simd_size_s32));
    xnn_simd_s32_t vin1_3 = (xnn_loadu_s32(input1 + 3 * xnn_simd_size_s32));
    input1 += 16;

    xnn_simd_s32_t vy_0 = xnn_mul_s32(vin1_0, vin2);
    xnn_simd_s32_t vy_1 = xnn_mul_s32(vin1_1, vin2);
    xnn_simd_s32_t vy_2 = xnn_mul_s32(vin1_2, vin2);
    xnn_simd_s32_t vy_3 = xnn_mul_s32(vin1_3, vin2);

    xnn_storeu_s32(output, vy_0);
    xnn_storeu_s32(output + 1 * xnn_simd_size_s32, vy_1);
    xnn_storeu_s32(output + 2 * xnn_simd_size_s32, vy_2);
    xnn_storeu_s32(output + 3 * xnn_simd_size_s32, vy_3);
    output += 16;
  }
  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input1);
    input1 += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s32_t vin1 = (xnn_load_tail_s32(input1, batch >> XNN_LOG2_SIZEOF_INT32_T));

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_store_tail_s32(output, vy, batch >> XNN_LOG2_SIZEOF_INT32_T);
  }
}
