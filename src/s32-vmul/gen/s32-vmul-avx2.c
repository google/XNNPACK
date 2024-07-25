// Auto-generated file. Do not edit!
//   Template: src/s32-vmul/s32-vmul.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/simd/s32-avx2.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_s32_vmul_ukernel__avx2_u8(
    size_t batch,
    const int32_t* input_a,
    const int32_t* input_b,
    int32_t* output,
    const union xnn_s32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 8);

  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input_a);
    input_a += xnn_simd_size_s32;

    xnn_simd_s32_t vin2 = xnn_loadu_s32(input_b);
    input_b += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s32_t vin1 = xnn_load_tail_s32(input_a, batch >> XNN_LOG2_SIZEOF_INT32_T);

    xnn_simd_s32_t vin2 = xnn_load_tail_s32(input_b, batch >> XNN_LOG2_SIZEOF_INT32_T);

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_store_tail_s32(output, vy, batch >> XNN_LOG2_SIZEOF_INT32_T);
  }
}

void xnn_s32_vmul_ukernel__avx2_u16(
    size_t batch,
    const int32_t* input_a,
    const int32_t* input_b,
    int32_t* output,
    const union xnn_s32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 8);

  for (; batch >= 16 * sizeof(int32_t); batch -= 16 * sizeof(int32_t)) {
    xnn_simd_s32_t vin1_0 = xnn_loadu_s32(input_a);
    xnn_simd_s32_t vin1_1 = xnn_loadu_s32(input_a + 1 * xnn_simd_size_s32);
    input_a += 16;

    xnn_simd_s32_t vin2_0 = xnn_loadu_s32(input_b);
    xnn_simd_s32_t vin2_1 = (xnn_loadu_s32(input_b + 1 * xnn_simd_size_s32));
    input_b += 16;

    xnn_simd_s32_t vy_0 = xnn_mul_s32(vin1_0, vin2_0);
    xnn_simd_s32_t vy_1 = xnn_mul_s32(vin1_1, vin2_1);

    xnn_storeu_s32(output, vy_0);
    xnn_storeu_s32(output + 1 * xnn_simd_size_s32, vy_1);
    output += 16;
  }
  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input_a);
    input_a += xnn_simd_size_s32;

    xnn_simd_s32_t vin2 = xnn_loadu_s32(input_b);
    input_b += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s32_t vin1 = xnn_load_tail_s32(input_a, batch >> XNN_LOG2_SIZEOF_INT32_T);

    xnn_simd_s32_t vin2 = xnn_load_tail_s32(input_b, batch >> XNN_LOG2_SIZEOF_INT32_T);

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_store_tail_s32(output, vy, batch >> XNN_LOG2_SIZEOF_INT32_T);
  }
}

void xnn_s32_vmul_ukernel__avx2_u24(
    size_t batch,
    const int32_t* input_a,
    const int32_t* input_b,
    int32_t* output,
    const union xnn_s32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 8);

  for (; batch >= 24 * sizeof(int32_t); batch -= 24 * sizeof(int32_t)) {
    xnn_simd_s32_t vin1_0 = xnn_loadu_s32(input_a);
    xnn_simd_s32_t vin1_1 = xnn_loadu_s32(input_a + 1 * xnn_simd_size_s32);
    xnn_simd_s32_t vin1_2 = xnn_loadu_s32(input_a + 2 * xnn_simd_size_s32);
    input_a += 24;

    xnn_simd_s32_t vin2_0 = xnn_loadu_s32(input_b);
    xnn_simd_s32_t vin2_1 = (xnn_loadu_s32(input_b + 1 * xnn_simd_size_s32));
    xnn_simd_s32_t vin2_2 = (xnn_loadu_s32(input_b + 2 * xnn_simd_size_s32));
    input_b += 24;

    xnn_simd_s32_t vy_0 = xnn_mul_s32(vin1_0, vin2_0);
    xnn_simd_s32_t vy_1 = xnn_mul_s32(vin1_1, vin2_1);
    xnn_simd_s32_t vy_2 = xnn_mul_s32(vin1_2, vin2_2);

    xnn_storeu_s32(output, vy_0);
    xnn_storeu_s32(output + 1 * xnn_simd_size_s32, vy_1);
    xnn_storeu_s32(output + 2 * xnn_simd_size_s32, vy_2);
    output += 24;
  }
  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input_a);
    input_a += xnn_simd_size_s32;

    xnn_simd_s32_t vin2 = xnn_loadu_s32(input_b);
    input_b += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s32_t vin1 = xnn_load_tail_s32(input_a, batch >> XNN_LOG2_SIZEOF_INT32_T);

    xnn_simd_s32_t vin2 = xnn_load_tail_s32(input_b, batch >> XNN_LOG2_SIZEOF_INT32_T);

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_store_tail_s32(output, vy, batch >> XNN_LOG2_SIZEOF_INT32_T);
  }
}

void xnn_s32_vmul_ukernel__avx2_u32(
    size_t batch,
    const int32_t* input_a,
    const int32_t* input_b,
    int32_t* output,
    const union xnn_s32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 8);

  for (; batch >= 32 * sizeof(int32_t); batch -= 32 * sizeof(int32_t)) {
    xnn_simd_s32_t vin1_0 = xnn_loadu_s32(input_a);
    xnn_simd_s32_t vin1_1 = xnn_loadu_s32(input_a + 1 * xnn_simd_size_s32);
    xnn_simd_s32_t vin1_2 = xnn_loadu_s32(input_a + 2 * xnn_simd_size_s32);
    xnn_simd_s32_t vin1_3 = xnn_loadu_s32(input_a + 3 * xnn_simd_size_s32);
    input_a += 32;

    xnn_simd_s32_t vin2_0 = xnn_loadu_s32(input_b);
    xnn_simd_s32_t vin2_1 = (xnn_loadu_s32(input_b + 1 * xnn_simd_size_s32));
    xnn_simd_s32_t vin2_2 = (xnn_loadu_s32(input_b + 2 * xnn_simd_size_s32));
    xnn_simd_s32_t vin2_3 = (xnn_loadu_s32(input_b + 3 * xnn_simd_size_s32));
    input_b += 32;

    xnn_simd_s32_t vy_0 = xnn_mul_s32(vin1_0, vin2_0);
    xnn_simd_s32_t vy_1 = xnn_mul_s32(vin1_1, vin2_1);
    xnn_simd_s32_t vy_2 = xnn_mul_s32(vin1_2, vin2_2);
    xnn_simd_s32_t vy_3 = xnn_mul_s32(vin1_3, vin2_3);

    xnn_storeu_s32(output, vy_0);
    xnn_storeu_s32(output + 1 * xnn_simd_size_s32, vy_1);
    xnn_storeu_s32(output + 2 * xnn_simd_size_s32, vy_2);
    xnn_storeu_s32(output + 3 * xnn_simd_size_s32, vy_3);
    output += 32;
  }
  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input_a);
    input_a += xnn_simd_size_s32;

    xnn_simd_s32_t vin2 = xnn_loadu_s32(input_b);
    input_b += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s32_t vin1 = xnn_load_tail_s32(input_a, batch >> XNN_LOG2_SIZEOF_INT32_T);

    xnn_simd_s32_t vin2 = xnn_load_tail_s32(input_b, batch >> XNN_LOG2_SIZEOF_INT32_T);

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_store_tail_s32(output, vy, batch >> XNN_LOG2_SIZEOF_INT32_T);
  }
}
