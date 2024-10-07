// Auto-generated file. Do not edit!
//   Template: src/u32-f32-vcvt/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>

#include "xnnpack/simd/f32-avx2.h"
#include "xnnpack/simd/u32-avx2.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"


void xnn_u32_f32_vcvt_ukernel__avx2_u8(
    size_t batch,
    const uint32_t* input,
    float* output,
    const struct xnn_u32_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);
  assert(xnn_simd_size_u32 == 8);

  const xnn_simd_u32_t zero_point = xnn_set1_u32(params->scalar.zero_point);


  for (; batch >= xnn_simd_bytes_u32; batch -= xnn_simd_bytes_u32) {
    const xnn_simd_u32_t vx = xnn_loadu_u32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_subw_f32_u32(vx, zero_point);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if (batch != 0) {
    const xnn_simd_u32_t vx =
        xnn_load_tail_u32(input, batch >> XNN_LOG2_SIZEOF_UINT32_T);

    const xnn_simd_f32_t vy = xnn_subw_f32_u32(vx, zero_point);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_u32_f32_vcvt_ukernel__avx2_u16(
    size_t batch,
    const uint32_t* input,
    float* output,
    const struct xnn_u32_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);
  assert(xnn_simd_size_u32 == 8);

  const xnn_simd_u32_t zero_point = xnn_set1_u32(params->scalar.zero_point);

  for (; batch >= 16 * sizeof(int32_t); batch -= 16 * sizeof(uint32_t)) {
    const xnn_simd_u32_t vx0 = xnn_loadu_u32(input);
    const xnn_simd_u32_t vx1 = xnn_loadu_u32(input + 1 * xnn_simd_size_u32);
    input += 2 * xnn_simd_size_u32;

    const xnn_simd_f32_t vy0 = xnn_subw_f32_u32(vx0, zero_point);
    const xnn_simd_f32_t vy1 = xnn_subw_f32_u32(vx1, zero_point);

    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    output += 2 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_u32; batch -= xnn_simd_bytes_u32) {
    const xnn_simd_u32_t vx = xnn_loadu_u32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_subw_f32_u32(vx, zero_point);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if (batch != 0) {
    const xnn_simd_u32_t vx =
        xnn_load_tail_u32(input, batch >> XNN_LOG2_SIZEOF_UINT32_T);

    const xnn_simd_f32_t vy = xnn_subw_f32_u32(vx, zero_point);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_u32_f32_vcvt_ukernel__avx2_u24(
    size_t batch,
    const uint32_t* input,
    float* output,
    const struct xnn_u32_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);
  assert(xnn_simd_size_u32 == 8);

  const xnn_simd_u32_t zero_point = xnn_set1_u32(params->scalar.zero_point);

  for (; batch >= 24 * sizeof(int32_t); batch -= 24 * sizeof(uint32_t)) {
    const xnn_simd_u32_t vx0 = xnn_loadu_u32(input);
    const xnn_simd_u32_t vx1 = xnn_loadu_u32(input + 1 * xnn_simd_size_u32);
    const xnn_simd_u32_t vx2 = xnn_loadu_u32(input + 2 * xnn_simd_size_u32);
    input += 3 * xnn_simd_size_u32;

    const xnn_simd_f32_t vy0 = xnn_subw_f32_u32(vx0, zero_point);
    const xnn_simd_f32_t vy1 = xnn_subw_f32_u32(vx1, zero_point);
    const xnn_simd_f32_t vy2 = xnn_subw_f32_u32(vx2, zero_point);

    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy2);
    output += 3 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_u32; batch -= xnn_simd_bytes_u32) {
    const xnn_simd_u32_t vx = xnn_loadu_u32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_subw_f32_u32(vx, zero_point);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if (batch != 0) {
    const xnn_simd_u32_t vx =
        xnn_load_tail_u32(input, batch >> XNN_LOG2_SIZEOF_UINT32_T);

    const xnn_simd_f32_t vy = xnn_subw_f32_u32(vx, zero_point);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_u32_f32_vcvt_ukernel__avx2_u32(
    size_t batch,
    const uint32_t* input,
    float* output,
    const struct xnn_u32_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);
  assert(xnn_simd_size_u32 == 8);

  const xnn_simd_u32_t zero_point = xnn_set1_u32(params->scalar.zero_point);

  for (; batch >= 32 * sizeof(int32_t); batch -= 32 * sizeof(uint32_t)) {
    const xnn_simd_u32_t vx0 = xnn_loadu_u32(input);
    const xnn_simd_u32_t vx1 = xnn_loadu_u32(input + 1 * xnn_simd_size_u32);
    const xnn_simd_u32_t vx2 = xnn_loadu_u32(input + 2 * xnn_simd_size_u32);
    const xnn_simd_u32_t vx3 = xnn_loadu_u32(input + 3 * xnn_simd_size_u32);
    input += 4 * xnn_simd_size_u32;

    const xnn_simd_f32_t vy0 = xnn_subw_f32_u32(vx0, zero_point);
    const xnn_simd_f32_t vy1 = xnn_subw_f32_u32(vx1, zero_point);
    const xnn_simd_f32_t vy2 = xnn_subw_f32_u32(vx2, zero_point);
    const xnn_simd_f32_t vy3 = xnn_subw_f32_u32(vx3, zero_point);

    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy3);
    output += 4 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_u32; batch -= xnn_simd_bytes_u32) {
    const xnn_simd_u32_t vx = xnn_loadu_u32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_subw_f32_u32(vx, zero_point);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if (batch != 0) {
    const xnn_simd_u32_t vx =
        xnn_load_tail_u32(input, batch >> XNN_LOG2_SIZEOF_UINT32_T);

    const xnn_simd_f32_t vy = xnn_subw_f32_u32(vx, zero_point);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
