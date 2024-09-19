// Auto-generated file. Do not edit!
//   Template: src/s32-f32-vcvt/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>

#include "xnnpack/simd/f32-wasmsimd.h"
#include "xnnpack/simd/s32-wasmsimd.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"


void xnn_s32_f32_vcvt_ukernel__wasmsimd_u4(
    size_t batch,
    const int32_t* input,
    float* output,
    const struct xnn_s32_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);
  assert(xnn_simd_size_s32 == 4);

  const xnn_simd_s32_t sub = xnn_set1_s32(params->scalar.zero_point);


  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    const xnn_simd_s32_t vx = xnn_loadu_s32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_cvt_f32_s32(xnn_sub_s32(vx, sub));

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if (batch != 0) {
    const xnn_simd_s32_t vx =
        xnn_load_tail_s32(input, batch >> XNN_LOG2_SIZEOF_INT32_T);

    const xnn_simd_f32_t vy = xnn_cvt_f32_s32(xnn_sub_s32(vx, sub));

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_s32_f32_vcvt_ukernel__wasmsimd_u8(
    size_t batch,
    const int32_t* input,
    float* output,
    const struct xnn_s32_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);
  assert(xnn_simd_size_s32 == 4);

  const xnn_simd_s32_t sub = xnn_set1_s32(params->scalar.zero_point);

  for (; batch >= 8 * sizeof(int32_t); batch -= 8 * sizeof(int32_t)) {
    const xnn_simd_s32_t vx0 = xnn_loadu_s32(input);
    const xnn_simd_s32_t vx1 = xnn_loadu_s32(input + 1 * xnn_simd_size_s32);
    input += 2 * xnn_simd_size_s32;

    const xnn_simd_f32_t vy0 = xnn_cvt_f32_s32(xnn_sub_s32(vx0, sub));
    const xnn_simd_f32_t vy1 = xnn_cvt_f32_s32(xnn_sub_s32(vx1, sub));

    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    output += 2 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    const xnn_simd_s32_t vx = xnn_loadu_s32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_cvt_f32_s32(xnn_sub_s32(vx, sub));

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if (batch != 0) {
    const xnn_simd_s32_t vx =
        xnn_load_tail_s32(input, batch >> XNN_LOG2_SIZEOF_INT32_T);

    const xnn_simd_f32_t vy = xnn_cvt_f32_s32(xnn_sub_s32(vx, sub));

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_s32_f32_vcvt_ukernel__wasmsimd_u12(
    size_t batch,
    const int32_t* input,
    float* output,
    const struct xnn_s32_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);
  assert(xnn_simd_size_s32 == 4);

  const xnn_simd_s32_t sub = xnn_set1_s32(params->scalar.zero_point);

  for (; batch >= 12 * sizeof(int32_t); batch -= 12 * sizeof(int32_t)) {
    const xnn_simd_s32_t vx0 = xnn_loadu_s32(input);
    const xnn_simd_s32_t vx1 = xnn_loadu_s32(input + 1 * xnn_simd_size_s32);
    const xnn_simd_s32_t vx2 = xnn_loadu_s32(input + 2 * xnn_simd_size_s32);
    input += 3 * xnn_simd_size_s32;

    const xnn_simd_f32_t vy0 = xnn_cvt_f32_s32(xnn_sub_s32(vx0, sub));
    const xnn_simd_f32_t vy1 = xnn_cvt_f32_s32(xnn_sub_s32(vx1, sub));
    const xnn_simd_f32_t vy2 = xnn_cvt_f32_s32(xnn_sub_s32(vx2, sub));

    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy2);
    output += 3 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    const xnn_simd_s32_t vx = xnn_loadu_s32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_cvt_f32_s32(xnn_sub_s32(vx, sub));

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if (batch != 0) {
    const xnn_simd_s32_t vx =
        xnn_load_tail_s32(input, batch >> XNN_LOG2_SIZEOF_INT32_T);

    const xnn_simd_f32_t vy = xnn_cvt_f32_s32(xnn_sub_s32(vx, sub));

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_s32_f32_vcvt_ukernel__wasmsimd_u16(
    size_t batch,
    const int32_t* input,
    float* output,
    const struct xnn_s32_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);
  assert(xnn_simd_size_s32 == 4);

  const xnn_simd_s32_t sub = xnn_set1_s32(params->scalar.zero_point);

  for (; batch >= 16 * sizeof(int32_t); batch -= 16 * sizeof(int32_t)) {
    const xnn_simd_s32_t vx0 = xnn_loadu_s32(input);
    const xnn_simd_s32_t vx1 = xnn_loadu_s32(input + 1 * xnn_simd_size_s32);
    const xnn_simd_s32_t vx2 = xnn_loadu_s32(input + 2 * xnn_simd_size_s32);
    const xnn_simd_s32_t vx3 = xnn_loadu_s32(input + 3 * xnn_simd_size_s32);
    input += 4 * xnn_simd_size_s32;

    const xnn_simd_f32_t vy0 = xnn_cvt_f32_s32(xnn_sub_s32(vx0, sub));
    const xnn_simd_f32_t vy1 = xnn_cvt_f32_s32(xnn_sub_s32(vx1, sub));
    const xnn_simd_f32_t vy2 = xnn_cvt_f32_s32(xnn_sub_s32(vx2, sub));
    const xnn_simd_f32_t vy3 = xnn_cvt_f32_s32(xnn_sub_s32(vx3, sub));

    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy3);
    output += 4 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    const xnn_simd_s32_t vx = xnn_loadu_s32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_cvt_f32_s32(xnn_sub_s32(vx, sub));

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if (batch != 0) {
    const xnn_simd_s32_t vx =
        xnn_load_tail_s32(input, batch >> XNN_LOG2_SIZEOF_INT32_T);

    const xnn_simd_f32_t vy = xnn_cvt_f32_s32(xnn_sub_s32(vx, sub));

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
