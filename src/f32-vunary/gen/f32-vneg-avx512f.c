// Auto-generated file. Do not edit!
//   Template: src/f32-vunary/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>

#include "xnnpack/simd/f32-avx512f.h"

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"
#include "xnnpack/microparams.h"


void xnn_f32_vneg_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 16);


  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_neg_f32(vx);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx =
        xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    const xnn_simd_f32_t vy = xnn_neg_f32(vx);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vneg_ukernel__avx512f_u32(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 16);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2 * xnn_simd_size_f32;

    const xnn_simd_f32_t vy0 = xnn_neg_f32(vx0);
    const xnn_simd_f32_t vy1 = xnn_neg_f32(vx1);

    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    output += 2 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_neg_f32(vx);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx =
        xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    const xnn_simd_f32_t vy = xnn_neg_f32(vx);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vneg_ukernel__avx512f_u48(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 16);

  for (; batch >= 48 * sizeof(float); batch -= 48 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    input += 3 * xnn_simd_size_f32;

    const xnn_simd_f32_t vy0 = xnn_neg_f32(vx0);
    const xnn_simd_f32_t vy1 = xnn_neg_f32(vx1);
    const xnn_simd_f32_t vy2 = xnn_neg_f32(vx2);

    xnn_storeu_f32(output, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy2);
    output += 3 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_neg_f32(vx);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx =
        xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    const xnn_simd_f32_t vy = xnn_neg_f32(vx);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
