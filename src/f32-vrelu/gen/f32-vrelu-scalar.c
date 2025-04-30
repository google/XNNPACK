// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vrelu/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/simd/f32-scalar.h"

#include "src/xnnpack/vunary.h"
#include "src/xnnpack/common.h"


void xnn_f32_vrelu_ukernel__scalar_u1(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_SIMD_CONST_F32(vzero, 0.0f);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vacc = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    vacc = xnn_max_f32(vacc, vzero);

    xnn_storeu_f32(output, vacc);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vacc = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    vacc = xnn_max_f32(vacc, vzero);

    xnn_store_tail_f32(output, vacc, batch >> XNN_LOG2_SIZEOF_FLOAT);      output += xnn_simd_size_f32;
  }
}

void xnn_f32_vrelu_ukernel__scalar_u2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_SIMD_CONST_F32(vzero, 0.0f);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    xnn_simd_f32_t vacc0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vacc1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2;

    vacc0 = xnn_max_f32(vacc0, vzero);
    vacc1 = xnn_max_f32(vacc1, vzero);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vacc0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vacc1);
    output += 2;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vacc = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    vacc = xnn_max_f32(vacc, vzero);

    xnn_storeu_f32(output, vacc);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vacc = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    vacc = xnn_max_f32(vacc, vzero);

    xnn_store_tail_f32(output, vacc, batch >> XNN_LOG2_SIZEOF_FLOAT);      output += xnn_simd_size_f32;
  }
}

void xnn_f32_vrelu_ukernel__scalar_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_SIMD_CONST_F32(vzero, 0.0f);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    xnn_simd_f32_t vacc0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vacc1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vacc2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vacc3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 4;

    vacc0 = xnn_max_f32(vacc0, vzero);
    vacc1 = xnn_max_f32(vacc1, vzero);
    vacc2 = xnn_max_f32(vacc2, vzero);
    vacc3 = xnn_max_f32(vacc3, vzero);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vacc0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vacc1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vacc2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vacc3);
    output += 4;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vacc = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    vacc = xnn_max_f32(vacc, vzero);

    xnn_storeu_f32(output, vacc);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vacc = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    vacc = xnn_max_f32(vacc, vzero);

    xnn_store_tail_f32(output, vacc, batch >> XNN_LOG2_SIZEOF_FLOAT);      output += xnn_simd_size_f32;
  }
}
