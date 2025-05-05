// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vclamp/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/simd/f32-avx512f.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vclamp_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  const xnn_simd_f32_t vmin = xnn_set1_f32(params->scalar.min);
  const xnn_simd_f32_t vmax = xnn_set1_f32(params->scalar.max);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vacc = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    vacc = xnn_max_f32(vmin, vacc);
    vacc = xnn_min_f32(vmax, vacc);

    xnn_storeu_f32(output, vacc);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vacc = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    vacc = xnn_max_f32(vmin, vacc);
    vacc = xnn_min_f32(vmax, vacc);

    xnn_store_tail_f32(output, vacc, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vclamp_ukernel__avx512f_u32(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  const xnn_simd_f32_t vmin = xnn_set1_f32(params->scalar.min);
  const xnn_simd_f32_t vmax = xnn_set1_f32(params->scalar.max);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    xnn_simd_f32_t vacc0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vacc1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 32;

    vacc0 = xnn_max_f32(vmin, vacc0);
    vacc1 = xnn_max_f32(vmin, vacc1);

    vacc0 = xnn_min_f32(vmax, vacc0);
    vacc1 = xnn_min_f32(vmax, vacc1);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vacc0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vacc1);
    output += 32;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vacc = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    vacc = xnn_max_f32(vmin, vacc);
    vacc = xnn_min_f32(vmax, vacc);

    xnn_storeu_f32(output, vacc);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vacc = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    vacc = xnn_max_f32(vmin, vacc);
    vacc = xnn_min_f32(vmax, vacc);

    xnn_store_tail_f32(output, vacc, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vclamp_ukernel__avx512f_u64(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  const xnn_simd_f32_t vmin = xnn_set1_f32(params->scalar.min);
  const xnn_simd_f32_t vmax = xnn_set1_f32(params->scalar.max);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    xnn_simd_f32_t vacc0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    xnn_simd_f32_t vacc1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vacc2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vacc3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 64;

    vacc0 = xnn_max_f32(vmin, vacc0);
    vacc1 = xnn_max_f32(vmin, vacc1);
    vacc2 = xnn_max_f32(vmin, vacc2);
    vacc3 = xnn_max_f32(vmin, vacc3);

    vacc0 = xnn_min_f32(vmax, vacc0);
    vacc1 = xnn_min_f32(vmax, vacc1);
    vacc2 = xnn_min_f32(vmax, vacc2);
    vacc3 = xnn_min_f32(vmax, vacc3);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vacc0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vacc1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vacc2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vacc3);
    output += 64;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vacc = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    vacc = xnn_max_f32(vmin, vacc);
    vacc = xnn_min_f32(vmax, vacc);

    xnn_storeu_f32(output, vacc);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vacc = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    vacc = xnn_max_f32(vmin, vacc);
    vacc = xnn_min_f32(vmax, vacc);

    xnn_store_tail_f32(output, vacc, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
