// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/simd/f32-avx512f.h"

static XNN_INLINE float load_tail_reduce_add_f32(xnn_simd_f32_t acc,
                                                 const float* input,
                                                 size_t num_elements) {
  assert(num_elements < xnn_simd_size_f32);
  if (num_elements != 0) {
    xnn_simd_f32_t tail = xnn_load_tail_safe_f32(input, num_elements);
    acc = xnn_add_f32(acc, tail);
  }
  return xnn_reduce_add_f32(acc);
}

void xnn_f32_rsum_ukernel__avx512f_u48_acc3(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_scale_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  xnn_simd_f32_t vacc1 = xnn_zero_f32();
  xnn_simd_f32_t vacc2 = xnn_zero_f32();
  for (; batch >= 48 * sizeof(float); batch -= 48 * sizeof(float)) {
    const xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    const xnn_simd_f32_t vt2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    input += 48;

    vacc0 = xnn_add_f32(vacc0, vt0);
    vacc1 = xnn_add_f32(vacc1, vt1);
    vacc2 = xnn_add_f32(vacc2, vt2);
  }
  if (batch >= 64) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 16;
    batch -= 64;
    vacc0 = xnn_add_f32(vacc0, vt);
  }
  if (batch >= 64) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 16;
    batch -= 64;
    vacc1 = xnn_add_f32(vacc1, vt);
  }
  vacc0 = xnn_add_f32(vacc0, vacc2);
  vacc0 = xnn_add_f32(vacc0, vacc1);
  const float vscale = params->scalar.scale;
  float vresult = load_tail_reduce_add_f32(vacc0, input, batch >> XNN_LOG2_SIZEOF_FLOAT);
  *output += vresult * vscale;
}
