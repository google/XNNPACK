// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rsum2/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2023-2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/simd/f32-sse2.h"

static XNN_INLINE float load_tail_reduce_add_squared_f32(xnn_simd_f32_t acc,
                                                         const float* input,
                                                         size_t num_elements) {
  assert(num_elements < xnn_simd_size_f32);
  for (; num_elements > 0; num_elements -= 1) {
    __m128 vt = _mm_load_ss(input);
    input += 1;
    vt = _mm_mul_ps(vt, vt);
    acc = _mm_add_ss(acc, vt);
  }
  return xnn_reduce_add_f32(acc);
}

void xnn_f32_rsum2_ukernel__sse2_u4(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    input += 4;

    vacc0 = xnn_fmadd_f32(vt0, vt0, vacc0);
  }
  const float vscale = params->scalar.scale;
  float vresult = load_tail_reduce_add_squared_f32(
      vacc0, input, batch >> XNN_LOG2_SIZEOF_FLOAT);
  *output += vresult * vscale;
}

void xnn_f32_rsum2_ukernel__sse2_u8_acc2(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  xnn_simd_f32_t vacc1 = xnn_zero_f32();
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 8;

    vacc0 = xnn_fmadd_f32(vt0, vt0, vacc0);
    vacc1 = xnn_fmadd_f32(vt1, vt1, vacc1);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc0 = xnn_fmadd_f32(vt, vt, vacc0);
  }
  vacc0 = xnn_add_f32(vacc0, vacc1);
  const float vscale = params->scalar.scale;
  float vresult = load_tail_reduce_add_squared_f32(
      vacc0, input, batch >> XNN_LOG2_SIZEOF_FLOAT);
  *output += vresult * vscale;
}

void xnn_f32_rsum2_ukernel__sse2_u8(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 8;

    vacc0 = xnn_fmadd_f32(vt0, vt0, vacc0);
    vacc0 = xnn_fmadd_f32(vt1, vt1, vacc0);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc0 = xnn_fmadd_f32(vt, vt, vacc0);
  }
  const float vscale = params->scalar.scale;
  float vresult = load_tail_reduce_add_squared_f32(
      vacc0, input, batch >> XNN_LOG2_SIZEOF_FLOAT);
  *output += vresult * vscale;
}

void xnn_f32_rsum2_ukernel__sse2_u12_acc3(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  xnn_simd_f32_t vacc1 = xnn_zero_f32();
  xnn_simd_f32_t vacc2 = xnn_zero_f32();
  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vt2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    input += 12;

    vacc0 = xnn_fmadd_f32(vt0, vt0, vacc0);
    vacc1 = xnn_fmadd_f32(vt1, vt1, vacc1);
    vacc2 = xnn_fmadd_f32(vt2, vt2, vacc2);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc0 = xnn_fmadd_f32(vt, vt, vacc0);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc1 = xnn_fmadd_f32(vt, vt, vacc1);
  }
  vacc0 = xnn_add_f32(vacc0, vacc2);
  vacc0 = xnn_add_f32(vacc0, vacc1);
  const float vscale = params->scalar.scale;
  float vresult = load_tail_reduce_add_squared_f32(
      vacc0, input, batch >> XNN_LOG2_SIZEOF_FLOAT);
  *output += vresult * vscale;
}

void xnn_f32_rsum2_ukernel__sse2_u12(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vt2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    input += 12;

    vacc0 = xnn_fmadd_f32(vt0, vt0, vacc0);
    vacc0 = xnn_fmadd_f32(vt1, vt1, vacc0);
    vacc0 = xnn_fmadd_f32(vt2, vt2, vacc0);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc0 = xnn_fmadd_f32(vt, vt, vacc0);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc0 = xnn_fmadd_f32(vt, vt, vacc0);
  }
  const float vscale = params->scalar.scale;
  float vresult = load_tail_reduce_add_squared_f32(
      vacc0, input, batch >> XNN_LOG2_SIZEOF_FLOAT);
  *output += vresult * vscale;
}

void xnn_f32_rsum2_ukernel__sse2_u16_acc4(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  xnn_simd_f32_t vacc1 = xnn_zero_f32();
  xnn_simd_f32_t vacc2 = xnn_zero_f32();
  xnn_simd_f32_t vacc3 = xnn_zero_f32();
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vt2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vt3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 16;

    vacc0 = xnn_fmadd_f32(vt0, vt0, vacc0);
    vacc1 = xnn_fmadd_f32(vt1, vt1, vacc1);
    vacc2 = xnn_fmadd_f32(vt2, vt2, vacc2);
    vacc3 = xnn_fmadd_f32(vt3, vt3, vacc3);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc0 = xnn_fmadd_f32(vt, vt, vacc0);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc1 = xnn_fmadd_f32(vt, vt, vacc1);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc2 = xnn_fmadd_f32(vt, vt, vacc2);
  }
  vacc0 = xnn_add_f32(vacc0, vacc2);
  vacc1 = xnn_add_f32(vacc1, vacc3);
  vacc0 = xnn_add_f32(vacc0, vacc1);
  const float vscale = params->scalar.scale;
  float vresult = load_tail_reduce_add_squared_f32(
      vacc0, input, batch >> XNN_LOG2_SIZEOF_FLOAT);
  *output += vresult * vscale;
}

void xnn_f32_rsum2_ukernel__sse2_u16_acc2(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  xnn_simd_f32_t vacc1 = xnn_zero_f32();
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vt2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vt3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 16;

    vacc0 = xnn_fmadd_f32(vt0, vt0, vacc0);
    vacc1 = xnn_fmadd_f32(vt1, vt1, vacc1);
    vacc0 = xnn_fmadd_f32(vt2, vt2, vacc0);
    vacc1 = xnn_fmadd_f32(vt3, vt3, vacc1);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc0 = xnn_fmadd_f32(vt, vt, vacc0);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc1 = xnn_fmadd_f32(vt, vt, vacc1);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc0 = xnn_fmadd_f32(vt, vt, vacc0);
  }
  vacc0 = xnn_add_f32(vacc0, vacc1);
  const float vscale = params->scalar.scale;
  float vresult = load_tail_reduce_add_squared_f32(
      vacc0, input, batch >> XNN_LOG2_SIZEOF_FLOAT);
  *output += vresult * vscale;
}

void xnn_f32_rsum2_ukernel__sse2_u16(
    size_t batch, const float* input, float* output,
    const struct xnn_f32_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vt2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vt3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 16;

    vacc0 = xnn_fmadd_f32(vt0, vt0, vacc0);
    vacc0 = xnn_fmadd_f32(vt1, vt1, vacc0);
    vacc0 = xnn_fmadd_f32(vt2, vt2, vacc0);
    vacc0 = xnn_fmadd_f32(vt3, vt3, vacc0);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc0 = xnn_fmadd_f32(vt, vt, vacc0);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc0 = xnn_fmadd_f32(vt, vt, vacc0);
  }
  if (batch >= 16) {
    xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;
    batch -= 16;
    vacc0 = xnn_fmadd_f32(vt, vt, vacc0);
  }
  const float vscale = params->scalar.scale;
  float vresult = load_tail_reduce_add_squared_f32(
      vacc0, input, batch >> XNN_LOG2_SIZEOF_FLOAT);
  *output += vresult * vscale;
}
