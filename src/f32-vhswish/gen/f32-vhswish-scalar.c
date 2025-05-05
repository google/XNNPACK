// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vhswish/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/simd/f32-scalar.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vhswish_ukernel__scalar_u1(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_SIMD_CONST_F32(vsixth, 0x1.555556p-3f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);
  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vzero, 0.0f);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;
    xnn_simd_f32_t vacc = xnn_fmadd_f32(vx, vsixth, vhalf);
    vacc = xnn_max_f32(vacc, vzero);
    vacc = xnn_min_f32(vacc, vone);
    vacc = xnn_mul_f32(vacc, vx);
    xnn_storeu_f32(output, vacc);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);
    xnn_simd_f32_t vacc = xnn_fmadd_f32(vx, vsixth, vhalf);
    vacc = xnn_max_f32(vacc, vzero);
    vacc = xnn_min_f32(vacc, vone);
    vacc = xnn_mul_f32(vacc, vx);
    xnn_store_tail_f32(output, vacc, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vhswish_ukernel__scalar_u2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_SIMD_CONST_F32(vsixth, 0x1.555556p-3f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);
  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vzero, 0.0f);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2;

    xnn_simd_f32_t vacc0 = xnn_fmadd_f32(vx0, vsixth, vhalf);
    xnn_simd_f32_t vacc1 = xnn_fmadd_f32(vx1, vsixth, vhalf);

    vacc0 = xnn_max_f32(vacc0, vzero);
    vacc1 = xnn_max_f32(vacc1, vzero);

    vacc0 = xnn_min_f32(vacc0, vone);
    vacc1 = xnn_min_f32(vacc1, vone);

    vacc0 = xnn_mul_f32(vacc0, vx0);
    vacc1 = xnn_mul_f32(vacc1, vx1);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vacc0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vacc1);
    output += 2;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;
    xnn_simd_f32_t vacc = xnn_fmadd_f32(vx, vsixth, vhalf);
    vacc = xnn_max_f32(vacc, vzero);
    vacc = xnn_min_f32(vacc, vone);
    vacc = xnn_mul_f32(vacc, vx);
    xnn_storeu_f32(output, vacc);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);
    xnn_simd_f32_t vacc = xnn_fmadd_f32(vx, vsixth, vhalf);
    vacc = xnn_max_f32(vacc, vzero);
    vacc = xnn_min_f32(vacc, vone);
    vacc = xnn_mul_f32(vacc, vx);
    xnn_store_tail_f32(output, vacc, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vhswish_ukernel__scalar_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_SIMD_CONST_F32(vsixth, 0x1.555556p-3f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);
  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vzero, 0.0f);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 4;

    xnn_simd_f32_t vacc0 = xnn_fmadd_f32(vx0, vsixth, vhalf);
    xnn_simd_f32_t vacc1 = xnn_fmadd_f32(vx1, vsixth, vhalf);
    xnn_simd_f32_t vacc2 = xnn_fmadd_f32(vx2, vsixth, vhalf);
    xnn_simd_f32_t vacc3 = xnn_fmadd_f32(vx3, vsixth, vhalf);

    vacc0 = xnn_max_f32(vacc0, vzero);
    vacc1 = xnn_max_f32(vacc1, vzero);
    vacc2 = xnn_max_f32(vacc2, vzero);
    vacc3 = xnn_max_f32(vacc3, vzero);

    vacc0 = xnn_min_f32(vacc0, vone);
    vacc1 = xnn_min_f32(vacc1, vone);
    vacc2 = xnn_min_f32(vacc2, vone);
    vacc3 = xnn_min_f32(vacc3, vone);

    vacc0 = xnn_mul_f32(vacc0, vx0);
    vacc1 = xnn_mul_f32(vacc1, vx1);
    vacc2 = xnn_mul_f32(vacc2, vx2);
    vacc3 = xnn_mul_f32(vacc3, vx3);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vacc0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vacc1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vacc2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vacc3);
    output += 4;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;
    xnn_simd_f32_t vacc = xnn_fmadd_f32(vx, vsixth, vhalf);
    vacc = xnn_max_f32(vacc, vzero);
    vacc = xnn_min_f32(vacc, vone);
    vacc = xnn_mul_f32(vacc, vx);
    xnn_storeu_f32(output, vacc);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);
    xnn_simd_f32_t vacc = xnn_fmadd_f32(vx, vsixth, vhalf);
    vacc = xnn_max_f32(vacc, vzero);
    vacc = xnn_min_f32(vacc, vone);
    vacc = xnn_mul_f32(vacc, vx);
    xnn_store_tail_f32(output, vacc, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
