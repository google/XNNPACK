// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vhswish/f16-f32acc.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/simd/f32-wasmrelaxedsimd.h"
#include "src/xnnpack/vunary.h"

#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"


static XNN_INLINE xnn_simd_f32_t load_f16_f32(const xnn_float16* input) {
  return xnn_cvt_f32_f16(xnn_loadu_f16(input));
}

static XNN_INLINE xnn_simd_f32_t load_tail_f16_f32(
    const xnn_float16* input, size_t num_elements)
{
  return xnn_cvt_f32_f16(xnn_load_tail_f16(input, num_elements));
}

static XNN_INLINE void store_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value)
{
  xnn_store_tail_f16(output, xnn_cvt_f16_f32(value), xnn_simd_size_f32);
}

static XNN_INLINE void store_tail_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value, size_t num_elements)
{
  xnn_store_tail_f16(output, xnn_cvt_f16_f32(value), num_elements);
}

static XNN_INLINE xnn_simd_f32_t hswish_f32(xnn_simd_f32_t vx) {
  XNN_SIMD_CONST_F32(vsixth, 0x1.555556p-3f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);
  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vzero, 0.0f);

  xnn_simd_f32_t vacc = xnn_fmadd_f32(vx, vsixth, vhalf);
  vacc = xnn_max_f32(vacc, vzero);
  vacc = xnn_min_f32(vacc, vone);
  return xnn_mul_f32(vacc, vx);
}

void xnn_f16_f32acc_vhswish_ukernel__wasmrelaxedsimd_u8(
    size_t batch,
    const xnn_float16* restrict input,
    xnn_float16* restrict output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 8 * sizeof(xnn_float16);
       batch -= 8 * sizeof(xnn_float16)) {
    const xnn_simd_f32_t vx0 = load_f16_f32(input + 0);
    const xnn_simd_f32_t vx1 = load_f16_f32(input + 4);
    input += 8;
    store_f32_f16(output + 0, hswish_f32(vx0));
    store_f32_f16(output + 4, hswish_f32(vx1));
    output += 8;
  }

  for (; batch >= 4 * sizeof(xnn_float16);
       batch -= 4 * sizeof(xnn_float16)) {
    const xnn_simd_f32_t vx = load_f16_f32(input);
    input += 4;
    store_f32_f16(output, hswish_f32(vx));
    output += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    const size_t num_elements = batch / sizeof(xnn_float16);
    const xnn_simd_f32_t vx = load_tail_f16_f32(input, num_elements);
    store_tail_f32_f16(output, hswish_f32(vx), num_elements);
  }
}
