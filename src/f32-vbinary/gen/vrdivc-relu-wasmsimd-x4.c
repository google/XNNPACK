// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


void xnn_f32_vrdivc_relu_ukernel__wasmsimd_x4(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const v128_t vzero = wasm_i32x4_const_splat(0);
  const v128_t vb = wasm_v128_load32_splat(b);
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const v128_t va0123 = wasm_v128_load(a);
    a += 4;

    v128_t vy0123 = wasm_f32x4_div(vb, va0123);


    vy0123 = wasm_i32x4_max(vy0123, vzero);

    wasm_v128_store(y, vy0123);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const v128_t va = wasm_v128_load(a);

    v128_t vy = wasm_f32x4_div(vb, va);

    vy = wasm_i32x4_max(vy, vzero);

    if (n & (2 * sizeof(float))) {
      *((double*) y) = wasm_f64x2_extract_lane(vy, 0);
      vy = wasm_v32x4_shuffle(vy, vy, 2, 3, 2, 3);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      *y = wasm_f32x4_extract_lane(vy, 0);
    }
  }
}
