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


void xnn_f32_vsubc_minmax_ukernel__wasmsimd_x86_x8(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const v128_t vy_min = wasm_v128_load32_splat(&params->scalar.min);
  const v128_t vy_max = wasm_v128_load32_splat(&params->scalar.max);
  const v128_t vb = wasm_v128_load32_splat(b);
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const v128_t va0123 = wasm_v128_load(a);
    const v128_t va4567 = wasm_v128_load(a + 4);
    a += 8;

    v128_t vy0123 = wasm_f32x4_sub(va0123, vb);
    v128_t vy4567 = wasm_f32x4_sub(va4567, vb);


    const v128_t vltmask0123 = wasm_f32x4_lt(vy0123, vy_min);
    const v128_t vltmask4567 = wasm_f32x4_lt(vy4567, vy_min);

    const v128_t vngtmask0123 = wasm_f32x4_le(vy0123, vy_max);
    vy0123 = wasm_v128_bitselect(vy_min, vy0123, vltmask0123);
    const v128_t vngtmask4567 = wasm_f32x4_le(vy4567, vy_max);
    vy4567 = wasm_v128_bitselect(vy_min, vy4567, vltmask4567);

    vy0123 = wasm_v128_bitselect(vy0123, vy_max, vngtmask0123);
    vy4567 = wasm_v128_bitselect(vy4567, vy_max, vngtmask4567);

    wasm_v128_store(y, vy0123);
    wasm_v128_store(y + 4, vy4567);
    y += 8;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const v128_t va = wasm_v128_load(a);
    a += 4;

    v128_t vy = wasm_f32x4_sub(va, vb);

    const v128_t vltmask = wasm_f32x4_lt(vy, vy_min);
    const v128_t vngtmask = wasm_f32x4_le(vy, vy_max);
    vy = wasm_v128_bitselect(vy_min, vy, vltmask);
    vy = wasm_v128_bitselect(vy, vy_max, vngtmask);

    wasm_v128_store(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const v128_t va = wasm_v128_load(a);

    v128_t vy = wasm_f32x4_sub(va, vb);

    const v128_t vltmask = wasm_f32x4_lt(vy, vy_min);
    const v128_t vngtmask = wasm_f32x4_le(vy, vy_max);
    vy = wasm_v128_bitselect(vy_min, vy, vltmask);
    vy = wasm_v128_bitselect(vy, vy_max, vngtmask);

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
