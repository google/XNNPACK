// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-wasmsimd.c.in
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


void xnn_f32_vmax_ukernel__wasmsimd_x86_x16(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);


  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const v128_t va0123 = wasm_v128_load(a);
    const v128_t va4567 = wasm_v128_load(a + 4);
    const v128_t va89AB = wasm_v128_load(a + 8);
    const v128_t vaCDEF = wasm_v128_load(a + 12);
    a += 16;

    const v128_t vb0123 = wasm_v128_load(b);
    const v128_t vb4567 = wasm_v128_load(b + 4);
    const v128_t vb89AB = wasm_v128_load(b + 8);
    const v128_t vbCDEF = wasm_v128_load(b + 12);
    b += 16;

    const v128_t vm0123 = wasm_f32x4_le(va0123, vb0123);
    const v128_t vm4567 = wasm_f32x4_le(va4567, vb4567);
    const v128_t vm89AB = wasm_f32x4_le(va89AB, vb89AB);
    const v128_t vmCDEF = wasm_f32x4_le(vaCDEF, vbCDEF);

    v128_t vy0123 = wasm_v128_bitselect(vb0123, va0123, vm0123);
    v128_t vy4567 = wasm_v128_bitselect(vb4567, va4567, vm4567);
    v128_t vy89AB = wasm_v128_bitselect(vb89AB, va89AB, vm89AB);
    v128_t vyCDEF = wasm_v128_bitselect(vbCDEF, vaCDEF, vmCDEF);


    wasm_v128_store(y, vy0123);
    wasm_v128_store(y + 4, vy4567);
    wasm_v128_store(y + 8, vy89AB);
    wasm_v128_store(y + 12, vyCDEF);
    y += 16;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const v128_t va = wasm_v128_load(a);
    a += 4;

    const v128_t vb = wasm_v128_load(b);
    b += 4;

    const v128_t vm = wasm_f32x4_le(va, vb);
    v128_t vy = wasm_v128_bitselect(vb, va, vm);


    wasm_v128_store(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const v128_t va = wasm_v128_load(a);
    const v128_t vb = wasm_v128_load(b);

    const v128_t vm = wasm_f32x4_le(va, vb);
    v128_t vy = wasm_v128_bitselect(vb, va, vm);


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
