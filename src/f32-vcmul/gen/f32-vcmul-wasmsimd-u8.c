// Auto-generated file. Do not edit!
//   Template: src/f32-vcmul/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vcmul_ukernel__wasmsimd_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float* ar = input_a;
  const float* ai = (const float*) ((uintptr_t) input_a + batch);
  const float* br = input_b;
  const float* bi = (const float*) ((uintptr_t) input_b + batch);
  float* or = output;
  float* oi = (float*) ((uintptr_t) output + batch);
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const v128_t va0r = wasm_v128_load(ar);
    const v128_t va0i = wasm_v128_load(ai);
    const v128_t vb0r = wasm_v128_load(br);
    const v128_t vb0i = wasm_v128_load(bi);
    const v128_t va1r = wasm_v128_load(ar + 4);
    const v128_t va1i = wasm_v128_load(ai + 4);
    const v128_t vb1r = wasm_v128_load(br + 4);
    const v128_t vb1i = wasm_v128_load(bi + 4);
    ar += 8;
    ai += 8;
    br += 8;
    bi += 8;

    v128_t vacc0r = wasm_f32x4_mul(va0r, vb0r);
    v128_t vacc0i = wasm_f32x4_mul(va0r, vb0i);
    v128_t vacc1r = wasm_f32x4_mul(va1r, vb1r);
    v128_t vacc1i = wasm_f32x4_mul(va1r, vb1i);

    vacc0r = wasm_f32x4_sub(vacc0r, wasm_f32x4_mul(va0i, vb0i));
    vacc0i = wasm_f32x4_add(vacc0i, wasm_f32x4_mul(va0i, vb0r));
    vacc1r = wasm_f32x4_sub(vacc1r, wasm_f32x4_mul(va1i, vb1i));
    vacc1i = wasm_f32x4_add(vacc1i, wasm_f32x4_mul(va1i, vb1r));

    wasm_v128_store(or, vacc0r);
    wasm_v128_store(oi, vacc0i);
    wasm_v128_store(or + 4, vacc1r);
    wasm_v128_store(oi + 4, vacc1i);
    or += 8;
    oi += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t var = wasm_v128_load(ar);
    ar += 4;
    const v128_t vai = wasm_v128_load(ai);
    ai += 4;
    const v128_t vbr = wasm_v128_load(br);
    br += 4;
    const v128_t vbi = wasm_v128_load(bi);
    bi += 4;

    v128_t vaccr = wasm_f32x4_mul(var, vbr);
    v128_t vacci = wasm_f32x4_mul(var, vbi);

    vaccr = wasm_f32x4_sub(vaccr, wasm_f32x4_mul(vai, vbi));
    vacci = wasm_f32x4_add(vacci, wasm_f32x4_mul(vai, vbr));

    wasm_v128_store(or, vaccr);
    or += 4;
    wasm_v128_store(oi, vacci);
    oi += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t var = wasm_v128_load(ar);
    ar += 4;
    const v128_t vai = wasm_v128_load(ai);
    ai += 4;
    const v128_t vbr = wasm_v128_load(br);
    br += 4;
    const v128_t vbi = wasm_v128_load(bi);
    bi += 4;

    v128_t vaccr = wasm_f32x4_mul(var, vbr);
    v128_t vacci = wasm_f32x4_mul(var, vbi);

    vaccr = wasm_f32x4_sub(vaccr, wasm_f32x4_mul(vai, vbi));
    vacci = wasm_f32x4_add(vacci, wasm_f32x4_mul(vai, vbr));

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(or, vaccr, 0);
      or += 2;
      wasm_v128_store64_lane(oi, vacci, 0);
      oi += 2;
      vaccr = wasm_v64x2_shuffle(vaccr, vaccr, 1, 1);
      vacci = wasm_v64x2_shuffle(vacci, vacci, 1, 1);
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(or, vaccr, 0);
      wasm_v128_store32_lane(oi, vacci, 0);
    }
  }
}
