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


void xnn_f32_vcmul_ukernel__wasmsimd_u16(
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
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const v128_t va0r = wasm_v128_load(ar);
    const v128_t va0i = wasm_v128_load(ai);
    const v128_t vb0r = wasm_v128_load(br);
    const v128_t vb0i = wasm_v128_load(bi);
    const v128_t va1r = wasm_v128_load(ar + 4);
    const v128_t va1i = wasm_v128_load(ai + 4);
    const v128_t vb1r = wasm_v128_load(br + 4);
    const v128_t vb1i = wasm_v128_load(bi + 4);
    const v128_t va2r = wasm_v128_load(ar + 8);
    const v128_t va2i = wasm_v128_load(ai + 8);
    const v128_t vb2r = wasm_v128_load(br + 8);
    const v128_t vb2i = wasm_v128_load(bi + 8);
    const v128_t va3r = wasm_v128_load(ar + 12);
    const v128_t va3i = wasm_v128_load(ai + 12);
    const v128_t vb3r = wasm_v128_load(br + 12);
    const v128_t vb3i = wasm_v128_load(bi + 12);
    ar += 16;
    ai += 16;
    br += 16;
    bi += 16;

    v128_t vacc0r = wasm_f32x4_mul(va0r, vb0r);
    v128_t vacc0i = wasm_f32x4_mul(va0r, vb0i);
    v128_t vacc1r = wasm_f32x4_mul(va1r, vb1r);
    v128_t vacc1i = wasm_f32x4_mul(va1r, vb1i);
    v128_t vacc2r = wasm_f32x4_mul(va2r, vb2r);
    v128_t vacc2i = wasm_f32x4_mul(va2r, vb2i);
    v128_t vacc3r = wasm_f32x4_mul(va3r, vb3r);
    v128_t vacc3i = wasm_f32x4_mul(va3r, vb3i);

    vacc0r = wasm_f32x4_sub(vacc0r, wasm_f32x4_mul(va0i, vb0i));
    vacc0i = wasm_f32x4_add(vacc0i, wasm_f32x4_mul(va0i, vb0r));
    vacc1r = wasm_f32x4_sub(vacc1r, wasm_f32x4_mul(va1i, vb1i));
    vacc1i = wasm_f32x4_add(vacc1i, wasm_f32x4_mul(va1i, vb1r));
    vacc2r = wasm_f32x4_sub(vacc2r, wasm_f32x4_mul(va2i, vb2i));
    vacc2i = wasm_f32x4_add(vacc2i, wasm_f32x4_mul(va2i, vb2r));
    vacc3r = wasm_f32x4_sub(vacc3r, wasm_f32x4_mul(va3i, vb3i));
    vacc3i = wasm_f32x4_add(vacc3i, wasm_f32x4_mul(va3i, vb3r));

    wasm_v128_store(or, vacc0r);
    wasm_v128_store(oi, vacc0i);
    wasm_v128_store(or + 4, vacc1r);
    wasm_v128_store(oi + 4, vacc1i);
    wasm_v128_store(or + 8, vacc2r);
    wasm_v128_store(oi + 8, vacc2i);
    wasm_v128_store(or + 12, vacc3r);
    wasm_v128_store(oi + 12, vacc3i);
    or += 16;
    oi += 16;
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
