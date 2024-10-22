// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/wasmrelaxedsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"

void xnn_qs8_rsum_ukernel__wasmrelaxedsimd_u64_acc2(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  XNN_ALIGN(32) static const int8_t mask_table[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  const v128_t vone = wasm_i8x16_const_splat(1);

  v128_t vacc0 = wasm_i32x4_const_splat(0);
  v128_t vacc1 = wasm_i32x4_const_splat(0);

  for (; batch >= 64; batch -= 64) {
    const v128_t vt0 = wasm_v128_load(input); input += 16;
    const v128_t vt1 = wasm_v128_load(input); input += 16;
    const v128_t vt2 = wasm_v128_load(input); input += 16;
    const v128_t vt3 = wasm_v128_load(input); input += 16;
    vacc0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vt0, vone, vacc0);
    vacc1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vt1, vone, vacc1);
    vacc0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vt2, vone, vacc0);
    vacc1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vt3, vone, vacc1);
  }

  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch < 32);
    for (; batch >= 16; batch -= 16) {
      const v128_t vt3 = wasm_v128_load(input); input += 16;
      vacc1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vt3, vone, vacc1);
    }
  }

  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch <= 15);
    const v128_t vmask = wasm_v128_load(&mask_table[16 - batch]);
    const v128_t vt = wasm_v128_bitselect(wasm_v128_load(input), wasm_i8x16_const_splat(0), vmask);
    vacc0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vt, vone, vacc0);
  }

  vacc0 = wasm_i32x4_add(vacc0, vacc1);

  v128_t vacc_shifted = wasm_i32x4_shuffle(vacc0, vacc0, 1, 0, 3, 2);
  v128_t vacc_lo = wasm_i32x4_add(vacc0, vacc_shifted);

  v128_t vacc_final_shifted = wasm_i32x4_shuffle(vacc_lo, vacc_lo, 2, 3, 0, 1);
  v128_t vacc_final = wasm_i32x4_add(vacc_lo, vacc_final_shifted);
  const int32_t vacc = wasm_i32x4_extract_lane(vacc_final, 0);

  *output += vacc;
}
