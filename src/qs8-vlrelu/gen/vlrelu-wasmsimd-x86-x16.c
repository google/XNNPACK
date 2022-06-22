// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/wasmsimd-x86.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/vcvt.h>


void xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x16(
    size_t n,
    const int8_t* x,
    int8_t* y,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(int8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const v128_t vinput_zero_point = wasm_v128_load64_splat(params->wasmsimd_x86.input_zero_point);
  const v128_t vmultiplier_diff = wasm_v128_load64_splat(params->wasmsimd_x86.multiplier_diff);
  const v128_t vmultiplier_base = wasm_v128_load64_splat(params->wasmsimd_x86.multiplier_base);
  const v128_t voutput_zero_point = wasm_v128_load64_splat(params->wasmsimd_x86.output_zero_point);
  for (; n >= 16 * sizeof(int8_t); n -= 16 * sizeof(int8_t)) {
    v128_t vacc0 = wasm_i16x8_load8x8(x);
    v128_t vacc1 = wasm_i16x8_load8x8(x + 8);
    x += 16;

    v128_t vmultiplier0 = wasm_i16x8_gt(vacc0, vinput_zero_point);
    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    v128_t vmultiplier1 = wasm_i16x8_gt(vacc1, vinput_zero_point);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);

    vmultiplier0 = wasm_v128_and(vmultiplier0, vmultiplier_diff);
    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = wasm_v128_xor(vmultiplier0, vmultiplier_base);
    vmultiplier1 = wasm_v128_and(vmultiplier1, vmultiplier_diff);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = wasm_v128_xor(vmultiplier1, vmultiplier_base);

    vacc0 = wasm_i16x8_q15mulr_sat(vacc0, vmultiplier0);
    vacc1 = wasm_i16x8_q15mulr_sat(vacc1, vmultiplier1);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);

    const v128_t vy0 = wasm_i8x16_narrow_i16x8(vacc0, vacc1);

    wasm_v128_store(y, vy0);
    y += 16;
  }
  for (; n >= 8 * sizeof(int8_t); n -= 8 * sizeof(int8_t)) {
    v128_t vacc = wasm_i16x8_load8x8(x);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = wasm_i16x8_q15mulr_sat(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    x += 8;

    const v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(y, vy, 0);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(int8_t));
    assert(n <= 7 * sizeof(int8_t));

    v128_t vacc = wasm_i16x8_load8x8(x);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = wasm_i16x8_q15mulr_sat(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    if (n & (4 * sizeof(int8_t))) {
      wasm_v128_store32_lane(y, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      y += 4;
    }
    if (n & (2 * sizeof(int8_t))) {
      wasm_v128_store16_lane(y, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      y += 2;
    }
    if (n & (1 * sizeof(int8_t))) {
      wasm_v128_store8_lane(y, vy, 0);
    }
  }
}
