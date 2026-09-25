// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vcvt.h"


void xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vbias = wasm_i32x4_splat(
      (int32_t) (((uint32_t) (int32_t) params->scalar.output_zero_point) << 16) -
      (int32_t) params->scalar.multiplier * (int32_t) params->scalar.input_zero_point +
      INT32_C(0x8000));
  const v128_t vmultiplier = wasm_i32x4_splat(params->scalar.multiplier);
  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(vmultiplier);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    v128_t vacc0 = wasm_i16x8_load8x8(input);
    v128_t vacc1 = wasm_i16x8_load8x8(input + 8);
    v128_t vacc2 = wasm_i16x8_load8x8(input + 16);
    v128_t vacc3 = wasm_i16x8_load8x8(input + 24);
    input += 32;

    v128_t vacc_lo0 = wasm_i32x4_extend_low_i16x8(vacc0);
    v128_t vacc_hi0 = wasm_i32x4_extend_high_i16x8(vacc0);
    v128_t vacc_lo1 = wasm_i32x4_extend_low_i16x8(vacc1);
    v128_t vacc_hi1 = wasm_i32x4_extend_high_i16x8(vacc1);
    v128_t vacc_lo2 = wasm_i32x4_extend_low_i16x8(vacc2);
    v128_t vacc_hi2 = wasm_i32x4_extend_high_i16x8(vacc2);
    v128_t vacc_lo3 = wasm_i32x4_extend_low_i16x8(vacc3);
    v128_t vacc_hi3 = wasm_i32x4_extend_high_i16x8(vacc3);

    vacc_lo0 = wasm_i32x4_add(wasm_i32x4_mul(vacc_lo0, vmultiplier), vbias);
    vacc_hi0 = wasm_i32x4_add(wasm_i32x4_mul(vacc_hi0, vmultiplier), vbias);
    vacc_lo1 = wasm_i32x4_add(wasm_i32x4_mul(vacc_lo1, vmultiplier), vbias);
    vacc_hi1 = wasm_i32x4_add(wasm_i32x4_mul(vacc_hi1, vmultiplier), vbias);
    vacc_lo2 = wasm_i32x4_add(wasm_i32x4_mul(vacc_lo2, vmultiplier), vbias);
    vacc_hi2 = wasm_i32x4_add(wasm_i32x4_mul(vacc_hi2, vmultiplier), vbias);
    vacc_lo3 = wasm_i32x4_add(wasm_i32x4_mul(vacc_lo3, vmultiplier), vbias);
    vacc_hi3 = wasm_i32x4_add(wasm_i32x4_mul(vacc_hi3, vmultiplier), vbias);

    vacc_lo0 = wasm_i32x4_shr(vacc_lo0, 16);
    vacc_hi0 = wasm_i32x4_shr(vacc_hi0, 16);
    vacc_lo1 = wasm_i32x4_shr(vacc_lo1, 16);
    vacc_hi1 = wasm_i32x4_shr(vacc_hi1, 16);
    vacc_lo2 = wasm_i32x4_shr(vacc_lo2, 16);
    vacc_hi2 = wasm_i32x4_shr(vacc_hi2, 16);
    vacc_lo3 = wasm_i32x4_shr(vacc_lo3, 16);
    vacc_hi3 = wasm_i32x4_shr(vacc_hi3, 16);

    vacc0 = wasm_i16x8_narrow_i32x4(vacc_lo0, vacc_hi0);
    vacc1 = wasm_i16x8_narrow_i32x4(vacc_lo1, vacc_hi1);
    vacc2 = wasm_i16x8_narrow_i32x4(vacc_lo2, vacc_hi2);
    vacc3 = wasm_i16x8_narrow_i32x4(vacc_lo3, vacc_hi3);

    const v128_t vy0 = wasm_i8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_i8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    v128_t vacc = wasm_i16x8_load8x8(input);
    v128_t vacc_lo = wasm_i32x4_extend_low_i16x8(vacc);
    v128_t vacc_hi = wasm_i32x4_extend_high_i16x8(vacc);
    vacc_lo = wasm_i32x4_add(wasm_i32x4_mul(vacc_lo, vmultiplier), vbias);
    vacc_hi = wasm_i32x4_add(wasm_i32x4_mul(vacc_hi, vmultiplier), vbias);
    vacc_lo = wasm_i32x4_shr(vacc_lo, 16);
    vacc_hi = wasm_i32x4_shr(vacc_hi, 16);
    vacc = wasm_i16x8_narrow_i32x4(vacc_lo, vacc_hi);
    input += 8;

    const v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    v128_t vacc = wasm_i16x8_load8x8(input);
    v128_t vacc_lo = wasm_i32x4_extend_low_i16x8(vacc);
    v128_t vacc_hi = wasm_i32x4_extend_high_i16x8(vacc);
    vacc_lo = wasm_i32x4_add(wasm_i32x4_mul(vacc_lo, vmultiplier), vbias);
    vacc_hi = wasm_i32x4_add(wasm_i32x4_mul(vacc_hi, vmultiplier), vbias);
    vacc_lo = wasm_i32x4_shr(vacc_lo, 16);
    vacc_hi = wasm_i32x4_shr(vacc_hi, 16);
    vacc = wasm_i16x8_narrow_i32x4(vacc_lo, vacc_hi);

    v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}
