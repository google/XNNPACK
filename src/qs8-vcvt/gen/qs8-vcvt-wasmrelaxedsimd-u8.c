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


void xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_u8(
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
