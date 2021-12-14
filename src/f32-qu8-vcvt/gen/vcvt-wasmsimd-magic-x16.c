// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/wasmsimd-magic.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vcvt.h>


void xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_x16(
    size_t n,
    const float* x,
    uint8_t* y,
    const union xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const v128_t vscale = wasm_v128_load64_splat(params->wasmsimd_magic.scale);
  const v128_t vmagic_bias = wasm_v128_load64_splat(params->wasmsimd_magic.magic_bias);
  const v128_t vmagic_min = wasm_v128_load64_splat(params->wasmsimd_magic.magic_min);
  const v128_t vmagic_bias_less_zero_point = wasm_v128_load64_splat(params->wasmsimd_magic.magic_bias_less_zero_point);
  const v128_t voutput_max = wasm_v128_load64_splat(params->wasmsimd_magic.output_max);
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    v128_t vx0123 = wasm_v128_load(x);
    v128_t vx4567 = wasm_v128_load(x + 4);
    v128_t vx89AB = wasm_v128_load(x + 8);
    v128_t vxCDEF = wasm_v128_load(x + 12);
    x += 16;

    vx0123 = wasm_f32x4_mul(vx0123, vscale);
    vx4567 = wasm_f32x4_mul(vx4567, vscale);
    vx89AB = wasm_f32x4_mul(vx89AB, vscale);
    vxCDEF = wasm_f32x4_mul(vxCDEF, vscale);

    vx0123 = wasm_f32x4_add(vx0123, vmagic_bias);
    vx4567 = wasm_f32x4_add(vx4567, vmagic_bias);
    vx89AB = wasm_f32x4_add(vx89AB, vmagic_bias);
    vxCDEF = wasm_f32x4_add(vxCDEF, vmagic_bias);

    v128_t vacc0123 = wasm_i32x4_max(vx0123, vmagic_min);
    v128_t vacc4567 = wasm_i32x4_max(vx4567, vmagic_min);
    v128_t vacc89AB = wasm_i32x4_max(vx89AB, vmagic_min);
    v128_t vaccCDEF = wasm_i32x4_max(vxCDEF, vmagic_min);

    vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_zero_point);
    vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_zero_point);
    vacc89AB = wasm_i32x4_sub(vacc89AB, vmagic_bias_less_zero_point);
    vaccCDEF = wasm_i32x4_sub(vaccCDEF, vmagic_bias_less_zero_point);

    const v128_t vacc01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
    const v128_t vacc89ABCDEF = wasm_i16x8_narrow_i32x4(vacc89AB, vaccCDEF);

    v128_t vy0123456789ABCDEF = wasm_u8x16_narrow_i16x8(vacc01234567, vacc89ABCDEF);

    vy0123456789ABCDEF = wasm_u8x16_min(vy0123456789ABCDEF, voutput_max);

    wasm_v128_store(y, vy0123456789ABCDEF);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    v128_t vx_lo = wasm_v128_load(x);
    v128_t vx_hi = wasm_v128_load(x + 4);
    x += 8;

    vx_lo = wasm_f32x4_mul(vx_lo, vscale);
    vx_hi = wasm_f32x4_mul(vx_hi, vscale);

    vx_lo = wasm_f32x4_add(vx_lo, vmagic_bias);
    vx_hi = wasm_f32x4_add(vx_hi, vmagic_bias);

    v128_t vacc_lo = wasm_i32x4_max(vx_lo, vmagic_min);
    v128_t vacc_hi = wasm_i32x4_max(vx_hi, vmagic_min);

    vacc_lo = wasm_i32x4_sub(vacc_lo, vmagic_bias_less_zero_point);
    vacc_hi = wasm_i32x4_sub(vacc_hi, vmagic_bias_less_zero_point);

    const v128_t vacc = wasm_i16x8_narrow_i32x4(vacc_lo, vacc_hi);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    vy = wasm_u8x16_min(vy, voutput_max);
    *((double*) y) = wasm_f64x2_extract_lane(vy, 0);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    v128_t vx_lo = wasm_v128_load(x);
    const float* x_hi = (const float*) ((uintptr_t) x + (n & (4 * sizeof(float))));
    v128_t vx_hi = wasm_v128_load(x_hi);

    vx_lo = wasm_f32x4_mul(vx_lo, vscale);
    vx_hi = wasm_f32x4_mul(vx_hi, vscale);

    vx_lo = wasm_f32x4_add(vx_lo, vmagic_bias);
    vx_hi = wasm_f32x4_add(vx_hi, vmagic_bias);

    v128_t vacc_lo = wasm_i32x4_max(vx_lo, vmagic_min);
    v128_t vacc_hi = wasm_i32x4_max(vx_hi, vmagic_min);

    vacc_lo = wasm_i32x4_sub(vacc_lo, vmagic_bias_less_zero_point);
    vacc_hi = wasm_i32x4_sub(vacc_hi, vmagic_bias_less_zero_point);

    const v128_t vacc = wasm_i16x8_narrow_i32x4(vacc_lo, vacc_hi);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    vy = wasm_u8x16_min(vy, voutput_max);

    if (n & (4 * sizeof(float))) {
      *((float*) y) = wasm_f32x4_extract_lane(vy, 0);
      y += 4;
      vy = wasm_u64x2_shr(vy, 32);
    }
    uint32_t vy_lo = (uint32_t) wasm_i32x4_extract_lane(vy, 0);
    if (n & (2 * sizeof(float))) {
      *((uint16_t*) y) = (uint16_t) vy_lo;
      y += 2;
      vy_lo >>= 16;
    }
    if (n & (1 * sizeof(float))) {
      *y = (uint8_t) vy_lo;
    }
  }
}
