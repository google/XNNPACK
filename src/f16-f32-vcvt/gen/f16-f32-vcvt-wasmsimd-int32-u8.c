// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/wasmsimd-int32.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/vcvt.h"


void xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u8(
    size_t batch,
    const void* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vsign_mask = wasm_i32x4_const_splat(UINT32_C(0x80000000));
  const v128_t vexp_offset = wasm_i32x4_const_splat(UINT32_C(0x70000000));
  const v128_t vexp_scale = wasm_f32x4_const_splat(0x1.0p-112f);
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0.5f);
  const v128_t vdenorm_cutoff = wasm_i32x4_const_splat(UINT32_C(0x08000000));

  XNN_FORCE_REALIZATION(vsign_mask);
  XNN_FORCE_REALIZATION(vexp_offset);
  XNN_FORCE_REALIZATION(vexp_scale);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const v128_t vh = wasm_v128_load(i);
    i += 8;

    const v128_t vzero = wasm_i16x8_const_splat(0);
    const v128_t vw_lo = wasm_v16x8_shuffle(vzero, vh, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vw_hi = wasm_v16x8_shuffle(vzero, vh, 4, 12, 5, 13, 6, 14, 7, 15);

    const v128_t vsign_lo = wasm_v128_and(vw_lo, vsign_mask);
    const v128_t vsign_hi = wasm_v128_and(vw_hi, vsign_mask);

    const v128_t vnonsign_lo = wasm_v128_xor(vw_lo, vsign_lo);
    const v128_t vnonsign_hi = wasm_v128_xor(vw_hi, vsign_hi);

    const v128_t vnorm_lo = wasm_f32x4_mul(wasm_i32x4_add(wasm_u32x4_shr(vnonsign_lo, 3), vexp_offset), vexp_scale);
    const v128_t vnorm_hi = wasm_f32x4_mul(wasm_i32x4_add(wasm_u32x4_shr(vnonsign_hi, 3), vexp_offset), vexp_scale);

    const v128_t vdenorm_lo = wasm_f32x4_sub(wasm_v128_or(wasm_u32x4_shr(vnonsign_lo, 16), vmagic_bias), vmagic_bias);
    const v128_t vdenorm_hi = wasm_f32x4_sub(wasm_v128_or(wasm_u32x4_shr(vnonsign_hi, 16), vmagic_bias), vmagic_bias);

    const v128_t vxmask_lo = wasm_i32x4_gt(vnonsign_lo, vdenorm_cutoff);
    const v128_t vxmask_hi = wasm_i32x4_gt(vnonsign_hi, vdenorm_cutoff);

    const v128_t vf_lo = wasm_v128_or(vsign_lo, wasm_v128_bitselect(vnorm_lo, vdenorm_lo, vxmask_lo));
    const v128_t vf_hi = wasm_v128_or(vsign_hi, wasm_v128_bitselect(vnorm_hi, vdenorm_hi, vxmask_hi));

    wasm_v128_store(output, vf_lo);
    wasm_v128_store(output + 4, vf_hi);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const v128_t vh = wasm_v128_load(i);

    const v128_t vzero = wasm_i16x8_const_splat(0);
    const v128_t vw_lo = wasm_v16x8_shuffle(vzero, vh, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vw_hi = wasm_v16x8_shuffle(vzero, vh, 4, 12, 5, 13, 6, 14, 7, 15);

    const v128_t vsign_lo = wasm_v128_and(vw_lo, vsign_mask);
    const v128_t vsign_hi = wasm_v128_and(vw_hi, vsign_mask);

    const v128_t vnonsign_lo = wasm_v128_xor(vw_lo, vsign_lo);
    const v128_t vnonsign_hi = wasm_v128_xor(vw_hi, vsign_hi);

    const v128_t vnorm_lo = wasm_f32x4_mul(wasm_i32x4_add(wasm_u32x4_shr(vnonsign_lo, 3), vexp_offset), vexp_scale);
    const v128_t vnorm_hi = wasm_f32x4_mul(wasm_i32x4_add(wasm_u32x4_shr(vnonsign_hi, 3), vexp_offset), vexp_scale);

    const v128_t vdenorm_lo = wasm_f32x4_sub(wasm_v128_or(wasm_u32x4_shr(vnonsign_lo, 16), vmagic_bias), vmagic_bias);
    const v128_t vdenorm_hi = wasm_f32x4_sub(wasm_v128_or(wasm_u32x4_shr(vnonsign_hi, 16), vmagic_bias), vmagic_bias);

    const v128_t vxmask_lo = wasm_i32x4_gt(vnonsign_lo, vdenorm_cutoff);
    v128_t vf = wasm_v128_or(vsign_lo, wasm_v128_bitselect(vnorm_lo, vdenorm_lo, vxmask_lo));

    if (batch & (4 * sizeof(uint16_t))) {
      wasm_v128_store(output, vf);
      output += 4;

      const v128_t vxmask_hi = wasm_i32x4_gt(vnonsign_hi, vdenorm_cutoff);
      vf = wasm_v128_or(vsign_hi, wasm_v128_bitselect(vnorm_hi, vdenorm_hi, vxmask_hi));
    }
    if (batch & (2 * sizeof(uint16_t))) {
      wasm_v128_store64_lane(output, vf, 0);
      vf = wasm_v64x2_shuffle(vf, vf, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      wasm_v128_store32_lane(output, vf, 0);
    }
  }
}
