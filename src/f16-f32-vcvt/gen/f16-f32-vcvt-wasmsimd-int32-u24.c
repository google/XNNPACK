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


void xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u24(
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
  for (; batch >= 24 * sizeof(uint16_t); batch -= 24 * sizeof(uint16_t)) {
    const v128_t vh0 = wasm_v128_load(i);
    const v128_t vh1 = wasm_v128_load(i + 8);
    const v128_t vh2 = wasm_v128_load(i + 16);
    i += 24;

    const v128_t vzero = wasm_i16x8_const_splat(0);
    const v128_t vw0 = wasm_v16x8_shuffle(vzero, vh0, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vw1 = wasm_v16x8_shuffle(vzero, vh0, 4, 12, 5, 13, 6, 14, 7, 15);
    const v128_t vw2 = wasm_v16x8_shuffle(vzero, vh1, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vw3 = wasm_v16x8_shuffle(vzero, vh1, 4, 12, 5, 13, 6, 14, 7, 15);
    const v128_t vw4 = wasm_v16x8_shuffle(vzero, vh2, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vw5 = wasm_v16x8_shuffle(vzero, vh2, 4, 12, 5, 13, 6, 14, 7, 15);

    const v128_t vsign0 = wasm_v128_and(vw0, vsign_mask);
    const v128_t vsign1 = wasm_v128_and(vw1, vsign_mask);
    const v128_t vsign2 = wasm_v128_and(vw2, vsign_mask);
    const v128_t vsign3 = wasm_v128_and(vw3, vsign_mask);
    const v128_t vsign4 = wasm_v128_and(vw4, vsign_mask);
    const v128_t vsign5 = wasm_v128_and(vw5, vsign_mask);

    const v128_t vnonsign0 = wasm_v128_xor(vw0, vsign0);
    const v128_t vnonsign1 = wasm_v128_xor(vw1, vsign1);
    const v128_t vnonsign2 = wasm_v128_xor(vw2, vsign2);
    const v128_t vnonsign3 = wasm_v128_xor(vw3, vsign3);
    const v128_t vnonsign4 = wasm_v128_xor(vw4, vsign4);
    const v128_t vnonsign5 = wasm_v128_xor(vw5, vsign5);

    const v128_t vnorm0 = wasm_f32x4_mul(wasm_i32x4_add(wasm_u32x4_shr(vnonsign0, 3), vexp_offset), vexp_scale);
    const v128_t vnorm1 = wasm_f32x4_mul(wasm_i32x4_add(wasm_u32x4_shr(vnonsign1, 3), vexp_offset), vexp_scale);
    const v128_t vnorm2 = wasm_f32x4_mul(wasm_i32x4_add(wasm_u32x4_shr(vnonsign2, 3), vexp_offset), vexp_scale);
    const v128_t vnorm3 = wasm_f32x4_mul(wasm_i32x4_add(wasm_u32x4_shr(vnonsign3, 3), vexp_offset), vexp_scale);
    const v128_t vnorm4 = wasm_f32x4_mul(wasm_i32x4_add(wasm_u32x4_shr(vnonsign4, 3), vexp_offset), vexp_scale);
    const v128_t vnorm5 = wasm_f32x4_mul(wasm_i32x4_add(wasm_u32x4_shr(vnonsign5, 3), vexp_offset), vexp_scale);

    const v128_t vdenorm0 = wasm_f32x4_sub(wasm_v128_or(wasm_u32x4_shr(vnonsign0, 16), vmagic_bias), vmagic_bias);
    const v128_t vdenorm1 = wasm_f32x4_sub(wasm_v128_or(wasm_u32x4_shr(vnonsign1, 16), vmagic_bias), vmagic_bias);
    const v128_t vdenorm2 = wasm_f32x4_sub(wasm_v128_or(wasm_u32x4_shr(vnonsign2, 16), vmagic_bias), vmagic_bias);
    const v128_t vdenorm3 = wasm_f32x4_sub(wasm_v128_or(wasm_u32x4_shr(vnonsign3, 16), vmagic_bias), vmagic_bias);
    const v128_t vdenorm4 = wasm_f32x4_sub(wasm_v128_or(wasm_u32x4_shr(vnonsign4, 16), vmagic_bias), vmagic_bias);
    const v128_t vdenorm5 = wasm_f32x4_sub(wasm_v128_or(wasm_u32x4_shr(vnonsign5, 16), vmagic_bias), vmagic_bias);

    const v128_t vxmask0 = wasm_i32x4_gt(vnonsign0, vdenorm_cutoff);
    const v128_t vxmask1 = wasm_i32x4_gt(vnonsign1, vdenorm_cutoff);
    const v128_t vxmask2 = wasm_i32x4_gt(vnonsign2, vdenorm_cutoff);
    const v128_t vxmask3 = wasm_i32x4_gt(vnonsign3, vdenorm_cutoff);
    const v128_t vxmask4 = wasm_i32x4_gt(vnonsign4, vdenorm_cutoff);
    const v128_t vxmask5 = wasm_i32x4_gt(vnonsign5, vdenorm_cutoff);

    const v128_t vf0 = wasm_v128_or(vsign0, wasm_v128_bitselect(vnorm0, vdenorm0, vxmask0));
    const v128_t vf1 = wasm_v128_or(vsign1, wasm_v128_bitselect(vnorm1, vdenorm1, vxmask1));
    const v128_t vf2 = wasm_v128_or(vsign2, wasm_v128_bitselect(vnorm2, vdenorm2, vxmask2));
    const v128_t vf3 = wasm_v128_or(vsign3, wasm_v128_bitselect(vnorm3, vdenorm3, vxmask3));
    const v128_t vf4 = wasm_v128_or(vsign4, wasm_v128_bitselect(vnorm4, vdenorm4, vxmask4));
    const v128_t vf5 = wasm_v128_or(vsign5, wasm_v128_bitselect(vnorm5, vdenorm5, vxmask5));

    wasm_v128_store(output, vf0);
    wasm_v128_store(output + 4, vf1);
    wasm_v128_store(output + 8, vf2);
    wasm_v128_store(output + 12, vf3);
    wasm_v128_store(output + 16, vf4);
    wasm_v128_store(output + 20, vf5);
    output += 24;
  }
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
