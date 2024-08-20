// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/wasmsimd-int16.c.in
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


void xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u24(
    size_t batch,
    const void* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vsign_mask = wasm_u16x8_const_splat(UINT16_C(0x8000));
  const v128_t vexp_offset = wasm_u16x8_const_splat(UINT16_C(0x7000));
  const v128_t vexp_scale = wasm_f32x4_const_splat(0x1.0p-112f);
  const v128_t vmagic_mask = wasm_u16x8_const_splat(UINT16_C(0x3F00));
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0.5f);
  const v128_t vdenorm_cutoff = wasm_u16x8_const_splat(UINT16_C(0x0400));

  XNN_FORCE_REALIZATION(vsign_mask);
  XNN_FORCE_REALIZATION(vexp_offset);
  XNN_FORCE_REALIZATION(vexp_scale);
  XNN_FORCE_REALIZATION(vmagic_mask);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 24 * sizeof(uint16_t); batch -= 24 * sizeof(uint16_t)) {
    const v128_t vh0 = wasm_v128_load(i);
    const v128_t vh1 = wasm_v128_load(i + 8);
    const v128_t vh2 = wasm_v128_load(i + 16);
    i += 24;

    const v128_t vsign0 = wasm_v128_and(vh0, vsign_mask);
    const v128_t vsign1 = wasm_v128_and(vh1, vsign_mask);
    const v128_t vsign2 = wasm_v128_and(vh2, vsign_mask);

    const v128_t vnonsign0 = wasm_v128_xor(vh0, vsign0);
    const v128_t vnonsign1 = wasm_v128_xor(vh1, vsign1);
    const v128_t vnonsign2 = wasm_v128_xor(vh2, vsign2);

    const v128_t vprenorm0 = wasm_i16x8_shl(vnonsign0, 13);
    const v128_t vprenorm1 = wasm_i16x8_add(wasm_u16x8_shr(vnonsign0, 3), vexp_offset);
    const v128_t vprenorm2 = wasm_i16x8_shl(vnonsign1, 13);
    const v128_t vprenorm3 = wasm_i16x8_add(wasm_u16x8_shr(vnonsign1, 3), vexp_offset);
    const v128_t vprenorm4 = wasm_i16x8_shl(vnonsign2, 13);
    const v128_t vprenorm5 = wasm_i16x8_add(wasm_u16x8_shr(vnonsign2, 3), vexp_offset);

    const v128_t vnorm0 = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm0, vprenorm1, 0,  8, 1,  9, 2, 10, 3, 11), vexp_scale);
    const v128_t vnorm1 = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm0, vprenorm1, 4, 12, 5, 13, 6, 14, 7, 15), vexp_scale);
    const v128_t vnorm2 = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm2, vprenorm3, 0,  8, 1,  9, 2, 10, 3, 11), vexp_scale);
    const v128_t vnorm3 = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm2, vprenorm3, 4, 12, 5, 13, 6, 14, 7, 15), vexp_scale);
    const v128_t vnorm4 = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm4, vprenorm5, 0,  8, 1,  9, 2, 10, 3, 11), vexp_scale);
    const v128_t vnorm5 = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm4, vprenorm5, 4, 12, 5, 13, 6, 14, 7, 15), vexp_scale);

    const v128_t vdenorm0 = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign0, vmagic_mask, 0,  8, 1,  9, 2, 10, 3, 11), vmagic_bias);
    const v128_t vdenorm1 = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign0, vmagic_mask, 4, 12, 5, 13, 6, 14, 7, 15), vmagic_bias);
    const v128_t vdenorm2 = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign1, vmagic_mask, 0,  8, 1,  9, 2, 10, 3, 11), vmagic_bias);
    const v128_t vdenorm3 = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign1, vmagic_mask, 4, 12, 5, 13, 6, 14, 7, 15), vmagic_bias);
    const v128_t vdenorm4 = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign2, vmagic_mask, 0,  8, 1,  9, 2, 10, 3, 11), vmagic_bias);
    const v128_t vdenorm5 = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign2, vmagic_mask, 4, 12, 5, 13, 6, 14, 7, 15), vmagic_bias);

    const v128_t vmask0 = wasm_i16x8_gt(vnonsign0, vdenorm_cutoff);
    const v128_t vmask1 = wasm_i16x8_gt(vnonsign1, vdenorm_cutoff);
    const v128_t vmask2 = wasm_i16x8_gt(vnonsign2, vdenorm_cutoff);
    const v128_t vzero = wasm_i16x8_const_splat(0);

    const v128_t vxmask0 = wasm_i32x4_extend_low_i16x8(vmask0);
    const v128_t vxmask1 = wasm_i32x4_extend_high_i16x8(vmask0);
    const v128_t vxmask2 = wasm_i32x4_extend_low_i16x8(vmask1);
    const v128_t vxmask3 = wasm_i32x4_extend_high_i16x8(vmask1);
    const v128_t vxmask4 = wasm_i32x4_extend_low_i16x8(vmask2);
    const v128_t vxmask5 = wasm_i32x4_extend_high_i16x8(vmask2);

    const v128_t vabsf0 = wasm_i32x4_relaxed_laneselect(vnorm0, vdenorm0, vxmask0);
    const v128_t vsignf0 = wasm_v16x8_shuffle(vzero, vsign0, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vabsf1 = wasm_i32x4_relaxed_laneselect(vnorm1, vdenorm1, vxmask1);
    const v128_t vsignf1 = wasm_v16x8_shuffle(vzero, vsign0, 4, 12, 5, 13, 6, 14, 7, 15);
    const v128_t vabsf2 = wasm_i32x4_relaxed_laneselect(vnorm2, vdenorm2, vxmask2);
    const v128_t vsignf2 = wasm_v16x8_shuffle(vzero, vsign1, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vabsf3 = wasm_i32x4_relaxed_laneselect(vnorm3, vdenorm3, vxmask3);
    const v128_t vsignf3 = wasm_v16x8_shuffle(vzero, vsign1, 4, 12, 5, 13, 6, 14, 7, 15);
    const v128_t vabsf4 = wasm_i32x4_relaxed_laneselect(vnorm4, vdenorm4, vxmask4);
    const v128_t vsignf4 = wasm_v16x8_shuffle(vzero, vsign2, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vabsf5 = wasm_i32x4_relaxed_laneselect(vnorm5, vdenorm5, vxmask5);
    const v128_t vsignf5 = wasm_v16x8_shuffle(vzero, vsign2, 4, 12, 5, 13, 6, 14, 7, 15);

    const v128_t vf0 = wasm_v128_or(vsignf0, vabsf0);
    const v128_t vf1 = wasm_v128_or(vsignf1, vabsf1);
    const v128_t vf2 = wasm_v128_or(vsignf2, vabsf2);
    const v128_t vf3 = wasm_v128_or(vsignf3, vabsf3);
    const v128_t vf4 = wasm_v128_or(vsignf4, vabsf4);
    const v128_t vf5 = wasm_v128_or(vsignf5, vabsf5);

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

    const v128_t vsign = wasm_v128_and(vh, vsign_mask);

    const v128_t vnonsign = wasm_v128_xor(vh, vsign);

    const v128_t vprenorm_lo = wasm_i16x8_shl(vnonsign, 13);
    const v128_t vprenorm_hi = wasm_i16x8_add(wasm_u16x8_shr(vnonsign, 3), vexp_offset);

    const v128_t vnorm_lo = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm_lo, vprenorm_hi, 0,  8, 1,  9, 2, 10, 3, 11), vexp_scale);
    const v128_t vnorm_hi = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm_lo, vprenorm_hi, 4, 12, 5, 13, 6, 14, 7, 15), vexp_scale);

    const v128_t vdenorm_lo = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign, vmagic_mask, 0,  8, 1,  9, 2, 10, 3, 11), vmagic_bias);
    const v128_t vdenorm_hi = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign, vmagic_mask, 4, 12, 5, 13, 6, 14, 7, 15), vmagic_bias);

    const v128_t vmask = wasm_i16x8_gt(vnonsign, vdenorm_cutoff);
    const v128_t vzero = wasm_i16x8_const_splat(0);

    const v128_t vxmask_lo = wasm_i32x4_extend_low_i16x8(vmask);
    const v128_t vxmask_hi = wasm_i32x4_extend_high_i16x8(vmask);

    const v128_t vabsf_lo = wasm_i32x4_relaxed_laneselect(vnorm_lo, vdenorm_lo, vxmask_lo);
    const v128_t vsignf_lo = wasm_v16x8_shuffle(vzero, vsign, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vabsf_hi = wasm_i32x4_relaxed_laneselect(vnorm_hi, vdenorm_hi, vxmask_hi);
    const v128_t vsignf_hi = wasm_v16x8_shuffle(vzero, vsign, 4, 12, 5, 13, 6, 14, 7, 15);

    const v128_t vf_lo = wasm_v128_or(vsignf_lo, vabsf_lo);
    const v128_t vf_hi = wasm_v128_or(vsignf_hi, vabsf_hi);

    wasm_v128_store(output, vf_lo);
    wasm_v128_store(output + 4, vf_hi);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const v128_t vh = wasm_v128_load(i);

    const v128_t vsign = wasm_v128_and(vh, vsign_mask);

    const v128_t vnonsign = wasm_v128_xor(vh, vsign);

    const v128_t vprenorm_lo = wasm_i16x8_shl(vnonsign, 13);
    const v128_t vprenorm_hi = wasm_i16x8_add(wasm_u16x8_shr(vnonsign, 3), vexp_offset);

    const v128_t vnorm_lo = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm_lo, vprenorm_hi, 0,  8, 1,  9, 2, 10, 3, 11), vexp_scale);
    const v128_t vnorm_hi = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm_lo, vprenorm_hi, 4, 12, 5, 13, 6, 14, 7, 15), vexp_scale);

    const v128_t vdenorm_lo = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign, vmagic_mask, 0,  8, 1,  9, 2, 10, 3, 11), vmagic_bias);
    const v128_t vdenorm_hi = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign, vmagic_mask, 4, 12, 5, 13, 6, 14, 7, 15), vmagic_bias);

    const v128_t vmask = wasm_i16x8_gt(vnonsign, vdenorm_cutoff);
    const v128_t vzero = wasm_i16x8_const_splat(0);

    const v128_t vxmask_lo = wasm_i32x4_extend_low_i16x8(vmask);
    const v128_t vxmask_hi = wasm_i32x4_extend_high_i16x8(vmask);

    const v128_t vabsf_lo = wasm_i32x4_relaxed_laneselect(vnorm_lo, vdenorm_lo, vxmask_lo);
    const v128_t vsignf_lo = wasm_v16x8_shuffle(vzero, vsign, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vabsf_hi = wasm_i32x4_relaxed_laneselect(vnorm_hi, vdenorm_hi, vxmask_hi);
    const v128_t vsignf_hi = wasm_v16x8_shuffle(vzero, vsign, 4, 12, 5, 13, 6, 14, 7, 15);

    v128_t vf = wasm_v128_or(vsignf_lo, vabsf_lo);

    if (batch & (4 * sizeof(uint16_t))) {
      wasm_v128_store(output, vf);
      output += 4;

      vf = wasm_v128_or(vsignf_hi, vabsf_hi);
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
