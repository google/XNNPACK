// Auto-generated file. Do not edit!
//   Template: src/f32-velu/wasmsimd-rr2-lut16-p3.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/vunary.h"
#include "xnnpack/common.h"


extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vsat_cutoff = wasm_f32x4_const_splat(-0x1.154246p+4f);
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.800000p19f);
  const v128_t vlog2e = wasm_f32x4_const_splat(0x1.715476p+0f);
  const v128_t vindex_mask = wasm_u32x4_const_splat(UINT32_C(0xF));
  const v128_t vminus_ln2_hi = wasm_f32x4_const_splat(-0x1.62E400p-1f);
  const v128_t vminus_ln2_lo = wasm_f32x4_const_splat(-0x1.7F7D1Cp-20f);
  const v128_t vc3 = wasm_f32x4_const_splat(0x1.55561Cp-3f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.0001ECp-1f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);

  const v128_t vprescale = wasm_v128_load32_splat(&params->scalar.prescale);
  const v128_t valpha = wasm_v128_load32_splat(&params->scalar.alpha);
  const v128_t vbeta = wasm_v128_load32_splat(&params->scalar.beta);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    v128_t vx0123 = wasm_v128_load(input);
    v128_t vx4567 = wasm_v128_load(input + 4);
    input += 8;

    const v128_t vz0123 = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx0123, vprescale));
    const v128_t vz4567 = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx4567, vprescale));

    v128_t vn0123 = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz0123, vlog2e));
    v128_t vn4567 = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz4567, vlog2e));

    const v128_t vidx0123 = wasm_i32x4_shl(wasm_v128_and(vn0123, vindex_mask), 2);
    const v128_t ven0123 = wasm_i32x4_shl(vn0123, 19);
    const v128_t vidx4567 = wasm_i32x4_shl(wasm_v128_and(vn4567, vindex_mask), 2);
    const v128_t ven4567 = wasm_i32x4_shl(vn4567, 19);

    const uint32_t vidx0 = wasm_u32x4_extract_lane(vidx0123, 0);
    v128_t vl0123 = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx0));
    const uint32_t vidx4 = wasm_u32x4_extract_lane(vidx4567, 0);
    v128_t vl4567 = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx4));

    const uint32_t vidx1 = wasm_u32x4_extract_lane(vidx0123, 1);
    vl0123 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx1), vl0123, 1);
    const uint32_t vidx5 = wasm_u32x4_extract_lane(vidx4567, 1);
    vl4567 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx5), vl4567, 1);
    const uint32_t vidx2 = wasm_u32x4_extract_lane(vidx0123, 2);
    vl0123 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx2), vl0123, 2);
    const uint32_t vidx6 = wasm_u32x4_extract_lane(vidx4567, 2);
    vl4567 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx6), vl4567, 2);
    const uint32_t vidx3 = wasm_u32x4_extract_lane(vidx0123, 3);
    vl0123 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx3), vl0123, 3);
    const uint32_t vidx7 = wasm_u32x4_extract_lane(vidx4567, 3);
    vl4567 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx7), vl4567, 3);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    v128_t vs0123 = wasm_i32x4_add(vl0123, ven0123);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    v128_t vs4567 = wasm_i32x4_add(vl4567, ven4567);

    v128_t vt0123 = wasm_f32x4_add(vz0123, wasm_f32x4_mul(vn0123, vminus_ln2_hi));
    v128_t vt4567 = wasm_f32x4_add(vz4567, wasm_f32x4_mul(vn4567, vminus_ln2_hi));

    vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vn0123, vminus_ln2_lo), vt0123);
    vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vn4567, vminus_ln2_lo), vt4567);

    v128_t vp0123 = wasm_f32x4_add(vc2, wasm_f32x4_mul(vc3, vt0123));
    v128_t vp4567 = wasm_f32x4_add(vc2, wasm_f32x4_mul(vc3, vt4567));

    vp0123 = wasm_f32x4_mul(vp0123, vt0123);
    vp4567 = wasm_f32x4_mul(vp4567, vt4567);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vs0123 = wasm_f32x4_sub(vs0123, vone);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vs4567 = wasm_f32x4_sub(vs4567, vone);

    vp0123 = wasm_f32x4_add(vt0123, wasm_f32x4_mul(vp0123, vt0123));
    vp4567 = wasm_f32x4_add(vt4567, wasm_f32x4_mul(vp4567, vt4567));

    const v128_t ve0123 = wasm_f32x4_mul(valpha, wasm_f32x4_add(vp0123, vs0123));
    const v128_t ve4567 = wasm_f32x4_mul(valpha, wasm_f32x4_add(vp4567, vs4567));

    const v128_t vsignm0123 = wasm_i32x4_shr(vx0123, 31);
    vx0123 = wasm_f32x4_mul(vx0123, vbeta);
    const v128_t vsignm4567 = wasm_i32x4_shr(vx4567, 31);
    vx4567 = wasm_f32x4_mul(vx4567, vbeta);

    const v128_t vy0123 = wasm_i32x4_relaxed_laneselect(ve0123, vx0123, vsignm0123);
    const v128_t vy4567 = wasm_i32x4_relaxed_laneselect(ve4567, vx4567, vsignm4567);

    wasm_v128_store(output, vy0123);
    wasm_v128_store(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    v128_t vx = wasm_v128_load(input);
    input += 4;

    const v128_t vz = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx, vprescale));

    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz, vlog2e));
    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const v128_t ven = wasm_i32x4_shl(vn, 19);

    const uint32_t vidx0 = wasm_u32x4_extract_lane(vidx, 0);
    v128_t vl = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx0));

    const uint32_t vidx1 = wasm_u32x4_extract_lane(vidx, 1);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx1), vl, 1);

    const uint32_t vidx2 = wasm_u32x4_extract_lane(vidx, 2);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx2), vl, 2);

    const uint32_t vidx3 = wasm_u32x4_extract_lane(vidx, 3);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx3), vl, 3);

    v128_t vs = wasm_i32x4_add(vl, ven);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(vz, wasm_f32x4_mul(vn, vminus_ln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vminus_ln2_lo));

    v128_t vp = wasm_f32x4_add(vc2, wasm_f32x4_mul(vc3, vt));
    vp = wasm_f32x4_mul(vp, vt);

    vt = wasm_f32x4_mul(vt, vs);
    vs = wasm_f32x4_sub(vs, vone);
    vp = wasm_f32x4_add(vt, wasm_f32x4_mul(vp, vt));
    const v128_t ve = wasm_f32x4_mul(wasm_f32x4_add(vp, vs), valpha);

    const v128_t vsignm = wasm_i32x4_shr(vx, 31);
    vx = wasm_f32x4_mul(vx, vbeta);
    const v128_t vy = wasm_i32x4_relaxed_laneselect(ve, vx, vsignm);

    wasm_v128_store(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    v128_t vx = wasm_v128_load(input);

    const v128_t vz = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx, vprescale));

    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz, vlog2e));
    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const v128_t ven = wasm_i32x4_shl(vn, 19);

    const uint32_t vidx0 = wasm_u32x4_extract_lane(vidx, 0);
    v128_t vl = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx0));

    const uint32_t vidx1 = wasm_u32x4_extract_lane(vidx, 1);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx1), vl, 1);

    const uint32_t vidx2 = wasm_u32x4_extract_lane(vidx, 2);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx2), vl, 2);

    const uint32_t vidx3 = wasm_u32x4_extract_lane(vidx, 3);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx3), vl, 3);

    v128_t vs = wasm_i32x4_add(vl, ven);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(vz, wasm_f32x4_mul(vn, vminus_ln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vminus_ln2_lo));

    v128_t vp = wasm_f32x4_add(vc2, wasm_f32x4_mul(vc3, vt));
    vp = wasm_f32x4_mul(vp, vt);

    vt = wasm_f32x4_mul(vt, vs);
    vs = wasm_f32x4_sub(vs, vone);
    vp = wasm_f32x4_add(vt, wasm_f32x4_mul(vp, vt));
    const v128_t ve = wasm_f32x4_mul(wasm_f32x4_add(vp, vs), valpha);

    const v128_t vsignm = wasm_i32x4_shr(vx, 31);
    vx = wasm_f32x4_mul(vx, vbeta);
    v128_t vy = wasm_i32x4_relaxed_laneselect(ve, vx, vsignm);

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vy, 0);
      vy = wasm_v64x2_shuffle(vy, vy, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vy, 0);
    }
  }
}
