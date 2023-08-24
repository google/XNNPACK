// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_64[64];

void xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vmagic_bias = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.magic_bias);
  const v128_t vminus_log2e = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.minus_log2e);
  const v128_t vindex_mask = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.index_mask);
  const v128_t vln2_hi = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.ln2_hi);
  const v128_t vln2_lo = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.ln2_lo);
  const v128_t vc2 = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.c2);
  const v128_t vone = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.one);
  const v128_t vdenorm_cutoff = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.denorm_cutoff);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    const v128_t vz = wasm_f32x4_abs(vx);

    v128_t vn = wasm_f32x4_relaxed_madd(vz, vminus_log2e, vmagic_bias);
    const v128_t ve = wasm_i32x4_shl(vn, 17);

    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);

    const uint32_t vidx0 = wasm_u32x4_extract_lane(vidx, 0);
    v128_t vl = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx0));

    const uint32_t vidx1 = wasm_u32x4_extract_lane(vidx, 1);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx1), vl, 1);

    const uint32_t vidx2 = wasm_u32x4_extract_lane(vidx, 2);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx2), vl, 2);

    const uint32_t vidx3 = wasm_u32x4_extract_lane(vidx, 3);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx3), vl, 3);

    const v128_t vs = wasm_i32x4_add(vl, ve);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vln2_hi, vz);
    vt = wasm_f32x4_relaxed_madd(vn, vln2_lo, vt);

    v128_t vp = wasm_f32x4_mul(vt, vc2);
    vp = wasm_f32x4_relaxed_nmadd(vp, vt, vt);

    const v128_t vy = wasm_f32x4_relaxed_nmadd(vs, vp, vs);
    const v128_t vd = wasm_f32x4_add(vy, vone);

    v128_t vf = wasm_f32x4_div(vy, vd);
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));
    const v128_t vcf = wasm_f32x4_sub(vone, vf);
    vf = wasm_i32x4_relaxed_laneselect(vf, vcf, wasm_i32x4_shr(vx, 31));

    wasm_v128_store(output, vf);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t vx = wasm_v128_load(input);

    const v128_t vz = wasm_f32x4_abs(vx);

    v128_t vn = wasm_f32x4_relaxed_madd(vz, vminus_log2e, vmagic_bias);
    const v128_t ve = wasm_i32x4_shl(vn, 17);

    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);

    const uint32_t vidx0 = wasm_u32x4_extract_lane(vidx, 0);
    v128_t vl = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx0));

    const uint32_t vidx1 = wasm_u32x4_extract_lane(vidx, 1);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx1), vl, 1);

    const uint32_t vidx2 = wasm_u32x4_extract_lane(vidx, 2);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx2), vl, 2);

    const uint32_t vidx3 = wasm_u32x4_extract_lane(vidx, 3);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx3), vl, 3);

    const v128_t vs = wasm_i32x4_add(vl, ve);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vln2_hi, vz);
    vt = wasm_f32x4_relaxed_madd(vn, vln2_lo, vt);

    v128_t vp = wasm_f32x4_mul(vt, vc2);
    vp = wasm_f32x4_relaxed_nmadd(vp, vt, vt);

    const v128_t vy = wasm_f32x4_relaxed_nmadd(vs, vp, vs);
    const v128_t vd = wasm_f32x4_add(vy, vone);

    v128_t vf = wasm_f32x4_div(vy, vd);
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));
    const v128_t vcf = wasm_f32x4_sub(vone, vf);
    vf = wasm_i32x4_relaxed_laneselect(vf, vcf, wasm_i32x4_shr(vx, 31));

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vf, 0);
      vf = wasm_v64x2_shuffle(vf, vf, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vf, 0);
    }
  }
}
