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

void xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n % sizeof(float) == 0);

  const v128_t vmagic_bias = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.magic_bias);
  const v128_t vminus_log2e = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.minus_log2e);
  const v128_t vindex_mask = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.index_mask);
  const v128_t vln2_hi = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.ln2_hi);
  const v128_t vln2_lo = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.ln2_lo);
  const v128_t vc2 = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.c2);
  const v128_t vone = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.one);
  const v128_t vdenorm_cutoff = wasm_v128_load64_splat(params->wasmsimd_rr2_lut64_p2.denorm_cutoff);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(x);
    x += 4;

    const v128_t vz = wasm_f32x4_abs(vx);

    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz, vminus_log2e));
    const v128_t ve = wasm_i32x4_shl(vn, 17);

    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const uint64_t vidx_lo = wasm_i64x2_extract_lane(vidx, 0);
    const uint64_t vidx_hi = wasm_i64x2_extract_lane(vidx, 1);
    const float vl0 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_lo));
    const float vl1 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_lo >> 32)));
    const float vl2 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_hi));
    const float vl3 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_hi >> 32)));
    const v128_t vl = wasm_f32x4_make(vl0, vl1, vl2, vl3);

    const v128_t vs = wasm_i32x4_add(vl, ve);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(vz, wasm_f32x4_mul(vn, vln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vln2_lo));

    v128_t vp = wasm_f32x4_mul(vt, vc2);
    vp = wasm_f32x4_sub(vt, wasm_f32x4_mul(vp, vt));

    const v128_t vy = wasm_f32x4_sub(vs, wasm_f32x4_mul(vs, vp));
    const v128_t vd = wasm_f32x4_add(vy, vone);

    v128_t vf = wasm_f32x4_div(vy, vd);
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));
    vf = wasm_v128_bitselect(vf, wasm_f32x4_sub(vone, vf), wasm_i32x4_shr(vx, 31));

    wasm_v128_store(y, vf);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const v128_t vx = wasm_v128_load(x);

    const v128_t vz = wasm_f32x4_abs(vx);

    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz, vminus_log2e));
    const v128_t ve = wasm_i32x4_shl(vn, 17);

    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const uint64_t vidx_lo = wasm_i64x2_extract_lane(vidx, 0);
    const uint64_t vidx_hi = wasm_i64x2_extract_lane(vidx, 1);
    const float vl0 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_lo));
    const float vl1 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_lo >> 32)));
    const float vl2 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_hi));
    const float vl3 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_hi >> 32)));
    const v128_t vl = wasm_f32x4_make(vl0, vl1, vl2, vl3);

    const v128_t vs = wasm_i32x4_add(vl, ve);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(vz, wasm_f32x4_mul(vn, vln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vln2_lo));

    v128_t vp = wasm_f32x4_mul(vt, vc2);
    vp = wasm_f32x4_sub(vt, wasm_f32x4_mul(vp, vt));

    const v128_t vy = wasm_f32x4_sub(vs, wasm_f32x4_mul(vs, vp));
    const v128_t vd = wasm_f32x4_add(vy, vone);

    v128_t vf = wasm_f32x4_div(vy, vd);
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));
    vf = wasm_v128_bitselect(vf, wasm_f32x4_sub(vone, vf), wasm_i32x4_shr(vx, 31));

    if (n & (2 * sizeof(float))) {
      *((double*) y) = wasm_f64x2_extract_lane(vf, 0);
      vf = wasm_v32x4_shuffle(vf, vf, 2, 3, 2, 3);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      *y = wasm_f32x4_extract_lane(vf, 0);
    }
  }
}
