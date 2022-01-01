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

#include <xnnpack/vunary.h>
#include <xnnpack/common.h>


extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const v128_t vprescale = wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.prescale);
  const v128_t valpha = wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.alpha);
  const v128_t vbeta = wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.beta);
  const v128_t vsat_cutoff = wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.sat_cutoff);
  const v128_t vmagic_bias = wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.magic_bias);
  const v128_t vlog2e = wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.log2e);
  const v128_t vindex_mask =  wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.index_mask);
  const v128_t vminus_ln2_hi = wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.minus_ln2_hi);
  const v128_t vminus_ln2_lo = wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.minus_ln2_lo);
  const v128_t vc3 = wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.c3);
  const v128_t vc2 = wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.c2);
  const v128_t vone = wasm_v128_load64_splat(params->wasmsimd_rr2_lut16_p3.one);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    v128_t vx = wasm_v128_load(x);
    x += 4;

    const v128_t vz = wasm_f32x4_mul(vx, vprescale);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vlog2e), vmagic_bias);
    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const v128_t ven = wasm_i32x4_shl(vn, 19);

    const uint64_t vidx_lo = wasm_i64x2_extract_lane(vidx, 0);
    const uint64_t vidx_hi = wasm_i64x2_extract_lane(vidx, 1);
    const float vl0 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo));
    const float vl1 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32)));
    const float vl2 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi));
    const float vl3 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32)));
    const v128_t vl = wasm_f32x4_make(vl0, vl1, vl2, vl3);

    v128_t vs = wasm_i32x4_add(vl, ven);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_hi), vz);
    const v128_t vsatm = wasm_f32x4_le(vz, vsat_cutoff);
    vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_lo), vt);
    vs = wasm_v128_andnot(vs, vsatm);
    vt = wasm_v128_andnot(vt, vsatm);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc3, vt), vc2);
    vp = wasm_f32x4_mul(vp, vt);

    vt = wasm_f32x4_mul(vt, vs);
    vs = wasm_f32x4_sub(vs, vone);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vt);
    const v128_t ve = wasm_f32x4_mul(wasm_f32x4_add(vp, vs), valpha);

    const v128_t vsignm = wasm_i32x4_shr(vx, 31);
    vx = wasm_f32x4_mul(vx, vbeta);
    const v128_t vy = wasm_v128_bitselect(ve, vx, vsignm);

    wasm_v128_store(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    v128_t vx = wasm_v128_load(x);

    const v128_t vz = wasm_f32x4_mul(vx, vprescale);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vlog2e), vmagic_bias);
    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const v128_t ven = wasm_i32x4_shl(vn, 19);

    const uint64_t vidx_lo = wasm_i64x2_extract_lane(vidx, 0);
    const uint64_t vidx_hi = wasm_i64x2_extract_lane(vidx, 1);
    const float vl0 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo));
    const float vl1 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32)));
    const float vl2 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi));
    const float vl3 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32)));
    const v128_t vl = wasm_f32x4_make(vl0, vl1, vl2, vl3);

    v128_t vs = wasm_i32x4_add(vl, ven);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_hi), vz);
    const v128_t vsatm = wasm_f32x4_le(vz, vsat_cutoff);
    vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_lo), vt);
    vs = wasm_v128_andnot(vs, vsatm);
    vt = wasm_v128_andnot(vt, vsatm);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc3, vt), vc2);
    vp = wasm_f32x4_mul(vp, vt);

    vt = wasm_f32x4_mul(vt, vs);
    vs = wasm_f32x4_sub(vs, vone);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vt);
    const v128_t ve = wasm_f32x4_mul(wasm_f32x4_add(vp, vs), valpha);

    const v128_t vsignm = wasm_i32x4_shr(vx, 31);
    vx = wasm_f32x4_mul(vx, vbeta);
    v128_t vy = wasm_v128_bitselect(ve, vx, vsignm);

    if (n & (2 * sizeof(float))) {
      *((double*) y) = wasm_f64x2_extract_lane(vy, 0);
      vy = wasm_v32x4_shuffle(vy, vy, 2, 3, 2, 3);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      *y = wasm_f32x4_extract_lane(vy, 0);
    }
  }
}
