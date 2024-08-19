// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"


extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];

void xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u12(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vsign_mask = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.sign_mask);
  const v128_t vsat_cutoff = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.sat_cutoff);
  const v128_t vlog2e = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.log2e);
  const v128_t vmagic_bias = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.magic_bias);
  const v128_t vindex_mask = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.index_mask);
  const v128_t vminus_ln2 = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.minus_ln2);
  const v128_t vc4 = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.c4);
  const v128_t vc3 = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.c3);
  const v128_t vc2 = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.c2);
  const v128_t vtwo = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.two);
  const v128_t vone = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.one);

  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    const v128_t vx0123 = wasm_v128_load(input);
    const v128_t vx4567 = wasm_v128_load(input + 4);
    const v128_t vx89AB = wasm_v128_load(input + 8);
    input += 12;

    v128_t vz0123 = wasm_v128_or(vx0123, vsign_mask);
    v128_t vz4567 = wasm_v128_or(vx4567, vsign_mask);
    v128_t vz89AB = wasm_v128_or(vx89AB, vsign_mask);

    const v128_t vinvsignx0123 = wasm_v128_xor(vx0123, vz0123);
    vz0123 = wasm_f32x4_pmax(vz0123, vsat_cutoff);
    const v128_t vinvsignx4567 = wasm_v128_xor(vx4567, vz4567);
    vz4567 = wasm_f32x4_pmax(vz4567, vsat_cutoff);
    const v128_t vinvsignx89AB = wasm_v128_xor(vx89AB, vz89AB);
    vz89AB = wasm_f32x4_pmax(vz89AB, vsat_cutoff);

    v128_t vn0123 = wasm_f32x4_add(wasm_f32x4_mul(vz0123, vlog2e), vmagic_bias);
    v128_t vn4567 = wasm_f32x4_add(wasm_f32x4_mul(vz4567, vlog2e), vmagic_bias);
    v128_t vn89AB = wasm_f32x4_add(wasm_f32x4_mul(vz89AB, vlog2e), vmagic_bias);

    const v128_t ve0123 = wasm_i32x4_shl(vn0123, 20);
    const v128_t vidx0123 = wasm_i32x4_shl(wasm_v128_and(vn0123, vindex_mask), 2);
    const v128_t ve4567 = wasm_i32x4_shl(vn4567, 20);
    const v128_t vidx4567 = wasm_i32x4_shl(wasm_v128_and(vn4567, vindex_mask), 2);
    const v128_t ve89AB = wasm_i32x4_shl(vn89AB, 20);
    const v128_t vidx89AB = wasm_i32x4_shl(wasm_v128_and(vn89AB, vindex_mask), 2);

    const uint64_t vidx01 = wasm_u64x2_extract_lane(vidx0123, 0);
    const uint64_t vidx45 = wasm_u64x2_extract_lane(vidx4567, 0);
    const uint64_t vidx89 = wasm_u64x2_extract_lane(vidx89AB, 0);

    v128_t vl0123 = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidx01));
    v128_t vl4567 = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidx45));
    v128_t vl89AB = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidx89));

    vl0123 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidx01 >> 32)), vl0123, 1);
    vl4567 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidx45 >> 32)), vl4567, 1);
    vl89AB = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidx89 >> 32)), vl89AB, 1);

    const uint64_t vidx23 = wasm_u64x2_extract_lane(vidx0123, 1);
    const uint64_t vidx67 = wasm_u64x2_extract_lane(vidx4567, 1);
    const uint64_t vidxAB = wasm_u64x2_extract_lane(vidx89AB, 1);

    vl0123 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidx23), vl0123, 2);
    vl4567 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidx67), vl4567, 2);
    vl89AB = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidxAB), vl89AB, 2);

    vl0123 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidx23 >> 32)), vl0123, 3);
    vl4567 = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidx67 >> 32)), vl4567, 3);
    vl89AB = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidxAB >> 32)), vl89AB, 3);

    const v128_t vs0123 = wasm_i32x4_add(vl0123, ve0123);
    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    const v128_t vs4567 = wasm_i32x4_add(vl4567, ve4567);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    const v128_t vs89AB = wasm_i32x4_add(vl89AB, ve89AB);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);

    const v128_t vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vn0123, vminus_ln2), vz0123);
    const v128_t vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vn4567, vminus_ln2), vz4567);
    const v128_t vt89AB = wasm_f32x4_add(wasm_f32x4_mul(vn89AB, vminus_ln2), vz89AB);

    v128_t vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vc4, vt0123), vc3);
    v128_t vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vc4, vt4567), vc3);
    v128_t vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vc4, vt89AB), vc3);
    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vc2);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vc2);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vt89AB), vc2);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vtwo);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vtwo);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vt89AB), vtwo);

    const v128_t vts0123 = wasm_f32x4_mul(vt0123, vs0123);
    const v128_t vsmo0123 = wasm_f32x4_sub(vs0123, vone);
    const v128_t vts4567 = wasm_f32x4_mul(vt4567, vs4567);
    const v128_t vsmo4567 = wasm_f32x4_sub(vs4567, vone);
    const v128_t vts89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    const v128_t vsmo89AB = wasm_f32x4_sub(vs89AB, vone);

    const v128_t vemo0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vts0123), vsmo0123);
    const v128_t vemo4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vts4567), vsmo4567);
    const v128_t vemo89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vts89AB), vsmo89AB);

    const v128_t vepo0123 = wasm_f32x4_add(vemo0123, vtwo);
    const v128_t vepo4567 = wasm_f32x4_add(vemo4567, vtwo);
    const v128_t vepo89AB = wasm_f32x4_add(vemo89AB, vtwo);

    v128_t vy0123 = wasm_f32x4_div(vemo0123, vepo0123);
    v128_t vy4567 = wasm_f32x4_div(vemo4567, vepo4567);
    v128_t vy89AB = wasm_f32x4_div(vemo89AB, vepo89AB);

    vy0123 = wasm_v128_xor(vy0123, vinvsignx0123);
    vy4567 = wasm_v128_xor(vy4567, vinvsignx4567);
    vy89AB = wasm_v128_xor(vy89AB, vinvsignx89AB);

    wasm_v128_store(output, vy0123);
    wasm_v128_store(output + 4, vy4567);
    wasm_v128_store(output + 8, vy89AB);
    output += 12;
  }

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    v128_t vz = wasm_v128_or(vx, vsign_mask);

    const v128_t vinvsignx = wasm_v128_xor(vx, vz);

    vz = wasm_f32x4_pmax(vz, vsat_cutoff);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vlog2e), vmagic_bias);

    const v128_t ve = wasm_i32x4_shl(vn, 20);

    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const uint64_t vidx_lo = wasm_u64x2_extract_lane(vidx, 0);
    v128_t vl = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidx_lo));
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidx_lo >> 32)), vl, 1);
    const uint64_t vidx_hi = wasm_u64x2_extract_lane(vidx, 1);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidx_hi), vl, 2);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidx_hi >> 32)), vl, 3);

    const v128_t vs = wasm_i32x4_add(vl, ve);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    const v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2), vz);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc4, vt), vc3);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc2);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vtwo);

    const v128_t vts = wasm_f32x4_mul(vt, vs);
    const v128_t vsmo = wasm_f32x4_sub(vs, vone);
    const v128_t vemo = wasm_f32x4_add(wasm_f32x4_mul(vp, vts), vsmo);

    const v128_t vepo = wasm_f32x4_add(vemo, vtwo);
    v128_t vy = wasm_f32x4_div(vemo, vepo);

    vy = wasm_v128_xor(vy, vinvsignx);

    wasm_v128_store(output, vy);
    output += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    const v128_t vx = wasm_v128_load(input);

    v128_t vz = wasm_v128_or(vx, vsign_mask);

    const v128_t vinvsignx = wasm_v128_xor(vx, vz);

    vz = wasm_f32x4_pmax(vz, vsat_cutoff);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vlog2e), vmagic_bias);

    const v128_t ve = wasm_i32x4_shl(vn, 20);

    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const uint64_t vidx_lo = wasm_u64x2_extract_lane(vidx, 0);
    v128_t vl = wasm_v128_load32_zero((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidx_lo));
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidx_lo >> 32)), vl, 1);
    const uint64_t vidx_hi = wasm_u64x2_extract_lane(vidx, 1);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) vidx_hi), vl, 2);
    vl = wasm_v128_load32_lane((const void*) ((uintptr_t) xnn_table_exp2minus_k_over_8 + (uint32_t) (vidx_hi >> 32)), vl, 3);

    const v128_t vs = wasm_i32x4_add(vl, ve);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    const v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2), vz);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc4, vt), vc3);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc2);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vtwo);

    const v128_t vts = wasm_f32x4_mul(vt, vs);
    const v128_t vsmo = wasm_f32x4_sub(vs, vone);
    const v128_t vemo = wasm_f32x4_add(wasm_f32x4_mul(vp, vts), vsmo);

    const v128_t vepo = wasm_f32x4_add(vemo, vtwo);
    v128_t vy = wasm_f32x4_div(vemo, vepo);

    vy = wasm_v128_xor(vy, vinvsignx);

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
