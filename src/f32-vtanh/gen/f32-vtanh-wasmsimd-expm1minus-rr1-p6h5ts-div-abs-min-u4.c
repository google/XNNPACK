// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/wasmsimd-expm1minus-abs.c.in
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


void xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vsat_cutoff = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.sat_cutoff);
  const v128_t vminus_log2e = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.minus_log2e);
  const v128_t vmagic_bias = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.magic_bias);
  const v128_t vln2 = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.ln2);
  const v128_t vc6 = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.c6);
  const v128_t vc5 = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.c5);
  const v128_t vc4 = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.c4);
  const v128_t vc3 = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.c3);
  const v128_t vc2 = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.c2);
  const v128_t vminus_two = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.minus_two);
  const v128_t vone = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.one);
  const v128_t vsign_mask = wasm_v128_load64_splat(params->wasmsimd_expm1minus_rr1_p6h5_abs.sign_mask);


  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    v128_t vz = wasm_f32x4_abs(vx);

    vz = wasm_f32x4_min(vz, vsat_cutoff);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vminus_log2e), vmagic_bias);

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    const v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vln2), vz);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt), vc5);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc4);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc3);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc2);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vminus_two);

    const v128_t vts = wasm_f32x4_mul(vt, vs);
    const v128_t vsmo = wasm_f32x4_sub(vs, vone);
    const v128_t vemo = wasm_f32x4_add(wasm_f32x4_mul(vp, vts), vsmo);

    const v128_t vepo = wasm_f32x4_sub(vemo, vminus_two);
    v128_t vy = wasm_f32x4_div(vemo, vepo);

    vy = wasm_v128_bitselect(vx, vy, vsign_mask);

    wasm_v128_store(output, vy);
    output += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    const v128_t vx = wasm_v128_load(input);

    v128_t vz = wasm_f32x4_abs(vx);

    vz = wasm_f32x4_min(vz, vsat_cutoff);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vminus_log2e), vmagic_bias);

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    const v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vln2), vz);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt), vc5);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc4);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc3);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc2);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vminus_two);

    const v128_t vts = wasm_f32x4_mul(vt, vs);
    const v128_t vsmo = wasm_f32x4_sub(vs, vone);
    const v128_t vemo = wasm_f32x4_add(wasm_f32x4_mul(vp, vts), vsmo);

    const v128_t vepo = wasm_f32x4_sub(vemo, vminus_two);
    v128_t vy = wasm_f32x4_div(vemo, vepo);

    vy = wasm_v128_bitselect(vx, vy, vsign_mask);

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
