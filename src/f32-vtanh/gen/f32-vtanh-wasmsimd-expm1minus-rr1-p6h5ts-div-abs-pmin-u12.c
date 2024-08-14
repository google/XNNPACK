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


void xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u12(
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

  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    const v128_t vx0123 = wasm_v128_load(input);
    const v128_t vx4567 = wasm_v128_load(input + 4);
    const v128_t vx89AB = wasm_v128_load(input + 8);
    input += 12;

    v128_t vz0123 = wasm_f32x4_abs(vx0123);
    v128_t vz4567 = wasm_f32x4_abs(vx4567);
    v128_t vz89AB = wasm_f32x4_abs(vx89AB);

    vz0123 = wasm_f32x4_pmin(vz0123, vsat_cutoff);
    vz4567 = wasm_f32x4_pmin(vz4567, vsat_cutoff);
    vz89AB = wasm_f32x4_pmin(vz89AB, vsat_cutoff);

    v128_t vn0123 = wasm_f32x4_add(wasm_f32x4_mul(vz0123, vminus_log2e), vmagic_bias);
    v128_t vn4567 = wasm_f32x4_add(wasm_f32x4_mul(vz4567, vminus_log2e), vmagic_bias);
    v128_t vn89AB = wasm_f32x4_add(wasm_f32x4_mul(vz89AB, vminus_log2e), vmagic_bias);

    const v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    const v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    const v128_t vs89AB = wasm_i32x4_shl(vn89AB, 23);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);


    const v128_t vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vn0123, vln2), vz0123);
    const v128_t vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vn4567, vln2), vz4567);
    const v128_t vt89AB = wasm_f32x4_add(wasm_f32x4_mul(vn89AB, vln2), vz89AB);

    v128_t vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt0123), vc5);
    v128_t vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt4567), vc5);
    v128_t vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt89AB), vc5);
    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vc4);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vc4);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vt89AB), vc4);
    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vc3);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vc3);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vt89AB), vc3);
    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vc2);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vc2);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vt89AB), vc2);
    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vminus_two);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vminus_two);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vt89AB), vminus_two);

    const v128_t vts0123 = wasm_f32x4_mul(vt0123, vs0123);
    const v128_t vsmo0123 = wasm_f32x4_sub(vs0123, vone);
    const v128_t vts4567 = wasm_f32x4_mul(vt4567, vs4567);
    const v128_t vsmo4567 = wasm_f32x4_sub(vs4567, vone);
    const v128_t vts89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    const v128_t vsmo89AB = wasm_f32x4_sub(vs89AB, vone);

    const v128_t vemo0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vts0123), vsmo0123);
    const v128_t vemo4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vts4567), vsmo4567);
    const v128_t vemo89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vts89AB), vsmo89AB);

    const v128_t vepo0123 = wasm_f32x4_sub(vemo0123, vminus_two);
    const v128_t vepo4567 = wasm_f32x4_sub(vemo4567, vminus_two);
    const v128_t vepo89AB = wasm_f32x4_sub(vemo89AB, vminus_two);

    v128_t vy0123 = wasm_f32x4_div(vemo0123, vepo0123);
    v128_t vy4567 = wasm_f32x4_div(vemo4567, vepo4567);
    v128_t vy89AB = wasm_f32x4_div(vemo89AB, vepo89AB);

    vy0123 = wasm_v128_bitselect(vx0123, vy0123, vsign_mask);
    vy4567 = wasm_v128_bitselect(vx4567, vy4567, vsign_mask);
    vy89AB = wasm_v128_bitselect(vx89AB, vy89AB, vsign_mask);

    wasm_v128_store(output, vy0123);
    wasm_v128_store(output + 4, vy4567);
    wasm_v128_store(output + 8, vy89AB);
    output += 12;
  }

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    v128_t vz = wasm_f32x4_abs(vx);

    vz = wasm_f32x4_pmin(vz, vsat_cutoff);

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

    vz = wasm_f32x4_pmin(vz, vsat_cutoff);

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
