// Auto-generated file. Do not edit!
//   Template: src/f32-velu/wasmsimd-rr2-p6.c.in
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


void xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const v128_t vprescale = wasm_v128_load32_splat(&params->scalar.prescale);
  const v128_t valpha = wasm_v128_load32_splat(&params->scalar.alpha);
  const v128_t vbeta = wasm_v128_load32_splat(&params->scalar.beta);

  const v128_t vsat_cutoff = wasm_f32x4_const_splat(-0x1.154246p+4f);
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.8000FEp23f);
  const v128_t vlog2e = wasm_f32x4_const_splat(0x1.715476p+0f);
  const v128_t vminus_ln2_hi = wasm_f32x4_const_splat(-0x1.62E440p-1f);
  const v128_t vminus_ln2_lo = wasm_f32x4_const_splat(0x1.0105C6p-21f);
  const v128_t vc6 = wasm_f32x4_const_splat(0x1.6b7338p-10f);
  const v128_t vc5 = wasm_f32x4_const_splat(0x1.12278Ep-7f);
  const v128_t vc4 = wasm_f32x4_const_splat(0x1.555716p-5f);
  const v128_t vc3 = wasm_f32x4_const_splat(0x1.5554B0p-3f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.FFFFFEp-2f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    v128_t vx0123 = wasm_v128_load(x);
    v128_t vx4567 = wasm_v128_load(x + 4);
    x += 8;

    const v128_t vz0123 = wasm_f32x4_max(wasm_f32x4_mul(vx0123, vprescale), vsat_cutoff);
    const v128_t vz4567 = wasm_f32x4_max(wasm_f32x4_mul(vx4567, vprescale), vsat_cutoff);

    v128_t vn0123 = wasm_f32x4_add(wasm_f32x4_mul(vz0123, vlog2e), vmagic_bias);
    v128_t vn4567 = wasm_f32x4_add(wasm_f32x4_mul(vz4567, vlog2e), vmagic_bias);

    v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);

    v128_t vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vn0123, vminus_ln2_hi), vz0123);
    v128_t vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vn4567, vminus_ln2_hi), vz4567);

    vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vn0123, vminus_ln2_lo), vt0123);
    vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vn4567, vminus_ln2_lo), vt4567);

    v128_t vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt0123), vc5);
    v128_t vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt4567), vc5);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vc4);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vc4);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vc3);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vc3);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vc2);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vc2);

    vp0123 = wasm_f32x4_mul(vp0123, vt0123);
    vp4567 = wasm_f32x4_mul(vp4567, vt4567);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vs0123 = wasm_f32x4_sub(vs0123, vone);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vs4567 = wasm_f32x4_sub(vs4567, vone);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vt0123);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vt4567);

    const v128_t ve0123 = wasm_f32x4_mul(wasm_f32x4_add(vp0123, vs0123), valpha);
    const v128_t ve4567 = wasm_f32x4_mul(wasm_f32x4_add(vp4567, vs4567), valpha);

    const v128_t vsignm0123 = wasm_i32x4_shr(vx0123, 31);
    vx0123 = wasm_f32x4_mul(vx0123, vbeta);
    const v128_t vsignm4567 = wasm_i32x4_shr(vx4567, 31);
    vx4567 = wasm_f32x4_mul(vx4567, vbeta);

    const v128_t vy0123 = wasm_v128_bitselect(ve0123, vx0123, vsignm0123);
    const v128_t vy4567 = wasm_v128_bitselect(ve4567, vx4567, vsignm4567);

    wasm_v128_store(y, vy0123);
    wasm_v128_store(y + 4, vy4567);
    y += 8;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    v128_t vx = wasm_v128_load(x);
    x += 4;

    const v128_t vz = wasm_f32x4_max(wasm_f32x4_mul(vx, vprescale), vsat_cutoff);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vlog2e), vmagic_bias);
    v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_hi), vz);
    vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_lo), vt);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt), vc5);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc4);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc3);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc2);
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

    const v128_t vz = wasm_f32x4_max(wasm_f32x4_mul(vx, vprescale), vsat_cutoff);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vlog2e), vmagic_bias);
    v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_hi), vz);
    vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_lo), vt);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt), vc5);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc4);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc3);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vp, vt), vc2);
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
