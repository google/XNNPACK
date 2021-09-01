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


void xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x16(
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

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    v128_t vx0123 = wasm_v128_load(x);
    v128_t vx4567 = wasm_v128_load(x + 4);
    v128_t vx89AB = wasm_v128_load(x + 8);
    v128_t vxCDEF = wasm_v128_load(x + 12);
    x += 16;

    const v128_t vz0123 = wasm_f32x4_max(wasm_f32x4_mul(vx0123, vprescale), vsat_cutoff);
    const v128_t vz4567 = wasm_f32x4_max(wasm_f32x4_mul(vx4567, vprescale), vsat_cutoff);
    const v128_t vz89AB = wasm_f32x4_max(wasm_f32x4_mul(vx89AB, vprescale), vsat_cutoff);
    const v128_t vzCDEF = wasm_f32x4_max(wasm_f32x4_mul(vxCDEF, vprescale), vsat_cutoff);

    v128_t vn0123 = wasm_f32x4_add(wasm_f32x4_mul(vz0123, vlog2e), vmagic_bias);
    v128_t vn4567 = wasm_f32x4_add(wasm_f32x4_mul(vz4567, vlog2e), vmagic_bias);
    v128_t vn89AB = wasm_f32x4_add(wasm_f32x4_mul(vz89AB, vlog2e), vmagic_bias);
    v128_t vnCDEF = wasm_f32x4_add(wasm_f32x4_mul(vzCDEF, vlog2e), vmagic_bias);

    v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);
    v128_t vs89AB = wasm_i32x4_shl(vn89AB, 23);
    v128_t vsCDEF = wasm_i32x4_shl(vnCDEF, 23);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);
    vnCDEF = wasm_f32x4_sub(vnCDEF, vmagic_bias);

    v128_t vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vn0123, vminus_ln2_hi), vz0123);
    v128_t vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vn4567, vminus_ln2_hi), vz4567);
    v128_t vt89AB = wasm_f32x4_add(wasm_f32x4_mul(vn89AB, vminus_ln2_hi), vz89AB);
    v128_t vtCDEF = wasm_f32x4_add(wasm_f32x4_mul(vnCDEF, vminus_ln2_hi), vzCDEF);

    vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vn0123, vminus_ln2_lo), vt0123);
    vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vn4567, vminus_ln2_lo), vt4567);
    vt89AB = wasm_f32x4_add(wasm_f32x4_mul(vn89AB, vminus_ln2_lo), vt89AB);
    vtCDEF = wasm_f32x4_add(wasm_f32x4_mul(vnCDEF, vminus_ln2_lo), vtCDEF);

    v128_t vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt0123), vc5);
    v128_t vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt4567), vc5);
    v128_t vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vc6, vt89AB), vc5);
    v128_t vpCDEF = wasm_f32x4_add(wasm_f32x4_mul(vc6, vtCDEF), vc5);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vc4);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vc4);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vt89AB), vc4);
    vpCDEF = wasm_f32x4_add(wasm_f32x4_mul(vpCDEF, vtCDEF), vc4);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vc3);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vc3);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vt89AB), vc3);
    vpCDEF = wasm_f32x4_add(wasm_f32x4_mul(vpCDEF, vtCDEF), vc3);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vc2);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vc2);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vt89AB), vc2);
    vpCDEF = wasm_f32x4_add(wasm_f32x4_mul(vpCDEF, vtCDEF), vc2);

    vp0123 = wasm_f32x4_mul(vp0123, vt0123);
    vp4567 = wasm_f32x4_mul(vp4567, vt4567);
    vp89AB = wasm_f32x4_mul(vp89AB, vt89AB);
    vpCDEF = wasm_f32x4_mul(vpCDEF, vtCDEF);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vs0123 = wasm_f32x4_sub(vs0123, vone);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vs4567 = wasm_f32x4_sub(vs4567, vone);
    vt89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    vs89AB = wasm_f32x4_sub(vs89AB, vone);
    vtCDEF = wasm_f32x4_mul(vtCDEF, vsCDEF);
    vsCDEF = wasm_f32x4_sub(vsCDEF, vone);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vt0123);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vt4567);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vt89AB), vt89AB);
    vpCDEF = wasm_f32x4_add(wasm_f32x4_mul(vpCDEF, vtCDEF), vtCDEF);

    const v128_t ve0123 = wasm_f32x4_mul(wasm_f32x4_add(vp0123, vs0123), valpha);
    const v128_t ve4567 = wasm_f32x4_mul(wasm_f32x4_add(vp4567, vs4567), valpha);
    const v128_t ve89AB = wasm_f32x4_mul(wasm_f32x4_add(vp89AB, vs89AB), valpha);
    const v128_t veCDEF = wasm_f32x4_mul(wasm_f32x4_add(vpCDEF, vsCDEF), valpha);

    const v128_t vsignm0123 = wasm_i32x4_shr(vx0123, 31);
    vx0123 = wasm_f32x4_mul(vx0123, vbeta);
    const v128_t vsignm4567 = wasm_i32x4_shr(vx4567, 31);
    vx4567 = wasm_f32x4_mul(vx4567, vbeta);
    const v128_t vsignm89AB = wasm_i32x4_shr(vx89AB, 31);
    vx89AB = wasm_f32x4_mul(vx89AB, vbeta);
    const v128_t vsignmCDEF = wasm_i32x4_shr(vxCDEF, 31);
    vxCDEF = wasm_f32x4_mul(vxCDEF, vbeta);

    const v128_t vy0123 = wasm_v128_bitselect(ve0123, vx0123, vsignm0123);
    const v128_t vy4567 = wasm_v128_bitselect(ve4567, vx4567, vsignm4567);
    const v128_t vy89AB = wasm_v128_bitselect(ve89AB, vx89AB, vsignm89AB);
    const v128_t vyCDEF = wasm_v128_bitselect(veCDEF, vxCDEF, vsignmCDEF);

    wasm_v128_store(y, vy0123);
    wasm_v128_store(y + 4, vy4567);
    wasm_v128_store(y + 8, vy89AB);
    wasm_v128_store(y + 12, vyCDEF);
    y += 16;
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
