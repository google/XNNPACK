// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/wasmsimd-p5-div.c.in
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


void xnn_f32_vsigmoid_ukernel__wasmsimd_p5_div_x16(
    size_t n,
    const float* x,
    float* y,
    const void* params) XNN_DISABLE_TSAN
{
  assert(n % sizeof(float) == 0);

  const v128_t vmagic_bias = wasm_f32x4_splat(0x1.8000FEp23f);
  const v128_t vminus_log2e = wasm_f32x4_splat(-0x1.715476p+0f);
  const v128_t vln2_hi = wasm_f32x4_splat(0x1.62E400p-1f);
  const v128_t vln2_lo = wasm_f32x4_splat(0x1.7F7D1Cp-20f);
  const v128_t vc5 = wasm_f32x4_splat(-0x1.0F9F9Cp-7f);
  const v128_t vc4 = wasm_f32x4_splat( 0x1.573A1Ap-5f);
  const v128_t vc3 = wasm_f32x4_splat(-0x1.555A80p-3f);
  const v128_t vc2 = wasm_f32x4_splat( 0x1.FFFDC6p-2f);
  const v128_t vc1 = wasm_f32x4_splat(-0x1.FFFFF6p-1f);
  const v128_t vone = wasm_f32x4_splat(1.0f);
  const v128_t vdenorm_cutoff = wasm_f32x4_splat(0x1.5D589Ep+6f);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const v128_t vx0123 = wasm_v128_load(x);
    const v128_t vx4567 = wasm_v128_load(x + 4);
    const v128_t vx89AB = wasm_v128_load(x + 8);
    const v128_t vxCDEF = wasm_v128_load(x + 12);
    x += 16;

    const v128_t vz0123 = wasm_f32x4_abs(vx0123);
    const v128_t vz4567 = wasm_f32x4_abs(vx4567);
    const v128_t vz89AB = wasm_f32x4_abs(vx89AB);
    const v128_t vzCDEF = wasm_f32x4_abs(vxCDEF);

    v128_t vn0123 = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz0123, vminus_log2e));
    v128_t vn4567 = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz4567, vminus_log2e));
    v128_t vn89AB = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz89AB, vminus_log2e));
    v128_t vnCDEF = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vzCDEF, vminus_log2e));

    const v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    const v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);
    const v128_t vs89AB = wasm_i32x4_shl(vn89AB, 23);
    const v128_t vsCDEF = wasm_i32x4_shl(vnCDEF, 23);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);
    vnCDEF = wasm_f32x4_sub(vnCDEF, vmagic_bias);

    v128_t vt0123 = wasm_f32x4_add(vz0123, wasm_f32x4_mul(vn0123, vln2_hi));
    v128_t vt4567 = wasm_f32x4_add(vz4567, wasm_f32x4_mul(vn4567, vln2_hi));
    v128_t vt89AB = wasm_f32x4_add(vz89AB, wasm_f32x4_mul(vn89AB, vln2_hi));
    v128_t vtCDEF = wasm_f32x4_add(vzCDEF, wasm_f32x4_mul(vnCDEF, vln2_hi));

    vt0123 = wasm_f32x4_add(vt0123, wasm_f32x4_mul(vn0123, vln2_lo));
    vt4567 = wasm_f32x4_add(vt4567, wasm_f32x4_mul(vn4567, vln2_lo));
    vt89AB = wasm_f32x4_add(vt89AB, wasm_f32x4_mul(vn89AB, vln2_lo));
    vtCDEF = wasm_f32x4_add(vtCDEF, wasm_f32x4_mul(vnCDEF, vln2_lo));

    v128_t vp0123 = wasm_f32x4_add(vc4, wasm_f32x4_mul(vt0123, vc5));
    v128_t vp4567 = wasm_f32x4_add(vc4, wasm_f32x4_mul(vt4567, vc5));
    v128_t vp89AB = wasm_f32x4_add(vc4, wasm_f32x4_mul(vt89AB, vc5));
    v128_t vpCDEF = wasm_f32x4_add(vc4, wasm_f32x4_mul(vtCDEF, vc5));

    vp0123 = wasm_f32x4_add(vc3, wasm_f32x4_mul(vt0123, vp0123));
    vp4567 = wasm_f32x4_add(vc3, wasm_f32x4_mul(vt4567, vp4567));
    vp89AB = wasm_f32x4_add(vc3, wasm_f32x4_mul(vt89AB, vp89AB));
    vpCDEF = wasm_f32x4_add(vc3, wasm_f32x4_mul(vtCDEF, vpCDEF));

    vp0123 = wasm_f32x4_add(vc2, wasm_f32x4_mul(vt0123, vp0123));
    vp4567 = wasm_f32x4_add(vc2, wasm_f32x4_mul(vt4567, vp4567));
    vp89AB = wasm_f32x4_add(vc2, wasm_f32x4_mul(vt89AB, vp89AB));
    vpCDEF = wasm_f32x4_add(vc2, wasm_f32x4_mul(vtCDEF, vpCDEF));

    vp0123 = wasm_f32x4_add(vc1, wasm_f32x4_mul(vt0123, vp0123));
    vp4567 = wasm_f32x4_add(vc1, wasm_f32x4_mul(vt4567, vp4567));
    vp89AB = wasm_f32x4_add(vc1, wasm_f32x4_mul(vt89AB, vp89AB));
    vpCDEF = wasm_f32x4_add(vc1, wasm_f32x4_mul(vtCDEF, vpCDEF));

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vt89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    vtCDEF = wasm_f32x4_mul(vtCDEF, vsCDEF);

    const v128_t ve0123 = wasm_f32x4_add(vs0123, wasm_f32x4_mul(vt0123, vp0123));
    const v128_t ve4567 = wasm_f32x4_add(vs4567, wasm_f32x4_mul(vt4567, vp4567));
    const v128_t ve89AB = wasm_f32x4_add(vs89AB, wasm_f32x4_mul(vt89AB, vp89AB));
    const v128_t veCDEF = wasm_f32x4_add(vsCDEF, wasm_f32x4_mul(vtCDEF, vpCDEF));

    const v128_t vd0123 = wasm_f32x4_add(ve0123, vone);
    const v128_t vd4567 = wasm_f32x4_add(ve4567, vone);
    const v128_t vd89AB = wasm_f32x4_add(ve89AB, vone);
    const v128_t vdCDEF = wasm_f32x4_add(veCDEF, vone);

    v128_t vf0123 = wasm_f32x4_div(ve0123, vd0123);
    v128_t vf4567 = wasm_f32x4_div(ve4567, vd4567);
    v128_t vf89AB = wasm_f32x4_div(ve89AB, vd89AB);
    v128_t vfCDEF = wasm_f32x4_div(veCDEF, vdCDEF);

    vf0123 = wasm_v128_andnot(vf0123, wasm_f32x4_gt(vz0123, vdenorm_cutoff));
    vf4567 = wasm_v128_andnot(vf4567, wasm_f32x4_gt(vz4567, vdenorm_cutoff));
    vf89AB = wasm_v128_andnot(vf89AB, wasm_f32x4_gt(vz89AB, vdenorm_cutoff));
    vfCDEF = wasm_v128_andnot(vfCDEF, wasm_f32x4_gt(vzCDEF, vdenorm_cutoff));

    vf0123 = wasm_v128_bitselect(vf0123, wasm_f32x4_sub(vone, vf0123), wasm_i32x4_shr(vx0123, 31));
    vf4567 = wasm_v128_bitselect(vf4567, wasm_f32x4_sub(vone, vf4567), wasm_i32x4_shr(vx4567, 31));
    vf89AB = wasm_v128_bitselect(vf89AB, wasm_f32x4_sub(vone, vf89AB), wasm_i32x4_shr(vx89AB, 31));
    vfCDEF = wasm_v128_bitselect(vfCDEF, wasm_f32x4_sub(vone, vfCDEF), wasm_i32x4_shr(vxCDEF, 31));

    wasm_v128_store(y, vf0123);
    wasm_v128_store(y + 4, vf4567);
    wasm_v128_store(y + 8, vf89AB);
    wasm_v128_store(y + 12, vfCDEF);
    y += 16;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(x);
    x += 4;

    const v128_t vz = wasm_f32x4_abs(vx);

    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz, vminus_log2e));
    const v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(vz, wasm_f32x4_mul(vn, vln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vln2_lo));

    v128_t vp = wasm_f32x4_add(vc4, wasm_f32x4_mul(vt, vc5));
    vp = wasm_f32x4_add(vc3, wasm_f32x4_mul(vt, vp));
    vp = wasm_f32x4_add(vc2, wasm_f32x4_mul(vt, vp));
    vp = wasm_f32x4_add(vc1, wasm_f32x4_mul(vt, vp));

    vt = wasm_f32x4_mul(vt, vs);
    const v128_t ve = wasm_f32x4_add(vs, wasm_f32x4_mul(vt, vp));
    const v128_t vd = wasm_f32x4_add(ve, vone);

    v128_t vf = wasm_f32x4_div(ve, vd);
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));
    vf = wasm_v128_bitselect(vf, wasm_f32x4_sub(vone, vf), wasm_i32x4_shr(vx, 31));

    wasm_v128_store(y, vf);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const v128_t vx = wasm_v128_load(x);

    const v128_t vz = wasm_f32x4_abs(vx);

    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz, vminus_log2e));
    const v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(vz, wasm_f32x4_mul(vn, vln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vln2_lo));

    v128_t vp = wasm_f32x4_add(vc4, wasm_f32x4_mul(vt, vc5));
    vp = wasm_f32x4_add(vc3, wasm_f32x4_mul(vt, vp));
    vp = wasm_f32x4_add(vc2, wasm_f32x4_mul(vt, vp));
    vp = wasm_f32x4_add(vc1, wasm_f32x4_mul(vt, vp));

    vt = wasm_f32x4_mul(vt, vs);
    const v128_t ve = wasm_f32x4_add(vs, wasm_f32x4_mul(vt, vp));
    const v128_t vd = wasm_f32x4_add(ve, vone);

    v128_t vf = wasm_f32x4_div(ve, vd);
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
