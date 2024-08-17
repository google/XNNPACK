// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u24(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.8000FEp23f);
  const v128_t vminus_log2e = wasm_f32x4_const_splat(-0x1.715476p+0f);
  const v128_t vln2_hi = wasm_f32x4_const_splat(0x1.62E400p-1f);
  const v128_t vln2_lo = wasm_f32x4_const_splat(0x1.7F7D1Cp-20f);
  const v128_t vc5 = wasm_f32x4_const_splat(-0x1.0F9F9Cp-7f);
  const v128_t vc4 = wasm_f32x4_const_splat(0x1.573A1Ap-5f);
  const v128_t vc3 = wasm_f32x4_const_splat(-0x1.555A80p-3f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.FFFDC6p-2f);
  const v128_t vc1 = wasm_f32x4_const_splat(-0x1.FFFFF6p-1f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);
  const v128_t vdenorm_cutoff = wasm_f32x4_const_splat(0x1.5D589Ep+6f);

  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_log2e);
  XNN_FORCE_REALIZATION(vln2_hi);
  XNN_FORCE_REALIZATION(vln2_lo);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  // XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    const v128_t vx0123 = wasm_v128_load(input);
    const v128_t vx4567 = wasm_v128_load(input + 4);
    const v128_t vx89AB = wasm_v128_load(input + 8);
    const v128_t vxCDEF = wasm_v128_load(input + 12);
    const v128_t vxGHIJ = wasm_v128_load(input + 16);
    const v128_t vxKLMN = wasm_v128_load(input + 20);
    input += 24;

    const v128_t vz0123 = wasm_f32x4_abs(vx0123);
    const v128_t vz4567 = wasm_f32x4_abs(vx4567);
    const v128_t vz89AB = wasm_f32x4_abs(vx89AB);
    const v128_t vzCDEF = wasm_f32x4_abs(vxCDEF);
    const v128_t vzGHIJ = wasm_f32x4_abs(vxGHIJ);
    const v128_t vzKLMN = wasm_f32x4_abs(vxKLMN);

    v128_t vn0123 = wasm_f32x4_add(wasm_f32x4_mul(vz0123, vminus_log2e), vmagic_bias);
    v128_t vn4567 = wasm_f32x4_add(wasm_f32x4_mul(vz4567, vminus_log2e), vmagic_bias);
    v128_t vn89AB = wasm_f32x4_add(wasm_f32x4_mul(vz89AB, vminus_log2e), vmagic_bias);
    v128_t vnCDEF = wasm_f32x4_add(wasm_f32x4_mul(vzCDEF, vminus_log2e), vmagic_bias);
    v128_t vnGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vzGHIJ, vminus_log2e), vmagic_bias);
    v128_t vnKLMN = wasm_f32x4_add(wasm_f32x4_mul(vzKLMN, vminus_log2e), vmagic_bias);

    const v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    const v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);
    const v128_t vs89AB = wasm_i32x4_shl(vn89AB, 23);
    const v128_t vsCDEF = wasm_i32x4_shl(vnCDEF, 23);
    const v128_t vsGHIJ = wasm_i32x4_shl(vnGHIJ, 23);
    const v128_t vsKLMN = wasm_i32x4_shl(vnKLMN, 23);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);
    vnCDEF = wasm_f32x4_sub(vnCDEF, vmagic_bias);
    vnGHIJ = wasm_f32x4_sub(vnGHIJ, vmagic_bias);
    vnKLMN = wasm_f32x4_sub(vnKLMN, vmagic_bias);

    v128_t vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vn0123, vln2_hi), vz0123);
    v128_t vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vn4567, vln2_hi), vz4567);
    v128_t vt89AB = wasm_f32x4_add(wasm_f32x4_mul(vn89AB, vln2_hi), vz89AB);
    v128_t vtCDEF = wasm_f32x4_add(wasm_f32x4_mul(vnCDEF, vln2_hi), vzCDEF);
    v128_t vtGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vnGHIJ, vln2_hi), vzGHIJ);
    v128_t vtKLMN = wasm_f32x4_add(wasm_f32x4_mul(vnKLMN, vln2_hi), vzKLMN);

    vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vn0123, vln2_lo), vt0123);
    vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vn4567, vln2_lo), vt4567);
    vt89AB = wasm_f32x4_add(wasm_f32x4_mul(vn89AB, vln2_lo), vt89AB);
    vtCDEF = wasm_f32x4_add(wasm_f32x4_mul(vnCDEF, vln2_lo), vtCDEF);
    vtGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vnGHIJ, vln2_lo), vtGHIJ);
    vtKLMN = wasm_f32x4_add(wasm_f32x4_mul(vnKLMN, vln2_lo), vtKLMN);

    v128_t vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vt0123, vc5), vc4);
    v128_t vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vt4567, vc5), vc4);
    v128_t vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vt89AB, vc5), vc4);
    v128_t vpCDEF = wasm_f32x4_add(wasm_f32x4_mul(vtCDEF, vc5), vc4);
    v128_t vpGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vtGHIJ, vc5), vc4);
    v128_t vpKLMN = wasm_f32x4_add(wasm_f32x4_mul(vtKLMN, vc5), vc4);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vt0123, vp0123), vc3);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vt4567, vp4567), vc3);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vt89AB, vp89AB), vc3);
    vpCDEF = wasm_f32x4_add(wasm_f32x4_mul(vtCDEF, vpCDEF), vc3);
    vpGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vtGHIJ, vpGHIJ), vc3);
    vpKLMN = wasm_f32x4_add(wasm_f32x4_mul(vtKLMN, vpKLMN), vc3);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vt0123, vp0123), vc2);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vt4567, vp4567), vc2);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vt89AB, vp89AB), vc2);
    vpCDEF = wasm_f32x4_add(wasm_f32x4_mul(vtCDEF, vpCDEF), vc2);
    vpGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vtGHIJ, vpGHIJ), vc2);
    vpKLMN = wasm_f32x4_add(wasm_f32x4_mul(vtKLMN, vpKLMN), vc2);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vt0123, vp0123), vc1);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vt4567, vp4567), vc1);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vt89AB, vp89AB), vc1);
    vpCDEF = wasm_f32x4_add(wasm_f32x4_mul(vtCDEF, vpCDEF), vc1);
    vpGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vtGHIJ, vpGHIJ), vc1);
    vpKLMN = wasm_f32x4_add(wasm_f32x4_mul(vtKLMN, vpKLMN), vc1);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vt89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    vtCDEF = wasm_f32x4_mul(vtCDEF, vsCDEF);
    vtGHIJ = wasm_f32x4_mul(vtGHIJ, vsGHIJ);
    vtKLMN = wasm_f32x4_mul(vtKLMN, vsKLMN);

    const v128_t ve0123 = wasm_f32x4_add(wasm_f32x4_mul(vt0123, vp0123), vs0123);
    const v128_t ve4567 = wasm_f32x4_add(wasm_f32x4_mul(vt4567, vp4567), vs4567);
    const v128_t ve89AB = wasm_f32x4_add(wasm_f32x4_mul(vt89AB, vp89AB), vs89AB);
    const v128_t veCDEF = wasm_f32x4_add(wasm_f32x4_mul(vtCDEF, vpCDEF), vsCDEF);
    const v128_t veGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vtGHIJ, vpGHIJ), vsGHIJ);
    const v128_t veKLMN = wasm_f32x4_add(wasm_f32x4_mul(vtKLMN, vpKLMN), vsKLMN);

    const v128_t vd0123 = wasm_f32x4_add(ve0123, vone);
    const v128_t vd4567 = wasm_f32x4_add(ve4567, vone);
    const v128_t vd89AB = wasm_f32x4_add(ve89AB, vone);
    const v128_t vdCDEF = wasm_f32x4_add(veCDEF, vone);
    const v128_t vdGHIJ = wasm_f32x4_add(veGHIJ, vone);
    const v128_t vdKLMN = wasm_f32x4_add(veKLMN, vone);

    v128_t vf0123 = wasm_f32x4_div(ve0123, vd0123);
    v128_t vf4567 = wasm_f32x4_div(ve4567, vd4567);
    v128_t vf89AB = wasm_f32x4_div(ve89AB, vd89AB);
    v128_t vfCDEF = wasm_f32x4_div(veCDEF, vdCDEF);
    v128_t vfGHIJ = wasm_f32x4_div(veGHIJ, vdGHIJ);
    v128_t vfKLMN = wasm_f32x4_div(veKLMN, vdKLMN);

    vf0123 = wasm_v128_andnot(vf0123, wasm_f32x4_gt(vz0123, vdenorm_cutoff));
    vf4567 = wasm_v128_andnot(vf4567, wasm_f32x4_gt(vz4567, vdenorm_cutoff));
    vf89AB = wasm_v128_andnot(vf89AB, wasm_f32x4_gt(vz89AB, vdenorm_cutoff));
    vfCDEF = wasm_v128_andnot(vfCDEF, wasm_f32x4_gt(vzCDEF, vdenorm_cutoff));
    vfGHIJ = wasm_v128_andnot(vfGHIJ, wasm_f32x4_gt(vzGHIJ, vdenorm_cutoff));
    vfKLMN = wasm_v128_andnot(vfKLMN, wasm_f32x4_gt(vzKLMN, vdenorm_cutoff));

    const v128_t vcf0123 = wasm_f32x4_sub(vone, vf0123);
    const v128_t vcf4567 = wasm_f32x4_sub(vone, vf4567);
    const v128_t vcf89AB = wasm_f32x4_sub(vone, vf89AB);
    const v128_t vcfCDEF = wasm_f32x4_sub(vone, vfCDEF);
    const v128_t vcfGHIJ = wasm_f32x4_sub(vone, vfGHIJ);
    const v128_t vcfKLMN = wasm_f32x4_sub(vone, vfKLMN);

    vf0123 = wasm_i32x4_relaxed_laneselect(vf0123, vcf0123, vx0123);
    vf4567 = wasm_i32x4_relaxed_laneselect(vf4567, vcf4567, vx4567);
    vf89AB = wasm_i32x4_relaxed_laneselect(vf89AB, vcf89AB, vx89AB);
    vfCDEF = wasm_i32x4_relaxed_laneselect(vfCDEF, vcfCDEF, vxCDEF);
    vfGHIJ = wasm_i32x4_relaxed_laneselect(vfGHIJ, vcfGHIJ, vxGHIJ);
    vfKLMN = wasm_i32x4_relaxed_laneselect(vfKLMN, vcfKLMN, vxKLMN);

    wasm_v128_store(output, vf0123);
    wasm_v128_store(output + 4, vf4567);
    wasm_v128_store(output + 8, vf89AB);
    wasm_v128_store(output + 12, vfCDEF);
    wasm_v128_store(output + 16, vfGHIJ);
    wasm_v128_store(output + 20, vfKLMN);
    output += 24;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    const v128_t vz = wasm_f32x4_abs(vx);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vminus_log2e), vmagic_bias);
    const v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vln2_hi), vz);
    vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vln2_lo), vt);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vt, vc5), vc4);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vt, vp), vc3);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vt, vp), vc2);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vt, vp), vc1);

    vt = wasm_f32x4_mul(vt, vs);
    const v128_t ve = wasm_f32x4_add(wasm_f32x4_mul(vt, vp), vs);
    const v128_t vd = wasm_f32x4_add(ve, vone);

    v128_t vf = wasm_f32x4_div(ve, vd);
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));
    const v128_t vcf = wasm_f32x4_sub(vone, vf);
    vf = wasm_i32x4_relaxed_laneselect(vf, vcf, vx);

    wasm_v128_store(output, vf);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t vx = wasm_v128_load(input);

    const v128_t vz = wasm_f32x4_abs(vx);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vminus_log2e), vmagic_bias);
    const v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vln2_hi), vz);
    vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vln2_lo), vt);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vt, vc5), vc4);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vt, vp), vc3);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vt, vp), vc2);
    vp = wasm_f32x4_add(wasm_f32x4_mul(vt, vp), vc1);

    vt = wasm_f32x4_mul(vt, vs);
    const v128_t ve = wasm_f32x4_add(wasm_f32x4_mul(vt, vp), vs);
    const v128_t vd = wasm_f32x4_add(ve, vone);

    v128_t vf = wasm_f32x4_div(ve, vd);
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));
    const v128_t vcf = wasm_f32x4_sub(vone, vf);
    vf = wasm_i32x4_relaxed_laneselect(vf, vcf, vx);

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
