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


void xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4(
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

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    const v128_t vz = wasm_f32x4_abs(vx);

    v128_t vn = wasm_f32x4_relaxed_madd(vz, vminus_log2e, vmagic_bias);
    const v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vln2_hi, vz);
    vt = wasm_f32x4_relaxed_madd(vn, vln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vt, vc5, vc4);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc3);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc2);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc1);

    vt = wasm_f32x4_mul(vt, vs);
    const v128_t ve = wasm_f32x4_relaxed_madd(vt, vp, vs);
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

    v128_t vn = wasm_f32x4_relaxed_madd(vz, vminus_log2e, vmagic_bias);
    const v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vln2_hi, vz);
    vt = wasm_f32x4_relaxed_madd(vn, vln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vt, vc5, vc4);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc3);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc2);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc1);

    vt = wasm_f32x4_mul(vt, vs);
    const v128_t ve = wasm_f32x4_relaxed_madd(vt, vp, vs);
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
