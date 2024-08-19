// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/raddstoreexpminusmax.h"


void xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const v128_t vlog2e = wasm_f32x4_const_splat(0x1.715476p+0f);
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.8000FEp23f);
  const v128_t vminus_ln2_hi = wasm_f32x4_const_splat(-0x1.62E400p-1f);
  const v128_t vminus_ln2_lo = wasm_f32x4_const_splat(-0x1.7F7D1Cp-20f);
  const v128_t vc5 = wasm_f32x4_const_splat(0x1.0F9F9Cp-7f);
  const v128_t vc4 = wasm_f32x4_const_splat(0x1.573A1Ap-5f);
  const v128_t vc3 = wasm_f32x4_const_splat(0x1.555A80p-3f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.FFFDC6p-2f);
  const v128_t vc1 = wasm_f32x4_const_splat(0x1.FFFFF6p-1f);
  const v128_t vdenorm_cutoff = wasm_f32x4_const_splat(-0x1.5D589Ep6f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const v128_t vi_max = wasm_v128_load32_splat(max);

  v128_t vacc0 = wasm_f32x4_const_splat(0.0f);
  for (; batch >= 20 * sizeof(float); batch -= 20 * sizeof(float)) {
    // Load 20 (5x4) inputs at a time.
    const v128_t vi0123 = wasm_v128_load(input);
    const v128_t vi4567 = wasm_v128_load(input + 4);
    const v128_t vi89AB = wasm_v128_load(input + 8);
    const v128_t viCDEF = wasm_v128_load(input + 12);
    const v128_t viGHIJ = wasm_v128_load(input + 16);
    input += 20;

    const v128_t vx0123 = wasm_f32x4_sub(vi0123, vi_max);
    const v128_t vx4567 = wasm_f32x4_sub(vi4567, vi_max);
    const v128_t vx89AB = wasm_f32x4_sub(vi89AB, vi_max);
    const v128_t vxCDEF = wasm_f32x4_sub(viCDEF, vi_max);
    const v128_t vxGHIJ = wasm_f32x4_sub(viGHIJ, vi_max);

    v128_t vn0123 = wasm_f32x4_relaxed_madd(vx0123, vlog2e, vmagic_bias);
    v128_t vn4567 = wasm_f32x4_relaxed_madd(vx4567, vlog2e, vmagic_bias);
    v128_t vn89AB = wasm_f32x4_relaxed_madd(vx89AB, vlog2e, vmagic_bias);
    v128_t vnCDEF = wasm_f32x4_relaxed_madd(vxCDEF, vlog2e, vmagic_bias);
    v128_t vnGHIJ = wasm_f32x4_relaxed_madd(vxGHIJ, vlog2e, vmagic_bias);

    const v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    const v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);
    const v128_t vs89AB = wasm_i32x4_shl(vn89AB, 23);
    const v128_t vsCDEF = wasm_i32x4_shl(vnCDEF, 23);
    const v128_t vsGHIJ = wasm_i32x4_shl(vnGHIJ, 23);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);
    vnCDEF = wasm_f32x4_sub(vnCDEF, vmagic_bias);
    vnGHIJ = wasm_f32x4_sub(vnGHIJ, vmagic_bias);

    v128_t vt0123 = wasm_f32x4_relaxed_madd(vn0123, vminus_ln2_hi, vx0123);
    v128_t vt4567 = wasm_f32x4_relaxed_madd(vn4567, vminus_ln2_hi, vx4567);
    v128_t vt89AB = wasm_f32x4_relaxed_madd(vn89AB, vminus_ln2_hi, vx89AB);
    v128_t vtCDEF = wasm_f32x4_relaxed_madd(vnCDEF, vminus_ln2_hi, vxCDEF);
    v128_t vtGHIJ = wasm_f32x4_relaxed_madd(vnGHIJ, vminus_ln2_hi, vxGHIJ);

    vt0123 = wasm_f32x4_relaxed_madd(vn0123, vminus_ln2_lo, vt0123);
    vt4567 = wasm_f32x4_relaxed_madd(vn4567, vminus_ln2_lo, vt4567);
    vt89AB = wasm_f32x4_relaxed_madd(vn89AB, vminus_ln2_lo, vt89AB);
    vtCDEF = wasm_f32x4_relaxed_madd(vnCDEF, vminus_ln2_lo, vtCDEF);
    vtGHIJ = wasm_f32x4_relaxed_madd(vnGHIJ, vminus_ln2_lo, vtGHIJ);

    v128_t vp0123 = wasm_f32x4_relaxed_madd(vc5, vt0123, vc4);
    v128_t vp4567 = wasm_f32x4_relaxed_madd(vc5, vt4567, vc4);
    v128_t vp89AB = wasm_f32x4_relaxed_madd(vc5, vt89AB, vc4);
    v128_t vpCDEF = wasm_f32x4_relaxed_madd(vc5, vtCDEF, vc4);
    v128_t vpGHIJ = wasm_f32x4_relaxed_madd(vc5, vtGHIJ, vc4);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc3);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc3);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc3);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc3);
    vpGHIJ = wasm_f32x4_relaxed_madd(vpGHIJ, vtGHIJ, vc3);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc2);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc2);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc2);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc2);
    vpGHIJ = wasm_f32x4_relaxed_madd(vpGHIJ, vtGHIJ, vc2);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc1);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc1);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc1);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc1);
    vpGHIJ = wasm_f32x4_relaxed_madd(vpGHIJ, vtGHIJ, vc1);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vt89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    vtCDEF = wasm_f32x4_mul(vtCDEF, vsCDEF);
    vtGHIJ = wasm_f32x4_mul(vtGHIJ, vsGHIJ);

    v128_t vf0123 = wasm_f32x4_relaxed_madd(vt0123, vp0123, vs0123);
    v128_t vf4567 = wasm_f32x4_relaxed_madd(vt4567, vp4567, vs4567);
    v128_t vf89AB = wasm_f32x4_relaxed_madd(vt89AB, vp89AB, vs89AB);
    v128_t vfCDEF = wasm_f32x4_relaxed_madd(vtCDEF, vpCDEF, vsCDEF);
    v128_t vfGHIJ = wasm_f32x4_relaxed_madd(vtGHIJ, vpGHIJ, vsGHIJ);

    vf0123 = wasm_v128_andnot(vf0123, wasm_f32x4_lt(vx0123, vdenorm_cutoff));
    vf4567 = wasm_v128_andnot(vf4567, wasm_f32x4_lt(vx4567, vdenorm_cutoff));
    vf89AB = wasm_v128_andnot(vf89AB, wasm_f32x4_lt(vx89AB, vdenorm_cutoff));
    vfCDEF = wasm_v128_andnot(vfCDEF, wasm_f32x4_lt(vxCDEF, vdenorm_cutoff));
    vfGHIJ = wasm_v128_andnot(vfGHIJ, wasm_f32x4_lt(vxGHIJ, vdenorm_cutoff));

    wasm_v128_store(output, vf0123);
    wasm_v128_store(output + 4, vf4567);
    wasm_v128_store(output + 8, vf89AB);
    wasm_v128_store(output + 12, vfCDEF);
    wasm_v128_store(output + 16, vfGHIJ);
    output += 20;

    vacc0 = wasm_f32x4_add(vacc0, vf0123);
    vacc0 = wasm_f32x4_add(vacc0, vf4567);
    vacc0 = wasm_f32x4_add(vacc0, vf89AB);
    vacc0 = wasm_f32x4_add(vacc0, vfCDEF);
    vacc0 = wasm_f32x4_add(vacc0, vfGHIJ);
  }

  v128_t vacc = vacc0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vi = wasm_v128_load(input);
    input += 4;

    const v128_t vx = wasm_f32x4_sub(vi, vi_max);

    v128_t vn = wasm_f32x4_relaxed_madd(vx, vlog2e, vmagic_bias);

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_hi, vx);
    vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vc5, vt, vc4);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc3);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc2);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc1);

    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = wasm_f32x4_relaxed_madd(vt, vp, vs);

    vf = wasm_v128_andnot(vf, wasm_f32x4_lt(vx, vdenorm_cutoff));

    wasm_v128_store(output, vf);
    output += 4;

    vacc = wasm_f32x4_add(vacc, vf);
  }
  vacc = wasm_f32x4_add(vacc, wasm_v64x2_shuffle(vacc, vacc, 1, 1));
  float vsum = wasm_f32x4_extract_lane(vacc, 0) + wasm_f32x4_extract_lane(vacc, 1);
  if (batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));

    const v128_t vi = wasm_v128_load(input);

    const v128_t vx = wasm_f32x4_sub(vi, vi_max);

    v128_t vn = wasm_f32x4_relaxed_madd(vx, vlog2e, vmagic_bias);

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_hi, vx);
    vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vc5, vt, vc4);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc3);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc2);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc1);

    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = wasm_f32x4_relaxed_madd(vt, vp, vs);

    vf = wasm_v128_andnot(vf, wasm_f32x4_lt(vx, vdenorm_cutoff));

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vf, 0);
      output += 2;

      vsum += wasm_f32x4_extract_lane(vf, 0) + wasm_f32x4_extract_lane(vf, 1);
      vf = wasm_v64x2_shuffle(vf, vf, 1, 1);
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vf, 0);
      vsum += wasm_f32x4_extract_lane(vf, 0);
    }
  }
  *sum = vsum;
}
