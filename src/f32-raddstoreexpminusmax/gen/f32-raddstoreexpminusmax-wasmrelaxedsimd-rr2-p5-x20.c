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

#include <xnnpack/common.h>
#include <xnnpack/raddstoreexpminusmax.h>


void xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x20(
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

  const v128_t vi_max = wasm_v128_load32_splat(max);
  const v128_t vlog2e = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.log2e);
  const v128_t vmagic_bias = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.magic_bias);
  const v128_t vminus_ln2_hi = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.minus_ln2_hi);
  const v128_t vminus_ln2_lo = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.minus_ln2_lo);
  const v128_t vc5 = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.c5);
  const v128_t vc4 = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.c4);
  const v128_t vc3 = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.c3);
  const v128_t vc2 = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.c2);
  const v128_t vc1 = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.c1);
  const v128_t vdenorm_cutoff = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.denorm_cutoff);

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

    v128_t vn0123 = __builtin_wasm_fma_f32x4(vmagic_bias, vx0123, vlog2e);
    v128_t vn4567 = __builtin_wasm_fma_f32x4(vmagic_bias, vx4567, vlog2e);
    v128_t vn89AB = __builtin_wasm_fma_f32x4(vmagic_bias, vx89AB, vlog2e);
    v128_t vnCDEF = __builtin_wasm_fma_f32x4(vmagic_bias, vxCDEF, vlog2e);
    v128_t vnGHIJ = __builtin_wasm_fma_f32x4(vmagic_bias, vxGHIJ, vlog2e);

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

    v128_t vt0123 = __builtin_wasm_fma_f32x4(vx0123, vn0123, vminus_ln2_hi);
    v128_t vt4567 = __builtin_wasm_fma_f32x4(vx4567, vn4567, vminus_ln2_hi);
    v128_t vt89AB = __builtin_wasm_fma_f32x4(vx89AB, vn89AB, vminus_ln2_hi);
    v128_t vtCDEF = __builtin_wasm_fma_f32x4(vxCDEF, vnCDEF, vminus_ln2_hi);
    v128_t vtGHIJ = __builtin_wasm_fma_f32x4(vxGHIJ, vnGHIJ, vminus_ln2_hi);

    vt0123 = __builtin_wasm_fma_f32x4(vt0123, vn0123, vminus_ln2_lo);
    vt4567 = __builtin_wasm_fma_f32x4(vt4567, vn4567, vminus_ln2_lo);
    vt89AB = __builtin_wasm_fma_f32x4(vt89AB, vn89AB, vminus_ln2_lo);
    vtCDEF = __builtin_wasm_fma_f32x4(vtCDEF, vnCDEF, vminus_ln2_lo);
    vtGHIJ = __builtin_wasm_fma_f32x4(vtGHIJ, vnGHIJ, vminus_ln2_lo);

    v128_t vp0123 = __builtin_wasm_fma_f32x4(vc4, vc5, vt0123);
    v128_t vp4567 = __builtin_wasm_fma_f32x4(vc4, vc5, vt4567);
    v128_t vp89AB = __builtin_wasm_fma_f32x4(vc4, vc5, vt89AB);
    v128_t vpCDEF = __builtin_wasm_fma_f32x4(vc4, vc5, vtCDEF);
    v128_t vpGHIJ = __builtin_wasm_fma_f32x4(vc4, vc5, vtGHIJ);

    vp0123 = __builtin_wasm_fma_f32x4(vc3, vp0123, vt0123);
    vp4567 = __builtin_wasm_fma_f32x4(vc3, vp4567, vt4567);
    vp89AB = __builtin_wasm_fma_f32x4(vc3, vp89AB, vt89AB);
    vpCDEF = __builtin_wasm_fma_f32x4(vc3, vpCDEF, vtCDEF);
    vpGHIJ = __builtin_wasm_fma_f32x4(vc3, vpGHIJ, vtGHIJ);

    vp0123 = __builtin_wasm_fma_f32x4(vc2, vp0123, vt0123);
    vp4567 = __builtin_wasm_fma_f32x4(vc2, vp4567, vt4567);
    vp89AB = __builtin_wasm_fma_f32x4(vc2, vp89AB, vt89AB);
    vpCDEF = __builtin_wasm_fma_f32x4(vc2, vpCDEF, vtCDEF);
    vpGHIJ = __builtin_wasm_fma_f32x4(vc2, vpGHIJ, vtGHIJ);

    vp0123 = __builtin_wasm_fma_f32x4(vc1, vp0123, vt0123);
    vp4567 = __builtin_wasm_fma_f32x4(vc1, vp4567, vt4567);
    vp89AB = __builtin_wasm_fma_f32x4(vc1, vp89AB, vt89AB);
    vpCDEF = __builtin_wasm_fma_f32x4(vc1, vpCDEF, vtCDEF);
    vpGHIJ = __builtin_wasm_fma_f32x4(vc1, vpGHIJ, vtGHIJ);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vt89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    vtCDEF = wasm_f32x4_mul(vtCDEF, vsCDEF);
    vtGHIJ = wasm_f32x4_mul(vtGHIJ, vsGHIJ);

    v128_t vf0123 = __builtin_wasm_fma_f32x4(vs0123, vt0123, vp0123);
    v128_t vf4567 = __builtin_wasm_fma_f32x4(vs4567, vt4567, vp4567);
    v128_t vf89AB = __builtin_wasm_fma_f32x4(vs89AB, vt89AB, vp89AB);
    v128_t vfCDEF = __builtin_wasm_fma_f32x4(vsCDEF, vtCDEF, vpCDEF);
    v128_t vfGHIJ = __builtin_wasm_fma_f32x4(vsGHIJ, vtGHIJ, vpGHIJ);

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

    v128_t vn = __builtin_wasm_fma_f32x4(vmagic_bias, vx, vlog2e);

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = __builtin_wasm_fma_f32x4(vx, vn, vminus_ln2_hi);
    vt = __builtin_wasm_fma_f32x4(vt, vn, vminus_ln2_lo);

    v128_t vp = __builtin_wasm_fma_f32x4(vc4, vc5, vt);
    vp = __builtin_wasm_fma_f32x4(vc3, vp, vt);
    vp = __builtin_wasm_fma_f32x4(vc2, vp, vt);
    vp = __builtin_wasm_fma_f32x4(vc1, vp, vt);

    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = __builtin_wasm_fma_f32x4(vs, vt, vp);

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

    v128_t vn = __builtin_wasm_fma_f32x4(vmagic_bias, vx, vlog2e);

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = __builtin_wasm_fma_f32x4(vx, vn, vminus_ln2_hi);
    vt = __builtin_wasm_fma_f32x4(vt, vn, vminus_ln2_lo);

    v128_t vp = __builtin_wasm_fma_f32x4(vc4, vc5, vt);
    vp = __builtin_wasm_fma_f32x4(vc3, vp, vt);
    vp = __builtin_wasm_fma_f32x4(vc2, vp, vt);
    vp = __builtin_wasm_fma_f32x4(vc1, vp, vt);

    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = __builtin_wasm_fma_f32x4(vs, vt, vp);

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
