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


void xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x4(
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
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    // Load 4 (1x4) inputs at a time.
    const v128_t vi0123 = wasm_v128_load(input);
    input += 4;

    const v128_t vx0123 = wasm_f32x4_sub(vi0123, vi_max);

    v128_t vn0123 = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vx0123, vlog2e));

    const v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);

    v128_t vt0123 = wasm_f32x4_add(vx0123, wasm_f32x4_mul(vn0123, vminus_ln2_hi));

    vt0123 = wasm_f32x4_add(vt0123, wasm_f32x4_mul(vn0123, vminus_ln2_lo));

    v128_t vp0123 = wasm_f32x4_add(vc4, wasm_f32x4_mul(vc5, vt0123));

    vp0123 = wasm_f32x4_add(vc3, wasm_f32x4_mul(vp0123, vt0123));

    vp0123 = wasm_f32x4_add(vc2, wasm_f32x4_mul(vp0123, vt0123));

    vp0123 = wasm_f32x4_add(vc1, wasm_f32x4_mul(vp0123, vt0123));

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);

    v128_t vf0123 = wasm_f32x4_add(vs0123, wasm_f32x4_mul(vt0123, vp0123));

    vf0123 = wasm_v128_andnot(vf0123, wasm_f32x4_lt(vx0123, vdenorm_cutoff));

    wasm_v128_store(output, vf0123);
    output += 4;

    vacc0 = wasm_f32x4_add(vacc0, vf0123);
  }

  v128_t vacc = vacc0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vi = wasm_v128_load(input);
    input += 4;

    const v128_t vx = wasm_f32x4_sub(vi, vi_max);

    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vx, vlog2e));

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(vx, wasm_f32x4_mul(vn, vminus_ln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vminus_ln2_lo));

    v128_t vp = wasm_f32x4_add(vc4, wasm_f32x4_mul(vc5, vt));
    vp = wasm_f32x4_add(vc3, wasm_f32x4_mul(vp, vt));
    vp = wasm_f32x4_add(vc2, wasm_f32x4_mul(vp, vt));
    vp = wasm_f32x4_add(vc1, wasm_f32x4_mul(vp, vt));

    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = wasm_f32x4_add(vs, wasm_f32x4_mul(vt, vp));

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

    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vx, vlog2e));

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(vx, wasm_f32x4_mul(vn, vminus_ln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vminus_ln2_lo));

    v128_t vp = wasm_f32x4_add(vc4, wasm_f32x4_mul(vc5, vt));
    vp = wasm_f32x4_add(vc3, wasm_f32x4_mul(vp, vt));
    vp = wasm_f32x4_add(vc2, wasm_f32x4_mul(vp, vt));
    vp = wasm_f32x4_add(vc1, wasm_f32x4_mul(vp, vt));

    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = wasm_f32x4_add(vs, wasm_f32x4_mul(vt, vp));

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
