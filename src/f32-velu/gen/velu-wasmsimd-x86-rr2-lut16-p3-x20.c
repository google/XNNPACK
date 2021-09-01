// Auto-generated file. Do not edit!
//   Template: src/f32-velu/wasmsimd-rr2-lut16-p3.c.in
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


extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x20(
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
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.800000p19f);
  const v128_t vlog2e = wasm_f32x4_const_splat(0x1.715476p+0f);
  const v128_t vindex_mask =  wasm_i32x4_const_splat(0xF);
  const v128_t vminus_ln2_hi = wasm_f32x4_const_splat(-0x1.62E400p-1f);
  const v128_t vminus_ln2_lo = wasm_f32x4_const_splat(-0x1.7F7D1Cp-20f);
  const v128_t vc3 = wasm_f32x4_const_splat(0x1.55561Cp-3f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.0001ECp-1f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);

  for (; n >= 20 * sizeof(float); n -= 20 * sizeof(float)) {
    v128_t vx0123 = wasm_v128_load(x);
    v128_t vx4567 = wasm_v128_load(x + 4);
    v128_t vx89AB = wasm_v128_load(x + 8);
    v128_t vxCDEF = wasm_v128_load(x + 12);
    v128_t vxGHIJ = wasm_v128_load(x + 16);
    x += 20;

    const v128_t vz0123 = wasm_f32x4_mul(vx0123, vprescale);
    const v128_t vz4567 = wasm_f32x4_mul(vx4567, vprescale);
    const v128_t vz89AB = wasm_f32x4_mul(vx89AB, vprescale);
    const v128_t vzCDEF = wasm_f32x4_mul(vxCDEF, vprescale);
    const v128_t vzGHIJ = wasm_f32x4_mul(vxGHIJ, vprescale);

    v128_t vn0123 = wasm_f32x4_add(wasm_f32x4_mul(vz0123, vlog2e), vmagic_bias);
    v128_t vn4567 = wasm_f32x4_add(wasm_f32x4_mul(vz4567, vlog2e), vmagic_bias);
    v128_t vn89AB = wasm_f32x4_add(wasm_f32x4_mul(vz89AB, vlog2e), vmagic_bias);
    v128_t vnCDEF = wasm_f32x4_add(wasm_f32x4_mul(vzCDEF, vlog2e), vmagic_bias);
    v128_t vnGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vzGHIJ, vlog2e), vmagic_bias);

    const v128_t vidx0123 = wasm_i32x4_shl(wasm_v128_and(vn0123, vindex_mask), 2);
    const v128_t ven0123 = wasm_i32x4_shl(vn0123, 19);
    const v128_t vidx4567 = wasm_i32x4_shl(wasm_v128_and(vn4567, vindex_mask), 2);
    const v128_t ven4567 = wasm_i32x4_shl(vn4567, 19);
    const v128_t vidx89AB = wasm_i32x4_shl(wasm_v128_and(vn89AB, vindex_mask), 2);
    const v128_t ven89AB = wasm_i32x4_shl(vn89AB, 19);
    const v128_t vidxCDEF = wasm_i32x4_shl(wasm_v128_and(vnCDEF, vindex_mask), 2);
    const v128_t venCDEF = wasm_i32x4_shl(vnCDEF, 19);
    const v128_t vidxGHIJ = wasm_i32x4_shl(wasm_v128_and(vnGHIJ, vindex_mask), 2);
    const v128_t venGHIJ = wasm_i32x4_shl(vnGHIJ, 19);

    const uint64_t vidx01 = wasm_i64x2_extract_lane(vidx0123, 0);
    const uint64_t vidx23 = wasm_i64x2_extract_lane(vidx0123, 1);
    const float vl0   = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx01));
    const float vl1 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx01 >> 32)));
    const float vl2 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx23));
    const float vl3 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx23 >> 32)));
    const v128_t vl0123 = wasm_f32x4_make(vl0, vl1, vl2, vl3);
    const uint64_t vidx45 = wasm_i64x2_extract_lane(vidx4567, 0);
    const uint64_t vidx67 = wasm_i64x2_extract_lane(vidx4567, 1);
    const float vl4   = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx45));
    const float vl5 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx45 >> 32)));
    const float vl6 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx67));
    const float vl7 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx67 >> 32)));
    const v128_t vl4567 = wasm_f32x4_make(vl4, vl5, vl6, vl7);
    const uint64_t vidx89 = wasm_i64x2_extract_lane(vidx89AB, 0);
    const uint64_t vidxAB = wasm_i64x2_extract_lane(vidx89AB, 1);
    const float vl8   = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx89));
    const float vl9 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx89 >> 32)));
    const float vlA = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxAB));
    const float vlB = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxAB >> 32)));
    const v128_t vl89AB = wasm_f32x4_make(vl8, vl9, vlA, vlB);
    const uint64_t vidxCD = wasm_i64x2_extract_lane(vidxCDEF, 0);
    const uint64_t vidxEF = wasm_i64x2_extract_lane(vidxCDEF, 1);
    const float vlC   = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxCD));
    const float vlD = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxCD >> 32)));
    const float vlE = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxEF));
    const float vlF = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxEF >> 32)));
    const v128_t vlCDEF = wasm_f32x4_make(vlC, vlD, vlE, vlF);
    const uint64_t vidxGH = wasm_i64x2_extract_lane(vidxGHIJ, 0);
    const uint64_t vidxIJ = wasm_i64x2_extract_lane(vidxGHIJ, 1);
    const float vlG   = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxGH));
    const float vlH = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxGH >> 32)));
    const float vlI = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxIJ));
    const float vlJ = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxIJ >> 32)));
    const v128_t vlGHIJ = wasm_f32x4_make(vlG, vlH, vlI, vlJ);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    v128_t vs0123 = wasm_i32x4_add(vl0123, ven0123);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    v128_t vs4567 = wasm_i32x4_add(vl4567, ven4567);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);
    v128_t vs89AB = wasm_i32x4_add(vl89AB, ven89AB);
    vnCDEF = wasm_f32x4_sub(vnCDEF, vmagic_bias);
    v128_t vsCDEF = wasm_i32x4_add(vlCDEF, venCDEF);
    vnGHIJ = wasm_f32x4_sub(vnGHIJ, vmagic_bias);
    v128_t vsGHIJ = wasm_i32x4_add(vlGHIJ, venGHIJ);

    v128_t vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vn0123, vminus_ln2_hi), vz0123);
    const v128_t vsatm0123 = wasm_f32x4_le(vz0123, vsat_cutoff);
    v128_t vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vn4567, vminus_ln2_hi), vz4567);
    const v128_t vsatm4567 = wasm_f32x4_le(vz4567, vsat_cutoff);
    v128_t vt89AB = wasm_f32x4_add(wasm_f32x4_mul(vn89AB, vminus_ln2_hi), vz89AB);
    const v128_t vsatm89AB = wasm_f32x4_le(vz89AB, vsat_cutoff);
    v128_t vtCDEF = wasm_f32x4_add(wasm_f32x4_mul(vnCDEF, vminus_ln2_hi), vzCDEF);
    const v128_t vsatmCDEF = wasm_f32x4_le(vzCDEF, vsat_cutoff);
    v128_t vtGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vnGHIJ, vminus_ln2_hi), vzGHIJ);
    const v128_t vsatmGHIJ = wasm_f32x4_le(vzGHIJ, vsat_cutoff);

    vt0123 = wasm_f32x4_add(wasm_f32x4_mul(vn0123, vminus_ln2_lo), vt0123);
    vs0123 = wasm_v128_andnot(vs0123, vsatm0123);
    vt4567 = wasm_f32x4_add(wasm_f32x4_mul(vn4567, vminus_ln2_lo), vt4567);
    vs4567 = wasm_v128_andnot(vs4567, vsatm4567);
    vt89AB = wasm_f32x4_add(wasm_f32x4_mul(vn89AB, vminus_ln2_lo), vt89AB);
    vs89AB = wasm_v128_andnot(vs89AB, vsatm89AB);
    vtCDEF = wasm_f32x4_add(wasm_f32x4_mul(vnCDEF, vminus_ln2_lo), vtCDEF);
    vsCDEF = wasm_v128_andnot(vsCDEF, vsatmCDEF);
    vtGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vnGHIJ, vminus_ln2_lo), vtGHIJ);
    vsGHIJ = wasm_v128_andnot(vsGHIJ, vsatmGHIJ);

    vt0123 = wasm_v128_andnot(vt0123, vsatm0123);
    v128_t vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vc3, vt0123), vc2);
    vt4567 = wasm_v128_andnot(vt4567, vsatm4567);
    v128_t vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vc3, vt4567), vc2);
    vt89AB = wasm_v128_andnot(vt89AB, vsatm89AB);
    v128_t vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vc3, vt89AB), vc2);
    vtCDEF = wasm_v128_andnot(vtCDEF, vsatmCDEF);
    v128_t vpCDEF = wasm_f32x4_add(wasm_f32x4_mul(vc3, vtCDEF), vc2);
    vtGHIJ = wasm_v128_andnot(vtGHIJ, vsatmGHIJ);
    v128_t vpGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vc3, vtGHIJ), vc2);

    vp0123 = wasm_f32x4_mul(vp0123, vt0123);
    vp4567 = wasm_f32x4_mul(vp4567, vt4567);
    vp89AB = wasm_f32x4_mul(vp89AB, vt89AB);
    vpCDEF = wasm_f32x4_mul(vpCDEF, vtCDEF);
    vpGHIJ = wasm_f32x4_mul(vpGHIJ, vtGHIJ);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vs0123 = wasm_f32x4_sub(vs0123, vone);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vs4567 = wasm_f32x4_sub(vs4567, vone);
    vt89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    vs89AB = wasm_f32x4_sub(vs89AB, vone);
    vtCDEF = wasm_f32x4_mul(vtCDEF, vsCDEF);
    vsCDEF = wasm_f32x4_sub(vsCDEF, vone);
    vtGHIJ = wasm_f32x4_mul(vtGHIJ, vsGHIJ);
    vsGHIJ = wasm_f32x4_sub(vsGHIJ, vone);

    vp0123 = wasm_f32x4_add(wasm_f32x4_mul(vp0123, vt0123), vt0123);
    vp4567 = wasm_f32x4_add(wasm_f32x4_mul(vp4567, vt4567), vt4567);
    vp89AB = wasm_f32x4_add(wasm_f32x4_mul(vp89AB, vt89AB), vt89AB);
    vpCDEF = wasm_f32x4_add(wasm_f32x4_mul(vpCDEF, vtCDEF), vtCDEF);
    vpGHIJ = wasm_f32x4_add(wasm_f32x4_mul(vpGHIJ, vtGHIJ), vtGHIJ);

    const v128_t ve0123 = wasm_f32x4_mul(wasm_f32x4_add(vp0123, vs0123), valpha);
    const v128_t ve4567 = wasm_f32x4_mul(wasm_f32x4_add(vp4567, vs4567), valpha);
    const v128_t ve89AB = wasm_f32x4_mul(wasm_f32x4_add(vp89AB, vs89AB), valpha);
    const v128_t veCDEF = wasm_f32x4_mul(wasm_f32x4_add(vpCDEF, vsCDEF), valpha);
    const v128_t veGHIJ = wasm_f32x4_mul(wasm_f32x4_add(vpGHIJ, vsGHIJ), valpha);

    const v128_t vsignm0123 = wasm_i32x4_shr(vx0123, 31);
    vx0123 = wasm_f32x4_mul(vx0123, vbeta);
    const v128_t vsignm4567 = wasm_i32x4_shr(vx4567, 31);
    vx4567 = wasm_f32x4_mul(vx4567, vbeta);
    const v128_t vsignm89AB = wasm_i32x4_shr(vx89AB, 31);
    vx89AB = wasm_f32x4_mul(vx89AB, vbeta);
    const v128_t vsignmCDEF = wasm_i32x4_shr(vxCDEF, 31);
    vxCDEF = wasm_f32x4_mul(vxCDEF, vbeta);
    const v128_t vsignmGHIJ = wasm_i32x4_shr(vxGHIJ, 31);
    vxGHIJ = wasm_f32x4_mul(vxGHIJ, vbeta);

    const v128_t vy0123 = wasm_v128_bitselect(ve0123, vx0123, vsignm0123);
    const v128_t vy4567 = wasm_v128_bitselect(ve4567, vx4567, vsignm4567);
    const v128_t vy89AB = wasm_v128_bitselect(ve89AB, vx89AB, vsignm89AB);
    const v128_t vyCDEF = wasm_v128_bitselect(veCDEF, vxCDEF, vsignmCDEF);
    const v128_t vyGHIJ = wasm_v128_bitselect(veGHIJ, vxGHIJ, vsignmGHIJ);

    wasm_v128_store(y, vy0123);
    wasm_v128_store(y + 4, vy4567);
    wasm_v128_store(y + 8, vy89AB);
    wasm_v128_store(y + 12, vyCDEF);
    wasm_v128_store(y + 16, vyGHIJ);
    y += 20;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    v128_t vx = wasm_v128_load(x);
    x += 4;

    const v128_t vz = wasm_f32x4_mul(vx, vprescale);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vlog2e), vmagic_bias);
    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const v128_t ven = wasm_i32x4_shl(vn, 19);

    const uint64_t vidx_lo = wasm_i64x2_extract_lane(vidx, 0);
    const uint64_t vidx_hi = wasm_i64x2_extract_lane(vidx, 1);
    const float vl0 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo));
    const float vl1 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32)));
    const float vl2 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi));
    const float vl3 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32)));
    const v128_t vl = wasm_f32x4_make(vl0, vl1, vl2, vl3);

    v128_t vs = wasm_i32x4_add(vl, ven);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_hi), vz);
    const v128_t vsatm = wasm_f32x4_le(vz, vsat_cutoff);
    vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_lo), vt);
    vs = wasm_v128_andnot(vs, vsatm);
    vt = wasm_v128_andnot(vt, vsatm);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc3, vt), vc2);
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

    const v128_t vz = wasm_f32x4_mul(vx, vprescale);

    v128_t vn = wasm_f32x4_add(wasm_f32x4_mul(vz, vlog2e), vmagic_bias);
    const v128_t vidx = wasm_i32x4_shl(wasm_v128_and(vn, vindex_mask), 2);
    const v128_t ven = wasm_i32x4_shl(vn, 19);

    const uint64_t vidx_lo = wasm_i64x2_extract_lane(vidx, 0);
    const uint64_t vidx_hi = wasm_i64x2_extract_lane(vidx, 1);
    const float vl0 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo));
    const float vl1 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32)));
    const float vl2 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi));
    const float vl3 = *((const float*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32)));
    const v128_t vl = wasm_f32x4_make(vl0, vl1, vl2, vl3);

    v128_t vs = wasm_i32x4_add(vl, ven);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_hi), vz);
    const v128_t vsatm = wasm_f32x4_le(vz, vsat_cutoff);
    vt = wasm_f32x4_add(wasm_f32x4_mul(vn, vminus_ln2_lo), vt);
    vs = wasm_v128_andnot(vs, vsatm);
    vt = wasm_v128_andnot(vt, vsatm);

    v128_t vp = wasm_f32x4_add(wasm_f32x4_mul(vc3, vt), vc2);
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
