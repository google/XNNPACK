// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/vcvt.h"


void xnn_f32_f16_vcvt_ukernel__wasmsimd_u32(
    size_t batch,
    const float* input,
    void* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vexp_bias = wasm_u32x4_const_splat(UINT32_C(0x07800000));
  const v128_t vscale_to_inf = wasm_f32x4_const_splat(0x1.0p+112f);
  const v128_t vexpw_max = wasm_u32x4_const_splat(UINT32_C(0x7F800000));
  const v128_t vscale_to_zero = wasm_f32x4_const_splat(0x1.0p-110f);
  const v128_t vbias_min = wasm_u32x4_const_splat(UINT32_C(0x40008000));
  const v128_t vmanth_mask = wasm_u32x4_const_splat(UINT32_C(0x00000FFF));
  const v128_t vexph_mask = wasm_u32x4_const_splat(UINT32_C(0x00007C00));
  const v128_t vnanh = wasm_u16x8_const_splat(UINT16_C(0x7E00));

  XNN_FORCE_REALIZATION(vexp_bias);
  XNN_FORCE_REALIZATION(vscale_to_inf);
  XNN_FORCE_REALIZATION(vexpw_max);
  XNN_FORCE_REALIZATION(vscale_to_zero);
  XNN_FORCE_REALIZATION(vbias_min);
  XNN_FORCE_REALIZATION(vmanth_mask);
  XNN_FORCE_REALIZATION(vexph_mask);
  XNN_FORCE_REALIZATION(vnanh);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const v128_t vx0 = wasm_v128_load(input);
    const v128_t vx1 = wasm_v128_load(input + 4);
    const v128_t vx2 = wasm_v128_load(input + 8);
    const v128_t vx3 = wasm_v128_load(input + 12);
    const v128_t vx4 = wasm_v128_load(input + 16);
    const v128_t vx5 = wasm_v128_load(input + 20);
    const v128_t vx6 = wasm_v128_load(input + 24);
    const v128_t vx7 = wasm_v128_load(input + 28);
    input += 32;

    const v128_t vabsx0 = wasm_f32x4_abs(vx0);
    const v128_t vabsx1 = wasm_f32x4_abs(vx1);
    const v128_t vabsx2 = wasm_f32x4_abs(vx2);
    const v128_t vabsx3 = wasm_f32x4_abs(vx3);
    const v128_t vabsx4 = wasm_f32x4_abs(vx4);
    const v128_t vabsx5 = wasm_f32x4_abs(vx5);
    const v128_t vabsx6 = wasm_f32x4_abs(vx6);
    const v128_t vabsx7 = wasm_f32x4_abs(vx7);

    const v128_t vsignx0 = wasm_v128_xor(vx0, vabsx0);
    const v128_t vsignx1 = wasm_v128_xor(vx1, vabsx1);
    const v128_t vsignx2 = wasm_v128_xor(vx2, vabsx2);
    const v128_t vsignx3 = wasm_v128_xor(vx3, vabsx3);
    const v128_t vsignx4 = wasm_v128_xor(vx4, vabsx4);
    const v128_t vsignx5 = wasm_v128_xor(vx5, vabsx5);
    const v128_t vsignx6 = wasm_v128_xor(vx6, vabsx6);
    const v128_t vsignx7 = wasm_v128_xor(vx7, vabsx7);

    v128_t vbias0 = wasm_i32x4_add(vabsx0, vexp_bias);
    v128_t vbias1 = wasm_i32x4_add(vabsx1, vexp_bias);
    v128_t vbias2 = wasm_i32x4_add(vabsx2, vexp_bias);
    v128_t vbias3 = wasm_i32x4_add(vabsx3, vexp_bias);
    v128_t vbias4 = wasm_i32x4_add(vabsx4, vexp_bias);
    v128_t vbias5 = wasm_i32x4_add(vabsx5, vexp_bias);
    v128_t vbias6 = wasm_i32x4_add(vabsx6, vexp_bias);
    v128_t vbias7 = wasm_i32x4_add(vabsx7, vexp_bias);

    v128_t vf0 = wasm_f32x4_mul(vabsx0, vscale_to_inf);
    v128_t vf1 = wasm_f32x4_mul(vabsx1, vscale_to_inf);
    v128_t vf2 = wasm_f32x4_mul(vabsx2, vscale_to_inf);
    v128_t vf3 = wasm_f32x4_mul(vabsx3, vscale_to_inf);
    v128_t vf4 = wasm_f32x4_mul(vabsx4, vscale_to_inf);
    v128_t vf5 = wasm_f32x4_mul(vabsx5, vscale_to_inf);
    v128_t vf6 = wasm_f32x4_mul(vabsx6, vscale_to_inf);
    v128_t vf7 = wasm_f32x4_mul(vabsx7, vscale_to_inf);

    const v128_t vnanmaskw0 = wasm_i32x4_gt(vabsx0, vexpw_max);
    const v128_t vnanmaskw1 = wasm_i32x4_gt(vabsx1, vexpw_max);
    const v128_t vnanmaskw2 = wasm_i32x4_gt(vabsx2, vexpw_max);
    const v128_t vnanmaskw3 = wasm_i32x4_gt(vabsx3, vexpw_max);
    const v128_t vnanmaskw4 = wasm_i32x4_gt(vabsx4, vexpw_max);
    const v128_t vnanmaskw5 = wasm_i32x4_gt(vabsx5, vexpw_max);
    const v128_t vnanmaskw6 = wasm_i32x4_gt(vabsx6, vexpw_max);
    const v128_t vnanmaskw7 = wasm_i32x4_gt(vabsx7, vexpw_max);

    vbias0 = wasm_v128_and(vbias0, vexpw_max);
    vbias1 = wasm_v128_and(vbias1, vexpw_max);
    vbias2 = wasm_v128_and(vbias2, vexpw_max);
    vbias3 = wasm_v128_and(vbias3, vexpw_max);
    vbias4 = wasm_v128_and(vbias4, vexpw_max);
    vbias5 = wasm_v128_and(vbias5, vexpw_max);
    vbias6 = wasm_v128_and(vbias6, vexpw_max);
    vbias7 = wasm_v128_and(vbias7, vexpw_max);

    vf0 = wasm_f32x4_mul(vf0, vscale_to_zero);
    vf1 = wasm_f32x4_mul(vf1, vscale_to_zero);
    vf2 = wasm_f32x4_mul(vf2, vscale_to_zero);
    vf3 = wasm_f32x4_mul(vf3, vscale_to_zero);
    vf4 = wasm_f32x4_mul(vf4, vscale_to_zero);
    vf5 = wasm_f32x4_mul(vf5, vscale_to_zero);
    vf6 = wasm_f32x4_mul(vf6, vscale_to_zero);
    vf7 = wasm_f32x4_mul(vf7, vscale_to_zero);

    const v128_t vnanmaskh0 = wasm_i16x8_narrow_i32x4(vnanmaskw0, vnanmaskw1);
    const v128_t vnanmaskh1 = wasm_i16x8_narrow_i32x4(vnanmaskw2, vnanmaskw3);
    const v128_t vnanmaskh2 = wasm_i16x8_narrow_i32x4(vnanmaskw4, vnanmaskw5);
    const v128_t vnanmaskh3 = wasm_i16x8_narrow_i32x4(vnanmaskw6, vnanmaskw7);

    const v128_t vsignh0 = wasm_i16x8_narrow_i32x4(vsignx0, vsignx1);
    const v128_t vsignh1 = wasm_i16x8_narrow_i32x4(vsignx2, vsignx3);
    const v128_t vsignh2 = wasm_i16x8_narrow_i32x4(vsignx4, vsignx5);
    const v128_t vsignh3 = wasm_i16x8_narrow_i32x4(vsignx6, vsignx7);

    vbias0 = wasm_i16x8_max(vbias0, vbias_min);
    vbias1 = wasm_i16x8_max(vbias1, vbias_min);
    vbias2 = wasm_i16x8_max(vbias2, vbias_min);
    vbias3 = wasm_i16x8_max(vbias3, vbias_min);
    vbias4 = wasm_i16x8_max(vbias4, vbias_min);
    vbias5 = wasm_i16x8_max(vbias5, vbias_min);
    vbias6 = wasm_i16x8_max(vbias6, vbias_min);
    vbias7 = wasm_i16x8_max(vbias7, vbias_min);

    vf0 = wasm_f32x4_add(vf0, vbias0);
    vf1 = wasm_f32x4_add(vf1, vbias1);
    vf2 = wasm_f32x4_add(vf2, vbias2);
    vf3 = wasm_f32x4_add(vf3, vbias3);
    vf4 = wasm_f32x4_add(vf4, vbias4);
    vf5 = wasm_f32x4_add(vf5, vbias5);
    vf6 = wasm_f32x4_add(vf6, vbias6);
    vf7 = wasm_f32x4_add(vf7, vbias7);

    v128_t vexpw0 = wasm_i32x4_shr(vf0, 13);
    v128_t vexpw1 = wasm_i32x4_shr(vf1, 13);
    v128_t vexpw2 = wasm_i32x4_shr(vf2, 13);
    v128_t vexpw3 = wasm_i32x4_shr(vf3, 13);
    v128_t vexpw4 = wasm_i32x4_shr(vf4, 13);
    v128_t vexpw5 = wasm_i32x4_shr(vf5, 13);
    v128_t vexpw6 = wasm_i32x4_shr(vf6, 13);
    v128_t vexpw7 = wasm_i32x4_shr(vf7, 13);

    const v128_t vmantw0 = wasm_v128_and(vf0, vmanth_mask);
    const v128_t vmantw1 = wasm_v128_and(vf1, vmanth_mask);
    const v128_t vmantw2 = wasm_v128_and(vf2, vmanth_mask);
    const v128_t vmantw3 = wasm_v128_and(vf3, vmanth_mask);
    const v128_t vmantw4 = wasm_v128_and(vf4, vmanth_mask);
    const v128_t vmantw5 = wasm_v128_and(vf5, vmanth_mask);
    const v128_t vmantw6 = wasm_v128_and(vf6, vmanth_mask);
    const v128_t vmantw7 = wasm_v128_and(vf7, vmanth_mask);

    vexpw0 = wasm_v128_and(vexpw0, vexph_mask);
    vexpw1 = wasm_v128_and(vexpw1, vexph_mask);
    vexpw2 = wasm_v128_and(vexpw2, vexph_mask);
    vexpw3 = wasm_v128_and(vexpw3, vexph_mask);
    vexpw4 = wasm_v128_and(vexpw4, vexph_mask);
    vexpw5 = wasm_v128_and(vexpw5, vexph_mask);
    vexpw6 = wasm_v128_and(vexpw6, vexph_mask);
    vexpw7 = wasm_v128_and(vexpw7, vexph_mask);

    const v128_t vnonsignw0 = wasm_i32x4_add(vmantw0, vexpw0);
    const v128_t vnonsignw1 = wasm_i32x4_add(vmantw1, vexpw1);
    const v128_t vnonsignw2 = wasm_i32x4_add(vmantw2, vexpw2);
    const v128_t vnonsignw3 = wasm_i32x4_add(vmantw3, vexpw3);
    const v128_t vnonsignw4 = wasm_i32x4_add(vmantw4, vexpw4);
    const v128_t vnonsignw5 = wasm_i32x4_add(vmantw5, vexpw5);
    const v128_t vnonsignw6 = wasm_i32x4_add(vmantw6, vexpw6);
    const v128_t vnonsignw7 = wasm_i32x4_add(vmantw7, vexpw7);

    const v128_t vnonsignh0 = wasm_i16x8_narrow_i32x4(vnonsignw0, vnonsignw1);
    const v128_t vnonsignh1 = wasm_i16x8_narrow_i32x4(vnonsignw2, vnonsignw3);
    const v128_t vnonsignh2 = wasm_i16x8_narrow_i32x4(vnonsignw4, vnonsignw5);
    const v128_t vnonsignh3 = wasm_i16x8_narrow_i32x4(vnonsignw6, vnonsignw7);

    const v128_t vabsh0 = wasm_v128_bitselect(vnanh, vnonsignh0, vnanmaskh0);
    const v128_t vabsh1 = wasm_v128_bitselect(vnanh, vnonsignh1, vnanmaskh1);
    const v128_t vabsh2 = wasm_v128_bitselect(vnanh, vnonsignh2, vnanmaskh2);
    const v128_t vabsh3 = wasm_v128_bitselect(vnanh, vnonsignh3, vnanmaskh3);

    const v128_t vh0 = wasm_v128_or(vabsh0, vsignh0);
    const v128_t vh1 = wasm_v128_or(vabsh1, vsignh1);
    const v128_t vh2 = wasm_v128_or(vabsh2, vsignh2);
    const v128_t vh3 = wasm_v128_or(vabsh3, vsignh3);

    wasm_v128_store(o, vh0);
    wasm_v128_store(o + 8, vh1);
    wasm_v128_store(o + 16, vh2);
    wasm_v128_store(o + 24, vh3);
    o += 32;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const v128_t vx_lo = wasm_v128_load(input);
    const v128_t vx_hi = wasm_v128_load(input + 4);
    input += 8;

    const v128_t vabsx_lo = wasm_f32x4_abs(vx_lo);
    const v128_t vabsx_hi = wasm_f32x4_abs(vx_hi);

    const v128_t vsignx_lo = wasm_v128_xor(vx_lo, vabsx_lo);
    const v128_t vsignx_hi = wasm_v128_xor(vx_hi, vabsx_hi);
    v128_t vbias_lo = wasm_i32x4_add(vabsx_lo, vexp_bias);
    v128_t vbias_hi = wasm_i32x4_add(vabsx_hi, vexp_bias);
    v128_t vf_lo = wasm_f32x4_mul(vabsx_lo, vscale_to_inf);
    v128_t vf_hi = wasm_f32x4_mul(vabsx_hi, vscale_to_inf);
    const v128_t vnanmaskw_lo = wasm_i32x4_gt(vabsx_lo, vexpw_max);
    const v128_t vnanmaskw_hi = wasm_i32x4_gt(vabsx_hi, vexpw_max);

    vbias_lo = wasm_v128_and(vbias_lo, vexpw_max);
    vbias_hi = wasm_v128_and(vbias_hi, vexpw_max);
    vf_lo = wasm_f32x4_mul(vf_lo, vscale_to_zero);
    vf_hi = wasm_f32x4_mul(vf_hi, vscale_to_zero);
    const v128_t vnanmaskh = wasm_i16x8_narrow_i32x4(vnanmaskw_lo, vnanmaskw_hi);
    const v128_t vsignh = wasm_i16x8_narrow_i32x4(vsignx_lo, vsignx_hi);

    vbias_lo = wasm_i16x8_max(vbias_lo, vbias_min);
    vbias_hi = wasm_i16x8_max(vbias_hi, vbias_min);

    vf_lo = wasm_f32x4_add(vf_lo, vbias_lo);
    vf_hi = wasm_f32x4_add(vf_hi, vbias_hi);

    v128_t vexpw_lo = wasm_i32x4_shr(vf_lo, 13);
    v128_t vexpw_hi = wasm_i32x4_shr(vf_hi, 13);
    const v128_t vmantw_lo = wasm_v128_and(vf_lo, vmanth_mask);
    const v128_t vmantw_hi = wasm_v128_and(vf_hi, vmanth_mask);

    vexpw_lo = wasm_v128_and(vexpw_lo, vexph_mask);
    vexpw_hi = wasm_v128_and(vexpw_hi, vexph_mask);

    const v128_t vnonsignw_lo = wasm_i32x4_add(vmantw_lo, vexpw_lo);
    const v128_t vnonsignw_hi = wasm_i32x4_add(vmantw_hi, vexpw_hi);

    const v128_t vnonsignh = wasm_i16x8_narrow_i32x4(vnonsignw_lo, vnonsignw_hi);

    const v128_t vabsh = wasm_v128_bitselect(vnanh, vnonsignh, vnanmaskh);

    const v128_t vh = wasm_v128_or(vabsh, vsignh);

    wasm_v128_store(o, vh);
    o += 8;
  }
  if XNN_UNPREDICTABLE(batch != 0) {
    const v128_t vx_lo = wasm_v128_load(input);
    const float* input_hi = (const float*) ((uintptr_t) input + (batch & (4 * sizeof(float))));
    const v128_t vx_hi = wasm_v128_load(input_hi);

    const v128_t vabsx_lo = wasm_f32x4_abs(vx_lo);
    const v128_t vabsx_hi = wasm_f32x4_abs(vx_hi);

    const v128_t vsignx_lo = wasm_v128_xor(vx_lo, vabsx_lo);
    const v128_t vsignx_hi = wasm_v128_xor(vx_hi, vabsx_hi);
    v128_t vbias_lo = wasm_i32x4_add(vabsx_lo, vexp_bias);
    v128_t vbias_hi = wasm_i32x4_add(vabsx_hi, vexp_bias);
    v128_t vf_lo = wasm_f32x4_mul(vabsx_lo, vscale_to_inf);
    v128_t vf_hi = wasm_f32x4_mul(vabsx_hi, vscale_to_inf);
    const v128_t vnanmaskw_lo = wasm_i32x4_gt(vabsx_lo, vexpw_max);
    const v128_t vnanmaskw_hi = wasm_i32x4_gt(vabsx_hi, vexpw_max);

    vbias_lo = wasm_v128_and(vbias_lo, vexpw_max);
    vbias_hi = wasm_v128_and(vbias_hi, vexpw_max);
    vf_lo = wasm_f32x4_mul(vf_lo, vscale_to_zero);
    vf_hi = wasm_f32x4_mul(vf_hi, vscale_to_zero);
    const v128_t vnanmaskh = wasm_i16x8_narrow_i32x4(vnanmaskw_lo, vnanmaskw_hi);
    const v128_t vsignh = wasm_i16x8_narrow_i32x4(vsignx_lo, vsignx_hi);

    vbias_lo = wasm_i16x8_max(vbias_lo, vbias_min);
    vbias_hi = wasm_i16x8_max(vbias_hi, vbias_min);

    vf_lo = wasm_f32x4_add(vf_lo, vbias_lo);
    vf_hi = wasm_f32x4_add(vf_hi, vbias_hi);

    v128_t vexpw_lo = wasm_i32x4_shr(vf_lo, 13);
    v128_t vexpw_hi = wasm_i32x4_shr(vf_hi, 13);
    const v128_t vmantw_lo = wasm_v128_and(vf_lo, vmanth_mask);
    const v128_t vmantw_hi = wasm_v128_and(vf_hi, vmanth_mask);

    vexpw_lo = wasm_v128_and(vexpw_lo, vexph_mask);
    vexpw_hi = wasm_v128_and(vexpw_hi, vexph_mask);

    const v128_t vnonsignw_lo = wasm_i32x4_add(vmantw_lo, vexpw_lo);
    const v128_t vnonsignw_hi = wasm_i32x4_add(vmantw_hi, vexpw_hi);

    const v128_t vnonsignh = wasm_i16x8_narrow_i32x4(vnonsignw_lo, vnonsignw_hi);

    const v128_t vabsh = wasm_v128_bitselect(vnanh, vnonsignh, vnanmaskh);

    v128_t vh = wasm_v128_or(vabsh, vsignh);

    if (batch & (4 * sizeof(float))) {
      wasm_v128_store64_lane(o, vh, 0);
      vh = wasm_v64x2_shuffle(vh, vh, 1, 1);
      o += 4;
    }
    if (batch & (2 * sizeof(float))) {
      wasm_v128_store32_lane(o, vh, 0);
      vh = wasm_i64x2_shr(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store16_lane(o, vh, 0);
    }
  }
}
