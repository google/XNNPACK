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


void xnn_f32_f16_vcvt_ukernel__wasmsimd_u8(
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
