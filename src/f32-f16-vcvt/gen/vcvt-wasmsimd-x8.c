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

#include <xnnpack/common.h>
#include <xnnpack/vcvt.h>


void xnn_f32_f16_vcvt_ukernel__wasmsimd_x8(
    size_t n,
    const float* input,
    void* output,
    const void* params)
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vexp_bias = wasm_i32x4_const_splat(0x07800000);
  const v128_t vscale_to_inf = wasm_f32x4_const_splat(0x1.0p+112f);
  const v128_t vexpw_max = wasm_i32x4_const_splat(0x7F800000);
  const v128_t vscale_to_zero = wasm_f32x4_const_splat(0x1.0p-110f);
  const v128_t vbias_min = wasm_i32x4_const_splat(0x40008000);
  const v128_t vmanth_mask = wasm_i32x4_const_splat(0x0FFF);
  const v128_t vexph_mask = wasm_i32x4_const_splat(0x7C00);
  const v128_t vnanh = wasm_i16x8_const_splat(0x7E00);

  uint16_t* o = (uint16_t*) output;
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
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
  if XNN_UNPREDICTABLE(n != 0) {
    const v128_t vx_lo = wasm_v128_load(input);
    const float* input_hi = input + 4;
    if XNN_UNPREDICTABLE((n & (4 * sizeof(uint16_t))) == 0) {
      input_hi = input;
    }
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

    if (n & (4 * sizeof(uint16_t))) {
      *((double*) o) = wasm_f64x2_extract_lane(vh, 0);
      vh = wasm_v64x2_shuffle(vh, vh, 1, 1);
      o += 4;
    }
    if (n & (2 * sizeof(uint16_t))) {
      *((float*) o) = (float) wasm_f32x4_extract_lane(vh, 0);
      vh = wasm_i64x2_shr(vh, 32);
      o += 2;
    }
    const uint32_t vh_lo = wasm_i32x4_extract_lane(vh, 0);
    if (n & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) vh_lo;
    }
  }
}
