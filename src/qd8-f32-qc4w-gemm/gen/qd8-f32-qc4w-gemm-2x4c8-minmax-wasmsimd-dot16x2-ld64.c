// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const v128_t vmask = wasm_i8x16_const_splat(0xF0);
  do {
    const v128_t vksum0123 = wasm_v128_load(w);
    const v128_t vinput_zero_point0 = wasm_v128_load32_splat(&quantization_params[0].zero_point);
    const v128_t vinput_zero_point1 = wasm_v128_load32_splat(&quantization_params[1].zero_point);
    const v128_t vinit0x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point0);
    const v128_t vinit1x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point1);

    const v128_t vzero = wasm_i32x4_const_splat(0);
    v128_t vacc0x0 = wasm_v32x4_shuffle(vinit0x0123, vzero, 0, 5, 6, 7);
    v128_t vacc0x1 = wasm_v32x4_shuffle(vinit0x0123, vzero, 4, 1, 6, 7);
    v128_t vacc0x2 = wasm_v32x4_shuffle(vinit0x0123, vzero, 4, 5, 2, 7);
    v128_t vacc0x3 = wasm_v32x4_shuffle(vinit0x0123, vzero, 4, 5, 6, 3);
    v128_t vacc1x0 = wasm_v32x4_shuffle(vinit1x0123, vzero, 0, 5, 6, 7);
    v128_t vacc1x1 = wasm_v32x4_shuffle(vinit1x0123, vzero, 4, 1, 6, 7);
    v128_t vacc1x2 = wasm_v32x4_shuffle(vinit1x0123, vzero, 4, 5, 2, 7);
    v128_t vacc1x3 = wasm_v32x4_shuffle(vinit1x0123, vzero, 4, 5, 6, 3);
    w = (const int32_t*) w + 4;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      v128_t vxa0;
      v128_t vxa1;
      vxa0 = wasm_i16x8_load8x8(a0);
      a0 += 8;
      vxa1 = wasm_i16x8_load8x8(a1);
      a1 += 8;
      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb2 = wasm_v128_load((const int8_t*) w + 16);
      const v128_t vbm0 = wasm_i8x16_shl(vb0, 4);
      const v128_t vbm2 = wasm_i8x16_shl(vb2, 4);
      const v128_t vxb0 = wasm_i16x8_extend_low_i8x16(vbm0);
      const v128_t vxb1 = wasm_i16x8_extend_high_i8x16(vbm0);
      const v128_t vxb2 = wasm_i16x8_extend_low_i8x16(vbm2);
      const v128_t vxb3 = wasm_i16x8_extend_high_i8x16(vbm2);
      vacc0x0 = wasm_i32x4_add(vacc0x0, wasm_i32x4_dot_i16x8(vxa0, vxb0));
      vacc1x0 = wasm_i32x4_add(vacc1x0, wasm_i32x4_dot_i16x8(vxa1, vxb0));
      vacc0x1 = wasm_i32x4_add(vacc0x1, wasm_i32x4_dot_i16x8(vxa0, vxb1));
      vacc1x1 = wasm_i32x4_add(vacc1x1, wasm_i32x4_dot_i16x8(vxa1, vxb1));
      vacc0x2 = wasm_i32x4_add(vacc0x2, wasm_i32x4_dot_i16x8(vxa0, vxb2));
      vacc1x2 = wasm_i32x4_add(vacc1x2, wasm_i32x4_dot_i16x8(vxa1, vxb2));
      vacc0x3 = wasm_i32x4_add(vacc0x3, wasm_i32x4_dot_i16x8(vxa0, vxb3));
      vacc1x3 = wasm_i32x4_add(vacc1x3, wasm_i32x4_dot_i16x8(vxa1, vxb3));
      vxa0 = wasm_i16x8_load8x8(a0);
      a0 += 8;
      vxa1 = wasm_i16x8_load8x8(a1);
      a1 += 8;
      const v128_t vbm4 = wasm_v128_and(vb0, vmask);
      const v128_t vbm6 = wasm_v128_and(vb2, vmask);
      const v128_t vxb4 = wasm_i16x8_extend_low_i8x16(vbm4);
      const v128_t vxb5 = wasm_i16x8_extend_high_i8x16(vbm4);
      const v128_t vxb6 = wasm_i16x8_extend_low_i8x16(vbm6);
      const v128_t vxb7 = wasm_i16x8_extend_high_i8x16(vbm6);
      vacc0x0 = wasm_i32x4_add(vacc0x0, wasm_i32x4_dot_i16x8(vxa0, vxb4));
      vacc1x0 = wasm_i32x4_add(vacc1x0, wasm_i32x4_dot_i16x8(vxa1, vxb4));
      vacc0x1 = wasm_i32x4_add(vacc0x1, wasm_i32x4_dot_i16x8(vxa0, vxb5));
      vacc1x1 = wasm_i32x4_add(vacc1x1, wasm_i32x4_dot_i16x8(vxa1, vxb5));
      vacc0x2 = wasm_i32x4_add(vacc0x2, wasm_i32x4_dot_i16x8(vxa0, vxb6));
      vacc1x2 = wasm_i32x4_add(vacc1x2, wasm_i32x4_dot_i16x8(vxa1, vxb6));
      vacc0x3 = wasm_i32x4_add(vacc0x3, wasm_i32x4_dot_i16x8(vxa0, vxb7));
      vacc1x3 = wasm_i32x4_add(vacc1x3, wasm_i32x4_dot_i16x8(vxa1, vxb7));
      w = (const int8_t*) w + 32;
      k -= 16 * sizeof(int8_t);
    }

    while (k >= 8 * sizeof(int8_t)) {
      const v128_t vxa0 = wasm_i16x8_load8x8(a0);
      a0 += 8;
      const v128_t vxa1 = wasm_i16x8_load8x8(a1);
      a1 += 8;

      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb2 = wasm_v128_load((const int8_t*) w + 16);
      const v128_t vbm0 = wasm_i8x16_shl(vb0, 4);
      const v128_t vbm2 = wasm_i8x16_shl(vb2, 4);
      const v128_t vxb0 = wasm_i16x8_extend_low_i8x16(vbm0);
      const v128_t vxb1 = wasm_i16x8_extend_high_i8x16(vbm0);
      const v128_t vxb2 = wasm_i16x8_extend_low_i8x16(vbm2);
      const v128_t vxb3 = wasm_i16x8_extend_high_i8x16(vbm2);

      vacc0x0 = wasm_i32x4_add(vacc0x0, wasm_i32x4_dot_i16x8(vxa0, vxb0));
      vacc1x0 = wasm_i32x4_add(vacc1x0, wasm_i32x4_dot_i16x8(vxa1, vxb0));

      vacc0x1 = wasm_i32x4_add(vacc0x1, wasm_i32x4_dot_i16x8(vxa0, vxb1));
      vacc1x1 = wasm_i32x4_add(vacc1x1, wasm_i32x4_dot_i16x8(vxa1, vxb1));

      vacc0x2 = wasm_i32x4_add(vacc0x2, wasm_i32x4_dot_i16x8(vxa0, vxb2));
      vacc1x2 = wasm_i32x4_add(vacc1x2, wasm_i32x4_dot_i16x8(vxa1, vxb2));

      vacc0x3 = wasm_i32x4_add(vacc0x3, wasm_i32x4_dot_i16x8(vxa0, vxb3));
      vacc1x3 = wasm_i32x4_add(vacc1x3, wasm_i32x4_dot_i16x8(vxa1, vxb3));

      w = (const int8_t*) w + 32;  // only low nibble used
      k -= 8 * sizeof(int8_t);
    };

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));
    const v128_t vacc1x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x0, vacc1x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x0, vacc1x2, 2, 6, 3, 7));
    const v128_t vacc1x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x1, vacc1x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x1, vacc1x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x02, vacc1x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x02, vacc1x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_i32x4_shr(vacc0x0123, 4);
    vacc1x0123 = wasm_i32x4_shr(vacc1x0123, 4);
    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);

    const v128_t vinput_scale0 = wasm_v128_load32_splat(&quantization_params[0].inv_scale);
    const v128_t vinput_scale1 = wasm_v128_load32_splat(&quantization_params[1].inv_scale);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale0);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vinput_scale1);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vfilter_output_scale0123);
    w = (const float*) w + 4;

    const v128_t vbias0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vbias0123);
    w = (const float*) w + 4;

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);
    vacc1x0123 = wasm_f32x4_pmax(vacc1x0123, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);
    vacc1x0123 = wasm_f32x4_pmin(vacc1x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c1, vacc1x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        c0 += 2;
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        c1 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
      }
      nc = 0;
    }
    } while (nc != 0);
}
