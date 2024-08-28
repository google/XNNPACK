// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c8-wasmdot.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmsdot_u2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }


  const v128_t vmagic_bias = wasm_f32x4_const_splat(12582912.0f);
  const int32_t output_min_less_zero_point = (int32_t) params->fp32_scalar.output_min - (int32_t) params->fp32_scalar.output_zero_point;
  const v128_t vmagic_min = wasm_i32x4_splat((int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point));
  const v128_t vmagic_bias_less_output_zero_point = wasm_i32x4_splat(INT32_C(0x4B400000) - (int32_t) params->fp32_scalar.output_zero_point);
  const v128_t voutput_max = wasm_i8x16_splat(params->fp32_scalar.output_max);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vmagic_min);
  XNN_FORCE_REALIZATION(vmagic_bias_less_output_zero_point);
  XNN_FORCE_REALIZATION(voutput_max);

  do {
    v128_t vacc0x01 = wasm_u64x2_load32x2(w); w = (const int32_t*) w + 2;
    v128_t vacc0x23 = wasm_u64x2_load32x2(w); w = (const int32_t*) w + 2;
    v128_t vacc0x45 = wasm_u64x2_load32x2(w); w = (const int32_t*) w + 2;
    v128_t vacc0x67 = wasm_u64x2_load32x2(w); w = (const int32_t*) w + 2;
    v128_t vacc1x01 = vacc0x01;
    v128_t vacc2x01 = vacc0x01;
    v128_t vacc1x23 = vacc0x23;
    v128_t vacc2x23 = vacc0x23;
    v128_t vacc1x45 = vacc0x45;
    v128_t vacc2x45 = vacc0x45;
    v128_t vacc1x67 = vacc0x67;
    v128_t vacc2x67 = vacc0x67;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const v128_t va0x01 = wasm_v128_load64_splat(a0);
      const v128_t va0x23 = wasm_v128_load64_splat((const int8_t*) a0 + 8);
      a0 += 16;
      const v128_t va1x01 = wasm_v128_load64_splat(a1);
      const v128_t va1x23 = wasm_v128_load64_splat((const int8_t*) a1 + 8);
      a1 += 16;
      const v128_t va2x01 = wasm_v128_load64_splat(a2);
      const v128_t va2x23 = wasm_v128_load64_splat((const int8_t*) a2 + 8);
      a2 += 16;

      const v128_t vb01x01 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb23x01 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb45x01 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb67x01 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb01x23 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb23x23 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb45x23 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb67x23 = wasm_v128_load(w); w = (const int8_t*) w + 16;

      vacc0x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01x01, va0x01, vacc0x01);
      vacc0x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23x01, va0x01, vacc0x23);
      vacc0x45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb45x01, va0x01, vacc0x45);
      vacc0x67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb67x01, va0x01, vacc0x67);
      vacc1x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01x01, va1x01, vacc1x01);
      vacc1x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23x01, va1x01, vacc1x23);
      vacc1x45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb45x01, va1x01, vacc1x45);
      vacc1x67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb67x01, va1x01, vacc1x67);
      vacc2x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01x01, va2x01, vacc2x01);
      vacc2x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23x01, va2x01, vacc2x23);
      vacc2x45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb45x01, va2x01, vacc2x45);
      vacc2x67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb67x01, va2x01, vacc2x67);
      vacc0x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01x23, va0x23, vacc0x01);
      vacc0x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23x23, va0x23, vacc0x23);
      vacc0x45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb45x23, va0x23, vacc0x45);
      vacc0x67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb67x23, va0x23, vacc0x67);
      vacc1x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01x23, va1x23, vacc1x01);
      vacc1x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23x23, va1x23, vacc1x23);
      vacc1x45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb45x23, va1x23, vacc1x45);
      vacc1x67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb67x23, va1x23, vacc1x67);
      vacc2x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01x23, va2x23, vacc2x01);
      vacc2x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23x23, va2x23, vacc2x23);
      vacc2x45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb45x23, va2x23, vacc2x45);
      vacc2x67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb67x23, va2x23, vacc2x67);

      k -= 16 * sizeof(int8_t);
    }

    if (k != 0) {
      const v128_t va0x01 = wasm_v128_load64_splat(a0);
      a0 += 8;
      const v128_t va1x01 = wasm_v128_load64_splat(a1);
      a1 += 8;
      const v128_t va2x01 = wasm_v128_load64_splat(a2);
      a2 += 8;

      const v128_t vb01 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb23 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb45 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb67 = wasm_v128_load(w); w = (const int8_t*) w + 16;

      vacc0x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va0x01, vacc0x01);
      vacc0x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va0x01, vacc0x23);
      vacc0x45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb45, va0x01, vacc0x45);
      vacc0x67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb67, va0x01, vacc0x67);
      vacc1x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va1x01, vacc1x01);
      vacc1x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va1x01, vacc1x23);
      vacc1x45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb45, va1x01, vacc1x45);
      vacc1x67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb67, va1x01, vacc1x67);
      vacc2x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va2x01, vacc2x01);
      vacc2x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va2x01, vacc2x23);
      vacc2x45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb45, va2x01, vacc2x45);
      vacc2x67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb67, va2x01, vacc2x67);

      k -= 8 * sizeof(int8_t);
    }

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x01, vacc0x23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc0x01, vacc0x23, 1, 3, 5, 7));
    v128_t vacc0x4567 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x45, vacc0x67, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc0x45, vacc0x67, 1, 3, 5, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x01, vacc1x23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc1x01, vacc1x23, 1, 3, 5, 7));
    v128_t vacc1x4567 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x45, vacc1x67, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc1x45, vacc1x67, 1, 3, 5, 7));
    v128_t vacc2x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x01, vacc2x23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc2x01, vacc2x23, 1, 3, 5, 7));
    v128_t vacc2x4567 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x45, vacc2x67, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc2x45, vacc2x67, 1, 3, 5, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc0x4567 = wasm_f32x4_convert_i32x4(vacc0x4567);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);
    vacc1x4567 = wasm_f32x4_convert_i32x4(vacc1x4567);
    vacc2x0123 = wasm_f32x4_convert_i32x4(vacc2x0123);
    vacc2x4567 = wasm_f32x4_convert_i32x4(vacc2x4567);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vscale4567 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc1x4567 = wasm_f32x4_mul(vacc1x4567, vscale4567);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vscale0123);
    vacc2x4567 = wasm_f32x4_mul(vacc2x4567, vscale4567);

    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);
    vacc0x4567 = wasm_f32x4_add(vacc0x4567, vmagic_bias);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vmagic_bias);
    vacc1x4567 = wasm_f32x4_add(vacc1x4567, vmagic_bias);
    vacc2x0123 = wasm_f32x4_add(vacc2x0123, vmagic_bias);
    vacc2x4567 = wasm_f32x4_add(vacc2x4567, vmagic_bias);

    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vmagic_min);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vmagic_min);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vmagic_min);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vmagic_min);
    vacc2x4567 = wasm_i32x4_max(vacc2x4567, vmagic_min);

    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc0x4567 = wasm_i32x4_sub(vacc0x4567, vmagic_bias_less_output_zero_point);
    vacc1x0123 = wasm_i32x4_sub(vacc1x0123, vmagic_bias_less_output_zero_point);
    vacc1x4567 = wasm_i32x4_sub(vacc1x4567, vmagic_bias_less_output_zero_point);
    vacc2x0123 = wasm_i32x4_sub(vacc2x0123, vmagic_bias_less_output_zero_point);
    vacc2x4567 = wasm_i32x4_sub(vacc2x4567, vmagic_bias_less_output_zero_point);

    v128_t vacc0x01234567 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc0x4567);
    v128_t vacc1x01234567 = wasm_i16x8_narrow_i32x4(vacc1x0123, vacc1x4567);
    v128_t vacc2x01234567 = wasm_i16x8_narrow_i32x4(vacc2x0123, vacc2x4567);

    v128_t vacc0x01234567_1x01234567 = wasm_i8x16_narrow_i16x8(vacc0x01234567, vacc1x01234567);
    v128_t vacc2x01234567_2x01234567 = wasm_i8x16_narrow_i16x8(vacc2x01234567, vacc2x01234567);

    vacc0x01234567_1x01234567 = wasm_i8x16_min(vacc0x01234567_1x01234567, voutput_max);
    vacc2x01234567_2x01234567 = wasm_i8x16_min(vacc2x01234567_2x01234567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store64_lane(c0, vacc0x01234567_1x01234567, 0);
      wasm_v128_store64_lane(c1, vacc0x01234567_1x01234567, 1);
      wasm_v128_store64_lane(c2, vacc2x01234567_2x01234567, 0);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store32_lane(c0, vacc0x01234567_1x01234567, 0);
        c0 += 4;
        wasm_v128_store32_lane(c1, vacc0x01234567_1x01234567, 2);
        c1 += 4;
        wasm_v128_store32_lane(c2, vacc2x01234567_2x01234567, 0);
        c2 += 4;

        vacc0x01234567_1x01234567 = wasm_u64x2_shr(vacc0x01234567_1x01234567, 32);
        vacc2x01234567_2x01234567 = wasm_u64x2_shr(vacc2x01234567_2x01234567, 32);
      }
      if (nc & 2) {
        wasm_v128_store16_lane(c0, vacc0x01234567_1x01234567, 0);
        c0 += 2;
        wasm_v128_store16_lane(c1, vacc0x01234567_1x01234567, 4);
        c1 += 2;
        wasm_v128_store16_lane(c2, vacc2x01234567_2x01234567, 0);
        c2 += 2;

        vacc0x01234567_1x01234567 = wasm_u32x4_shr(vacc0x01234567_1x01234567, 16);
        vacc2x01234567_2x01234567 = wasm_u32x4_shr(vacc2x01234567_2x01234567, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c0, vacc0x01234567_1x01234567, 0);
        wasm_v128_store8_lane(c1, vacc0x01234567_1x01234567, 8);
        wasm_v128_store8_lane(c2, vacc2x01234567_2x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
