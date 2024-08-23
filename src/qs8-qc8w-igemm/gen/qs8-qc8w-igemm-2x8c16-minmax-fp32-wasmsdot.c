// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx4c16-wasmdot.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c16__wasmsdot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
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
    v128_t vacc0x0 = wasm_v128_load32_zero(w);
    v128_t vacc0x1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vacc0x2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vacc0x3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    v128_t vacc0x4 = wasm_v128_load32_zero((const int32_t*) w + 4);
    v128_t vacc0x5 = wasm_v128_load32_zero((const int32_t*) w + 5);
    v128_t vacc0x6 = wasm_v128_load32_zero((const int32_t*) w + 6);
    v128_t vacc0x7 = wasm_v128_load32_zero((const int32_t*) w + 7);
    v128_t vacc1x0 = vacc0x0;
    v128_t vacc1x1 = vacc0x1;
    v128_t vacc1x2 = vacc0x2;
    v128_t vacc1x3 = vacc0x3;
    v128_t vacc1x4 = vacc0x4;
    v128_t vacc1x5 = vacc0x5;
    v128_t vacc1x6 = vacc0x6;
    v128_t vacc1x7 = vacc0x7;
    w = (const int32_t*) w + 8;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = kc;
      do {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 16;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 16;

        const v128_t vb0 = wasm_v128_load(w);
        vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
        vacc1x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va1, vacc1x0);
        const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);
        vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
        vacc1x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va1, vacc1x1);
        const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);
        vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
        vacc1x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va1, vacc1x2);
        const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);
        vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);
        vacc1x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va1, vacc1x3);
        const v128_t vb4 = wasm_v128_load((const int8_t*) w + 64);
        vacc0x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb4, va0, vacc0x4);
        vacc1x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb4, va1, vacc1x4);
        const v128_t vb5 = wasm_v128_load((const int8_t*) w + 80);
        vacc0x5 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb5, va0, vacc0x5);
        vacc1x5 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb5, va1, vacc1x5);
        const v128_t vb6 = wasm_v128_load((const int8_t*) w + 96);
        vacc0x6 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb6, va0, vacc0x6);
        vacc1x6 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb6, va1, vacc1x6);
        const v128_t vb7 = wasm_v128_load((const int8_t*) w + 112);
        vacc0x7 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb7, va0, vacc0x7);
        vacc1x7 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb7, va1, vacc1x7);

        w = (const int8_t*) w + 128;
        k -= 16 * sizeof(int8_t);
      } while (k != 0);
      p -= 2 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));
    const v128_t vacc0x46 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x4, vacc0x6, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x4, vacc0x6, 2, 6, 3, 7));
    const v128_t vacc0x57 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x5, vacc0x7, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x5, vacc0x7, 2, 6, 3, 7));
    const v128_t vacc1x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x0, vacc1x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x0, vacc1x2, 2, 6, 3, 7));
    const v128_t vacc1x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x1, vacc1x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x1, vacc1x3, 2, 6, 3, 7));
    const v128_t vacc1x46 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x4, vacc1x6, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x4, vacc1x6, 2, 6, 3, 7));
    const v128_t vacc1x57 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x5, vacc1x7, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x5, vacc1x7, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));
    v128_t vacc0x4567 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x46, vacc0x57, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x46, vacc0x57, 2, 6, 3, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x02, vacc1x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x02, vacc1x13, 2, 6, 3, 7));
    v128_t vacc1x4567 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x46, vacc1x57, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x46, vacc1x57, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc0x4567 = wasm_f32x4_convert_i32x4(vacc0x4567);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);
    vacc1x4567 = wasm_f32x4_convert_i32x4(vacc1x4567);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vscale4567 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc1x4567 = wasm_f32x4_mul(vacc1x4567, vscale4567);

    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);
    vacc0x4567 = wasm_f32x4_add(vacc0x4567, vmagic_bias);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vmagic_bias);
    vacc1x4567 = wasm_f32x4_add(vacc1x4567, vmagic_bias);

    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vmagic_min);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vmagic_min);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vmagic_min);

    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc0x4567 = wasm_i32x4_sub(vacc0x4567, vmagic_bias_less_output_zero_point);
    vacc1x0123 = wasm_i32x4_sub(vacc1x0123, vmagic_bias_less_output_zero_point);
    vacc1x4567 = wasm_i32x4_sub(vacc1x4567, vmagic_bias_less_output_zero_point);

    v128_t vacc0x01234567 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc0x4567);
    v128_t vacc1x01234567 = wasm_i16x8_narrow_i32x4(vacc1x0123, vacc1x4567);

    v128_t vout0x01234567_1x01234567 = wasm_i8x16_narrow_i16x8(vacc0x01234567, vacc1x01234567);

    vout0x01234567_1x01234567 = wasm_i8x16_min(vout0x01234567_1x01234567, voutput_max);

    if (nc >= 8) {
      wasm_v128_store64_lane(c1, vout0x01234567_1x01234567, 1);
      wasm_v128_store64_lane(c0, vout0x01234567_1x01234567, 0);

      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store32_lane(c1, vout0x01234567_1x01234567, 2);
        c1 += 4;
        wasm_v128_store32_lane(c0, vout0x01234567_1x01234567, 0);
        c0 += 4;

        vout0x01234567_1x01234567 = wasm_u64x2_shr(vout0x01234567_1x01234567, 32);
      }
      if (nc & 2) {
        wasm_v128_store16_lane(c1, vout0x01234567_1x01234567, 4);
        c1 += 2;
        wasm_v128_store16_lane(c0, vout0x01234567_1x01234567, 0);
        c0 += 2;

        vout0x01234567_1x01234567 = wasm_u32x4_shr(vout0x01234567_1x01234567, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c1, vout0x01234567_1x01234567, 8);
        wasm_v128_store8_lane(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
