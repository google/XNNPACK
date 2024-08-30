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


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__wasmusdot_u2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const v128_t vsign_mask = wasm_u8x16_const_splat(UINT8_C(0x80));
  do {
    const v128_t vinput_zero_point0 = wasm_i32x4_splat((int32_t) quantization_params[0].zero_point + 128);
    v128_t vksum0123 = wasm_v128_load(w); w = (const int32_t*) w + 4;
    v128_t vsum0x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point0);
    v128_t vacc0x01 = wasm_u64x2_extend_low_u32x4(vsum0x0123);
    v128_t vacc0x23 = wasm_u64x2_extend_high_u32x4(vsum0x0123);
    v128_t vksum4567 = wasm_v128_load(w); w = (const int32_t*) w + 4;
    v128_t vsum0x4567 = wasm_i32x4_mul(vksum4567, vinput_zero_point0);
    v128_t vacc0x45 = wasm_u64x2_extend_low_u32x4(vsum0x4567);
    v128_t vacc0x67 = wasm_u64x2_extend_high_u32x4(vsum0x4567);

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const v128_t va0x01 = wasm_v128_xor(wasm_v128_load64_splat(a0), vsign_mask);
      const v128_t va0x23 = wasm_v128_xor(wasm_v128_load64_splat((const int8_t*) a0 + 8), vsign_mask);
      a0 += 16;

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
      vacc0x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01x23, va0x23, vacc0x01);
      vacc0x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23x23, va0x23, vacc0x23);
      vacc0x45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb45x23, va0x23, vacc0x45);
      vacc0x67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb67x23, va0x23, vacc0x67);

      k -= 16 * sizeof(int8_t);
    }

    if (k != 0) {
      const v128_t va0x01 = wasm_v128_xor(wasm_v128_load64_splat(a0), vsign_mask);
      a0 += 8;

      const v128_t vb01 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb23 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb45 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb67 = wasm_v128_load(w); w = (const int8_t*) w + 16;

      vacc0x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va0x01, vacc0x01);
      vacc0x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va0x01, vacc0x23);
      vacc0x45 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb45, va0x01, vacc0x45);
      vacc0x67 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb67, va0x01, vacc0x67);

      k -= 8 * sizeof(int8_t);
    }

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x01, vacc0x23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc0x01, vacc0x23, 1, 3, 5, 7));
    v128_t vacc0x4567 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x45, vacc0x67, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc0x45, vacc0x67, 1, 3, 5, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc0x4567 = wasm_f32x4_convert_i32x4(vacc0x4567);

    const v128_t vinput_scale0 = wasm_v128_load32_splat(&quantization_params[0].inv_scale);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale0);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vinput_scale0);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vfilter_output_scale4567 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vfilter_output_scale4567);

    const v128_t vbias0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vbias4567 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);
    vacc0x4567 = wasm_f32x4_add(vacc0x4567, vbias4567);

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);
    vacc0x4567 = wasm_f32x4_pmax(vacc0x4567, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);
    vacc0x4567 = wasm_f32x4_pmin(vacc0x4567, vmax);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        vacc0x0123 = vacc0x4567;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
