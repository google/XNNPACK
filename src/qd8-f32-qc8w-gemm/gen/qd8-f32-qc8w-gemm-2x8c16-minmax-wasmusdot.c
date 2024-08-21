// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c16-wasmdot.c.in
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

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c16__wasmusdot(
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
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
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

  const v128_t vsign_mask = wasm_u8x16_const_splat(UINT8_C(0x80));
  do {
    v128_t vksum0 = wasm_v128_load32_zero(w);
    v128_t vksum1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vksum2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vksum3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    v128_t vksum4 = wasm_v128_load32_zero((const int32_t*) w + 4);
    v128_t vksum5 = wasm_v128_load32_zero((const int32_t*) w + 5);
    v128_t vksum6 = wasm_v128_load32_zero((const int32_t*) w + 6);
    v128_t vksum7 = wasm_v128_load32_zero((const int32_t*) w + 7);
    const v128_t vinput_zero_point0 = wasm_i32x4_splat((int32_t) quantization_params[0].zero_point + 128);
    const v128_t vinput_zero_point1 = wasm_i32x4_splat((int32_t) quantization_params[1].zero_point + 128);
    v128_t vacc0x0 = wasm_i32x4_mul(vksum0, vinput_zero_point0);
    v128_t vacc0x1 = wasm_i32x4_mul(vksum1, vinput_zero_point0);
    v128_t vacc0x2 = wasm_i32x4_mul(vksum2, vinput_zero_point0);
    v128_t vacc0x3 = wasm_i32x4_mul(vksum3, vinput_zero_point0);
    v128_t vacc0x4 = wasm_i32x4_mul(vksum4, vinput_zero_point0);
    v128_t vacc0x5 = wasm_i32x4_mul(vksum5, vinput_zero_point0);
    v128_t vacc0x6 = wasm_i32x4_mul(vksum6, vinput_zero_point0);
    v128_t vacc0x7 = wasm_i32x4_mul(vksum7, vinput_zero_point0);
    v128_t vacc1x0 = wasm_i32x4_mul(vksum0, vinput_zero_point1);
    v128_t vacc1x1 = wasm_i32x4_mul(vksum1, vinput_zero_point1);
    v128_t vacc1x2 = wasm_i32x4_mul(vksum2, vinput_zero_point1);
    v128_t vacc1x3 = wasm_i32x4_mul(vksum3, vinput_zero_point1);
    v128_t vacc1x4 = wasm_i32x4_mul(vksum4, vinput_zero_point1);
    v128_t vacc1x5 = wasm_i32x4_mul(vksum5, vinput_zero_point1);
    v128_t vacc1x6 = wasm_i32x4_mul(vksum6, vinput_zero_point1);
    v128_t vacc1x7 = wasm_i32x4_mul(vksum7, vinput_zero_point1);
    w = (const int32_t*) w + 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_xor(wasm_v128_load(a0), vsign_mask);
      a0 += 16;
      const v128_t va1 = wasm_v128_xor(wasm_v128_load(a1), vsign_mask);
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

    const v128_t vinput_scale0 = wasm_v128_load32_splat(&quantization_params[0].inv_scale);
    const v128_t vinput_scale1 = wasm_v128_load32_splat(&quantization_params[1].inv_scale);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale0);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vinput_scale0);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vinput_scale1);
    vacc1x4567 = wasm_f32x4_mul(vacc1x4567, vinput_scale1);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vfilter_output_scale4567 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vfilter_output_scale4567);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vfilter_output_scale0123);
    vacc1x4567 = wasm_f32x4_mul(vacc1x4567, vfilter_output_scale4567);

    const v128_t vbias0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vbias4567 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);
    vacc0x4567 = wasm_f32x4_add(vacc0x4567, vbias4567);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vbias0123);
    vacc1x4567 = wasm_f32x4_add(vacc1x4567, vbias4567);

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);
    vacc0x4567 = wasm_f32x4_pmax(vacc0x4567, vmin);
    vacc1x0123 = wasm_f32x4_pmax(vacc1x0123, vmin);
    vacc1x4567 = wasm_f32x4_pmax(vacc1x4567, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);
    vacc0x4567 = wasm_f32x4_pmin(vacc0x4567, vmax);
    vacc1x0123 = wasm_f32x4_pmin(vacc1x0123, vmax);
    vacc1x4567 = wasm_f32x4_pmin(vacc1x4567, vmax);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        vacc0x0123 = vacc0x4567;
        c0 += 4;
        wasm_v128_store(c1, vacc1x0123);
        vacc1x0123 = vacc1x4567;
        c1 += 4;
      }
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
