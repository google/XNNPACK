// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx16c4-wasmdot.c.in
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


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__wasmusdot(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const v128_t vsign_mask = wasm_u8x16_const_splat(UINT8_C(0x80));
  XNN_FORCE_REALIZATION(vsign_mask);
  do {
    v128_t vksum0123 = wasm_v128_load((const int32_t*) w);
    v128_t vksum4567 = wasm_v128_load((const int32_t*) w + 4);
    v128_t vksum89AB = wasm_v128_load((const int32_t*) w + 8);
    v128_t vksumCDEF = wasm_v128_load((const int32_t*) w + 12);
    const v128_t vinput_zero_point = wasm_i32x4_splat((int32_t) quantization_params->zero_point + 128);
    v128_t vacc0x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point);
    v128_t vacc0x4567 = wasm_i32x4_mul(vksum4567, vinput_zero_point);
    v128_t vacc0x89AB = wasm_i32x4_mul(vksum89AB, vinput_zero_point);
    v128_t vacc0xCDEF = wasm_i32x4_mul(vksumCDEF, vinput_zero_point);
    w = (const int32_t*) w + 16;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = kc;

      while (k != 0) {
        v128_t va0x0123 = wasm_v128_load32_splat(a0);
        a0 += 4;

        va0x0123 = wasm_v128_xor(va0x0123, vsign_mask);

        const v128_t vb0123 = wasm_v128_load((const int8_t*) w);
        const v128_t vb4567 = wasm_v128_load((const int8_t*) w + 16);
        const v128_t vb89AB = wasm_v128_load((const int8_t*) w + 32);
        const v128_t vbCDEF = wasm_v128_load((const int8_t*) w + 48);

        vacc0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0123, va0x0123, vacc0x0123);
        vacc0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb4567, va0x0123, vacc0x4567);
        vacc0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb89AB, va0x0123, vacc0x89AB);
        vacc0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vbCDEF, va0x0123, vacc0xCDEF);

        w = (const int8_t*) w + 64;
        k -= 4 * sizeof(int8_t);
      }

      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc0x4567 = wasm_f32x4_convert_i32x4(vacc0x4567);
    vacc0x89AB = wasm_f32x4_convert_i32x4(vacc0x89AB);
    vacc0xCDEF = wasm_f32x4_convert_i32x4(vacc0xCDEF);

    const v128_t vinput_scale = wasm_v128_load32_splat(&quantization_params->inv_scale);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vinput_scale);
    vacc0x89AB = wasm_f32x4_mul(vacc0x89AB, vinput_scale);
    vacc0xCDEF = wasm_f32x4_mul(vacc0xCDEF, vinput_scale);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vfilter_output_scale4567 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vfilter_output_scale89AB = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vfilter_output_scaleCDEF = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vfilter_output_scale4567);
    vacc0x89AB = wasm_f32x4_mul(vacc0x89AB, vfilter_output_scale89AB);
    vacc0xCDEF = wasm_f32x4_mul(vacc0xCDEF, vfilter_output_scaleCDEF);

    const v128_t vbias0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vbias4567 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vbias89AB = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vbiasCDEF = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);
    vacc0x4567 = wasm_f32x4_add(vacc0x4567, vbias4567);
    vacc0x89AB = wasm_f32x4_add(vacc0x89AB, vbias89AB);
    vacc0xCDEF = wasm_f32x4_add(vacc0xCDEF, vbiasCDEF);

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);
    vacc0x4567 = wasm_f32x4_pmax(vacc0x4567, vmin);
    vacc0x89AB = wasm_f32x4_pmax(vacc0x89AB, vmin);
    vacc0xCDEF = wasm_f32x4_pmax(vacc0xCDEF, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);
    vacc0x4567 = wasm_f32x4_pmin(vacc0x4567, vmax);
    vacc0x89AB = wasm_f32x4_pmin(vacc0x89AB, vmax);
    vacc0xCDEF = wasm_f32x4_pmin(vacc0xCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      wasm_v128_store(c0 + 8, vacc0x89AB);
      wasm_v128_store(c0 + 12, vacc0xCDEF);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      if (nc & 8) {
        wasm_v128_store(c0, vacc0x0123);
        vacc0x0123 = vacc0x89AB;
        c0 += 4;
        wasm_v128_store(c0, vacc0x4567);
        vacc0x4567 = vacc0xCDEF;
        c0 += 4;
      }
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
