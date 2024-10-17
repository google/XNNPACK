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


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__wasmusdot_u2(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

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
    v128_t vacc0x0x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point);
    v128_t vacc0x0x4567 = wasm_i32x4_mul(vksum4567, vinput_zero_point);
    v128_t vacc0x0x89AB = wasm_i32x4_mul(vksum89AB, vinput_zero_point);
    v128_t vacc0x0xCDEF = wasm_i32x4_mul(vksumCDEF, vinput_zero_point);
    v128_t vacc1x0x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point);
    v128_t vacc1x0x4567 = wasm_i32x4_mul(vksum4567, vinput_zero_point);
    v128_t vacc1x0x89AB = wasm_i32x4_mul(vksum89AB, vinput_zero_point);
    v128_t vacc1x0xCDEF = wasm_i32x4_mul(vksumCDEF, vinput_zero_point);
    v128_t vacc2x0x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point);
    v128_t vacc2x0x4567 = wasm_i32x4_mul(vksum4567, vinput_zero_point);
    v128_t vacc2x0x89AB = wasm_i32x4_mul(vksum89AB, vinput_zero_point);
    v128_t vacc2x0xCDEF = wasm_i32x4_mul(vksumCDEF, vinput_zero_point);
    v128_t vacc3x0x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point);
    v128_t vacc3x0x4567 = wasm_i32x4_mul(vksum4567, vinput_zero_point);
    v128_t vacc3x0x89AB = wasm_i32x4_mul(vksum89AB, vinput_zero_point);
    v128_t vacc3x0xCDEF = wasm_i32x4_mul(vksumCDEF, vinput_zero_point);
    w = (const int32_t*) w + 16;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      a += 4;

      size_t k = kc;
      while (k >= 8 * sizeof(int8_t)) {
        v128_t va0x0x0123 = wasm_v128_load32_splat(a0);
        v128_t va0x1x0123 = wasm_v128_load32_splat(a0 + 4);
        a0 += 8;
        v128_t va1x0x0123 = wasm_v128_load32_splat(a1);
        v128_t va1x1x0123 = wasm_v128_load32_splat(a1 + 4);
        a1 += 8;
        v128_t va2x0x0123 = wasm_v128_load32_splat(a2);
        v128_t va2x1x0123 = wasm_v128_load32_splat(a2 + 4);
        a2 += 8;
        v128_t va3x0x0123 = wasm_v128_load32_splat(a3);
        v128_t va3x1x0123 = wasm_v128_load32_splat(a3 + 4);
        a3 += 8;

        va0x0x0123 = wasm_v128_xor(va0x0x0123, vsign_mask);
        va0x1x0123 = wasm_v128_xor(va0x1x0123, vsign_mask);
        va1x0x0123 = wasm_v128_xor(va1x0x0123, vsign_mask);
        va1x1x0123 = wasm_v128_xor(va1x1x0123, vsign_mask);
        va2x0x0123 = wasm_v128_xor(va2x0x0123, vsign_mask);
        va2x1x0123 = wasm_v128_xor(va2x1x0123, vsign_mask);
        va3x0x0123 = wasm_v128_xor(va3x0x0123, vsign_mask);
        va3x1x0123 = wasm_v128_xor(va3x1x0123, vsign_mask);

        const v128_t vb0x0123 = wasm_v128_load((const int8_t*) w);
        const v128_t vb0x4567 = wasm_v128_load((const int8_t*) w + 16);
        const v128_t vb0x89AB = wasm_v128_load((const int8_t*) w + 32);
        const v128_t vb0xCDEF = wasm_v128_load((const int8_t*) w + 48);
        const v128_t vb1x0123 = wasm_v128_load((const int8_t*) w + 64);
        const v128_t vb1x4567 = wasm_v128_load((const int8_t*) w + 80);
        const v128_t vb1x89AB = wasm_v128_load((const int8_t*) w + 96);
        const v128_t vb1xCDEF = wasm_v128_load((const int8_t*) w + 112);

        vacc0x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x0123, va0x0x0123, vacc0x0x0123);
        vacc0x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x4567, va0x0x0123, vacc0x0x4567);
        vacc0x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x89AB, va0x0x0123, vacc0x0x89AB);
        vacc0x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0xCDEF, va0x0x0123, vacc0x0xCDEF);
        vacc1x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x0123, va1x0x0123, vacc1x0x0123);
        vacc1x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x4567, va1x0x0123, vacc1x0x4567);
        vacc1x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x89AB, va1x0x0123, vacc1x0x89AB);
        vacc1x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0xCDEF, va1x0x0123, vacc1x0xCDEF);
        vacc2x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x0123, va2x0x0123, vacc2x0x0123);
        vacc2x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x4567, va2x0x0123, vacc2x0x4567);
        vacc2x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x89AB, va2x0x0123, vacc2x0x89AB);
        vacc2x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0xCDEF, va2x0x0123, vacc2x0xCDEF);
        vacc3x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x0123, va3x0x0123, vacc3x0x0123);
        vacc3x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x4567, va3x0x0123, vacc3x0x4567);
        vacc3x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0x89AB, va3x0x0123, vacc3x0x89AB);
        vacc3x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0xCDEF, va3x0x0123, vacc3x0xCDEF);
        vacc0x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x0123, va0x1x0123, vacc0x0x0123);
        vacc0x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x4567, va0x1x0123, vacc0x0x4567);
        vacc0x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x89AB, va0x1x0123, vacc0x0x89AB);
        vacc0x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1xCDEF, va0x1x0123, vacc0x0xCDEF);
        vacc1x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x0123, va1x1x0123, vacc1x0x0123);
        vacc1x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x4567, va1x1x0123, vacc1x0x4567);
        vacc1x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x89AB, va1x1x0123, vacc1x0x89AB);
        vacc1x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1xCDEF, va1x1x0123, vacc1x0xCDEF);
        vacc2x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x0123, va2x1x0123, vacc2x0x0123);
        vacc2x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x4567, va2x1x0123, vacc2x0x4567);
        vacc2x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x89AB, va2x1x0123, vacc2x0x89AB);
        vacc2x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1xCDEF, va2x1x0123, vacc2x0xCDEF);
        vacc3x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x0123, va3x1x0123, vacc3x0x0123);
        vacc3x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x4567, va3x1x0123, vacc3x0x4567);
        vacc3x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1x89AB, va3x1x0123, vacc3x0x89AB);
        vacc3x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1xCDEF, va3x1x0123, vacc3x0xCDEF);

        w = (const int8_t*) w + 128;
        k -= 8 * sizeof(int8_t);
      }

      while (k != 0) {
        v128_t va0x0123 = wasm_v128_load32_splat(a0);
        a0 += 4;
        v128_t va1x0123 = wasm_v128_load32_splat(a1);
        a1 += 4;
        v128_t va2x0123 = wasm_v128_load32_splat(a2);
        a2 += 4;
        v128_t va3x0123 = wasm_v128_load32_splat(a3);
        a3 += 4;

        va0x0123 = wasm_v128_xor(va0x0123, vsign_mask);
        va1x0123 = wasm_v128_xor(va1x0123, vsign_mask);
        va2x0123 = wasm_v128_xor(va2x0123, vsign_mask);
        va3x0123 = wasm_v128_xor(va3x0123, vsign_mask);

        const v128_t vb0123 = wasm_v128_load((const int8_t*) w);
        const v128_t vb4567 = wasm_v128_load((const int8_t*) w + 16);
        const v128_t vb89AB = wasm_v128_load((const int8_t*) w + 32);
        const v128_t vbCDEF = wasm_v128_load((const int8_t*) w + 48);

        vacc0x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0123, va0x0123, vacc0x0x0123);
        vacc0x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb4567, va0x0123, vacc0x0x4567);
        vacc0x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb89AB, va0x0123, vacc0x0x89AB);
        vacc0x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vbCDEF, va0x0123, vacc0x0xCDEF);
        vacc1x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0123, va1x0123, vacc1x0x0123);
        vacc1x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb4567, va1x0123, vacc1x0x4567);
        vacc1x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb89AB, va1x0123, vacc1x0x89AB);
        vacc1x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vbCDEF, va1x0123, vacc1x0xCDEF);
        vacc2x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0123, va2x0123, vacc2x0x0123);
        vacc2x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb4567, va2x0123, vacc2x0x4567);
        vacc2x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb89AB, va2x0123, vacc2x0x89AB);
        vacc2x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vbCDEF, va2x0123, vacc2x0xCDEF);
        vacc3x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0123, va3x0123, vacc3x0x0123);
        vacc3x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb4567, va3x0123, vacc3x0x4567);
        vacc3x0x89AB = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb89AB, va3x0123, vacc3x0x89AB);
        vacc3x0xCDEF = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vbCDEF, va3x0123, vacc3x0xCDEF);

        w = (const int8_t*) w + 64;
        k -= 4 * sizeof(int8_t);
      }

      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc0x0x0123 = wasm_f32x4_convert_i32x4(vacc0x0x0123);
    vacc0x0x4567 = wasm_f32x4_convert_i32x4(vacc0x0x4567);
    vacc0x0x89AB = wasm_f32x4_convert_i32x4(vacc0x0x89AB);
    vacc0x0xCDEF = wasm_f32x4_convert_i32x4(vacc0x0xCDEF);
    vacc1x0x0123 = wasm_f32x4_convert_i32x4(vacc1x0x0123);
    vacc1x0x4567 = wasm_f32x4_convert_i32x4(vacc1x0x4567);
    vacc1x0x89AB = wasm_f32x4_convert_i32x4(vacc1x0x89AB);
    vacc1x0xCDEF = wasm_f32x4_convert_i32x4(vacc1x0xCDEF);
    vacc2x0x0123 = wasm_f32x4_convert_i32x4(vacc2x0x0123);
    vacc2x0x4567 = wasm_f32x4_convert_i32x4(vacc2x0x4567);
    vacc2x0x89AB = wasm_f32x4_convert_i32x4(vacc2x0x89AB);
    vacc2x0xCDEF = wasm_f32x4_convert_i32x4(vacc2x0xCDEF);
    vacc3x0x0123 = wasm_f32x4_convert_i32x4(vacc3x0x0123);
    vacc3x0x4567 = wasm_f32x4_convert_i32x4(vacc3x0x4567);
    vacc3x0x89AB = wasm_f32x4_convert_i32x4(vacc3x0x89AB);
    vacc3x0xCDEF = wasm_f32x4_convert_i32x4(vacc3x0xCDEF);

    const v128_t vinput_scale = wasm_v128_load32_splat(&quantization_params->inv_scale);

    vacc0x0x0123 = wasm_f32x4_mul(vacc0x0x0123, vinput_scale);
    vacc0x0x4567 = wasm_f32x4_mul(vacc0x0x4567, vinput_scale);
    vacc0x0x89AB = wasm_f32x4_mul(vacc0x0x89AB, vinput_scale);
    vacc0x0xCDEF = wasm_f32x4_mul(vacc0x0xCDEF, vinput_scale);
    vacc1x0x0123 = wasm_f32x4_mul(vacc1x0x0123, vinput_scale);
    vacc1x0x4567 = wasm_f32x4_mul(vacc1x0x4567, vinput_scale);
    vacc1x0x89AB = wasm_f32x4_mul(vacc1x0x89AB, vinput_scale);
    vacc1x0xCDEF = wasm_f32x4_mul(vacc1x0xCDEF, vinput_scale);
    vacc2x0x0123 = wasm_f32x4_mul(vacc2x0x0123, vinput_scale);
    vacc2x0x4567 = wasm_f32x4_mul(vacc2x0x4567, vinput_scale);
    vacc2x0x89AB = wasm_f32x4_mul(vacc2x0x89AB, vinput_scale);
    vacc2x0xCDEF = wasm_f32x4_mul(vacc2x0xCDEF, vinput_scale);
    vacc3x0x0123 = wasm_f32x4_mul(vacc3x0x0123, vinput_scale);
    vacc3x0x4567 = wasm_f32x4_mul(vacc3x0x4567, vinput_scale);
    vacc3x0x89AB = wasm_f32x4_mul(vacc3x0x89AB, vinput_scale);
    vacc3x0xCDEF = wasm_f32x4_mul(vacc3x0xCDEF, vinput_scale);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vfilter_output_scale4567 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vfilter_output_scale89AB = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vfilter_output_scaleCDEF = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0x0123 = wasm_f32x4_mul(vacc0x0x0123, vfilter_output_scale0123);
    vacc0x0x4567 = wasm_f32x4_mul(vacc0x0x4567, vfilter_output_scale4567);
    vacc0x0x89AB = wasm_f32x4_mul(vacc0x0x89AB, vfilter_output_scale89AB);
    vacc0x0xCDEF = wasm_f32x4_mul(vacc0x0xCDEF, vfilter_output_scaleCDEF);
    vacc1x0x0123 = wasm_f32x4_mul(vacc1x0x0123, vfilter_output_scale0123);
    vacc1x0x4567 = wasm_f32x4_mul(vacc1x0x4567, vfilter_output_scale4567);
    vacc1x0x89AB = wasm_f32x4_mul(vacc1x0x89AB, vfilter_output_scale89AB);
    vacc1x0xCDEF = wasm_f32x4_mul(vacc1x0xCDEF, vfilter_output_scaleCDEF);
    vacc2x0x0123 = wasm_f32x4_mul(vacc2x0x0123, vfilter_output_scale0123);
    vacc2x0x4567 = wasm_f32x4_mul(vacc2x0x4567, vfilter_output_scale4567);
    vacc2x0x89AB = wasm_f32x4_mul(vacc2x0x89AB, vfilter_output_scale89AB);
    vacc2x0xCDEF = wasm_f32x4_mul(vacc2x0xCDEF, vfilter_output_scaleCDEF);
    vacc3x0x0123 = wasm_f32x4_mul(vacc3x0x0123, vfilter_output_scale0123);
    vacc3x0x4567 = wasm_f32x4_mul(vacc3x0x4567, vfilter_output_scale4567);
    vacc3x0x89AB = wasm_f32x4_mul(vacc3x0x89AB, vfilter_output_scale89AB);
    vacc3x0xCDEF = wasm_f32x4_mul(vacc3x0xCDEF, vfilter_output_scaleCDEF);

    const v128_t vbias0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vbias4567 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vbias89AB = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vbiasCDEF = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0x0123 = wasm_f32x4_add(vacc0x0x0123, vbias0123);
    vacc0x0x4567 = wasm_f32x4_add(vacc0x0x4567, vbias4567);
    vacc0x0x89AB = wasm_f32x4_add(vacc0x0x89AB, vbias89AB);
    vacc0x0xCDEF = wasm_f32x4_add(vacc0x0xCDEF, vbiasCDEF);
    vacc1x0x0123 = wasm_f32x4_add(vacc1x0x0123, vbias0123);
    vacc1x0x4567 = wasm_f32x4_add(vacc1x0x4567, vbias4567);
    vacc1x0x89AB = wasm_f32x4_add(vacc1x0x89AB, vbias89AB);
    vacc1x0xCDEF = wasm_f32x4_add(vacc1x0xCDEF, vbiasCDEF);
    vacc2x0x0123 = wasm_f32x4_add(vacc2x0x0123, vbias0123);
    vacc2x0x4567 = wasm_f32x4_add(vacc2x0x4567, vbias4567);
    vacc2x0x89AB = wasm_f32x4_add(vacc2x0x89AB, vbias89AB);
    vacc2x0xCDEF = wasm_f32x4_add(vacc2x0xCDEF, vbiasCDEF);
    vacc3x0x0123 = wasm_f32x4_add(vacc3x0x0123, vbias0123);
    vacc3x0x4567 = wasm_f32x4_add(vacc3x0x4567, vbias4567);
    vacc3x0x89AB = wasm_f32x4_add(vacc3x0x89AB, vbias89AB);
    vacc3x0xCDEF = wasm_f32x4_add(vacc3x0xCDEF, vbiasCDEF);

    vacc0x0x0123 = wasm_f32x4_pmax(vacc0x0x0123, vmin);
    vacc0x0x4567 = wasm_f32x4_pmax(vacc0x0x4567, vmin);
    vacc0x0x89AB = wasm_f32x4_pmax(vacc0x0x89AB, vmin);
    vacc0x0xCDEF = wasm_f32x4_pmax(vacc0x0xCDEF, vmin);
    vacc1x0x0123 = wasm_f32x4_pmax(vacc1x0x0123, vmin);
    vacc1x0x4567 = wasm_f32x4_pmax(vacc1x0x4567, vmin);
    vacc1x0x89AB = wasm_f32x4_pmax(vacc1x0x89AB, vmin);
    vacc1x0xCDEF = wasm_f32x4_pmax(vacc1x0xCDEF, vmin);
    vacc2x0x0123 = wasm_f32x4_pmax(vacc2x0x0123, vmin);
    vacc2x0x4567 = wasm_f32x4_pmax(vacc2x0x4567, vmin);
    vacc2x0x89AB = wasm_f32x4_pmax(vacc2x0x89AB, vmin);
    vacc2x0xCDEF = wasm_f32x4_pmax(vacc2x0xCDEF, vmin);
    vacc3x0x0123 = wasm_f32x4_pmax(vacc3x0x0123, vmin);
    vacc3x0x4567 = wasm_f32x4_pmax(vacc3x0x4567, vmin);
    vacc3x0x89AB = wasm_f32x4_pmax(vacc3x0x89AB, vmin);
    vacc3x0xCDEF = wasm_f32x4_pmax(vacc3x0xCDEF, vmin);

    vacc0x0x0123 = wasm_f32x4_pmin(vacc0x0x0123, vmax);
    vacc0x0x4567 = wasm_f32x4_pmin(vacc0x0x4567, vmax);
    vacc0x0x89AB = wasm_f32x4_pmin(vacc0x0x89AB, vmax);
    vacc0x0xCDEF = wasm_f32x4_pmin(vacc0x0xCDEF, vmax);
    vacc1x0x0123 = wasm_f32x4_pmin(vacc1x0x0123, vmax);
    vacc1x0x4567 = wasm_f32x4_pmin(vacc1x0x4567, vmax);
    vacc1x0x89AB = wasm_f32x4_pmin(vacc1x0x89AB, vmax);
    vacc1x0xCDEF = wasm_f32x4_pmin(vacc1x0xCDEF, vmax);
    vacc2x0x0123 = wasm_f32x4_pmin(vacc2x0x0123, vmax);
    vacc2x0x4567 = wasm_f32x4_pmin(vacc2x0x4567, vmax);
    vacc2x0x89AB = wasm_f32x4_pmin(vacc2x0x89AB, vmax);
    vacc2x0xCDEF = wasm_f32x4_pmin(vacc2x0xCDEF, vmax);
    vacc3x0x0123 = wasm_f32x4_pmin(vacc3x0x0123, vmax);
    vacc3x0x4567 = wasm_f32x4_pmin(vacc3x0x4567, vmax);
    vacc3x0x89AB = wasm_f32x4_pmin(vacc3x0x89AB, vmax);
    vacc3x0xCDEF = wasm_f32x4_pmin(vacc3x0xCDEF, vmax);

    if XNN_LIKELY(nc >= 16) {
      wasm_v128_store(c3, vacc3x0x0123);
      wasm_v128_store(c3 + 4, vacc3x0x4567);
      wasm_v128_store(c3 + 8, vacc3x0x89AB);
      wasm_v128_store(c3 + 12, vacc3x0xCDEF);
      wasm_v128_store(c2, vacc2x0x0123);
      wasm_v128_store(c2 + 4, vacc2x0x4567);
      wasm_v128_store(c2 + 8, vacc2x0x89AB);
      wasm_v128_store(c2 + 12, vacc2x0xCDEF);
      wasm_v128_store(c1, vacc1x0x0123);
      wasm_v128_store(c1 + 4, vacc1x0x4567);
      wasm_v128_store(c1 + 8, vacc1x0x89AB);
      wasm_v128_store(c1 + 12, vacc1x0xCDEF);
      wasm_v128_store(c0, vacc0x0x0123);
      wasm_v128_store(c0 + 4, vacc0x0x4567);
      wasm_v128_store(c0 + 8, vacc0x0x89AB);
      wasm_v128_store(c0 + 12, vacc0x0xCDEF);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      if (nc & 8) {
        wasm_v128_store(c3, vacc3x0x0123);
        vacc3x0x0123 = vacc3x0x89AB;
        c3 += 4;
        wasm_v128_store(c3, vacc3x0x4567);
        vacc3x0x4567 = vacc3x0xCDEF;
        c3 += 4;
        wasm_v128_store(c2, vacc2x0x0123);
        vacc2x0x0123 = vacc2x0x89AB;
        c2 += 4;
        wasm_v128_store(c2, vacc2x0x4567);
        vacc2x0x4567 = vacc2x0xCDEF;
        c2 += 4;
        wasm_v128_store(c1, vacc1x0x0123);
        vacc1x0x0123 = vacc1x0x89AB;
        c1 += 4;
        wasm_v128_store(c1, vacc1x0x4567);
        vacc1x0x4567 = vacc1x0xCDEF;
        c1 += 4;
        wasm_v128_store(c0, vacc0x0x0123);
        vacc0x0x0123 = vacc0x0x89AB;
        c0 += 4;
        wasm_v128_store(c0, vacc0x0x4567);
        vacc0x0x4567 = vacc0x0xCDEF;
        c0 += 4;
      }
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0x0123);
        vacc3x0x0123 = vacc3x0x4567;
        c3 += 4;
        wasm_v128_store(c2, vacc2x0x0123);
        vacc2x0x0123 = vacc2x0x4567;
        c2 += 4;
        wasm_v128_store(c1, vacc1x0x0123);
        vacc1x0x0123 = vacc1x0x4567;
        c1 += 4;
        wasm_v128_store(c0, vacc0x0x0123);
        vacc0x0x0123 = vacc0x0x4567;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0x0123, 0);
        vacc3x0x0123 = wasm_v64x2_shuffle(vacc3x0x0123, vacc3x0x0123, 1, 1);
        c3 += 2;
        wasm_v128_store64_lane(c2, vacc2x0x0123, 0);
        vacc2x0x0123 = wasm_v64x2_shuffle(vacc2x0x0123, vacc2x0x0123, 1, 1);
        c2 += 2;
        wasm_v128_store64_lane(c1, vacc1x0x0123, 0);
        vacc1x0x0123 = wasm_v64x2_shuffle(vacc1x0x0123, vacc1x0x0123, 1, 1);
        c1 += 2;
        wasm_v128_store64_lane(c0, vacc0x0x0123, 0);
        vacc0x0x0123 = wasm_v64x2_shuffle(vacc0x0x0123, vacc0x0x0123, 1, 1);
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0x0123, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
