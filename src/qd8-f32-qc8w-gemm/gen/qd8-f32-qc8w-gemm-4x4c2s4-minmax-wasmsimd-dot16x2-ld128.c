// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    const v128_t vksum0123 = wasm_v128_load(w);
    const v128_t vinput_zero_point0 = wasm_v128_load32_splat(&quantization_params[0].zero_point);
    const v128_t vinput_zero_point1 = wasm_v128_load32_splat(&quantization_params[1].zero_point);
    const v128_t vinput_zero_point2 = wasm_v128_load32_splat(&quantization_params[2].zero_point);
    const v128_t vinput_zero_point3 = wasm_v128_load32_splat(&quantization_params[3].zero_point);
    v128_t vacc0x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point0);
    v128_t vacc1x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point1);
    v128_t vacc2x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point2);
    v128_t vacc3x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point3);
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      v128_t vxa0 = wasm_i16x8_load8x8((const v128_t*) a0);
      a0 += 8;
      v128_t vxa1 = wasm_i16x8_load8x8((const v128_t*) a1);
      a1 += 8;
      v128_t vxa2 = wasm_i16x8_load8x8((const v128_t*) a2);
      a2 += 8;
      v128_t vxa3 = wasm_i16x8_load8x8((const v128_t*) a3);
      a3 += 8;

      const v128_t vb01 = wasm_v128_load(w);
      const v128_t vxb0 = wasm_i16x8_extend_low_i8x16(vb01);
      const v128_t vxb1 = wasm_i16x8_extend_high_i8x16(vb01);

      vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb0));
      vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
      vacc1x0123 = wasm_i32x4_add(vacc1x0123, wasm_i32x4_dot_i16x8(vxa1, vxb0));
      vxa1 = wasm_v32x4_shuffle(vxa1, vxa1, 1, 2, 3, 4);
      vacc2x0123 = wasm_i32x4_add(vacc2x0123, wasm_i32x4_dot_i16x8(vxa2, vxb0));
      vxa2 = wasm_v32x4_shuffle(vxa2, vxa2, 1, 2, 3, 4);
      vacc3x0123 = wasm_i32x4_add(vacc3x0123, wasm_i32x4_dot_i16x8(vxa3, vxb0));
      vxa3 = wasm_v32x4_shuffle(vxa3, vxa3, 1, 2, 3, 4);

      vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb1));
      vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
      vacc1x0123 = wasm_i32x4_add(vacc1x0123, wasm_i32x4_dot_i16x8(vxa1, vxb1));
      vxa1 = wasm_v32x4_shuffle(vxa1, vxa1, 1, 2, 3, 4);
      vacc2x0123 = wasm_i32x4_add(vacc2x0123, wasm_i32x4_dot_i16x8(vxa2, vxb1));
      vxa2 = wasm_v32x4_shuffle(vxa2, vxa2, 1, 2, 3, 4);
      vacc3x0123 = wasm_i32x4_add(vacc3x0123, wasm_i32x4_dot_i16x8(vxa3, vxb1));
      vxa3 = wasm_v32x4_shuffle(vxa3, vxa3, 1, 2, 3, 4);
      const v128_t vb23 = wasm_v128_load((const int8_t*) w + 16);
      const v128_t vxb2 = wasm_i16x8_extend_low_i8x16(vb23);
      const v128_t vxb3 = wasm_i16x8_extend_high_i8x16(vb23);

      vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb2));
      vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
      vacc1x0123 = wasm_i32x4_add(vacc1x0123, wasm_i32x4_dot_i16x8(vxa1, vxb2));
      vxa1 = wasm_v32x4_shuffle(vxa1, vxa1, 1, 2, 3, 4);
      vacc2x0123 = wasm_i32x4_add(vacc2x0123, wasm_i32x4_dot_i16x8(vxa2, vxb2));
      vxa2 = wasm_v32x4_shuffle(vxa2, vxa2, 1, 2, 3, 4);
      vacc3x0123 = wasm_i32x4_add(vacc3x0123, wasm_i32x4_dot_i16x8(vxa3, vxb2));
      vxa3 = wasm_v32x4_shuffle(vxa3, vxa3, 1, 2, 3, 4);

      vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb3));
      vacc1x0123 = wasm_i32x4_add(vacc1x0123, wasm_i32x4_dot_i16x8(vxa1, vxb3));
      vacc2x0123 = wasm_i32x4_add(vacc2x0123, wasm_i32x4_dot_i16x8(vxa2, vxb3));
      vacc3x0123 = wasm_i32x4_add(vacc3x0123, wasm_i32x4_dot_i16x8(vxa3, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    } while (k != 0);

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);
    vacc2x0123 = wasm_f32x4_convert_i32x4(vacc2x0123);
    vacc3x0123 = wasm_f32x4_convert_i32x4(vacc3x0123);

    const v128_t vinput_scale0 = wasm_v128_load32_splat(&quantization_params[0].inv_scale);
    const v128_t vinput_scale1 = wasm_v128_load32_splat(&quantization_params[1].inv_scale);
    const v128_t vinput_scale2 = wasm_v128_load32_splat(&quantization_params[2].inv_scale);
    const v128_t vinput_scale3 = wasm_v128_load32_splat(&quantization_params[3].inv_scale);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale0);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vinput_scale1);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vinput_scale2);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vinput_scale3);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vfilter_output_scale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vfilter_output_scale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vfilter_output_scale0123);
    w = (const float*) w + 4;

    const v128_t vbias0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vbias0123);
    vacc2x0123 = wasm_f32x4_add(vacc2x0123, vbias0123);
    vacc3x0123 = wasm_f32x4_add(vacc3x0123, vbias0123);
    w = (const float*) w + 4;

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);
    vacc1x0123 = wasm_f32x4_pmax(vacc1x0123, vmin);
    vacc2x0123 = wasm_f32x4_pmax(vacc2x0123, vmin);
    vacc3x0123 = wasm_f32x4_pmax(vacc3x0123, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);
    vacc1x0123 = wasm_f32x4_pmin(vacc1x0123, vmax);
    vacc2x0123 = wasm_f32x4_pmin(vacc2x0123, vmax);
    vacc3x0123 = wasm_f32x4_pmin(vacc3x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c3, vacc3x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        c0 += 2;
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        c1 += 2;
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        c2 += 2;
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        c3 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
