// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c8-wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>


void xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x4c8__wasmsimd(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const v128_t vzero = wasm_f64x2_splat(0.0);
  do {
    v128_t vacc0x0 = wasm_f32x4_replace_lane(vzero, 0, ((const float*) w)[0]);
    v128_t vacc0x1 = wasm_f32x4_replace_lane(vzero, 0, ((const float*) w)[1]);
    v128_t vacc0x2 = wasm_f32x4_replace_lane(vzero, 0, ((const float*) w)[2]);
    v128_t vacc0x3 = wasm_f32x4_replace_lane(vzero, 0, ((const float*) w)[3]);
    v128_t vacc1x0 = vacc0x0;
    v128_t vacc1x1 = vacc0x1;
    v128_t vacc1x2 = vacc0x2;
    v128_t vacc1x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t k = 0;
    while (k < kc) {
      const v128_t vxa0 = wasm_i16x8_load8x8(a0);
      a0 += 8;
      const v128_t vxa1 = wasm_i16x8_load8x8(a1);
      a1 += 8;

      const v128_t vxb0 = wasm_v128_load(w);

      const v128_t vprod0x0 = wasm_i16x8_mul(vxa0, vxb0);
      vacc0x0 = wasm_i32x4_add(vacc0x0, wasm_i32x4_extend_low_i16x8(vprod0x0));
      vacc0x0 = wasm_i32x4_add(vacc0x0, wasm_i32x4_extend_high_i16x8(vprod0x0));
      const v128_t vprod1x0 = wasm_i16x8_mul(vxa1, vxb0);
      vacc1x0 = wasm_i32x4_add(vacc1x0, wasm_i32x4_extend_low_i16x8(vprod1x0));
      vacc1x0 = wasm_i32x4_add(vacc1x0, wasm_i32x4_extend_high_i16x8(vprod1x0));
      const v128_t vxb1 = wasm_v128_load((const int16_t*) w + 8);

      const v128_t vprod0x1 = wasm_i16x8_mul(vxa0, vxb1);
      vacc0x1 = wasm_i32x4_add(vacc0x1, wasm_i32x4_extend_low_i16x8(vprod0x1));
      vacc0x1 = wasm_i32x4_add(vacc0x1, wasm_i32x4_extend_high_i16x8(vprod0x1));
      const v128_t vprod1x1 = wasm_i16x8_mul(vxa1, vxb1);
      vacc1x1 = wasm_i32x4_add(vacc1x1, wasm_i32x4_extend_low_i16x8(vprod1x1));
      vacc1x1 = wasm_i32x4_add(vacc1x1, wasm_i32x4_extend_high_i16x8(vprod1x1));
      const v128_t vxb2 = wasm_v128_load((const int16_t*) w + 16);

      const v128_t vprod0x2 = wasm_i16x8_mul(vxa0, vxb2);
      vacc0x2 = wasm_i32x4_add(vacc0x2, wasm_i32x4_extend_low_i16x8(vprod0x2));
      vacc0x2 = wasm_i32x4_add(vacc0x2, wasm_i32x4_extend_high_i16x8(vprod0x2));
      const v128_t vprod1x2 = wasm_i16x8_mul(vxa1, vxb2);
      vacc1x2 = wasm_i32x4_add(vacc1x2, wasm_i32x4_extend_low_i16x8(vprod1x2));
      vacc1x2 = wasm_i32x4_add(vacc1x2, wasm_i32x4_extend_high_i16x8(vprod1x2));
      const v128_t vxb3 = wasm_v128_load((const int16_t*) w + 24);

      const v128_t vprod0x3 = wasm_i16x8_mul(vxa0, vxb3);
      vacc0x3 = wasm_i32x4_add(vacc0x3, wasm_i32x4_extend_low_i16x8(vprod0x3));
      vacc0x3 = wasm_i32x4_add(vacc0x3, wasm_i32x4_extend_high_i16x8(vprod0x3));
      const v128_t vprod1x3 = wasm_i16x8_mul(vxa1, vxb3);
      vacc1x3 = wasm_i32x4_add(vacc1x3, wasm_i32x4_extend_low_i16x8(vprod1x3));
      vacc1x3 = wasm_i32x4_add(vacc1x3, wasm_i32x4_extend_high_i16x8(vprod1x3));

      w = (const void*) ((const int16_t*) w + 32);
      k += 8 * sizeof(int8_t);
    }

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));
    const v128_t vacc1x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x0, vacc1x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x0, vacc1x2, 2, 6, 3, 7));
    const v128_t vacc1x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x1, vacc1x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x1, vacc1x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x02, vacc1x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x02, vacc1x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);

    const v128_t vscale = wasm_v128_load(params->fp32_wasmsimd.scale);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale);

    const v128_t voutput_min_less_zero_point = wasm_v128_load(params->fp32_wasmsimd.output_min_less_zero_point);
    vacc0x0123 = wasm_f32x4_max(vacc0x0123, voutput_min_less_zero_point);
    vacc1x0123 = wasm_f32x4_max(vacc1x0123, voutput_min_less_zero_point);

    const v128_t voutput_max_less_zero_point = wasm_v128_load(params->fp32_wasmsimd.output_max_less_zero_point);
    vacc0x0123 = wasm_f32x4_min(vacc0x0123, voutput_max_less_zero_point);
    vacc1x0123 = wasm_f32x4_min(vacc1x0123, voutput_max_less_zero_point);

    const v128_t vmagic_bias = wasm_v128_load(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vmagic_bias);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc1x0123 = wasm_i32x4_sub(vacc1x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc01x0123 = wasm_v16x8_shuffle(vacc0x0123, vacc1x0123, 0, 2, 4, 6, 8, 10, 12, 14);

    v128_t vout = wasm_v8x16_shuffle(vacc01x0123, vacc01x0123, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);

    if (nc >= 4) {
      *((float*) c0) = (float) wasm_f32x4_extract_lane(vout, 0);
      *((float*) c1) = (float) wasm_f32x4_extract_lane(vout, 1);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) wasm_i16x8_extract_lane(vout, 0);
        c0 += 2;
        *((uint16_t*) c1) = (uint16_t) wasm_i16x8_extract_lane(vout, 2);
        c1 += 2;
        vout = wasm_u32x4_shr(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) wasm_i8x16_extract_lane(vout, 0);
        *c1 = (int8_t) wasm_i8x16_extract_lane(vout, 4);
      }

      nc = 0;
    }
  } while (nc != 0);
}
