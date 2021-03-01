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


void xnn_qs8_gemm_minmax_ukernel_3x4c8__wasmsimd_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(mr != 0);
  assert(mr <= 3);
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
    v128_t vacc2x0 = vacc0x0;
    v128_t vacc2x1 = vacc0x1;
    v128_t vacc2x2 = vacc0x2;
    v128_t vacc2x3 = vacc0x3;
    w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));

    size_t k = 0;
    while (k < kc) {
      const v128_t vxa0 = wasm_i16x8_load_8x8(a0);
      a0 += 8;
      const v128_t vxa1 = wasm_i16x8_load_8x8(a1);
      a1 += 8;
      const v128_t vxa2 = wasm_i16x8_load_8x8(a2);
      a2 += 8;

      const v128_t vxb0 = wasm_i16x8_load_8x8(w);

      const v128_t vprod0x0 = wasm_i16x8_mul(vxa0, vxb0);
      vacc0x0 = wasm_i32x4_add(vacc0x0, wasm_i32x4_widen_low_i16x8(vprod0x0));
      vacc0x0 = wasm_i32x4_add(vacc0x0, wasm_i32x4_widen_high_i16x8(vprod0x0));
      const v128_t vprod1x0 = wasm_i16x8_mul(vxa1, vxb0);
      vacc1x0 = wasm_i32x4_add(vacc1x0, wasm_i32x4_widen_low_i16x8(vprod1x0));
      vacc1x0 = wasm_i32x4_add(vacc1x0, wasm_i32x4_widen_high_i16x8(vprod1x0));
      const v128_t vprod2x0 = wasm_i16x8_mul(vxa2, vxb0);
      vacc2x0 = wasm_i32x4_add(vacc2x0, wasm_i32x4_widen_low_i16x8(vprod2x0));
      vacc2x0 = wasm_i32x4_add(vacc2x0, wasm_i32x4_widen_high_i16x8(vprod2x0));
      const v128_t vxb1 = wasm_i16x8_load_8x8((const void*) ((uintptr_t) w + 8 * sizeof(int8_t)));

      const v128_t vprod0x1 = wasm_i16x8_mul(vxa0, vxb1);
      vacc0x1 = wasm_i32x4_add(vacc0x1, wasm_i32x4_widen_low_i16x8(vprod0x1));
      vacc0x1 = wasm_i32x4_add(vacc0x1, wasm_i32x4_widen_high_i16x8(vprod0x1));
      const v128_t vprod1x1 = wasm_i16x8_mul(vxa1, vxb1);
      vacc1x1 = wasm_i32x4_add(vacc1x1, wasm_i32x4_widen_low_i16x8(vprod1x1));
      vacc1x1 = wasm_i32x4_add(vacc1x1, wasm_i32x4_widen_high_i16x8(vprod1x1));
      const v128_t vprod2x1 = wasm_i16x8_mul(vxa2, vxb1);
      vacc2x1 = wasm_i32x4_add(vacc2x1, wasm_i32x4_widen_low_i16x8(vprod2x1));
      vacc2x1 = wasm_i32x4_add(vacc2x1, wasm_i32x4_widen_high_i16x8(vprod2x1));
      const v128_t vxb2 = wasm_i16x8_load_8x8((const void*) ((uintptr_t) w + 16 * sizeof(int8_t)));

      const v128_t vprod0x2 = wasm_i16x8_mul(vxa0, vxb2);
      vacc0x2 = wasm_i32x4_add(vacc0x2, wasm_i32x4_widen_low_i16x8(vprod0x2));
      vacc0x2 = wasm_i32x4_add(vacc0x2, wasm_i32x4_widen_high_i16x8(vprod0x2));
      const v128_t vprod1x2 = wasm_i16x8_mul(vxa1, vxb2);
      vacc1x2 = wasm_i32x4_add(vacc1x2, wasm_i32x4_widen_low_i16x8(vprod1x2));
      vacc1x2 = wasm_i32x4_add(vacc1x2, wasm_i32x4_widen_high_i16x8(vprod1x2));
      const v128_t vprod2x2 = wasm_i16x8_mul(vxa2, vxb2);
      vacc2x2 = wasm_i32x4_add(vacc2x2, wasm_i32x4_widen_low_i16x8(vprod2x2));
      vacc2x2 = wasm_i32x4_add(vacc2x2, wasm_i32x4_widen_high_i16x8(vprod2x2));
      const v128_t vxb3 = wasm_i16x8_load_8x8((const void*) ((uintptr_t) w + 24 * sizeof(int8_t)));

      const v128_t vprod0x3 = wasm_i16x8_mul(vxa0, vxb3);
      vacc0x3 = wasm_i32x4_add(vacc0x3, wasm_i32x4_widen_low_i16x8(vprod0x3));
      vacc0x3 = wasm_i32x4_add(vacc0x3, wasm_i32x4_widen_high_i16x8(vprod0x3));
      const v128_t vprod1x3 = wasm_i16x8_mul(vxa1, vxb3);
      vacc1x3 = wasm_i32x4_add(vacc1x3, wasm_i32x4_widen_low_i16x8(vprod1x3));
      vacc1x3 = wasm_i32x4_add(vacc1x3, wasm_i32x4_widen_high_i16x8(vprod1x3));
      const v128_t vprod2x3 = wasm_i16x8_mul(vxa2, vxb3);
      vacc2x3 = wasm_i32x4_add(vacc2x3, wasm_i32x4_widen_low_i16x8(vprod2x3));
      vacc2x3 = wasm_i32x4_add(vacc2x3, wasm_i32x4_widen_high_i16x8(vprod2x3));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int8_t));
      k += 8 * sizeof(int8_t);
    }

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));
    const v128_t vacc1x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x0, vacc1x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x0, vacc1x2, 2, 6, 3, 7));
    const v128_t vacc1x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x1, vacc1x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x1, vacc1x3, 2, 6, 3, 7));
    const v128_t vacc2x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x0, vacc2x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x0, vacc2x2, 2, 6, 3, 7));
    const v128_t vacc2x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x1, vacc2x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x1, vacc2x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x02, vacc1x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x02, vacc1x13, 2, 6, 3, 7));
    v128_t vacc2x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x02, vacc2x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x02, vacc2x13, 2, 6, 3, 7));

    const v128_t vsign0x0123 = wasm_i32x4_lt(vacc0x0123, vzero);
    const v128_t vsign1x0123 = wasm_i32x4_lt(vacc1x0123, vzero);
    const v128_t vsign2x0123 = wasm_i32x4_lt(vacc2x0123, vzero);

    const v128_t vacc0x01 = wasm_v32x4_shuffle(vacc0x0123, vsign0x0123, 0, 4, 1, 5);
    const v128_t vacc1x01 = wasm_v32x4_shuffle(vacc1x0123, vsign1x0123, 0, 4, 1, 5);
    const v128_t vacc2x01 = wasm_v32x4_shuffle(vacc2x0123, vsign2x0123, 0, 4, 1, 5);

    const v128_t vmultiplier = wasm_v128_load(params->wasmsimd.multiplier);
    const v128_t vrounding = wasm_v128_load(params->wasmsimd.rounding);
    const v128_t vprod0x01 = wasm_i64x2_add(wasm_i64x2_mul(vacc0x01, vmultiplier), vrounding);
    const v128_t vacc0x23 = wasm_v32x4_shuffle(vacc0x0123, vsign0x0123, 2, 6, 3, 7);
    const v128_t vprod1x01 = wasm_i64x2_add(wasm_i64x2_mul(vacc1x01, vmultiplier), vrounding);
    const v128_t vacc1x23 = wasm_v32x4_shuffle(vacc1x0123, vsign1x0123, 2, 6, 3, 7);
    const v128_t vprod2x01 = wasm_i64x2_add(wasm_i64x2_mul(vacc2x01, vmultiplier), vrounding);
    const v128_t vacc2x23 = wasm_v32x4_shuffle(vacc2x0123, vsign2x0123, 2, 6, 3, 7);

    const v128_t vprod0x23 = wasm_i64x2_add(wasm_i64x2_mul(vacc0x23, vmultiplier), vrounding);
    const v128_t vprod1x23 = wasm_i64x2_add(wasm_i64x2_mul(vacc1x23, vmultiplier), vrounding);
    const v128_t vprod2x23 = wasm_i64x2_add(wasm_i64x2_mul(vacc2x23, vmultiplier), vrounding);

    const v128_t vq31prod0x0123 = wasm_v32x4_shuffle(vprod0x01, vprod0x23, 1, 3, 5, 7);
    const v128_t vq31prod1x0123 = wasm_v32x4_shuffle(vprod1x01, vprod1x23, 1, 3, 5, 7);
    const v128_t vq31prod2x0123 = wasm_v32x4_shuffle(vprod2x01, vprod2x23, 1, 3, 5, 7);

    const v128_t vremainder_mask = wasm_v128_load(params->wasmsimd.remainder_mask);
    const v128_t vrem0x0123 = wasm_i32x4_add(wasm_v128_and(vq31prod0x0123, vremainder_mask), wasm_i32x4_lt(vq31prod0x0123, vzero));
    const v128_t vrem1x0123 = wasm_i32x4_add(wasm_v128_and(vq31prod1x0123, vremainder_mask), wasm_i32x4_lt(vq31prod1x0123, vzero));
    const v128_t vrem2x0123 = wasm_i32x4_add(wasm_v128_and(vq31prod2x0123, vremainder_mask), wasm_i32x4_lt(vq31prod2x0123, vzero));

    const v128_t vthreshold = wasm_v128_load(params->wasmsimd.remainder_threshold);
    const int32_t vshift = params->wasmsimd.shift;
    vacc0x0123 = wasm_i32x4_sub(wasm_i32x4_shr(vq31prod0x0123, vshift), wasm_i32x4_gt(vrem0x0123, vthreshold));
    vacc1x0123 = wasm_i32x4_sub(wasm_i32x4_shr(vq31prod1x0123, vshift), wasm_i32x4_gt(vrem1x0123, vthreshold));
    vacc2x0123 = wasm_i32x4_sub(wasm_i32x4_shr(vq31prod2x0123, vshift), wasm_i32x4_gt(vrem2x0123, vthreshold));

    const v128_t voutput_zero_point = wasm_v128_load(params->wasmsimd.output_zero_point);
    v128_t vacc01x0123 = wasm_i16x8_add_saturate(wasm_i16x8_narrow_i32x4(vacc0x0123, vacc1x0123), voutput_zero_point);
    v128_t vacc22x0123 = wasm_i16x8_add_saturate(wasm_i16x8_narrow_i32x4(vacc2x0123, vacc2x0123), voutput_zero_point);

    v128_t vout = wasm_i8x16_narrow_i16x8(vacc01x0123, vacc22x0123);

    const v128_t voutput_min = wasm_v128_load(params->wasmsimd.output_min);
    vout = wasm_i8x16_max(vout, voutput_min);

    const v128_t voutput_max = wasm_v128_load(params->wasmsimd.output_max);
    vout = wasm_i8x16_min(vout, voutput_max);

    if (nc >= 4) {
      *((float*) c0) = (float) wasm_f32x4_extract_lane(vout, 0);
      *((float*) c1) = (float) wasm_f32x4_extract_lane(vout, 1);
      *((float*) c2) = (float) wasm_f32x4_extract_lane(vout, 2);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) wasm_i16x8_extract_lane(vout, 0);
        c0 += 2;
        *((uint16_t*) c1) = (uint16_t) wasm_i16x8_extract_lane(vout, 2);
        c1 += 2;
        *((uint16_t*) c2) = (uint16_t) wasm_i16x8_extract_lane(vout, 4);
        c2 += 2;
        vout = wasm_u32x4_shr(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) wasm_i8x16_extract_lane(vout, 0);
        *c1 = (int8_t) wasm_i8x16_extract_lane(vout, 4);
        *c2 = (int8_t) wasm_i8x16_extract_lane(vout, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}
