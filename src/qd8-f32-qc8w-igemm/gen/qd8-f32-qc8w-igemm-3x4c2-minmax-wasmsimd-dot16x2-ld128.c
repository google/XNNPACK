// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/igemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld128(
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
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 2 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  const v128_t vinput_zero_point = wasm_v128_load32_splat(&quantization_params->zero_point);
  const v128_t vinput_scale = wasm_v128_load32_splat(&quantization_params->inv_scale);
  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const v128_t vksum0123 = wasm_v128_load(w);
    v128_t vacc0x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point);
    v128_t vacc1x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point);
    v128_t vacc2x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point);
    w = (const void*) ((const int32_t*) w + 4);

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
      a += 3;

      size_t k = kc;
      while (k >= 8 * sizeof(int8_t)) {
        const v128_t vxa0 = wasm_i16x8_load8x8(a0);
        a0 += 8;
        const v128_t vxa1 = wasm_i16x8_load8x8(a1);
        a1 += 8;
        const v128_t vxa2 = wasm_i16x8_load8x8(a2);
        a2 += 8;

        const v128_t vb01 = wasm_v128_load(w);
        const v128_t vxb0 = wasm_i16x8_extend_low_i8x16(vb01);
        const v128_t vxb1 = wasm_i16x8_extend_high_i8x16(vb01);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 0, 0, 0, 0), vxb0));
        vacc1x0123 = wasm_i32x4_add(vacc1x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa1, vxa1, 0, 0, 0, 0), vxb0));
        vacc2x0123 = wasm_i32x4_add(vacc2x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa2, vxa2, 0, 0, 0, 0), vxb0));

        vacc0x0123 = wasm_i32x4_add(vacc0x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 1, 1, 1, 1), vxb1));
        vacc1x0123 = wasm_i32x4_add(vacc1x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa1, vxa1, 1, 1, 1, 1), vxb1));
        vacc2x0123 = wasm_i32x4_add(vacc2x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa2, vxa2, 1, 1, 1, 1), vxb1));
        const v128_t vb23 = wasm_v128_load((const int8_t*) w + 16);
        const v128_t vxb2 = wasm_i16x8_extend_low_i8x16(vb23);
        const v128_t vxb3 = wasm_i16x8_extend_high_i8x16(vb23);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 2, 2, 2, 2), vxb2));
        vacc1x0123 = wasm_i32x4_add(vacc1x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa1, vxa1, 2, 2, 2, 2), vxb2));
        vacc2x0123 = wasm_i32x4_add(vacc2x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa2, vxa2, 2, 2, 2, 2), vxb2));

        vacc0x0123 = wasm_i32x4_add(vacc0x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 3, 3, 3, 3), vxb3));
        vacc1x0123 = wasm_i32x4_add(vacc1x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa1, vxa1, 3, 3, 3, 3), vxb3));
        vacc2x0123 = wasm_i32x4_add(vacc2x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa2, vxa2, 3, 3, 3, 3), vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k -= 8 * sizeof(int8_t);
      }
      if (k != 0) {
        const v128_t vxa0 = wasm_i16x8_load8x8(a0);
        a0 = (const int8_t*) ((uintptr_t) a0 + k);
        const v128_t vxa1 = wasm_i16x8_load8x8(a1);
        a1 = (const int8_t*) ((uintptr_t) a1 + k);
        const v128_t vxa2 = wasm_i16x8_load8x8(a2);
        a2 = (const int8_t*) ((uintptr_t) a2 + k);

        const v128_t vxb0 = wasm_i16x8_load8x8(w);
        w = (const int8_t*) w + 8;

        vacc0x0123 = wasm_i32x4_add(vacc0x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 0, 0, 0, 0), vxb0));
        vacc1x0123 = wasm_i32x4_add(vacc1x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa1, vxa1, 0, 0, 0, 0), vxb0));
        vacc2x0123 = wasm_i32x4_add(vacc2x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa2, vxa2, 0, 0, 0, 0), vxb0));

        if (k > 2 * sizeof(int8_t)) {
          const v128_t vxb1 = wasm_i16x8_load8x8(w);
          w = (const int8_t*) w + 8;

          vacc0x0123 = wasm_i32x4_add(vacc0x0123,
            wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 1, 1, 1, 1), vxb1));
          vacc1x0123 = wasm_i32x4_add(vacc1x0123,
            wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa1, vxa1, 1, 1, 1, 1), vxb1));
          vacc2x0123 = wasm_i32x4_add(vacc2x0123,
            wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa2, vxa2, 1, 1, 1, 1), vxb1));

          if (k > 4 * sizeof(int8_t)) {
            const v128_t vxb2 = wasm_i16x8_load8x8(w);
            w = (const int8_t*) w + 8;

            vacc0x0123 = wasm_i32x4_add(vacc0x0123,
              wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 2, 2, 2, 2), vxb2));
            vacc1x0123 = wasm_i32x4_add(vacc1x0123,
              wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa1, vxa1, 2, 2, 2, 2), vxb2));
            vacc2x0123 = wasm_i32x4_add(vacc2x0123,
              wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa2, vxa2, 2, 2, 2, 2), vxb2));
          }
        }
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);
    vacc2x0123 = wasm_f32x4_convert_i32x4(vacc2x0123);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vinput_scale);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vinput_scale);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vfilter_output_scale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vfilter_output_scale0123);
    w = (const float*) w + 4;

    const v128_t vbias0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vbias0123);
    vacc2x0123 = wasm_f32x4_add(vacc2x0123, vbias0123);
    w = (const float*) w + 4;

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);
    vacc1x0123 = wasm_f32x4_pmax(vacc1x0123, vmin);
    vacc2x0123 = wasm_f32x4_pmax(vacc2x0123, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);
    vacc1x0123 = wasm_f32x4_pmin(vacc1x0123, vmax);
    vacc2x0123 = wasm_f32x4_pmin(vacc2x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c0, vacc0x0123);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        c2 += 2;
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        c1 += 2;
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
