// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/wasmsimd-loadsplat.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/gemm.h"


void xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    w = (const float*) w + 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;

      const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
      const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
      const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
      const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
      w = (const int8_t*) w + 8;

      vacc0x0123 = wasm_f32x4_add(vacc0x0123, wasm_f32x4_mul(va0, vb0123));
      vacc0x4567 = wasm_f32x4_add(vacc0x4567, wasm_f32x4_mul(va0, vb4567));

      k -= sizeof(float);
    } while (k != 0);

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    w = (const float*) w + 8;
    vacc0x0123 = wasm_f32x4_max(vmin, vacc0x0123);
    vacc0x4567 = wasm_f32x4_max(vmin, vacc0x4567);

    vacc0x0123 = wasm_f32x4_min(vmax, vacc0x0123);
    vacc0x4567 = wasm_f32x4_min(vmax, vacc0x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

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
