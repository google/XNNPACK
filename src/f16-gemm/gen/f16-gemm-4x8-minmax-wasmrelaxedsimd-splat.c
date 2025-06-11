// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/wasmrelaxedsimd-splat.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/gemm.h"

void xnn_f16_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_float16* restrict a,
    size_t a_stride,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;

  const v128_t vmin = wasm_v128_load16_splat(&params->scalar.min);;
  const v128_t vmax = wasm_v128_load16_splat(&params->scalar.max);;

  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    v128_t vacc0x0 = wasm_v128_load(w);
    v128_t vacc1x0 = vacc0x0;
    v128_t vacc2x0 = vacc0x0;
    v128_t vacc3x0 = vacc0x0;
    w = (const xnn_float16*) w + 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_i16x8_splat(*a0);
      a0 += 1;
      const v128_t va1 = wasm_i16x8_splat(*a1);
      a1 += 1;
      const v128_t va2 = wasm_i16x8_splat(*a2);
      a2 += 1;
      const v128_t va3 = wasm_i16x8_splat(*a3);
      a3 += 1;

      const v128_t vb0 = wasm_v128_load(w);
      w = (const xnn_float16*) w + 8;

      vacc0x0 = wasm_f16x8_relaxed_madd(va0, vb0, vacc0x0);
      vacc1x0 = wasm_f16x8_relaxed_madd(va1, vb0, vacc1x0);
      vacc2x0 = wasm_f16x8_relaxed_madd(va2, vb0, vacc2x0);
      vacc3x0 = wasm_f16x8_relaxed_madd(va3, vb0, vacc3x0);

      k -= sizeof(uint16_t);
    } while (k != 0);

    vacc0x0 = wasm_f16x8_pmax(vacc0x0, vmin);
    vacc1x0 = wasm_f16x8_pmax(vacc1x0, vmin);
    vacc2x0 = wasm_f16x8_pmax(vacc2x0, vmin);
    vacc3x0 = wasm_f16x8_pmax(vacc3x0, vmin);

    vacc0x0 = wasm_f16x8_pmin(vacc0x0, vmax);
    vacc1x0 = wasm_f16x8_pmin(vacc1x0, vmax);
    vacc2x0 = wasm_f16x8_pmin(vacc2x0, vmax);
    vacc3x0 = wasm_f16x8_pmin(vacc3x0, vmax);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint16_t*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      v128_t vh0x0 = vacc0x0;
      v128_t vh1x0 = vacc1x0;
      v128_t vh2x0 = vacc2x0;
      v128_t vh3x0 = vacc3x0;
      if (nc & 4) {
        wasm_v128_store64_lane(c0, vh0x0, 0);
        wasm_v128_store64_lane(c1, vh1x0, 0);
        wasm_v128_store64_lane(c2, vh2x0, 0);
        wasm_v128_store64_lane(c3, vh3x0, 0);

        vh0x0 = wasm_i64x2_shuffle(vh0x0, vh0x0, 1, 1);
        vh1x0 = wasm_i64x2_shuffle(vh1x0, vh1x0, 1, 1);
        vh2x0 = wasm_i64x2_shuffle(vh2x0, vh2x0, 1, 1);
        vh3x0 = wasm_i64x2_shuffle(vh3x0, vh3x0, 1, 1);

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        wasm_v128_store32_lane(c0, vh0x0, 0);
        wasm_v128_store32_lane(c1, vh1x0, 0);
        wasm_v128_store32_lane(c2, vh2x0, 0);
        wasm_v128_store32_lane(c3, vh3x0, 0);

        wasm_i32x4_shuffle(vh0x0, vh0x0, 1, 2, 3, 1);
        wasm_i32x4_shuffle(vh1x0, vh1x0, 1, 2, 3, 1);
        wasm_i32x4_shuffle(vh2x0, vh2x0, 1, 2, 3, 1);
        wasm_i32x4_shuffle(vh3x0, vh3x0, 1, 2, 3, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        wasm_v128_store16_lane(c0, vh0x0, 0);
        wasm_v128_store16_lane(c1, vh1x0, 0);
        wasm_v128_store16_lane(c2, vh2x0, 0);
        wasm_v128_store16_lane(c3, vh3x0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
