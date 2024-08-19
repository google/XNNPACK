// Auto-generated file. Do not edit!
//   Template: src/f32-ppmm/wasmsimd-splat.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/ppmm.h"


void xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat(
  size_t mr,
  size_t nc,
  size_t kc,
  const float* restrict a,
  const float* restrict w,
  float* restrict c,
  size_t cm_stride,
  size_t cn_stride,
  const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

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

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    do {
      const v128_t va0123 = wasm_v128_load(a);
      a += 4;

      const v128_t vb0123 = wasm_v128_load(w);
      const v128_t vb4567 = wasm_v128_load(w + 4);
      w += 8;

      const v128_t va0000 = wasm_v32x4_shuffle(va0123, va0123, 0, 0, 0, 0);
      const v128_t va1111 = wasm_v32x4_shuffle(va0123, va0123, 1, 1, 1, 1);
      const v128_t va2222 = wasm_v32x4_shuffle(va0123, va0123, 2, 2, 2, 2);
      const v128_t va3333 = wasm_v32x4_shuffle(va0123, va0123, 3, 3, 3, 3);

      vacc0x0123 = wasm_f32x4_add(vacc0x0123, wasm_f32x4_mul(va0000, vb0123));
      vacc1x0123 = wasm_f32x4_add(vacc1x0123, wasm_f32x4_mul(va1111, vb0123));
      vacc2x0123 = wasm_f32x4_add(vacc2x0123, wasm_f32x4_mul(va2222, vb0123));
      vacc3x0123 = wasm_f32x4_add(vacc3x0123, wasm_f32x4_mul(va3333, vb0123));
      vacc0x4567 = wasm_f32x4_add(vacc0x4567, wasm_f32x4_mul(va0000, vb4567));
      vacc1x4567 = wasm_f32x4_add(vacc1x4567, wasm_f32x4_mul(va1111, vb4567));
      vacc2x4567 = wasm_f32x4_add(vacc2x4567, wasm_f32x4_mul(va2222, vb4567));
      vacc3x4567 = wasm_f32x4_add(vacc3x4567, wasm_f32x4_mul(va3333, vb4567));

      k -= sizeof(float);
    } while (k != 0);

    vacc0x0123 = wasm_f32x4_pmax(vmin, vacc0x0123);
    vacc1x0123 = wasm_f32x4_pmax(vmin, vacc1x0123);
    vacc2x0123 = wasm_f32x4_pmax(vmin, vacc2x0123);
    vacc3x0123 = wasm_f32x4_pmax(vmin, vacc3x0123);
    vacc0x4567 = wasm_f32x4_pmax(vmin, vacc0x4567);
    vacc1x4567 = wasm_f32x4_pmax(vmin, vacc1x4567);
    vacc2x4567 = wasm_f32x4_pmax(vmin, vacc2x4567);
    vacc3x4567 = wasm_f32x4_pmax(vmin, vacc3x4567);

    vacc0x0123 = wasm_f32x4_pmin(vmax, vacc0x0123);
    vacc1x0123 = wasm_f32x4_pmin(vmax, vacc1x0123);
    vacc2x0123 = wasm_f32x4_pmin(vmax, vacc2x0123);
    vacc3x0123 = wasm_f32x4_pmin(vmax, vacc3x0123);
    vacc0x4567 = wasm_f32x4_pmin(vmax, vacc0x4567);
    vacc1x4567 = wasm_f32x4_pmin(vmax, vacc1x4567);
    vacc2x4567 = wasm_f32x4_pmin(vmax, vacc2x4567);
    vacc3x4567 = wasm_f32x4_pmin(vmax, vacc3x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);

      a = (const float*) ((uintptr_t) a - kc * 4);

      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
