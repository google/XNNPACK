// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/wasmsimd-loadsplat.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "src/xnnpack/igemm.h"


void xnn_f32_igemm_minmax_ukernel_4x16__wasmrelaxedsimd_fma_loadsplat_u2(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

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
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc0x89AB = wasm_v128_load(w + 8);
    v128_t vacc0xCDEF = wasm_v128_load(w + 12);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc1x89AB = vacc0x89AB;
    v128_t vacc1xCDEF = vacc0xCDEF;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc2x89AB = vacc0x89AB;
    v128_t vacc2xCDEF = vacc0xCDEF;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc3x89AB = vacc0x89AB;
    v128_t vacc3xCDEF = vacc0xCDEF;
    w += 16;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      while (k >= 2 * sizeof(float)) {
        const v128_t vb0x0123 = wasm_v128_load(w);
        const v128_t vb0x4567 = wasm_v128_load(w + 4);
        const v128_t vb0x89AB = wasm_v128_load(w + 8);
        const v128_t vb0xCDEF = wasm_v128_load(w + 12);
        const v128_t vb1x0123 = wasm_v128_load(w + 16);
        const v128_t vb1x4567 = wasm_v128_load(w + 20);
        const v128_t vb1x89AB = wasm_v128_load(w + 24);
        const v128_t vb1xCDEF = wasm_v128_load(w + 28);
        w += 32;

        const v128_t va0x0 = wasm_v128_load32_splat(a0);
        const v128_t va0x1 = wasm_v128_load32_splat(a0 + 1);
        a0 += 2;
        const v128_t va1x0 = wasm_v128_load32_splat(a1);
        const v128_t va1x1 = wasm_v128_load32_splat(a1 + 1);
        a1 += 2;
        const v128_t va2x0 = wasm_v128_load32_splat(a2);
        const v128_t va2x1 = wasm_v128_load32_splat(a2 + 1);
        a2 += 2;
        const v128_t va3x0 = wasm_v128_load32_splat(a3);
        const v128_t va3x1 = wasm_v128_load32_splat(a3 + 1);
        a3 += 2;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0x0, vb0x0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0x0, vb0x4567, vacc0x4567);
        vacc0x89AB = wasm_f32x4_relaxed_madd(va0x0, vb0x89AB, vacc0x89AB);
        vacc0xCDEF = wasm_f32x4_relaxed_madd(va0x0, vb0xCDEF, vacc0xCDEF);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1x0, vb0x0123, vacc1x0123);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1x0, vb0x4567, vacc1x4567);
        vacc1x89AB = wasm_f32x4_relaxed_madd(va1x0, vb0x89AB, vacc1x89AB);
        vacc1xCDEF = wasm_f32x4_relaxed_madd(va1x0, vb0xCDEF, vacc1xCDEF);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2x0, vb0x0123, vacc2x0123);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2x0, vb0x4567, vacc2x4567);
        vacc2x89AB = wasm_f32x4_relaxed_madd(va2x0, vb0x89AB, vacc2x89AB);
        vacc2xCDEF = wasm_f32x4_relaxed_madd(va2x0, vb0xCDEF, vacc2xCDEF);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3x0, vb0x0123, vacc3x0123);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3x0, vb0x4567, vacc3x4567);
        vacc3x89AB = wasm_f32x4_relaxed_madd(va3x0, vb0x89AB, vacc3x89AB);
        vacc3xCDEF = wasm_f32x4_relaxed_madd(va3x0, vb0xCDEF, vacc3xCDEF);
        vacc0x0123 = wasm_f32x4_relaxed_madd(va0x1, vb1x0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0x1, vb1x4567, vacc0x4567);
        vacc0x89AB = wasm_f32x4_relaxed_madd(va0x1, vb1x89AB, vacc0x89AB);
        vacc0xCDEF = wasm_f32x4_relaxed_madd(va0x1, vb1xCDEF, vacc0xCDEF);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1x1, vb1x0123, vacc1x0123);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1x1, vb1x4567, vacc1x4567);
        vacc1x89AB = wasm_f32x4_relaxed_madd(va1x1, vb1x89AB, vacc1x89AB);
        vacc1xCDEF = wasm_f32x4_relaxed_madd(va1x1, vb1xCDEF, vacc1xCDEF);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2x1, vb1x0123, vacc2x0123);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2x1, vb1x4567, vacc2x4567);
        vacc2x89AB = wasm_f32x4_relaxed_madd(va2x1, vb1x89AB, vacc2x89AB);
        vacc2xCDEF = wasm_f32x4_relaxed_madd(va2x1, vb1xCDEF, vacc2xCDEF);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3x1, vb1x0123, vacc3x0123);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3x1, vb1x4567, vacc3x4567);
        vacc3x89AB = wasm_f32x4_relaxed_madd(va3x1, vb1x89AB, vacc3x89AB);
        vacc3xCDEF = wasm_f32x4_relaxed_madd(va3x1, vb1xCDEF, vacc3xCDEF);
        k -= 2 * sizeof(float);
      }

      while (k != 0) {
        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        const v128_t vb89AB = wasm_v128_load(w + 8);
        const v128_t vbCDEF = wasm_v128_load(w + 12);
        w += 16;

        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        vacc0x89AB = wasm_f32x4_relaxed_madd(va0, vb89AB, vacc0x89AB);
        vacc0xCDEF = wasm_f32x4_relaxed_madd(va0, vbCDEF, vacc0xCDEF);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
        vacc1x89AB = wasm_f32x4_relaxed_madd(va1, vb89AB, vacc1x89AB);
        vacc1xCDEF = wasm_f32x4_relaxed_madd(va1, vbCDEF, vacc1xCDEF);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
        vacc2x89AB = wasm_f32x4_relaxed_madd(va2, vb89AB, vacc2x89AB);
        vacc2xCDEF = wasm_f32x4_relaxed_madd(va2, vbCDEF, vacc2xCDEF);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
        vacc3x89AB = wasm_f32x4_relaxed_madd(va3, vb89AB, vacc3x89AB);
        vacc3xCDEF = wasm_f32x4_relaxed_madd(va3, vbCDEF, vacc3xCDEF);
        k -= sizeof(float);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_max(vmin, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_max(vmin, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_max(vmin, vacc3x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_max(vmin, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_max(vmin, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_max(vmin, vacc3x4567);
    vacc0x89AB = wasm_f32x4_relaxed_max(vmin, vacc0x89AB);
    vacc1x89AB = wasm_f32x4_relaxed_max(vmin, vacc1x89AB);
    vacc2x89AB = wasm_f32x4_relaxed_max(vmin, vacc2x89AB);
    vacc3x89AB = wasm_f32x4_relaxed_max(vmin, vacc3x89AB);
    vacc0xCDEF = wasm_f32x4_relaxed_max(vmin, vacc0xCDEF);
    vacc1xCDEF = wasm_f32x4_relaxed_max(vmin, vacc1xCDEF);
    vacc2xCDEF = wasm_f32x4_relaxed_max(vmin, vacc2xCDEF);
    vacc3xCDEF = wasm_f32x4_relaxed_max(vmin, vacc3xCDEF);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_min(vmax, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_min(vmax, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_min(vmax, vacc3x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_min(vmax, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_min(vmax, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_min(vmax, vacc3x4567);
    vacc0x89AB = wasm_f32x4_relaxed_min(vmax, vacc0x89AB);
    vacc1x89AB = wasm_f32x4_relaxed_min(vmax, vacc1x89AB);
    vacc2x89AB = wasm_f32x4_relaxed_min(vmax, vacc2x89AB);
    vacc3x89AB = wasm_f32x4_relaxed_min(vmax, vacc3x89AB);
    vacc0xCDEF = wasm_f32x4_relaxed_min(vmax, vacc0xCDEF);
    vacc1xCDEF = wasm_f32x4_relaxed_min(vmax, vacc1xCDEF);
    vacc2xCDEF = wasm_f32x4_relaxed_min(vmax, vacc2xCDEF);
    vacc3xCDEF = wasm_f32x4_relaxed_min(vmax, vacc3xCDEF);
    if XNN_LIKELY(nc >= 16) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      wasm_v128_store(c3 + 8, vacc3x89AB);
      wasm_v128_store(c3 + 12, vacc3xCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      wasm_v128_store(c2 + 8, vacc2x89AB);
      wasm_v128_store(c2 + 12, vacc2xCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      wasm_v128_store(c1 + 8, vacc1x89AB);
      wasm_v128_store(c1 + 12, vacc1xCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      wasm_v128_store(c0 + 8, vacc0x89AB);
      wasm_v128_store(c0 + 12, vacc0xCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c3 + 4, vacc3x4567);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c2 + 4, vacc2x4567);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c1 + 4, vacc1x4567);
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c0 + 4, vacc0x4567);

        vacc3x0123 = vacc3x89AB;
        vacc3x4567 = vacc3xCDEF;
        vacc2x0123 = vacc2x89AB;
        vacc2x4567 = vacc2xCDEF;
        vacc1x0123 = vacc1x89AB;
        vacc1x4567 = vacc1xCDEF;
        vacc0x0123 = vacc0x89AB;
        vacc0x4567 = vacc0xCDEF;

        c3 += 8;
        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc3x4567 = vacc3x89AB;
        vacc3x89AB = vacc3xCDEF;
        vacc2x0123 = vacc2x4567;
        vacc2x4567 = vacc2x89AB;
        vacc2x89AB = vacc2xCDEF;
        vacc1x0123 = vacc1x4567;
        vacc1x4567 = vacc1x89AB;
        vacc1x89AB = vacc1xCDEF;
        vacc0x0123 = vacc0x4567;
        vacc0x4567 = vacc0x89AB;
        vacc0x89AB = vacc0xCDEF;

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
