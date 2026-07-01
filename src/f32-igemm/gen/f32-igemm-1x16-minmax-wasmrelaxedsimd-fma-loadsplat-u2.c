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


void xnn_f32_igemm_minmax_ukernel_1x16__wasmrelaxedsimd_fma_loadsplat_u2(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc0x89AB = wasm_v128_load(w + 8);
    v128_t vacc0xCDEF = wasm_v128_load(w + 12);
    w += 16;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

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

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0x0, vb0x0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0x0, vb0x4567, vacc0x4567);
        vacc0x89AB = wasm_f32x4_relaxed_madd(va0x0, vb0x89AB, vacc0x89AB);
        vacc0xCDEF = wasm_f32x4_relaxed_madd(va0x0, vb0xCDEF, vacc0xCDEF);
        vacc0x0123 = wasm_f32x4_relaxed_madd(va0x1, vb1x0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0x1, vb1x4567, vacc0x4567);
        vacc0x89AB = wasm_f32x4_relaxed_madd(va0x1, vb1x89AB, vacc0x89AB);
        vacc0xCDEF = wasm_f32x4_relaxed_madd(va0x1, vb1xCDEF, vacc0xCDEF);
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

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        vacc0x89AB = wasm_f32x4_relaxed_madd(va0, vb89AB, vacc0x89AB);
        vacc0xCDEF = wasm_f32x4_relaxed_madd(va0, vbCDEF, vacc0xCDEF);
        k -= sizeof(float);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);
    vacc0x89AB = wasm_f32x4_relaxed_max(vmin, vacc0x89AB);
    vacc0xCDEF = wasm_f32x4_relaxed_max(vmin, vacc0xCDEF);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);
    vacc0x89AB = wasm_f32x4_relaxed_min(vmax, vacc0x89AB);
    vacc0xCDEF = wasm_f32x4_relaxed_min(vmax, vacc0xCDEF);
    if XNN_LIKELY(nc >= 16) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      wasm_v128_store(c0 + 8, vacc0x89AB);
      wasm_v128_store(c0 + 12, vacc0xCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c0 + 4, vacc0x4567);

        vacc0x0123 = vacc0x89AB;
        vacc0x4567 = vacc0xCDEF;

        c0 += 8;
      }
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;
        vacc0x4567 = vacc0x89AB;
        vacc0x89AB = vacc0xCDEF;

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
