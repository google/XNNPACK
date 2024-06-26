// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/MRx2c4-wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/gemm.h"


void xnn_f32_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    v128_t vacc0x0c4 = wasm_v128_load32_zero(w);
    v128_t vacc0x1c4 = wasm_v128_load32_zero(w + 1);
    v128_t vacc1x0c4 = vacc0x0c4;
    v128_t vacc1x1c4 = vacc0x1c4;
    v128_t vacc2x0c4 = vacc0x0c4;
    v128_t vacc2x1c4 = vacc0x1c4;
    v128_t vacc3x0c4 = vacc0x0c4;
    v128_t vacc3x1c4 = vacc0x1c4;
    w += 2;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;

      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb1 = wasm_v128_load(w + 4);
      w += 8;

      vacc0x0c4 = wasm_f32x4_relaxed_madd(va0, vb0, vacc0x0c4);
      vacc0x1c4 = wasm_f32x4_relaxed_madd(va0, vb1, vacc0x1c4);
      vacc1x0c4 = wasm_f32x4_relaxed_madd(va1, vb0, vacc1x0c4);
      vacc1x1c4 = wasm_f32x4_relaxed_madd(va1, vb1, vacc1x1c4);
      vacc2x0c4 = wasm_f32x4_relaxed_madd(va2, vb0, vacc2x0c4);
      vacc2x1c4 = wasm_f32x4_relaxed_madd(va2, vb1, vacc2x1c4);
      vacc3x0c4 = wasm_f32x4_relaxed_madd(va3, vb0, vacc3x0c4);
      vacc3x1c4 = wasm_f32x4_relaxed_madd(va3, vb1, vacc3x1c4);
    }
    if XNN_UNLIKELY(k != 0) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 = (const float*) ((uintptr_t) a0 + k);
      const v128_t va1 = wasm_v128_load(a1);
      a1 = (const float*) ((uintptr_t) a1 + k);
      const v128_t va2 = wasm_v128_load(a2);
      a2 = (const float*) ((uintptr_t) a2 + k);
      const v128_t va3 = wasm_v128_load(a3);
      a3 = (const float*) ((uintptr_t) a3 + k);

      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb1 = wasm_v128_load(w + 4);
      w += 8;

      const v128_t vzero = wasm_f32x4_const_splat(0.0f);
      const v128_t vmask0 = wasm_f32x4_eq(vb0, vzero);
      const v128_t vmask1 = wasm_f32x4_eq(vb1, vzero);

      vacc0x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va0, vmask0), vb0, vacc0x0c4);
      vacc0x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va0, vmask1), vb1, vacc0x1c4);
      vacc1x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va1, vmask0), vb0, vacc1x0c4);
      vacc1x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va1, vmask1), vb1, vacc1x1c4);
      vacc2x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va2, vmask0), vb0, vacc2x0c4);
      vacc2x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va2, vmask1), vb1, vacc2x1c4);
      vacc3x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va3, vmask0), vb0, vacc3x0c4);
      vacc3x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va3, vmask1), vb1, vacc3x1c4);
    }

    const v128_t vacc0x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 2, 6, 3, 7));
    const v128_t vacc1x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 2, 6, 3, 7));
    const v128_t vacc2x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 2, 6, 3, 7));
    const v128_t vacc3x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 2, 6, 3, 7));

    v128_t vacc01x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 2, 3, 6, 7));
    v128_t vacc23x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 2, 3, 6, 7));


    if XNN_LIKELY(nc >= 2) {
      wasm_v128_store64_lane(c0, vacc01x01, 0);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a0 = (const float*) ((uintptr_t) a0 - kc);
      wasm_v128_store64_lane(c1, vacc01x01, 1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      wasm_v128_store64_lane(c2, vacc23x01, 0);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      wasm_v128_store64_lane(c3, vacc23x01, 1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 2;
    } else {
      assert(nc == 1);
      wasm_v128_store32_lane(c0, vacc01x01, 0);
      wasm_v128_store32_lane(c1, vacc01x01, 2);
      wasm_v128_store32_lane(c2, vacc23x01, 0);
      wasm_v128_store32_lane(c3, vacc23x01, 2);

      nc = 0;
    }
  } while (nc != 0);
}
