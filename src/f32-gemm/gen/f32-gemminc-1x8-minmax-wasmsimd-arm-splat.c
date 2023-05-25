// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/wasmsimd-splat.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/gemm.h>


void xnn_f32_gemminc_minmax_ukernel_1x8__wasmsimd_arm_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const float* restrict acc,
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
  assert(acc != NULL);

  const float* a0 = a;
  float* c0 = c;

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  do {
    v128_t vacc0x0123 = wasm_v128_load(acc + 0);
    v128_t vacc0x4567 = wasm_v128_load(acc + 4);
    acc += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = wasm_f32x4_add(wasm_f32x4_mul(va0c0, vb0123c0), vacc0x0123);
      vacc0x4567 = wasm_f32x4_add(wasm_f32x4_mul(va0c0, vb4567c0), vacc0x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = wasm_f32x4_add(wasm_f32x4_mul(va0c1, vb0123c1), vacc0x0123);
      vacc0x4567 = wasm_f32x4_add(wasm_f32x4_mul(va0c1, vb4567c1), vacc0x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = wasm_f32x4_add(wasm_f32x4_mul(va0c2, vb0123c2), vacc0x0123);
      vacc0x4567 = wasm_f32x4_add(wasm_f32x4_mul(va0c2, vb4567c2), vacc0x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = wasm_f32x4_add(wasm_f32x4_mul(va0c3, vb0123c3), vacc0x0123);
      vacc0x4567 = wasm_f32x4_add(wasm_f32x4_mul(va0c3, vb4567c3), vacc0x4567);
      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = wasm_f32x4_add(wasm_f32x4_mul(va0, vb0123), vacc0x0123);
        vacc0x4567 = wasm_f32x4_add(wasm_f32x4_mul(va0, vb4567), vacc0x4567);

        k -= sizeof(float);
      } while (k != 0);
    }

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
