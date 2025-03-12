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

void xnn_f16_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_float16* restrict a,
    size_t a_stride,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
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


  do {
    v128_t vacc0x0 = wasm_v128_load(w);
    w = (const xnn_float16*) w + 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_i16x8_splat(*a0);
      a0 += 1;

      const v128_t vb0 = wasm_v128_load(w);
      w = (const xnn_float16*) w + 8;

      vacc0x0 = wasm_f16x8_relaxed_madd(va0, vb0, vacc0x0);

      k -= sizeof(uint16_t);
    } while (k != 0);

    vacc0x0 = wasm_f16x8_pmax(vacc0x0, vmin);

    vacc0x0 = wasm_f16x8_pmin(vacc0x0, vmax);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      v128_t vh0x0 = vacc0x0;
      if (nc & 4) {
        wasm_v128_store64_lane(c0, vh0x0, 0);

        vh0x0 = wasm_i64x2_shuffle(vh0x0, vh0x0, 1, 1);

        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store32_lane(c0, vh0x0, 0);

        wasm_i32x4_shuffle(vh0x0, vh0x0, 1, 2, 3, 1);

        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store16_lane(c0, vh0x0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
