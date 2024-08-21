// Auto-generated file. Do not edit!
//   Template: src/f32-vmulcaddc/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/math.h"
#include "xnnpack/vmulcaddc.h"


void xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const v128_t vscale0123 = wasm_v128_load(w);

      v128_t vacc0x0123 = wasm_v128_load(i0);
      i0 += 4;
      v128_t vacc1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vbias0123 = wasm_v128_load(w + 4);

      vacc0x0123 = wasm_f32x4_relaxed_madd(vscale0123, vacc0x0123, vbias0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(vscale0123, vacc1x0123, vbias0123);

      vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_max(vmin, vacc1x0123);

      vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_min(vmax, vacc1x0123);

      wasm_v128_store(o0, vacc0x0123);
      o0 += 4;
      wasm_v128_store(o1, vacc1x0123);
      o1 += 4;

      w += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const v128_t vscale = wasm_v128_load(w);

      v128_t vacc0 = wasm_v128_load(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      v128_t vacc1 = wasm_v128_load(i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      const v128_t vbias = wasm_v128_load(w + 4);

      vacc0 = wasm_f32x4_relaxed_madd(vscale, vacc0, vbias);
      vacc1 = wasm_f32x4_relaxed_madd(vscale, vacc1, vbias);

      vacc0 = wasm_f32x4_relaxed_max(vmin, vacc0);
      vacc1 = wasm_f32x4_relaxed_max(vmin, vacc1);

      vacc0 = wasm_f32x4_relaxed_min(vmax, vacc0);
      vacc1 = wasm_f32x4_relaxed_min(vmax, vacc1);

      if (c & (2 * sizeof(float))) {
        wasm_v128_store64_lane(o0, vacc0, 0);
        wasm_v128_store64_lane(o1, vacc1, 0);

        vacc0 = wasm_v64x2_shuffle(vacc0, vacc0, 1, 1);
        vacc1 = wasm_v64x2_shuffle(vacc1, vacc1, 1, 1);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(float))) {
        wasm_v128_store32_lane(o0, vacc0, 0);
        o0 += 1;
        wasm_v128_store32_lane(o1, vacc1, 0);
        o1 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}
