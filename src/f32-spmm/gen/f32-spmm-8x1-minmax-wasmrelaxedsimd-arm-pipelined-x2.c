// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/wasmsimd-pipelined.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/spmm.h"


void xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_pipelined_x2(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  size_t output_decrement = output_stride * nc - 8 * sizeof(float);
  while XNN_LIKELY(mc >= 8 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    v128_t vw = wasm_v128_load32_splat(w); w += 1;
    intptr_t diff = *dmap++;
    v128_t vi0123 = wasm_v128_load(input + 0);
    v128_t vi4567 = wasm_v128_load(input + 4);
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
       v128_t vacc0123 = vw;
       v128_t vacc4567 = vw;
      vw = wasm_v128_load32_splat(w); w += 1;

      for (; nnz >= 2; nnz -= 2) {
        vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
        vacc4567 = wasm_f32x4_relaxed_madd(vi4567, vw, vacc4567);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
        diff = *dmap++;
        vw = wasm_v128_load32_splat(w); w += 1;
        vi0123 = wasm_v128_load(input + 0);
        vi4567 = wasm_v128_load(input + 4);
        vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
        vacc4567 = wasm_f32x4_relaxed_madd(vi4567, vw, vacc4567);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
        diff = *dmap++;
        vw = wasm_v128_load32_splat(w); w += 1;
        vi0123 = wasm_v128_load(input + 0);
        vi4567 = wasm_v128_load(input + 4);
      }

      if XNN_LIKELY(nnz != 0) {
        do {
          vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
          vacc4567 = wasm_f32x4_relaxed_madd(vi4567, vw, vacc4567);
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);

          diff = *dmap++;
          vw = wasm_v128_load32_splat(w); w += 1;
          vi0123 = wasm_v128_load(input + 0);
          vi4567 = wasm_v128_load(input + 4);
        } while (--nnz != 0);
      }
      v128_t vout0123 = wasm_f32x4_min(vmax, vacc0123);
      v128_t vout4567 = wasm_f32x4_min(vmax, vacc4567);
      vout0123 = wasm_f32x4_max(vmin, vout0123);
      vout4567 = wasm_f32x4_max(vmin, vout4567);
      wasm_v128_store(output, vout0123);
      wasm_v128_store(output + 4, vout4567);
      output = (float*restrict) ((uintptr_t) output + output_stride);
    } while (--n != 0);
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 8;
    mc -= 8 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0123 = wasm_v128_load32_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0123 = wasm_v128_load(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_f32x4_min(vmax, vacc0123);
        vout0123 = wasm_f32x4_max(vmin, vout0123);
        wasm_v128_store(output, vout0123);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc01 = wasm_v128_load32_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi01 = wasm_v128_load64_splat(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc01 = wasm_f32x4_relaxed_madd(vi01, vw, vacc01);
          } while (--nnz != 0);
        }
        v128_t vout01 = wasm_f32x4_min(vmax, vacc01);
        vout01 = wasm_f32x4_max(vmin, vout01);
        wasm_v128_store64_lane(output, vout01, 0);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0 = wasm_v128_load32_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0 = wasm_v128_load32_splat(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc0 = wasm_f32x4_relaxed_madd(vi0, vw, vacc0);
          } while (--nnz != 0);
        }
        v128_t vout0 = wasm_f32x4_min(vmax, vacc0);
        vout0 = wasm_f32x4_max(vmin, vout0);
        wasm_v128_store32_lane(output, vout0, 0);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}
