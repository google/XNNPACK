// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/spmm.h>


void xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_x4(
    size_t mc,
    size_t nc,
    const float*restrict input,
    const float*restrict weights,
    const int32_t*restrict widx_dmap,
    const uint32_t*restrict nidx_nnzmap,
    float*restrict output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.max);
  const v128_t vzero = wasm_f64x2_splat(0.0);
  size_t output_decrement = output_stride * nc - 8 * sizeof(float);
  while XNN_LIKELY(mc >= 8 * sizeof(float)) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      v128_t vacc0123x0 = wasm_v128_load32_splat(w);
      w += 1;
      v128_t vacc0123x1 = vzero;
      v128_t vacc0123x2 = vzero;
      v128_t vacc0123x3 = vzero;
      v128_t vacc4567x0 = vacc0123x0;
      v128_t vacc4567x1 = vzero;
      v128_t vacc4567x2 = vzero;
      v128_t vacc4567x3 = vzero;
      for (; nnz >= 4; nnz -= 4) {
        const intptr_t diff0 = dmap[0];
        const intptr_t diff1 = dmap[1];
        const intptr_t diff2 = dmap[2];
        const intptr_t diff3 = dmap[3];
        dmap += 4;
        const v128_t vi0123x0 = wasm_v128_load(input);
        const v128_t vi4567x0 = wasm_v128_load(input + 4);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff0);
        const v128_t vw0 = wasm_v128_load32_splat(w);
        w += 1;
        vacc0123x0 = wasm_f32x4_add(vacc0123x0, wasm_f32x4_mul(vi0123x0, vw0));
        vacc4567x0 = wasm_f32x4_add(vacc4567x0, wasm_f32x4_mul(vi4567x0, vw0));
        const v128_t vi0123x1 = wasm_v128_load(input);
        const v128_t vi4567x1 = wasm_v128_load(input + 4);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff1);
        const v128_t vw1 = wasm_v128_load32_splat(w);
        w += 1;
        vacc0123x1 = wasm_f32x4_add(vacc0123x1, wasm_f32x4_mul(vi0123x1, vw1));
        vacc4567x1 = wasm_f32x4_add(vacc4567x1, wasm_f32x4_mul(vi4567x1, vw1));
        const v128_t vi0123x2 = wasm_v128_load(input);
        const v128_t vi4567x2 = wasm_v128_load(input + 4);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff2);
        const v128_t vw2 = wasm_v128_load32_splat(w);
        w += 1;
        vacc0123x2 = wasm_f32x4_add(vacc0123x2, wasm_f32x4_mul(vi0123x2, vw2));
        vacc4567x2 = wasm_f32x4_add(vacc4567x2, wasm_f32x4_mul(vi4567x2, vw2));
        const v128_t vi0123x3 = wasm_v128_load(input);
        const v128_t vi4567x3 = wasm_v128_load(input + 4);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff3);
        const v128_t vw3 = wasm_v128_load32_splat(w);
        w += 1;
        vacc0123x3 = wasm_f32x4_add(vacc0123x3, wasm_f32x4_mul(vi0123x3, vw3));
        vacc4567x3 = wasm_f32x4_add(vacc4567x3, wasm_f32x4_mul(vi4567x3, vw3));
      }
      v128_t vacc0123 = vacc0123x0;
      v128_t vacc4567 = vacc4567x0;
      vacc0123 = wasm_f32x4_add(vacc0123, vacc0123x1);
      vacc4567 = wasm_f32x4_add(vacc4567, vacc4567x1);
      vacc0123 = wasm_f32x4_add(vacc0123, vacc0123x2);
      vacc4567 = wasm_f32x4_add(vacc4567, vacc4567x2);
      vacc0123 = wasm_f32x4_add(vacc0123, vacc0123x3);
      vacc4567 = wasm_f32x4_add(vacc4567, vacc4567x3);
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const v128_t vi0123 = wasm_v128_load(input);
          const v128_t vi4567 = wasm_v128_load(input + 4);
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const v128_t vw = wasm_v128_load32_splat(w); w += 1;
          vacc0123 = wasm_f32x4_add(vacc0123, wasm_f32x4_mul(vi0123, vw));
          vacc4567 = wasm_f32x4_add(vacc4567, wasm_f32x4_mul(vi4567, vw));
        } while (--nnz != 0);
      }
      v128_t vout0123 = wasm_v128_bitselect(vacc0123, vmax, wasm_f32x4_le(vacc0123, vmax));
      v128_t vout4567 = wasm_v128_bitselect(vacc4567, vmax, wasm_f32x4_le(vacc4567, vmax));
      vout0123 = wasm_v128_bitselect(vmin, vout0123, wasm_f32x4_lt(vout0123, vmin));
      vout4567 = wasm_v128_bitselect(vmin, vout4567, wasm_f32x4_lt(vout4567, vmin));
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
      const float*restrict w = weights;
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
            vacc0123 = wasm_f32x4_add(vacc0123, wasm_f32x4_mul(vi0123, vw));
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_v128_bitselect(vacc0123, vmax, wasm_f32x4_le(vacc0123, vmax));
        vout0123 = wasm_v128_bitselect(vmin, vout0123, wasm_f32x4_lt(vout0123, vmin));
        wasm_v128_store(output, vout0123);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float*restrict w = weights;
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
            vacc01 = wasm_f32x4_add(vacc01, wasm_f32x4_mul(vi01, vw));
          } while (--nnz != 0);
        }
        v128_t vout01 = wasm_v128_bitselect(vacc01, vmax, wasm_f32x4_le(vacc01, vmax));
        vout01 = wasm_v128_bitselect(vmin, vout01, wasm_f32x4_lt(vout01, vmin));
        *((double*) output) = wasm_f64x2_extract_lane(vout01, 0);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float*restrict w = weights;
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
            vacc0 = wasm_f32x4_add(vacc0, wasm_f32x4_mul(vi0, vw));
          } while (--nnz != 0);
        }
        v128_t vout0 = wasm_v128_bitselect(vacc0, vmax, wasm_f32x4_le(vacc0, vmax));
        vout0 = wasm_v128_bitselect(vmin, vout0, wasm_f32x4_lt(vout0, vmin));
        *output = wasm_f32x4_extract_lane(vout0, 0);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}
