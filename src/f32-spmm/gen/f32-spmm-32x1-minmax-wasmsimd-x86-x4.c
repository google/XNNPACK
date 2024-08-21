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

#include "xnnpack/spmm.h"


void xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_x4(
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
  size_t output_decrement = output_stride * nc - 32 * sizeof(float);
  while XNN_LIKELY(mc >= 32 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      v128_t vacc0123x0 = wasm_v128_load32_splat(w);
      w += 1;
      v128_t vacc0123x1 = wasm_f32x4_const_splat(0.0f);
      v128_t vacc0123x2 = wasm_f32x4_const_splat(0.0f);
      v128_t vacc0123x3 = wasm_f32x4_const_splat(0.0f);
      v128_t vacc4567x0 = vacc0123x0;
      v128_t vacc4567x1 = wasm_f32x4_const_splat(0.0f);
      v128_t vacc4567x2 = wasm_f32x4_const_splat(0.0f);
      v128_t vacc4567x3 = wasm_f32x4_const_splat(0.0f);
      v128_t vacc89ABx0 = vacc0123x0;
      v128_t vacc89ABx1 = wasm_f32x4_const_splat(0.0f);
      v128_t vacc89ABx2 = wasm_f32x4_const_splat(0.0f);
      v128_t vacc89ABx3 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccCDEFx0 = vacc0123x0;
      v128_t vaccCDEFx1 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccCDEFx2 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccCDEFx3 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccGHIJx0 = vacc0123x0;
      v128_t vaccGHIJx1 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccGHIJx2 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccGHIJx3 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccKLMNx0 = vacc0123x0;
      v128_t vaccKLMNx1 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccKLMNx2 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccKLMNx3 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccOPQRx0 = vacc0123x0;
      v128_t vaccOPQRx1 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccOPQRx2 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccOPQRx3 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccSTUVx0 = vacc0123x0;
      v128_t vaccSTUVx1 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccSTUVx2 = wasm_f32x4_const_splat(0.0f);
      v128_t vaccSTUVx3 = wasm_f32x4_const_splat(0.0f);
      for (; nnz >= 4; nnz -= 4) {
        const intptr_t diff0 = dmap[0];
        const intptr_t diff1 = dmap[1];
        const intptr_t diff2 = dmap[2];
        const intptr_t diff3 = dmap[3];
        dmap += 4;
        const v128_t vi0123x0 = wasm_v128_load(input);
        const v128_t vi4567x0 = wasm_v128_load(input + 4);
        const v128_t vi89ABx0 = wasm_v128_load(input + 8);
        const v128_t viCDEFx0 = wasm_v128_load(input + 12);
        const v128_t viGHIJx0 = wasm_v128_load(input + 16);
        const v128_t viKLMNx0 = wasm_v128_load(input + 20);
        const v128_t viOPQRx0 = wasm_v128_load(input + 24);
        const v128_t viSTUVx0 = wasm_v128_load(input + 28);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff0);
        const v128_t vw0 = wasm_v128_load32_splat(w);
        w += 1;
        vacc0123x0 = wasm_f32x4_add(vacc0123x0, wasm_f32x4_mul(vi0123x0, vw0));
        vacc4567x0 = wasm_f32x4_add(vacc4567x0, wasm_f32x4_mul(vi4567x0, vw0));
        vacc89ABx0 = wasm_f32x4_add(vacc89ABx0, wasm_f32x4_mul(vi89ABx0, vw0));
        vaccCDEFx0 = wasm_f32x4_add(vaccCDEFx0, wasm_f32x4_mul(viCDEFx0, vw0));
        vaccGHIJx0 = wasm_f32x4_add(vaccGHIJx0, wasm_f32x4_mul(viGHIJx0, vw0));
        vaccKLMNx0 = wasm_f32x4_add(vaccKLMNx0, wasm_f32x4_mul(viKLMNx0, vw0));
        vaccOPQRx0 = wasm_f32x4_add(vaccOPQRx0, wasm_f32x4_mul(viOPQRx0, vw0));
        vaccSTUVx0 = wasm_f32x4_add(vaccSTUVx0, wasm_f32x4_mul(viSTUVx0, vw0));
        const v128_t vi0123x1 = wasm_v128_load(input);
        const v128_t vi4567x1 = wasm_v128_load(input + 4);
        const v128_t vi89ABx1 = wasm_v128_load(input + 8);
        const v128_t viCDEFx1 = wasm_v128_load(input + 12);
        const v128_t viGHIJx1 = wasm_v128_load(input + 16);
        const v128_t viKLMNx1 = wasm_v128_load(input + 20);
        const v128_t viOPQRx1 = wasm_v128_load(input + 24);
        const v128_t viSTUVx1 = wasm_v128_load(input + 28);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff1);
        const v128_t vw1 = wasm_v128_load32_splat(w);
        w += 1;
        vacc0123x1 = wasm_f32x4_add(vacc0123x1, wasm_f32x4_mul(vi0123x1, vw1));
        vacc4567x1 = wasm_f32x4_add(vacc4567x1, wasm_f32x4_mul(vi4567x1, vw1));
        vacc89ABx1 = wasm_f32x4_add(vacc89ABx1, wasm_f32x4_mul(vi89ABx1, vw1));
        vaccCDEFx1 = wasm_f32x4_add(vaccCDEFx1, wasm_f32x4_mul(viCDEFx1, vw1));
        vaccGHIJx1 = wasm_f32x4_add(vaccGHIJx1, wasm_f32x4_mul(viGHIJx1, vw1));
        vaccKLMNx1 = wasm_f32x4_add(vaccKLMNx1, wasm_f32x4_mul(viKLMNx1, vw1));
        vaccOPQRx1 = wasm_f32x4_add(vaccOPQRx1, wasm_f32x4_mul(viOPQRx1, vw1));
        vaccSTUVx1 = wasm_f32x4_add(vaccSTUVx1, wasm_f32x4_mul(viSTUVx1, vw1));
        const v128_t vi0123x2 = wasm_v128_load(input);
        const v128_t vi4567x2 = wasm_v128_load(input + 4);
        const v128_t vi89ABx2 = wasm_v128_load(input + 8);
        const v128_t viCDEFx2 = wasm_v128_load(input + 12);
        const v128_t viGHIJx2 = wasm_v128_load(input + 16);
        const v128_t viKLMNx2 = wasm_v128_load(input + 20);
        const v128_t viOPQRx2 = wasm_v128_load(input + 24);
        const v128_t viSTUVx2 = wasm_v128_load(input + 28);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff2);
        const v128_t vw2 = wasm_v128_load32_splat(w);
        w += 1;
        vacc0123x2 = wasm_f32x4_add(vacc0123x2, wasm_f32x4_mul(vi0123x2, vw2));
        vacc4567x2 = wasm_f32x4_add(vacc4567x2, wasm_f32x4_mul(vi4567x2, vw2));
        vacc89ABx2 = wasm_f32x4_add(vacc89ABx2, wasm_f32x4_mul(vi89ABx2, vw2));
        vaccCDEFx2 = wasm_f32x4_add(vaccCDEFx2, wasm_f32x4_mul(viCDEFx2, vw2));
        vaccGHIJx2 = wasm_f32x4_add(vaccGHIJx2, wasm_f32x4_mul(viGHIJx2, vw2));
        vaccKLMNx2 = wasm_f32x4_add(vaccKLMNx2, wasm_f32x4_mul(viKLMNx2, vw2));
        vaccOPQRx2 = wasm_f32x4_add(vaccOPQRx2, wasm_f32x4_mul(viOPQRx2, vw2));
        vaccSTUVx2 = wasm_f32x4_add(vaccSTUVx2, wasm_f32x4_mul(viSTUVx2, vw2));
        const v128_t vi0123x3 = wasm_v128_load(input);
        const v128_t vi4567x3 = wasm_v128_load(input + 4);
        const v128_t vi89ABx3 = wasm_v128_load(input + 8);
        const v128_t viCDEFx3 = wasm_v128_load(input + 12);
        const v128_t viGHIJx3 = wasm_v128_load(input + 16);
        const v128_t viKLMNx3 = wasm_v128_load(input + 20);
        const v128_t viOPQRx3 = wasm_v128_load(input + 24);
        const v128_t viSTUVx3 = wasm_v128_load(input + 28);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff3);
        const v128_t vw3 = wasm_v128_load32_splat(w);
        w += 1;
        vacc0123x3 = wasm_f32x4_add(vacc0123x3, wasm_f32x4_mul(vi0123x3, vw3));
        vacc4567x3 = wasm_f32x4_add(vacc4567x3, wasm_f32x4_mul(vi4567x3, vw3));
        vacc89ABx3 = wasm_f32x4_add(vacc89ABx3, wasm_f32x4_mul(vi89ABx3, vw3));
        vaccCDEFx3 = wasm_f32x4_add(vaccCDEFx3, wasm_f32x4_mul(viCDEFx3, vw3));
        vaccGHIJx3 = wasm_f32x4_add(vaccGHIJx3, wasm_f32x4_mul(viGHIJx3, vw3));
        vaccKLMNx3 = wasm_f32x4_add(vaccKLMNx3, wasm_f32x4_mul(viKLMNx3, vw3));
        vaccOPQRx3 = wasm_f32x4_add(vaccOPQRx3, wasm_f32x4_mul(viOPQRx3, vw3));
        vaccSTUVx3 = wasm_f32x4_add(vaccSTUVx3, wasm_f32x4_mul(viSTUVx3, vw3));
      }
      v128_t vacc0123 = vacc0123x0;
      v128_t vacc4567 = vacc4567x0;
      v128_t vacc89AB = vacc89ABx0;
      v128_t vaccCDEF = vaccCDEFx0;
      v128_t vaccGHIJ = vaccGHIJx0;
      v128_t vaccKLMN = vaccKLMNx0;
      v128_t vaccOPQR = vaccOPQRx0;
      v128_t vaccSTUV = vaccSTUVx0;
      vacc0123 = wasm_f32x4_add(vacc0123, vacc0123x1);
      vacc4567 = wasm_f32x4_add(vacc4567, vacc4567x1);
      vacc89AB = wasm_f32x4_add(vacc89AB, vacc89ABx1);
      vaccCDEF = wasm_f32x4_add(vaccCDEF, vaccCDEFx1);
      vaccGHIJ = wasm_f32x4_add(vaccGHIJ, vaccGHIJx1);
      vaccKLMN = wasm_f32x4_add(vaccKLMN, vaccKLMNx1);
      vaccOPQR = wasm_f32x4_add(vaccOPQR, vaccOPQRx1);
      vaccSTUV = wasm_f32x4_add(vaccSTUV, vaccSTUVx1);
      vacc0123 = wasm_f32x4_add(vacc0123, vacc0123x2);
      vacc4567 = wasm_f32x4_add(vacc4567, vacc4567x2);
      vacc89AB = wasm_f32x4_add(vacc89AB, vacc89ABx2);
      vaccCDEF = wasm_f32x4_add(vaccCDEF, vaccCDEFx2);
      vaccGHIJ = wasm_f32x4_add(vaccGHIJ, vaccGHIJx2);
      vaccKLMN = wasm_f32x4_add(vaccKLMN, vaccKLMNx2);
      vaccOPQR = wasm_f32x4_add(vaccOPQR, vaccOPQRx2);
      vaccSTUV = wasm_f32x4_add(vaccSTUV, vaccSTUVx2);
      vacc0123 = wasm_f32x4_add(vacc0123, vacc0123x3);
      vacc4567 = wasm_f32x4_add(vacc4567, vacc4567x3);
      vacc89AB = wasm_f32x4_add(vacc89AB, vacc89ABx3);
      vaccCDEF = wasm_f32x4_add(vaccCDEF, vaccCDEFx3);
      vaccGHIJ = wasm_f32x4_add(vaccGHIJ, vaccGHIJx3);
      vaccKLMN = wasm_f32x4_add(vaccKLMN, vaccKLMNx3);
      vaccOPQR = wasm_f32x4_add(vaccOPQR, vaccOPQRx3);
      vaccSTUV = wasm_f32x4_add(vaccSTUV, vaccSTUVx3);
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const v128_t vi0123 = wasm_v128_load(input);
          const v128_t vi4567 = wasm_v128_load(input + 4);
          const v128_t vi89AB = wasm_v128_load(input + 8);
          const v128_t viCDEF = wasm_v128_load(input + 12);
          const v128_t viGHIJ = wasm_v128_load(input + 16);
          const v128_t viKLMN = wasm_v128_load(input + 20);
          const v128_t viOPQR = wasm_v128_load(input + 24);
          const v128_t viSTUV = wasm_v128_load(input + 28);
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const v128_t vw = wasm_v128_load32_splat(w); w += 1;
          vacc0123 = wasm_f32x4_add(vacc0123, wasm_f32x4_mul(vi0123, vw));
          vacc4567 = wasm_f32x4_add(vacc4567, wasm_f32x4_mul(vi4567, vw));
          vacc89AB = wasm_f32x4_add(vacc89AB, wasm_f32x4_mul(vi89AB, vw));
          vaccCDEF = wasm_f32x4_add(vaccCDEF, wasm_f32x4_mul(viCDEF, vw));
          vaccGHIJ = wasm_f32x4_add(vaccGHIJ, wasm_f32x4_mul(viGHIJ, vw));
          vaccKLMN = wasm_f32x4_add(vaccKLMN, wasm_f32x4_mul(viKLMN, vw));
          vaccOPQR = wasm_f32x4_add(vaccOPQR, wasm_f32x4_mul(viOPQR, vw));
          vaccSTUV = wasm_f32x4_add(vaccSTUV, wasm_f32x4_mul(viSTUV, vw));
        } while (--nnz != 0);
      }
      v128_t vout0123 = wasm_f32x4_pmin(vmax, vacc0123);
      v128_t vout4567 = wasm_f32x4_pmin(vmax, vacc4567);
      v128_t vout89AB = wasm_f32x4_pmin(vmax, vacc89AB);
      v128_t voutCDEF = wasm_f32x4_pmin(vmax, vaccCDEF);
      v128_t voutGHIJ = wasm_f32x4_pmin(vmax, vaccGHIJ);
      v128_t voutKLMN = wasm_f32x4_pmin(vmax, vaccKLMN);
      v128_t voutOPQR = wasm_f32x4_pmin(vmax, vaccOPQR);
      v128_t voutSTUV = wasm_f32x4_pmin(vmax, vaccSTUV);
      vout0123 = wasm_f32x4_pmax(vmin, vout0123);
      vout4567 = wasm_f32x4_pmax(vmin, vout4567);
      vout89AB = wasm_f32x4_pmax(vmin, vout89AB);
      voutCDEF = wasm_f32x4_pmax(vmin, voutCDEF);
      voutGHIJ = wasm_f32x4_pmax(vmin, voutGHIJ);
      voutKLMN = wasm_f32x4_pmax(vmin, voutKLMN);
      voutOPQR = wasm_f32x4_pmax(vmin, voutOPQR);
      voutSTUV = wasm_f32x4_pmax(vmin, voutSTUV);
      wasm_v128_store(output, vout0123);
      wasm_v128_store(output + 4, vout4567);
      wasm_v128_store(output + 8, vout89AB);
      wasm_v128_store(output + 12, voutCDEF);
      wasm_v128_store(output + 16, voutGHIJ);
      wasm_v128_store(output + 20, voutKLMN);
      wasm_v128_store(output + 24, voutOPQR);
      wasm_v128_store(output + 28, voutSTUV);
      output = (float*restrict) ((uintptr_t) output + output_stride);
    } while (--n != 0);
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 32;
    mc -= 32 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0123 = wasm_v128_load32_splat(w); w += 1;
        v128_t vacc4567 = vacc0123;
        v128_t vacc89AB = vacc0123;
        v128_t vaccCDEF = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0123 = wasm_v128_load(input);
            const v128_t vi4567 = wasm_v128_load(input + 4);
            const v128_t vi89AB = wasm_v128_load(input + 8);
            const v128_t viCDEF = wasm_v128_load(input + 12);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc0123 = wasm_f32x4_add(vacc0123, wasm_f32x4_mul(vi0123, vw));
            vacc4567 = wasm_f32x4_add(vacc4567, wasm_f32x4_mul(vi4567, vw));
            vacc89AB = wasm_f32x4_add(vacc89AB, wasm_f32x4_mul(vi89AB, vw));
            vaccCDEF = wasm_f32x4_add(vaccCDEF, wasm_f32x4_mul(viCDEF, vw));
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_f32x4_pmin(vmax, vacc0123);
        v128_t vout4567 = wasm_f32x4_pmin(vmax, vacc4567);
        v128_t vout89AB = wasm_f32x4_pmin(vmax, vacc89AB);
        v128_t voutCDEF = wasm_f32x4_pmin(vmax, vaccCDEF);
        vout0123 = wasm_f32x4_pmax(vmin, vout0123);
        vout4567 = wasm_f32x4_pmax(vmin, vout4567);
        vout89AB = wasm_f32x4_pmax(vmin, vout89AB);
        voutCDEF = wasm_f32x4_pmax(vmin, voutCDEF);
        wasm_v128_store(output, vout0123);

        wasm_v128_store(output + 4, vout4567);
        wasm_v128_store(output + 8, vout89AB);
        wasm_v128_store(output + 12, voutCDEF);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0123 = wasm_v128_load32_splat(w); w += 1;
        v128_t vacc4567 = vacc0123;
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
        v128_t vout0123 = wasm_f32x4_pmin(vmax, vacc0123);
        v128_t vout4567 = wasm_f32x4_pmin(vmax, vacc4567);
        vout0123 = wasm_f32x4_pmax(vmin, vout0123);
        vout4567 = wasm_f32x4_pmax(vmin, vout4567);
        wasm_v128_store(output, vout0123);

        wasm_v128_store(output + 4, vout4567);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 8;
    }
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
            vacc0123 = wasm_f32x4_add(vacc0123, wasm_f32x4_mul(vi0123, vw));
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_f32x4_pmin(vmax, vacc0123);
        vout0123 = wasm_f32x4_pmax(vmin, vout0123);
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
            vacc01 = wasm_f32x4_add(vacc01, wasm_f32x4_mul(vi01, vw));
          } while (--nnz != 0);
        }
        v128_t vout01 = wasm_f32x4_pmin(vmax, vacc01);
        vout01 = wasm_f32x4_pmax(vmin, vout01);
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
            vacc0 = wasm_f32x4_add(vacc0, wasm_f32x4_mul(vi0, vw));
          } while (--nnz != 0);
        }
        v128_t vout0 = wasm_f32x4_pmin(vmax, vacc0);
        vout0 = wasm_f32x4_pmax(vmin, vout0);
        wasm_v128_store32_lane(output, vout0, 0);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}
