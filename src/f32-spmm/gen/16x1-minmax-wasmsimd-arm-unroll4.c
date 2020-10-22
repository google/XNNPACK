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


void xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_unroll4(
    uint32_t batch_size,
    uint32_t output_channels,
    const float*restrict input,
    const float*restrict weights,
    const int32_t*restrict widx_dmap,
    const uint32_t*restrict nidx_nnzmap,
    float*restrict output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch_size != 0);

  const v128_t vmin = wasm_v32x4_load_splat(&params->scalar.min);
  const v128_t vmax = wasm_v32x4_load_splat(&params->scalar.max);
  const v128_t vzero = wasm_f32x4_splat(0.0f);
  size_t n = batch_size;
  while XNN_LIKELY(n >= 16) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t c = output_channels;
    do {
      uint32_t nnz = *nnzmap++;
      v128_t vacc0123x0 = wasm_v32x4_load_splat(w);
      w += 1;
      v128_t vacc0123x1 = vzero;
      v128_t vacc0123x2 = vzero;
      v128_t vacc0123x3 = vzero;
      v128_t vacc4567x0 = vacc0123x0;
      v128_t vacc4567x1 = vzero;
      v128_t vacc4567x2 = vzero;
      v128_t vacc4567x3 = vzero;
      v128_t vacc89ABx0 = vacc0123x0;
      v128_t vacc89ABx1 = vzero;
      v128_t vacc89ABx2 = vzero;
      v128_t vacc89ABx3 = vzero;
      v128_t vaccCDEFx0 = vacc0123x0;
      v128_t vaccCDEFx1 = vzero;
      v128_t vaccCDEFx2 = vzero;
      v128_t vaccCDEFx3 = vzero;
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
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff0);
        const v128_t vw0 = wasm_v32x4_load_splat(w);
        w += 1;
        vacc0123x0 = wasm_f32x4_add(vacc0123x0, wasm_f32x4_mul(vi0123x0, vw0));
        vacc4567x0 = wasm_f32x4_add(vacc4567x0, wasm_f32x4_mul(vi4567x0, vw0));
        vacc89ABx0 = wasm_f32x4_add(vacc89ABx0, wasm_f32x4_mul(vi89ABx0, vw0));
        vaccCDEFx0 = wasm_f32x4_add(vaccCDEFx0, wasm_f32x4_mul(viCDEFx0, vw0));
        const v128_t vi0123x1 = wasm_v128_load(input);
        const v128_t vi4567x1 = wasm_v128_load(input + 4);
        const v128_t vi89ABx1 = wasm_v128_load(input + 8);
        const v128_t viCDEFx1 = wasm_v128_load(input + 12);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff1);
        const v128_t vw1 = wasm_v32x4_load_splat(w);
        w += 1;
        vacc0123x1 = wasm_f32x4_add(vacc0123x1, wasm_f32x4_mul(vi0123x1, vw1));
        vacc4567x1 = wasm_f32x4_add(vacc4567x1, wasm_f32x4_mul(vi4567x1, vw1));
        vacc89ABx1 = wasm_f32x4_add(vacc89ABx1, wasm_f32x4_mul(vi89ABx1, vw1));
        vaccCDEFx1 = wasm_f32x4_add(vaccCDEFx1, wasm_f32x4_mul(viCDEFx1, vw1));
        const v128_t vi0123x2 = wasm_v128_load(input);
        const v128_t vi4567x2 = wasm_v128_load(input + 4);
        const v128_t vi89ABx2 = wasm_v128_load(input + 8);
        const v128_t viCDEFx2 = wasm_v128_load(input + 12);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff2);
        const v128_t vw2 = wasm_v32x4_load_splat(w);
        w += 1;
        vacc0123x2 = wasm_f32x4_add(vacc0123x2, wasm_f32x4_mul(vi0123x2, vw2));
        vacc4567x2 = wasm_f32x4_add(vacc4567x2, wasm_f32x4_mul(vi4567x2, vw2));
        vacc89ABx2 = wasm_f32x4_add(vacc89ABx2, wasm_f32x4_mul(vi89ABx2, vw2));
        vaccCDEFx2 = wasm_f32x4_add(vaccCDEFx2, wasm_f32x4_mul(viCDEFx2, vw2));
        const v128_t vi0123x3 = wasm_v128_load(input);
        const v128_t vi4567x3 = wasm_v128_load(input + 4);
        const v128_t vi89ABx3 = wasm_v128_load(input + 8);
        const v128_t viCDEFx3 = wasm_v128_load(input + 12);
        input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff3);
        const v128_t vw3 = wasm_v32x4_load_splat(w);
        w += 1;
        vacc0123x3 = wasm_f32x4_add(vacc0123x3, wasm_f32x4_mul(vi0123x3, vw3));
        vacc4567x3 = wasm_f32x4_add(vacc4567x3, wasm_f32x4_mul(vi4567x3, vw3));
        vacc89ABx3 = wasm_f32x4_add(vacc89ABx3, wasm_f32x4_mul(vi89ABx3, vw3));
        vaccCDEFx3 = wasm_f32x4_add(vaccCDEFx3, wasm_f32x4_mul(viCDEFx3, vw3));
      }
      v128_t vacc0123 = vacc0123x0;
      v128_t vacc4567 = vacc4567x0;
      v128_t vacc89AB = vacc89ABx0;
      v128_t vaccCDEF = vaccCDEFx0;
      vacc0123 = wasm_f32x4_add(vacc0123, vacc0123x1);
      vacc4567 = wasm_f32x4_add(vacc4567, vacc4567x1);
      vacc89AB = wasm_f32x4_add(vacc89AB, vacc89ABx1);
      vaccCDEF = wasm_f32x4_add(vaccCDEF, vaccCDEFx1);
      vacc0123 = wasm_f32x4_add(vacc0123, vacc0123x2);
      vacc4567 = wasm_f32x4_add(vacc4567, vacc4567x2);
      vacc89AB = wasm_f32x4_add(vacc89AB, vacc89ABx2);
      vaccCDEF = wasm_f32x4_add(vaccCDEF, vaccCDEFx2);
      vacc0123 = wasm_f32x4_add(vacc0123, vacc0123x3);
      vacc4567 = wasm_f32x4_add(vacc4567, vacc4567x3);
      vacc89AB = wasm_f32x4_add(vacc89AB, vacc89ABx3);
      vaccCDEF = wasm_f32x4_add(vaccCDEF, vaccCDEFx3);
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const v128_t vi0123 = wasm_v128_load(input);
          const v128_t vi4567 = wasm_v128_load(input + 4);
          const v128_t vi89AB = wasm_v128_load(input + 8);
          const v128_t viCDEF = wasm_v128_load(input + 12);
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const v128_t vw = wasm_v32x4_load_splat(w); w += 1;
          vacc0123 = wasm_f32x4_add(vacc0123, wasm_f32x4_mul(vi0123, vw));
          vacc4567 = wasm_f32x4_add(vacc4567, wasm_f32x4_mul(vi4567, vw));
          vacc89AB = wasm_f32x4_add(vacc89AB, wasm_f32x4_mul(vi89AB, vw));
          vaccCDEF = wasm_f32x4_add(vaccCDEF, wasm_f32x4_mul(viCDEF, vw));
        } while (--nnz != 0);
      }
      v128_t vout0123 = wasm_f32x4_min(vacc0123, vmax);
      v128_t vout4567 = wasm_f32x4_min(vacc4567, vmax);
      v128_t vout89AB = wasm_f32x4_min(vacc89AB, vmax);
      v128_t voutCDEF = wasm_f32x4_min(vaccCDEF, vmax);
      vout0123 = wasm_f32x4_max(vout0123, vmin);
      vout4567 = wasm_f32x4_max(vout4567, vmin);
      vout89AB = wasm_f32x4_max(vout89AB, vmin);
      voutCDEF = wasm_f32x4_max(voutCDEF, vmin);
      wasm_v128_store(output, vout0123);
      wasm_v128_store(output + 4, vout4567);
      wasm_v128_store(output + 8, vout89AB);
      wasm_v128_store(output + 12, voutCDEF);
      output += 1 * batch_size;
    } while (--c != 0);
    output -= batch_size * output_channels;
    output += 16;
    input += 16;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0123 = wasm_v32x4_load_splat(w); w += 1;
        v128_t vacc4567 = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0123 = wasm_v128_load(input);
            const v128_t vi4567 = wasm_v128_load(input + 4);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v32x4_load_splat(w); w += 1;
            vacc0123 = wasm_f32x4_add(vacc0123, wasm_f32x4_mul(vi0123, vw));
            vacc4567 = wasm_f32x4_add(vacc4567, wasm_f32x4_mul(vi4567, vw));
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_f32x4_min(vacc0123, vmax);
        v128_t vout4567 = wasm_f32x4_min(vacc4567, vmax);
        vout0123 = wasm_f32x4_max(vout0123, vmin);
        vout4567 = wasm_f32x4_max(vout4567, vmin);
        wasm_v128_store(output, vout0123);

        wasm_v128_store(output + 4, vout4567);
        output += 1 * batch_size;
      } while (--c != 0);
      output -= batch_size * output_channels;
      output += 8;
      input += 8;
    }
    if (n & 4) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0123 = wasm_v32x4_load_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0123 = wasm_v128_load(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v32x4_load_splat(w); w += 1;
            vacc0123 = wasm_f32x4_add(vacc0123, wasm_f32x4_mul(vi0123, vw));
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_f32x4_min(vacc0123, vmax);
        vout0123 = wasm_f32x4_max(vout0123, vmin);
        wasm_v128_store(output, vout0123);

        output += 1 * batch_size;
      } while (--c != 0);
      output -= batch_size * output_channels;
      output += 4;
      input += 4;
    }
    if (n & 2) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc01 = wasm_v32x4_load_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi01 = wasm_v64x2_load_splat(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v32x4_load_splat(w); w += 1;
            vacc01 = wasm_f32x4_add(vacc01, wasm_f32x4_mul(vi01, vw));
          } while (--nnz != 0);
        }
        v128_t vout01 = wasm_f32x4_min(vacc01, vmax);
        vout01 = wasm_f32x4_max(vout01, vmin);
        *((double*) output) = wasm_f64x2_extract_lane(vout01, 0);

        output += 1 * batch_size;
      } while (--c != 0);
      output -= batch_size * output_channels;
      output += 2;
      input += 2;
    }
    if (n & 1) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0 = wasm_v32x4_load_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0 = wasm_v32x4_load_splat(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v32x4_load_splat(w); w += 1;
            vacc0 = wasm_f32x4_add(vacc0, wasm_f32x4_mul(vi0, vw));
          } while (--nnz != 0);
        }
        v128_t vout0 = wasm_f32x4_min(vacc0, vmax);
        vout0 = wasm_f32x4_max(vout0, vmin);
        *output = wasm_f32x4_extract_lane(vout0, 0);

        output += 1 * batch_size;
      } while (--c != 0);
      output -= batch_size * output_channels;
      output += 1;
      input += 1;
    }
  }
}
