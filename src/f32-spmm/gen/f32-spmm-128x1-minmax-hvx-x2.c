// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <hvx_hexagon_protos.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "xnnpack/simd/f32-hvx.h"

#include "xnnpack/prefetch.h"
#include "xnnpack/spmm.h"

void xnn_f32_spmm_minmax_ukernel_128x1__hvx_x2(
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

  const HVX_Vector vmin = xnn_set1_f32(params->scalar.min);
  const HVX_Vector vmax = xnn_set1_f32(params->scalar.max);

  size_t output_decrement = output_stride * nc - 128 * sizeof(float);
  while XNN_LIKELY(mc >= 128 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      HVX_Vector vacc0x0 = xnn_set1_f32(*w); w += 1;
      HVX_Vector vacc0x1 = xnn_zero_f32();
      HVX_Vector vacc1x0 = vacc0x0;
      HVX_Vector vacc1x1 = xnn_zero_f32();
      HVX_Vector vacc2x0 = vacc0x0;
      HVX_Vector vacc2x1 = xnn_zero_f32();
      HVX_Vector vacc3x0 = vacc0x0;
      HVX_Vector vacc3x1 = xnn_zero_f32();
      for (; nnz >= 2; nnz -= 2) {
        const intptr_t diff0 = dmap[0];
        const intptr_t diff1 = dmap[1];
        dmap += 2;
        const HVX_Vector vi0x0 = xnn_loadu_f32(input);
        const HVX_Vector vi1x0 = xnn_loadu_f32(input + 32);
        const HVX_Vector vi2x0 = xnn_loadu_f32(input + 64);
        const HVX_Vector vi3x0 = xnn_loadu_f32(input + 96);
        input = (const float*) ((uintptr_t) input + (uintptr_t) diff0);

        const HVX_Vector vw0 = xnn_set1_f32(*w); w += 1;

        vacc0x0 = xnn_fmadd_f32(vi0x0, vw0, vacc0x0);
        vacc1x0 = xnn_fmadd_f32(vi1x0, vw0, vacc1x0);
        vacc2x0 = xnn_fmadd_f32(vi2x0, vw0, vacc2x0);
        vacc3x0 = xnn_fmadd_f32(vi3x0, vw0, vacc3x0);
        const HVX_Vector vi0x1 = xnn_loadu_f32(input);
        const HVX_Vector vi1x1 = xnn_loadu_f32(input + 32);
        const HVX_Vector vi2x1 = xnn_loadu_f32(input + 64);
        const HVX_Vector vi3x1 = xnn_loadu_f32(input + 96);
        input = (const float*) ((uintptr_t) input + (uintptr_t) diff1);

        const HVX_Vector vw1 = xnn_set1_f32(*w); w += 1;

        vacc0x1 = xnn_fmadd_f32(vi0x1, vw1, vacc0x1);
        vacc1x1 = xnn_fmadd_f32(vi1x1, vw1, vacc1x1);
        vacc2x1 = xnn_fmadd_f32(vi2x1, vw1, vacc2x1);
        vacc3x1 = xnn_fmadd_f32(vi3x1, vw1, vacc3x1);
      }
      HVX_Vector vacc0 = vacc0x0;
      HVX_Vector vacc1 = vacc1x0;
      HVX_Vector vacc2 = vacc2x0;
      HVX_Vector vacc3 = vacc3x0;
      vacc0 = xnn_add_f32(vacc0, vacc0x1);
      vacc1 = xnn_add_f32(vacc1, vacc1x1);
      vacc2 = xnn_add_f32(vacc2, vacc2x1);
      vacc3 = xnn_add_f32(vacc3, vacc3x1);
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const HVX_Vector vi0 = xnn_loadu_f32(input);
          const HVX_Vector vi1 = xnn_loadu_f32(input + 32);
          const HVX_Vector vi2 = xnn_loadu_f32(input + 64);
          const HVX_Vector vi3 = xnn_loadu_f32(input + 96);
          input = (const float*) ((uintptr_t) input + (uintptr_t) diff);

          const HVX_Vector vw = xnn_set1_f32(*w); w += 1;

          vacc0 = xnn_fmadd_f32(vi0, vw, vacc0);
          vacc1 = xnn_fmadd_f32(vi1, vw, vacc1);
          vacc2 = xnn_fmadd_f32(vi2, vw, vacc2);
          vacc3 = xnn_fmadd_f32(vi3, vw, vacc3);
        } while (--nnz != 0);
      }
      HVX_Vector vout0 = xnn_min_f32(vacc0, vmax);
      HVX_Vector vout1 = xnn_min_f32(vacc1, vmax);
      HVX_Vector vout2 = xnn_min_f32(vacc2, vmax);
      HVX_Vector vout3 = xnn_min_f32(vacc3, vmax);
      vout0 = xnn_max_f32(vout0, vmin);
      vout1 = xnn_max_f32(vout1, vmin);
      vout2 = xnn_max_f32(vout2, vmin);
      vout3 = xnn_max_f32(vout3, vmin);

      xnn_storeu_f32(output, vout0);
      xnn_storeu_f32(output + 32, vout1);
      xnn_storeu_f32(output + 64, vout2);
      xnn_storeu_f32(output + 96, vout3);
      output = (float*) ((uintptr_t) output + output_stride);
    } while (--n != 0);
    output = (float*) ((uintptr_t) output - output_decrement);
    input += 128;
    mc -= 128 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 64 * sizeof(float);
    if (mc & (64 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        HVX_Vector vacc0 = xnn_set1_f32(*w); w += 1;
        HVX_Vector vacc1 = vacc0;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const HVX_Vector vi0 = xnn_loadu_f32(input);
            const HVX_Vector vi1 = xnn_loadu_f32(input + 32);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const HVX_Vector vw = xnn_set1_f32(*w); w += 1;
            vacc0 = xnn_fmadd_f32(vi0, vw, vacc0);
            vacc1 = xnn_fmadd_f32(vi1, vw, vacc1);
          } while (--nnz != 0);
        }
        HVX_Vector vout0 = xnn_min_f32(vacc0, vmax);
        HVX_Vector vout1 = xnn_min_f32(vacc1, vmax);
        vout0 = xnn_max_f32(vout0, vmin);
        vout1 = xnn_max_f32(vout1, vmin);

        xnn_storeu_f32(output, vout0);
        xnn_storeu_f32(output + 32, vout1);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 64;
    }
    output_decrement += 32 * sizeof(float);
    if (mc & (32 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        HVX_Vector vacc0 = xnn_set1_f32(*w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const HVX_Vector vi0 = xnn_loadu_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const HVX_Vector vw = xnn_set1_f32(*w); w += 1;
            vacc0 = xnn_fmadd_f32(vi0, vw, vacc0);
          } while (--nnz != 0);
        }
        HVX_Vector vout0 = xnn_min_f32(vacc0, vmax);
        vout0 = xnn_max_f32(vout0, vmin);

        xnn_storeu_f32(output, vout0);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 32;
    }
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        HVX_Vector vacc0 = xnn_set1_f32(*w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const HVX_Vector vi0 = xnn_loadu_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const HVX_Vector vw = xnn_set1_f32(*w); w += 1;
            vacc0 = xnn_fmadd_f32(vi0, vw, vacc0);
          } while (--nnz != 0);
        }
        HVX_Vector vout0 = xnn_min_f32(vacc0, vmax);
        vout0 = xnn_max_f32(vout0, vmin);

        xnn_store_tail_f32(output, vout0, 16);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
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
        HVX_Vector vacc0 = xnn_set1_f32(*w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const HVX_Vector vi0 = xnn_loadu_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const HVX_Vector vw = xnn_set1_f32(*w); w += 1;
            vacc0 = xnn_fmadd_f32(vi0, vw, vacc0);
          } while (--nnz != 0);
        }
        HVX_Vector vout0 = xnn_min_f32(vacc0, vmax);
        vout0 = xnn_max_f32(vout0, vmin);

        xnn_store_tail_f32(output, vout0, 8);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
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
        HVX_Vector vacc0 = xnn_set1_f32(*w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const HVX_Vector vi0 = xnn_loadu_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const HVX_Vector vw = xnn_set1_f32(*w); w += 1;
            vacc0 = xnn_fmadd_f32(vi0, vw, vacc0);
          } while (--nnz != 0);
        }
        HVX_Vector vout0 = xnn_min_f32(vacc0, vmax);
        vout0 = xnn_max_f32(vout0, vmin);

        xnn_store_tail_f32(output, vout0, 4);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
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
        HVX_Vector vacc0 = xnn_set1_f32(*w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const HVX_Vector vi0 = xnn_loadu_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const HVX_Vector vw = xnn_set1_f32(*w); w += 1;
            vacc0 = xnn_fmadd_f32(vi0, vw, vacc0);
          } while (--nnz != 0);
        }
        HVX_Vector vout0 = xnn_min_f32(vacc0, vmax);
        vout0 = xnn_max_f32(vout0, vmin);

        xnn_store_tail_f32(output, vout0, 2);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        HVX_Vector vacc0 = xnn_set1_f32(*w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const HVX_Vector vi0 = xnn_loadu_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const HVX_Vector vw = xnn_set1_f32(*w); w += 1;
            vacc0 = xnn_fmadd_f32(vi0, vw, vacc0);
          } while (--nnz != 0);
        }
        HVX_Vector vout0 = xnn_min_f32(vacc0, vmax);
        vout0 = xnn_max_f32(vout0, vmin);

        xnn_store_tail_f32(output, vout0, 1);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
    }
  }
}
