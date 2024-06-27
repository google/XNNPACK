// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/spmm.h"


void xnn_f32_spmm_minmax_ukernel_8x4__scalar(
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

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  size_t output_decrement = output_stride * nc - 8 * sizeof(float);
  while (mc >= 8 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 4) {
      uint32_t nnz = *nnzmap++;
      float vacc0x0 = *w++;
      float vacc0x1 = *w++;
      float vacc0x2 = *w++;
      float vacc0x3 = *w++;
      float vacc1x0 = vacc0x0;
      float vacc1x1 = vacc0x1;
      float vacc1x2 = vacc0x2;
      float vacc1x3 = vacc0x3;
      float vacc2x0 = vacc0x0;
      float vacc2x1 = vacc0x1;
      float vacc2x2 = vacc0x2;
      float vacc2x3 = vacc0x3;
      float vacc3x0 = vacc0x0;
      float vacc3x1 = vacc0x1;
      float vacc3x2 = vacc0x2;
      float vacc3x3 = vacc0x3;
      float vacc4x0 = vacc0x0;
      float vacc4x1 = vacc0x1;
      float vacc4x2 = vacc0x2;
      float vacc4x3 = vacc0x3;
      float vacc5x0 = vacc0x0;
      float vacc5x1 = vacc0x1;
      float vacc5x2 = vacc0x2;
      float vacc5x3 = vacc0x3;
      float vacc6x0 = vacc0x0;
      float vacc6x1 = vacc0x1;
      float vacc6x2 = vacc0x2;
      float vacc6x3 = vacc0x3;
      float vacc7x0 = vacc0x0;
      float vacc7x1 = vacc0x1;
      float vacc7x2 = vacc0x2;
      float vacc7x3 = vacc0x3;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float vi0 = input[0];
          const float vi1 = input[1];
          const float vi2 = input[2];
          const float vi3 = input[3];
          const float vi4 = input[4];
          const float vi5 = input[5];
          const float vi6 = input[6];
          const float vi7 = input[7];
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const float vw0 = *w++;
          const float vw1 = *w++;
          const float vw2 = *w++;
          const float vw3 = *w++;
          vacc0x0 += vi0 * vw0;
          vacc1x0 += vi1 * vw0;
          vacc2x0 += vi2 * vw0;
          vacc3x0 += vi3 * vw0;
          vacc4x0 += vi4 * vw0;
          vacc5x0 += vi5 * vw0;
          vacc6x0 += vi6 * vw0;
          vacc7x0 += vi7 * vw0;
          vacc0x1 += vi0 * vw1;
          vacc1x1 += vi1 * vw1;
          vacc2x1 += vi2 * vw1;
          vacc3x1 += vi3 * vw1;
          vacc4x1 += vi4 * vw1;
          vacc5x1 += vi5 * vw1;
          vacc6x1 += vi6 * vw1;
          vacc7x1 += vi7 * vw1;
          vacc0x2 += vi0 * vw2;
          vacc1x2 += vi1 * vw2;
          vacc2x2 += vi2 * vw2;
          vacc3x2 += vi3 * vw2;
          vacc4x2 += vi4 * vw2;
          vacc5x2 += vi5 * vw2;
          vacc6x2 += vi6 * vw2;
          vacc7x2 += vi7 * vw2;
          vacc0x3 += vi0 * vw3;
          vacc1x3 += vi1 * vw3;
          vacc2x3 += vi2 * vw3;
          vacc3x3 += vi3 * vw3;
          vacc4x3 += vi4 * vw3;
          vacc5x3 += vi5 * vw3;
          vacc6x3 += vi6 * vw3;
          vacc7x3 += vi7 * vw3;
        } while (--nnz != 0);
      }
      float vout0x0 = math_min_f32(vacc0x0, vmax);
      float vout1x0 = math_min_f32(vacc1x0, vmax);
      float vout2x0 = math_min_f32(vacc2x0, vmax);
      float vout3x0 = math_min_f32(vacc3x0, vmax);
      float vout4x0 = math_min_f32(vacc4x0, vmax);
      float vout5x0 = math_min_f32(vacc5x0, vmax);
      float vout6x0 = math_min_f32(vacc6x0, vmax);
      float vout7x0 = math_min_f32(vacc7x0, vmax);
      float vout0x1 = math_min_f32(vacc0x1, vmax);
      float vout1x1 = math_min_f32(vacc1x1, vmax);
      float vout2x1 = math_min_f32(vacc2x1, vmax);
      float vout3x1 = math_min_f32(vacc3x1, vmax);
      float vout4x1 = math_min_f32(vacc4x1, vmax);
      float vout5x1 = math_min_f32(vacc5x1, vmax);
      float vout6x1 = math_min_f32(vacc6x1, vmax);
      float vout7x1 = math_min_f32(vacc7x1, vmax);
      float vout0x2 = math_min_f32(vacc0x2, vmax);
      float vout1x2 = math_min_f32(vacc1x2, vmax);
      float vout2x2 = math_min_f32(vacc2x2, vmax);
      float vout3x2 = math_min_f32(vacc3x2, vmax);
      float vout4x2 = math_min_f32(vacc4x2, vmax);
      float vout5x2 = math_min_f32(vacc5x2, vmax);
      float vout6x2 = math_min_f32(vacc6x2, vmax);
      float vout7x2 = math_min_f32(vacc7x2, vmax);
      float vout0x3 = math_min_f32(vacc0x3, vmax);
      float vout1x3 = math_min_f32(vacc1x3, vmax);
      float vout2x3 = math_min_f32(vacc2x3, vmax);
      float vout3x3 = math_min_f32(vacc3x3, vmax);
      float vout4x3 = math_min_f32(vacc4x3, vmax);
      float vout5x3 = math_min_f32(vacc5x3, vmax);
      float vout6x3 = math_min_f32(vacc6x3, vmax);
      float vout7x3 = math_min_f32(vacc7x3, vmax);
      vout0x0 = math_max_f32(vout0x0, vmin);
      vout1x0 = math_max_f32(vout1x0, vmin);
      vout2x0 = math_max_f32(vout2x0, vmin);
      vout3x0 = math_max_f32(vout3x0, vmin);
      vout4x0 = math_max_f32(vout4x0, vmin);
      vout5x0 = math_max_f32(vout5x0, vmin);
      vout6x0 = math_max_f32(vout6x0, vmin);
      vout7x0 = math_max_f32(vout7x0, vmin);
      vout0x1 = math_max_f32(vout0x1, vmin);
      vout1x1 = math_max_f32(vout1x1, vmin);
      vout2x1 = math_max_f32(vout2x1, vmin);
      vout3x1 = math_max_f32(vout3x1, vmin);
      vout4x1 = math_max_f32(vout4x1, vmin);
      vout5x1 = math_max_f32(vout5x1, vmin);
      vout6x1 = math_max_f32(vout6x1, vmin);
      vout7x1 = math_max_f32(vout7x1, vmin);
      vout0x2 = math_max_f32(vout0x2, vmin);
      vout1x2 = math_max_f32(vout1x2, vmin);
      vout2x2 = math_max_f32(vout2x2, vmin);
      vout3x2 = math_max_f32(vout3x2, vmin);
      vout4x2 = math_max_f32(vout4x2, vmin);
      vout5x2 = math_max_f32(vout5x2, vmin);
      vout6x2 = math_max_f32(vout6x2, vmin);
      vout7x2 = math_max_f32(vout7x2, vmin);
      vout0x3 = math_max_f32(vout0x3, vmin);
      vout1x3 = math_max_f32(vout1x3, vmin);
      vout2x3 = math_max_f32(vout2x3, vmin);
      vout3x3 = math_max_f32(vout3x3, vmin);
      vout4x3 = math_max_f32(vout4x3, vmin);
      vout5x3 = math_max_f32(vout5x3, vmin);
      vout6x3 = math_max_f32(vout6x3, vmin);
      vout7x3 = math_max_f32(vout7x3, vmin);
      output[0] = vout0x3;
      output[1] = vout1x3;
      output[2] = vout2x3;
      output[3] = vout3x3;
      output[4] = vout4x3;
      output[5] = vout5x3;
      output[6] = vout6x3;
      output[7] = vout7x3;
      output[0] = vout0x0;
      output[1] = vout1x0;
      output[2] = vout2x0;
      output[3] = vout3x0;
      output[4] = vout4x0;
      output[5] = vout5x0;
      output[6] = vout6x0;
      output[7] = vout7x0;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      output[0] = vout0x1;
      output[1] = vout1x1;
      output[2] = vout2x1;
      output[3] = vout3x1;
      output[4] = vout4x1;
      output[5] = vout5x1;
      output[6] = vout6x1;
      output[7] = vout7x1;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      output[0] = vout0x2;
      output[1] = vout1x2;
      output[2] = vout2x2;
      output[3] = vout3x2;
      output[4] = vout4x2;
      output[5] = vout5x2;
      output[6] = vout6x2;
      output[7] = vout7x2;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      output[0] = vout0x3;
      output[1] = vout1x3;
      output[2] = vout2x3;
      output[3] = vout3x3;
      output[4] = vout4x3;
      output[5] = vout5x3;
      output[6] = vout6x3;
      output[7] = vout7x3;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      n -= 4;
    }
    if XNN_UNLIKELY(n != 0) {
      do {
        uint32_t nnz = *nnzmap++;
        float vacc0 = *w++;
        float vacc1 = vacc0;
        float vacc2 = vacc0;
        float vacc3 = vacc0;
        float vacc4 = vacc0;
        float vacc5 = vacc0;
        float vacc6 = vacc0;
        float vacc7 = vacc0;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            const float vi2 = input[2];
            const float vi3 = input[3];
            const float vi4 = input[4];
            const float vi5 = input[5];
            const float vi6 = input[6];
            const float vi7 = input[7];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw = *w++;
            vacc0 += vi0 * vw;
            vacc1 += vi1 * vw;
            vacc2 += vi2 * vw;
            vacc3 += vi3 * vw;
            vacc4 += vi4 * vw;
            vacc5 += vi5 * vw;
            vacc6 += vi6 * vw;
            vacc7 += vi7 * vw;
          } while (--nnz != 0);
        }
        float vout0 = math_min_f32(vacc0, vmax);
        float vout1 = math_min_f32(vacc1, vmax);
        float vout2 = math_min_f32(vacc2, vmax);
        float vout3 = math_min_f32(vacc3, vmax);
        float vout4 = math_min_f32(vacc4, vmax);
        float vout5 = math_min_f32(vacc5, vmax);
        float vout6 = math_min_f32(vacc6, vmax);
        float vout7 = math_min_f32(vacc7, vmax);
        vout0 = math_max_f32(vout0, vmin);
        vout1 = math_max_f32(vout1, vmin);
        vout2 = math_max_f32(vout2, vmin);
        vout3 = math_max_f32(vout3, vmin);
        vout4 = math_max_f32(vout4, vmin);
        vout5 = math_max_f32(vout5, vmin);
        vout6 = math_max_f32(vout6, vmin);
        vout7 = math_max_f32(vout7, vmin);
        output[0] = vout0;
        output[1] = vout1;
        output[2] = vout2;
        output[3] = vout3;
        output[4] = vout4;
        output[5] = vout5;
        output[6] = vout6;
        output[7] = vout7;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      } while (n != 0);
    }
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
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc0x1 = *w++;
        float vacc0x2 = *w++;
        float vacc0x3 = *w++;
        float vacc1x0 = vacc0x0;
        float vacc2x0 = vacc0x0;
        float vacc3x0 = vacc0x0;
        float vacc1x1 = vacc0x1;
        float vacc2x1 = vacc0x1;
        float vacc3x1 = vacc0x1;
        float vacc1x2 = vacc0x2;
        float vacc2x2 = vacc0x2;
        float vacc3x2 = vacc0x2;
        float vacc1x3 = vacc0x3;
        float vacc2x3 = vacc0x3;
        float vacc3x3 = vacc0x3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            const float vi2 = input[2];
            const float vi3 = input[3];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            const float vw1 = *w++;
            const float vw2 = *w++;
            const float vw3 = *w++;
            vacc0x0 += vi0 * vw0;
            vacc1x0 += vi1 * vw0;
            vacc2x0 += vi2 * vw0;
            vacc3x0 += vi3 * vw0;
            vacc0x1 += vi0 * vw1;
            vacc1x1 += vi1 * vw1;
            vacc2x1 += vi2 * vw1;
            vacc3x1 += vi3 * vw1;
            vacc0x2 += vi0 * vw2;
            vacc1x2 += vi1 * vw2;
            vacc2x2 += vi2 * vw2;
            vacc3x2 += vi3 * vw2;
            vacc0x3 += vi0 * vw3;
            vacc1x3 += vi1 * vw3;
            vacc2x3 += vi2 * vw3;
            vacc3x3 += vi3 * vw3;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout1x0 = math_min_f32(vacc1x0, vmax);
        float vout2x0 = math_min_f32(vacc2x0, vmax);
        float vout3x0 = math_min_f32(vacc3x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        float vout1x1 = math_min_f32(vacc1x1, vmax);
        float vout2x1 = math_min_f32(vacc2x1, vmax);
        float vout3x1 = math_min_f32(vacc3x1, vmax);
        float vout0x2 = math_min_f32(vacc0x2, vmax);
        float vout1x2 = math_min_f32(vacc1x2, vmax);
        float vout2x2 = math_min_f32(vacc2x2, vmax);
        float vout3x2 = math_min_f32(vacc3x2, vmax);
        float vout0x3 = math_min_f32(vacc0x3, vmax);
        float vout1x3 = math_min_f32(vacc1x3, vmax);
        float vout2x3 = math_min_f32(vacc2x3, vmax);
        float vout3x3 = math_min_f32(vacc3x3, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout1x0 = math_max_f32(vout1x0, vmin);
        vout2x0 = math_max_f32(vout2x0, vmin);
        vout3x0 = math_max_f32(vout3x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        vout1x1 = math_max_f32(vout1x1, vmin);
        vout2x1 = math_max_f32(vout2x1, vmin);
        vout3x1 = math_max_f32(vout3x1, vmin);
        vout0x2 = math_max_f32(vout0x2, vmin);
        vout1x2 = math_max_f32(vout1x2, vmin);
        vout2x2 = math_max_f32(vout2x2, vmin);
        vout3x2 = math_max_f32(vout3x2, vmin);
        vout0x3 = math_max_f32(vout0x3, vmin);
        vout1x3 = math_max_f32(vout1x3, vmin);
        vout2x3 = math_max_f32(vout2x3, vmin);
        vout3x3 = math_max_f32(vout3x3, vmin);
        output[0] = vout0x0;
        output[1] = vout1x0;
        output[2] = vout2x0;
        output[3] = vout3x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x1;
        output[1] = vout1x1;
        output[2] = vout2x1;
        output[3] = vout3x1;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x2;
        output[1] = vout1x2;
        output[2] = vout2x2;
        output[3] = vout3x2;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x3;
        output[1] = vout1x3;
        output[2] = vout2x3;
        output[3] = vout3x3;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          float vacc1 = vacc0;
          float vacc2 = vacc0;
          float vacc3 = vacc0;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              const float vi1 = input[1];
              const float vi2 = input[2];
              const float vi3 = input[3];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
              vacc1 += vi1 * vw;
              vacc2 += vi2 * vw;
              vacc3 += vi3 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          float vout1 = math_min_f32(vacc1, vmax);
          float vout2 = math_min_f32(vacc2, vmax);
          float vout3 = math_min_f32(vacc3, vmax);
          vout0 = math_max_f32(vout0, vmin);
          vout1 = math_max_f32(vout1, vmin);
          vout2 = math_max_f32(vout2, vmin);
          vout3 = math_max_f32(vout3, vmin);
          output[0] = vout0;
          output[1] = vout1;
          output[2] = vout2;
          output[3] = vout3;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc0x1 = *w++;
        float vacc0x2 = *w++;
        float vacc0x3 = *w++;
        float vacc1x0 = vacc0x0;
        float vacc1x1 = vacc0x1;
        float vacc1x2 = vacc0x2;
        float vacc1x3 = vacc0x3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            const float vw1 = *w++;
            const float vw2 = *w++;
            const float vw3 = *w++;
            vacc0x0 += vi0 * vw0;
            vacc1x0 += vi1 * vw0;
            vacc0x1 += vi0 * vw1;
            vacc1x1 += vi1 * vw1;
            vacc0x2 += vi0 * vw2;
            vacc1x2 += vi1 * vw2;
            vacc0x3 += vi0 * vw3;
            vacc1x3 += vi1 * vw3;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout1x0 = math_min_f32(vacc1x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        float vout1x1 = math_min_f32(vacc1x1, vmax);
        float vout0x2 = math_min_f32(vacc0x2, vmax);
        float vout1x2 = math_min_f32(vacc1x2, vmax);
        float vout0x3 = math_min_f32(vacc0x3, vmax);
        float vout1x3 = math_min_f32(vacc1x3, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout1x0 = math_max_f32(vout1x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        vout1x1 = math_max_f32(vout1x1, vmin);
        vout0x2 = math_max_f32(vout0x2, vmin);
        vout1x2 = math_max_f32(vout1x2, vmin);
        vout0x3 = math_max_f32(vout0x3, vmin);
        vout1x3 = math_max_f32(vout1x3, vmin);
        output[0] = vout0x0;
        output[1] = vout1x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x1;
        output[1] = vout1x1;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x2;
        output[1] = vout1x2;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x3;
        output[1] = vout1x3;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          float vacc1 = vacc0;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              const float vi1 = input[1];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
              vacc1 += vi1 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          float vout1 = math_min_f32(vacc1, vmax);
          vout0 = math_max_f32(vout0, vmin);
          vout1 = math_max_f32(vout1, vmin);
          output[0] = vout0;
          output[1] = vout1;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc0x1 = *w++;
        float vacc0x2 = *w++;
        float vacc0x3 = *w++;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = *w++;
            const float vw1 = *w++;
            const float vw2 = *w++;
            const float vw3 = *w++;
            vacc0x0 += vi0 * vw0;
            vacc0x1 += vi0 * vw1;
            vacc0x2 += vi0 * vw2;
            vacc0x3 += vi0 * vw3;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        float vout0x2 = math_min_f32(vacc0x2, vmax);
        float vout0x3 = math_min_f32(vacc0x3, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        vout0x2 = math_max_f32(vout0x2, vmin);
        vout0x3 = math_max_f32(vout0x3, vmin);
        output[0] = vout0x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x1;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x2;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x3;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = *w++;
              vacc0 += vi0 * vw;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          vout0 = math_max_f32(vout0, vmin);
          output[0] = vout0;
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}
