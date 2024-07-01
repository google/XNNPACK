// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/scalar-pipelined.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/spmm.h"


void xnn_f32_spmm_minmax_ukernel_8x1__scalar_pipelined(
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
  while XNN_LIKELY(mc >= 8 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    float vw = *w++;
    intptr_t diff = *dmap++;
    float vi0 = input[0];
    float vi1 = input[1];
    float vi2 = input[2];
    float vi3 = input[3];
    float vi4 = input[4];
    float vi5 = input[5];
    float vi6 = input[6];
    float vi7 = input[7];
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      float vacc0 = vw;
      float vacc1 = vw;
      float vacc2 = vw;
      float vacc3 = vw;
      float vacc4 = vw;
      float vacc5 = vw;
      float vacc6 = vw;
      float vacc7 = vw;
      vw = *w++;
      if XNN_LIKELY(nnz != 0) {
        do {
          vacc0 += vi0 * vw;
          vacc1 += vi1 * vw;
          vacc2 += vi2 * vw;
          vacc3 += vi3 * vw;
          vacc4 += vi4 * vw;
          vacc5 += vi5 * vw;
          vacc6 += vi6 * vw;
          vacc7 += vi7 * vw;
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);

          diff = *dmap++;
          vw = *w++;
          vi0 = input[0];
          vi1 = input[1];
          vi2 = input[2];
          vi3 = input[3];
          vi4 = input[4];
          vi5 = input[5];
          vi6 = input[6];
          vi7 = input[7];
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
      float vw = *w++;
      intptr_t diff = *dmap++;
      float vi0 = input[0];
      float vi1 = input[1];
      float vi2 = input[2];
      float vi3 = input[3];
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float vacc0 = vw;
        float vacc1 = vw;
        float vacc2 = vw;
        float vacc3 = vw;
        vw = *w++;
        if XNN_LIKELY(nnz != 0) {
          do {
            vacc0 += vi0 * vw;
            vacc1 += vi1 * vw;
            vacc2 += vi2 * vw;
            vacc3 += vi3 * vw;
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);

            diff = *dmap++;
            vw = *w++;
            vi0 = input[0];
            vi1 = input[1];
            vi2 = input[2];
            vi3 = input[3];
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
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      float vw = *w++;
      intptr_t diff = *dmap++;
      float vi0 = input[0];
      float vi1 = input[1];
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float vacc0 = vw;
        float vacc1 = vw;
        vw = *w++;
        if XNN_LIKELY(nnz != 0) {
          do {
            vacc0 += vi0 * vw;
            vacc1 += vi1 * vw;
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);

            diff = *dmap++;
            vw = *w++;
            vi0 = input[0];
            vi1 = input[1];
          } while (--nnz != 0);
        }
        float vout0 = math_min_f32(vacc0, vmax);
        float vout1 = math_min_f32(vacc1, vmax);
        vout0 = math_max_f32(vout0, vmin);
        vout1 = math_max_f32(vout1, vmin);
        output[0] = vout0;
        output[1] = vout1;
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
      float vw = *w++;
      intptr_t diff = *dmap++;
      float vi0 = input[0];
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float vacc0 = vw;
        vw = *w++;
        if XNN_LIKELY(nnz != 0) {
          do {
            vacc0 += vi0 * vw;
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);

            diff = *dmap++;
            vw = *w++;
            vi0 = input[0];
          } while (--nnz != 0);
        }
        float vout0 = math_min_f32(vacc0, vmax);
        vout0 = math_max_f32(vout0, vmin);
        output[0] = vout0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}
