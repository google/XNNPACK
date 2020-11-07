// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/scalar-pipelined.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/spmm.h>


void xnn_f32_spmm_minmax_ukernel_8x1__scalar_pipelined(
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

  const uintptr_t output_stride = batch_size * sizeof(float);
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  size_t n = batch_size;
  while XNN_LIKELY(n >= 8) {
    const float*restrict w = weights;
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
    size_t c = output_channels;
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
    } while (--c != 0);
    output -= batch_size * output_channels;
    output += 8;
    input += 8;
    n -= 8;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 4) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      float vw = *w++;
      intptr_t diff = *dmap++;
      float vi0 = input[0];
      float vi1 = input[1];
      float vi2 = input[2];
      float vi3 = input[3];
      size_t c = output_channels;
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
      } while (--c != 0);
      output -= batch_size * output_channels;
      output += 4;
      input += 4;
    }
    if (n & 2) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      float vw = *w++;
      intptr_t diff = *dmap++;
      float vi0 = input[0];
      float vi1 = input[1];
      size_t c = output_channels;
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
      } while (--c != 0);
      output -= batch_size * output_channels;
      output += 2;
      input += 2;
    }
    if (n & 1) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      float vw = *w++;
      intptr_t diff = *dmap++;
      float vi0 = input[0];
      size_t c = output_channels;
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
      } while (--c != 0);
      output -= batch_size * output_channels;
      output += 1;
      input += 1;
    }
  }
}
