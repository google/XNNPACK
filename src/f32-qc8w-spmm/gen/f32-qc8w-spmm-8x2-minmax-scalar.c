// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <xnnpack/unaligned.h>

#include <xnnpack/math.h>
#include <xnnpack/spmm.h>


void xnn_f32_qc8w_spmm_minmax_ukernel_8x2__scalar(
    size_t mc,
    size_t nc,
    const float* input,
    const void* weights,
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
    const void* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 2) {
      uint32_t nnz = *nnzmap++;
      float vacc0x0 = unaligned_indexed_load_f32(w, 0);
      float vacc0x1 = unaligned_indexed_load_f32(w, 1);
      w = (const float*) w + 2;
      float vacc1x0 = vacc0x0;
      float vacc1x1 = vacc0x1;
      float vacc2x0 = vacc0x0;
      float vacc2x1 = vacc0x1;
      float vacc3x0 = vacc0x0;
      float vacc3x1 = vacc0x1;
      float vacc4x0 = vacc0x0;
      float vacc4x1 = vacc0x1;
      float vacc5x0 = vacc0x0;
      float vacc5x1 = vacc0x1;
      float vacc6x0 = vacc0x0;
      float vacc6x1 = vacc0x1;
      float vacc7x0 = vacc0x0;
      float vacc7x1 = vacc0x1;
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
          const float vw0 = (float) ((const int8_t*) w)[0];
          const float vw1 = (float) ((const int8_t*) w)[1];
          w = (const int8_t*) w + 2;
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
        } while (--nnz != 0);
      }
      const float vscale0 = unaligned_indexed_load_f32(w, 0);
      const float vscale1 = unaligned_indexed_load_f32(w, 1);
      w = (const float*) w + 2;
      vacc0x0 *= vscale0;
      vacc1x0 *= vscale0;
      vacc2x0 *= vscale0;
      vacc3x0 *= vscale0;
      vacc4x0 *= vscale0;
      vacc5x0 *= vscale0;
      vacc6x0 *= vscale0;
      vacc7x0 *= vscale0;
      vacc0x1 *= vscale1;
      vacc1x1 *= vscale1;
      vacc2x1 *= vscale1;
      vacc3x1 *= vscale1;
      vacc4x1 *= vscale1;
      vacc5x1 *= vscale1;
      vacc6x1 *= vscale1;
      vacc7x1 *= vscale1;
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
      output[0] = vout0x1;
      output[1] = vout1x1;
      output[2] = vout2x1;
      output[3] = vout3x1;
      output[4] = vout4x1;
      output[5] = vout5x1;
      output[6] = vout6x1;
      output[7] = vout7x1;
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
      n -= 2;
    }
    if XNN_UNLIKELY(n != 0) {
      do {
        uint32_t nnz = *nnzmap++;
        float vacc0 = unaligned_load_f32(w);
        w = (const float*) w + 1;
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
            const float vw = (float) ((const int8_t*) w)[1];
            w = (const int8_t*) w + 1;
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
        float vscale = unaligned_load_f32(w);
        w = (const float*) w + 1;
        vacc0 *= vscale;
        vacc1 *= vscale;
        vacc2 *= vscale;
        vacc3 *= vscale;
        vacc4 *= vscale;
        vacc5 *= vscale;
        vacc6 *= vscale;
        vacc7 *= vscale;
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
      const void* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = unaligned_indexed_load_f32(w, 0);
        float vacc0x1 = unaligned_indexed_load_f32(w, 1);
        w = (const float*) w + 2;
        float vacc1x0 = vacc0x0;
        float vacc2x0 = vacc0x0;
        float vacc3x0 = vacc0x0;
        float vacc1x1 = vacc0x1;
        float vacc2x1 = vacc0x1;
        float vacc3x1 = vacc0x1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            const float vi2 = input[2];
            const float vi3 = input[3];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = (float) ((const int8_t*) w)[0];
            const float vw1 = (float) ((const int8_t*) w)[1];
            w = (const int8_t*) w + 2;
            vacc0x0 += vi0 * vw0;
            vacc1x0 += vi1 * vw0;
            vacc2x0 += vi2 * vw0;
            vacc3x0 += vi3 * vw0;
            vacc0x1 += vi0 * vw1;
            vacc1x1 += vi1 * vw1;
            vacc2x1 += vi2 * vw1;
            vacc3x1 += vi3 * vw1;
          } while (--nnz != 0);
        }
        const float vscale0 = unaligned_indexed_load_f32(w, 0);
        const float vscale1 = unaligned_indexed_load_f32(w, 1);
        w = (const float*) w + 2;
        vacc0x0 *= vscale0;
        vacc1x0 *= vscale0;
        vacc2x0 *= vscale0;
        vacc3x0 *= vscale0;
        vacc0x1 *= vscale1;
        vacc1x1 *= vscale1;
        vacc2x1 *= vscale1;
        vacc3x1 *= vscale1;
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout1x0 = math_min_f32(vacc1x0, vmax);
        float vout2x0 = math_min_f32(vacc2x0, vmax);
        float vout3x0 = math_min_f32(vacc3x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        float vout1x1 = math_min_f32(vacc1x1, vmax);
        float vout2x1 = math_min_f32(vacc2x1, vmax);
        float vout3x1 = math_min_f32(vacc3x1, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout1x0 = math_max_f32(vout1x0, vmin);
        vout2x0 = math_max_f32(vout2x0, vmin);
        vout3x0 = math_max_f32(vout3x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        vout1x1 = math_max_f32(vout1x1, vmin);
        vout2x1 = math_max_f32(vout2x1, vmin);
        vout3x1 = math_max_f32(vout3x1, vmin);
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
        n -= 2;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = unaligned_load_f32(w);
          w = (const float*) w + 1;
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
              const float vw = (float) ((const int8_t*) w)[1];
              w = (const int8_t*) w + 1;
              vacc0 += vi0 * vw;
              vacc1 += vi1 * vw;
              vacc2 += vi2 * vw;
              vacc3 += vi3 * vw;
            } while (--nnz != 0);
          }
          float vscale = unaligned_load_f32(w);
          w = (const float*) w + 1;
          vacc0 *= vscale;
          vacc1 *= vscale;
          vacc2 *= vscale;
          vacc3 *= vscale;
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
      const void* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = unaligned_indexed_load_f32(w, 0);
        float vacc0x1 = unaligned_indexed_load_f32(w, 1);
        w = (const float*) w + 2;
        float vacc1x0 = vacc0x0;
        float vacc1x1 = vacc0x1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            const float vi1 = input[1];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = (float) ((const int8_t*) w)[0];
            const float vw1 = (float) ((const int8_t*) w)[1];
            w = (const int8_t*) w + 2;
            vacc0x0 += vi0 * vw0;
            vacc1x0 += vi1 * vw0;
            vacc0x1 += vi0 * vw1;
            vacc1x1 += vi1 * vw1;
          } while (--nnz != 0);
        }
        const float vscale0 = unaligned_indexed_load_f32(w, 0);
        const float vscale1 = unaligned_indexed_load_f32(w, 1);
        w = (const float*) w + 2;
        vacc0x0 *= vscale0;
        vacc1x0 *= vscale0;
        vacc0x1 *= vscale1;
        vacc1x1 *= vscale1;
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout1x0 = math_min_f32(vacc1x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        float vout1x1 = math_min_f32(vacc1x1, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout1x0 = math_max_f32(vout1x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        vout1x1 = math_max_f32(vout1x1, vmin);
        output[0] = vout0x0;
        output[1] = vout1x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x1;
        output[1] = vout1x1;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 2;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = unaligned_load_f32(w);
          w = (const float*) w + 1;
          float vacc1 = vacc0;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              const float vi1 = input[1];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = (float) ((const int8_t*) w)[1];
              w = (const int8_t*) w + 1;
              vacc0 += vi0 * vw;
              vacc1 += vi1 * vw;
            } while (--nnz != 0);
          }
          float vscale = unaligned_load_f32(w);
          w = (const float*) w + 1;
          vacc0 *= vscale;
          vacc1 *= vscale;
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
      const void* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = unaligned_indexed_load_f32(w, 0);
        float vacc0x1 = unaligned_indexed_load_f32(w, 1);
        w = (const float*) w + 2;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float vi0 = input[0];
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float vw0 = (float) ((const int8_t*) w)[0];
            const float vw1 = (float) ((const int8_t*) w)[1];
            w = (const int8_t*) w + 2;
            vacc0x0 += vi0 * vw0;
            vacc0x1 += vi0 * vw1;
          } while (--nnz != 0);
        }
        const float vscale0 = unaligned_indexed_load_f32(w, 0);
        const float vscale1 = unaligned_indexed_load_f32(w, 1);
        w = (const float*) w + 2;
        vacc0x0 *= vscale0;
        vacc0x1 *= vscale1;
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        float vout0x1 = math_min_f32(vacc0x1, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        vout0x1 = math_max_f32(vout0x1, vmin);
        output[0] = vout0x0;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        output[0] = vout0x1;
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 2;
      }
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = unaligned_load_f32(w);
          w = (const float*) w + 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float vi0 = input[0];
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float vw = (float) ((const int8_t*) w)[1];
              w = (const int8_t*) w + 1;
              vacc0 += vi0 * vw;
            } while (--nnz != 0);
          }
          float vscale = unaligned_load_f32(w);
          w = (const float*) w + 1;
          vacc0 *= vscale;
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
