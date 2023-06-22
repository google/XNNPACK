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


void xnn_f32_qc8w_spmm_minmax_ukernel_1x1__scalar(
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
  size_t output_decrement = output_stride * nc - 1 * sizeof(float);
  while (mc >= 1 * sizeof(float)) {
    const void* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 1) {
      uint32_t nnz = *nnzmap++;
      float vacc0x0 = unaligned_indexed_load_f32(w, 0);
      w = (const float*) w + 1;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float vi0 = input[0];
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const float vw0 = (float) ((const int8_t*) w)[0];
          w = (const int8_t*) w + 1;
          vacc0x0 += vi0 * vw0;
        } while (--nnz != 0);
      }
      const float vscale0 = unaligned_indexed_load_f32(w, 0);
      w = (const float*) w + 1;
      vacc0x0 *= vscale0;
      float vout0x0 = math_min_f32(vacc0x0, vmax);
      vout0x0 = math_max_f32(vout0x0, vmin);
      output[0] = vout0x0;
      output[0] = vout0x0;
      output = (float*restrict) ((uintptr_t) output + output_stride);
      n -= 1;
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
            const float vw = (float) ((const int8_t*) w)[0];
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
    mc -= 1 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
  }
}
