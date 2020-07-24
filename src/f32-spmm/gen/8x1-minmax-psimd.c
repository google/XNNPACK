// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/psimd.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/spmm.h>


void xnn_f32_spmm_minmax_ukernel_8x1__psimd(
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

  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);
  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
  size_t n = batch_size;
  while XNN_LIKELY(n >= 8) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t c = output_channels;
    do {
      uint32_t nnz = *nnzmap++;
      psimd_f32 vacc0123 = psimd_load_splat_f32(w); w += 1;
      psimd_f32 vacc4567 = vacc0123;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const psimd_f32 vi0123 = psimd_load_f32(input);
          const psimd_f32 vi4567 = psimd_load_f32(input + 4);
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const psimd_f32 vw = psimd_load_splat_f32(w); w += 1;
          vacc0123 = psimd_qfma_f32(vacc0123, vi0123, vw);
          vacc4567 = psimd_qfma_f32(vacc4567, vi4567, vw);
        } while (--nnz != 0);
      }
      psimd_f32 vout0123 = psimd_min_f32(vacc0123, vmax);
      psimd_f32 vout4567 = psimd_min_f32(vacc4567, vmax);
      vout0123 = psimd_max_f32(vout0123, vmin);
      vout4567 = psimd_max_f32(vout4567, vmin);
      psimd_store_f32(output, vout0123);
      psimd_store_f32(output + 4, vout4567);
      output += 1 * batch_size;
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
      size_t c = output_channels;
      do {
        uint32_t nnz = *nnzmap++;
        psimd_f32 vacc0123 = psimd_load_splat_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const psimd_f32 vi0123 = psimd_load_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const psimd_f32 vw = psimd_load_splat_f32(w); w += 1;
            vacc0123 = psimd_qfma_f32(vacc0123, vi0123, vw);
          } while (--nnz != 0);
        }
        psimd_f32 vout0123 = psimd_min_f32(vacc0123, vmax);
        vout0123 = psimd_max_f32(vout0123, vmin);
        psimd_store_f32(output, vout0123);
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
        psimd_f32 vacc01 = psimd_load_splat_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const psimd_f32 vi01 = psimd_load2_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const psimd_f32 vw = psimd_load_splat_f32(w); w += 1;
            vacc01 = psimd_qfma_f32(vacc01, vi01, vw);
          } while (--nnz != 0);
        }
        psimd_f32 vout01 = psimd_min_f32(vacc01, vmax);
        vout01 = psimd_max_f32(vout01, vmin);
        psimd_store2_f32(output, vout01);
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
        psimd_f32 vacc0 = psimd_load_splat_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const psimd_f32 vi0 = psimd_load_splat_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const psimd_f32 vw = psimd_load_splat_f32(w); w += 1;
            vacc0 = psimd_qfma_f32(vacc0, vi0, vw);
          } while (--nnz != 0);
        }
        psimd_f32 vout0 = psimd_min_f32(vacc0, vmax);
        vout0 = psimd_max_f32(vout0, vmin);
        psimd_store1_f32(output, vout0);
        output += 1 * batch_size;
      } while (--c != 0);
      output -= batch_size * output_channels;
      output += 1;
      input += 1;
    }
  }
}
