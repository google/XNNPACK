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
    uint32_t m,
    uint32_t n,
    const float*restrict a,
    const float*restrict weights,
    const int32_t*restrict widx_dmap,
    const uint32_t*restrict nidx_nnzmap,
    float*restrict c,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(m != 0);

  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);
  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
  size_t i = m;
  while XNN_LIKELY(i >= 8) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t j = n;
    do {
      uint32_t nnz = *nnzmap++;
      psimd_f32 vacc0123 = psimd_load_splat_f32(w); w += 1;
      psimd_f32 vacc4567 = vacc0123;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const psimd_f32 va0123 = psimd_load_f32(a);
          const psimd_f32 va4567 = psimd_load_f32(a + 4);
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
          const psimd_f32 vb = psimd_load_splat_f32(w); w += 1;
          vacc0123 = psimd_qfma_f32(vacc0123, va0123, vb);
          vacc4567 = psimd_qfma_f32(vacc4567, va4567, vb);
        } while (--nnz != 0);
      }
      psimd_f32 vout0123 = psimd_min_f32(vacc0123, vmax);
      psimd_f32 vout4567 = psimd_min_f32(vacc4567, vmax);
      vout0123 = psimd_max_f32(vout0123, vmin);
      vout4567 = psimd_max_f32(vout4567, vmin);
      psimd_store_f32(c, vout0123);
      psimd_store_f32(c + 4, vout4567);
      c += 1 * m;
    } while (--j != 0);
    c -= m * n;
    c += 8;
    a += 8;
    i -= 8;
  }
  if XNN_UNLIKELY(i != 0) {
    if (i & 4) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t j = n;
      do {
        uint32_t nnz = *nnzmap++;
        psimd_f32 vacc0123 = psimd_load_splat_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const psimd_f32 va0123 = psimd_load_f32(a);
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const psimd_f32 vb = psimd_load_splat_f32(w); w += 1;
            vacc0123 = psimd_qfma_f32(vacc0123, va0123, vb);
          } while (--nnz != 0);
        }
        psimd_f32 vout0123 = psimd_min_f32(vacc0123, vmax);
        vout0123 = psimd_max_f32(vout0123, vmin);
        psimd_store_f32(c, vout0123);
        c += 1 * m;
      } while (--j != 0);
      c -= m * n;
      c += 4;
      a += 4;
    }
    if (i & 2) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t j = n;
      do {
        uint32_t nnz = *nnzmap++;
        psimd_f32 vacc01 = psimd_load_splat_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const psimd_f32 va01 = psimd_load2_f32(a);
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const psimd_f32 vb = psimd_load_splat_f32(w); w += 1;
            vacc01 = psimd_qfma_f32(vacc01, va01, vb);
          } while (--nnz != 0);
        }
        psimd_f32 vout01 = psimd_min_f32(vacc01, vmax);
        vout01 = psimd_max_f32(vout01, vmin);
        psimd_store2_f32(c, vout01);
        c += 1 * m;
      } while (--j != 0);
      c -= m * n;
      c += 2;
      a += 2;
    }
    if (i & 1) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t j = n;
      do {
        uint32_t nnz = *nnzmap++;
        psimd_f32 vacc0 = psimd_load_splat_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const psimd_f32 va0 = psimd_load_splat_f32(a);
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const psimd_f32 vb = psimd_load_splat_f32(w); w += 1;
            vacc0 = psimd_qfma_f32(vacc0, va0, vb);
          } while (--nnz != 0);
        }
        psimd_f32 vout0 = psimd_min_f32(vacc0, vmax);
        vout0 = psimd_max_f32(vout0, vmin);
        psimd_store1_f32(c, vout0);
        c += 1 * m;
      } while (--j != 0);
      c -= m * n;
      c += 1;
      a += 1;
    }
  }
}
