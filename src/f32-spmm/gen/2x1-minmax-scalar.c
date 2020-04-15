// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/spmm.h>


void xnn_f32_spmm_minmax_ukernel_2x1__scalar(
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

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  size_t i = m;
  while (i >= 2) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t j = n;
    while (j >= 1) {
      uint32_t nnz = *nnzmap++;
      float vacc0x0 = *w++;
      float vacc1x0 = vacc0x0;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float va0 = a[0];
          const float va1 = a[1];
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
          const float vb0 = *w++;
          vacc0x0 += va0 * vb0;
          vacc1x0 += va1 * vb0;
        } while (--nnz != 0);
      }
      float vout0x0 = math_min_f32(vacc0x0, vmax);
      float vout1x0 = math_min_f32(vacc1x0, vmax);
      vout0x0 = math_max_f32(vout0x0, vmin);
      vout1x0 = math_max_f32(vout1x0, vmin);
      c[0 * m + 0] = vout0x0;
      c[0 * m + 1] = vout1x0;
      c += 1 * m;
      j -= 1;
    }
    if XNN_UNLIKELY(j != 0) {
      do {
        uint32_t nnz = *nnzmap++;
        float vacc0 = *w++;
        float vacc1 = vacc0;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float va0 = a[0];
            const float va1 = a[1];
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const float vb = *w++;
            vacc0 += va0 * vb;
            vacc1 += va1 * vb;
          } while (--nnz != 0);
        }
        float vout0 = math_min_f32(vacc0, vmax);
        float vout1 = math_min_f32(vacc1, vmax);
        vout0 = math_max_f32(vout0, vmin);
        vout1 = math_max_f32(vout1, vmin);
        c[0] = vout0;
        c[1] = vout1;
        c += m;
        j -= 1;
      } while (j != 0);
    }
    c -= m * n;
    c += 2;
    a += 2;
    i -= 2;
  }
  if XNN_UNLIKELY(i != 0) {
    if (i & 1) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t j = n;
      while (j >= 1) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float va0 = a[0];
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const float vb0 = *w++;
            vacc0x0 += va0 * vb0;
          } while (--nnz != 0);
        }
        float vout0x0 = math_min_f32(vacc0x0, vmax);
        vout0x0 = math_max_f32(vout0x0, vmin);
        c[0 * m + 0] = vout0x0;
        c += 1 * m;
        j -= 1;
      }
      if XNN_UNLIKELY(j != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float va0 = a[0];
              a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
              const float vb = *w++;
              vacc0 += va0 * vb;
            } while (--nnz != 0);
          }
          float vout0 = math_min_f32(vacc0, vmax);
          vout0 = math_max_f32(vout0, vmin);
          c[0] = vout0;
          c += m;
          j -= 1;
        } while (j != 0);
      }
      c -= m * n;
      c += 1;
      a += 1;
    }
  }
}
