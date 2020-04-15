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


void xnn_f32_spmm_minmax_ukernel_8x4__scalar(
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
  while (i >= 8) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t j = n;
    while (j >= 4) {
      uint32_t nnz = *nnzmap++;
      float vacc0x0 = *w++;
      float vacc1x0 = vacc0x0;
      float vacc2x0 = vacc0x0;
      float vacc3x0 = vacc0x0;
      float vacc4x0 = vacc0x0;
      float vacc5x0 = vacc0x0;
      float vacc6x0 = vacc0x0;
      float vacc7x0 = vacc0x0;
      float vacc0x1 = *w++;
      float vacc1x1 = vacc0x1;
      float vacc2x1 = vacc0x1;
      float vacc3x1 = vacc0x1;
      float vacc4x1 = vacc0x1;
      float vacc5x1 = vacc0x1;
      float vacc6x1 = vacc0x1;
      float vacc7x1 = vacc0x1;
      float vacc0x2 = *w++;
      float vacc1x2 = vacc0x2;
      float vacc2x2 = vacc0x2;
      float vacc3x2 = vacc0x2;
      float vacc4x2 = vacc0x2;
      float vacc5x2 = vacc0x2;
      float vacc6x2 = vacc0x2;
      float vacc7x2 = vacc0x2;
      float vacc0x3 = *w++;
      float vacc1x3 = vacc0x3;
      float vacc2x3 = vacc0x3;
      float vacc3x3 = vacc0x3;
      float vacc4x3 = vacc0x3;
      float vacc5x3 = vacc0x3;
      float vacc6x3 = vacc0x3;
      float vacc7x3 = vacc0x3;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float va0 = a[0];
          const float va1 = a[1];
          const float va2 = a[2];
          const float va3 = a[3];
          const float va4 = a[4];
          const float va5 = a[5];
          const float va6 = a[6];
          const float va7 = a[7];
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
          const float vb0 = *w++;
          const float vb1 = *w++;
          const float vb2 = *w++;
          const float vb3 = *w++;
          vacc0x0 += va0 * vb0;
          vacc1x0 += va1 * vb0;
          vacc2x0 += va2 * vb0;
          vacc3x0 += va3 * vb0;
          vacc4x0 += va4 * vb0;
          vacc5x0 += va5 * vb0;
          vacc6x0 += va6 * vb0;
          vacc7x0 += va7 * vb0;
          vacc0x1 += va0 * vb1;
          vacc1x1 += va1 * vb1;
          vacc2x1 += va2 * vb1;
          vacc3x1 += va3 * vb1;
          vacc4x1 += va4 * vb1;
          vacc5x1 += va5 * vb1;
          vacc6x1 += va6 * vb1;
          vacc7x1 += va7 * vb1;
          vacc0x2 += va0 * vb2;
          vacc1x2 += va1 * vb2;
          vacc2x2 += va2 * vb2;
          vacc3x2 += va3 * vb2;
          vacc4x2 += va4 * vb2;
          vacc5x2 += va5 * vb2;
          vacc6x2 += va6 * vb2;
          vacc7x2 += va7 * vb2;
          vacc0x3 += va0 * vb3;
          vacc1x3 += va1 * vb3;
          vacc2x3 += va2 * vb3;
          vacc3x3 += va3 * vb3;
          vacc4x3 += va4 * vb3;
          vacc5x3 += va5 * vb3;
          vacc6x3 += va6 * vb3;
          vacc7x3 += va7 * vb3;
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
      c[0 * m + 0] = vout0x0;
      c[0 * m + 1] = vout1x0;
      c[0 * m + 2] = vout2x0;
      c[0 * m + 3] = vout3x0;
      c[0 * m + 4] = vout4x0;
      c[0 * m + 5] = vout5x0;
      c[0 * m + 6] = vout6x0;
      c[0 * m + 7] = vout7x0;
      c[1 * m + 0] = vout0x1;
      c[1 * m + 1] = vout1x1;
      c[1 * m + 2] = vout2x1;
      c[1 * m + 3] = vout3x1;
      c[1 * m + 4] = vout4x1;
      c[1 * m + 5] = vout5x1;
      c[1 * m + 6] = vout6x1;
      c[1 * m + 7] = vout7x1;
      c[2 * m + 0] = vout0x2;
      c[2 * m + 1] = vout1x2;
      c[2 * m + 2] = vout2x2;
      c[2 * m + 3] = vout3x2;
      c[2 * m + 4] = vout4x2;
      c[2 * m + 5] = vout5x2;
      c[2 * m + 6] = vout6x2;
      c[2 * m + 7] = vout7x2;
      c[3 * m + 0] = vout0x3;
      c[3 * m + 1] = vout1x3;
      c[3 * m + 2] = vout2x3;
      c[3 * m + 3] = vout3x3;
      c[3 * m + 4] = vout4x3;
      c[3 * m + 5] = vout5x3;
      c[3 * m + 6] = vout6x3;
      c[3 * m + 7] = vout7x3;
      c += 4 * m;
      j -= 4;
    }
    if XNN_UNLIKELY(j != 0) {
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
            const float va0 = a[0];
            const float va1 = a[1];
            const float va2 = a[2];
            const float va3 = a[3];
            const float va4 = a[4];
            const float va5 = a[5];
            const float va6 = a[6];
            const float va7 = a[7];
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const float vb = *w++;
            vacc0 += va0 * vb;
            vacc1 += va1 * vb;
            vacc2 += va2 * vb;
            vacc3 += va3 * vb;
            vacc4 += va4 * vb;
            vacc5 += va5 * vb;
            vacc6 += va6 * vb;
            vacc7 += va7 * vb;
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
        c[0] = vout0;
        c[1] = vout1;
        c[2] = vout2;
        c[3] = vout3;
        c[4] = vout4;
        c[5] = vout5;
        c[6] = vout6;
        c[7] = vout7;
        c += m;
        j -= 1;
      } while (j != 0);
    }
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
      while (j >= 4) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc1x0 = vacc0x0;
        float vacc2x0 = vacc0x0;
        float vacc3x0 = vacc0x0;
        float vacc0x1 = *w++;
        float vacc1x1 = vacc0x1;
        float vacc2x1 = vacc0x1;
        float vacc3x1 = vacc0x1;
        float vacc0x2 = *w++;
        float vacc1x2 = vacc0x2;
        float vacc2x2 = vacc0x2;
        float vacc3x2 = vacc0x2;
        float vacc0x3 = *w++;
        float vacc1x3 = vacc0x3;
        float vacc2x3 = vacc0x3;
        float vacc3x3 = vacc0x3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float va0 = a[0];
            const float va1 = a[1];
            const float va2 = a[2];
            const float va3 = a[3];
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const float vb0 = *w++;
            const float vb1 = *w++;
            const float vb2 = *w++;
            const float vb3 = *w++;
            vacc0x0 += va0 * vb0;
            vacc1x0 += va1 * vb0;
            vacc2x0 += va2 * vb0;
            vacc3x0 += va3 * vb0;
            vacc0x1 += va0 * vb1;
            vacc1x1 += va1 * vb1;
            vacc2x1 += va2 * vb1;
            vacc3x1 += va3 * vb1;
            vacc0x2 += va0 * vb2;
            vacc1x2 += va1 * vb2;
            vacc2x2 += va2 * vb2;
            vacc3x2 += va3 * vb2;
            vacc0x3 += va0 * vb3;
            vacc1x3 += va1 * vb3;
            vacc2x3 += va2 * vb3;
            vacc3x3 += va3 * vb3;
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
        c[0 * m + 0] = vout0x0;
        c[0 * m + 1] = vout1x0;
        c[0 * m + 2] = vout2x0;
        c[0 * m + 3] = vout3x0;
        c[1 * m + 0] = vout0x1;
        c[1 * m + 1] = vout1x1;
        c[1 * m + 2] = vout2x1;
        c[1 * m + 3] = vout3x1;
        c[2 * m + 0] = vout0x2;
        c[2 * m + 1] = vout1x2;
        c[2 * m + 2] = vout2x2;
        c[2 * m + 3] = vout3x2;
        c[3 * m + 0] = vout0x3;
        c[3 * m + 1] = vout1x3;
        c[3 * m + 2] = vout2x3;
        c[3 * m + 3] = vout3x3;
        c += 4 * m;
        j -= 4;
      }
      if XNN_UNLIKELY(j != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float vacc0 = *w++;
          float vacc1 = vacc0;
          float vacc2 = vacc0;
          float vacc3 = vacc0;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float va0 = a[0];
              const float va1 = a[1];
              const float va2 = a[2];
              const float va3 = a[3];
              a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
              const float vb = *w++;
              vacc0 += va0 * vb;
              vacc1 += va1 * vb;
              vacc2 += va2 * vb;
              vacc3 += va3 * vb;
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
          c[0] = vout0;
          c[1] = vout1;
          c[2] = vout2;
          c[3] = vout3;
          c += m;
          j -= 1;
        } while (j != 0);
      }
      c -= m * n;
      c += 4;
      a += 4;
    }
    if (i & 2) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t j = n;
      while (j >= 4) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc1x0 = vacc0x0;
        float vacc0x1 = *w++;
        float vacc1x1 = vacc0x1;
        float vacc0x2 = *w++;
        float vacc1x2 = vacc0x2;
        float vacc0x3 = *w++;
        float vacc1x3 = vacc0x3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float va0 = a[0];
            const float va1 = a[1];
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const float vb0 = *w++;
            const float vb1 = *w++;
            const float vb2 = *w++;
            const float vb3 = *w++;
            vacc0x0 += va0 * vb0;
            vacc1x0 += va1 * vb0;
            vacc0x1 += va0 * vb1;
            vacc1x1 += va1 * vb1;
            vacc0x2 += va0 * vb2;
            vacc1x2 += va1 * vb2;
            vacc0x3 += va0 * vb3;
            vacc1x3 += va1 * vb3;
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
        c[0 * m + 0] = vout0x0;
        c[0 * m + 1] = vout1x0;
        c[1 * m + 0] = vout0x1;
        c[1 * m + 1] = vout1x1;
        c[2 * m + 0] = vout0x2;
        c[2 * m + 1] = vout1x2;
        c[3 * m + 0] = vout0x3;
        c[3 * m + 1] = vout1x3;
        c += 4 * m;
        j -= 4;
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
    }
    if (i & 1) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t j = n;
      while (j >= 4) {
        uint32_t nnz = *nnzmap++;
        float vacc0x0 = *w++;
        float vacc0x1 = *w++;
        float vacc0x2 = *w++;
        float vacc0x3 = *w++;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float va0 = a[0];
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const float vb0 = *w++;
            const float vb1 = *w++;
            const float vb2 = *w++;
            const float vb3 = *w++;
            vacc0x0 += va0 * vb0;
            vacc0x1 += va0 * vb1;
            vacc0x2 += va0 * vb2;
            vacc0x3 += va0 * vb3;
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
        c[0 * m + 0] = vout0x0;
        c[1 * m + 0] = vout0x1;
        c[2 * m + 0] = vout0x2;
        c[3 * m + 0] = vout0x3;
        c += 4 * m;
        j -= 4;
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
