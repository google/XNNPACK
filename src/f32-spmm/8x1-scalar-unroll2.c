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


void xnn_f32_spmm_ukernel_8x1__scalar_unroll2(
    uint32_t m,
    uint32_t n,
    const float*restrict a,
    const float*restrict weights,
    const int32_t*restrict widx_dmap,
    const uint32_t*restrict nidx_nnzmap,
    float*restrict c,
    const union xnn_f32_output_params params[restrict static 1])
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
    do {
      uint32_t nnz = *nnzmap++;
      float vacc0x0 = *w++;
      float vacc0x1 = 0.0f;
      float vacc1x0 = vacc0x0;
      float vacc1x1 = 0.0f;
      float vacc2x0 = vacc0x0;
      float vacc2x1 = 0.0f;
      float vacc3x0 = vacc0x0;
      float vacc3x1 = 0.0f;
      float vacc4x0 = vacc0x0;
      float vacc4x1 = 0.0f;
      float vacc5x0 = vacc0x0;
      float vacc5x1 = 0.0f;
      float vacc6x0 = vacc0x0;
      float vacc6x1 = 0.0f;
      float vacc7x0 = vacc0x0;
      float vacc7x1 = 0.0f;
      for (; nnz >= 2; nnz -= 2) {
        const intptr_t diff0 = dmap[0];
        const intptr_t diff1 = dmap[1];
        dmap += 2;
        const float va0x0 = a[0];
        const float va1x0 = a[1];
        const float va2x0 = a[2];
        const float va3x0 = a[3];
        const float va4x0 = a[4];
        const float va5x0 = a[5];
        const float va6x0 = a[6];
        const float va7x0 = a[7];
        a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff0);
        const float vb0 = *w++;
        vacc0x0 += va0x0 * vb0;
        vacc1x0 += va1x0 * vb0;
        vacc2x0 += va2x0 * vb0;
        vacc3x0 += va3x0 * vb0;
        vacc4x0 += va4x0 * vb0;
        vacc5x0 += va5x0 * vb0;
        vacc6x0 += va6x0 * vb0;
        vacc7x0 += va7x0 * vb0;
        const float va0x1 = a[0];
        const float va1x1 = a[1];
        const float va2x1 = a[2];
        const float va3x1 = a[3];
        const float va4x1 = a[4];
        const float va5x1 = a[5];
        const float va6x1 = a[6];
        const float va7x1 = a[7];
        a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff1);
        const float vb1 = *w++;
        vacc0x1 += va0x1 * vb1;
        vacc1x1 += va1x1 * vb1;
        vacc2x1 += va2x1 * vb1;
        vacc3x1 += va3x1 * vb1;
        vacc4x1 += va4x1 * vb1;
        vacc5x1 += va5x1 * vb1;
        vacc6x1 += va6x1 * vb1;
        vacc7x1 += va7x1 * vb1;
      }
      float vacc0 = vacc0x0;
      float vacc1 = vacc1x0;
      float vacc2 = vacc2x0;
      float vacc3 = vacc3x0;
      float vacc4 = vacc4x0;
      float vacc5 = vacc5x0;
      float vacc6 = vacc6x0;
      float vacc7 = vacc7x0;
      vacc0 += vacc0x1;
      vacc1 += vacc1x1;
      vacc2 += vacc2x1;
      vacc3 += vacc3x1;
      vacc4 += vacc4x1;
      vacc5 += vacc5x1;
      vacc6 += vacc6x1;
      vacc7 += vacc7x1;
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
        float vacc0x0 = *w++;
        float vacc0x1 = 0.0f;
        float vacc1x0 = vacc0x0;
        float vacc1x1 = 0.0f;
        float vacc2x0 = vacc0x0;
        float vacc2x1 = 0.0f;
        float vacc3x0 = vacc0x0;
        float vacc3x1 = 0.0f;
        for (; nnz >= 2; nnz -= 2) {
          const intptr_t diff0 = dmap[0];
          const intptr_t diff1 = dmap[1];
          dmap += 2;
          const float va0x0 = a[0];
          const float va1x0 = a[1];
          const float va2x0 = a[2];
          const float va3x0 = a[3];
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff0);
          const float vb0 = *w++;
          vacc0x0 += va0x0 * vb0;
          vacc1x0 += va1x0 * vb0;
          vacc2x0 += va2x0 * vb0;
          vacc3x0 += va3x0 * vb0;
          const float va0x1 = a[0];
          const float va1x1 = a[1];
          const float va2x1 = a[2];
          const float va3x1 = a[3];
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff1);
          const float vb1 = *w++;
          vacc0x1 += va0x1 * vb1;
          vacc1x1 += va1x1 * vb1;
          vacc2x1 += va2x1 * vb1;
          vacc3x1 += va3x1 * vb1;
        }
        float vacc0 = vacc0x0;
        float vacc1 = vacc1x0;
        float vacc2 = vacc2x0;
        float vacc3 = vacc3x0;
        vacc0 += vacc0x1;
        vacc1 += vacc1x1;
        vacc2 += vacc2x1;
        vacc3 += vacc3x1;
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
        float vacc0x0 = *w++;
        float vacc0x1 = 0.0f;
        float vacc1x0 = vacc0x0;
        float vacc1x1 = 0.0f;
        for (; nnz >= 2; nnz -= 2) {
          const intptr_t diff0 = dmap[0];
          const intptr_t diff1 = dmap[1];
          dmap += 2;
          const float va0x0 = a[0];
          const float va1x0 = a[1];
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff0);
          const float vb0 = *w++;
          vacc0x0 += va0x0 * vb0;
          vacc1x0 += va1x0 * vb0;
          const float va0x1 = a[0];
          const float va1x1 = a[1];
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff1);
          const float vb1 = *w++;
          vacc0x1 += va0x1 * vb1;
          vacc1x1 += va1x1 * vb1;
        }
        float vacc0 = vacc0x0;
        float vacc1 = vacc1x0;
        vacc0 += vacc0x1;
        vacc1 += vacc1x1;
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
        float vacc0x0 = *w++;
        float vacc0x1 = 0.0f;
        for (; nnz >= 2; nnz -= 2) {
          const intptr_t diff0 = dmap[0];
          const intptr_t diff1 = dmap[1];
          dmap += 2;
          const float va0x0 = a[0];
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff0);
          const float vb0 = *w++;
          vacc0x0 += va0x0 * vb0;
          const float va0x1 = a[0];
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff1);
          const float vb1 = *w++;
          vacc0x1 += va0x1 * vb1;
        }
        float vacc0 = vacc0x0;
        vacc0 += vacc0x1;
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
      } while (--j != 0);
      c -= m * n;
      c += 1;
      a += 1;
    }
  }
}
