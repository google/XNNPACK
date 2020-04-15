// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/spmm.h>


void xnn_f32_spmm_minmax_ukernel_16x1__neonfma_unroll2(
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

  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
  size_t i = m;
  while XNN_LIKELY(i >= 16) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t j = n;
    do {
      uint32_t nnz = *nnzmap++;
      float32x4_t vacc0123x0 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc0123x1 = vmovq_n_f32(0.0f);
      float32x4_t vacc4567x0 = vacc0123x0;
      float32x4_t vacc4567x1 = vmovq_n_f32(0.0f);
      float32x4_t vacc89ABx0 = vacc0123x0;
      float32x4_t vacc89ABx1 = vmovq_n_f32(0.0f);
      float32x4_t vaccCDEFx0 = vacc0123x0;
      float32x4_t vaccCDEFx1 = vmovq_n_f32(0.0f);
      for (; nnz >= 2; nnz -= 2) {
        const intptr_t diff0 = dmap[0];
        const intptr_t diff1 = dmap[1];
        dmap += 2;
        const float32x4_t va0123x0 = vld1q_f32(a);
        const float32x4_t va4567x0 = vld1q_f32(a + 4);
        const float32x4_t va89ABx0 = vld1q_f32(a + 8);
        const float32x4_t vaCDEFx0 = vld1q_f32(a + 12);
        __builtin_prefetch(a + 16);
        a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff0);
        const float32x4_t vb0 = vld1q_dup_f32(w); w += 1;
        vacc0123x0 = vfmaq_f32(vacc0123x0, va0123x0, vb0);
        vacc4567x0 = vfmaq_f32(vacc4567x0, va4567x0, vb0);
        vacc89ABx0 = vfmaq_f32(vacc89ABx0, va89ABx0, vb0);
        vaccCDEFx0 = vfmaq_f32(vaccCDEFx0, vaCDEFx0, vb0);
        const float32x4_t va0123x1 = vld1q_f32(a);
        const float32x4_t va4567x1 = vld1q_f32(a + 4);
        const float32x4_t va89ABx1 = vld1q_f32(a + 8);
        const float32x4_t vaCDEFx1 = vld1q_f32(a + 12);
        __builtin_prefetch(a + 16);
        a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff1);
        const float32x4_t vb1 = vld1q_dup_f32(w); w += 1;
        vacc0123x1 = vfmaq_f32(vacc0123x1, va0123x1, vb1);
        vacc4567x1 = vfmaq_f32(vacc4567x1, va4567x1, vb1);
        vacc89ABx1 = vfmaq_f32(vacc89ABx1, va89ABx1, vb1);
        vaccCDEFx1 = vfmaq_f32(vaccCDEFx1, vaCDEFx1, vb1);
      }
      float32x4_t vacc0123 = vacc0123x0;
      float32x4_t vacc4567 = vacc4567x0;
      float32x4_t vacc89AB = vacc89ABx0;
      float32x4_t vaccCDEF = vaccCDEFx0;
      vacc0123 = vaddq_f32(vacc0123, vacc0123x1);
      vacc4567 = vaddq_f32(vacc4567, vacc4567x1);
      vacc89AB = vaddq_f32(vacc89AB, vacc89ABx1);
      vaccCDEF = vaddq_f32(vaccCDEF, vaccCDEFx1);
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float32x4_t va0123 = vld1q_f32(a);
          const float32x4_t va4567 = vld1q_f32(a + 4);
          const float32x4_t va89AB = vld1q_f32(a + 8);
          const float32x4_t vaCDEF = vld1q_f32(a + 12);
          __builtin_prefetch(a + 16);
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
          const float32x4_t vb = vld1q_dup_f32(w); w += 1;
          vacc0123 = vfmaq_f32(vacc0123, va0123, vb);
          vacc4567 = vfmaq_f32(vacc4567, va4567, vb);
          vacc89AB = vfmaq_f32(vacc89AB, va89AB, vb);
          vaccCDEF = vfmaq_f32(vaccCDEF, vaCDEF, vb);
        } while (--nnz != 0);
      }
      float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
      float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
      float32x4_t vout89AB = vminq_f32(vacc89AB, vmax);
      float32x4_t voutCDEF = vminq_f32(vaccCDEF, vmax);
      vout0123 = vmaxq_f32(vout0123, vmin);
      vout4567 = vmaxq_f32(vout4567, vmin);
      vout89AB = vmaxq_f32(vout89AB, vmin);
      voutCDEF = vmaxq_f32(voutCDEF, vmin);
      vst1q_f32(c, vout0123);
      vst1q_f32(c + 4, vout4567);
      vst1q_f32(c + 8, vout89AB);
      vst1q_f32(c + 12, voutCDEF);
      c += m;
    } while (--j != 0);
    c -= m * n;
    c += 16;
    a += 16;
    i -= 16;
  }
  if XNN_UNLIKELY(i != 0) {
    if (i & 8) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t j = n;
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567 = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t va0123 = vld1q_f32(a);
            const float32x4_t va4567 = vld1q_f32(a + 4);
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const float32x4_t vb = vld1q_dup_f32(w); w += 1;
            vacc0123 = vfmaq_f32(vacc0123, va0123, vb);
            vacc4567 = vfmaq_f32(vacc4567, va4567, vb);
          } while (--nnz != 0);
        }
        float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
        float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
        vout0123 = vmaxq_f32(vout0123, vmin);
        vout4567 = vmaxq_f32(vout4567, vmin);
        vst1q_f32(c, vout0123);
        vst1q_f32(c + 4, vout4567);
        c += m;
      } while (--j != 0);
      c -= m * n;
      c += 8;
      a += 8;
    }
    if (i & 4) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t j = n;
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t va0123 = vld1q_f32(a);
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const float32x4_t vb = vld1q_dup_f32(w); w += 1;
            vacc0123 = vfmaq_f32(vacc0123, va0123, vb);
          } while (--nnz != 0);
        }
        float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
        vout0123 = vmaxq_f32(vout0123, vmin);
        vst1q_f32(c, vout0123);
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
        float32x2_t vacc01 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t va01 = vld1_f32(a);
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const float32x2_t vb = vld1_dup_f32(w); w += 1;
            vacc01 = vfma_f32(vacc01, va01, vb);
          } while (--nnz != 0);
        }
        float32x2_t vout01 = vmin_f32(vacc01, vget_low_f32(vmax));
        vout01 = vmax_f32(vout01, vget_low_f32(vmin));
        vst1_f32(c, vout01);
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
        float32x2_t vacc0 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t va0 = vld1_dup_f32(a);
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const float32x2_t vb = vld1_dup_f32(w); w += 1;
            vacc0 = vfma_f32(vacc0, va0, vb);
          } while (--nnz != 0);
        }
        float32x2_t vout0 = vmin_f32(vacc0, vget_low_f32(vmax));
        vout0 = vmax_f32(vout0, vget_low_f32(vmin));
        vst1_lane_f32(c, vout0, 0);
        c += m;
      } while (--j != 0);
      c -= m * n;
      c += 1;
      a += 1;
    }
  }
}
