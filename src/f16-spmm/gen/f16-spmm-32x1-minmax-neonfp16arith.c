// Auto-generated file. Do not edit!
//   Template: src/f16-spmm/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/prefetch.h"
#include "xnnpack/spmm.h"


void xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith(
    size_t mc,
    size_t nc,
    const void* input,
    const void* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    void* output,
    size_t output_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(uint16_t) == 0);
  assert(nc != 0);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  #if XNN_ARCH_ARM64
    const uint16x8x2_t vminmax = vld2q_dup_u16(&params->fp16arith.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vminmax.val[0]);
    const float16x8_t vmax = vreinterpretq_f16_u16(vminmax.val[1]);
  #else
    // vld2_dup is to work around aarch32 clang bug with vld1q_dup
    const uint16x4x2_t vminmax = vld2_dup_u16(&params->fp16arith.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
    const float16x8_t vmax = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
  #endif

  size_t output_decrement = output_stride * nc - 32 * sizeof(uint16_t);
  while XNN_LIKELY(mc >= 32 * sizeof(uint16_t)) {
    const uint16_t* w = (const uint16_t*) weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      float16x8_t vacc01234567 = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
      float16x8_t vacc89ABCDEF = vacc01234567;
      float16x8_t vaccGHIJKLMN = vacc01234567;
      float16x8_t vaccOPQRSTUV = vacc01234567;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(i));
          const float16x8_t va89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i + 8));
          const float16x8_t vaGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i + 16));
          const float16x8_t vaOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i + 24));
          i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
          xnn_prefetch_to_l1(i + 32);
          const float16x8_t vw = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
          xnn_prefetch_to_l1(w + 64);
          vacc01234567 = vfmaq_f16(vacc01234567, va01234567, vw);
          vacc89ABCDEF = vfmaq_f16(vacc89ABCDEF, va89ABCDEF, vw);
          vaccGHIJKLMN = vfmaq_f16(vaccGHIJKLMN, vaGHIJKLMN, vw);
          vaccOPQRSTUV = vfmaq_f16(vaccOPQRSTUV, vaOPQRSTUV, vw);
        } while (--nnz != 0);
      }
      float16x8_t vout01234567 = vminq_f16(vacc01234567, vmax);
      float16x8_t vout89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);
      float16x8_t voutGHIJKLMN = vminq_f16(vaccGHIJKLMN, vmax);
      float16x8_t voutOPQRSTUV = vminq_f16(vaccOPQRSTUV, vmax);
      vout01234567 = vmaxq_f16(vout01234567, vmin);
      vout89ABCDEF = vmaxq_f16(vout89ABCDEF, vmin);
      voutGHIJKLMN = vmaxq_f16(voutGHIJKLMN, vmin);
      voutOPQRSTUV = vmaxq_f16(voutOPQRSTUV, vmin);
      vst1q_u16(o, vreinterpretq_u16_f16(vout01234567));
      vst1q_u16(o + 8, vreinterpretq_u16_f16(vout89ABCDEF));
      vst1q_u16(o + 16, vreinterpretq_u16_f16(voutGHIJKLMN));
      vst1q_u16(o + 24, vreinterpretq_u16_f16(voutOPQRSTUV));
      o = (uint16_t*) ((uintptr_t) o + output_stride);
    } while (--n != 0);
    o = (uint16_t*) ((uintptr_t) o - output_decrement);
    i += 32;
    mc -= 32 * sizeof(uint16_t);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 16 * sizeof(uint16_t);
    if (mc & (16 * sizeof(uint16_t))) {
      const uint16_t* w = (const uint16_t*) weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float16x8_t vacc01234567 = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
        float16x8_t vacc89ABCDEF = vacc01234567;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(i));
            const float16x8_t va89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i + 8));
            i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
            const float16x8_t vw = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
            vacc01234567 = vfmaq_f16(vacc01234567, va01234567, vw);
            vacc89ABCDEF = vfmaq_f16(vacc89ABCDEF, va89ABCDEF, vw);
          } while (--nnz != 0);
        }
        float16x8_t vout01234567 = vminq_f16(vacc01234567, vmax);
        float16x8_t vout89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);
        vout01234567 = vmaxq_f16(vout01234567, vmin);
        vout89ABCDEF = vmaxq_f16(vout89ABCDEF, vmin);
        vst1q_u16(o, vreinterpretq_u16_f16(vout01234567));
        vst1q_u16(o + 8, vreinterpretq_u16_f16(vout89ABCDEF));
        o = (uint16_t*) ((uintptr_t) o + output_stride);
      } while (--n != 0);
      o = (uint16_t*) ((uintptr_t) o - output_decrement);
      i += 16;
    }
    output_decrement += 8 * sizeof(uint16_t);
    if (mc & (8 * sizeof(uint16_t))) {
      const uint16_t* w = (const uint16_t*) weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float16x8_t vacc01234567 = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(i));
            i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
            const float16x8_t vw = vreinterpretq_f16_u16(vld1q_dup_u16(w)); w += 1;
            vacc01234567 = vfmaq_f16(vacc01234567, va01234567, vw);
          } while (--nnz != 0);
        }
        float16x8_t vout01234567 = vminq_f16(vacc01234567, vmax);
        vout01234567 = vmaxq_f16(vout01234567, vmin);
        vst1q_u16(o, vreinterpretq_u16_f16(vout01234567));
        o = (uint16_t*) ((uintptr_t) o + output_stride);
      } while (--n != 0);
      o = (uint16_t*) ((uintptr_t) o - output_decrement);
      i += 8;
    }
    output_decrement += 4 * sizeof(uint16_t);
    if (mc & (4 * sizeof(uint16_t))) {
      const uint16_t* w = (const uint16_t*) weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float16x4_t vacc0123 = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float16x4_t va0123 = vreinterpret_f16_u16(vld1_u16(i));
            i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
            const float16x4_t vw = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
            vacc0123 = vfma_f16(vacc0123, va0123, vw);
          } while (--nnz != 0);
        }
        float16x4_t vout0123 = vmin_f16(vacc0123, vget_low_f16(vmax));
        vout0123 = vmax_f16(vout0123, vget_low_f16(vmin));
        vst1_u16(o, vreinterpret_u16_f16(vout0123));
        o = (uint16_t*) ((uintptr_t) o + output_stride);
      } while (--n != 0);
      o = (uint16_t*) ((uintptr_t) o - output_decrement);
      i += 4;
    }
    output_decrement += 2 * sizeof(uint16_t);
    if (mc & (2 * sizeof(uint16_t))) {
      const uint16_t* w = (const uint16_t*) weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float16x4_t vacc01 = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float16x4_t va01 = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
            i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
            const float16x4_t vw = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
            vacc01 = vfma_f16(vacc01, va01, vw);
          } while (--nnz != 0);
        }
        float16x4_t vout01 = vmin_f16(vacc01, vget_low_f16(vmax));
        vout01 = vmax_f16(vout01, vget_low_f16(vmin));
        vst1_lane_u32((void*) o, vreinterpret_u32_f16(vout01), 0);
        o = (uint16_t*) ((uintptr_t) o + output_stride);
      } while (--n != 0);
      o = (uint16_t*) ((uintptr_t) o - output_decrement);
      i += 2;
    }
    output_decrement += 1 * sizeof(uint16_t);
    if (mc & (1 * sizeof(uint16_t))) {
      const uint16_t* w = (const uint16_t*) weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float16x4_t vacc0 = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float16x4_t va0 = vreinterpret_f16_u16(vld1_dup_u16(i));
            i = (const uint16_t*) ((uintptr_t) i + (uintptr_t) diff);
            const float16x4_t vw = vreinterpret_f16_u16(vld1_dup_u16(w)); w += 1;
            vacc0 = vfma_f16(vacc0, va0, vw);
          } while (--nnz != 0);
        }
        float16x4_t vout0 = vmin_f16(vacc0, vget_low_f16(vmax));
        vout0 = vmax_f16(vout0, vget_low_f16(vmin));
        vst1_lane_u16(o, vreinterpret_u16_f16(vout0), 0);
        o = (uint16_t*) ((uintptr_t) o + output_stride);
      } while (--n != 0);
      o = (uint16_t*) ((uintptr_t) o - output_decrement);
      i += 1;
    }
  }
}
