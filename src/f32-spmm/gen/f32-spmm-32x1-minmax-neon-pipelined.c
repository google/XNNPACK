// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/neon-pipelined.c.in
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


void xnn_f32_spmm_minmax_ukernel_32x1__neon_pipelined(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  #if XNN_ARCH_ARM64
    const float32x4x2_t vminmax = vld2q_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vminmax.val[0];
    const float32x4_t vmax = vminmax.val[1];
  #else
    const float32x2x2_t vminmax = vld2_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vcombine_f32(vminmax.val[0], vminmax.val[0]);
    const float32x4_t vmax = vcombine_f32(vminmax.val[1], vminmax.val[1]);
  #endif

  size_t output_decrement = output_stride * nc - 32 * sizeof(float);
  while XNN_LIKELY(mc >= 32 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    float32x4_t vw = vld1q_dup_f32(w); w += 1;
    intptr_t diff = *dmap++;
    float32x4_t vi0123 = vld1q_f32(input);
    float32x4_t vi4567 = vld1q_f32(input + 4);
    float32x4_t vi89AB = vld1q_f32(input + 8);
    float32x4_t viCDEF = vld1q_f32(input + 12);
    float32x4_t viGHIJ = vld1q_f32(input + 16);
    float32x4_t viKLMN = vld1q_f32(input + 20);
    float32x4_t viOPQR = vld1q_f32(input + 24);
    float32x4_t viSTUV = vld1q_f32(input + 28);
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      float32x4_t vacc0123 = vw;
      float32x4_t vacc4567 = vw;
      float32x4_t vacc89AB = vw;
      float32x4_t vaccCDEF = vw;
      float32x4_t vaccGHIJ = vw;
      float32x4_t vaccKLMN = vw;
      float32x4_t vaccOPQR = vw;
      float32x4_t vaccSTUV = vw;
      vw = vld1q_dup_f32(w); w += 1;
      if XNN_LIKELY(nnz != 0) {
        do {
          vacc0123 = vmlaq_f32(vacc0123, vi0123, vw);
          vacc4567 = vmlaq_f32(vacc4567, vi4567, vw);
          vacc89AB = vmlaq_f32(vacc89AB, vi89AB, vw);
          vaccCDEF = vmlaq_f32(vaccCDEF, viCDEF, vw);
          vaccGHIJ = vmlaq_f32(vaccGHIJ, viGHIJ, vw);
          vaccKLMN = vmlaq_f32(vaccKLMN, viKLMN, vw);
          vaccOPQR = vmlaq_f32(vaccOPQR, viOPQR, vw);
          vaccSTUV = vmlaq_f32(vaccSTUV, viSTUV, vw);
          input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
          xnn_prefetch_to_l1(input + 16);
          xnn_prefetch_to_l1(input + 32);
          diff = *dmap++;
          vw = vld1q_dup_f32(w); w += 1;
          xnn_prefetch_to_l1(w + 32);
          vi0123 = vld1q_f32(input);
          vi4567 = vld1q_f32(input + 4);
          vi89AB = vld1q_f32(input + 8);
          viCDEF = vld1q_f32(input + 12);
          viGHIJ = vld1q_f32(input + 16);
          viKLMN = vld1q_f32(input + 20);
          viOPQR = vld1q_f32(input + 24);
          viSTUV = vld1q_f32(input + 28);
        } while (--nnz != 0);
      }
      float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
      float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
      float32x4_t vout89AB = vminq_f32(vacc89AB, vmax);
      float32x4_t voutCDEF = vminq_f32(vaccCDEF, vmax);
      float32x4_t voutGHIJ = vminq_f32(vaccGHIJ, vmax);
      float32x4_t voutKLMN = vminq_f32(vaccKLMN, vmax);
      float32x4_t voutOPQR = vminq_f32(vaccOPQR, vmax);
      float32x4_t voutSTUV = vminq_f32(vaccSTUV, vmax);
      vout0123 = vmaxq_f32(vout0123, vmin);
      vout4567 = vmaxq_f32(vout4567, vmin);
      vout89AB = vmaxq_f32(vout89AB, vmin);
      voutCDEF = vmaxq_f32(voutCDEF, vmin);
      voutGHIJ = vmaxq_f32(voutGHIJ, vmin);
      voutKLMN = vmaxq_f32(voutKLMN, vmin);
      voutOPQR = vmaxq_f32(voutOPQR, vmin);
      voutSTUV = vmaxq_f32(voutSTUV, vmin);
      vst1q_f32(output, vout0123);
      vst1q_f32(output + 4, vout4567);
      vst1q_f32(output + 8, vout89AB);
      vst1q_f32(output + 12, voutCDEF);
      vst1q_f32(output + 16, voutGHIJ);
      vst1q_f32(output + 20, voutKLMN);
      vst1q_f32(output + 24, voutOPQR);
      vst1q_f32(output + 28, voutSTUV);
      output = (float*) ((uintptr_t) output + output_stride);
    } while (--n != 0);
    output = (float*) ((uintptr_t) output - output_decrement);
    input += 32;
    mc -= 32 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567 = vacc0123;
        float32x4_t vacc89AB = vacc0123;
        float32x4_t vaccCDEF = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x4_t vb = vld1q_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc0123 = vmlaq_f32(vacc0123, vi0123, vb);
            vacc4567 = vmlaq_f32(vacc4567, vi4567, vb);
            vacc89AB = vmlaq_f32(vacc89AB, vi89AB, vb);
            vaccCDEF = vmlaq_f32(vaccCDEF, viCDEF, vb);
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
        vst1q_f32(output, vout0123);
        vst1q_f32(output + 4, vout4567);
        vst1q_f32(output + 8, vout89AB);
        vst1q_f32(output + 12, voutCDEF);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567 = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x4_t vb = vld1q_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc0123 = vmlaq_f32(vacc0123, vi0123, vb);
            vacc4567 = vmlaq_f32(vacc4567, vi4567, vb);
          } while (--nnz != 0);
        }
        float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
        float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
        vout0123 = vmaxq_f32(vout0123, vmin);
        vout4567 = vmaxq_f32(vout4567, vmin);
        vst1q_f32(output, vout0123);
        vst1q_f32(output + 4, vout4567);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 8;
    }
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x4_t vb = vld1q_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc0123 = vmlaq_f32(vacc0123, vi0123, vb);
          } while (--nnz != 0);
        }
        float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
        vout0123 = vmaxq_f32(vout0123, vmin);
        vst1q_f32(output, vout0123);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc01 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi01 = vld1_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x2_t vb = vld1_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc01 = vmla_f32(vacc01, vi01, vb);
          } while (--nnz != 0);
        }
        float32x2_t vout01 = vmin_f32(vacc01, vget_low_f32(vmax));
        vout01 = vmax_f32(vout01, vget_low_f32(vmin));
        vst1_f32(output, vout01);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc0 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi0 = vld1_dup_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x2_t vb = vld1_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc0 = vmla_f32(vacc0, vi0, vb);
          } while (--nnz != 0);
        }
        float32x2_t vout0 = vmin_f32(vacc0, vget_low_f32(vmax));
        vout0 = vmax_f32(vout0, vget_low_f32(vmin));
        vst1_lane_f32(output, vout0, 0);
        output = (float*) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}
