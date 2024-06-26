// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/neon-blocked.c.in
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


void xnn_f32_spmm_minmax_ukernel_32x2__aarch64_neonfma(
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
    size_t n = nc;
    while (n >= 2) {
      uint32_t nnz = *nnzmap++;
      float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n0 = vacc0123n0;
      float32x4_t vacc89ABn0 = vacc0123n0;
      float32x4_t vaccCDEFn0 = vacc0123n0;
      float32x4_t vaccGHIJn0 = vacc0123n0;
      float32x4_t vaccKLMNn0 = vacc0123n0;
      float32x4_t vaccOPQRn0 = vacc0123n0;
      float32x4_t vaccSTUVn0 = vacc0123n0;
      float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n1 = vacc0123n1;
      float32x4_t vacc89ABn1 = vacc0123n1;
      float32x4_t vaccCDEFn1 = vacc0123n1;
      float32x4_t vaccGHIJn1 = vacc0123n1;
      float32x4_t vaccKLMNn1 = vacc0123n1;
      float32x4_t vaccOPQRn1 = vacc0123n1;
      float32x4_t vaccSTUVn1 = vacc0123n1;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float32x4_t vi0123 = vld1q_f32(input);
          const float32x4_t vi4567 = vld1q_f32(input + 4);
          const float32x4_t vi89AB = vld1q_f32(input + 8);
          const float32x4_t viCDEF = vld1q_f32(input + 12);
          const float32x4_t viGHIJ = vld1q_f32(input + 16);
          const float32x4_t viKLMN = vld1q_f32(input + 20);
          const float32x4_t viOPQR = vld1q_f32(input + 24);
          const float32x4_t viSTUV = vld1q_f32(input + 28);
          input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
          xnn_prefetch_to_l1(input + 16);
          xnn_prefetch_to_l1(input + 32);
          const float32x2_t vw = vld1_f32(w); w += 2;
          xnn_prefetch_to_l1(w + 32);
          vacc0123n0 = vfmaq_lane_f32(vacc0123n0, vi0123, vw, 0);
          vacc4567n0 = vfmaq_lane_f32(vacc4567n0, vi4567, vw, 0);
          vacc89ABn0 = vfmaq_lane_f32(vacc89ABn0, vi89AB, vw, 0);
          vaccCDEFn0 = vfmaq_lane_f32(vaccCDEFn0, viCDEF, vw, 0);
          vaccGHIJn0 = vfmaq_lane_f32(vaccGHIJn0, viGHIJ, vw, 0);
          vaccKLMNn0 = vfmaq_lane_f32(vaccKLMNn0, viKLMN, vw, 0);
          vaccOPQRn0 = vfmaq_lane_f32(vaccOPQRn0, viOPQR, vw, 0);
          vaccSTUVn0 = vfmaq_lane_f32(vaccSTUVn0, viSTUV, vw, 0);
          vacc0123n1 = vfmaq_lane_f32(vacc0123n1, vi0123, vw, 1);
          vacc4567n1 = vfmaq_lane_f32(vacc4567n1, vi4567, vw, 1);
          vacc89ABn1 = vfmaq_lane_f32(vacc89ABn1, vi89AB, vw, 1);
          vaccCDEFn1 = vfmaq_lane_f32(vaccCDEFn1, viCDEF, vw, 1);
          vaccGHIJn1 = vfmaq_lane_f32(vaccGHIJn1, viGHIJ, vw, 1);
          vaccKLMNn1 = vfmaq_lane_f32(vaccKLMNn1, viKLMN, vw, 1);
          vaccOPQRn1 = vfmaq_lane_f32(vaccOPQRn1, viOPQR, vw, 1);
          vaccSTUVn1 = vfmaq_lane_f32(vaccSTUVn1, viSTUV, vw, 1);
        } while (--nnz != 0);
      }
      float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
      float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
      float32x4_t vout89ABn0 = vminq_f32(vacc89ABn0, vmax);
      float32x4_t voutCDEFn0 = vminq_f32(vaccCDEFn0, vmax);
      float32x4_t voutGHIJn0 = vminq_f32(vaccGHIJn0, vmax);
      float32x4_t voutKLMNn0 = vminq_f32(vaccKLMNn0, vmax);
      float32x4_t voutOPQRn0 = vminq_f32(vaccOPQRn0, vmax);
      float32x4_t voutSTUVn0 = vminq_f32(vaccSTUVn0, vmax);
      float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
      float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
      float32x4_t vout89ABn1 = vminq_f32(vacc89ABn1, vmax);
      float32x4_t voutCDEFn1 = vminq_f32(vaccCDEFn1, vmax);
      float32x4_t voutGHIJn1 = vminq_f32(vaccGHIJn1, vmax);
      float32x4_t voutKLMNn1 = vminq_f32(vaccKLMNn1, vmax);
      float32x4_t voutOPQRn1 = vminq_f32(vaccOPQRn1, vmax);
      float32x4_t voutSTUVn1 = vminq_f32(vaccSTUVn1, vmax);

      vout0123n0 = vmaxq_f32(vout0123n0, vmin);
      vout4567n0 = vmaxq_f32(vout4567n0, vmin);
      vout89ABn0 = vmaxq_f32(vout89ABn0, vmin);
      voutCDEFn0 = vmaxq_f32(voutCDEFn0, vmin);
      voutGHIJn0 = vmaxq_f32(voutGHIJn0, vmin);
      voutKLMNn0 = vmaxq_f32(voutKLMNn0, vmin);
      voutOPQRn0 = vmaxq_f32(voutOPQRn0, vmin);
      voutSTUVn0 = vmaxq_f32(voutSTUVn0, vmin);
      vout0123n1 = vmaxq_f32(vout0123n1, vmin);
      vout4567n1 = vmaxq_f32(vout4567n1, vmin);
      vout89ABn1 = vmaxq_f32(vout89ABn1, vmin);
      voutCDEFn1 = vmaxq_f32(voutCDEFn1, vmin);
      voutGHIJn1 = vmaxq_f32(voutGHIJn1, vmin);
      voutKLMNn1 = vmaxq_f32(voutKLMNn1, vmin);
      voutOPQRn1 = vmaxq_f32(voutOPQRn1, vmin);
      voutSTUVn1 = vmaxq_f32(voutSTUVn1, vmin);

      vst1q_f32(output + 0, vout0123n0);
      vst1q_f32(output + 4, vout4567n0);
      vst1q_f32(output + 8, vout89ABn0);
      vst1q_f32(output + 12, voutCDEFn0);
      vst1q_f32(output + 16, voutGHIJn0);
      vst1q_f32(output + 20, voutKLMNn0);
      vst1q_f32(output + 24, voutOPQRn0);
      vst1q_f32(output + 28, voutSTUVn0);
      output = (float*) ((uintptr_t) output + output_stride);
      vst1q_f32(output + 0, vout0123n1);
      vst1q_f32(output + 4, vout4567n1);
      vst1q_f32(output + 8, vout89ABn1);
      vst1q_f32(output + 12, voutCDEFn1);
      vst1q_f32(output + 16, voutGHIJn1);
      vst1q_f32(output + 20, voutKLMNn1);
      vst1q_f32(output + 24, voutOPQRn1);
      vst1q_f32(output + 28, voutSTUVn1);
      output = (float*) ((uintptr_t) output + output_stride);
      n -= 2;
    }

    // clean up loop, fall back to nr=1
    if XNN_UNLIKELY(n != 0) {
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567 = vacc0123;
        float32x4_t vacc89AB = vacc0123;
        float32x4_t vaccCDEF = vacc0123;
        float32x4_t vaccGHIJ = vacc0123;
        float32x4_t vaccKLMN = vacc0123;
        float32x4_t vaccOPQR = vacc0123;
        float32x4_t vaccSTUV = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            const float32x4_t viGHIJ = vld1q_f32(input + 16);
            const float32x4_t viKLMN = vld1q_f32(input + 20);
            const float32x4_t viOPQR = vld1q_f32(input + 24);
            const float32x4_t viSTUV = vld1q_f32(input + 28);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x4_t vw = vld1q_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
            vacc4567 = vfmaq_f32(vacc4567, vi4567, vw);
            vacc89AB = vfmaq_f32(vacc89AB, vi89AB, vw);
            vaccCDEF = vfmaq_f32(vaccCDEF, viCDEF, vw);
            vaccGHIJ = vfmaq_f32(vaccGHIJ, viGHIJ, vw);
            vaccKLMN = vfmaq_f32(vaccKLMN, viKLMN, vw);
            vaccOPQR = vfmaq_f32(vaccOPQR, viOPQR, vw);
            vaccSTUV = vfmaq_f32(vaccSTUV, viSTUV, vw);
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

        vst1q_f32(output + 0, vout0123);
        vst1q_f32(output + 4, vout4567);
        vst1q_f32(output + 8, vout89AB);
        vst1q_f32(output + 12, voutCDEF);
        vst1q_f32(output + 16, voutGHIJ);
        vst1q_f32(output + 20, voutKLMN);
        vst1q_f32(output + 24, voutOPQR);
        vst1q_f32(output + 28, voutSTUV);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 1;
      } while (n != 0);
    }
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
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n0 = vacc0123n0;
        float32x4_t vacc89ABn0 = vacc0123n0;
        float32x4_t vaccCDEFn0 = vacc0123n0;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n1 = vacc0123n1;
        float32x4_t vacc89ABn1 = vacc0123n1;
        float32x4_t vaccCDEFn1 = vacc0123n1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123n0 = vfmaq_lane_f32(vacc0123n0, vi0123, vw, 0);
            vacc4567n0 = vfmaq_lane_f32(vacc4567n0, vi4567, vw, 0);
            vacc89ABn0 = vfmaq_lane_f32(vacc89ABn0, vi89AB, vw, 0);
            vaccCDEFn0 = vfmaq_lane_f32(vaccCDEFn0, viCDEF, vw, 0);
            vacc0123n1 = vfmaq_lane_f32(vacc0123n1, vi0123, vw, 1);
            vacc4567n1 = vfmaq_lane_f32(vacc4567n1, vi4567, vw, 1);
            vacc89ABn1 = vfmaq_lane_f32(vacc89ABn1, vi89AB, vw, 1);
            vaccCDEFn1 = vfmaq_lane_f32(vaccCDEFn1, viCDEF, vw, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
        float32x4_t vout89ABn0 = vminq_f32(vacc89ABn0, vmax);
        float32x4_t voutCDEFn0 = vminq_f32(vaccCDEFn0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
        float32x4_t vout89ABn1 = vminq_f32(vacc89ABn1, vmax);
        float32x4_t voutCDEFn1 = vminq_f32(vaccCDEFn1, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);
        vout89ABn0 = vmaxq_f32(vout89ABn0, vmin);
        voutCDEFn0 = vmaxq_f32(voutCDEFn0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout4567n1 = vmaxq_f32(vout4567n1, vmin);
        vout89ABn1 = vmaxq_f32(vout89ABn1, vmin);
        voutCDEFn1 = vmaxq_f32(voutCDEFn1, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        vst1q_f32(output + 8, vout89ABn0);
        vst1q_f32(output + 12, voutCDEFn0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        vst1q_f32(output + 4, vout4567n1);
        vst1q_f32(output + 8, vout89ABn1);
        vst1q_f32(output + 12, voutCDEFn1);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
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
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
              vacc4567 = vfmaq_f32(vacc4567, vi4567, vw);
              vacc89AB = vfmaq_f32(vacc89AB, vi89AB, vw);
              vaccCDEF = vfmaq_f32(vaccCDEF, viCDEF, vw);
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

          vst1q_f32(output + 0, vout0123);
          vst1q_f32(output + 4, vout4567);
          vst1q_f32(output + 8, vout89AB);
          vst1q_f32(output + 12, voutCDEF);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n0 = vacc0123n0;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n1 = vacc0123n1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123n0 = vfmaq_lane_f32(vacc0123n0, vi0123, vw, 0);
            vacc4567n0 = vfmaq_lane_f32(vacc4567n0, vi4567, vw, 0);
            vacc0123n1 = vfmaq_lane_f32(vacc0123n1, vi0123, vw, 1);
            vacc4567n1 = vfmaq_lane_f32(vacc4567n1, vi4567, vw, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout4567n1 = vmaxq_f32(vout4567n1, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        vst1q_f32(output + 4, vout4567n1);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
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
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
              vacc4567 = vfmaq_f32(vacc4567, vi4567, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
          float32x4_t vout4567 = vminq_f32(vacc4567, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);
          vout4567 = vmaxq_f32(vout4567, vmin);

          vst1q_f32(output + 0, vout0123);
          vst1q_f32(output + 4, vout4567);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 8;
    }
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123n0 = vfmaq_lane_f32(vacc0123n0, vi0123, vw, 0);
            vacc0123n1 = vfmaq_lane_f32(vacc0123n1, vi0123, vw, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);

        vst1q_f32(output + 0, vout0123n0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);

          vst1q_f32(output + 0, vout0123);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc01n0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01n1 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi01 = vld1_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc01n0 = vfma_lane_f32(vacc01n0, vi01, vw, 0);
            vacc01n1 = vfma_lane_f32(vacc01n1, vi01, vw, 1);
          } while (--nnz != 0);
        }
        float32x2_t vout01n0 = vmin_f32(vacc01n0, vget_low_f32(vmax));
        float32x2_t vout01n1 = vmin_f32(vacc01n1, vget_low_f32(vmax));

        vout01n0 = vmax_f32(vout01n0, vget_low_f32(vmin));
        vout01n1 = vmax_f32(vout01n1, vget_low_f32(vmin));

        vst1_f32(output + 0, vout01n0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_f32(output + 0, vout01n1);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc01 = vld1_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x2_t vi01 = vld1_f32(input);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x2_t vw = vld1_dup_f32(w); w += 1;
              vacc01 = vfma_f32(vacc01, vi01, vw);
            } while (--nnz != 0);
          }
          float32x2_t vout01 = vmin_f32(vacc01, vget_low_f32(vmax));
          vout01 = vmax_f32(vout01, vget_low_f32(vmin));

          vst1_f32(output, vout01);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc0n0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0n1 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi0 = vld1_dup_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0n0 = vfma_lane_f32(vacc0n0, vi0, vw, 0);
            vacc0n1 = vfma_lane_f32(vacc0n1, vi0, vw, 1);
          } while (--nnz != 0);
        }
        float32x2_t vout0n0 = vmin_f32(vacc0n0, vget_low_f32(vmax));
        float32x2_t vout0n1 = vmin_f32(vacc0n1, vget_low_f32(vmax));

        vout0n0 = vmax_f32(vout0n0, vget_low_f32(vmin));
        vout0n1 = vmax_f32(vout0n1, vget_low_f32(vmin));

        vst1_lane_f32(output + 0, vout0n0, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_lane_f32(output + 0, vout0n1, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc0 = vld1_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x2_t vi0 = vld1_dup_f32(input);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x2_t vw = vld1_dup_f32(w); w += 1;
              vacc0 = vfma_f32(vacc0, vi0, vw);
            } while (--nnz != 0);
          }
          float32x2_t vout0 = vmin_f32(vacc0, vget_low_f32(vmax));
          vout0 = vmax_f32(vout0, vget_low_f32(vmin));

          vst1_lane_f32(output, vout0, 1);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 1;
    }
    }
}
