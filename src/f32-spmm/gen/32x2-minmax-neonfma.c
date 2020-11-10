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

#include <xnnpack/spmm.h>


void xnn_f32_spmm_minmax_ukernel_32x2__neonfma(
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

  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
  size_t n = batch_size;
  while XNN_LIKELY(n >= 32) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t c = output_channels;
    while (c >= 2) {
      uint32_t nnz = *nnzmap++;
      float32x4_t vacc0123c0 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567c0 = vacc0123c0;
      float32x4_t vacc89ABc0 = vacc0123c0;
      float32x4_t vaccCDEFc0 = vacc0123c0;
      float32x4_t vaccGHIJc0 = vacc0123c0;
      float32x4_t vaccKLMNc0 = vacc0123c0;
      float32x4_t vaccOPQRc0 = vacc0123c0;
      float32x4_t vaccSTUVc0 = vacc0123c0;
      float32x4_t vacc0123c1 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567c1 = vacc0123c1;
      float32x4_t vacc89ABc1 = vacc0123c1;
      float32x4_t vaccCDEFc1 = vacc0123c1;
      float32x4_t vaccGHIJc1 = vacc0123c1;
      float32x4_t vaccKLMNc1 = vacc0123c1;
      float32x4_t vaccOPQRc1 = vacc0123c1;
      float32x4_t vaccSTUVc1 = vacc0123c1;
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
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          __builtin_prefetch(input + 16);
          __builtin_prefetch(input + 32);
          const float32x2_t vw = vld1_f32(w); w += 2;
          __builtin_prefetch(w + 32);
          vacc0123c0 = vfmaq_lane_f32(vacc0123c0, vi0123, vw, 0);
          vacc4567c0 = vfmaq_lane_f32(vacc4567c0, vi4567, vw, 0);
          vacc89ABc0 = vfmaq_lane_f32(vacc89ABc0, vi89AB, vw, 0);
          vaccCDEFc0 = vfmaq_lane_f32(vaccCDEFc0, viCDEF, vw, 0);
          vaccGHIJc0 = vfmaq_lane_f32(vaccGHIJc0, viGHIJ, vw, 0);
          vaccKLMNc0 = vfmaq_lane_f32(vaccKLMNc0, viKLMN, vw, 0);
          vaccOPQRc0 = vfmaq_lane_f32(vaccOPQRc0, viOPQR, vw, 0);
          vaccSTUVc0 = vfmaq_lane_f32(vaccSTUVc0, viSTUV, vw, 0);
          vacc0123c1 = vfmaq_lane_f32(vacc0123c1, vi0123, vw, 1);
          vacc4567c1 = vfmaq_lane_f32(vacc4567c1, vi4567, vw, 1);
          vacc89ABc1 = vfmaq_lane_f32(vacc89ABc1, vi89AB, vw, 1);
          vaccCDEFc1 = vfmaq_lane_f32(vaccCDEFc1, viCDEF, vw, 1);
          vaccGHIJc1 = vfmaq_lane_f32(vaccGHIJc1, viGHIJ, vw, 1);
          vaccKLMNc1 = vfmaq_lane_f32(vaccKLMNc1, viKLMN, vw, 1);
          vaccOPQRc1 = vfmaq_lane_f32(vaccOPQRc1, viOPQR, vw, 1);
          vaccSTUVc1 = vfmaq_lane_f32(vaccSTUVc1, viSTUV, vw, 1);
        } while (--nnz != 0);
      }
      float32x4_t vout0123c0 = vminq_f32(vacc0123c0, vmax);
      float32x4_t vout4567c0 = vminq_f32(vacc4567c0, vmax);
      float32x4_t vout89ABc0 = vminq_f32(vacc89ABc0, vmax);
      float32x4_t voutCDEFc0 = vminq_f32(vaccCDEFc0, vmax);
      float32x4_t voutGHIJc0 = vminq_f32(vaccGHIJc0, vmax);
      float32x4_t voutKLMNc0 = vminq_f32(vaccKLMNc0, vmax);
      float32x4_t voutOPQRc0 = vminq_f32(vaccOPQRc0, vmax);
      float32x4_t voutSTUVc0 = vminq_f32(vaccSTUVc0, vmax);
      float32x4_t vout0123c1 = vminq_f32(vacc0123c1, vmax);
      float32x4_t vout4567c1 = vminq_f32(vacc4567c1, vmax);
      float32x4_t vout89ABc1 = vminq_f32(vacc89ABc1, vmax);
      float32x4_t voutCDEFc1 = vminq_f32(vaccCDEFc1, vmax);
      float32x4_t voutGHIJc1 = vminq_f32(vaccGHIJc1, vmax);
      float32x4_t voutKLMNc1 = vminq_f32(vaccKLMNc1, vmax);
      float32x4_t voutOPQRc1 = vminq_f32(vaccOPQRc1, vmax);
      float32x4_t voutSTUVc1 = vminq_f32(vaccSTUVc1, vmax);

      vout0123c0 = vmaxq_f32(vout0123c0, vmin);
      vout4567c0 = vmaxq_f32(vout4567c0, vmin);
      vout89ABc0 = vmaxq_f32(vout89ABc0, vmin);
      voutCDEFc0 = vmaxq_f32(voutCDEFc0, vmin);
      voutGHIJc0 = vmaxq_f32(voutGHIJc0, vmin);
      voutKLMNc0 = vmaxq_f32(voutKLMNc0, vmin);
      voutOPQRc0 = vmaxq_f32(voutOPQRc0, vmin);
      voutSTUVc0 = vmaxq_f32(voutSTUVc0, vmin);
      vout0123c1 = vmaxq_f32(vout0123c1, vmin);
      vout4567c1 = vmaxq_f32(vout4567c1, vmin);
      vout89ABc1 = vmaxq_f32(vout89ABc1, vmin);
      voutCDEFc1 = vmaxq_f32(voutCDEFc1, vmin);
      voutGHIJc1 = vmaxq_f32(voutGHIJc1, vmin);
      voutKLMNc1 = vmaxq_f32(voutKLMNc1, vmin);
      voutOPQRc1 = vmaxq_f32(voutOPQRc1, vmin);
      voutSTUVc1 = vmaxq_f32(voutSTUVc1, vmin);

      vst1q_f32(output + 0 * batch_size + 0, vout0123c0);
      vst1q_f32(output + 0 * batch_size + 4, vout4567c0);
      vst1q_f32(output + 0 * batch_size + 8, vout89ABc0);
      vst1q_f32(output + 0 * batch_size + 12, voutCDEFc0);
      vst1q_f32(output + 0 * batch_size + 16, voutGHIJc0);
      vst1q_f32(output + 0 * batch_size + 20, voutKLMNc0);
      vst1q_f32(output + 0 * batch_size + 24, voutOPQRc0);
      vst1q_f32(output + 0 * batch_size + 28, voutSTUVc0);
      vst1q_f32(output + 1 * batch_size + 0, vout0123c1);
      vst1q_f32(output + 1 * batch_size + 4, vout4567c1);
      vst1q_f32(output + 1 * batch_size + 8, vout89ABc1);
      vst1q_f32(output + 1 * batch_size + 12, voutCDEFc1);
      vst1q_f32(output + 1 * batch_size + 16, voutGHIJc1);
      vst1q_f32(output + 1 * batch_size + 20, voutKLMNc1);
      vst1q_f32(output + 1 * batch_size + 24, voutOPQRc1);
      vst1q_f32(output + 1 * batch_size + 28, voutSTUVc1);
      output += 2 * batch_size;
      c -= 2;
    }

    // clean up loop, fall back to nr=1
    if XNN_UNLIKELY(c != 0) {
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
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            __builtin_prefetch(input + 16);
            __builtin_prefetch(input + 32);
            const float32x4_t vw = vld1q_dup_f32(w); w += 1;
            __builtin_prefetch(w + 32);
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
        output += batch_size;
        c -= 1;
      } while (c != 0);
    }
    output -= batch_size * output_channels;
    output += 32;
    input += 32;
    n -= 32;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 16) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
      while (c >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123c0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c0 = vacc0123c0;
        float32x4_t vacc89ABc0 = vacc0123c0;
        float32x4_t vaccCDEFc0 = vacc0123c0;
        float32x4_t vacc0123c1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c1 = vacc0123c1;
        float32x4_t vacc89ABc1 = vacc0123c1;
        float32x4_t vaccCDEFc1 = vacc0123c1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123c0 = vfmaq_lane_f32(vacc0123c0, vi0123, vw, 0);
            vacc4567c0 = vfmaq_lane_f32(vacc4567c0, vi4567, vw, 0);
            vacc89ABc0 = vfmaq_lane_f32(vacc89ABc0, vi89AB, vw, 0);
            vaccCDEFc0 = vfmaq_lane_f32(vaccCDEFc0, viCDEF, vw, 0);
            vacc0123c1 = vfmaq_lane_f32(vacc0123c1, vi0123, vw, 1);
            vacc4567c1 = vfmaq_lane_f32(vacc4567c1, vi4567, vw, 1);
            vacc89ABc1 = vfmaq_lane_f32(vacc89ABc1, vi89AB, vw, 1);
            vaccCDEFc1 = vfmaq_lane_f32(vaccCDEFc1, viCDEF, vw, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123c0 = vminq_f32(vacc0123c0, vmax);
        float32x4_t vout4567c0 = vminq_f32(vacc4567c0, vmax);
        float32x4_t vout89ABc0 = vminq_f32(vacc89ABc0, vmax);
        float32x4_t voutCDEFc0 = vminq_f32(vaccCDEFc0, vmax);
        float32x4_t vout0123c1 = vminq_f32(vacc0123c1, vmax);
        float32x4_t vout4567c1 = vminq_f32(vacc4567c1, vmax);
        float32x4_t vout89ABc1 = vminq_f32(vacc89ABc1, vmax);
        float32x4_t voutCDEFc1 = vminq_f32(vaccCDEFc1, vmax);

        vout0123c0 = vmaxq_f32(vout0123c0, vmin);
        vout4567c0 = vmaxq_f32(vout4567c0, vmin);
        vout89ABc0 = vmaxq_f32(vout89ABc0, vmin);
        voutCDEFc0 = vmaxq_f32(voutCDEFc0, vmin);
        vout0123c1 = vmaxq_f32(vout0123c1, vmin);
        vout4567c1 = vmaxq_f32(vout4567c1, vmin);
        vout89ABc1 = vmaxq_f32(vout89ABc1, vmin);
        voutCDEFc1 = vmaxq_f32(voutCDEFc1, vmin);

        vst1q_f32(output + 0 * batch_size + 0, vout0123c0);
        vst1q_f32(output + 0 * batch_size + 4, vout4567c0);
        vst1q_f32(output + 0 * batch_size + 8, vout89ABc0);
        vst1q_f32(output + 0 * batch_size + 12, voutCDEFc0);
        vst1q_f32(output + 1 * batch_size + 0, vout0123c1);
        vst1q_f32(output + 1 * batch_size + 4, vout4567c1);
        vst1q_f32(output + 1 * batch_size + 8, vout89ABc1);
        vst1q_f32(output + 1 * batch_size + 12, voutCDEFc1);
        output += 2 * batch_size;
        c -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(c != 0) {
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
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
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
          output += batch_size;
          c -= 1;
        } while (c != 0);
      }
      output -= batch_size * output_channels;
      output += 16;
      input += 16;
    }
    if (n & 8) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
      while (c >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123c0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c0 = vacc0123c0;
        float32x4_t vacc0123c1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c1 = vacc0123c1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123c0 = vfmaq_lane_f32(vacc0123c0, vi0123, vw, 0);
            vacc4567c0 = vfmaq_lane_f32(vacc4567c0, vi4567, vw, 0);
            vacc0123c1 = vfmaq_lane_f32(vacc0123c1, vi0123, vw, 1);
            vacc4567c1 = vfmaq_lane_f32(vacc4567c1, vi4567, vw, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123c0 = vminq_f32(vacc0123c0, vmax);
        float32x4_t vout4567c0 = vminq_f32(vacc4567c0, vmax);
        float32x4_t vout0123c1 = vminq_f32(vacc0123c1, vmax);
        float32x4_t vout4567c1 = vminq_f32(vacc4567c1, vmax);

        vout0123c0 = vmaxq_f32(vout0123c0, vmin);
        vout4567c0 = vmaxq_f32(vout4567c0, vmin);
        vout0123c1 = vmaxq_f32(vout0123c1, vmin);
        vout4567c1 = vmaxq_f32(vout4567c1, vmin);

        vst1q_f32(output + 0 * batch_size + 0, vout0123c0);
        vst1q_f32(output + 0 * batch_size + 4, vout4567c0);
        vst1q_f32(output + 1 * batch_size + 0, vout0123c1);
        vst1q_f32(output + 1 * batch_size + 4, vout4567c1);
        output += 2 * batch_size;
        c -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(c != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          float32x4_t vacc4567 = vacc0123;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              const float32x4_t vi4567 = vld1q_f32(input + 4);
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
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
          output += batch_size;
          c -= 1;
        } while (c != 0);
      }
      output -= batch_size * output_channels;
      output += 8;
      input += 8;
    }
    if (n & 4) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
      while (c >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123c0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123c1 = vld1q_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123c0 = vfmaq_lane_f32(vacc0123c0, vi0123, vw, 0);
            vacc0123c1 = vfmaq_lane_f32(vacc0123c1, vi0123, vw, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123c0 = vminq_f32(vacc0123c0, vmax);
        float32x4_t vout0123c1 = vminq_f32(vacc0123c1, vmax);

        vout0123c0 = vmaxq_f32(vout0123c0, vmin);
        vout0123c1 = vmaxq_f32(vout0123c1, vmin);

        vst1q_f32(output + 0 * batch_size + 0, vout0123c0);
        vst1q_f32(output + 1 * batch_size + 0, vout0123c1);
        output += 2 * batch_size;
        c -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(c != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);

          vst1q_f32(output + 0, vout0123);
          output += batch_size;
          c -= 1;
        } while (c != 0);
      }
      output -= batch_size * output_channels;
      output += 4;
      input += 4;
    }
    if (n & 2) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
      while (c >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc01c0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01c1 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi01 = vld1_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc01c0 = vfma_lane_f32(vacc01c0, vi01, vw, 0);
            vacc01c1 = vfma_lane_f32(vacc01c1, vi01, vw, 1);
          } while (--nnz != 0);
        }
        float32x2_t vout01c0 = vmin_f32(vacc01c0, vget_low_f32(vmax));
        float32x2_t vout01c1 = vmin_f32(vacc01c1, vget_low_f32(vmax));

        vout01c0 = vmax_f32(vout01c0, vget_low_f32(vmin));
        vout01c1 = vmax_f32(vout01c1, vget_low_f32(vmin));

        vst1_f32(output + 0 * batch_size + 0, vout01c0);
        vst1_f32(output + 1 * batch_size + 0, vout01c1);
        output += 2 * batch_size;
        c -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(c != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc01 = vld1_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x2_t vi01 = vld1_f32(input);
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float32x2_t vw = vld1_dup_f32(w); w += 1;
              vacc01 = vfma_f32(vacc01, vi01, vw);
            } while (--nnz != 0);
          }
          float32x2_t vout01 = vmin_f32(vacc01, vget_low_f32(vmax));
          vout01 = vmax_f32(vout01, vget_low_f32(vmin));

          vst1_f32(output, vout01);
          output += batch_size;
          c -= 1;
        } while (c != 0);
      }
      output -= batch_size * output_channels;
      output += 2;
      input += 2;
    }
    if (n & 1) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
      while (c >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc0c0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0c1 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi0 = vld1_dup_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0c0 = vfma_lane_f32(vacc0c0, vi0, vw, 0);
            vacc0c1 = vfma_lane_f32(vacc0c1, vi0, vw, 1);
          } while (--nnz != 0);
        }
        float32x2_t vout0c0 = vmin_f32(vacc0c0, vget_low_f32(vmax));
        float32x2_t vout0c1 = vmin_f32(vacc0c1, vget_low_f32(vmax));

        vout0c0 = vmax_f32(vout0c0, vget_low_f32(vmin));
        vout0c1 = vmax_f32(vout0c1, vget_low_f32(vmin));

        vst1_lane_f32(output + 0 * batch_size + 0, vout0c0, 0);
        vst1_lane_f32(output + 1 * batch_size + 0, vout0c1, 0);
        output += 2 * batch_size;
        c -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(c != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc0 = vld1_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x2_t vi0 = vld1_dup_f32(input);
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float32x2_t vw = vld1_dup_f32(w); w += 1;
              vacc0 = vfma_f32(vacc0, vi0, vw);
            } while (--nnz != 0);
          }
          float32x2_t vout0 = vmin_f32(vacc0, vget_low_f32(vmax));
          vout0 = vmax_f32(vout0, vget_low_f32(vmin));

          vst1_lane_f32(output, vout0, 1);
          output += batch_size;
          c -= 1;
        } while (c != 0);
      }
      output -= batch_size * output_channels;
      output += 1;
      input += 1;
    }
    }
}
