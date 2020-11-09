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


void xnn_f32_spmm_minmax_ukernel_32x4__neonfma(
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
    while (c >= 4) {
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
      float32x4_t vacc0123c2 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567c2 = vacc0123c2;
      float32x4_t vacc89ABc2 = vacc0123c2;
      float32x4_t vaccCDEFc2 = vacc0123c2;
      float32x4_t vaccGHIJc2 = vacc0123c2;
      float32x4_t vaccKLMNc2 = vacc0123c2;
      float32x4_t vaccOPQRc2 = vacc0123c2;
      float32x4_t vaccSTUVc2 = vacc0123c2;
      float32x4_t vacc0123c3 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567c3 = vacc0123c3;
      float32x4_t vacc89ABc3 = vacc0123c3;
      float32x4_t vaccCDEFc3 = vacc0123c3;
      float32x4_t vaccGHIJc3 = vacc0123c3;
      float32x4_t vaccKLMNc3 = vacc0123c3;
      float32x4_t vaccOPQRc3 = vacc0123c3;
      float32x4_t vaccSTUVc3 = vacc0123c3;
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
          const float32x4_t vw = vld1q_f32(w); w += 4;
          __builtin_prefetch(w + 16);
          vacc0123c0 = vfmaq_laneq_f32(vacc0123c0, vi0123, vw, 0);
          vacc4567c0 = vfmaq_laneq_f32(vacc4567c0, vi4567, vw, 0);
          vacc89ABc0 = vfmaq_laneq_f32(vacc89ABc0, vi89AB, vw, 0);
          vaccCDEFc0 = vfmaq_laneq_f32(vaccCDEFc0, viCDEF, vw, 0);
          vaccGHIJc0 = vfmaq_laneq_f32(vaccGHIJc0, viGHIJ, vw, 0);
          vaccKLMNc0 = vfmaq_laneq_f32(vaccKLMNc0, viKLMN, vw, 0);
          vaccOPQRc0 = vfmaq_laneq_f32(vaccOPQRc0, viOPQR, vw, 0);
          vaccSTUVc0 = vfmaq_laneq_f32(vaccSTUVc0, viSTUV, vw, 0);
          vacc0123c1 = vfmaq_laneq_f32(vacc0123c1, vi0123, vw, 1);
          vacc4567c1 = vfmaq_laneq_f32(vacc4567c1, vi4567, vw, 1);
          vacc89ABc1 = vfmaq_laneq_f32(vacc89ABc1, vi89AB, vw, 1);
          vaccCDEFc1 = vfmaq_laneq_f32(vaccCDEFc1, viCDEF, vw, 1);
          vaccGHIJc1 = vfmaq_laneq_f32(vaccGHIJc1, viGHIJ, vw, 1);
          vaccKLMNc1 = vfmaq_laneq_f32(vaccKLMNc1, viKLMN, vw, 1);
          vaccOPQRc1 = vfmaq_laneq_f32(vaccOPQRc1, viOPQR, vw, 1);
          vaccSTUVc1 = vfmaq_laneq_f32(vaccSTUVc1, viSTUV, vw, 1);
          vacc0123c2 = vfmaq_laneq_f32(vacc0123c2, vi0123, vw, 2);
          vacc4567c2 = vfmaq_laneq_f32(vacc4567c2, vi4567, vw, 2);
          vacc89ABc2 = vfmaq_laneq_f32(vacc89ABc2, vi89AB, vw, 2);
          vaccCDEFc2 = vfmaq_laneq_f32(vaccCDEFc2, viCDEF, vw, 2);
          vaccGHIJc2 = vfmaq_laneq_f32(vaccGHIJc2, viGHIJ, vw, 2);
          vaccKLMNc2 = vfmaq_laneq_f32(vaccKLMNc2, viKLMN, vw, 2);
          vaccOPQRc2 = vfmaq_laneq_f32(vaccOPQRc2, viOPQR, vw, 2);
          vaccSTUVc2 = vfmaq_laneq_f32(vaccSTUVc2, viSTUV, vw, 2);
          vacc0123c3 = vfmaq_laneq_f32(vacc0123c3, vi0123, vw, 3);
          vacc4567c3 = vfmaq_laneq_f32(vacc4567c3, vi4567, vw, 3);
          vacc89ABc3 = vfmaq_laneq_f32(vacc89ABc3, vi89AB, vw, 3);
          vaccCDEFc3 = vfmaq_laneq_f32(vaccCDEFc3, viCDEF, vw, 3);
          vaccGHIJc3 = vfmaq_laneq_f32(vaccGHIJc3, viGHIJ, vw, 3);
          vaccKLMNc3 = vfmaq_laneq_f32(vaccKLMNc3, viKLMN, vw, 3);
          vaccOPQRc3 = vfmaq_laneq_f32(vaccOPQRc3, viOPQR, vw, 3);
          vaccSTUVc3 = vfmaq_laneq_f32(vaccSTUVc3, viSTUV, vw, 3);
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
      float32x4_t vout0123c2 = vminq_f32(vacc0123c2, vmax);
      float32x4_t vout4567c2 = vminq_f32(vacc4567c2, vmax);
      float32x4_t vout89ABc2 = vminq_f32(vacc89ABc2, vmax);
      float32x4_t voutCDEFc2 = vminq_f32(vaccCDEFc2, vmax);
      float32x4_t voutGHIJc2 = vminq_f32(vaccGHIJc2, vmax);
      float32x4_t voutKLMNc2 = vminq_f32(vaccKLMNc2, vmax);
      float32x4_t voutOPQRc2 = vminq_f32(vaccOPQRc2, vmax);
      float32x4_t voutSTUVc2 = vminq_f32(vaccSTUVc2, vmax);
      float32x4_t vout0123c3 = vminq_f32(vacc0123c3, vmax);
      float32x4_t vout4567c3 = vminq_f32(vacc4567c3, vmax);
      float32x4_t vout89ABc3 = vminq_f32(vacc89ABc3, vmax);
      float32x4_t voutCDEFc3 = vminq_f32(vaccCDEFc3, vmax);
      float32x4_t voutGHIJc3 = vminq_f32(vaccGHIJc3, vmax);
      float32x4_t voutKLMNc3 = vminq_f32(vaccKLMNc3, vmax);
      float32x4_t voutOPQRc3 = vminq_f32(vaccOPQRc3, vmax);
      float32x4_t voutSTUVc3 = vminq_f32(vaccSTUVc3, vmax);

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
      vout0123c2 = vmaxq_f32(vout0123c2, vmin);
      vout4567c2 = vmaxq_f32(vout4567c2, vmin);
      vout89ABc2 = vmaxq_f32(vout89ABc2, vmin);
      voutCDEFc2 = vmaxq_f32(voutCDEFc2, vmin);
      voutGHIJc2 = vmaxq_f32(voutGHIJc2, vmin);
      voutKLMNc2 = vmaxq_f32(voutKLMNc2, vmin);
      voutOPQRc2 = vmaxq_f32(voutOPQRc2, vmin);
      voutSTUVc2 = vmaxq_f32(voutSTUVc2, vmin);
      vout0123c3 = vmaxq_f32(vout0123c3, vmin);
      vout4567c3 = vmaxq_f32(vout4567c3, vmin);
      vout89ABc3 = vmaxq_f32(vout89ABc3, vmin);
      voutCDEFc3 = vmaxq_f32(voutCDEFc3, vmin);
      voutGHIJc3 = vmaxq_f32(voutGHIJc3, vmin);
      voutKLMNc3 = vmaxq_f32(voutKLMNc3, vmin);
      voutOPQRc3 = vmaxq_f32(voutOPQRc3, vmin);
      voutSTUVc3 = vmaxq_f32(voutSTUVc3, vmin);

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
      vst1q_f32(output + 2 * batch_size + 0, vout0123c2);
      vst1q_f32(output + 2 * batch_size + 4, vout4567c2);
      vst1q_f32(output + 2 * batch_size + 8, vout89ABc2);
      vst1q_f32(output + 2 * batch_size + 12, voutCDEFc2);
      vst1q_f32(output + 2 * batch_size + 16, voutGHIJc2);
      vst1q_f32(output + 2 * batch_size + 20, voutKLMNc2);
      vst1q_f32(output + 2 * batch_size + 24, voutOPQRc2);
      vst1q_f32(output + 2 * batch_size + 28, voutSTUVc2);
      vst1q_f32(output + 3 * batch_size + 0, vout0123c3);
      vst1q_f32(output + 3 * batch_size + 4, vout4567c3);
      vst1q_f32(output + 3 * batch_size + 8, vout89ABc3);
      vst1q_f32(output + 3 * batch_size + 12, voutCDEFc3);
      vst1q_f32(output + 3 * batch_size + 16, voutGHIJc3);
      vst1q_f32(output + 3 * batch_size + 20, voutKLMNc3);
      vst1q_f32(output + 3 * batch_size + 24, voutOPQRc3);
      vst1q_f32(output + 3 * batch_size + 28, voutSTUVc3);
      output += 4 * batch_size;
      c -= 4;
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
            __builtin_prefetch(w + 16);
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
      while (c >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123c0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c0 = vacc0123c0;
        float32x4_t vacc89ABc0 = vacc0123c0;
        float32x4_t vaccCDEFc0 = vacc0123c0;
        float32x4_t vacc0123c1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c1 = vacc0123c1;
        float32x4_t vacc89ABc1 = vacc0123c1;
        float32x4_t vaccCDEFc1 = vacc0123c1;
        float32x4_t vacc0123c2 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c2 = vacc0123c2;
        float32x4_t vacc89ABc2 = vacc0123c2;
        float32x4_t vaccCDEFc2 = vacc0123c2;
        float32x4_t vacc0123c3 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c3 = vacc0123c3;
        float32x4_t vacc89ABc3 = vacc0123c3;
        float32x4_t vaccCDEFc3 = vacc0123c3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc0123c0 = vfmaq_laneq_f32(vacc0123c0, vi0123, vw, 0);
            vacc4567c0 = vfmaq_laneq_f32(vacc4567c0, vi4567, vw, 0);
            vacc89ABc0 = vfmaq_laneq_f32(vacc89ABc0, vi89AB, vw, 0);
            vaccCDEFc0 = vfmaq_laneq_f32(vaccCDEFc0, viCDEF, vw, 0);
            vacc0123c1 = vfmaq_laneq_f32(vacc0123c1, vi0123, vw, 1);
            vacc4567c1 = vfmaq_laneq_f32(vacc4567c1, vi4567, vw, 1);
            vacc89ABc1 = vfmaq_laneq_f32(vacc89ABc1, vi89AB, vw, 1);
            vaccCDEFc1 = vfmaq_laneq_f32(vaccCDEFc1, viCDEF, vw, 1);
            vacc0123c2 = vfmaq_laneq_f32(vacc0123c2, vi0123, vw, 2);
            vacc4567c2 = vfmaq_laneq_f32(vacc4567c2, vi4567, vw, 2);
            vacc89ABc2 = vfmaq_laneq_f32(vacc89ABc2, vi89AB, vw, 2);
            vaccCDEFc2 = vfmaq_laneq_f32(vaccCDEFc2, viCDEF, vw, 2);
            vacc0123c3 = vfmaq_laneq_f32(vacc0123c3, vi0123, vw, 3);
            vacc4567c3 = vfmaq_laneq_f32(vacc4567c3, vi4567, vw, 3);
            vacc89ABc3 = vfmaq_laneq_f32(vacc89ABc3, vi89AB, vw, 3);
            vaccCDEFc3 = vfmaq_laneq_f32(vaccCDEFc3, viCDEF, vw, 3);
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
        float32x4_t vout0123c2 = vminq_f32(vacc0123c2, vmax);
        float32x4_t vout4567c2 = vminq_f32(vacc4567c2, vmax);
        float32x4_t vout89ABc2 = vminq_f32(vacc89ABc2, vmax);
        float32x4_t voutCDEFc2 = vminq_f32(vaccCDEFc2, vmax);
        float32x4_t vout0123c3 = vminq_f32(vacc0123c3, vmax);
        float32x4_t vout4567c3 = vminq_f32(vacc4567c3, vmax);
        float32x4_t vout89ABc3 = vminq_f32(vacc89ABc3, vmax);
        float32x4_t voutCDEFc3 = vminq_f32(vaccCDEFc3, vmax);

        vout0123c0 = vmaxq_f32(vout0123c0, vmin);
        vout4567c0 = vmaxq_f32(vout4567c0, vmin);
        vout89ABc0 = vmaxq_f32(vout89ABc0, vmin);
        voutCDEFc0 = vmaxq_f32(voutCDEFc0, vmin);
        vout0123c1 = vmaxq_f32(vout0123c1, vmin);
        vout4567c1 = vmaxq_f32(vout4567c1, vmin);
        vout89ABc1 = vmaxq_f32(vout89ABc1, vmin);
        voutCDEFc1 = vmaxq_f32(voutCDEFc1, vmin);
        vout0123c2 = vmaxq_f32(vout0123c2, vmin);
        vout4567c2 = vmaxq_f32(vout4567c2, vmin);
        vout89ABc2 = vmaxq_f32(vout89ABc2, vmin);
        voutCDEFc2 = vmaxq_f32(voutCDEFc2, vmin);
        vout0123c3 = vmaxq_f32(vout0123c3, vmin);
        vout4567c3 = vmaxq_f32(vout4567c3, vmin);
        vout89ABc3 = vmaxq_f32(vout89ABc3, vmin);
        voutCDEFc3 = vmaxq_f32(voutCDEFc3, vmin);

        vst1q_f32(output + 0 * batch_size + 0, vout0123c0);
        vst1q_f32(output + 0 * batch_size + 4, vout4567c0);
        vst1q_f32(output + 0 * batch_size + 8, vout89ABc0);
        vst1q_f32(output + 0 * batch_size + 12, voutCDEFc0);
        vst1q_f32(output + 1 * batch_size + 0, vout0123c1);
        vst1q_f32(output + 1 * batch_size + 4, vout4567c1);
        vst1q_f32(output + 1 * batch_size + 8, vout89ABc1);
        vst1q_f32(output + 1 * batch_size + 12, voutCDEFc1);
        vst1q_f32(output + 2 * batch_size + 0, vout0123c2);
        vst1q_f32(output + 2 * batch_size + 4, vout4567c2);
        vst1q_f32(output + 2 * batch_size + 8, vout89ABc2);
        vst1q_f32(output + 2 * batch_size + 12, voutCDEFc2);
        vst1q_f32(output + 3 * batch_size + 0, vout0123c3);
        vst1q_f32(output + 3 * batch_size + 4, vout4567c3);
        vst1q_f32(output + 3 * batch_size + 8, vout89ABc3);
        vst1q_f32(output + 3 * batch_size + 12, voutCDEFc3);
        output += 4 * batch_size;
        c -= 4;
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
      while (c >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123c0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c0 = vacc0123c0;
        float32x4_t vacc0123c1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c1 = vacc0123c1;
        float32x4_t vacc0123c2 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c2 = vacc0123c2;
        float32x4_t vacc0123c3 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567c3 = vacc0123c3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc0123c0 = vfmaq_laneq_f32(vacc0123c0, vi0123, vw, 0);
            vacc4567c0 = vfmaq_laneq_f32(vacc4567c0, vi4567, vw, 0);
            vacc0123c1 = vfmaq_laneq_f32(vacc0123c1, vi0123, vw, 1);
            vacc4567c1 = vfmaq_laneq_f32(vacc4567c1, vi4567, vw, 1);
            vacc0123c2 = vfmaq_laneq_f32(vacc0123c2, vi0123, vw, 2);
            vacc4567c2 = vfmaq_laneq_f32(vacc4567c2, vi4567, vw, 2);
            vacc0123c3 = vfmaq_laneq_f32(vacc0123c3, vi0123, vw, 3);
            vacc4567c3 = vfmaq_laneq_f32(vacc4567c3, vi4567, vw, 3);
          } while (--nnz != 0);
        }
        float32x4_t vout0123c0 = vminq_f32(vacc0123c0, vmax);
        float32x4_t vout4567c0 = vminq_f32(vacc4567c0, vmax);
        float32x4_t vout0123c1 = vminq_f32(vacc0123c1, vmax);
        float32x4_t vout4567c1 = vminq_f32(vacc4567c1, vmax);
        float32x4_t vout0123c2 = vminq_f32(vacc0123c2, vmax);
        float32x4_t vout4567c2 = vminq_f32(vacc4567c2, vmax);
        float32x4_t vout0123c3 = vminq_f32(vacc0123c3, vmax);
        float32x4_t vout4567c3 = vminq_f32(vacc4567c3, vmax);

        vout0123c0 = vmaxq_f32(vout0123c0, vmin);
        vout4567c0 = vmaxq_f32(vout4567c0, vmin);
        vout0123c1 = vmaxq_f32(vout0123c1, vmin);
        vout4567c1 = vmaxq_f32(vout4567c1, vmin);
        vout0123c2 = vmaxq_f32(vout0123c2, vmin);
        vout4567c2 = vmaxq_f32(vout4567c2, vmin);
        vout0123c3 = vmaxq_f32(vout0123c3, vmin);
        vout4567c3 = vmaxq_f32(vout4567c3, vmin);

        vst1q_f32(output + 0 * batch_size + 0, vout0123c0);
        vst1q_f32(output + 0 * batch_size + 4, vout4567c0);
        vst1q_f32(output + 1 * batch_size + 0, vout0123c1);
        vst1q_f32(output + 1 * batch_size + 4, vout4567c1);
        vst1q_f32(output + 2 * batch_size + 0, vout0123c2);
        vst1q_f32(output + 2 * batch_size + 4, vout4567c2);
        vst1q_f32(output + 3 * batch_size + 0, vout0123c3);
        vst1q_f32(output + 3 * batch_size + 4, vout4567c3);
        output += 4 * batch_size;
        c -= 4;
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
      while (c >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123c0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123c1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123c2 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123c3 = vld1q_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc0123c0 = vfmaq_laneq_f32(vacc0123c0, vi0123, vw, 0);
            vacc0123c1 = vfmaq_laneq_f32(vacc0123c1, vi0123, vw, 1);
            vacc0123c2 = vfmaq_laneq_f32(vacc0123c2, vi0123, vw, 2);
            vacc0123c3 = vfmaq_laneq_f32(vacc0123c3, vi0123, vw, 3);
          } while (--nnz != 0);
        }
        float32x4_t vout0123c0 = vminq_f32(vacc0123c0, vmax);
        float32x4_t vout0123c1 = vminq_f32(vacc0123c1, vmax);
        float32x4_t vout0123c2 = vminq_f32(vacc0123c2, vmax);
        float32x4_t vout0123c3 = vminq_f32(vacc0123c3, vmax);

        vout0123c0 = vmaxq_f32(vout0123c0, vmin);
        vout0123c1 = vmaxq_f32(vout0123c1, vmin);
        vout0123c2 = vmaxq_f32(vout0123c2, vmin);
        vout0123c3 = vmaxq_f32(vout0123c3, vmin);

        vst1q_f32(output + 0 * batch_size + 0, vout0123c0);
        vst1q_f32(output + 1 * batch_size + 0, vout0123c1);
        vst1q_f32(output + 2 * batch_size + 0, vout0123c2);
        vst1q_f32(output + 3 * batch_size + 0, vout0123c3);
        output += 4 * batch_size;
        c -= 4;
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
      while (c >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc01c0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01c1 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01c2 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01c3 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi01 = vld1_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc01c0 = vfma_laneq_f32(vacc01c0, vi01, vw, 0);
            vacc01c1 = vfma_laneq_f32(vacc01c1, vi01, vw, 1);
            vacc01c2 = vfma_laneq_f32(vacc01c2, vi01, vw, 2);
            vacc01c3 = vfma_laneq_f32(vacc01c3, vi01, vw, 3);
          } while (--nnz != 0);
        }
        float32x2_t vout01c0 = vmin_f32(vacc01c0, vget_low_f32(vmax));
        float32x2_t vout01c1 = vmin_f32(vacc01c1, vget_low_f32(vmax));
        float32x2_t vout01c2 = vmin_f32(vacc01c2, vget_low_f32(vmax));
        float32x2_t vout01c3 = vmin_f32(vacc01c3, vget_low_f32(vmax));

        vout01c0 = vmax_f32(vout01c0, vget_low_f32(vmin));
        vout01c1 = vmax_f32(vout01c1, vget_low_f32(vmin));
        vout01c2 = vmax_f32(vout01c2, vget_low_f32(vmin));
        vout01c3 = vmax_f32(vout01c3, vget_low_f32(vmin));

        vst1_f32(output + 0 * batch_size + 0, vout01c0);
        vst1_f32(output + 1 * batch_size + 0, vout01c1);
        vst1_f32(output + 2 * batch_size + 0, vout01c2);
        vst1_f32(output + 3 * batch_size + 0, vout01c3);
        output += 4 * batch_size;
        c -= 4;
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
      while (c >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc0c0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0c1 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0c2 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0c3 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi0 = vld1_dup_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc0c0 = vfma_laneq_f32(vacc0c0, vi0, vw, 0);
            vacc0c1 = vfma_laneq_f32(vacc0c1, vi0, vw, 1);
            vacc0c2 = vfma_laneq_f32(vacc0c2, vi0, vw, 2);
            vacc0c3 = vfma_laneq_f32(vacc0c3, vi0, vw, 3);
          } while (--nnz != 0);
        }
        float32x2_t vout0c0 = vmin_f32(vacc0c0, vget_low_f32(vmax));
        float32x2_t vout0c1 = vmin_f32(vacc0c1, vget_low_f32(vmax));
        float32x2_t vout0c2 = vmin_f32(vacc0c2, vget_low_f32(vmax));
        float32x2_t vout0c3 = vmin_f32(vacc0c3, vget_low_f32(vmax));

        vout0c0 = vmax_f32(vout0c0, vget_low_f32(vmin));
        vout0c1 = vmax_f32(vout0c1, vget_low_f32(vmin));
        vout0c2 = vmax_f32(vout0c2, vget_low_f32(vmin));
        vout0c3 = vmax_f32(vout0c3, vget_low_f32(vmin));

        vst1_lane_f32(output + 0 * batch_size + 0, vout0c0, 0);
        vst1_lane_f32(output + 1 * batch_size + 0, vout0c1, 0);
        vst1_lane_f32(output + 2 * batch_size + 0, vout0c2, 0);
        vst1_lane_f32(output + 3 * batch_size + 0, vout0c3, 0);
        output += 4 * batch_size;
        c -= 4;
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
