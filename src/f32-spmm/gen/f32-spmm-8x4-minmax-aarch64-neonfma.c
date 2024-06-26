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


void xnn_f32_spmm_minmax_ukernel_8x4__aarch64_neonfma(
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

  size_t output_decrement = output_stride * nc - 8 * sizeof(float);
  while XNN_LIKELY(mc >= 8 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 4) {
      uint32_t nnz = *nnzmap++;
      float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n0 = vacc0123n0;
      float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n1 = vacc0123n1;
      float32x4_t vacc0123n2 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n2 = vacc0123n2;
      float32x4_t vacc0123n3 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n3 = vacc0123n3;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float32x4_t vi0123 = vld1q_f32(input);
          const float32x4_t vi4567 = vld1q_f32(input + 4);
          input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
          xnn_prefetch_to_l1(input + 16);
          const float32x4_t vw = vld1q_f32(w); w += 4;
          xnn_prefetch_to_l1(w + 32);
          vacc0123n0 = vfmaq_laneq_f32(vacc0123n0, vi0123, vw, 0);
          vacc4567n0 = vfmaq_laneq_f32(vacc4567n0, vi4567, vw, 0);
          vacc0123n1 = vfmaq_laneq_f32(vacc0123n1, vi0123, vw, 1);
          vacc4567n1 = vfmaq_laneq_f32(vacc4567n1, vi4567, vw, 1);
          vacc0123n2 = vfmaq_laneq_f32(vacc0123n2, vi0123, vw, 2);
          vacc4567n2 = vfmaq_laneq_f32(vacc4567n2, vi4567, vw, 2);
          vacc0123n3 = vfmaq_laneq_f32(vacc0123n3, vi0123, vw, 3);
          vacc4567n3 = vfmaq_laneq_f32(vacc4567n3, vi4567, vw, 3);
        } while (--nnz != 0);
      }
      float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
      float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
      float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
      float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
      float32x4_t vout0123n2 = vminq_f32(vacc0123n2, vmax);
      float32x4_t vout4567n2 = vminq_f32(vacc4567n2, vmax);
      float32x4_t vout0123n3 = vminq_f32(vacc0123n3, vmax);
      float32x4_t vout4567n3 = vminq_f32(vacc4567n3, vmax);

      vout0123n0 = vmaxq_f32(vout0123n0, vmin);
      vout4567n0 = vmaxq_f32(vout4567n0, vmin);
      vout0123n1 = vmaxq_f32(vout0123n1, vmin);
      vout4567n1 = vmaxq_f32(vout4567n1, vmin);
      vout0123n2 = vmaxq_f32(vout0123n2, vmin);
      vout4567n2 = vmaxq_f32(vout4567n2, vmin);
      vout0123n3 = vmaxq_f32(vout0123n3, vmin);
      vout4567n3 = vmaxq_f32(vout4567n3, vmin);

      vst1q_f32(output + 0, vout0123n0);
      vst1q_f32(output + 4, vout4567n0);
      output = (float*) ((uintptr_t) output + output_stride);
      vst1q_f32(output + 0, vout0123n1);
      vst1q_f32(output + 4, vout4567n1);
      output = (float*) ((uintptr_t) output + output_stride);
      vst1q_f32(output + 0, vout0123n2);
      vst1q_f32(output + 4, vout4567n2);
      output = (float*) ((uintptr_t) output + output_stride);
      vst1q_f32(output + 0, vout0123n3);
      vst1q_f32(output + 4, vout4567n3);
      output = (float*) ((uintptr_t) output + output_stride);
      n -= 4;
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
            xnn_prefetch_to_l1(input + 16);
            const float32x4_t vw = vld1q_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
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
    mc -= 8 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123n2 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123n3 = vld1q_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc0123n0 = vfmaq_laneq_f32(vacc0123n0, vi0123, vw, 0);
            vacc0123n1 = vfmaq_laneq_f32(vacc0123n1, vi0123, vw, 1);
            vacc0123n2 = vfmaq_laneq_f32(vacc0123n2, vi0123, vw, 2);
            vacc0123n3 = vfmaq_laneq_f32(vacc0123n3, vi0123, vw, 3);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout0123n2 = vminq_f32(vacc0123n2, vmax);
        float32x4_t vout0123n3 = vminq_f32(vacc0123n3, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout0123n2 = vmaxq_f32(vout0123n2, vmin);
        vout0123n3 = vmaxq_f32(vout0123n3, vmin);

        vst1q_f32(output + 0, vout0123n0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n2);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n3);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 4;
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
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc01n0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01n1 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01n2 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01n3 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi01 = vld1_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc01n0 = vfma_laneq_f32(vacc01n0, vi01, vw, 0);
            vacc01n1 = vfma_laneq_f32(vacc01n1, vi01, vw, 1);
            vacc01n2 = vfma_laneq_f32(vacc01n2, vi01, vw, 2);
            vacc01n3 = vfma_laneq_f32(vacc01n3, vi01, vw, 3);
          } while (--nnz != 0);
        }
        float32x2_t vout01n0 = vmin_f32(vacc01n0, vget_low_f32(vmax));
        float32x2_t vout01n1 = vmin_f32(vacc01n1, vget_low_f32(vmax));
        float32x2_t vout01n2 = vmin_f32(vacc01n2, vget_low_f32(vmax));
        float32x2_t vout01n3 = vmin_f32(vacc01n3, vget_low_f32(vmax));

        vout01n0 = vmax_f32(vout01n0, vget_low_f32(vmin));
        vout01n1 = vmax_f32(vout01n1, vget_low_f32(vmin));
        vout01n2 = vmax_f32(vout01n2, vget_low_f32(vmin));
        vout01n3 = vmax_f32(vout01n3, vget_low_f32(vmin));

        vst1_f32(output + 0, vout01n0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_f32(output + 0, vout01n1);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_f32(output + 0, vout01n2);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_f32(output + 0, vout01n3);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 4;
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
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc0n0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0n1 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0n2 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0n3 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi0 = vld1_dup_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc0n0 = vfma_laneq_f32(vacc0n0, vi0, vw, 0);
            vacc0n1 = vfma_laneq_f32(vacc0n1, vi0, vw, 1);
            vacc0n2 = vfma_laneq_f32(vacc0n2, vi0, vw, 2);
            vacc0n3 = vfma_laneq_f32(vacc0n3, vi0, vw, 3);
          } while (--nnz != 0);
        }
        float32x2_t vout0n0 = vmin_f32(vacc0n0, vget_low_f32(vmax));
        float32x2_t vout0n1 = vmin_f32(vacc0n1, vget_low_f32(vmax));
        float32x2_t vout0n2 = vmin_f32(vacc0n2, vget_low_f32(vmax));
        float32x2_t vout0n3 = vmin_f32(vacc0n3, vget_low_f32(vmax));

        vout0n0 = vmax_f32(vout0n0, vget_low_f32(vmin));
        vout0n1 = vmax_f32(vout0n1, vget_low_f32(vmin));
        vout0n2 = vmax_f32(vout0n2, vget_low_f32(vmin));
        vout0n3 = vmax_f32(vout0n3, vget_low_f32(vmin));

        vst1_lane_f32(output + 0, vout0n0, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_lane_f32(output + 0, vout0n1, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_lane_f32(output + 0, vout0n2, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_lane_f32(output + 0, vout0n3, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 4;
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
