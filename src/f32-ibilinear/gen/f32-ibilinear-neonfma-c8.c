// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/ibilinear.h"


void xnn_f32_ibilinear_ukernel__neonfma_c8(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  do {
    const float* i0 = (const float*) ((uintptr_t) input[0] + input_offset);
    const float* i1 = (const float*) ((uintptr_t) input[1] + input_offset);
    const float* i2 = (const float*) ((uintptr_t) input[2] + input_offset);
    const float* i3 = (const float*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const float32x2_t valphahv = vld1_f32(weights); weights += 2;
    #if XNN_ARCH_ARM
      const float32x4_t valphah = vdupq_lane_f32(valphahv, 0);
      const float32x4_t valphav = vdupq_lane_f32(valphahv, 1);
    #endif

    size_t c = channels;
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const float32x4_t vtl0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vtr0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vbl0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vbr0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vtl4567 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vtr4567 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vbl4567 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vbr4567 = vld1q_f32(i3); i3 += 4;

      const float32x4_t vtd0123 = vsubq_f32(vtr0123, vtl0123);
      const float32x4_t vbd0123 = vsubq_f32(vbr0123, vbl0123);
      const float32x4_t vtd4567 = vsubq_f32(vtr4567, vtl4567);
      const float32x4_t vbd4567 = vsubq_f32(vbr4567, vbl4567);

      #if XNN_ARCH_ARM
      const float32x4_t vt0123 = vfmaq_f32(vtl0123, vtd0123, valphah);
      const float32x4_t vb0123 = vfmaq_f32(vbl0123, vbd0123, valphah);
      const float32x4_t vt4567 = vfmaq_f32(vtl4567, vtd4567, valphah);
      const float32x4_t vb4567 = vfmaq_f32(vbl4567, vbd4567, valphah);
      #else
      const float32x4_t vt0123 = vfmaq_lane_f32(vtl0123, vtd0123, valphahv, 0);
      const float32x4_t vb0123 = vfmaq_lane_f32(vbl0123, vbd0123, valphahv, 0);
      const float32x4_t vt4567 = vfmaq_lane_f32(vtl4567, vtd4567, valphahv, 0);
      const float32x4_t vb4567 = vfmaq_lane_f32(vbl4567, vbd4567, valphahv, 0);
      #endif

      const float32x4_t vd0123 = vsubq_f32(vb0123, vt0123);
      const float32x4_t vd4567 = vsubq_f32(vb4567, vt4567);

      #if XNN_ARCH_ARM
      const float32x4_t vo0123 = vfmaq_f32(vt0123, vd0123, valphav);
      const float32x4_t vo4567 = vfmaq_f32(vt4567, vd4567, valphav);
      #else
      const float32x4_t vo0123 = vfmaq_lane_f32(vt0123, vd0123, valphahv, 1);
      const float32x4_t vo4567 = vfmaq_lane_f32(vt4567, vd4567, valphahv, 1);
      #endif

      vst1q_f32(output, vo0123); output += 4;
      vst1q_f32(output, vo4567); output += 4;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float32x4_t vtl0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vtr0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vbl0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vbr0123 = vld1q_f32(i3); i3 += 4;

      const float32x4_t vtd0123 = vsubq_f32(vtr0123, vtl0123);
      const float32x4_t vbd0123 = vsubq_f32(vbr0123, vbl0123);

      #if XNN_ARCH_ARM
      const float32x4_t vt0123 = vfmaq_f32(vtl0123, vtd0123, valphah);
      const float32x4_t vb0123 = vfmaq_f32(vbl0123, vbd0123, valphah);
      #else
      const float32x4_t vt0123 = vfmaq_lane_f32(vtl0123, vtd0123, valphahv, 0);
      const float32x4_t vb0123 = vfmaq_lane_f32(vbl0123, vbd0123, valphahv, 0);
      #endif

      const float32x4_t vd0123 = vsubq_f32(vb0123, vt0123);

      #if XNN_ARCH_ARM
      const float32x4_t vo0123 = vfmaq_f32(vt0123, vd0123, valphav);
      #else
      const float32x4_t vo0123 = vfmaq_lane_f32(vt0123, vd0123, valphahv, 1);
      #endif

      vst1q_f32(output, vo0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const float32x4_t vtl0123 = vld1q_f32(i0);
      const float32x4_t vtr0123 = vld1q_f32(i1);
      const float32x4_t vbl0123 = vld1q_f32(i2);
      const float32x4_t vbr0123 = vld1q_f32(i3);

      const float32x4_t vtd0123 = vsubq_f32(vtr0123, vtl0123);
      const float32x4_t vbd0123 = vsubq_f32(vbr0123, vbl0123);

        #if XNN_ARCH_ARM
        const float32x4_t vt0123 = vfmaq_f32(vtl0123, vtd0123, valphah);
        const float32x4_t vb0123 = vfmaq_f32(vbl0123, vbd0123, valphah);
        #else
        const float32x4_t vt0123 = vfmaq_lane_f32(vtl0123, vtd0123, valphahv, 0);
        const float32x4_t vb0123 = vfmaq_lane_f32(vbl0123, vbd0123, valphahv, 0);
        #endif

      const float32x4_t vd0123 = vsubq_f32(vb0123, vt0123);

      #if XNN_ARCH_ARM
      float32x4_t vo0123 = vfmaq_f32(vt0123, vd0123, valphav);
      #else
      float32x4_t vo0123 = vfmaq_lane_f32(vt0123, vd0123, valphahv, 1);
      #endif

      float32x2_t vo01 = vget_low_f32(vo0123);
      if (c & (2 * sizeof(float))) {
        vst1_f32(output, vo01); output += 2;
        vo01 = vget_high_f32(vo0123);
      }
      if (c & (1 * sizeof(float))) {
        vst1_lane_f32(output, vo01, 0); output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
