// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear-chw/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/ibilinear.h"


void xnn_f32_ibilinear_chw_ukernel__neon_p4(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t input_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(float) == 0);

  do {
    const float** i = input;
    const float* w = weights;
    size_t p = output_pixels;

    for (; p >= 4; p -= 4) {
      const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
      const float* itl2 = (const float*) ((uintptr_t) i[4] + input_offset);
      const float* ibl2 = (const float*) ((uintptr_t) i[5] + input_offset);
      const float* itl3 = (const float*) ((uintptr_t) i[6] + input_offset);
      const float* ibl3 = (const float*) ((uintptr_t) i[7] + input_offset);
      i += 8;

      const float32x4x2_t vw = vld2q_f32(w);
      w += 8;

      const float32x2_t vtltr0 = vld1_f32(itl0);
      const float32x2_t vblbr0 = vld1_f32(ibl0);
      const float32x2_t vtltr1 = vld1_f32(itl1);
      const float32x2_t vblbr1 = vld1_f32(ibl1);
      const float32x2_t vtltr2 = vld1_f32(itl2);
      const float32x2_t vblbr2 = vld1_f32(ibl2);
      const float32x2_t vtltr3 = vld1_f32(itl3);
      const float32x2_t vblbr3 = vld1_f32(ibl3);

      const float32x4_t valphah = vw.val[0];
      const float32x4_t valphav = vw.val[1];

      const float32x4_t vtltr01 = vcombine_f32(vtltr0, vtltr1);
      const float32x4_t vblbr01 = vcombine_f32(vblbr0, vblbr1);
      const float32x4_t vtltr23 = vcombine_f32(vtltr2, vtltr3);
      const float32x4_t vblbr23 = vcombine_f32(vblbr2, vblbr3);

      const float32x4_t vldrd01 = vsubq_f32(vblbr01, vtltr01);
      const float32x4_t vldrd23 = vsubq_f32(vblbr23, vtltr23);

      const float32x4x2_t vld_t = vuzpq_f32(vldrd01, vldrd23);
      const float32x4_t vld = vld_t.val[0];
      const float32x4_t vrd = vld_t.val[1];

      const float32x4x2_t vtl_t = vuzpq_f32(vtltr01, vtltr23);
      const float32x4_t vtl = vtl_t.val[0];
      const float32x4_t vtr = vtl_t.val[1];

      const float32x4_t vl = vmlaq_f32(vtl, vld, valphav);
      const float32x4_t vr = vmlaq_f32(vtr, vrd, valphav);

      const float32x4_t vd = vsubq_f32(vr, vl);
      const float32x4_t vo = vmlaq_f32(vl, vd, valphah);

      vst1q_f32(output, vo);
      output += 4;
    }

    if XNN_UNLIKELY(p != 0) {
      if (p & 2) {
        const float32x2x2_t vw = vld2_f32(w);
        w += 4;

        const float32x2_t valphah = vw.val[0];
        const float32x2_t valphav = vw.val[1];

        const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
        const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
        const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
        i += 4;

        const float32x2_t vtltr0 = vld1_f32(itl0);
        const float32x2_t vblbr0 = vld1_f32(ibl0);
        const float32x2_t vtltr1 = vld1_f32(itl1);
        const float32x2_t vblbr1 = vld1_f32(ibl1);

        const float32x2_t vldrd0 = vsub_f32(vblbr0, vtltr0);
        const float32x2_t vldrd1 = vsub_f32(vblbr1, vtltr1);

        const float32x2x2_t vld_t = vuzp_f32(vldrd0, vldrd1);
        const float32x2_t vld = vld_t.val[0];
        const float32x2_t vrd = vld_t.val[1];

        const float32x2x2_t vtl_t = vuzp_f32(vtltr0, vtltr1);
        const float32x2_t vtl = vtl_t.val[0];
        const float32x2_t vtr = vtl_t.val[1];

        const float32x2_t vl = vmla_f32(vtl, vld, valphav);
        const float32x2_t vr = vmla_f32(vtr, vrd, valphav);

        const float32x2_t vd = vsub_f32(vr, vl);
        const float32x2_t vo = vmla_f32(vl, vd, valphah);

        vst1_f32(output, vo);
        output += 2;
      }

      if (p & 1) {
        // We are computing the following formula:
        //   result = (1 - alpha_h) * (1 - alpha_v) * top_left +
        //                 alpha_h  * (1 - alpha_v) * top_right +
        //            (1 - alpha_h) *      alpha_v  * bottom_left +
        //                 alpha_h  *      alpha_v  * bottom_right.
        //
        // Rearranging gives
        //   result =    left + alpha_h * (right        - left),
        // where
        //   left =  top_left + alpha_v * (bottom_left  - top_left),
        //  right = top_right + alpha_v * (bottom_right - top_right).

        const float alphah = *w;
        const float32x2_t valphav = vld1_dup_f32(w + 1);
        w += 2;

        const float* itl = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl = (const float*) ((uintptr_t) i[1] + input_offset);
        i += 2;

        const float32x2_t vtltr = vld1_f32(itl);
        const float32x2_t vblbr = vld1_f32(ibl);

        // Compute at once
        //    left_diff = bottom_left  - top_left
        //   right_diff = bottom_right - top_right
        const float32x2_t vldrd = vsub_f32(vblbr, vtltr);
        const float32x2_t vlr = vmla_f32(vtltr, vldrd, valphav);

        // Extract them and compute the result.
        const float l = vget_lane_f32(vlr, 0);
        const float r = vget_lane_f32(vlr, 1);

        *output++ = l + alphah * (r - l);
      }
    }

    input_offset += input_increment;
  } while (--channels != 0);
}
