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


void xnn_f32_ibilinear_chw_ukernel__neon_p8(
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
    for (; p >= 8; p -= 8) {
      const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
      const float* itl2 = (const float*) ((uintptr_t) i[4] + input_offset);
      const float* ibl2 = (const float*) ((uintptr_t) i[5] + input_offset);
      const float* itl3 = (const float*) ((uintptr_t) i[6] + input_offset);
      const float* ibl3 = (const float*) ((uintptr_t) i[7] + input_offset);
      const float* itl4 = (const float*) ((uintptr_t) i[8] + input_offset);
      const float* ibl4 = (const float*) ((uintptr_t) i[9] + input_offset);
      const float* itl5 = (const float*) ((uintptr_t) i[10] + input_offset);
      const float* ibl5 = (const float*) ((uintptr_t) i[11] + input_offset);
      const float* itl6 = (const float*) ((uintptr_t) i[12] + input_offset);
      const float* ibl6 = (const float*) ((uintptr_t) i[13] + input_offset);
      const float* itl7 = (const float*) ((uintptr_t) i[14] + input_offset);
      const float* ibl7 = (const float*) ((uintptr_t) i[15] + input_offset);
      i += 2 * 8;

      const float32x4x2_t vw0123 = vld2q_f32(w + 0);
      const float32x4x2_t vw4567 = vld2q_f32(w + 8);
      w += 2 * 8;

      const float32x2_t vtltr0 = vld1_f32(itl0);
      const float32x2_t vblbr0 = vld1_f32(ibl0);
      const float32x2_t vtltr1 = vld1_f32(itl1);
      const float32x2_t vblbr1 = vld1_f32(ibl1);
      const float32x2_t vtltr2 = vld1_f32(itl2);
      const float32x2_t vblbr2 = vld1_f32(ibl2);
      const float32x2_t vtltr3 = vld1_f32(itl3);
      const float32x2_t vblbr3 = vld1_f32(ibl3);
      const float32x2_t vtltr4 = vld1_f32(itl4);
      const float32x2_t vblbr4 = vld1_f32(ibl4);
      const float32x2_t vtltr5 = vld1_f32(itl5);
      const float32x2_t vblbr5 = vld1_f32(ibl5);
      const float32x2_t vtltr6 = vld1_f32(itl6);
      const float32x2_t vblbr6 = vld1_f32(ibl6);
      const float32x2_t vtltr7 = vld1_f32(itl7);
      const float32x2_t vblbr7 = vld1_f32(ibl7);

      const float32x4_t valphah0123 = vw0123.val[0];
      const float32x4_t valphav0123 = vw0123.val[1];
      const float32x4_t valphah4567 = vw4567.val[0];
      const float32x4_t valphav4567 = vw4567.val[1];

      const float32x4_t vtltr01 = vcombine_f32(vtltr0, vtltr1);
      const float32x4_t vblbr01 = vcombine_f32(vblbr0, vblbr1);
      const float32x4_t vtltr23 = vcombine_f32(vtltr2, vtltr3);
      const float32x4_t vblbr23 = vcombine_f32(vblbr2, vblbr3);
      const float32x4_t vtltr45 = vcombine_f32(vtltr4, vtltr5);
      const float32x4_t vblbr45 = vcombine_f32(vblbr4, vblbr5);
      const float32x4_t vtltr67 = vcombine_f32(vtltr6, vtltr7);
      const float32x4_t vblbr67 = vcombine_f32(vblbr6, vblbr7);

      const float32x4_t vldrd01 = vsubq_f32(vblbr01, vtltr01);
      const float32x4_t vldrd23 = vsubq_f32(vblbr23, vtltr23);
      const float32x4_t vldrd45 = vsubq_f32(vblbr45, vtltr45);
      const float32x4_t vldrd67 = vsubq_f32(vblbr67, vtltr67);

      const float32x4x2_t vld_t0123 = vuzpq_f32(vldrd01, vldrd23);
      const float32x4_t vld0123 = vld_t0123.val[0];
      const float32x4_t vrd0123 = vld_t0123.val[1];
      const float32x4x2_t vld_t4567 = vuzpq_f32(vldrd45, vldrd67);
      const float32x4_t vld4567 = vld_t4567.val[0];
      const float32x4_t vrd4567 = vld_t4567.val[1];

      const float32x4x2_t vtl_t0123 = vuzpq_f32(vtltr01, vtltr23);
      const float32x4_t vtl0123 = vtl_t0123.val[0];
      const float32x4_t vtr0123 = vtl_t0123.val[1];
      const float32x4x2_t vtl_t4567 = vuzpq_f32(vtltr45, vtltr67);
      const float32x4_t vtl4567 = vtl_t4567.val[0];
      const float32x4_t vtr4567 = vtl_t4567.val[1];

      const float32x4_t vl0123 = vmlaq_f32(vtl0123, vld0123, valphav0123);
      const float32x4_t vr0123 = vmlaq_f32(vtr0123, vrd0123, valphav0123);
      const float32x4_t vl4567 = vmlaq_f32(vtl4567, vld4567, valphav4567);
      const float32x4_t vr4567 = vmlaq_f32(vtr4567, vrd4567, valphav4567);

      const float32x4_t vd0123 = vsubq_f32(vr0123, vl0123);
      const float32x4_t vd4567 = vsubq_f32(vr4567, vl4567);

      const float32x4_t vo0123 = vmlaq_f32(vl0123, vd0123, valphah0123);
      const float32x4_t vo4567 = vmlaq_f32(vl4567, vd4567, valphah4567);

      vst1q_f32(output + 0, vo0123);
      vst1q_f32(output + 4, vo4567);
      output += 8;
    }

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
