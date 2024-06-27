// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear-chw/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/ibilinear.h"


void xnn_f32_ibilinear_chw_ukernel__sse_p4(
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

      const __m128 vw0 = _mm_loadu_ps(w);
      const __m128 vw1 = _mm_loadu_ps(w + 4);
      w += 8;

      const __m128 vtltr0 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl0);
      const __m128 vblbr0 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl0);
      const __m128 vtltr2 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl2);
      const __m128 vblbr2 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl2);

      const __m128 valphah = _mm_shuffle_ps(vw0, vw1, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 valphav = _mm_shuffle_ps(vw0, vw1, _MM_SHUFFLE(3, 1, 3, 1));

      const __m128 vtltr01 = _mm_loadh_pi(vtltr0, (const __m64*) itl1);
      const __m128 vblbr01 = _mm_loadh_pi(vblbr0, (const __m64*) ibl1);
      const __m128 vtltr23 = _mm_loadh_pi(vtltr2, (const __m64*) itl3);
      const __m128 vblbr23 = _mm_loadh_pi(vblbr2, (const __m64*) ibl3);

      const __m128 vldrd01 = _mm_sub_ps(vblbr01, vtltr01);
      const __m128 vldrd23 = _mm_sub_ps(vblbr23, vtltr23);

      const __m128 vld = _mm_shuffle_ps(vldrd01, vldrd23, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vrd = _mm_shuffle_ps(vldrd01, vldrd23, _MM_SHUFFLE(3, 1, 3, 1));

      const __m128 vtl = _mm_shuffle_ps(vtltr01, vtltr23, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vtr = _mm_shuffle_ps(vtltr01, vtltr23, _MM_SHUFFLE(3, 1, 3, 1));

      const __m128 vl = _mm_add_ps(vtl, _mm_mul_ps(vld, valphav));
      const __m128 vr = _mm_add_ps(vtr, _mm_mul_ps(vrd, valphav));

      const __m128 vd = _mm_sub_ps(vr, vl);
      const __m128 vo = _mm_add_ps(vl, _mm_mul_ps(vd, valphah));

      _mm_storeu_ps(output, vo);
      output += 4;
    }

    if XNN_UNLIKELY(p != 0) {
      if (p & 2) {
        const __m128 vw = _mm_loadu_ps(w);
        w += 4;

        const __m128 valphah = _mm_shuffle_ps(vw, vw, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 valphav = _mm_shuffle_ps(vw, vw, _MM_SHUFFLE(3, 1, 3, 1));

        const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
        const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
        const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
        i += 4;

        const __m128 vtltr = _mm_loadh_pi(_mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl0), (const __m64*) itl1);
        const __m128 vblbr = _mm_loadh_pi(_mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl0), (const __m64*) ibl1);

        const __m128 vldrd = _mm_sub_ps(vblbr, vtltr);
        const __m128 vld = _mm_shuffle_ps(vldrd, vldrd, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 vrd = _mm_shuffle_ps(vldrd, vldrd, _MM_SHUFFLE(3, 1, 3, 1));

        const __m128 vtl = _mm_shuffle_ps(vtltr, vtltr, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 vtr = _mm_shuffle_ps(vtltr, vtltr, _MM_SHUFFLE(3, 1, 3, 1));

        const __m128 vl = _mm_add_ps(vtl, _mm_mul_ps(vld, valphav));
        const __m128 vr = _mm_add_ps(vtr, _mm_mul_ps(vrd, valphav));

        const __m128 vd = _mm_sub_ps(vr, vl);
        const __m128 vo = _mm_add_ps(vl, _mm_mul_ps(vd, valphah));

        _mm_storel_pi((__m64*) output, vo);
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
        const __m128 valphav = _mm_load_ps1(w + 1);
        w += 2;

        const float* itl = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl = (const float*) ((uintptr_t) i[1] + input_offset);
        i += 2;

        const __m128 vtltr = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl);
        const __m128 vblbr = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl);

        // Compute at once
        //    left_diff = bottom_left  - top_left
        //   right_diff = bottom_right - top_right
        const __m128 vldrd = _mm_sub_ps(vblbr, vtltr);
        const __m128 vlr = _mm_add_ps(vtltr, _mm_mul_ps(vldrd, valphav));

        // Extract them and compute the result.
        const float l = _mm_cvtss_f32(vlr);
        const float r = _mm_cvtss_f32(_mm_shuffle_ps(vlr, vlr, 1));

        *output++ = l + alphah * (r - l);
      }
    }

    input_offset += input_increment;
  } while (--channels != 0);
}
