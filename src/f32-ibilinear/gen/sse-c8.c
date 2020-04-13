// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/ibilinear.h>


void xnn_f32_ibilinear_ukernel__sse_c8(
    size_t output_pixels,
    size_t channels,
    const float**restrict input,
    size_t input_offset,
    const float*restrict weights,
    float*restrict output,
    size_t output_increment)
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

    __m128 valphahv = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) weights);
    valphahv = _mm_unpacklo_ps(valphahv, valphahv);
    const __m128 valphah = _mm_movelh_ps(valphahv, valphahv);
    const __m128 valphav = _mm_movehl_ps(valphahv, valphahv);
    weights += 2;

    size_t c = channels;
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const __m128 vtl0123 = _mm_loadu_ps(i0);
      const __m128 vtr0123 = _mm_loadu_ps(i1);
      const __m128 vbl0123 = _mm_loadu_ps(i2);
      const __m128 vbr0123 = _mm_loadu_ps(i3);
      const __m128 vtl4567 = _mm_loadu_ps(i0 + 4);
      const __m128 vtr4567 = _mm_loadu_ps(i1 + 4);
      const __m128 vbl4567 = _mm_loadu_ps(i2 + 4);
      const __m128 vbr4567 = _mm_loadu_ps(i3 + 4);
      i0 += 8;
      i1 += 8;
      i2 += 8;
      i3 += 8;

      const __m128 vtd0123 = _mm_sub_ps(vtr0123, vtl0123);
      const __m128 vbd0123 = _mm_sub_ps(vbr0123, vbl0123);
      const __m128 vtd4567 = _mm_sub_ps(vtr4567, vtl4567);
      const __m128 vbd4567 = _mm_sub_ps(vbr4567, vbl4567);

      const __m128 vt0123 = _mm_add_ps(vtl0123, _mm_mul_ps(vtd0123, valphah));
      const __m128 vb0123 = _mm_add_ps(vbl0123, _mm_mul_ps(vbd0123, valphah));
      const __m128 vt4567 = _mm_add_ps(vtl4567, _mm_mul_ps(vtd4567, valphah));
      const __m128 vb4567 = _mm_add_ps(vbl4567, _mm_mul_ps(vbd4567, valphah));

      const __m128 vd0123 = _mm_sub_ps(vb0123, vt0123);
      const __m128 vd4567 = _mm_sub_ps(vb4567, vt4567);

      const __m128 vo0123 = _mm_add_ps(vt0123, _mm_mul_ps(vd0123, valphav));
      const __m128 vo4567 = _mm_add_ps(vt4567, _mm_mul_ps(vd4567, valphav));

      _mm_storeu_ps(output, vo0123);
      _mm_storeu_ps(output + 4, vo4567);
      output += 8;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const __m128 vtl0123 = _mm_loadu_ps(i0);
      const __m128 vtr0123 = _mm_loadu_ps(i1);
      const __m128 vbl0123 = _mm_loadu_ps(i2);
      const __m128 vbr0123 = _mm_loadu_ps(i3);
      i0 += 4;
      i1 += 4;
      i2 += 4;
      i3 += 4;

      const __m128 vtd0123 = _mm_sub_ps(vtr0123, vtl0123);
      const __m128 vbd0123 = _mm_sub_ps(vbr0123, vbl0123);

      const __m128 vt0123 = _mm_add_ps(vtl0123, _mm_mul_ps(vtd0123, valphah));
      const __m128 vb0123 = _mm_add_ps(vbl0123, _mm_mul_ps(vbd0123, valphah));

      const __m128 vd0123 = _mm_sub_ps(vb0123, vt0123);

      const __m128 vo0123 = _mm_add_ps(vt0123, _mm_mul_ps(vd0123, valphav));

      _mm_storeu_ps(output, vo0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m128 vtl0123 = _mm_loadu_ps(i0);
      const __m128 vtr0123 = _mm_loadu_ps(i1);
      const __m128 vbl0123 = _mm_loadu_ps(i2);
      const __m128 vbr0123 = _mm_loadu_ps(i3);

      const __m128 vtd0123 = _mm_sub_ps(vtr0123, vtl0123);
      const __m128 vbd0123 = _mm_sub_ps(vbr0123, vbl0123);

      const __m128 vt0123 = _mm_add_ps(vtl0123, _mm_mul_ps(vtd0123, valphah));
      const __m128 vb0123 = _mm_add_ps(vbl0123, _mm_mul_ps(vbd0123, valphah));

      const __m128 vd0123 = _mm_sub_ps(vb0123, vt0123);

      __m128 vo0123 = _mm_add_ps(vt0123, _mm_mul_ps(vd0123, valphav));

      if (c & (2 * sizeof(float))) {
        _mm_storel_pi((__m64*) output, vo0123);
        vo0123 = _mm_movehl_ps(vo0123, vo0123);
        output += 2;
      }
      if (c & (1 * sizeof(float))) {
        _mm_store_ss(output, vo0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
