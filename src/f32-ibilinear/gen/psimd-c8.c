// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear/psimd.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/ibilinear.h>


void xnn_f32_ibilinear_ukernel__psimd_c8(
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

    const psimd_f32 valphah = psimd_load_splat_f32(weights);
    const psimd_f32 valphav = psimd_load_splat_f32(weights + 1);
    weights += 2;

    size_t c = channels;
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const psimd_f32 vtl0123 = psimd_load_f32(i0);
      const psimd_f32 vtr0123 = psimd_load_f32(i1);
      const psimd_f32 vbl0123 = psimd_load_f32(i2);
      const psimd_f32 vbr0123 = psimd_load_f32(i3);
      const psimd_f32 vtl4567 = psimd_load_f32(i0 + 4);
      const psimd_f32 vtr4567 = psimd_load_f32(i1 + 4);
      const psimd_f32 vbl4567 = psimd_load_f32(i2 + 4);
      const psimd_f32 vbr4567 = psimd_load_f32(i3 + 4);
      i0 += 8;
      i1 += 8;
      i2 += 8;
      i3 += 8;

      const psimd_f32 vtd0123 = psimd_sub_f32(vtr0123, vtl0123);
      const psimd_f32 vbd0123 = psimd_sub_f32(vbr0123, vbl0123);
      const psimd_f32 vtd4567 = psimd_sub_f32(vtr4567, vtl4567);
      const psimd_f32 vbd4567 = psimd_sub_f32(vbr4567, vbl4567);

      const psimd_f32 vt0123 = psimd_qfma_f32(vtl0123, vtd0123, valphah);
      const psimd_f32 vb0123 = psimd_qfma_f32(vbl0123, vbd0123, valphah);
      const psimd_f32 vt4567 = psimd_qfma_f32(vtl4567, vtd4567, valphah);
      const psimd_f32 vb4567 = psimd_qfma_f32(vbl4567, vbd4567, valphah);

      const psimd_f32 vd0123 = psimd_sub_f32(vb0123, vt0123);
      const psimd_f32 vd4567 = psimd_sub_f32(vb4567, vt4567);

      const psimd_f32 vo0123 = psimd_qfma_f32(vt0123, vd0123, valphav);
      const psimd_f32 vo4567 = psimd_qfma_f32(vt4567, vd4567, valphav);

      psimd_store_f32(output, vo0123);
      psimd_store_f32(output + 4, vo4567);
      output += 8;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const psimd_f32 vtl0123 = psimd_load_f32(i0);
      const psimd_f32 vtr0123 = psimd_load_f32(i1);
      const psimd_f32 vbl0123 = psimd_load_f32(i2);
      const psimd_f32 vbr0123 = psimd_load_f32(i3);
      i0 += 4;
      i1 += 4;
      i2 += 4;
      i3 += 4;

      const psimd_f32 vtd0123 = psimd_sub_f32(vtr0123, vtl0123);
      const psimd_f32 vbd0123 = psimd_sub_f32(vbr0123, vbl0123);

      const psimd_f32 vt0123 = psimd_qfma_f32(vtl0123, vtd0123, valphah);
      const psimd_f32 vb0123 = psimd_qfma_f32(vbl0123, vbd0123, valphah);

      const psimd_f32 vd0123 = psimd_sub_f32(vb0123, vt0123);

      const psimd_f32 vo0123 = psimd_qfma_f32(vt0123, vd0123, valphav);

      psimd_store_f32(output, vo0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const psimd_f32 vtl0123 = psimd_load_f32(i0);
      const psimd_f32 vtr0123 = psimd_load_f32(i1);
      const psimd_f32 vbl0123 = psimd_load_f32(i2);
      const psimd_f32 vbr0123 = psimd_load_f32(i3);

      const psimd_f32 vtd0123 = psimd_sub_f32(vtr0123, vtl0123);
      const psimd_f32 vbd0123 = psimd_sub_f32(vbr0123, vbl0123);

      const psimd_f32 vt0123 = psimd_qfma_f32(vtl0123, vtd0123, valphah);
      const psimd_f32 vb0123 = psimd_qfma_f32(vbl0123, vbd0123, valphah);

      const psimd_f32 vd0123 = psimd_sub_f32(vb0123, vt0123);

      psimd_f32 vo0123 = psimd_qfma_f32(vt0123, vd0123, valphav);

      if (c & (2 * sizeof(float))) {
        psimd_store2_f32(output, vo0123);
        vo0123 = psimd_concat_hi_f32(vo0123, vo0123);
        output += 2;
      }
      if (c & (1 * sizeof(float))) {
        psimd_store1_f32(output, vo0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
