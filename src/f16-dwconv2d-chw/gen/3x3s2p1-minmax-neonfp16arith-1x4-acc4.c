// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv2d-chw/3x3s2p1-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x4_acc4(
    size_t input_height,
    size_t input_width,
    const void* input,
    const void* weights,
    const void* zero,
    void* output,
    uint32_t padding_top,
    const union xnn_f16_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(__fp16) == 0);
  assert(padding_top <= 1);

  const uint16x4_t vmask_even = vld1_u16(params->neonfp16arith.maskx4_even);
  const uint16x4_t vmask_odd  = vld1_u16(params->neonfp16arith.maskx4_odd);
  const float16x4_t vmax = vld1_dup_f16(&params->neonfp16arith.max);
  const float16x4_t vmin = vld1_dup_f16(&params->neonfp16arith.min);

  const __fp16* w0 = (const __fp16*)weights;
  const float16x8_t vw01234567 = vld1q_f16(w0);
  const float16x4_t vw89 = vreinterpret_f16_u32(vld1_dup_u32((const void*)(w0 + 8)));

  const size_t input_decrement = round_down_po2(input_width, 4 /* SIMD output width */ * 2 /* subsampling */ * sizeof(__fp16));

  const __fp16* i0 = (const __fp16*) ((uintptr_t) input - ((-padding_top) & input_width));
  const __fp16* i1 = (const __fp16*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const __fp16* i2 = (const __fp16*) ((uintptr_t) i1 + input_width);

  __fp16* o0 = output;

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }

    float16x4_t vi0x1357 = vmov_n_f16(0);
    float16x4_t vi1x1357 = vmov_n_f16(0);
    float16x4_t vi2x1357 = vmov_n_f16(0);

    size_t w = input_width;
    for (; w >= 8 * sizeof(__fp16); w -= 8 * sizeof(__fp16)) {
      float16x4_t vo0p0 = vdup_laneq_f16(vw01234567, 0);

      const float16x4x2_t vi0x8ACE9BDF = vld2_f16(i0); i0 += 8;
      const float16x4x2_t vi1x8ACE9BDF = vld2_f16(i1); i1 += 8;
      const float16x4x2_t vi2x8ACE9BDF = vld2_f16(i2); i2 += 8;

      // Center column
      float16x4_t vo0p1 = vmul_laneq_f16(vi0x8ACE9BDF.val[0], vw01234567, 2);

      float16x4_t vo0p2 = vmul_laneq_f16(vi1x8ACE9BDF.val[0], vw01234567, 5);

      float16x4_t vo0p3 = vmul_lane_f16(vi2x8ACE9BDF.val[0], vw89, 0);

      // Left column
      const float16x4_t vi0x7BDF = vext_f16(vi0x1357, vi0x8ACE9BDF.val[1], 3);
      vi0x1357 = vi0x8ACE9BDF.val[1];
      const float16x4_t vi1x7BDF = vext_f16(vi1x1357, vi1x8ACE9BDF.val[1], 3);
      vi1x1357 = vi1x8ACE9BDF.val[1];
      const float16x4_t vi2x7BDF = vext_f16(vi2x1357, vi2x8ACE9BDF.val[1], 3);
      vi2x1357 = vi2x8ACE9BDF.val[1];

      vo0p1 = vfma_laneq_f16(vo0p1, vi0x7BDF, vw01234567, 1);

      vo0p2 = vfma_laneq_f16(vo0p2, vi1x7BDF, vw01234567, 4);

      vo0p3 = vfma_laneq_f16(vo0p3, vi2x7BDF, vw01234567, 7);

      // Right column
      vo0p0 = vfma_laneq_f16(vo0p0, vi0x8ACE9BDF.val[1], vw01234567, 3);

      vo0p1 = vfma_laneq_f16(vo0p1, vi1x8ACE9BDF.val[1], vw01234567, 6);

      vo0p2 = vfma_lane_f16(vo0p2, vi2x8ACE9BDF.val[1], vw89, 1);

      vo0p0 = vadd_f16(vo0p0, vo0p1);
      vo0p2 = vadd_f16(vo0p2, vo0p3);
      vo0p0 = vadd_f16(vo0p0, vo0p2);

      float16x4_t vo0 = vmax_f16(vo0p0, vmin);

      vo0 = vmin_f16(vo0, vmax);

      vst1_f16(o0, vo0); o0 += 4;
    }

    // Last block has 0-7 pixels to process.
    assert(w < 8 * sizeof(__fp16));
    if XNN_LIKELY(w != 0) {
      float16x4_t vo0p0 = vdup_laneq_f16(vw01234567, 0);

      const float16x4x2_t vi0x8ACE9BDF = vld2_f16(i0);
      const float16x4x2_t vi1x8ACE9BDF = vld2_f16(i1);
      const float16x4x2_t vi2x8ACE9BDF = vld2_f16(i2);

      const float16x4_t vi0x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi0x8ACE9BDF.val[0])));
      const float16x4_t vi0x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi0x8ACE9BDF.val[1])));
      const float16x4_t vi1x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi1x8ACE9BDF.val[0])));
      const float16x4_t vi1x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi1x8ACE9BDF.val[1])));
      const float16x4_t vi2x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi2x8ACE9BDF.val[0])));
      const float16x4_t vi2x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi2x8ACE9BDF.val[1])));

      // Center column
      float16x4_t vo0p1 = vmul_laneq_f16(vi0x8ACE, vw01234567, 2);

      float16x4_t vo0p2 = vmul_laneq_f16(vi1x8ACE, vw01234567, 5);

      float16x4_t vo0p3 = vmul_lane_f16(vi2x8ACE, vw89, 0);

      // Left column
      const float16x4_t vi0x7BDF = vext_f16(vi0x1357, vi0x9BDF, 3);
      const float16x4_t vi1x7BDF = vext_f16(vi1x1357, vi1x9BDF, 3);
      const float16x4_t vi2x7BDF = vext_f16(vi2x1357, vi2x9BDF, 3);

      vo0p1 = vfma_laneq_f16(vo0p1, vi0x7BDF, vw01234567, 1);

      vo0p2 = vfma_laneq_f16(vo0p2, vi1x7BDF, vw01234567, 4);

      vo0p3 = vfma_laneq_f16(vo0p3, vi2x7BDF, vw01234567, 7);

      // Right column
      vo0p0 = vfma_laneq_f16(vo0p0, vi0x9BDF, vw01234567, 3);

      vo0p1 = vfma_laneq_f16(vo0p1, vi1x9BDF, vw01234567, 6);

      vo0p2 = vfma_lane_f16(vo0p2, vi2x9BDF, vw89, 1);

      vo0p0 = vadd_f16(vo0p0, vo0p1);
      vo0p2 = vadd_f16(vo0p2, vo0p3);
      vo0p0 = vadd_f16(vo0p0, vo0p2);

      float16x4_t vo0 = vmax_f16(vo0p0, vmin);

      vo0 = vmin_f16(vo0, vmax);

      w += 1 * sizeof(__fp16);

      if XNN_LIKELY(w == 8 * sizeof(__fp16)) {
        vst1_f16(o0, vo0); o0 += 4;
      } else {
        if (w & (4 * sizeof(__fp16))) {
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0), 0); o0 += 2;

          vo0 = vext_f16(vo0, vo0, 2);
        }
        if (w & (2 * sizeof(__fp16))) {
          vst1_lane_f16(o0, vo0, 0); o0 += 1;
        }
      }

    }

    i0 = (const __fp16*) ((uintptr_t) i2 - input_decrement);
    i1 = (const __fp16*) ((uintptr_t) i0 + input_width);
    i2 = (const __fp16*) ((uintptr_t) i1 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}
