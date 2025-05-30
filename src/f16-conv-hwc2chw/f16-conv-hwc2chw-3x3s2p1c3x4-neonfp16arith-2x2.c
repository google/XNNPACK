// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "src/xnnpack/conv.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"


void xnn_f16_conv_hwc2chw_ukernel_3x3s2p1c3x4__neonfp16arith_2x2(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const xnn_float16* input,
    const xnn_float16* zero,
    const xnn_float16* weights,
    xnn_float16* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const struct xnn_f16_minmax_params* restrict params) XNN_OOB_READS
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(uint16_t);
  const size_t input_width_increment = round_down_po2(input_width, 4) * 3 /* channels */ * sizeof(uint16_t);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * 4 - output_width * sizeof(uint16_t);

  // Adjustment for padding processed below
  const uint16_t* i0 = (const uint16_t*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_height_stride);
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_height_stride);
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_height_stride);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_height_stride);
  uint16_t* output0 = (uint16_t*) ((uintptr_t) output + output_height_stride * output_y_start);
  uint16_t* output1 = (uint16_t*) ((uintptr_t) output0 + output_height_stride);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = (const uint16_t*) zero;
  }

  const float16x4_t vmax = vreinterpret_f16_u16(vdup_n_u16(*(const uint16_t*) &params->scalar.max));
  const float16x4_t vmin = vreinterpret_f16_u16(vdup_n_u16(*(const uint16_t*) &params->scalar.min));

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 2) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    const size_t input_y4 = input_y2 + 2;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = (const uint16_t*) zero;
    }
    if XNN_UNPREDICTABLE(input_y4 > input_height) {
      i3 = (const uint16_t*) zero;
    }
    if XNN_UNPREDICTABLE(input_y4 >= input_height) {
      i4 = (const uint16_t*) zero;
    }
    if XNN_UNPREDICTABLE(output_y + 2 > output_y_end) {
      output1 = output0;
    }

    const uint16_t* w = (const uint16_t*) weights;
    size_t c = output_channels;
    uint16_t* o0c0 = output0;
    uint16_t* o1c0 = output1;
    uint16_t* o0c1 = (uint16_t*) ((uintptr_t) o0c0 + output_channel_stride);
    uint16_t* o1c1 = (uint16_t*) ((uintptr_t) o1c0 + output_channel_stride);
    uint16_t* o0c2 = (uint16_t*) ((uintptr_t) o0c1 + output_channel_stride);
    uint16_t* o1c2 = (uint16_t*) ((uintptr_t) o1c1 + output_channel_stride);
    uint16_t* o0c3 = (uint16_t*) ((uintptr_t) o0c2 + output_channel_stride);
    uint16_t* o1c3 = (uint16_t*) ((uintptr_t) o1c2 + output_channel_stride);
    do {
      if XNN_UNPREDICTABLE(c < 2) {
        o0c1 = o0c0;
        o1c1 = o1c0;
      }
      if XNN_UNPREDICTABLE(c <= 2) {
        o0c2 = o0c1;
        o1c2 = o1c1;
      }
      if XNN_UNPREDICTABLE(c < 4) {
        o0c3 = o0c2;
        o1c3 = o1c2;
      }

      // viMx0 = ( iM0c2, iM0c1, iM0c0, --- )
      float16x4_t vi0x0 = vreinterpret_f16_u16(vmov_n_u16(0));
      float16x4_t vi1x0 = vreinterpret_f16_u16(vmov_n_u16(0));
      float16x4_t vi2x0 = vreinterpret_f16_u16(vmov_n_u16(0));
      float16x4_t vi3x0 = vreinterpret_f16_u16(vmov_n_u16(0));
      float16x4_t vi4x0 = vreinterpret_f16_u16(vmov_n_u16(0));

      size_t iw = input_width;
      for (; iw >= 4; iw -= 4) {
        float16x4_t vo0x0 = vreinterpret_f16_u16(vld1_u16(w));
        float16x4_t vo1x0 = vo0x0;
        float16x4_t vo0x1 = vo0x0;
        float16x4_t vo1x1 = vo0x0;

        const float16x4_t vk00c0 = vreinterpret_f16_u16(vld1_u16(w + 4));

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        const float16x4_t vi0x1 = vreinterpret_f16_u16(vld1_u16(i0)); i0 += 4;
        const float16x4_t vi1x1 = vreinterpret_f16_u16(vld1_u16(i1)); i1 += 4;
        const float16x4_t vi2x1 = vreinterpret_f16_u16(vld1_u16(i2)); i2 += 4;
        const float16x4_t vi3x1 = vreinterpret_f16_u16(vld1_u16(i3)); i3 += 4;
        const float16x4_t vi4x1 = vreinterpret_f16_u16(vld1_u16(i4)); i4 += 4;

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c0, vi0x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c0, vi2x0, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk00c0, vi0x1, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk00c0, vi2x1, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c0, vi0x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c0, vi2x0, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk00c0, vi0x1, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk00c0, vi2x1, 3);
#endif
        const float16x4_t vk10c0 = vreinterpret_f16_u16(vld1_u16(w + 8));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c0, vi1x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c0, vi3x0, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk10c0, vi1x1, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk10c0, vi3x1, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c0, vi1x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c0, vi3x0, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk10c0, vi1x1, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk10c0, vi3x1, 3);
#endif
        const float16x4_t vk20c0 = vreinterpret_f16_u16(vld1_u16(w + 12));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c0, vi2x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c0, vi4x0, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk20c0, vi2x1, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk20c0, vi4x1, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c0, vi2x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c0, vi4x0, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk20c0, vi2x1, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk20c0, vi4x1, 3);
#endif
        const float16x4_t vk00c1 = vreinterpret_f16_u16(vld1_u16(w + 16));

        // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
        const float16x4_t vi0x2 = vreinterpret_f16_u16(vld1_u16(i0)); i0 += 4;
        const float16x4_t vi1x2 = vreinterpret_f16_u16(vld1_u16(i1)); i1 += 4;
        const float16x4_t vi2x2 = vreinterpret_f16_u16(vld1_u16(i2)); i2 += 4;
        const float16x4_t vi3x2 = vreinterpret_f16_u16(vld1_u16(i3)); i3 += 4;
        const float16x4_t vi4x2 = vreinterpret_f16_u16(vld1_u16(i4)); i4 += 4;

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c1, vi0x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c1, vi2x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk00c1, vi0x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk00c1, vi2x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c1, vi0x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c1, vi2x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk00c1, vi0x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk00c1, vi2x2, 0);
#endif
        const float16x4_t vk10c1 = vreinterpret_f16_u16(vld1_u16(w + 20));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c1, vi1x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c1, vi3x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk10c1, vi1x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk10c1, vi3x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c1, vi1x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c1, vi3x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk10c1, vi1x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk10c1, vi3x2, 0);
#endif
        const float16x4_t vk20c1 = vreinterpret_f16_u16(vld1_u16(w + 24));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c1, vi2x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c1, vi4x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk20c1, vi2x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk20c1, vi4x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c1, vi2x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c1, vi4x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk20c1, vi2x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk20c1, vi4x2, 0);
#endif
        const float16x4_t vk00c2 = vreinterpret_f16_u16(vld1_u16(w + 28));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c2, vi0x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c2, vi2x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk00c2, vi0x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk00c2, vi2x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c2, vi0x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c2, vi2x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk00c2, vi0x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk00c2, vi2x2, 1);
#endif
        const float16x4_t vk10c2 = vreinterpret_f16_u16(vld1_u16(w + 32));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c2, vi1x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c2, vi3x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk10c2, vi1x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk10c2, vi3x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c2, vi1x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c2, vi3x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk10c2, vi1x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk10c2, vi3x2, 1);
#endif
        const float16x4_t vk20c2 = vreinterpret_f16_u16(vld1_u16(w + 36));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c2, vi2x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c2, vi4x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk20c2, vi2x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk20c2, vi4x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c2, vi2x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c2, vi4x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk20c2, vi2x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk20c2, vi4x2, 1);
#endif
        const float16x4_t vk01c0 = vreinterpret_f16_u16(vld1_u16(w + 40));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c0, vi0x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c0, vi2x1, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk01c0, vi0x2, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk01c0, vi2x2, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c0, vi0x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c0, vi2x1, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk01c0, vi0x2, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk01c0, vi2x2, 2);
#endif
        const float16x4_t vk11c0 = vreinterpret_f16_u16(vld1_u16(w + 44));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c0, vi1x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c0, vi3x1, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk11c0, vi1x2, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk11c0, vi3x2, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c0, vi1x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c0, vi3x1, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk11c0, vi1x2, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk11c0, vi3x2, 2);
#endif
        const float16x4_t vk21c0 = vreinterpret_f16_u16(vld1_u16(w + 48));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c0, vi2x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c0, vi4x1, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk21c0, vi2x2, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk21c0, vi4x2, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c0, vi2x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c0, vi4x1, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk21c0, vi2x2, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk21c0, vi4x2, 2);
#endif
        const float16x4_t vk01c1 = vreinterpret_f16_u16(vld1_u16(w + 52));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c1, vi0x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c1, vi2x1, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk01c1, vi0x2, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk01c1, vi2x2, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c1, vi0x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c1, vi2x1, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk01c1, vi0x2, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk01c1, vi2x2, 3);
#endif
        const float16x4_t vk11c1 = vreinterpret_f16_u16(vld1_u16(w + 56));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c1, vi1x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c1, vi3x1, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk11c1, vi1x2, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk11c1, vi3x2, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c1, vi1x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c1, vi3x1, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk11c1, vi1x2, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk11c1, vi3x2, 3);
#endif
        const float16x4_t vk21c1 = vreinterpret_f16_u16(vld1_u16(w + 60));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c1, vi2x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c1, vi4x1, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk21c1, vi2x2, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk21c1, vi4x2, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c1, vi2x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c1, vi4x1, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk21c1, vi2x2, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk21c1, vi4x2, 3);
#endif
        const float16x4_t vk01c2 = vreinterpret_f16_u16(vld1_u16(w + 64));

        // viMx3 = ( iM4c2, iM4c1, iM4c0, iM3c2 )
        const float16x4_t vi0x3 = vreinterpret_f16_u16(vld1_u16(i0)); i0 += 4;
        const float16x4_t vi1x3 = vreinterpret_f16_u16(vld1_u16(i1)); i1 += 4;
        const float16x4_t vi2x3 = vreinterpret_f16_u16(vld1_u16(i2)); i2 += 4;
        const float16x4_t vi3x3 = vreinterpret_f16_u16(vld1_u16(i3)); i3 += 4;
        const float16x4_t vi4x3 = vreinterpret_f16_u16(vld1_u16(i4)); i4 += 4;

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c2, vi0x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c2, vi2x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk01c2, vi0x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk01c2, vi2x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c2, vi0x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c2, vi2x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk01c2, vi0x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk01c2, vi2x3, 0);
#endif
        const float16x4_t vk11c2 = vreinterpret_f16_u16(vld1_u16(w + 68));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c2, vi1x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c2, vi3x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk11c2, vi1x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk11c2, vi3x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c2, vi1x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c2, vi3x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk11c2, vi1x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk11c2, vi3x3, 0);
#endif
        const float16x4_t vk21c2 = vreinterpret_f16_u16(vld1_u16(w + 72));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c2, vi2x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c2, vi4x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk21c2, vi2x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk21c2, vi4x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c2, vi2x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c2, vi4x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk21c2, vi2x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk21c2, vi4x3, 0);
#endif
        const float16x4_t vk02c0 = vreinterpret_f16_u16(vld1_u16(w + 76));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk02c0, vi0x1, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk02c0, vi2x1, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk02c0, vi0x3, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk02c0, vi2x3, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk02c0, vi0x1, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk02c0, vi2x1, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk02c0, vi0x3, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk02c0, vi2x3, 1);
#endif
        const float16x4_t vk12c0 = vreinterpret_f16_u16(vld1_u16(w + 80));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk12c0, vi1x1, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk12c0, vi3x1, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk12c0, vi1x3, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk12c0, vi3x3, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk12c0, vi1x1, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk12c0, vi3x1, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk12c0, vi1x3, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk12c0, vi3x3, 1);
#endif
        const float16x4_t vk22c0 = vreinterpret_f16_u16(vld1_u16(w + 84));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk22c0, vi2x1, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk22c0, vi4x1, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk22c0, vi2x3, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk22c0, vi4x3, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk22c0, vi2x1, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk22c0, vi4x1, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk22c0, vi2x3, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk22c0, vi4x3, 1);
#endif
        const float16x4_t vk02c1 = vreinterpret_f16_u16(vld1_u16(w + 88));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk02c1, vi0x2, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk02c1, vi2x2, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk02c1, vi0x3, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk02c1, vi2x3, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk02c1, vi0x2, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk02c1, vi2x2, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk02c1, vi0x3, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk02c1, vi2x3, 2);
#endif
        const float16x4_t vk12c1 = vreinterpret_f16_u16(vld1_u16(w + 92));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk12c1, vi1x2, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk12c1, vi3x2, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk12c1, vi1x3, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk12c1, vi3x3, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk12c1, vi1x2, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk12c1, vi3x2, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk12c1, vi1x3, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk12c1, vi3x3, 2);
#endif
        const float16x4_t vk22c1 = vreinterpret_f16_u16(vld1_u16(w + 96));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk22c1, vi2x2, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk22c1, vi4x2, 0);
        vo0x1 = vfma_lane_f16(vo0x1, vk22c1, vi2x3, 2);
        vo1x1 = vfma_lane_f16(vo1x1, vk22c1, vi4x3, 2);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk22c1, vi2x2, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk22c1, vi4x2, 0);
        vo0x1 = vmla_lane_f16(vo0x1, vk22c1, vi2x3, 2);
        vo1x1 = vmla_lane_f16(vo1x1, vk22c1, vi4x3, 2);
#endif
        const float16x4_t vk02c2 = vreinterpret_f16_u16(vld1_u16(w + 100));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk02c2, vi0x2, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk02c2, vi2x2, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk02c2, vi0x3, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk02c2, vi2x3, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk02c2, vi0x2, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk02c2, vi2x2, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk02c2, vi0x3, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk02c2, vi2x3, 3);
#endif
        const float16x4_t vk12c2 = vreinterpret_f16_u16(vld1_u16(w + 104));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk12c2, vi1x2, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk12c2, vi3x2, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk12c2, vi1x3, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk12c2, vi3x3, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk12c2, vi1x2, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk12c2, vi3x2, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk12c2, vi1x3, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk12c2, vi3x3, 3);
#endif
        const float16x4_t vk22c2 = vreinterpret_f16_u16(vld1_u16(w + 108));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk22c2, vi2x2, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk22c2, vi4x2, 1);
        vo0x1 = vfma_lane_f16(vo0x1, vk22c2, vi2x3, 3);
        vo1x1 = vfma_lane_f16(vo1x1, vk22c2, vi4x3, 3);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk22c2, vi2x2, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk22c2, vi4x2, 1);
        vo0x1 = vmla_lane_f16(vo0x1, vk22c2, vi2x3, 3);
        vo1x1 = vmla_lane_f16(vo1x1, vk22c2, vi4x3, 3);
#endif
        vi0x0 = vi0x3;
        vi1x0 = vi1x3;
        vi2x0 = vi2x3;
        vi3x0 = vi3x3;
        vi4x0 = vi4x3;

        vo0x0 = vmax_f16(vo0x0, vmin);
        vo1x0 = vmax_f16(vo1x0, vmin);
        vo0x1 = vmax_f16(vo0x1, vmin);
        vo1x1 = vmax_f16(vo1x1, vmin);

        vo0x0 = vmin_f16(vo0x0, vmax);
        vo1x0 = vmin_f16(vo1x0, vmax);
        vo0x1 = vmin_f16(vo0x1, vmax);
        vo1x1 = vmin_f16(vo1x1, vmax);

        const float16x4x2_t vo0c0123 = vzip_f16(vo0x0, vo0x1);
        const float16x4x2_t vo1c0123 = vzip_f16(vo1x0, vo1x1);

        // Always 2+ output width elements remaining
        vst1_lane_u32((void*) o1c0, vreinterpret_u32_f16(vo1c0123.val[0]), 0); o1c0 += 2;
        vst1_lane_u32((void*) o1c1, vreinterpret_u32_f16(vo1c0123.val[0]), 1); o1c1 += 2;
        vst1_lane_u32((void*) o1c2, vreinterpret_u32_f16(vo1c0123.val[1]), 0); o1c2 += 2;
        vst1_lane_u32((void*) o1c3, vreinterpret_u32_f16(vo1c0123.val[1]), 1); o1c3 += 2;

        vst1_lane_u32((void*) o0c0, vreinterpret_u32_f16(vo0c0123.val[0]), 0); o0c0 += 2;
        vst1_lane_u32((void*) o0c1, vreinterpret_u32_f16(vo0c0123.val[0]), 1); o0c1 += 2;
        vst1_lane_u32((void*) o0c2, vreinterpret_u32_f16(vo0c0123.val[1]), 0); o0c2 += 2;
        vst1_lane_u32((void*) o0c3, vreinterpret_u32_f16(vo0c0123.val[1]), 1); o0c3 += 2;
      }
      assert(iw < 4);
      if XNN_UNLIKELY(iw != 0) {
        float16x4_t vo0x0 = vreinterpret_f16_u16(vld1_u16(w));
        float16x4_t vo1x0 = vo0x0;
        float16x4_t vo0x1 = vo0x0;
        float16x4_t vo1x1 = vo0x0;

        const float16x4_t vk00c0 = vreinterpret_f16_u16(vld1_u16(w + 4));

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        float16x4_t vi0x1 = vreinterpret_f16_u16(vld1_u16(i0));
        float16x4_t vi1x1 = vreinterpret_f16_u16(vld1_u16(i1));
        float16x4_t vi2x1 = vreinterpret_f16_u16(vld1_u16(i2));
        float16x4_t vi3x1 = vreinterpret_f16_u16(vld1_u16(i3));
        float16x4_t vi4x1 = vreinterpret_f16_u16(vld1_u16(i4));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c0, vi0x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c0, vi2x0, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk00c0, vi0x1, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk00c0, vi2x1, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c0, vi0x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c0, vi2x0, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk00c0, vi0x1, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk00c0, vi2x1, 3);
        }
#endif
        const float16x4_t vk10c0 = vreinterpret_f16_u16(vld1_u16(w + 8));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c0, vi1x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c0, vi3x0, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk10c0, vi1x1, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk10c0, vi3x1, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c0, vi1x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c0, vi3x0, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk10c0, vi1x1, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk10c0, vi3x1, 3);
        }
#endif
        const float16x4_t vk20c0 = vreinterpret_f16_u16(vld1_u16(w + 12));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c0, vi2x0, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c0, vi4x0, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk20c0, vi2x1, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk20c0, vi4x1, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c0, vi2x0, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c0, vi4x0, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk20c0, vi2x1, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk20c0, vi4x1, 3);
        }
#endif
        const float16x4_t vk00c1 = vreinterpret_f16_u16(vld1_u16(w + 16));

        float16x4_t vi0x2 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi1x2 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi2x2 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi3x2 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi4x2 = vreinterpret_f16_u16(vmov_n_u16(0));
        if (iw >= 2) {
          // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
          vi0x2 = vreinterpret_f16_u16(vld1_u16(i0 + 4));
          vi1x2 = vreinterpret_f16_u16(vld1_u16(i1 + 4));
          vi2x2 = vreinterpret_f16_u16(vld1_u16(i2 + 4));
          vi3x2 = vreinterpret_f16_u16(vld1_u16(i3 + 4));
          vi4x2 = vreinterpret_f16_u16(vld1_u16(i4 + 4));
        }

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c1, vi0x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c1, vi2x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk00c1, vi0x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk00c1, vi2x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c1, vi0x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c1, vi2x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk00c1, vi0x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk00c1, vi2x2, 0);
#endif
        const float16x4_t vk10c1 = vreinterpret_f16_u16(vld1_u16(w + 20));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c1, vi1x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c1, vi3x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk10c1, vi1x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk10c1, vi3x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c1, vi1x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c1, vi3x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk10c1, vi1x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk10c1, vi3x2, 0);
#endif
        const float16x4_t vk20c1 = vreinterpret_f16_u16(vld1_u16(w + 24));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c1, vi2x0, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c1, vi4x0, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk20c1, vi2x2, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk20c1, vi4x2, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c1, vi2x0, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c1, vi4x0, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk20c1, vi2x2, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk20c1, vi4x2, 0);
#endif
        const float16x4_t vk00c2 = vreinterpret_f16_u16(vld1_u16(w + 28));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk00c2, vi0x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk00c2, vi2x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk00c2, vi0x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk00c2, vi2x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk00c2, vi0x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk00c2, vi2x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk00c2, vi0x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk00c2, vi2x2, 1);
#endif
        const float16x4_t vk10c2 = vreinterpret_f16_u16(vld1_u16(w + 32));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk10c2, vi1x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk10c2, vi3x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk10c2, vi1x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk10c2, vi3x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk10c2, vi1x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk10c2, vi3x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk10c2, vi1x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk10c2, vi3x2, 1);
#endif
        const float16x4_t vk20c2 = vreinterpret_f16_u16(vld1_u16(w + 36));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk20c2, vi2x0, 3);
        vo1x0 = vfma_lane_f16(vo1x0, vk20c2, vi4x0, 3);
        vo0x1 = vfma_lane_f16(vo0x1, vk20c2, vi2x2, 1);
        vo1x1 = vfma_lane_f16(vo1x1, vk20c2, vi4x2, 1);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk20c2, vi2x0, 3);
        vo1x0 = vmla_lane_f16(vo1x0, vk20c2, vi4x0, 3);
        vo0x1 = vmla_lane_f16(vo0x1, vk20c2, vi2x2, 1);
        vo1x1 = vmla_lane_f16(vo1x1, vk20c2, vi4x2, 1);
#endif
        const float16x4_t vk01c0 = vreinterpret_f16_u16(vld1_u16(w + 40));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c0, vi0x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c0, vi2x1, 0);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk01c0, vi0x2, 2);
          vo1x1 = vfma_lane_f16(vo1x1, vk01c0, vi2x2, 2);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c0, vi0x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c0, vi2x1, 0);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk01c0, vi0x2, 2);
          vo1x1 = vmla_lane_f16(vo1x1, vk01c0, vi2x2, 2);
        }
#endif
        const float16x4_t vk11c0 = vreinterpret_f16_u16(vld1_u16(w + 44));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c0, vi1x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c0, vi3x1, 0);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk11c0, vi1x2, 2);
          vo1x1 = vfma_lane_f16(vo1x1, vk11c0, vi3x2, 2);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c0, vi1x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c0, vi3x1, 0);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk11c0, vi1x2, 2);
          vo1x1 = vmla_lane_f16(vo1x1, vk11c0, vi3x2, 2);
        }
#endif
        const float16x4_t vk21c0 = vreinterpret_f16_u16(vld1_u16(w + 48));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c0, vi2x1, 0);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c0, vi4x1, 0);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk21c0, vi2x2, 2);
          vo1x1 = vfma_lane_f16(vo1x1, vk21c0, vi4x2, 2);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c0, vi2x1, 0);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c0, vi4x1, 0);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk21c0, vi2x2, 2);
          vo1x1 = vmla_lane_f16(vo1x1, vk21c0, vi4x2, 2);
        }
#endif
        const float16x4_t vk01c1 = vreinterpret_f16_u16(vld1_u16(w + 52));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c1, vi0x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c1, vi2x1, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk01c1, vi0x2, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk01c1, vi2x2, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c1, vi0x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c1, vi2x1, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk01c1, vi0x2, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk01c1, vi2x2, 3);
        }
#endif
        const float16x4_t vk11c1 = vreinterpret_f16_u16(vld1_u16(w + 56));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c1, vi1x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c1, vi3x1, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk11c1, vi1x2, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk11c1, vi3x2, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c1, vi1x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c1, vi3x1, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk11c1, vi1x2, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk11c1, vi3x2, 3);
        }
#endif
        const float16x4_t vk21c1 = vreinterpret_f16_u16(vld1_u16(w + 60));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c1, vi2x1, 1);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c1, vi4x1, 1);
        if (iw > 2) {
          vo0x1 = vfma_lane_f16(vo0x1, vk21c1, vi2x2, 3);
          vo1x1 = vfma_lane_f16(vo1x1, vk21c1, vi4x2, 3);
        }
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c1, vi2x1, 1);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c1, vi4x1, 1);
        if (iw > 2) {
          vo0x1 = vmla_lane_f16(vo0x1, vk21c1, vi2x2, 3);
          vo1x1 = vmla_lane_f16(vo1x1, vk21c1, vi4x2, 3);
        }
#endif
        const float16x4_t vk01c2 = vreinterpret_f16_u16(vld1_u16(w + 64));

        float16x4_t vi0x3 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi1x3 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi2x3 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi3x3 = vreinterpret_f16_u16(vmov_n_u16(0));
        float16x4_t vi4x3 = vreinterpret_f16_u16(vmov_n_u16(0));
        if (iw > 2) {
          // viMx3 = ( 0.0, 0.0, 0.0, iM3c2 )
          vi0x3 = vreinterpret_f16_u16(vld1_lane_u16(i0 + 8, vreinterpret_u16_f16(vi0x3), 0));
          vi1x3 = vreinterpret_f16_u16(vld1_lane_u16(i1 + 8, vreinterpret_u16_f16(vi1x3), 0));
          vi2x3 = vreinterpret_f16_u16(vld1_lane_u16(i2 + 8, vreinterpret_u16_f16(vi2x3), 0));
          vi3x3 = vreinterpret_f16_u16(vld1_lane_u16(i3 + 8, vreinterpret_u16_f16(vi3x3), 0));
          vi4x3 = vreinterpret_f16_u16(vld1_lane_u16(i4 + 8, vreinterpret_u16_f16(vi4x3), 0));
        }

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk01c2, vi0x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk01c2, vi2x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk01c2, vi0x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk01c2, vi2x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk01c2, vi0x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk01c2, vi2x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk01c2, vi0x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk01c2, vi2x3, 0);
#endif
        const float16x4_t vk11c2 = vreinterpret_f16_u16(vld1_u16(w + 68));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk11c2, vi1x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk11c2, vi3x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk11c2, vi1x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk11c2, vi3x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk11c2, vi1x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk11c2, vi3x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk11c2, vi1x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk11c2, vi3x3, 0);
#endif
        const float16x4_t vk21c2 = vreinterpret_f16_u16(vld1_u16(w + 72));

#if XNN_ARCH_ARM64
        vo0x0 = vfma_lane_f16(vo0x0, vk21c2, vi2x1, 2);
        vo1x0 = vfma_lane_f16(vo1x0, vk21c2, vi4x1, 2);
        vo0x1 = vfma_lane_f16(vo0x1, vk21c2, vi2x3, 0);
        vo1x1 = vfma_lane_f16(vo1x1, vk21c2, vi4x3, 0);
#else
        vo0x0 = vmla_lane_f16(vo0x0, vk21c2, vi2x1, 2);
        vo1x0 = vmla_lane_f16(vo1x0, vk21c2, vi4x1, 2);
        vo0x1 = vmla_lane_f16(vo0x1, vk21c2, vi2x3, 0);
        vo1x1 = vmla_lane_f16(vo1x1, vk21c2, vi4x3, 0);
#endif
        if (iw >= 2) {
          const float16x4_t vk02c0 = vreinterpret_f16_u16(vld1_u16(w + 76));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk02c0, vi0x1, 3);
          vo1x0 = vfma_lane_f16(vo1x0, vk02c0, vi2x1, 3);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk02c0, vi0x1, 3);
          vo1x0 = vmla_lane_f16(vo1x0, vk02c0, vi2x1, 3);
#endif
          const float16x4_t vk12c0 = vreinterpret_f16_u16(vld1_u16(w + 80));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk12c0, vi1x1, 3);
          vo1x0 = vfma_lane_f16(vo1x0, vk12c0, vi3x1, 3);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk12c0, vi1x1, 3);
          vo1x0 = vmla_lane_f16(vo1x0, vk12c0, vi3x1, 3);
#endif
          const float16x4_t vk22c0 = vreinterpret_f16_u16(vld1_u16(w + 84));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk22c0, vi2x1, 3);
          vo1x0 = vfma_lane_f16(vo1x0, vk22c0, vi4x1, 3);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk22c0, vi2x1, 3);
          vo1x0 = vmla_lane_f16(vo1x0, vk22c0, vi4x1, 3);
#endif
          const float16x4_t vk02c1 = vreinterpret_f16_u16(vld1_u16(w + 88));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk02c1, vi0x2, 0);
          vo1x0 = vfma_lane_f16(vo1x0, vk02c1, vi2x2, 0);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk02c1, vi0x2, 0);
          vo1x0 = vmla_lane_f16(vo1x0, vk02c1, vi2x2, 0);
#endif
          const float16x4_t vk12c1 = vreinterpret_f16_u16(vld1_u16(w + 92));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk12c1, vi1x2, 0);
          vo1x0 = vfma_lane_f16(vo1x0, vk12c1, vi3x2, 0);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk12c1, vi1x2, 0);
          vo1x0 = vmla_lane_f16(vo1x0, vk12c1, vi3x2, 0);
#endif
          const float16x4_t vk22c1 = vreinterpret_f16_u16(vld1_u16(w + 96));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk22c1, vi2x2, 0);
          vo1x0 = vfma_lane_f16(vo1x0, vk22c1, vi4x2, 0);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk22c1, vi2x2, 0);
          vo1x0 = vmla_lane_f16(vo1x0, vk22c1, vi4x2, 0);
#endif
          const float16x4_t vk02c2 = vreinterpret_f16_u16(vld1_u16(w + 100));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk02c2, vi0x2, 1);
          vo1x0 = vfma_lane_f16(vo1x0, vk02c2, vi2x2, 1);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk02c2, vi0x2, 1);
          vo1x0 = vmla_lane_f16(vo1x0, vk02c2, vi2x2, 1);
#endif
          const float16x4_t vk12c2 = vreinterpret_f16_u16(vld1_u16(w + 104));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk12c2, vi1x2, 1);
          vo1x0 = vfma_lane_f16(vo1x0, vk12c2, vi3x2, 1);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk12c2, vi1x2, 1);
          vo1x0 = vmla_lane_f16(vo1x0, vk12c2, vi3x2, 1);
#endif
          const float16x4_t vk22c2 = vreinterpret_f16_u16(vld1_u16(w + 108));

#if XNN_ARCH_ARM64
          vo0x0 = vfma_lane_f16(vo0x0, vk22c2, vi2x2, 1);
          vo1x0 = vfma_lane_f16(vo1x0, vk22c2, vi4x2, 1);
#else
          vo0x0 = vmla_lane_f16(vo0x0, vk22c2, vi2x2, 1);
          vo1x0 = vmla_lane_f16(vo1x0, vk22c2, vi4x2, 1);
#endif
        }

        vo0x0 = vmax_f16(vo0x0, vmin);
        vo1x0 = vmax_f16(vo1x0, vmin);
        vo0x1 = vmax_f16(vo0x1, vmin);
        vo1x1 = vmax_f16(vo1x1, vmin);

        vo0x0 = vmin_f16(vo0x0, vmax);
        vo1x0 = vmin_f16(vo1x0, vmax);
        vo0x1 = vmin_f16(vo0x1, vmax);
        vo1x1 = vmin_f16(vo1x1, vmax);

        if (iw == 3) {
          // Exactly 2 output width elements remaining
          const float16x4x2_t vo0c0123 = vzip_f16(vo0x0, vo0x1);
          const float16x4x2_t vo1c0123 = vzip_f16(vo1x0, vo1x1);

          // Always 2+ output width elements remaining
          vst1_lane_u32((void*) o1c0, vreinterpret_u32_f16(vo1c0123.val[0]), 0); o1c0 += 2;
          vst1_lane_u32((void*) o1c1, vreinterpret_u32_f16(vo1c0123.val[0]), 1); o1c1 += 2;
          vst1_lane_u32((void*) o1c2, vreinterpret_u32_f16(vo1c0123.val[1]), 0); o1c2 += 2;
          vst1_lane_u32((void*) o1c3, vreinterpret_u32_f16(vo1c0123.val[1]), 1); o1c3 += 2;

          vst1_lane_u32((void*) o0c0, vreinterpret_u32_f16(vo0c0123.val[0]), 0); o0c0 += 2;
          vst1_lane_u32((void*) o0c1, vreinterpret_u32_f16(vo0c0123.val[0]), 1); o0c1 += 2;
          vst1_lane_u32((void*) o0c2, vreinterpret_u32_f16(vo0c0123.val[1]), 0); o0c2 += 2;
          vst1_lane_u32((void*) o0c3, vreinterpret_u32_f16(vo0c0123.val[1]), 1); o0c3 += 2;
        } else {
          // Exactly 1 output width element remaining

          vst1_lane_u16(o1c0, vreinterpret_u16_f16(vo1x0), 0); o1c0 += 1;
          vst1_lane_u16(o1c1, vreinterpret_u16_f16(vo1x0), 1); o1c1 += 1;
          vst1_lane_u16(o1c2, vreinterpret_u16_f16(vo1x0), 2); o1c2 += 1;
          vst1_lane_u16(o1c3, vreinterpret_u16_f16(vo1x0), 3); o1c3 += 1;

          vst1_lane_u16(o0c0, vreinterpret_u16_f16(vo0x0), 0); o0c0 += 1;
          vst1_lane_u16(o0c1, vreinterpret_u16_f16(vo0x0), 1); o0c1 += 1;
          vst1_lane_u16(o0c2, vreinterpret_u16_f16(vo0x0), 2); o0c2 += 1;
          vst1_lane_u16(o0c3, vreinterpret_u16_f16(vo0x0), 3); o0c3 += 1;
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0c0 = (uint16_t*) ((uintptr_t) o0c0 + output_channel_increment);
      o0c1 = (uint16_t*) ((uintptr_t) o0c1 + output_channel_increment);
      o0c2 = (uint16_t*) ((uintptr_t) o0c2 + output_channel_increment);
      o0c3 = (uint16_t*) ((uintptr_t) o0c3 + output_channel_increment);
      o1c0 = (uint16_t*) ((uintptr_t) o1c0 + output_channel_increment);
      o1c1 = (uint16_t*) ((uintptr_t) o1c1 + output_channel_increment);
      o1c2 = (uint16_t*) ((uintptr_t) o1c2 + output_channel_increment);
      o1c3 = (uint16_t*) ((uintptr_t) o1c3 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const uint16_t*) ((uintptr_t) i0 - input_width_increment);
      i1 = (const uint16_t*) ((uintptr_t) i1 - input_width_increment);
      i2 = (const uint16_t*) ((uintptr_t) i2 - input_width_increment);
      i3 = (const uint16_t*) ((uintptr_t) i3 - input_width_increment);
      i4 = (const uint16_t*) ((uintptr_t) i4 - input_width_increment);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers forward to the next two rows
    output0 = (uint16_t*) ((uintptr_t) output1 + output_height_stride);
    output1 = (uint16_t*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next four rows
    i0 = i4;
    i1 = (const uint16_t*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const uint16_t*) ((uintptr_t) i1 + input_height_stride);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_height_stride);
    i4 = (const uint16_t*) ((uintptr_t) i3 + input_height_stride);
  }
}
