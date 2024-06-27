// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/argmaxpool.h"


void xnn_f32_argmaxpool_ukernel_9p8x__neon_c4(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* accumulation_buffer,
    uint32_t* index_buffer,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements > 9);
  assert(channels != 0);

  do {
    {
      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;

      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);

      for (size_t c = 0; c < channels; c += 4) {
        const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7); i7 += 4;
        const float32x4_t vi8 = vld1q_f32(i8); i8 += 4;

        float32x4_t vmax = vi0;
        uint32x4_t vidx = vmovq_n_u32(0);

        const uint32x4_t vm1 = vcgtq_f32(vi1, vmax);
        vmax = vbslq_f32(vm1, vi1, vmax);
        vidx = vbslq_u32(vm1, vmovq_n_u32(1), vidx);

        const uint32x4_t vm2 = vcgtq_f32(vi2, vmax);
        vmax = vbslq_f32(vm2, vi2, vmax);
        vidx = vbslq_u32(vm2, vmovq_n_u32(2), vidx);

        const uint32x4_t vm3 = vcgtq_f32(vi3, vmax);
        vmax = vbslq_f32(vm3, vi3, vmax);
        vidx = vbslq_u32(vm3, vmovq_n_u32(3), vidx);

        const uint32x4_t vm4 = vcgtq_f32(vi4, vmax);
        vmax = vbslq_f32(vm4, vi4, vmax);
        vidx = vbslq_u32(vm4, vmovq_n_u32(4), vidx);

        const uint32x4_t vm5 = vcgtq_f32(vi5, vmax);
        vmax = vbslq_f32(vm5, vi5, vmax);
        vidx = vbslq_u32(vm5, vmovq_n_u32(5), vidx);

        const uint32x4_t vm6 = vcgtq_f32(vi6, vmax);
        vmax = vbslq_f32(vm6, vi6, vmax);
        vidx = vbslq_u32(vm6, vmovq_n_u32(6), vidx);

        const uint32x4_t vm7 = vcgtq_f32(vi7, vmax);
        vmax = vbslq_f32(vm7, vi7, vmax);
        vidx = vbslq_u32(vm7, vmovq_n_u32(7), vidx);

        const uint32x4_t vm8 = vcgtq_f32(vi8, vmax);
        vmax = vbslq_f32(vm8, vi8, vmax);
        vidx = vbslq_u32(vm8, vmovq_n_u32(8), vidx);

        vst1q_f32(ab, vmax); ab += 4;
        vst1q_u32(ib, vidx); ib += 4;
      }
    }
    const uint32x4_t v1 = vmovq_n_u32(1);
    const uint32x4_t v8 = vmovq_n_u32(8);
    uint32x4_t vidx0 = vaddq_u32(v1, v8);

    size_t k = pooling_elements;
    for (k -= 9; k > 8; k -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);

      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;

      for (size_t c = 0; c < channels; c += 4) {
        const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7); i7 += 4;

        float32x4_t vmax = vld1q_f32(ab);
        uint32x4_t vidx = vld1q_u32(ib);

        const uint32x4_t vm0 = vcgtq_f32(vi0, vmax);
        vmax = vbslq_f32(vm0, vi0, vmax);
        vidx = vbslq_u32(vm0, vidx0, vidx);

        const uint32x4_t vm1 = vcgtq_f32(vi1, vmax);
        const uint32x4_t vidx1 = vaddq_u32(vidx0, v1);
        vmax = vbslq_f32(vm1, vi1, vmax);
        vidx = vbslq_u32(vm1, vidx1, vidx);

        const uint32x4_t vm2 = vcgtq_f32(vi2, vmax);
        const uint32x4_t vidx2 = vaddq_u32(vidx1, v1);
        vmax = vbslq_f32(vm2, vi2, vmax);
        vidx = vbslq_u32(vm2, vidx2, vidx);

        const uint32x4_t vm3 = vcgtq_f32(vi3, vmax);
        const uint32x4_t vidx3 = vaddq_u32(vidx2, v1);
        vmax = vbslq_f32(vm3, vi3, vmax);
        vidx = vbslq_u32(vm3, vidx3, vidx);

        const uint32x4_t vm4 = vcgtq_f32(vi4, vmax);
        const uint32x4_t vidx4 = vaddq_u32(vidx3, v1);
        vmax = vbslq_f32(vm4, vi4, vmax);
        vidx = vbslq_u32(vm4, vidx4, vidx);

        const uint32x4_t vm5 = vcgtq_f32(vi5, vmax);
        const uint32x4_t vidx5 = vaddq_u32(vidx4, v1);
        vmax = vbslq_f32(vm5, vi5, vmax);
        vidx = vbslq_u32(vm5, vidx5, vidx);

        const uint32x4_t vm6 = vcgtq_f32(vi6, vmax);
        const uint32x4_t vidx6 = vaddq_u32(vidx5, v1);
        vmax = vbslq_f32(vm6, vi6, vmax);
        vidx = vbslq_u32(vm6, vidx6, vidx);

        const uint32x4_t vm7 = vcgtq_f32(vi7, vmax);
        const uint32x4_t vidx7 = vaddq_u32(vidx6, v1);
        vmax = vbslq_f32(vm7, vi7, vmax);
        vidx = vbslq_u32(vm7, vidx7, vidx);

        vst1q_f32(ab, vmax); ab += 4;
        vst1q_u32(ib, vidx); ib += 4;
      }
      vidx0 = vaddq_u32(vidx0, v8);
    }

    float* o = output;
    uint32_t* i = index;
    {
      const float* i0 = input[0];
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      input = (const float**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k != 8) {
        i7 = i0;
      }

      size_t c = channels;
      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;
      for (; c >= 4; c -= 4) {
        const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7); i7 += 4;

        float32x4_t vmax = vld1q_f32(ab); ab += 4;
        uint32x4_t vidx = vld1q_u32(ib); ib += 4;

        const uint32x4_t vm0 = vcgtq_f32(vi0, vmax);
        vmax = vbslq_f32(vm0, vi0, vmax);
        vidx = vbslq_u32(vm0, vidx0, vidx);

        const uint32x4_t vm1 = vcgtq_f32(vi1, vmax);
        const uint32x4_t vidx1 = vaddq_u32(vidx0, v1);
        vmax = vbslq_f32(vm1, vi1, vmax);
        vidx = vbslq_u32(vm1, vidx1, vidx);

        const uint32x4_t vm2 = vcgtq_f32(vi2, vmax);
        const uint32x4_t vidx2 = vaddq_u32(vidx1, v1);
        vmax = vbslq_f32(vm2, vi2, vmax);
        vidx = vbslq_u32(vm2, vidx2, vidx);

        const uint32x4_t vm3 = vcgtq_f32(vi3, vmax);
        const uint32x4_t vidx3 = vaddq_u32(vidx2, v1);
        vmax = vbslq_f32(vm3, vi3, vmax);
        vidx = vbslq_u32(vm3, vidx3, vidx);

        const uint32x4_t vm4 = vcgtq_f32(vi4, vmax);
        const uint32x4_t vidx4 = vaddq_u32(vidx3, v1);
        vmax = vbslq_f32(vm4, vi4, vmax);
        vidx = vbslq_u32(vm4, vidx4, vidx);

        const uint32x4_t vm5 = vcgtq_f32(vi5, vmax);
        const uint32x4_t vidx5 = vaddq_u32(vidx4, v1);
        vmax = vbslq_f32(vm5, vi5, vmax);
        vidx = vbslq_u32(vm5, vidx5, vidx);

        const uint32x4_t vm6 = vcgtq_f32(vi6, vmax);
        const uint32x4_t vidx6 = vaddq_u32(vidx5, v1);
        vmax = vbslq_f32(vm6, vi6, vmax);
        vidx = vbslq_u32(vm6, vidx6, vidx);

        const uint32x4_t vm7 = vcgtq_f32(vi7, vmax);
        const uint32x4_t vidx7 = vaddq_u32(vidx6, v1);
        vmax = vbslq_f32(vm7, vi7, vmax);
        vidx = vbslq_u32(vm7, vidx7, vidx);

        vst1q_f32(o, vmax); o += 4;
        vst1q_u32(i, vidx); i += 4;
      }
      if (c != 0) {
        const float32x4_t vi0 = vld1q_f32(i0);
        const float32x4_t vi1 = vld1q_f32(i1);
        const float32x4_t vi2 = vld1q_f32(i2);
        const float32x4_t vi3 = vld1q_f32(i3);
        const float32x4_t vi4 = vld1q_f32(i4);
        const float32x4_t vi5 = vld1q_f32(i5);
        const float32x4_t vi6 = vld1q_f32(i6);
        const float32x4_t vi7 = vld1q_f32(i7);

        float32x4_t vmax = vld1q_f32(ab);
        uint32x4_t vidx = vld1q_u32(ib);

        const uint32x4_t vm0 = vcgtq_f32(vi0, vmax);
        vmax = vbslq_f32(vm0, vi0, vmax);
        vidx = vbslq_u32(vm0, vidx0, vidx);

        const uint32x4_t vm1 = vcgtq_f32(vi1, vmax);
        const uint32x4_t vidx1 = vaddq_u32(vidx0, v1);
        vmax = vbslq_f32(vm1, vi1, vmax);
        vidx = vbslq_u32(vm1, vidx1, vidx);

        const uint32x4_t vm2 = vcgtq_f32(vi2, vmax);
        const uint32x4_t vidx2 = vaddq_u32(vidx1, v1);
        vmax = vbslq_f32(vm2, vi2, vmax);
        vidx = vbslq_u32(vm2, vidx2, vidx);

        const uint32x4_t vm3 = vcgtq_f32(vi3, vmax);
        const uint32x4_t vidx3 = vaddq_u32(vidx2, v1);
        vmax = vbslq_f32(vm3, vi3, vmax);
        vidx = vbslq_u32(vm3, vidx3, vidx);

        const uint32x4_t vm4 = vcgtq_f32(vi4, vmax);
        const uint32x4_t vidx4 = vaddq_u32(vidx3, v1);
        vmax = vbslq_f32(vm4, vi4, vmax);
        vidx = vbslq_u32(vm4, vidx4, vidx);

        const uint32x4_t vm5 = vcgtq_f32(vi5, vmax);
        const uint32x4_t vidx5 = vaddq_u32(vidx4, v1);
        vmax = vbslq_f32(vm5, vi5, vmax);
        vidx = vbslq_u32(vm5, vidx5, vidx);

        const uint32x4_t vm6 = vcgtq_f32(vi6, vmax);
        const uint32x4_t vidx6 = vaddq_u32(vidx5, v1);
        vmax = vbslq_f32(vm6, vi6, vmax);
        vidx = vbslq_u32(vm6, vidx6, vidx);

        const uint32x4_t vm7 = vcgtq_f32(vi7, vmax);
        const uint32x4_t vidx7 = vaddq_u32(vidx6, v1);
        vmax = vbslq_f32(vm7, vi7, vmax);
        vidx = vbslq_u32(vm7, vidx7, vidx);

        float32x2_t vmax_lo = vget_low_f32(vmax);
        uint32x2_t vidx_lo = vget_low_u32(vidx);
        if (c & 2) {
          vst1_f32(o, vmax_lo); o += 2;
          vst1_u32(i, vidx_lo); i += 2;
          vmax_lo = vget_high_f32(vmax);
          vidx_lo = vget_high_u32(vidx);
        }
        if (c & 1) {
          vst1_lane_f32(o, vmax_lo, 0); o += 1;
          vst1_lane_u32(i, vidx_lo, 0); i += 1;
        }
      }
    }

    output = (float*) ((uintptr_t) o + output_increment);
    index = (uint32_t*) i;
  } while (--output_pixels != 0);
}
