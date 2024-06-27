// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/argmaxpool.h"


void xnn_f32_argmaxpool_ukernel_9x__neon_c4(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements <= 9);
  assert(channels != 0);

  do {
    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    i0 = (const float*) ((uintptr_t) i0 + input_offset);
    i1 = (const float*) ((uintptr_t) i1 + input_offset);
    i2 = (const float*) ((uintptr_t) i2 + input_offset);
    i3 = (const float*) ((uintptr_t) i3 + input_offset);
    i4 = (const float*) ((uintptr_t) i4 + input_offset);
    i5 = (const float*) ((uintptr_t) i5 + input_offset);
    i6 = (const float*) ((uintptr_t) i6 + input_offset);
    i7 = (const float*) ((uintptr_t) i7 + input_offset);
    i8 = (const float*) ((uintptr_t) i8 + input_offset);
    if (pooling_elements < 2) {
      i1 = i0;
    }
    if (pooling_elements <= 2) {
      i2 = i0;
    }
    if (pooling_elements < 4) {
      i3 = i0;
    }
    if (pooling_elements <= 4) {
      i4 = i0;
    }
    if (pooling_elements < 6) {
      i5 = i0;
    }
    if (pooling_elements <= 6) {
      i6 = i0;
    }
    if (pooling_elements < 8) {
      i7 = i0;
    }
    if (pooling_elements <= 8) {
      i8 = i0;
    }

    size_t c = channels;
    for (; c >= 4; c -= 4) {
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

      vst1q_f32(output, vmax); output += 4;
      vst1q_u32(index, vidx); index += 4;
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
      const float32x4_t vi8 = vld1q_f32(i8);

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

      float32x2_t vmax_lo = vget_low_f32(vmax);
      uint32x2_t vidx_lo = vget_low_u32(vidx);
      if (c & 2) {
        vst1_f32(output, vmax_lo); output += 2;
        vst1_u32(index, vidx_lo); index += 2;
        vmax_lo = vget_high_f32(vmax);
        vidx_lo = vget_high_u32(vidx);
      }
      if (c & 1) {
        vst1_lane_f32(output, vmax_lo, 0); output += 1;
        vst1_lane_u32(index, vidx_lo, 0); index += 1;
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
