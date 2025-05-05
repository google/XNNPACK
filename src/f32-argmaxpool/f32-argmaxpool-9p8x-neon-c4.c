// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/argmaxpool.h"
#include "src/xnnpack/simd/f32-neon.h"

static XNN_INLINE uint32x4_t
xnn_load_tail_safe_u32(const uint32_t* input, size_t num_elements) {
  return vreinterpretq_u32_f32(xnn_load_tail_safe_f32((const float*) input, num_elements));
}

void xnn_f32_argmaxpool_ukernel_9p8x__neon_c4(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    size_t input_pixel_stride,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    size_t index_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(channels != 0);

  const uint32x4_t v1 = vdupq_n_u32(1);

  do {
    // Accumulators start out null, after each pass the accumulator is set to
    // the output.
    const float* ab = NULL;
    const uint32_t* ib = NULL;
    const float** id = input;

    uint32x4_t vidx0 = vdupq_n_u32(0);
    uint32x4_t vidx8;

    assert(!ab);
    assert(!ib);

    ptrdiff_t k = pooling_elements;
    for (; k > 0; k -= 9) {
      const float* i0 = *id++;
      const float* i1 = 1 < k ? *id++ : i0;
      const float* i2 = 2 < k ? *id++ : i0;
      const float* i3 = 3 < k ? *id++ : i0;
      const float* i4 = 4 < k ? *id++ : i0;
      const float* i5 = 5 < k ? *id++ : i0;
      const float* i6 = 6 < k ? *id++ : i0;
      const float* i7 = 7 < k ? *id++ : i0;
      const float* i8 = 8 < k ? *id++ : i0;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);

      float* o = output;
      uint32_t* i = index;
      size_t c = channels;
      for (; c >= 4; c -= 4) {
        const float32x4_t vi0 = vld1q_f32(i0);
        i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1);
        i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2);
        i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3);
        i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4);
        i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5);
        i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6);
        i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7);
        i7 += 4;
        const float32x4_t vi8 = vld1q_f32(i8);
        i8 += 4;

        float32x4_t vmax;
        uint32x4_t vidx;
        if (ab) {
          vmax = vld1q_f32(ab); ab += 4;
          vidx = vld1q_u32(ib); ib += 4;

          const uint32x4_t vm0 = vcgtq_f32(vi0, vmax);
          vmax = vbslq_f32(vm0, vi0, vmax);
          vidx = vbslq_u32(vm0, vidx0, vidx);
        } else {
          vmax = vi0;
          vidx = vidx0;
        }

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

        const uint32x4_t vm8 = vcgtq_f32(vi8, vmax);
        vidx8 = vaddq_u32(vidx7, v1);
        vmax = vbslq_f32(vm8, vi8, vmax);
        vidx = vbslq_u32(vm8, vidx8, vidx);

        vst1q_f32(o, vmax);
        o += 4;
        vst1q_u32(i, vidx);
        i += 4;
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

        float32x4_t vmax;
        uint32x4_t vidx;
        if (ab) {
          vmax = xnn_load_tail_safe_f32(ab, c);
          vidx = xnn_load_tail_safe_u32(ib, c);

          const uint32x4_t vm0 = vcgtq_f32(vi0, vmax);
          vmax = vbslq_f32(vm0, vi0, vmax);
          vidx = vbslq_u32(vm0, vidx0, vidx);
        } else {
          vmax = vi0;
          vidx = vidx0;
        }

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

        const uint32x4_t vm8 = vcgtq_f32(vi8, vmax);
        vidx8 = vaddq_u32(vidx7, v1);
        vmax = vbslq_f32(vm8, vi8, vmax);
        vidx = vbslq_u32(vm8, vidx8, vidx);

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
      vidx0 = vaddq_u32(vidx8, v1);
      ab = output;
      ib = index;
    }

    input = (const float**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
    index = (uint32_t*) ((uintptr_t) index + index_increment);
  } while (--output_pixels != 0);
}
