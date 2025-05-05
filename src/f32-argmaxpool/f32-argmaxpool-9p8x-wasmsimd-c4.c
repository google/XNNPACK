// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/argmaxpool.h"
#include "src/xnnpack/simd/f32-wasmsimd.h"

static XNN_INLINE v128_t
xnn_load_tail_safe_u32(const uint32_t* input, size_t num_elements) {
  return xnn_load_tail_safe_f32((const float*) input, num_elements);
}

void xnn_f32_argmaxpool_ukernel_9p8x__wasmsimd_c4(
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

  const v128_t v1 = wasm_i32x4_const_splat(1);

  do {
    // Accumulators start out null, after each pass the accumulator is set to
    // the output.
    const float* ab = NULL;
    const uint32_t* ib = NULL;
    const float** id = input;

    v128_t vidx0 = wasm_i32x4_const_splat(0);
    v128_t vidx8;

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
        const v128_t vi0 = wasm_v128_load(i0);
        i0 += 4;
        const v128_t vi1 = wasm_v128_load(i1);
        i1 += 4;
        const v128_t vi2 = wasm_v128_load(i2);
        i2 += 4;
        const v128_t vi3 = wasm_v128_load(i3);
        i3 += 4;
        const v128_t vi4 = wasm_v128_load(i4);
        i4 += 4;
        const v128_t vi5 = wasm_v128_load(i5);
        i5 += 4;
        const v128_t vi6 = wasm_v128_load(i6);
        i6 += 4;
        const v128_t vi7 = wasm_v128_load(i7);
        i7 += 4;
        const v128_t vi8 = wasm_v128_load(i8);
        i8 += 4;

        v128_t vmax;
        v128_t vidx;
        if (ab) {
          vmax = wasm_v128_load(ab);
          ab += 4;
          vidx = wasm_v128_load(ib);
          ib += 4;

          const v128_t vm0 = wasm_f32x4_gt(vi0, vmax);
          vmax = wasm_v128_bitselect(vi0, vmax, vm0);
          vidx = wasm_v128_bitselect(vidx0, vidx, vm0);
        } else {
          vmax = vi0;
          vidx = vidx0;
        }

        const v128_t vm1 = wasm_f32x4_gt(vi1, vmax);
        const v128_t vidx1 = wasm_i32x4_add(vidx0, v1);
        vmax = wasm_v128_bitselect(vi1, vmax, vm1);
        vidx = wasm_v128_bitselect(vidx1, vidx, vm1);

        const v128_t vm2 = wasm_f32x4_gt(vi2, vmax);
        const v128_t vidx2 = wasm_i32x4_add(vidx1, v1);
        vmax = wasm_v128_bitselect(vi2, vmax, vm2);
        vidx = wasm_v128_bitselect(vidx2, vidx, vm2);

        const v128_t vm3 = wasm_f32x4_gt(vi3, vmax);
        const v128_t vidx3 = wasm_i32x4_add(vidx2, v1);
        vmax = wasm_v128_bitselect(vi3, vmax, vm3);
        vidx = wasm_v128_bitselect(vidx3, vidx, vm3);

        const v128_t vm4 = wasm_f32x4_gt(vi4, vmax);
        const v128_t vidx4 = wasm_i32x4_add(vidx3, v1);
        vmax = wasm_v128_bitselect(vi4, vmax, vm4);
        vidx = wasm_v128_bitselect(vidx4, vidx, vm4);

        const v128_t vm5 = wasm_f32x4_gt(vi5, vmax);
        const v128_t vidx5 = wasm_i32x4_add(vidx4, v1);
        vmax = wasm_v128_bitselect(vi5, vmax, vm5);
        vidx = wasm_v128_bitselect(vidx5, vidx, vm5);

        const v128_t vm6 = wasm_f32x4_gt(vi6, vmax);
        const v128_t vidx6 = wasm_i32x4_add(vidx5, v1);
        vmax = wasm_v128_bitselect(vi6, vmax, vm6);
        vidx = wasm_v128_bitselect(vidx6, vidx, vm6);

        const v128_t vm7 = wasm_f32x4_gt(vi7, vmax);
        const v128_t vidx7 = wasm_i32x4_add(vidx6, v1);
        vmax = wasm_v128_bitselect(vi7, vmax, vm7);
        vidx = wasm_v128_bitselect(vidx7, vidx, vm7);

        const v128_t vm8 = wasm_f32x4_gt(vi8, vmax);
        vidx8 = wasm_i32x4_add(vidx7, v1);
        vmax = wasm_v128_bitselect(vi8, vmax, vm8);
        vidx = wasm_v128_bitselect(vidx8, vidx, vm8);

        wasm_v128_store(o, vmax);
        o += 4;
        wasm_v128_store(i, vidx);
        i += 4;
      }
      if (c != 0) {
        const v128_t vi0 = wasm_v128_load(i0);
        const v128_t vi1 = wasm_v128_load(i1);
        const v128_t vi2 = wasm_v128_load(i2);
        const v128_t vi3 = wasm_v128_load(i3);
        const v128_t vi4 = wasm_v128_load(i4);
        const v128_t vi5 = wasm_v128_load(i5);
        const v128_t vi6 = wasm_v128_load(i6);
        const v128_t vi7 = wasm_v128_load(i7);
        const v128_t vi8 = wasm_v128_load(i8);

        v128_t vmax;
        v128_t vidx;
        if (ab) {
          vmax = xnn_load_tail_safe_f32(ab, c);
          vidx = xnn_load_tail_safe_u32(ib, c);

          const v128_t vm0 = wasm_f32x4_gt(vi0, vmax);
          vmax = wasm_v128_bitselect(vi0, vmax, vm0);
          vidx = wasm_v128_bitselect(vidx0, vidx, vm0);
        } else {
          vmax = vi0;
          vidx = vidx0;
        }

        const v128_t vm1 = wasm_f32x4_gt(vi1, vmax);
        const v128_t vidx1 = wasm_i32x4_add(vidx0, v1);
        vmax = wasm_v128_bitselect(vi1, vmax, vm1);
        vidx = wasm_v128_bitselect(vidx1, vidx, vm1);

        const v128_t vm2 = wasm_f32x4_gt(vi2, vmax);
        const v128_t vidx2 = wasm_i32x4_add(vidx1, v1);
        vmax = wasm_v128_bitselect(vi2, vmax, vm2);
        vidx = wasm_v128_bitselect(vidx2, vidx, vm2);

        const v128_t vm3 = wasm_f32x4_gt(vi3, vmax);
        const v128_t vidx3 = wasm_i32x4_add(vidx2, v1);
        vmax = wasm_v128_bitselect(vi3, vmax, vm3);
        vidx = wasm_v128_bitselect(vidx3, vidx, vm3);

        const v128_t vm4 = wasm_f32x4_gt(vi4, vmax);
        const v128_t vidx4 = wasm_i32x4_add(vidx3, v1);
        vmax = wasm_v128_bitselect(vi4, vmax, vm4);
        vidx = wasm_v128_bitselect(vidx4, vidx, vm4);

        const v128_t vm5 = wasm_f32x4_gt(vi5, vmax);
        const v128_t vidx5 = wasm_i32x4_add(vidx4, v1);
        vmax = wasm_v128_bitselect(vi5, vmax, vm5);
        vidx = wasm_v128_bitselect(vidx5, vidx, vm5);

        const v128_t vm6 = wasm_f32x4_gt(vi6, vmax);
        const v128_t vidx6 = wasm_i32x4_add(vidx5, v1);
        vmax = wasm_v128_bitselect(vi6, vmax, vm6);
        vidx = wasm_v128_bitselect(vidx6, vidx, vm6);

        const v128_t vm7 = wasm_f32x4_gt(vi7, vmax);
        const v128_t vidx7 = wasm_i32x4_add(vidx6, v1);
        vmax = wasm_v128_bitselect(vi7, vmax, vm7);
        vidx = wasm_v128_bitselect(vidx7, vidx, vm7);

        const v128_t vm8 = wasm_f32x4_gt(vi8, vmax);
        vidx8 = wasm_i32x4_add(vidx7, v1);
        vmax = wasm_v128_bitselect(vi8, vmax, vm8);
        vidx = wasm_v128_bitselect(vidx8, vidx, vm8);

        if (c & 2) {
          wasm_v128_store64_lane(o, vmax, 0);
          wasm_v128_store64_lane(i, vidx, 0);
          vmax = wasm_v64x2_shuffle(vmax, vmax, 1, 1);
          vidx = wasm_v64x2_shuffle(vidx, vidx, 1, 1);
          o += 2;
          i += 2;
        }
        if (c & 1) {
          wasm_v128_store32_lane(o, vmax, 0);
          wasm_v128_store32_lane(i, vidx, 0);
          o += 1;
          i += 1;
        }
      }
      vidx0 = wasm_i32x4_add(vidx8, v1);
      ab = output;
      ib = index;
    }

    input = (const float**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
    index = (uint32_t*) ((uintptr_t) index + index_increment);
  } while (--output_pixels != 0);
}
