// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/argmaxpool.h>


void xnn_f32_argmaxpool_ukernel_9p8x__wasmsimd_c4(
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

        v128_t vmax = vi0;
        v128_t vidx = wasm_i32x4_const_splat(0);

        const v128_t vm1 = wasm_f32x4_gt(vi1, vmax);
        vmax = wasm_v128_bitselect(vi1, vmax, vm1);
        vidx = wasm_v128_bitselect(wasm_i32x4_const_splat(1), vidx, vm1);

        const v128_t vm2 = wasm_f32x4_gt(vi2, vmax);
        vmax = wasm_v128_bitselect(vi2, vmax, vm2);
        vidx = wasm_v128_bitselect(wasm_i32x4_const_splat(2), vidx, vm2);

        const v128_t vm3 = wasm_f32x4_gt(vi3, vmax);
        vmax = wasm_v128_bitselect(vi3, vmax, vm3);
        vidx = wasm_v128_bitselect(wasm_i32x4_const_splat(3), vidx, vm3);

        const v128_t vm4 = wasm_f32x4_gt(vi4, vmax);
        vmax = wasm_v128_bitselect(vi4, vmax, vm4);
        vidx = wasm_v128_bitselect(wasm_i32x4_const_splat(4), vidx, vm4);

        const v128_t vm5 = wasm_f32x4_gt(vi5, vmax);
        vmax = wasm_v128_bitselect(vi5, vmax, vm5);
        vidx = wasm_v128_bitselect(wasm_i32x4_const_splat(5), vidx, vm5);

        const v128_t vm6 = wasm_f32x4_gt(vi6, vmax);
        vmax = wasm_v128_bitselect(vi6, vmax, vm6);
        vidx = wasm_v128_bitselect(wasm_i32x4_const_splat(6), vidx, vm6);

        const v128_t vm7 = wasm_f32x4_gt(vi7, vmax);
        vmax = wasm_v128_bitselect(vi7, vmax, vm7);
        vidx = wasm_v128_bitselect(wasm_i32x4_const_splat(7), vidx, vm7);

        const v128_t vm8 = wasm_f32x4_gt(vi8, vmax);
        vmax = wasm_v128_bitselect(vi8, vmax, vm8);
        vidx = wasm_v128_bitselect(wasm_i32x4_const_splat(8), vidx, vm8);

        wasm_v128_store(ab, vmax);
        ab += 4;
        wasm_v128_store(ib, vidx);
        ib += 4;
      }
    }
    const v128_t v1 = wasm_i32x4_const_splat(1);
    const v128_t v8 = wasm_i32x4_const_splat(8);
    v128_t vidx0 = wasm_i32x4_add(v1, v8);

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

        v128_t vmax = wasm_v128_load(ab);
        v128_t vidx = wasm_v128_load(ib);

        const v128_t vm0 = wasm_f32x4_gt(vi0, vmax);
        vmax = wasm_v128_bitselect(vi0, vmax, vm0);
        vidx = wasm_v128_bitselect(vidx0, vidx, vm0);

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

        wasm_v128_store(ab, vmax);
        ab += 4;
        wasm_v128_store(ib, vidx);
        ib += 4;
      }
      vidx0 = wasm_i32x4_add(vidx0, v8);
    }

    float* o = output;
    float* i = (float*) index;
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

        v128_t vmax = wasm_v128_load(ab);
        ab += 4;
        v128_t vidx = wasm_v128_load(ib);
        ib += 4;

        const v128_t vm0 = wasm_f32x4_gt(vi0, vmax);
        vmax = wasm_v128_bitselect(vi0, vmax, vm0);
        vidx = wasm_v128_bitselect(vidx0, vidx, vm0);

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

        v128_t vmax = wasm_v128_load(ab);
        v128_t vidx = wasm_v128_load(ib);

        const v128_t vm0 = wasm_f32x4_gt(vi0, vmax);
        vmax = wasm_v128_bitselect(vi0, vmax, vm0);
        vidx = wasm_v128_bitselect(vidx0, vidx, vm0);

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
    }

    output = (float*) ((uintptr_t) o + output_increment);
    index = (uint32_t*) i;
  } while (--output_pixels != 0);
}
