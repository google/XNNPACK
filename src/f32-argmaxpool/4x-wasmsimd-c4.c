// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/argmaxpool.h>


void xnn_f32_argmaxpool_ukernel_4x__wasmsimd_c4(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index_ptr,
    size_t input_increment,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements <= 4);
  assert(channels != 0);

  float* index = (float*) index_ptr;
  do {
    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    i0 = (const float*) ((uintptr_t) i0 + input_offset);
    i1 = (const float*) ((uintptr_t) i1 + input_offset);
    i2 = (const float*) ((uintptr_t) i2 + input_offset);
    i3 = (const float*) ((uintptr_t) i3 + input_offset);
    if (pooling_elements < 2) {
      i1 = i0;
    }
    if (pooling_elements <= 2) {
      i2 = i0;
    }
    if (pooling_elements != 4) {
      i3 = i0;
    }

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

      wasm_v128_store(output, vmax);
      output += 4;
      wasm_v128_store(index, vidx);
      index += 4;
    }
    if (c != 0) {
      const v128_t vi0 = wasm_v128_load(i0);
      const v128_t vi1 = wasm_v128_load(i1);
      const v128_t vi2 = wasm_v128_load(i2);
      const v128_t vi3 = wasm_v128_load(i3);

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

      if (c & 2) {
        wasm_v128_store64_lane(output, vmax, 0);
        wasm_v128_store64_lane(index, vidx, 0);
        vmax = wasm_v64x2_shuffle(vmax, vmax, 1, 1);
        vidx = wasm_v64x2_shuffle(vidx, vidx, 1, 1);
        output += 2;
        index += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vmax, 0);
        wasm_v128_store32_lane(index, vidx, 0);
        output += 1;
        index += 1;
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
