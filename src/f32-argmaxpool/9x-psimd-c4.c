// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/argmaxpool.h>


void xnn_f32_argmaxpool_ukernel_9x__psimd_c4(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements <= 9);
  assert(channels != 0);

  const psimd_f32 voutput_max = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 voutput_min = psimd_load_splat_f32(&params->scalar.min);
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
      const psimd_f32 vi0 = psimd_load_f32(i0);
      i0 += 4;
      const psimd_f32 vi1 = psimd_load_f32(i1);
      i1 += 4;
      const psimd_f32 vi2 = psimd_load_f32(i2);
      i2 += 4;
      const psimd_f32 vi3 = psimd_load_f32(i3);
      i3 += 4;
      const psimd_f32 vi4 = psimd_load_f32(i4);
      i4 += 4;
      const psimd_f32 vi5 = psimd_load_f32(i5);
      i5 += 4;
      const psimd_f32 vi6 = psimd_load_f32(i6);
      i6 += 4;
      const psimd_f32 vi7 = psimd_load_f32(i7);
      i7 += 4;
      const psimd_f32 vi8 = psimd_load_f32(i8);
      i8 += 4;

      psimd_f32 vmax = vi0;
      psimd_u32 vidx = psimd_splat_u32(0);

      const psimd_s32 vm1 = vi1 > vmax;
      vmax = psimd_blend_f32(vm1, vi1, vmax);
      vidx = psimd_blend_u32(vm1, psimd_splat_u32(1), vidx);

      const psimd_s32 vm2 = vi2 > vmax;
      vmax = psimd_blend_f32(vm2, vi2, vmax);
      vidx = psimd_blend_u32(vm2, psimd_splat_u32(2), vidx);

      const psimd_s32 vm3 = vi3 > vmax;
      vmax = psimd_blend_f32(vm3, vi3, vmax);
      vidx = psimd_blend_u32(vm3, psimd_splat_u32(3), vidx);

      const psimd_s32 vm4 = vi4 > vmax;
      vmax = psimd_blend_f32(vm4, vi4, vmax);
      vidx = psimd_blend_u32(vm4, psimd_splat_u32(4), vidx);

      const psimd_s32 vm5 = vi5 > vmax;
      vmax = psimd_blend_f32(vm5, vi5, vmax);
      vidx = psimd_blend_u32(vm5, psimd_splat_u32(5), vidx);

      const psimd_s32 vm6 = vi6 > vmax;
      vmax = psimd_blend_f32(vm6, vi6, vmax);
      vidx = psimd_blend_u32(vm6, psimd_splat_u32(6), vidx);

      const psimd_s32 vm7 = vi7 > vmax;
      vmax = psimd_blend_f32(vm7, vi7, vmax);
      vidx = psimd_blend_u32(vm7, psimd_splat_u32(7), vidx);

      const psimd_s32 vm8 = vi8 > vmax;
      vmax = psimd_blend_f32(vm8, vi8, vmax);
      vidx = psimd_blend_u32(vm8, psimd_splat_u32(8), vidx);

      const psimd_f32 vout = psimd_max_f32(psimd_min_f32(vmax, voutput_max), voutput_min);

      psimd_store_f32(output, vout);
      output += 4;
      psimd_store_u32(index, vidx);
      index += 4;
    }
    if (c != 0) {
      const psimd_f32 vi0 = psimd_load_f32(i0);
      const psimd_f32 vi1 = psimd_load_f32(i1);
      const psimd_f32 vi2 = psimd_load_f32(i2);
      const psimd_f32 vi3 = psimd_load_f32(i3);
      const psimd_f32 vi4 = psimd_load_f32(i4);
      const psimd_f32 vi5 = psimd_load_f32(i5);
      const psimd_f32 vi6 = psimd_load_f32(i6);
      const psimd_f32 vi7 = psimd_load_f32(i7);
      const psimd_f32 vi8 = psimd_load_f32(i8);

      psimd_f32 vmax = vi0;
      psimd_u32 vidx = psimd_splat_u32(0);

      const psimd_s32 vm1 = vi1 > vmax;
      vmax = psimd_blend_f32(vm1, vi1, vmax);
      vidx = psimd_blend_u32(vm1, psimd_splat_u32(1), vidx);

      const psimd_s32 vm2 = vi2 > vmax;
      vmax = psimd_blend_f32(vm2, vi2, vmax);
      vidx = psimd_blend_u32(vm2, psimd_splat_u32(2), vidx);

      const psimd_s32 vm3 = vi3 > vmax;
      vmax = psimd_blend_f32(vm3, vi3, vmax);
      vidx = psimd_blend_u32(vm3, psimd_splat_u32(3), vidx);

      const psimd_s32 vm4 = vi4 > vmax;
      vmax = psimd_blend_f32(vm4, vi4, vmax);
      vidx = psimd_blend_u32(vm4, psimd_splat_u32(4), vidx);

      const psimd_s32 vm5 = vi5 > vmax;
      vmax = psimd_blend_f32(vm5, vi5, vmax);
      vidx = psimd_blend_u32(vm5, psimd_splat_u32(5), vidx);

      const psimd_s32 vm6 = vi6 > vmax;
      vmax = psimd_blend_f32(vm6, vi6, vmax);
      vidx = psimd_blend_u32(vm6, psimd_splat_u32(6), vidx);

      const psimd_s32 vm7 = vi7 > vmax;
      vmax = psimd_blend_f32(vm7, vi7, vmax);
      vidx = psimd_blend_u32(vm7, psimd_splat_u32(7), vidx);

      const psimd_s32 vm8 = vi8 > vmax;
      vmax = psimd_blend_f32(vm8, vi8, vmax);
      vidx = psimd_blend_u32(vm8, psimd_splat_u32(8), vidx);

      psimd_f32 vout = psimd_max_f32(psimd_min_f32(vmax, voutput_max), voutput_min);

      if (c & 2) {
        psimd_store2_f32(output, vout);
        psimd_store2_u32(index, vidx);
        vout = psimd_concat_hi_f32(vout, vout);
        vidx = psimd_concat_hi_u32(vidx, vidx);
        output += 2;
        index += 2;
      }
      if (c & 1) {
        psimd_store1_f32(output, vout);
        psimd_store1_u32(index, vidx);
        output += 1;
        index += 1;
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
