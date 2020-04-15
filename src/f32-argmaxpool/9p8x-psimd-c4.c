// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/argmaxpool.h>


void xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4(
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
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements > 9);
  assert(channels != 0);

  const psimd_f32 voutput_max = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 voutput_min = psimd_load_splat_f32(&params->scalar.min);
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

        psimd_store_f32(ab, vmax);
        ab += 4;
        psimd_store_u32(ib, vidx);
        ib += 4;
      }
    }
    const psimd_u32 v1 = psimd_splat_u32(1);
    const psimd_u32 v8 = psimd_splat_u32(8);
    psimd_u32 vidx0 = psimd_add_u32(v1, v8);

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

        psimd_f32 vmax = psimd_load_f32(ab);
        psimd_u32 vidx = psimd_load_u32(ib);

        const psimd_s32 vm0 = vi0 > vmax;
        vmax = psimd_blend_f32(vm0, vi0, vmax);
        vidx = psimd_blend_u32(vm0, vidx0, vidx);

        const psimd_s32 vm1 = vi1 > vmax;
        const psimd_u32 vidx1 = psimd_add_u32(vidx0, v1);
        vmax = psimd_blend_f32(vm1, vi1, vmax);
        vidx = psimd_blend_u32(vm1, vidx1, vidx);

        const psimd_s32 vm2 = vi2 > vmax;
        const psimd_u32 vidx2 = psimd_add_u32(vidx1, v1);
        vmax = psimd_blend_f32(vm2, vi2, vmax);
        vidx = psimd_blend_u32(vm2, vidx2, vidx);

        const psimd_s32 vm3 = vi3 > vmax;
        const psimd_u32 vidx3 = psimd_add_u32(vidx2, v1);
        vmax = psimd_blend_f32(vm3, vi3, vmax);
        vidx = psimd_blend_u32(vm3, vidx3, vidx);

        const psimd_s32 vm4 = vi4 > vmax;
        const psimd_u32 vidx4 = psimd_add_u32(vidx3, v1);
        vmax = psimd_blend_f32(vm4, vi4, vmax);
        vidx = psimd_blend_u32(vm4, vidx4, vidx);

        const psimd_s32 vm5 = vi5 > vmax;
        const psimd_u32 vidx5 = psimd_add_u32(vidx4, v1);
        vmax = psimd_blend_f32(vm5, vi5, vmax);
        vidx = psimd_blend_u32(vm5, vidx5, vidx);

        const psimd_s32 vm6 = vi6 > vmax;
        const psimd_u32 vidx6 = psimd_add_u32(vidx5, v1);
        vmax = psimd_blend_f32(vm6, vi6, vmax);
        vidx = psimd_blend_u32(vm6, vidx6, vidx);

        const psimd_s32 vm7 = vi7 > vmax;
        const psimd_u32 vidx7 = psimd_add_u32(vidx6, v1);
        vmax = psimd_blend_f32(vm7, vi7, vmax);
        vidx = psimd_blend_u32(vm7, vidx7, vidx);

        psimd_store_f32(ab, vmax);
        ab += 4;
        psimd_store_u32(ib, vidx);
        ib += 4;
      }
      vidx0 = psimd_add_u32(vidx0, v8);
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

        psimd_f32 vmax = psimd_load_f32(ab);
        ab += 4;
        psimd_u32 vidx = psimd_load_u32(ib);
        ib += 4;

        const psimd_s32 vm0 = vi0 > vmax;
        vmax = psimd_blend_f32(vm0, vi0, vmax);
        vidx = psimd_blend_u32(vm0, vidx0, vidx);

        const psimd_s32 vm1 = vi1 > vmax;
        const psimd_u32 vidx1 = psimd_add_u32(vidx0, v1);
        vmax = psimd_blend_f32(vm1, vi1, vmax);
        vidx = psimd_blend_u32(vm1, vidx1, vidx);

        const psimd_s32 vm2 = vi2 > vmax;
        const psimd_u32 vidx2 = psimd_add_u32(vidx1, v1);
        vmax = psimd_blend_f32(vm2, vi2, vmax);
        vidx = psimd_blend_u32(vm2, vidx2, vidx);

        const psimd_s32 vm3 = vi3 > vmax;
        const psimd_u32 vidx3 = psimd_add_u32(vidx2, v1);
        vmax = psimd_blend_f32(vm3, vi3, vmax);
        vidx = psimd_blend_u32(vm3, vidx3, vidx);

        const psimd_s32 vm4 = vi4 > vmax;
        const psimd_u32 vidx4 = psimd_add_u32(vidx3, v1);
        vmax = psimd_blend_f32(vm4, vi4, vmax);
        vidx = psimd_blend_u32(vm4, vidx4, vidx);

        const psimd_s32 vm5 = vi5 > vmax;
        const psimd_u32 vidx5 = psimd_add_u32(vidx4, v1);
        vmax = psimd_blend_f32(vm5, vi5, vmax);
        vidx = psimd_blend_u32(vm5, vidx5, vidx);

        const psimd_s32 vm6 = vi6 > vmax;
        const psimd_u32 vidx6 = psimd_add_u32(vidx5, v1);
        vmax = psimd_blend_f32(vm6, vi6, vmax);
        vidx = psimd_blend_u32(vm6, vidx6, vidx);

        const psimd_s32 vm7 = vi7 > vmax;
        const psimd_u32 vidx7 = psimd_add_u32(vidx6, v1);
        vmax = psimd_blend_f32(vm7, vi7, vmax);
        vidx = psimd_blend_u32(vm7, vidx7, vidx);

        psimd_f32 vout = psimd_max_f32(psimd_min_f32(vmax, voutput_max), voutput_min);

        psimd_store_f32(o, vout);
        o += 4;
        psimd_store_u32(i, vidx);
        i += 4;
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

        psimd_f32 vmax = psimd_load_f32(ab);
        psimd_u32 vidx = psimd_load_u32(ib);

        const psimd_s32 vm0 = vi0 > vmax;
        vmax = psimd_blend_f32(vm0, vi0, vmax);
        vidx = psimd_blend_u32(vm0, vidx0, vidx);

        const psimd_s32 vm1 = vi1 > vmax;
        const psimd_u32 vidx1 = psimd_add_u32(vidx0, v1);
        vmax = psimd_blend_f32(vm1, vi1, vmax);
        vidx = psimd_blend_u32(vm1, vidx1, vidx);

        const psimd_s32 vm2 = vi2 > vmax;
        const psimd_u32 vidx2 = psimd_add_u32(vidx1, v1);
        vmax = psimd_blend_f32(vm2, vi2, vmax);
        vidx = psimd_blend_u32(vm2, vidx2, vidx);

        const psimd_s32 vm3 = vi3 > vmax;
        const psimd_u32 vidx3 = psimd_add_u32(vidx2, v1);
        vmax = psimd_blend_f32(vm3, vi3, vmax);
        vidx = psimd_blend_u32(vm3, vidx3, vidx);

        const psimd_s32 vm4 = vi4 > vmax;
        const psimd_u32 vidx4 = psimd_add_u32(vidx3, v1);
        vmax = psimd_blend_f32(vm4, vi4, vmax);
        vidx = psimd_blend_u32(vm4, vidx4, vidx);

        const psimd_s32 vm5 = vi5 > vmax;
        const psimd_u32 vidx5 = psimd_add_u32(vidx4, v1);
        vmax = psimd_blend_f32(vm5, vi5, vmax);
        vidx = psimd_blend_u32(vm5, vidx5, vidx);

        const psimd_s32 vm6 = vi6 > vmax;
        const psimd_u32 vidx6 = psimd_add_u32(vidx5, v1);
        vmax = psimd_blend_f32(vm6, vi6, vmax);
        vidx = psimd_blend_u32(vm6, vidx6, vidx);

        const psimd_s32 vm7 = vi7 > vmax;
        const psimd_u32 vidx7 = psimd_add_u32(vidx6, v1);
        vmax = psimd_blend_f32(vm7, vi7, vmax);
        vidx = psimd_blend_u32(vm7, vidx7, vidx);

        psimd_f32 vout = psimd_max_f32(psimd_min_f32(vmax, voutput_max), voutput_min);

        if (c & 2) {
          psimd_store2_f32(o, vout);
          psimd_store2_u32(i, vidx);
          vout = psimd_concat_hi_f32(vout, vout);
          vidx = psimd_concat_hi_u32(vidx, vidx);
          o += 2;
          i += 2;
        }
        if (c & 1) {
          psimd_store1_f32(o, vout);
          psimd_store1_u32(i, vidx);
          o += 1;
          i += 1;
        }
      }
    }

    output = (float*) ((uintptr_t) o + output_increment);
    index = (uint32_t*) i;
  } while (--output_pixels != 0);
}
