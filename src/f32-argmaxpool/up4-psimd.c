// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/argmaxpool.h>


void xnn_f32_argmaxpool_ukernel_up4__psimd(
    size_t n,
    size_t ks,
    size_t kc,
    const float** input,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(ks != 0);
  assert(ks <= 4);
  assert(kc != 0);

  const psimd_f32 voutput_max = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 voutput_min = psimd_load_splat_f32(&params->scalar.min);
  do {
    float* o = output;
    uint32_t* i = index;

    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    if (ks < 2) {
      i1 = i0;
    }
    if (ks <= 2) {
      i2 = i0;
    }
    if (ks != 4) {
      i3 = i0;
    }

    size_t k = kc;
    for (; k >= 4; k -= 4) {
      const psimd_f32 vi0 = psimd_load_f32(i0);
      i0 += 4;
      const psimd_f32 vi1 = psimd_load_f32(i1);
      i1 += 4;
      const psimd_f32 vi2 = psimd_load_f32(i2);
      i2 += 4;
      const psimd_f32 vi3 = psimd_load_f32(i3);
      i3 += 4;

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

      const psimd_f32 vout = psimd_max_f32(psimd_min_f32(vmax, voutput_max), voutput_min);

      psimd_store_f32(o, vout);
      o += 4;
      psimd_store_u32(i, vidx);
      i += 4;
    }
    if (k != 0) {
      const psimd_f32 vi0 = psimd_load_f32(i0);
      const psimd_f32 vi1 = psimd_load_f32(i1);
      const psimd_f32 vi2 = psimd_load_f32(i2);
      const psimd_f32 vi3 = psimd_load_f32(i3);

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

      psimd_f32 vout = psimd_max_f32(psimd_min_f32(vmax, voutput_max), voutput_min);

      if (k & 2) {
        psimd_store2_f32(o, vout);
        psimd_store2_u32(i, vidx);
        vout = psimd_concat_hi_f32(vout, vout);
        vidx = psimd_concat_hi_u32(vidx, vidx);
        o += 2;
        i += 2;
      }
      if (k & 1) {
        psimd_store1_f32(o, vout);
        psimd_store1_u32(i, vidx);
        o += 1;
        i += 1;
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
    index = (uint32_t*) i;
  } while (--n != 0);
}
