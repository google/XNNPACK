// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/3x3p1-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
// Copyright 2025 Andes Technology
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>

#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_3x3p1__rvv_4x2v(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 1);

  size_t vlmax = __riscv_vsetvlmax_e32m2();

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  const float vbias = weights[0];
  const float vk00 = weights[1];
  const float vk01 = weights[2];
  const float vk02 = weights[3];
  const float vk10 = weights[4];
  const float vk11 = weights[5];
  const float vk12 = weights[6];
  const float vk20 = weights[7];
  const float vk21 = weights[8];
  const float vk22 = weights[9];

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);
  float* o3 = (float*) ((uintptr_t) o2 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i3 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i4 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(output_height < 5) {
      i5 = zero;
    }

    size_t w = input_width >> XNN_LOG2_SIZEOF_FLOAT;
    size_t vl =  __riscv_vsetvl_e32m2(w);
    vfloat32m2_t vi0x1 =  __riscv_vle32_v_f32m2(i0, vl);
    vfloat32m2_t vi0x0 =  __riscv_vfslide1up_vf_f32m2(vi0x1, 0.0f, vl);
    i0 += vl;
    vfloat32m2_t vi1x1 =  __riscv_vle32_v_f32m2(i1, vl);
    vfloat32m2_t vi1x0 =  __riscv_vfslide1up_vf_f32m2(vi1x1, 0.0f, vl);
    i1 += vl;
    vfloat32m2_t vi2x1 =  __riscv_vle32_v_f32m2(i2, vl);
    vfloat32m2_t vi2x0 =  __riscv_vfslide1up_vf_f32m2(vi2x1, 0.0f, vl);
    i2 += vl;
    vfloat32m2_t vi3x1 =  __riscv_vle32_v_f32m2(i3, vl);
    vfloat32m2_t vi3x0 =  __riscv_vfslide1up_vf_f32m2(vi3x1, 0.0f, vl);
    i3 += vl;
    vfloat32m2_t vi4x1 =  __riscv_vle32_v_f32m2(i4, vl);
    vfloat32m2_t vi4x0 =  __riscv_vfslide1up_vf_f32m2(vi4x1, 0.0f, vl);
    i4 += vl;
    vfloat32m2_t vi5x1 =  __riscv_vle32_v_f32m2(i5, vl);
    vfloat32m2_t vi5x0 =  __riscv_vfslide1up_vf_f32m2(vi5x1, 0.0f, vl);
    i5 += vl;

    while (w > vlmax) {
      vfloat32m2_t vi0x2 = __riscv_vfslide1down_vf_f32m2(vi0x1, *i0, vl);
      vfloat32m2_t vi1x2 = __riscv_vfslide1down_vf_f32m2(vi1x1, *i1, vl);
      vfloat32m2_t vi2x2 = __riscv_vfslide1down_vf_f32m2(vi2x1, *i2, vl);
      vfloat32m2_t vi3x2 = __riscv_vfslide1down_vf_f32m2(vi3x1, *i3, vl);
      vfloat32m2_t vi4x2 = __riscv_vfslide1down_vf_f32m2(vi4x1, *i4, vl);
      vfloat32m2_t vi5x2 = __riscv_vfslide1down_vf_f32m2(vi5x1, *i5, vl);

      vfloat32m2_t vo0p0 = __riscv_vfmv_v_f_f32m2(vbias, vl);
      vfloat32m2_t vo1p0 = __riscv_vfmv_v_f_f32m2(vbias, vl);
      vfloat32m2_t vo2p0 = __riscv_vfmv_v_f_f32m2(vbias, vl);
      vfloat32m2_t vo3p0 = __riscv_vfmv_v_f_f32m2(vbias, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk00, vi0x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk00, vi1x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk00, vi2x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk00, vi3x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk10, vi1x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk10, vi2x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk10, vi3x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk10, vi4x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk20, vi2x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk20, vi3x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk20, vi4x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk20, vi5x0, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk01, vi0x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk01, vi1x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk01, vi2x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk01, vi3x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk11, vi1x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk11, vi2x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk11, vi3x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk11, vi4x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk21, vi2x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk21, vi3x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk21, vi4x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk21, vi5x1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk02, vi0x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk02, vi1x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk02, vi2x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk02, vi3x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk12, vi1x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk12, vi2x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk12, vi3x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk12, vi4x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk22, vi2x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk22, vi3x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk22, vi4x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk22, vi5x2, vl);


      vo0p0 = __riscv_vfmax_vf_f32m2(vo0p0, vmin, vl);
      vo1p0 = __riscv_vfmax_vf_f32m2(vo1p0, vmin, vl);
      vo2p0 = __riscv_vfmax_vf_f32m2(vo2p0, vmin, vl);
      vo3p0 = __riscv_vfmax_vf_f32m2(vo3p0, vmin, vl);

      vo0p0 = __riscv_vfmin_vf_f32m2(vo0p0, vmax, vl);
      vo1p0 = __riscv_vfmin_vf_f32m2(vo1p0, vmax, vl);
      vo2p0 = __riscv_vfmin_vf_f32m2(vo2p0, vmax, vl);
      vo3p0 = __riscv_vfmin_vf_f32m2(vo3p0, vmax, vl);

      __riscv_vse32_v_f32m2(o3, vo3p0, vl);
      o3 += vl;
      __riscv_vse32_v_f32m2(o2, vo2p0, vl);
      o2 += vl;
      __riscv_vse32_v_f32m2(o1, vo1p0, vl);
      o1 += vl;
      __riscv_vse32_v_f32m2(o0, vo0p0, vl);
      o0 += vl;

      w -= vl;
      vl = __riscv_vsetvl_e32m2(w);
      vi0x1 =  __riscv_vle32_v_f32m2(i0, vl);
      vi0x0 =  __riscv_vfslide1up_vf_f32m2(vi0x1, *(i0-1), vl);
      i0 += vl;
      vi1x1 =  __riscv_vle32_v_f32m2(i1, vl);
      vi1x0 =  __riscv_vfslide1up_vf_f32m2(vi1x1, *(i1-1), vl);
      i1 += vl;
      vi2x1 =  __riscv_vle32_v_f32m2(i2, vl);
      vi2x0 =  __riscv_vfslide1up_vf_f32m2(vi2x1, *(i2-1), vl);
      i2 += vl;
      vi3x1 =  __riscv_vle32_v_f32m2(i3, vl);
      vi3x0 =  __riscv_vfslide1up_vf_f32m2(vi3x1, *(i3-1), vl);
      i3 += vl;
      vi4x1 =  __riscv_vle32_v_f32m2(i4, vl);
      vi4x0 =  __riscv_vfslide1up_vf_f32m2(vi4x1, *(i4-1), vl);
      i4 += vl;
      vi5x1 =  __riscv_vle32_v_f32m2(i5, vl);
      vi5x0 =  __riscv_vfslide1up_vf_f32m2(vi5x1, *(i5-1), vl);
      i5 += vl;
    }
    // Always process the last tile separately to account for right edge.
    assert(w <= vlmax);
    {
      vfloat32m2_t vi0x2 = __riscv_vfslide1down_vf_f32m2(vi0x1, 0.0f, vl);
      vfloat32m2_t vi1x2 = __riscv_vfslide1down_vf_f32m2(vi1x1, 0.0f, vl);
      vfloat32m2_t vi2x2 = __riscv_vfslide1down_vf_f32m2(vi2x1, 0.0f, vl);
      vfloat32m2_t vi3x2 = __riscv_vfslide1down_vf_f32m2(vi3x1, 0.0f, vl);
      vfloat32m2_t vi4x2 = __riscv_vfslide1down_vf_f32m2(vi4x1, 0.0f, vl);
      vfloat32m2_t vi5x2 = __riscv_vfslide1down_vf_f32m2(vi5x1, 0.0f, vl);

      vfloat32m2_t vo0p0 = __riscv_vfmv_v_f_f32m2(vbias, vl);
      vfloat32m2_t vo1p0 = __riscv_vfmv_v_f_f32m2(vbias, vl);
      vfloat32m2_t vo2p0 = __riscv_vfmv_v_f_f32m2(vbias, vl);
      vfloat32m2_t vo3p0 = __riscv_vfmv_v_f_f32m2(vbias, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk00, vi0x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk00, vi1x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk00, vi2x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk00, vi3x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk10, vi1x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk10, vi2x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk10, vi3x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk10, vi4x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk20, vi2x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk20, vi3x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk20, vi4x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk20, vi5x0, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk01, vi0x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk01, vi1x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk01, vi2x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk01, vi3x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk11, vi1x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk11, vi2x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk11, vi3x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk11, vi4x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk21, vi2x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk21, vi3x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk21, vi4x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk21, vi5x1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk02, vi0x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk02, vi1x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk02, vi2x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk02, vi3x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk12, vi1x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk12, vi2x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk12, vi3x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk12, vi4x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk22, vi2x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, vk22, vi3x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, vk22, vi4x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m2(vo3p0, vk22, vi5x2, vl);


      vo0p0 = __riscv_vfmax_vf_f32m2(vo0p0, vmin, vl);
      vo1p0 = __riscv_vfmax_vf_f32m2(vo1p0, vmin, vl);
      vo2p0 = __riscv_vfmax_vf_f32m2(vo2p0, vmin, vl);
      vo3p0 = __riscv_vfmax_vf_f32m2(vo3p0, vmin, vl);

      vo0p0 = __riscv_vfmin_vf_f32m2(vo0p0, vmax, vl);
      vo1p0 = __riscv_vfmin_vf_f32m2(vo1p0, vmax, vl);
      vo2p0 = __riscv_vfmin_vf_f32m2(vo2p0, vmax, vl);
      vo3p0 = __riscv_vfmin_vf_f32m2(vo3p0, vmax, vl);

      __riscv_vse32_v_f32m2(o3, vo3p0, vl);
      o3 += vl;
      __riscv_vse32_v_f32m2(o2, vo2p0, vl);
      o2 += vl;
      __riscv_vse32_v_f32m2(o1, vo1p0, vl);
      o1 += vl;
      __riscv_vse32_v_f32m2(o0, vo0p0, vl);
      o0 += vl;
    }

    i0 = (const float*) ((uintptr_t) i4 - input_width);
    i1 = (const float*) ((uintptr_t) i5 - input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);

    o0 = o3;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);
    o3 = (float*) ((uintptr_t) o2 + input_width);

    output_height = doz(output_height, 4);
  } while (output_height != 0);
}
