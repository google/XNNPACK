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


void xnn_f32_dwconv2d_chw_ukernel_3x3p1__rvv_8x1v(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 1);

  size_t vlmax = __riscv_vsetvlmax_e32m1();

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
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);
  const float* i8 = (const float*) ((uintptr_t) i7 + input_width);
  const float* i9 = (const float*) ((uintptr_t) i8 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);
  float* o3 = (float*) ((uintptr_t) o2 + input_width);
  float* o4 = (float*) ((uintptr_t) o3 + input_width);
  float* o5 = (float*) ((uintptr_t) o4 + input_width);
  float* o6 = (float*) ((uintptr_t) o5 + input_width);
  float* o7 = (float*) ((uintptr_t) o6 + input_width);

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
      o4 = o3;
    }
    if XNN_UNPREDICTABLE(output_height < 6) {
      i6 = zero;
      o5 = o4;
    }
    if XNN_UNPREDICTABLE(output_height < 7) {
      i7 = zero;
      o6 = o5;
    }
    if XNN_UNPREDICTABLE(output_height < 8) {
      i8 = zero;
      o7 = o6;
    }
    if XNN_UNPREDICTABLE(output_height < 9) {
      i9 = zero;
    }

    size_t w = input_width >> XNN_LOG2_SIZEOF_FLOAT;
    size_t vl =  __riscv_vsetvl_e32m1(w);
    vfloat32m1_t vi0x1 =  __riscv_vle32_v_f32m1(i0, vl);
    vfloat32m1_t vi0x0 =  __riscv_vfslide1up_vf_f32m1(vi0x1, 0.0f, vl);
    i0 += vl;
    vfloat32m1_t vi1x1 =  __riscv_vle32_v_f32m1(i1, vl);
    vfloat32m1_t vi1x0 =  __riscv_vfslide1up_vf_f32m1(vi1x1, 0.0f, vl);
    i1 += vl;
    vfloat32m1_t vi2x1 =  __riscv_vle32_v_f32m1(i2, vl);
    vfloat32m1_t vi2x0 =  __riscv_vfslide1up_vf_f32m1(vi2x1, 0.0f, vl);
    i2 += vl;
    vfloat32m1_t vi3x1 =  __riscv_vle32_v_f32m1(i3, vl);
    vfloat32m1_t vi3x0 =  __riscv_vfslide1up_vf_f32m1(vi3x1, 0.0f, vl);
    i3 += vl;
    vfloat32m1_t vi4x1 =  __riscv_vle32_v_f32m1(i4, vl);
    vfloat32m1_t vi4x0 =  __riscv_vfslide1up_vf_f32m1(vi4x1, 0.0f, vl);
    i4 += vl;
    vfloat32m1_t vi5x1 =  __riscv_vle32_v_f32m1(i5, vl);
    vfloat32m1_t vi5x0 =  __riscv_vfslide1up_vf_f32m1(vi5x1, 0.0f, vl);
    i5 += vl;
    vfloat32m1_t vi6x1 =  __riscv_vle32_v_f32m1(i6, vl);
    vfloat32m1_t vi6x0 =  __riscv_vfslide1up_vf_f32m1(vi6x1, 0.0f, vl);
    i6 += vl;
    vfloat32m1_t vi7x1 =  __riscv_vle32_v_f32m1(i7, vl);
    vfloat32m1_t vi7x0 =  __riscv_vfslide1up_vf_f32m1(vi7x1, 0.0f, vl);
    i7 += vl;
    vfloat32m1_t vi8x1 =  __riscv_vle32_v_f32m1(i8, vl);
    vfloat32m1_t vi8x0 =  __riscv_vfslide1up_vf_f32m1(vi8x1, 0.0f, vl);
    i8 += vl;
    vfloat32m1_t vi9x1 =  __riscv_vle32_v_f32m1(i9, vl);
    vfloat32m1_t vi9x0 =  __riscv_vfslide1up_vf_f32m1(vi9x1, 0.0f, vl);
    i9 += vl;

    while (w > vlmax) {
      vfloat32m1_t vi0x2 = __riscv_vfslide1down_vf_f32m1(vi0x1, *i0, vl);
      vfloat32m1_t vi1x2 = __riscv_vfslide1down_vf_f32m1(vi1x1, *i1, vl);
      vfloat32m1_t vi2x2 = __riscv_vfslide1down_vf_f32m1(vi2x1, *i2, vl);
      vfloat32m1_t vi3x2 = __riscv_vfslide1down_vf_f32m1(vi3x1, *i3, vl);
      vfloat32m1_t vi4x2 = __riscv_vfslide1down_vf_f32m1(vi4x1, *i4, vl);
      vfloat32m1_t vi5x2 = __riscv_vfslide1down_vf_f32m1(vi5x1, *i5, vl);
      vfloat32m1_t vi6x2 = __riscv_vfslide1down_vf_f32m1(vi6x1, *i6, vl);
      vfloat32m1_t vi7x2 = __riscv_vfslide1down_vf_f32m1(vi7x1, *i7, vl);
      vfloat32m1_t vi8x2 = __riscv_vfslide1down_vf_f32m1(vi8x1, *i8, vl);
      vfloat32m1_t vi9x2 = __riscv_vfslide1down_vf_f32m1(vi9x1, *i9, vl);

      vfloat32m1_t vo0p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo1p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo2p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo3p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo4p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo5p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo6p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo7p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk00, vi0x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk00, vi1x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk00, vi2x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk00, vi3x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk00, vi4x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk00, vi5x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk00, vi6x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk00, vi7x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk10, vi1x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk10, vi2x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk10, vi3x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk10, vi4x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk10, vi5x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk10, vi6x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk10, vi7x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk10, vi8x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk20, vi2x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk20, vi3x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk20, vi4x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk20, vi5x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk20, vi6x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk20, vi7x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk20, vi8x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk20, vi9x0, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk01, vi0x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk01, vi1x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk01, vi2x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk01, vi3x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk01, vi4x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk01, vi5x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk01, vi6x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk01, vi7x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk11, vi1x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk11, vi2x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk11, vi3x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk11, vi4x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk11, vi5x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk11, vi6x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk11, vi7x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk11, vi8x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk21, vi2x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk21, vi3x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk21, vi4x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk21, vi5x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk21, vi6x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk21, vi7x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk21, vi8x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk21, vi9x1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk02, vi0x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk02, vi1x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk02, vi2x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk02, vi3x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk02, vi4x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk02, vi5x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk02, vi6x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk02, vi7x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk12, vi1x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk12, vi2x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk12, vi3x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk12, vi4x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk12, vi5x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk12, vi6x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk12, vi7x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk12, vi8x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk22, vi2x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk22, vi3x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk22, vi4x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk22, vi5x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk22, vi6x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk22, vi7x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk22, vi8x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk22, vi9x2, vl);


      vo0p0 = __riscv_vfmax_vf_f32m1(vo0p0, vmin, vl);
      vo1p0 = __riscv_vfmax_vf_f32m1(vo1p0, vmin, vl);
      vo2p0 = __riscv_vfmax_vf_f32m1(vo2p0, vmin, vl);
      vo3p0 = __riscv_vfmax_vf_f32m1(vo3p0, vmin, vl);
      vo4p0 = __riscv_vfmax_vf_f32m1(vo4p0, vmin, vl);
      vo5p0 = __riscv_vfmax_vf_f32m1(vo5p0, vmin, vl);
      vo6p0 = __riscv_vfmax_vf_f32m1(vo6p0, vmin, vl);
      vo7p0 = __riscv_vfmax_vf_f32m1(vo7p0, vmin, vl);

      vo0p0 = __riscv_vfmin_vf_f32m1(vo0p0, vmax, vl);
      vo1p0 = __riscv_vfmin_vf_f32m1(vo1p0, vmax, vl);
      vo2p0 = __riscv_vfmin_vf_f32m1(vo2p0, vmax, vl);
      vo3p0 = __riscv_vfmin_vf_f32m1(vo3p0, vmax, vl);
      vo4p0 = __riscv_vfmin_vf_f32m1(vo4p0, vmax, vl);
      vo5p0 = __riscv_vfmin_vf_f32m1(vo5p0, vmax, vl);
      vo6p0 = __riscv_vfmin_vf_f32m1(vo6p0, vmax, vl);
      vo7p0 = __riscv_vfmin_vf_f32m1(vo7p0, vmax, vl);

      __riscv_vse32_v_f32m1(o7, vo7p0, vl);
      o7 += vl;
      __riscv_vse32_v_f32m1(o6, vo6p0, vl);
      o6 += vl;
      __riscv_vse32_v_f32m1(o5, vo5p0, vl);
      o5 += vl;
      __riscv_vse32_v_f32m1(o4, vo4p0, vl);
      o4 += vl;
      __riscv_vse32_v_f32m1(o3, vo3p0, vl);
      o3 += vl;
      __riscv_vse32_v_f32m1(o2, vo2p0, vl);
      o2 += vl;
      __riscv_vse32_v_f32m1(o1, vo1p0, vl);
      o1 += vl;
      __riscv_vse32_v_f32m1(o0, vo0p0, vl);
      o0 += vl;

      w -= vl;
      vl = __riscv_vsetvl_e32m1(w);
      vi0x1 =  __riscv_vle32_v_f32m1(i0, vl);
      vi0x0 =  __riscv_vfslide1up_vf_f32m1(vi0x1, *(i0-1), vl);
      i0 += vl;
      vi1x1 =  __riscv_vle32_v_f32m1(i1, vl);
      vi1x0 =  __riscv_vfslide1up_vf_f32m1(vi1x1, *(i1-1), vl);
      i1 += vl;
      vi2x1 =  __riscv_vle32_v_f32m1(i2, vl);
      vi2x0 =  __riscv_vfslide1up_vf_f32m1(vi2x1, *(i2-1), vl);
      i2 += vl;
      vi3x1 =  __riscv_vle32_v_f32m1(i3, vl);
      vi3x0 =  __riscv_vfslide1up_vf_f32m1(vi3x1, *(i3-1), vl);
      i3 += vl;
      vi4x1 =  __riscv_vle32_v_f32m1(i4, vl);
      vi4x0 =  __riscv_vfslide1up_vf_f32m1(vi4x1, *(i4-1), vl);
      i4 += vl;
      vi5x1 =  __riscv_vle32_v_f32m1(i5, vl);
      vi5x0 =  __riscv_vfslide1up_vf_f32m1(vi5x1, *(i5-1), vl);
      i5 += vl;
      vi6x1 =  __riscv_vle32_v_f32m1(i6, vl);
      vi6x0 =  __riscv_vfslide1up_vf_f32m1(vi6x1, *(i6-1), vl);
      i6 += vl;
      vi7x1 =  __riscv_vle32_v_f32m1(i7, vl);
      vi7x0 =  __riscv_vfslide1up_vf_f32m1(vi7x1, *(i7-1), vl);
      i7 += vl;
      vi8x1 =  __riscv_vle32_v_f32m1(i8, vl);
      vi8x0 =  __riscv_vfslide1up_vf_f32m1(vi8x1, *(i8-1), vl);
      i8 += vl;
      vi9x1 =  __riscv_vle32_v_f32m1(i9, vl);
      vi9x0 =  __riscv_vfslide1up_vf_f32m1(vi9x1, *(i9-1), vl);
      i9 += vl;
    }
    // Always process the last tile separately to account for right edge.
    assert(w <= vlmax);
    {
      vfloat32m1_t vi0x2 = __riscv_vfslide1down_vf_f32m1(vi0x1, 0.0f, vl);
      vfloat32m1_t vi1x2 = __riscv_vfslide1down_vf_f32m1(vi1x1, 0.0f, vl);
      vfloat32m1_t vi2x2 = __riscv_vfslide1down_vf_f32m1(vi2x1, 0.0f, vl);
      vfloat32m1_t vi3x2 = __riscv_vfslide1down_vf_f32m1(vi3x1, 0.0f, vl);
      vfloat32m1_t vi4x2 = __riscv_vfslide1down_vf_f32m1(vi4x1, 0.0f, vl);
      vfloat32m1_t vi5x2 = __riscv_vfslide1down_vf_f32m1(vi5x1, 0.0f, vl);
      vfloat32m1_t vi6x2 = __riscv_vfslide1down_vf_f32m1(vi6x1, 0.0f, vl);
      vfloat32m1_t vi7x2 = __riscv_vfslide1down_vf_f32m1(vi7x1, 0.0f, vl);
      vfloat32m1_t vi8x2 = __riscv_vfslide1down_vf_f32m1(vi8x1, 0.0f, vl);
      vfloat32m1_t vi9x2 = __riscv_vfslide1down_vf_f32m1(vi9x1, 0.0f, vl);

      vfloat32m1_t vo0p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo1p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo2p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo3p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo4p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo5p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo6p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo7p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk00, vi0x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk00, vi1x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk00, vi2x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk00, vi3x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk00, vi4x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk00, vi5x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk00, vi6x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk00, vi7x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk10, vi1x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk10, vi2x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk10, vi3x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk10, vi4x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk10, vi5x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk10, vi6x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk10, vi7x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk10, vi8x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk20, vi2x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk20, vi3x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk20, vi4x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk20, vi5x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk20, vi6x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk20, vi7x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk20, vi8x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk20, vi9x0, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk01, vi0x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk01, vi1x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk01, vi2x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk01, vi3x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk01, vi4x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk01, vi5x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk01, vi6x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk01, vi7x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk11, vi1x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk11, vi2x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk11, vi3x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk11, vi4x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk11, vi5x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk11, vi6x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk11, vi7x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk11, vi8x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk21, vi2x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk21, vi3x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk21, vi4x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk21, vi5x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk21, vi6x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk21, vi7x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk21, vi8x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk21, vi9x1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk02, vi0x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk02, vi1x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk02, vi2x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk02, vi3x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk02, vi4x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk02, vi5x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk02, vi6x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk02, vi7x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk12, vi1x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk12, vi2x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk12, vi3x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk12, vi4x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk12, vi5x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk12, vi6x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk12, vi7x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk12, vi8x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk22, vi2x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk22, vi3x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk22, vi4x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk22, vi5x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk22, vi6x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk22, vi7x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk22, vi8x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk22, vi9x2, vl);


      vo0p0 = __riscv_vfmax_vf_f32m1(vo0p0, vmin, vl);
      vo1p0 = __riscv_vfmax_vf_f32m1(vo1p0, vmin, vl);
      vo2p0 = __riscv_vfmax_vf_f32m1(vo2p0, vmin, vl);
      vo3p0 = __riscv_vfmax_vf_f32m1(vo3p0, vmin, vl);
      vo4p0 = __riscv_vfmax_vf_f32m1(vo4p0, vmin, vl);
      vo5p0 = __riscv_vfmax_vf_f32m1(vo5p0, vmin, vl);
      vo6p0 = __riscv_vfmax_vf_f32m1(vo6p0, vmin, vl);
      vo7p0 = __riscv_vfmax_vf_f32m1(vo7p0, vmin, vl);

      vo0p0 = __riscv_vfmin_vf_f32m1(vo0p0, vmax, vl);
      vo1p0 = __riscv_vfmin_vf_f32m1(vo1p0, vmax, vl);
      vo2p0 = __riscv_vfmin_vf_f32m1(vo2p0, vmax, vl);
      vo3p0 = __riscv_vfmin_vf_f32m1(vo3p0, vmax, vl);
      vo4p0 = __riscv_vfmin_vf_f32m1(vo4p0, vmax, vl);
      vo5p0 = __riscv_vfmin_vf_f32m1(vo5p0, vmax, vl);
      vo6p0 = __riscv_vfmin_vf_f32m1(vo6p0, vmax, vl);
      vo7p0 = __riscv_vfmin_vf_f32m1(vo7p0, vmax, vl);

      __riscv_vse32_v_f32m1(o7, vo7p0, vl);
      o7 += vl;
      __riscv_vse32_v_f32m1(o6, vo6p0, vl);
      o6 += vl;
      __riscv_vse32_v_f32m1(o5, vo5p0, vl);
      o5 += vl;
      __riscv_vse32_v_f32m1(o4, vo4p0, vl);
      o4 += vl;
      __riscv_vse32_v_f32m1(o3, vo3p0, vl);
      o3 += vl;
      __riscv_vse32_v_f32m1(o2, vo2p0, vl);
      o2 += vl;
      __riscv_vse32_v_f32m1(o1, vo1p0, vl);
      o1 += vl;
      __riscv_vse32_v_f32m1(o0, vo0p0, vl);
      o0 += vl;
    }

    i0 = (const float*) ((uintptr_t) i8 - input_width);
    i1 = (const float*) ((uintptr_t) i9 - input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);
    i8 = (const float*) ((uintptr_t) i7 + input_width);
    i9 = (const float*) ((uintptr_t) i8 + input_width);

    o0 = o7;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);
    o3 = (float*) ((uintptr_t) o2 + input_width);
    o4 = (float*) ((uintptr_t) o3 + input_width);
    o5 = (float*) ((uintptr_t) o4 + input_width);
    o6 = (float*) ((uintptr_t) o5 + input_width);
    o7 = (float*) ((uintptr_t) o6 + input_width);

    output_height = doz(output_height, 8);
  } while (output_height != 0);
}
