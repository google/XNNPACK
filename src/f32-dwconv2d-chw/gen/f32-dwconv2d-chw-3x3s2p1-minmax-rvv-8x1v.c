// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/3x3s2p1-rvv.c.in
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


void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__rvv_8x1v(
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
  assert(padding_top >= 0);
  assert(padding_top <= 1);

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

  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(float)) / 2, sizeof(float));

  const float* i0 = (const float*) ((uintptr_t) input - ((-padding_top) & input_width));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);
  const float* i8 = (const float*) ((uintptr_t) i7 + input_width);
  const float* i9 = (const float*) ((uintptr_t) i8 + input_width);
  const float* i10 = (const float*) ((uintptr_t) i9 + input_width);
  const float* i11 = (const float*) ((uintptr_t) i10 + input_width);
  const float* i12 = (const float*) ((uintptr_t) i11 + input_width);
  const float* i13 = (const float*) ((uintptr_t) i12 + input_width);
  const float* i14 = (const float*) ((uintptr_t) i13 + input_width);
  const float* i15 = (const float*) ((uintptr_t) i14 + input_width);
  const float* i16 = (const float*) ((uintptr_t) i15 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + output_width);
  float* o2 = (float*) ((uintptr_t) o1 + output_width);
  float* o3 = (float*) ((uintptr_t) o2 + output_width);
  float* o4 = (float*) ((uintptr_t) o3 + output_width);
  float* o5 = (float*) ((uintptr_t) o4 + output_width);
  float* o6 = (float*) ((uintptr_t) o5 + output_width);
  float* o7 = (float*) ((uintptr_t) o6 + output_width);

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 5) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i5 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 8) {
      i6 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 9) {
      i7 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 10) {
      i8 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 11) {
      i9 = zero;
      o4 = o3;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 12) {
      i10 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 13) {
      i11 = zero;
      o5 = o4;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 14) {
      i12 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 15) {
      i13 = zero;
      o6 = o5;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 16) {
      i14 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 17) {
      i15 = zero;
      o7 = o6;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 18) {
      i16 = zero;
    }

    size_t w = input_width >> XNN_LOG2_SIZEOF_FLOAT;
    size_t vl =  __riscv_vsetvl_e32m1((w + 1) >> 1);
    vfloat32m1x2_t tuple;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i0, vl);
    vfloat32m1_t vi0x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi0x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi0x2;
    vi0x0 = __riscv_vfslide1up_vf_f32m1(vi0x0, 0.0f, vl);
    i0 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i1, vl);
    vfloat32m1_t vi1x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi1x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi1x2;
    vi1x0 = __riscv_vfslide1up_vf_f32m1(vi1x0, 0.0f, vl);
    i1 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i2, vl);
    vfloat32m1_t vi2x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi2x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi2x2;
    vi2x0 = __riscv_vfslide1up_vf_f32m1(vi2x0, 0.0f, vl);
    i2 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i3, vl);
    vfloat32m1_t vi3x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi3x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi3x2;
    vi3x0 = __riscv_vfslide1up_vf_f32m1(vi3x0, 0.0f, vl);
    i3 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i4, vl);
    vfloat32m1_t vi4x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi4x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi4x2;
    vi4x0 = __riscv_vfslide1up_vf_f32m1(vi4x0, 0.0f, vl);
    i4 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i5, vl);
    vfloat32m1_t vi5x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi5x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi5x2;
    vi5x0 = __riscv_vfslide1up_vf_f32m1(vi5x0, 0.0f, vl);
    i5 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i6, vl);
    vfloat32m1_t vi6x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi6x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi6x2;
    vi6x0 = __riscv_vfslide1up_vf_f32m1(vi6x0, 0.0f, vl);
    i6 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i7, vl);
    vfloat32m1_t vi7x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi7x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi7x2;
    vi7x0 = __riscv_vfslide1up_vf_f32m1(vi7x0, 0.0f, vl);
    i7 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i8, vl);
    vfloat32m1_t vi8x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi8x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi8x2;
    vi8x0 = __riscv_vfslide1up_vf_f32m1(vi8x0, 0.0f, vl);
    i8 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i9, vl);
    vfloat32m1_t vi9x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi9x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi9x2;
    vi9x0 = __riscv_vfslide1up_vf_f32m1(vi9x0, 0.0f, vl);
    i9 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i10, vl);
    vfloat32m1_t vi10x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi10x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi10x2;
    vi10x0 = __riscv_vfslide1up_vf_f32m1(vi10x0, 0.0f, vl);
    i10 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i11, vl);
    vfloat32m1_t vi11x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi11x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi11x2;
    vi11x0 = __riscv_vfslide1up_vf_f32m1(vi11x0, 0.0f, vl);
    i11 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i12, vl);
    vfloat32m1_t vi12x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi12x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi12x2;
    vi12x0 = __riscv_vfslide1up_vf_f32m1(vi12x0, 0.0f, vl);
    i12 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i13, vl);
    vfloat32m1_t vi13x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi13x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi13x2;
    vi13x0 = __riscv_vfslide1up_vf_f32m1(vi13x0, 0.0f, vl);
    i13 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i14, vl);
    vfloat32m1_t vi14x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi14x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi14x2;
    vi14x0 = __riscv_vfslide1up_vf_f32m1(vi14x0, 0.0f, vl);
    i14 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i15, vl);
    vfloat32m1_t vi15x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi15x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi15x2;
    vi15x0 = __riscv_vfslide1up_vf_f32m1(vi15x0, 0.0f, vl);
    i15 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m1x2(i16, vl);
    vfloat32m1_t vi16x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
    vfloat32m1_t vi16x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
    vfloat32m1_t vi16x2;
    vi16x0 = __riscv_vfslide1up_vf_f32m1(vi16x0, 0.0f, vl);
    i16 += 2 * vl - 1;

    while (w > 2 * vlmax) {
      vi0x2 = __riscv_vfslide1down_vf_f32m1(vi0x0, *i0, vl);
      vi1x2 = __riscv_vfslide1down_vf_f32m1(vi1x0, *i1, vl);
      vi2x2 = __riscv_vfslide1down_vf_f32m1(vi2x0, *i2, vl);
      vi3x2 = __riscv_vfslide1down_vf_f32m1(vi3x0, *i3, vl);
      vi4x2 = __riscv_vfslide1down_vf_f32m1(vi4x0, *i4, vl);
      vi5x2 = __riscv_vfslide1down_vf_f32m1(vi5x0, *i5, vl);
      vi6x2 = __riscv_vfslide1down_vf_f32m1(vi6x0, *i6, vl);
      vi7x2 = __riscv_vfslide1down_vf_f32m1(vi7x0, *i7, vl);
      vi8x2 = __riscv_vfslide1down_vf_f32m1(vi8x0, *i8, vl);
      vi9x2 = __riscv_vfslide1down_vf_f32m1(vi9x0, *i9, vl);
      vi10x2 = __riscv_vfslide1down_vf_f32m1(vi10x0, *i10, vl);
      vi11x2 = __riscv_vfslide1down_vf_f32m1(vi11x0, *i11, vl);
      vi12x2 = __riscv_vfslide1down_vf_f32m1(vi12x0, *i12, vl);
      vi13x2 = __riscv_vfslide1down_vf_f32m1(vi13x0, *i13, vl);
      vi14x2 = __riscv_vfslide1down_vf_f32m1(vi14x0, *i14, vl);
      vi15x2 = __riscv_vfslide1down_vf_f32m1(vi15x0, *i15, vl);
      vi16x2 = __riscv_vfslide1down_vf_f32m1(vi16x0, *i16, vl);

      vfloat32m1_t vo0p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo1p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo2p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo3p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo4p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo5p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo6p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo7p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk00, vi0x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk00, vi2x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk00, vi4x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk00, vi6x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk00, vi8x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk00, vi10x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk00, vi12x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk00, vi14x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk10, vi1x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk10, vi3x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk10, vi5x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk10, vi7x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk10, vi9x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk10, vi11x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk10, vi13x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk10, vi15x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk20, vi2x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk20, vi4x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk20, vi6x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk20, vi8x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk20, vi10x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk20, vi12x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk20, vi14x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk20, vi16x0, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk01, vi0x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk01, vi2x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk01, vi4x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk01, vi6x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk01, vi8x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk01, vi10x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk01, vi12x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk01, vi14x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk11, vi1x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk11, vi3x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk11, vi5x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk11, vi7x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk11, vi9x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk11, vi11x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk11, vi13x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk11, vi15x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk21, vi2x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk21, vi4x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk21, vi6x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk21, vi8x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk21, vi10x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk21, vi12x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk21, vi14x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk21, vi16x1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk02, vi0x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk02, vi2x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk02, vi4x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk02, vi6x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk02, vi8x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk02, vi10x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk02, vi12x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk02, vi14x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk12, vi1x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk12, vi3x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk12, vi5x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk12, vi7x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk12, vi9x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk12, vi11x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk12, vi13x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk12, vi15x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk22, vi2x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk22, vi4x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk22, vi6x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk22, vi8x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk22, vi10x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk22, vi12x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk22, vi14x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk22, vi16x2, vl);


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

      w -= 2 * vl;
      vl = __riscv_vsetvl_e32m1((w + 1) >> 1);
      tuple = __riscv_vlseg2e32_v_f32m1x2(i0, vl);
      vi0x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi0x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i0 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i1, vl);
      vi1x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi1x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i1 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i2, vl);
      vi2x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi2x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i2 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i3, vl);
      vi3x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi3x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i3 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i4, vl);
      vi4x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi4x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i4 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i5, vl);
      vi5x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi5x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i5 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i6, vl);
      vi6x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi6x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i6 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i7, vl);
      vi7x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi7x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i7 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i8, vl);
      vi8x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi8x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i8 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i9, vl);
      vi9x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi9x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i9 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i10, vl);
      vi10x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi10x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i10 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i11, vl);
      vi11x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi11x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i11 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i12, vl);
      vi12x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi12x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i12 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i13, vl);
      vi13x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi13x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i13 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i14, vl);
      vi14x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi14x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i14 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i15, vl);
      vi15x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi15x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i15 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m1x2(i16, vl);
      vi16x0 = __riscv_vget_v_f32m1x2_f32m1(tuple, 0);
      vi16x1 = __riscv_vget_v_f32m1x2_f32m1(tuple, 1);
      i16 += 2 * vl;
    }
    //  Always process the last tile separately to account for right edge.
    assert(w <= 2*vlmax);
    {
      if (w & 1) {
        vi0x2 = __riscv_vfslide1down_vf_f32m1(vi0x0, 0.0f, vl);
        vi1x2 = __riscv_vfslide1down_vf_f32m1(vi1x0, 0.0f, vl);
        vi2x2 = __riscv_vfslide1down_vf_f32m1(vi2x0, 0.0f, vl);
        vi3x2 = __riscv_vfslide1down_vf_f32m1(vi3x0, 0.0f, vl);
        vi4x2 = __riscv_vfslide1down_vf_f32m1(vi4x0, 0.0f, vl);
        vi5x2 = __riscv_vfslide1down_vf_f32m1(vi5x0, 0.0f, vl);
        vi6x2 = __riscv_vfslide1down_vf_f32m1(vi6x0, 0.0f, vl);
        vi7x2 = __riscv_vfslide1down_vf_f32m1(vi7x0, 0.0f, vl);
        vi8x2 = __riscv_vfslide1down_vf_f32m1(vi8x0, 0.0f, vl);
        vi9x2 = __riscv_vfslide1down_vf_f32m1(vi9x0, 0.0f, vl);
        vi10x2 = __riscv_vfslide1down_vf_f32m1(vi10x0, 0.0f, vl);
        vi11x2 = __riscv_vfslide1down_vf_f32m1(vi11x0, 0.0f, vl);
        vi12x2 = __riscv_vfslide1down_vf_f32m1(vi12x0, 0.0f, vl);
        vi13x2 = __riscv_vfslide1down_vf_f32m1(vi13x0, 0.0f, vl);
        vi14x2 = __riscv_vfslide1down_vf_f32m1(vi14x0, 0.0f, vl);
        vi15x2 = __riscv_vfslide1down_vf_f32m1(vi15x0, 0.0f, vl);
        vi16x2 = __riscv_vfslide1down_vf_f32m1(vi16x0, 0.0f, vl);
      } else {
        vi0x2 = __riscv_vfslide1down_vf_f32m1(vi0x0, *i0, vl);
        i0++;
        vi1x2 = __riscv_vfslide1down_vf_f32m1(vi1x0, *i1, vl);
        i1++;
        vi2x2 = __riscv_vfslide1down_vf_f32m1(vi2x0, *i2, vl);
        i2++;
        vi3x2 = __riscv_vfslide1down_vf_f32m1(vi3x0, *i3, vl);
        i3++;
        vi4x2 = __riscv_vfslide1down_vf_f32m1(vi4x0, *i4, vl);
        i4++;
        vi5x2 = __riscv_vfslide1down_vf_f32m1(vi5x0, *i5, vl);
        i5++;
        vi6x2 = __riscv_vfslide1down_vf_f32m1(vi6x0, *i6, vl);
        i6++;
        vi7x2 = __riscv_vfslide1down_vf_f32m1(vi7x0, *i7, vl);
        i7++;
        vi8x2 = __riscv_vfslide1down_vf_f32m1(vi8x0, *i8, vl);
        i8++;
        vi9x2 = __riscv_vfslide1down_vf_f32m1(vi9x0, *i9, vl);
        i9++;
        vi10x2 = __riscv_vfslide1down_vf_f32m1(vi10x0, *i10, vl);
        i10++;
        vi11x2 = __riscv_vfslide1down_vf_f32m1(vi11x0, *i11, vl);
        i11++;
        vi12x2 = __riscv_vfslide1down_vf_f32m1(vi12x0, *i12, vl);
        i12++;
        vi13x2 = __riscv_vfslide1down_vf_f32m1(vi13x0, *i13, vl);
        i13++;
        vi14x2 = __riscv_vfslide1down_vf_f32m1(vi14x0, *i14, vl);
        i14++;
        vi15x2 = __riscv_vfslide1down_vf_f32m1(vi15x0, *i15, vl);
        i15++;
        vi16x2 = __riscv_vfslide1down_vf_f32m1(vi16x0, *i16, vl);
        i16++;
      }

      vfloat32m1_t vo0p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo1p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo2p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo3p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo4p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo5p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo6p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);
      vfloat32m1_t vo7p0 = __riscv_vfmv_v_f_f32m1(vbias, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk00, vi0x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk00, vi2x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk00, vi4x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk00, vi6x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk00, vi8x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk00, vi10x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk00, vi12x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk00, vi14x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk10, vi1x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk10, vi3x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk10, vi5x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk10, vi7x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk10, vi9x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk10, vi11x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk10, vi13x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk10, vi15x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk20, vi2x0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk20, vi4x0, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk20, vi6x0, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk20, vi8x0, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk20, vi10x0, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk20, vi12x0, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk20, vi14x0, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk20, vi16x0, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk01, vi0x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk01, vi2x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk01, vi4x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk01, vi6x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk01, vi8x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk01, vi10x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk01, vi12x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk01, vi14x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk11, vi1x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk11, vi3x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk11, vi5x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk11, vi7x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk11, vi9x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk11, vi11x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk11, vi13x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk11, vi15x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk21, vi2x1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk21, vi4x1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk21, vi6x1, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk21, vi8x1, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk21, vi10x1, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk21, vi12x1, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk21, vi14x1, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk21, vi16x1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk02, vi0x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk02, vi2x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk02, vi4x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk02, vi6x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk02, vi8x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk02, vi10x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk02, vi12x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk02, vi14x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk12, vi1x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk12, vi3x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk12, vi5x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk12, vi7x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk12, vi9x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk12, vi11x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk12, vi13x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk12, vi15x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m1(vo0p0, vk22, vi2x2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m1(vo1p0, vk22, vi4x2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m1(vo2p0, vk22, vi6x2, vl);
      vo3p0 = __riscv_vfmacc_vf_f32m1(vo3p0, vk22, vi8x2, vl);
      vo4p0 = __riscv_vfmacc_vf_f32m1(vo4p0, vk22, vi10x2, vl);
      vo5p0 = __riscv_vfmacc_vf_f32m1(vo5p0, vk22, vi12x2, vl);
      vo6p0 = __riscv_vfmacc_vf_f32m1(vo6p0, vk22, vi14x2, vl);
      vo7p0 = __riscv_vfmacc_vf_f32m1(vo7p0, vk22, vi16x2, vl);


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

    i0 = (const float*) ((uintptr_t) i15);
    i1 = (const float*) ((uintptr_t) i16);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);
    i8 = (const float*) ((uintptr_t) i7 + input_width);
    i9 = (const float*) ((uintptr_t) i8 + input_width);
    i10 = (const float*) ((uintptr_t) i9 + input_width);
    i11 = (const float*) ((uintptr_t) i10 + input_width);
    i12 = (const float*) ((uintptr_t) i11 + input_width);
    i13 = (const float*) ((uintptr_t) i12 + input_width);
    i14 = (const float*) ((uintptr_t) i13 + input_width);
    i15 = (const float*) ((uintptr_t) i14 + input_width);
    i16 = (const float*) ((uintptr_t) i15 + input_width);

    o0 = o7;
    o1 = (float*) ((uintptr_t) o0 + output_width);
    o2 = (float*) ((uintptr_t) o1 + output_width);
    o3 = (float*) ((uintptr_t) o2 + output_width);
    o4 = (float*) ((uintptr_t) o3 + output_width);
    o5 = (float*) ((uintptr_t) o4 + output_width);
    o6 = (float*) ((uintptr_t) o5 + output_width);
    o7 = (float*) ((uintptr_t) o6 + output_width);

    output_height = doz(output_height, 8);
    padded_input_height = doz(padded_input_height, 16);
  } while (output_height != 0);
}
