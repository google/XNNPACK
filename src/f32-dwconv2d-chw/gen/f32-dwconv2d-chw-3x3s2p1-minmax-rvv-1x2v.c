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


void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__rvv_1x2v(
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


  const float* i0 = (const float*) ((uintptr_t) input - ((-padding_top) & input_width));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);

  float* o0 = output;

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }

    size_t w = input_width >> XNN_LOG2_SIZEOF_FLOAT;
    size_t vl =  __riscv_vsetvl_e32m2((w + 1) >> 1);
    vfloat32m2x2_t tuple;
    tuple = __riscv_vlseg2e32_v_f32m2x2(i0, vl);
    vfloat32m2_t vi0x1 = __riscv_vget_v_f32m2x2_f32m2(tuple, 0);
    vfloat32m2_t vi0x0 = __riscv_vget_v_f32m2x2_f32m2(tuple, 1);
    vfloat32m2_t vi0x2;
    vi0x0 = __riscv_vfslide1up_vf_f32m2(vi0x0, 0.0f, vl);
    i0 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m2x2(i1, vl);
    vfloat32m2_t vi1x1 = __riscv_vget_v_f32m2x2_f32m2(tuple, 0);
    vfloat32m2_t vi1x0 = __riscv_vget_v_f32m2x2_f32m2(tuple, 1);
    vfloat32m2_t vi1x2;
    vi1x0 = __riscv_vfslide1up_vf_f32m2(vi1x0, 0.0f, vl);
    i1 += 2 * vl - 1;
    tuple = __riscv_vlseg2e32_v_f32m2x2(i2, vl);
    vfloat32m2_t vi2x1 = __riscv_vget_v_f32m2x2_f32m2(tuple, 0);
    vfloat32m2_t vi2x0 = __riscv_vget_v_f32m2x2_f32m2(tuple, 1);
    vfloat32m2_t vi2x2;
    vi2x0 = __riscv_vfslide1up_vf_f32m2(vi2x0, 0.0f, vl);
    i2 += 2 * vl - 1;

    while (w > 2 * vlmax) {
      vi0x2 = __riscv_vfslide1down_vf_f32m2(vi0x0, *i0, vl);
      vi1x2 = __riscv_vfslide1down_vf_f32m2(vi1x0, *i1, vl);
      vi2x2 = __riscv_vfslide1down_vf_f32m2(vi2x0, *i2, vl);

      vfloat32m2_t vo0p0 = __riscv_vfmv_v_f_f32m2(vbias, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk00, vi0x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk10, vi1x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk20, vi2x0, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk01, vi0x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk11, vi1x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk21, vi2x1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk02, vi0x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk12, vi1x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk22, vi2x2, vl);


      vo0p0 = __riscv_vfmax_vf_f32m2(vo0p0, vmin, vl);

      vo0p0 = __riscv_vfmin_vf_f32m2(vo0p0, vmax, vl);

      __riscv_vse32_v_f32m2(o0, vo0p0, vl);
      o0 += vl;

      w -= 2 * vl;
      vl = __riscv_vsetvl_e32m2((w + 1) >> 1);
      tuple = __riscv_vlseg2e32_v_f32m2x2(i0, vl);
      vi0x0 = __riscv_vget_v_f32m2x2_f32m2(tuple, 0);
      vi0x1 = __riscv_vget_v_f32m2x2_f32m2(tuple, 1);
      i0 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m2x2(i1, vl);
      vi1x0 = __riscv_vget_v_f32m2x2_f32m2(tuple, 0);
      vi1x1 = __riscv_vget_v_f32m2x2_f32m2(tuple, 1);
      i1 += 2 * vl;
      tuple = __riscv_vlseg2e32_v_f32m2x2(i2, vl);
      vi2x0 = __riscv_vget_v_f32m2x2_f32m2(tuple, 0);
      vi2x1 = __riscv_vget_v_f32m2x2_f32m2(tuple, 1);
      i2 += 2 * vl;
    }
    //  Always process the last tile separately to account for right edge.
    assert(w <= 2*vlmax);
    {
      if (w & 1) {
        vi0x2 = __riscv_vfslide1down_vf_f32m2(vi0x0, 0.0f, vl);
        vi1x2 = __riscv_vfslide1down_vf_f32m2(vi1x0, 0.0f, vl);
        vi2x2 = __riscv_vfslide1down_vf_f32m2(vi2x0, 0.0f, vl);
      } else {
        vi0x2 = __riscv_vfslide1down_vf_f32m2(vi0x0, *i0, vl);
        i0++;
        vi1x2 = __riscv_vfslide1down_vf_f32m2(vi1x0, *i1, vl);
        i1++;
        vi2x2 = __riscv_vfslide1down_vf_f32m2(vi2x0, *i2, vl);
        i2++;
      }

      vfloat32m2_t vo0p0 = __riscv_vfmv_v_f_f32m2(vbias, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk00, vi0x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk10, vi1x0, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk20, vi2x0, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk01, vi0x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk11, vi1x1, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk21, vi2x1, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk02, vi0x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk12, vi1x2, vl);
      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, vk22, vi2x2, vl);


      vo0p0 = __riscv_vfmax_vf_f32m2(vo0p0, vmin, vl);

      vo0p0 = __riscv_vfmin_vf_f32m2(vo0p0, vmax, vl);

      __riscv_vse32_v_f32m2(o0, vo0p0, vl);
      o0 += vl;
    }

    i0 = (const float*) ((uintptr_t) i1);
    i1 = (const float*) ((uintptr_t) i2);
    i2 = (const float*) ((uintptr_t) i1 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}
