// Auto-generated file. Do not edit!
//   Template: src/f32-maxpool/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include "xnnpack/maxpool.h"
#include <riscv_vector.h>

void xnn_f32_maxpool_minmax_ukernel_9p8x__rvv_c2v(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  do {
    float* o = output;
    {
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
      if (kernel_elements < 2) {
        i1 = i0;
      }
      if (kernel_elements <= 2) {
        i2 = i0;
      }
      if (kernel_elements < 4) {
        i3 = i0;
      }
      if (kernel_elements <= 4) {
        i4 = i0;
      }
      if (kernel_elements < 6) {
        i5 = i0;
      }
      if (kernel_elements <= 6) {
        i6 = i0;
      }
      if (kernel_elements < 8) {
        i7 = i0;
      }
      if (kernel_elements <= 8) {
        i8 = i0;
      }

      size_t c = channels;
      do {
        int32_t n = __riscv_vsetvl_e32m2(c);

        vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i0, n); i0 += n;
        vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i1, n); i1 += n;
        vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i2, n); i2 += n;
        vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i3, n); i3 += n;
        vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i4, n); i4 += n;
        vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i5, n); i5 += n;
        vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i6, n); i6 += n;
        vfloat32m2_t i7_f32v = __riscv_vle32_v_f32m2(i7, n); i7 += n;
        vfloat32m2_t i8_f32v = __riscv_vle32_v_f32m2(i8, n); i8 += n;

        vfloat32m2_t max01_f32v = __riscv_vfmax_vv_f32m2(i0_f32v, i1_f32v, n);
        vfloat32m2_t max23_f32v = __riscv_vfmax_vv_f32m2(i2_f32v, i3_f32v, n);
        vfloat32m2_t max45_f32v = __riscv_vfmax_vv_f32m2(i4_f32v, i5_f32v, n);
        vfloat32m2_t max67_f32v = __riscv_vfmax_vv_f32m2(i6_f32v, i7_f32v, n);
        vfloat32m2_t max018_f32v = __riscv_vfmax_vv_f32m2(max01_f32v, i8_f32v, n);

        vfloat32m2_t max2345_f32v = __riscv_vfmax_vv_f32m2(max23_f32v, max45_f32v, n);
        vfloat32m2_t max01678_f32v = __riscv_vfmax_vv_f32m2(max67_f32v, max018_f32v, n);
        vfloat32m2_t out_f32v = __riscv_vfmax_vv_f32m2(max2345_f32v, max01678_f32v, n);
        out_f32v = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(out_f32v, output_min, n), output_max, n);
        __riscv_vse32_v_f32m2(o, out_f32v, n); o += n;

        c -= n;
      } while (c != 0);
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
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
      if (k < 8) {
        i7 = i0;
      }

      o = output;
      size_t c = channels;
      do {
        int32_t n = __riscv_vsetvl_e32m2(c);

        vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i0, n); i0 += n;
        vfloat32m2_t i1_f32v = __riscv_vle32_v_f32m2(i1, n); i1 += n;
        vfloat32m2_t i2_f32v = __riscv_vle32_v_f32m2(i2, n); i2 += n;
        vfloat32m2_t i3_f32v = __riscv_vle32_v_f32m2(i3, n); i3 += n;
        vfloat32m2_t i4_f32v = __riscv_vle32_v_f32m2(i4, n); i4 += n;
        vfloat32m2_t i5_f32v = __riscv_vle32_v_f32m2(i5, n); i5 += n;
        vfloat32m2_t i6_f32v = __riscv_vle32_v_f32m2(i6, n); i6 += n;
        vfloat32m2_t i7_f32v = __riscv_vle32_v_f32m2(i7, n); i7 += n;
        vfloat32m2_t i8_f32v = __riscv_vle32_v_f32m2(o, n);

        vfloat32m2_t max01_f32v = __riscv_vfmax_vv_f32m2(i0_f32v, i1_f32v, n);
        vfloat32m2_t max23_f32v = __riscv_vfmax_vv_f32m2(i2_f32v, i3_f32v, n);
        vfloat32m2_t max45_f32v = __riscv_vfmax_vv_f32m2(i4_f32v, i5_f32v, n);
        vfloat32m2_t max67_f32v = __riscv_vfmax_vv_f32m2(i6_f32v, i7_f32v, n);
        vfloat32m2_t max018_f32v = __riscv_vfmax_vv_f32m2(max01_f32v, i8_f32v, n);

        vfloat32m2_t max2345_f32v = __riscv_vfmax_vv_f32m2(max23_f32v, max45_f32v, n);
        vfloat32m2_t max01678_f32v = __riscv_vfmax_vv_f32m2(max67_f32v, max018_f32v, n);
        vfloat32m2_t out_f32v = __riscv_vfmax_vv_f32m2(max2345_f32v, max01678_f32v, n);
        out_f32v = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(out_f32v, output_min, n), output_max, n);
        __riscv_vse32_v_f32m2(o, out_f32v, n); o += n;

        c -= n;
      } while (c != 0);
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}
