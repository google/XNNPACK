// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-avgpool/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include "src/xnnpack/maxpool.h"
#include <riscv_vector.h>


void xnn_f32_avgpool_minmax_ukernel_9p__rvv_u2v(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* zero,
    const float* multiplier,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f32_scaleminmax_params* restrict params)
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const float min = params->scalar.min;
  const float max = params->scalar.max;
  float scale = params->scalar.scale;

  do {
    // Start with the previous output as the zero buffer.
    const float* prev_output = zero;

    const float** i = input;

    // Passes 0 - n-1: load the output, add 9 inputs.
    size_t k = kernel_elements;
    for (; k > 9; k -= 9) {
      const float* i0 = *i++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *i++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *i++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *i++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *i++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *i++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *i++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *i++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      const float* i8 = *i++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const float*) ((uintptr_t) i8 + input_offset);
      }

      float* o = output;
      size_t c = channels;
      do {
        size_t vl = __riscv_vsetvl_e32m2(c);
        vfloat32m2_t vi0 = __riscv_vle32_v_f32m2(i0, vl); i0 += vl;
        vfloat32m2_t vi1 = __riscv_vle32_v_f32m2(i1, vl); i1 += vl;
        vfloat32m2_t vi2 = __riscv_vle32_v_f32m2(i2, vl); i2 += vl;
        vfloat32m2_t vi3 = __riscv_vle32_v_f32m2(i3, vl); i3 += vl;
        vfloat32m2_t vi4 = __riscv_vle32_v_f32m2(i4, vl); i4 += vl;
        vfloat32m2_t vi5 = __riscv_vle32_v_f32m2(i5, vl); i5 += vl;
        vfloat32m2_t vi6 = __riscv_vle32_v_f32m2(i6, vl); i6 += vl;
        vfloat32m2_t vi7 = __riscv_vle32_v_f32m2(i7, vl); i7 += vl;
        vfloat32m2_t vi8 = __riscv_vle32_v_f32m2(i8, vl); i8 += vl;
        vfloat32m2_t vprev = __riscv_vle32_v_f32m2(prev_output, vl); prev_output += vl;

        vfloat32m2_t vsum01 = __riscv_vfadd(vi0, vi1, vl);
        vfloat32m2_t vsum23 = __riscv_vfadd(vi2, vi3, vl);
        vfloat32m2_t vsum45 = __riscv_vfadd(vi4, vi5, vl);
        vfloat32m2_t vsum67 = __riscv_vfadd(vi6, vi7, vl);
        vfloat32m2_t vsum018 = __riscv_vfadd(vsum01, vi8, vl);

        vfloat32m2_t vsum2345 = __riscv_vfadd(vsum23, vsum45, vl);
        vfloat32m2_t vsum01678 = __riscv_vfadd(vsum67, vsum018, vl);
        vfloat32m2_t vsum012345678 = __riscv_vfadd(vsum2345, vsum01678, vl);
        vfloat32m2_t vacc = __riscv_vfadd(vprev, vsum012345678, vl);
        __riscv_vse32_v_f32m2(o, vacc, vl); o += vl;

        c -= vl;
      } while (c != 0);

      // Subsequent passes read from the previous output.
      prev_output = output;
    }

    // Final pass: load the output, add remaining kernel elements, apply scaling/min/max
    const float* i0 = 0 < k ? *i++ : zero;
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = 1 < k ? *i++ : zero;
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = 2 < k ? *i++ : zero;
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = 3 < k ? *i++ : zero;
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = 4 < k ? *i++ : zero;
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = 5 < k ? *i++ : zero;
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = 6 < k ? *i++ : zero;
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = 7 < k ? *i++ : zero;
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = 8 < k ? *i++ : zero;
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }

    if (multiplier != NULL) {
      scale = *multiplier++;
    }
    float* o = output;
    size_t c = channels;
    do {
      size_t vl = __riscv_vsetvl_e32m2(c);
      vfloat32m2_t vi0 = __riscv_vle32_v_f32m2(i0, vl); i0 += vl;
      vfloat32m2_t vi1 = __riscv_vle32_v_f32m2(i1, vl); i1 += vl;
      vfloat32m2_t vi2 = __riscv_vle32_v_f32m2(i2, vl); i2 += vl;
      vfloat32m2_t vi3 = __riscv_vle32_v_f32m2(i3, vl); i3 += vl;
      vfloat32m2_t vi4 = __riscv_vle32_v_f32m2(i4, vl); i4 += vl;
      vfloat32m2_t vi5 = __riscv_vle32_v_f32m2(i5, vl); i5 += vl;
      vfloat32m2_t vi6 = __riscv_vle32_v_f32m2(i6, vl); i6 += vl;
      vfloat32m2_t vi7 = __riscv_vle32_v_f32m2(i7, vl); i7 += vl;
      vfloat32m2_t vi8 = __riscv_vle32_v_f32m2(i8, vl); i8 += vl;
      vfloat32m2_t vprev = __riscv_vle32_v_f32m2(prev_output, vl); prev_output += vl;

      vfloat32m2_t vsum01 = __riscv_vfadd(vi0, vi1, vl);
      vfloat32m2_t vsum23 = __riscv_vfadd(vi2, vi3, vl);
      vfloat32m2_t vsum45 = __riscv_vfadd(vi4, vi5, vl);
      vfloat32m2_t vsum67 = __riscv_vfadd(vi6, vi7, vl);
      vfloat32m2_t vsum018 = __riscv_vfadd(vsum01, vi8, vl);

      vfloat32m2_t vsum2345 = __riscv_vfadd(vsum23, vsum45, vl);
      vfloat32m2_t vsum01678 = __riscv_vfadd(vsum67, vsum018, vl);
      vfloat32m2_t vsum012345678 = __riscv_vfadd(vsum2345, vsum01678, vl);
      vfloat32m2_t vacc = __riscv_vfadd(vprev, vsum012345678, vl);

      vacc = __riscv_vfmul(vacc, scale, vl);
      vacc = __riscv_vfmax(vacc, min, vl);
      vacc = __riscv_vfmin(vacc, max, vl);

      __riscv_vse32_v_f32m2(o, vacc, vl); o += vl;

      c -= vl;
    } while (c != 0);

    input = (const float**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
