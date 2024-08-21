// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/gemm.h"
#include "xnnpack/igemm.h"
#include "xnnpack/math.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/microparams.h"
#include "xnnpack/pavgpool.h"
#include "xnnpack/prelu.h"
#include "xnnpack/reduce.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vbinary.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/vmulcaddc.h"
#include "xnnpack/vunary.h"


void xnn_f32_avgpool_minmax_ukernel_9p8x__wasm_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const float vscale = params->scalar.scale;
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  do {
    {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      const float* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const float*) ((uintptr_t) i8 + input_offset);
      }

      float* b = buffer;
      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vi8 = *i8++;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum018 = vsum01 + vi8;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum01678 = vsum018 + vsum67;
        const float vsum = vsum2345 + vsum01678;

        *b++ = vsum;
      } while (--c != 0);
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      float* b = buffer;
      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vacc = *b;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum01a = vsum01 + vacc;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum0167a = vsum01a + vsum67;
        const float vsum = vsum2345 + vsum0167a;

        *b++ = vsum;
      } while (--c != 0);
    }

    {
      const float* i0 = input[0];
      assert(i0 != NULL);
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      input = (const float**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      size_t c = channels;
      float* b = buffer;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vacc = *b++;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum01a = vsum01 + vacc;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum0167a = vsum01a + vsum67;
        const float vsum = vsum2345 + vsum0167a;

        float vout = vsum * vscale;
        vout = __builtin_wasm_max_f32(vout, vmin);
        vout = __builtin_wasm_min_f32(vout, vmax);

        *output++ = vout;
      } while (--c != 0);
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_avgpool_minmax_ukernel_9x__wasm_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const float vscale = params->scalar.scale;
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    input = (const float**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }

    size_t c = channels;
    do {
      const float vi0 = *i0++;
      const float vi1 = *i1++;
      const float vi2 = *i2++;
      const float vi3 = *i3++;
      const float vi4 = *i4++;
      const float vi5 = *i5++;
      const float vi6 = *i6++;
      const float vi7 = *i7++;
      const float vi8 = *i8++;

      const float vsum01 = vi0 + vi1;
      const float vsum23 = vi2 + vi3;
      const float vsum45 = vi4 + vi5;
      const float vsum67 = vi6 + vi7;
      const float vsum018 = vsum01 + vi8;
      const float vsum2345 = vsum23 + vsum45;
      const float vsum01678 = vsum018 + vsum67;
      const float vsum = vsum2345 + vsum01678;

      float vout = vsum * vscale;
      vout = __builtin_wasm_max_f32(vout, vmin);
      vout = __builtin_wasm_min_f32(vout, vmax);

      *output++ = vout;
    } while (--c != 0);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_dwconv_minmax_ukernel_25p1c__wasm_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);

      const float vi4 = *i4++;
      const float vk4 = w[5];
      vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);

      const float vi5 = *i5++;
      const float vk5 = w[6];
      vacc0p1 = math_muladd_f32(vi5, vk5, vacc0p1);

      const float vi6 = *i6++;
      const float vk6 = w[7];
      vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);

      const float vi7 = *i7++;
      const float vk7 = w[8];
      vacc0p1 = math_muladd_f32(vi7, vk7, vacc0p1);

      const float vi8 = *i8++;
      const float vk8 = w[9];
      vacc0p0 = math_muladd_f32(vi8, vk8, vacc0p0);

      const float vi9 = *i9++;
      const float vk9 = w[10];
      vacc0p1 = math_muladd_f32(vi9, vk9, vacc0p1);

      const float vi10 = *i10++;
      const float vk10 = w[11];
      vacc0p0 = math_muladd_f32(vi10, vk10, vacc0p0);

      const float vi11 = *i11++;
      const float vk11 = w[12];
      vacc0p1 = math_muladd_f32(vi11, vk11, vacc0p1);

      const float vi12 = *i12++;
      const float vk12 = w[13];
      vacc0p0 = math_muladd_f32(vi12, vk12, vacc0p0);

      const float vi13 = *i13++;
      const float vk13 = w[14];
      vacc0p1 = math_muladd_f32(vi13, vk13, vacc0p1);

      const float vi14 = *i14++;
      const float vk14 = w[15];
      vacc0p0 = math_muladd_f32(vi14, vk14, vacc0p0);

      const float vi15 = *i15++;
      const float vk15 = w[16];
      vacc0p1 = math_muladd_f32(vi15, vk15, vacc0p1);

      const float vi16 = *i16++;
      const float vk16 = w[17];
      vacc0p0 = math_muladd_f32(vi16, vk16, vacc0p0);

      const float vi17 = *i17++;
      const float vk17 = w[18];
      vacc0p1 = math_muladd_f32(vi17, vk17, vacc0p1);

      const float vi18 = *i18++;
      const float vk18 = w[19];
      vacc0p0 = math_muladd_f32(vi18, vk18, vacc0p0);

      const float vi19 = *i19++;
      const float vk19 = w[20];
      vacc0p1 = math_muladd_f32(vi19, vk19, vacc0p1);

      const float vi20 = *i20++;
      const float vk20 = w[21];
      vacc0p0 = math_muladd_f32(vi20, vk20, vacc0p0);

      const float vi21 = *i21++;
      const float vk21 = w[22];
      vacc0p1 = math_muladd_f32(vi21, vk21, vacc0p1);

      const float vi22 = *i22++;
      const float vk22 = w[23];
      vacc0p0 = math_muladd_f32(vi22, vk22, vacc0p0);

      const float vi23 = *i23++;
      const float vk23 = w[24];
      vacc0p1 = math_muladd_f32(vi23, vk23, vacc0p1);

      const float vi24 = *i24++;
      const float vk24 = w[25];
      vacc0p0 = math_muladd_f32(vi24, vk24, vacc0p0);

      w += 26;

      vacc0p0 += vacc0p1;

      float vacc0 = __builtin_wasm_max_f32(vacc0p0, vmin);
      vacc0 = __builtin_wasm_min_f32(vacc0, vmax);
      *output++ = vacc0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_3p1c__wasm_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      w += 4;

      vacc0p0 += vacc0p1;

      float vacc0 = __builtin_wasm_max_f32(vacc0p0, vmin);
      vacc0 = __builtin_wasm_min_f32(vacc0, vmax);
      *output++ = vacc0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_4p1c__wasm_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);

      w += 5;

      vacc0p0 += vacc0p1;

      float vacc0 = __builtin_wasm_max_f32(vacc0p0, vmin);
      vacc0 = __builtin_wasm_min_f32(vacc0, vmax);
      *output++ = vacc0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* w = weights;

    // First pass to process 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      // Process c channels and write to buffer.
      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = w[0];

        const float vi0 = *i0++;
        const float vk0 = w[1];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[2];
        float vacc0p1 = vi1 * vk1;
        const float vi2 = *i2++;
        const float vk2 = w[3];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);
        const float vi3 = *i3++;
        const float vk3 = w[4];
        vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);
        const float vi4 = *i4++;
        const float vk4 = w[5];
        vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);

        w += 6;

        // Add up all accumulators to vacc0p0
        vacc0p0 = vacc0p0 + vacc0p1;

        *b++ = vacc0p0;
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = *b;

        const float vi0 = *i0++;
        const float vk0 = w[0];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[1];
        float vacc0p1 = vi1 * vk1;
        const float vi2 = *i2++;
        const float vk2 = w[2];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);
        const float vi3 = *i3++;
        const float vk3 = w[3];
        vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);
        const float vi4 = *i4++;
        const float vk4 = w[4];
        vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);

        // Add up all accumulators to vacc0p0
        vacc0p0 = vacc0p0 + vacc0p1;

        w += 5;
        *b++ = vacc0p0;
      }
    }

    // Last pass to process up to 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }

      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = *b++;

        const float vi0 = *i0++;
        const float vk0 = w[0];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[1];
        float vacc0p1 = vi1 * vk1;
        const float vi2 = *i2++;
        const float vk2 = w[2];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);
        const float vi3 = *i3++;
        const float vk3 = w[3];
        vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);
        const float vi4 = *i4++;
        const float vk4 = w[4];
        vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);

        w += 5;

        // Add up all accumulators to vacc0p0
        vacc0p0 = vacc0p0 + vacc0p1;

        float vacc0 = __builtin_wasm_max_f32(vacc0p0, vmin);
        vacc0 = __builtin_wasm_min_f32(vacc0, vmax);
        *output++ = vacc0;
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_9p1c__wasm_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);

      const float vi4 = *i4++;
      const float vk4 = w[5];
      vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);

      const float vi5 = *i5++;
      const float vk5 = w[6];
      vacc0p1 = math_muladd_f32(vi5, vk5, vacc0p1);

      const float vi6 = *i6++;
      const float vk6 = w[7];
      vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);

      const float vi7 = *i7++;
      const float vk7 = w[8];
      vacc0p1 = math_muladd_f32(vi7, vk7, vacc0p1);

      const float vi8 = *i8++;
      const float vk8 = w[9];
      vacc0p0 = math_muladd_f32(vi8, vk8, vacc0p0);

      w += 10;

      vacc0p0 += vacc0p1;

      float vacc0 = __builtin_wasm_max_f32(vacc0p0, vmin);
      vacc0 = __builtin_wasm_min_f32(vacc0, vmax);
      *output++ = vacc0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* buffer,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows > 7);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - channels * sizeof(float);

  float* b = buffer;
  size_t c = channels;
  do {
    const float vi0 = *i0++;
    const float vi1 = *i1++;
    const float vi2 = *i2++;
    const float vi3 = *i3++;
    const float vi4 = *i4++;
    const float vi5 = *i5++;
    const float vi6 = *i6++;

    const float vsum01 = vi0 + vi1;
    const float vsum23 = vi2 + vi3;
    const float vsum45 = vi4 + vi5;

    const float vsum016 = vsum01 + vi6;
    const float vsum2345 = vsum23 + vsum45;

    const float vsum = vsum016 + vsum2345;

    *b++ = vsum;
  } while (--c != 0);
  for (rows -= 7; rows > 7; rows -= 7) {
    b = buffer;

    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_increment);
    i3 = (const float*) ((uintptr_t) i3 + input_increment);
    i4 = (const float*) ((uintptr_t) i4 + input_increment);
    i5 = (const float*) ((uintptr_t) i5 + input_increment);
    i6 = (const float*) ((uintptr_t) i6 + input_increment);

    size_t c = channels;
    do {
      const float vi0 = *i0++;
      const float vi1 = *i1++;
      const float vi2 = *i2++;
      const float vi3 = *i3++;
      const float vi4 = *i4++;
      const float vi5 = *i5++;
      const float vi6 = *i6++;
      const float vacc = *b;

      const float vsum01 = vi0 + vi1;
      const float vsum23 = vi2 + vi3;
      const float vsum45 = vi4 + vi5;
      const float vsum6a = vi6 + vacc;

      const float vsum0123 = vsum01 + vsum23;
      const float vsum456a = vsum45 + vsum6a;

      const float vsum = vsum0123 + vsum456a;

      *b++ = vsum;
    } while (--c != 0);
  }

  i0 = (const float*) ((uintptr_t) i0 + input_increment);
  i1 = (const float*) ((uintptr_t) i1 + input_increment);
  if (rows < 2) {
    i1 = zero;
  }
  i2 = (const float*) ((uintptr_t) i2 + input_increment);
  if (rows <= 2) {
    i2 = zero;
  }
  i3 = (const float*) ((uintptr_t) i3 + input_increment);
  if (rows < 4) {
    i3 = zero;
  }
  i4 = (const float*) ((uintptr_t) i4 + input_increment);
  if (rows <= 4) {
    i4 = zero;
  }
  i5 = (const float*) ((uintptr_t) i5 + input_increment);
  if (rows < 6) {
    i5 = zero;
  }
  i6 = (const float*) ((uintptr_t) i6 + input_increment);
  if (rows <= 6) {
    i6 = zero;
  }
  const float vscale = params->scalar.scale;
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  b = buffer;
  do {
    const float vi0 = *i0++;
    const float vi1 = *i1++;
    const float vi2 = *i2++;
    const float vi3 = *i3++;
    const float vi4 = *i4++;
    const float vi5 = *i5++;
    const float vi6 = *i6++;
    const float vacc = *b++;

    const float vsum01 = vi0 + vi1;
    const float vsum23 = vi2 + vi3;
    const float vsum45 = vi4 + vi5;
    const float vsum6a = vi6 + vacc;

    const float vsum0123 = vsum01 + vsum23;
    const float vsum456a = vsum45 + vsum6a;

    const float vsum = vsum0123 + vsum456a;

    float vout = vsum * vscale;
    vout = __builtin_wasm_max_f32(vout, vmin);
    vout = __builtin_wasm_min_f32(vout, vmax);

    *output++ = vout;
  } while (--channels != 0);
}

void xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  if (rows < 2) {
    i1 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  if (rows <= 2) {
    i2 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  if (rows < 4) {
    i3 = zero;
  }
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  if (rows <= 4) {
    i4 = zero;
  }
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  if (rows < 6) {
    i5 = zero;
  }
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  if (rows <= 6) {
    i6 = zero;
  }

  const float vscale = params->scalar.scale;
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float vi0 = *i0++;
    const float vi1 = *i1++;
    const float vi2 = *i2++;
    const float vi3 = *i3++;
    const float vi4 = *i4++;
    const float vi5 = *i5++;
    const float vi6 = *i6++;

    const float vsum01 = vi0 + vi1;
    const float vsum23 = vi2 + vi3;
    const float vsum45 = vi4 + vi5;

    const float vsum016 = vsum01 + vi6;
    const float vsum2345 = vsum23 + vsum45;

    const float vsum = vsum016 + vsum2345;

    float vout = vsum * vscale;
    vout = __builtin_wasm_max_f32(vout, vmin);
    vout = __builtin_wasm_min_f32(vout, vmax);

    *output++ = vout;
  } while (--channels != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t k = kc;
    do {
      const float va0 = *a0++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);

      k -= sizeof(float);
    } while (k != 0);

    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc02 = __builtin_wasm_max_f32(vacc02, vmin);
    vacc03 = __builtin_wasm_max_f32(vacc03, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc02 = __builtin_wasm_min_f32(vacc02, vmax);
    vacc03 = __builtin_wasm_min_f32(vacc03, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_1x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t k = kc;
    do {
      const float va0 = *a0++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);

      k -= sizeof(float);
    } while (k != 0);

    vacc00 = __builtin_wasm_max_f32(vacc00, 0.0f);
    vacc01 = __builtin_wasm_max_f32(vacc01, 0.0f);
    vacc02 = __builtin_wasm_max_f32(vacc02, 0.0f);
    vacc03 = __builtin_wasm_max_f32(vacc03, 0.0f);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x2__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    w += 2;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc30 = vacc00;
    float vacc31 = vacc01;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      w += 2;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);

      k -= sizeof(float);
    } while (k != 0);

    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc10 = __builtin_wasm_max_f32(vacc10, vmin);
    vacc11 = __builtin_wasm_max_f32(vacc11, vmin);
    vacc20 = __builtin_wasm_max_f32(vacc20, vmin);
    vacc21 = __builtin_wasm_max_f32(vacc21, vmin);
    vacc30 = __builtin_wasm_max_f32(vacc30, vmin);
    vacc31 = __builtin_wasm_max_f32(vacc31, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc10 = __builtin_wasm_min_f32(vacc10, vmax);
    vacc11 = __builtin_wasm_min_f32(vacc11, vmax);
    vacc20 = __builtin_wasm_min_f32(vacc20, vmax);
    vacc21 = __builtin_wasm_min_f32(vacc21, vmax);
    vacc30 = __builtin_wasm_min_f32(vacc30, vmax);
    vacc31 = __builtin_wasm_min_f32(vacc31, vmax);

    if XNN_LIKELY(nc >= 2) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a3 = (const void*) ((uintptr_t) a3 - kc);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = vacc00;
        c1[0] = vacc10;
        c2[0] = vacc20;
        c3[0] = vacc30;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc12 = math_muladd_f32(va1, vb2, vacc12);
      vacc13 = math_muladd_f32(va1, vb3, vacc13);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc22 = math_muladd_f32(va2, vb2, vacc22);
      vacc23 = math_muladd_f32(va2, vb3, vacc23);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);
      vacc32 = math_muladd_f32(va3, vb2, vacc32);
      vacc33 = math_muladd_f32(va3, vb3, vacc33);

      k -= sizeof(float);
    } while (k != 0);

    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc02 = __builtin_wasm_max_f32(vacc02, vmin);
    vacc03 = __builtin_wasm_max_f32(vacc03, vmin);
    vacc10 = __builtin_wasm_max_f32(vacc10, vmin);
    vacc11 = __builtin_wasm_max_f32(vacc11, vmin);
    vacc12 = __builtin_wasm_max_f32(vacc12, vmin);
    vacc13 = __builtin_wasm_max_f32(vacc13, vmin);
    vacc20 = __builtin_wasm_max_f32(vacc20, vmin);
    vacc21 = __builtin_wasm_max_f32(vacc21, vmin);
    vacc22 = __builtin_wasm_max_f32(vacc22, vmin);
    vacc23 = __builtin_wasm_max_f32(vacc23, vmin);
    vacc30 = __builtin_wasm_max_f32(vacc30, vmin);
    vacc31 = __builtin_wasm_max_f32(vacc31, vmin);
    vacc32 = __builtin_wasm_max_f32(vacc32, vmin);
    vacc33 = __builtin_wasm_max_f32(vacc33, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc02 = __builtin_wasm_min_f32(vacc02, vmax);
    vacc03 = __builtin_wasm_min_f32(vacc03, vmax);
    vacc10 = __builtin_wasm_min_f32(vacc10, vmax);
    vacc11 = __builtin_wasm_min_f32(vacc11, vmax);
    vacc12 = __builtin_wasm_min_f32(vacc12, vmax);
    vacc13 = __builtin_wasm_min_f32(vacc13, vmax);
    vacc20 = __builtin_wasm_min_f32(vacc20, vmax);
    vacc21 = __builtin_wasm_min_f32(vacc21, vmax);
    vacc22 = __builtin_wasm_min_f32(vacc22, vmax);
    vacc23 = __builtin_wasm_min_f32(vacc23, vmax);
    vacc30 = __builtin_wasm_min_f32(vacc30, vmax);
    vacc31 = __builtin_wasm_min_f32(vacc31, vmax);
    vacc32 = __builtin_wasm_min_f32(vacc32, vmax);
    vacc33 = __builtin_wasm_min_f32(vacc33, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a3 = (const void*) ((uintptr_t) a3 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
        c1[0] = vacc10;
        c2[0] = vacc20;
        c3[0] = vacc30;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_4x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc12 = math_muladd_f32(va1, vb2, vacc12);
      vacc13 = math_muladd_f32(va1, vb3, vacc13);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc22 = math_muladd_f32(va2, vb2, vacc22);
      vacc23 = math_muladd_f32(va2, vb3, vacc23);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);
      vacc32 = math_muladd_f32(va3, vb2, vacc32);
      vacc33 = math_muladd_f32(va3, vb3, vacc33);

      k -= sizeof(float);
    } while (k != 0);

    vacc00 = __builtin_wasm_max_f32(vacc00, 0.0f);
    vacc01 = __builtin_wasm_max_f32(vacc01, 0.0f);
    vacc02 = __builtin_wasm_max_f32(vacc02, 0.0f);
    vacc03 = __builtin_wasm_max_f32(vacc03, 0.0f);
    vacc10 = __builtin_wasm_max_f32(vacc10, 0.0f);
    vacc11 = __builtin_wasm_max_f32(vacc11, 0.0f);
    vacc12 = __builtin_wasm_max_f32(vacc12, 0.0f);
    vacc13 = __builtin_wasm_max_f32(vacc13, 0.0f);
    vacc20 = __builtin_wasm_max_f32(vacc20, 0.0f);
    vacc21 = __builtin_wasm_max_f32(vacc21, 0.0f);
    vacc22 = __builtin_wasm_max_f32(vacc22, 0.0f);
    vacc23 = __builtin_wasm_max_f32(vacc23, 0.0f);
    vacc30 = __builtin_wasm_max_f32(vacc30, 0.0f);
    vacc31 = __builtin_wasm_max_f32(vacc31, 0.0f);
    vacc32 = __builtin_wasm_max_f32(vacc32, 0.0f);
    vacc33 = __builtin_wasm_max_f32(vacc33, 0.0f);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a3 = (const void*) ((uintptr_t) a3 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
        c1[0] = vacc10;
        c2[0] = vacc20;
        c3[0] = vacc30;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const float va0 = *a0++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        const float vb2 = w[2];
        const float vb3 = w[3];
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc02 = __builtin_wasm_max_f32(vacc02, vmin);
    vacc03 = __builtin_wasm_max_f32(vacc03, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc02 = __builtin_wasm_min_f32(vacc02, vmax);
    vacc03 = __builtin_wasm_min_f32(vacc03, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_1x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const float va0 = *a0++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        const float vb2 = w[2];
        const float vb3 = w[3];
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc00 = __builtin_wasm_max_f32(vacc00, 0.0f);
    vacc01 = __builtin_wasm_max_f32(vacc01, 0.0f);
    vacc02 = __builtin_wasm_max_f32(vacc02, 0.0f);
    vacc03 = __builtin_wasm_max_f32(vacc03, 0.0f);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x2__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    w += 2;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        w += 2;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc10 = math_muladd_f32(va1, vb0, vacc10);
        vacc11 = math_muladd_f32(va1, vb1, vacc11);
        vacc20 = math_muladd_f32(va2, vb0, vacc20);
        vacc21 = math_muladd_f32(va2, vb1, vacc21);
        vacc30 = math_muladd_f32(va3, vb0, vacc30);
        vacc31 = math_muladd_f32(va3, vb1, vacc31);

        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc10 = __builtin_wasm_max_f32(vacc10, vmin);
    vacc11 = __builtin_wasm_max_f32(vacc11, vmin);
    vacc20 = __builtin_wasm_max_f32(vacc20, vmin);
    vacc21 = __builtin_wasm_max_f32(vacc21, vmin);
    vacc30 = __builtin_wasm_max_f32(vacc30, vmin);
    vacc31 = __builtin_wasm_max_f32(vacc31, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc10 = __builtin_wasm_min_f32(vacc10, vmax);
    vacc11 = __builtin_wasm_min_f32(vacc11, vmax);
    vacc20 = __builtin_wasm_min_f32(vacc20, vmax);
    vacc21 = __builtin_wasm_min_f32(vacc21, vmax);
    vacc30 = __builtin_wasm_min_f32(vacc30, vmax);
    vacc31 = __builtin_wasm_min_f32(vacc31, vmax);

    if XNN_LIKELY(nc >= 2) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;
    w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        const float vb2 = w[2];
        const float vb3 = w[3];
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);
        vacc10 = math_muladd_f32(va1, vb0, vacc10);
        vacc11 = math_muladd_f32(va1, vb1, vacc11);
        vacc12 = math_muladd_f32(va1, vb2, vacc12);
        vacc13 = math_muladd_f32(va1, vb3, vacc13);
        vacc20 = math_muladd_f32(va2, vb0, vacc20);
        vacc21 = math_muladd_f32(va2, vb1, vacc21);
        vacc22 = math_muladd_f32(va2, vb2, vacc22);
        vacc23 = math_muladd_f32(va2, vb3, vacc23);
        vacc30 = math_muladd_f32(va3, vb0, vacc30);
        vacc31 = math_muladd_f32(va3, vb1, vacc31);
        vacc32 = math_muladd_f32(va3, vb2, vacc32);
        vacc33 = math_muladd_f32(va3, vb3, vacc33);

        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc02 = __builtin_wasm_max_f32(vacc02, vmin);
    vacc03 = __builtin_wasm_max_f32(vacc03, vmin);
    vacc10 = __builtin_wasm_max_f32(vacc10, vmin);
    vacc11 = __builtin_wasm_max_f32(vacc11, vmin);
    vacc12 = __builtin_wasm_max_f32(vacc12, vmin);
    vacc13 = __builtin_wasm_max_f32(vacc13, vmin);
    vacc20 = __builtin_wasm_max_f32(vacc20, vmin);
    vacc21 = __builtin_wasm_max_f32(vacc21, vmin);
    vacc22 = __builtin_wasm_max_f32(vacc22, vmin);
    vacc23 = __builtin_wasm_max_f32(vacc23, vmin);
    vacc30 = __builtin_wasm_max_f32(vacc30, vmin);
    vacc31 = __builtin_wasm_max_f32(vacc31, vmin);
    vacc32 = __builtin_wasm_max_f32(vacc32, vmin);
    vacc33 = __builtin_wasm_max_f32(vacc33, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc02 = __builtin_wasm_min_f32(vacc02, vmax);
    vacc03 = __builtin_wasm_min_f32(vacc03, vmax);
    vacc10 = __builtin_wasm_min_f32(vacc10, vmax);
    vacc11 = __builtin_wasm_min_f32(vacc11, vmax);
    vacc12 = __builtin_wasm_min_f32(vacc12, vmax);
    vacc13 = __builtin_wasm_min_f32(vacc13, vmax);
    vacc20 = __builtin_wasm_min_f32(vacc20, vmax);
    vacc21 = __builtin_wasm_min_f32(vacc21, vmax);
    vacc22 = __builtin_wasm_min_f32(vacc22, vmax);
    vacc23 = __builtin_wasm_min_f32(vacc23, vmax);
    vacc30 = __builtin_wasm_min_f32(vacc30, vmax);
    vacc31 = __builtin_wasm_min_f32(vacc31, vmax);
    vacc32 = __builtin_wasm_min_f32(vacc32, vmax);
    vacc33 = __builtin_wasm_min_f32(vacc33, vmax);

    if XNN_LIKELY(nc >= 4) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_4x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;
    w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        const float vb2 = w[2];
        const float vb3 = w[3];
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);
        vacc10 = math_muladd_f32(va1, vb0, vacc10);
        vacc11 = math_muladd_f32(va1, vb1, vacc11);
        vacc12 = math_muladd_f32(va1, vb2, vacc12);
        vacc13 = math_muladd_f32(va1, vb3, vacc13);
        vacc20 = math_muladd_f32(va2, vb0, vacc20);
        vacc21 = math_muladd_f32(va2, vb1, vacc21);
        vacc22 = math_muladd_f32(va2, vb2, vacc22);
        vacc23 = math_muladd_f32(va2, vb3, vacc23);
        vacc30 = math_muladd_f32(va3, vb0, vacc30);
        vacc31 = math_muladd_f32(va3, vb1, vacc31);
        vacc32 = math_muladd_f32(va3, vb2, vacc32);
        vacc33 = math_muladd_f32(va3, vb3, vacc33);

        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc00 = __builtin_wasm_max_f32(vacc00, 0.0f);
    vacc01 = __builtin_wasm_max_f32(vacc01, 0.0f);
    vacc02 = __builtin_wasm_max_f32(vacc02, 0.0f);
    vacc03 = __builtin_wasm_max_f32(vacc03, 0.0f);
    vacc10 = __builtin_wasm_max_f32(vacc10, 0.0f);
    vacc11 = __builtin_wasm_max_f32(vacc11, 0.0f);
    vacc12 = __builtin_wasm_max_f32(vacc12, 0.0f);
    vacc13 = __builtin_wasm_max_f32(vacc13, 0.0f);
    vacc20 = __builtin_wasm_max_f32(vacc20, 0.0f);
    vacc21 = __builtin_wasm_max_f32(vacc21, 0.0f);
    vacc22 = __builtin_wasm_max_f32(vacc22, 0.0f);
    vacc23 = __builtin_wasm_max_f32(vacc23, 0.0f);
    vacc30 = __builtin_wasm_max_f32(vacc30, 0.0f);
    vacc31 = __builtin_wasm_max_f32(vacc31, 0.0f);
    vacc32 = __builtin_wasm_max_f32(vacc32, 0.0f);
    vacc33 = __builtin_wasm_max_f32(vacc33, 0.0f);

    if XNN_LIKELY(nc >= 4) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1(
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

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
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
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vi8 = *i8++;

        const float vmax01 = __builtin_wasm_max_f32(vi0, vi1);
        const float vmax23 = __builtin_wasm_max_f32(vi2, vi3);
        const float vmax45 = __builtin_wasm_max_f32(vi4, vi5);
        const float vmax67 = __builtin_wasm_max_f32(vi6, vi7);
        const float vmax018 = __builtin_wasm_max_f32(vmax01, vi8);

        const float vmax2345 = __builtin_wasm_max_f32(vmax23, vmax45);
        const float vmax01678 = __builtin_wasm_max_f32(vmax018, vmax67);
        float vout = __builtin_wasm_max_f32(vmax2345, vmax01678);
        vout = __builtin_wasm_max_f32(vout, voutput_min);
        vout = __builtin_wasm_min_f32(vout, voutput_max);

        *o++ = vout;
      } while (--c != 0);
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
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vi8 = *o;

        const float vmax01 = __builtin_wasm_max_f32(vi0, vi1);
        const float vmax23 = __builtin_wasm_max_f32(vi2, vi3);
        const float vmax45 = __builtin_wasm_max_f32(vi4, vi5);
        const float vmax67 = __builtin_wasm_max_f32(vi6, vi7);
        const float vmax018 = __builtin_wasm_max_f32(vmax01, vi8);

        const float vmax2345 = __builtin_wasm_max_f32(vmax23, vmax45);
        const float vmax01678 = __builtin_wasm_max_f32(vmax018, vmax67);
        float vout = __builtin_wasm_max_f32(vmax2345, vmax01678);
        vout = __builtin_wasm_max_f32(vout, voutput_min);
        vout = __builtin_wasm_min_f32(vout, voutput_max);

        *o++ = vout;
      } while (--c != 0);
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_pavgpool_minmax_ukernel_9p8x__wasm_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    const float* multiplier,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  do {
    {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      const float* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const float*) ((uintptr_t) i8 + input_offset);
      }

      float* b = buffer;
      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vi8 = *i8++;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum018 = vsum01 + vi8;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum01678 = vsum018 + vsum67;
        const float vsum = vsum2345 + vsum01678;

        *b++ = vsum;
      } while (--c != 0);
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const float* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      float* b = buffer;
      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vacc = *b;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum01a = vsum01 + vacc;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum0167a = vsum01a + vsum67;
        const float vsum = vsum2345 + vsum0167a;

        *b++ = vsum;
      } while (--c != 0);
    }

    {
      const float* i0 = input[0];
      assert(i0 != NULL);
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      input = (const float**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }

      const float vmultiplier = *multiplier++;

      size_t c = channels;
      float* b = buffer;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vacc = *b++;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum01a = vsum01 + vacc;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum0167a = vsum01a + vsum67;
        const float vsum = vsum2345 + vsum0167a;

        float vout = vsum * vmultiplier;
        vout = __builtin_wasm_max_f32(vout, voutput_min);
        vout = __builtin_wasm_min_f32(vout, voutput_max);

        *output++ = vout;
      } while (--c != 0);
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_pavgpool_minmax_ukernel_9x__wasm_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    const float* multiplier,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    input = (const float**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }

    const float vmultiplier = *multiplier++;

    size_t c = channels;
    do {
      const float vi0 = *i0++;
      const float vi1 = *i1++;
      const float vi2 = *i2++;
      const float vi3 = *i3++;
      const float vi4 = *i4++;
      const float vi5 = *i5++;
      const float vi6 = *i6++;
      const float vi7 = *i7++;
      const float vi8 = *i8++;

      const float vsum01 = vi0 + vi1;
      const float vsum23 = vi2 + vi3;
      const float vsum45 = vi4 + vi5;
      const float vsum67 = vi6 + vi7;
      const float vsum018 = vsum01 + vi8;
      const float vsum2345 = vsum23 + vsum45;
      const float vsum01678 = vsum018 + vsum67;
      const float vsum = vsum2345 + vsum01678;

      float vout = vsum * vmultiplier;
      vout = __builtin_wasm_max_f32(vout, voutput_min);
      vout = __builtin_wasm_min_f32(vout, voutput_max);

      *output++ = vout;
    } while (--c != 0);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_prelu_ukernel__wasm_2x4(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const float vzero = 0.0f;
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float vw0 = w[0];
      const float vw1 = w[1];
      const float vw2 = w[2];
      const float vw3 = w[3];

      float vi0x0 = i0[0];
      float vi0x1 = i0[1];
      float vi0x2 = i0[2];
      float vi0x3 = i0[3];
      i0 += 4;
      float vi1x0 = i1[0];
      float vi1x1 = i1[1];
      float vi1x2 = i1[2];
      float vi1x3 = i1[3];
      i1 += 4;

      float vacc0x0 = __builtin_wasm_max_f32(vi0x0, vzero);
      vi0x0 = __builtin_wasm_min_f32(vi0x0, vzero);
      float vacc0x1 = __builtin_wasm_max_f32(vi0x1, vzero);
      vi0x1 = __builtin_wasm_min_f32(vi0x1, vzero);
      float vacc0x2 = __builtin_wasm_max_f32(vi0x2, vzero);
      vi0x2 = __builtin_wasm_min_f32(vi0x2, vzero);
      float vacc0x3 = __builtin_wasm_max_f32(vi0x3, vzero);
      vi0x3 = __builtin_wasm_min_f32(vi0x3, vzero);
      float vacc1x0 = __builtin_wasm_max_f32(vi1x0, vzero);
      vi1x0 = __builtin_wasm_min_f32(vi1x0, vzero);
      float vacc1x1 = __builtin_wasm_max_f32(vi1x1, vzero);
      vi1x1 = __builtin_wasm_min_f32(vi1x1, vzero);
      float vacc1x2 = __builtin_wasm_max_f32(vi1x2, vzero);
      vi1x2 = __builtin_wasm_min_f32(vi1x2, vzero);
      float vacc1x3 = __builtin_wasm_max_f32(vi1x3, vzero);
      vi1x3 = __builtin_wasm_min_f32(vi1x3, vzero);

      vacc0x0 += vi0x0 * vw0;
      vacc0x1 += vi0x1 * vw1;
      vacc0x2 += vi0x2 * vw2;
      vacc0x3 += vi0x3 * vw3;
      vacc1x0 += vi1x0 * vw0;
      vacc1x1 += vi1x1 * vw1;
      vacc1x2 += vi1x2 * vw2;
      vacc1x3 += vi1x3 * vw3;

      o0[0] = vacc0x0;
      o0[1] = vacc0x1;
      o0[2] = vacc0x2;
      o0[3] = vacc0x3;
      o0 += 4;
      o1[0] = vacc1x0;
      o1[1] = vacc1x1;
      o1[2] = vacc1x2;
      o1[3] = vacc1x3;
      o1 += 4;

      w += 4;
    }
    for (; c != 0; c -= sizeof(float)) {
      const float vw = *w++;

      float vi0 = *i0++;
      float vi1 = *i1++;

      float vacc0 = __builtin_wasm_max_f32(vi0, vzero);
      vi0 = __builtin_wasm_min_f32(vi0, vzero);
      float vacc1 = __builtin_wasm_max_f32(vi1, vzero);
      vi1 = __builtin_wasm_min_f32(vi1, vzero);

      vacc0 += vi0 * vw;
      vacc1 += vi1 * vw;

      *o0++ = vacc0;
      *o1++ = vacc1;
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_qc4w_gemm_minmax_ukernel_1x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  const int32_t vminus_kernel_zero_point = params->scalar.minus_kernel_zero_point;
  do {
    float vacc00 = ((const float*)w)[0];
    float vacc01 = ((const float*)w)[1];
    float vacc02 = ((const float*)w)[2];
    float vacc03 = ((const float*)w)[3];
    w = (const float*) w + 4;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float va00 = *a0++;
      const float va01 = *a0++;

      const uint8_t vbi0 = ((const uint8_t*) w)[0];
      const uint8_t vbi1 = ((const uint8_t*) w)[1];
      const uint8_t vbi2 = ((const uint8_t*) w)[2];
      const uint8_t vbi3 = ((const uint8_t*) w)[3];
      const float vb00 = (float) ((int32_t) (vbi0 & 0xF) + vminus_kernel_zero_point);
      const float vb10 = (float) ((int32_t) (vbi1 & 0xF) + vminus_kernel_zero_point);
      const float vb20 = (float) ((int32_t) (vbi2 & 0xF) + vminus_kernel_zero_point);
      const float vb30 = (float) ((int32_t) (vbi3 & 0xF) + vminus_kernel_zero_point);
      const float vb01 = (float) ((int32_t) (vbi0 >> 4) + vminus_kernel_zero_point);
      const float vb11 = (float) ((int32_t) (vbi1 >> 4) + vminus_kernel_zero_point);
      const float vb21 = (float) ((int32_t) (vbi2 >> 4) + vminus_kernel_zero_point);
      const float vb31 = (float) ((int32_t) (vbi3 >> 4) + vminus_kernel_zero_point);
      w = (const int8_t*) w + 4;

      vacc00 = math_muladd_f32(va00, vb00, vacc00);
      vacc01 = math_muladd_f32(va00, vb10, vacc01);
      vacc02 = math_muladd_f32(va00, vb20, vacc02);
      vacc03 = math_muladd_f32(va00, vb30, vacc03);
      vacc00 = math_muladd_f32(va01, vb01, vacc00);
      vacc01 = math_muladd_f32(va01, vb11, vacc01);
      vacc02 = math_muladd_f32(va01, vb21, vacc02);
      vacc03 = math_muladd_f32(va01, vb31, vacc03);
    }
    if XNN_UNLIKELY(k != 0) {
      const float va0 = *a0++;

      const uint8_t vbi0 = ((const uint8_t*) w)[0];
      const uint8_t vbi1 = ((const uint8_t*) w)[1];
      const uint8_t vbi2 = ((const uint8_t*) w)[2];
      const uint8_t vbi3 = ((const uint8_t*) w)[3];
      const float vb0 = (float) ((int32_t) vbi0 + vminus_kernel_zero_point);
      const float vb1 = (float) ((int32_t) vbi1 + vminus_kernel_zero_point);
      const float vb2 = (float) ((int32_t) vbi2 + vminus_kernel_zero_point);
      const float vb3 = (float) ((int32_t) vbi3 + vminus_kernel_zero_point);
      w = (const int8_t*) w + 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);
    }

    const float vscale0 = ((const float*)w)[0];
    const float vscale1 = ((const float*)w)[1];
    const float vscale2 = ((const float*)w)[2];
    const float vscale3 = ((const float*)w)[3];
    w = (const float*) w + 4;
    vacc00 *= vscale0;
    vacc01 *= vscale1;
    vacc02 *= vscale2;
    vacc03 *= vscale3;
    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc02 = __builtin_wasm_max_f32(vacc02, vmin);
    vacc03 = __builtin_wasm_max_f32(vacc03, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc02 = __builtin_wasm_min_f32(vacc02, vmax);
    vacc03 = __builtin_wasm_min_f32(vacc03, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc4w_gemm_minmax_ukernel_4x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  const int32_t vminus_kernel_zero_point = params->scalar.minus_kernel_zero_point;
  do {
    float vacc00 = ((const float*)w)[0];
    float vacc01 = ((const float*)w)[1];
    float vacc02 = ((const float*)w)[2];
    float vacc03 = ((const float*)w)[3];
    w = (const float*) w + 4;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float va00 = *a0++;
      const float va01 = *a0++;
      const float va10 = *a1++;
      const float va11 = *a1++;
      const float va20 = *a2++;
      const float va21 = *a2++;
      const float va30 = *a3++;
      const float va31 = *a3++;

      const uint8_t vbi0 = ((const uint8_t*) w)[0];
      const uint8_t vbi1 = ((const uint8_t*) w)[1];
      const uint8_t vbi2 = ((const uint8_t*) w)[2];
      const uint8_t vbi3 = ((const uint8_t*) w)[3];
      const float vb00 = (float) ((int32_t) (vbi0 & 0xF) + vminus_kernel_zero_point);
      const float vb10 = (float) ((int32_t) (vbi1 & 0xF) + vminus_kernel_zero_point);
      const float vb20 = (float) ((int32_t) (vbi2 & 0xF) + vminus_kernel_zero_point);
      const float vb30 = (float) ((int32_t) (vbi3 & 0xF) + vminus_kernel_zero_point);
      const float vb01 = (float) ((int32_t) (vbi0 >> 4) + vminus_kernel_zero_point);
      const float vb11 = (float) ((int32_t) (vbi1 >> 4) + vminus_kernel_zero_point);
      const float vb21 = (float) ((int32_t) (vbi2 >> 4) + vminus_kernel_zero_point);
      const float vb31 = (float) ((int32_t) (vbi3 >> 4) + vminus_kernel_zero_point);
      w = (const int8_t*) w + 4;

      vacc00 = math_muladd_f32(va00, vb00, vacc00);
      vacc01 = math_muladd_f32(va00, vb10, vacc01);
      vacc02 = math_muladd_f32(va00, vb20, vacc02);
      vacc03 = math_muladd_f32(va00, vb30, vacc03);
      vacc10 = math_muladd_f32(va10, vb00, vacc10);
      vacc11 = math_muladd_f32(va10, vb10, vacc11);
      vacc12 = math_muladd_f32(va10, vb20, vacc12);
      vacc13 = math_muladd_f32(va10, vb30, vacc13);
      vacc20 = math_muladd_f32(va20, vb00, vacc20);
      vacc21 = math_muladd_f32(va20, vb10, vacc21);
      vacc22 = math_muladd_f32(va20, vb20, vacc22);
      vacc23 = math_muladd_f32(va20, vb30, vacc23);
      vacc30 = math_muladd_f32(va30, vb00, vacc30);
      vacc31 = math_muladd_f32(va30, vb10, vacc31);
      vacc32 = math_muladd_f32(va30, vb20, vacc32);
      vacc33 = math_muladd_f32(va30, vb30, vacc33);
      vacc00 = math_muladd_f32(va01, vb01, vacc00);
      vacc01 = math_muladd_f32(va01, vb11, vacc01);
      vacc02 = math_muladd_f32(va01, vb21, vacc02);
      vacc03 = math_muladd_f32(va01, vb31, vacc03);
      vacc10 = math_muladd_f32(va11, vb01, vacc10);
      vacc11 = math_muladd_f32(va11, vb11, vacc11);
      vacc12 = math_muladd_f32(va11, vb21, vacc12);
      vacc13 = math_muladd_f32(va11, vb31, vacc13);
      vacc20 = math_muladd_f32(va21, vb01, vacc20);
      vacc21 = math_muladd_f32(va21, vb11, vacc21);
      vacc22 = math_muladd_f32(va21, vb21, vacc22);
      vacc23 = math_muladd_f32(va21, vb31, vacc23);
      vacc30 = math_muladd_f32(va31, vb01, vacc30);
      vacc31 = math_muladd_f32(va31, vb11, vacc31);
      vacc32 = math_muladd_f32(va31, vb21, vacc32);
      vacc33 = math_muladd_f32(va31, vb31, vacc33);
    }
    if XNN_UNLIKELY(k != 0) {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const uint8_t vbi0 = ((const uint8_t*) w)[0];
      const uint8_t vbi1 = ((const uint8_t*) w)[1];
      const uint8_t vbi2 = ((const uint8_t*) w)[2];
      const uint8_t vbi3 = ((const uint8_t*) w)[3];
      const float vb0 = (float) ((int32_t) vbi0 + vminus_kernel_zero_point);
      const float vb1 = (float) ((int32_t) vbi1 + vminus_kernel_zero_point);
      const float vb2 = (float) ((int32_t) vbi2 + vminus_kernel_zero_point);
      const float vb3 = (float) ((int32_t) vbi3 + vminus_kernel_zero_point);
      w = (const int8_t*) w + 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc12 = math_muladd_f32(va1, vb2, vacc12);
      vacc13 = math_muladd_f32(va1, vb3, vacc13);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc22 = math_muladd_f32(va2, vb2, vacc22);
      vacc23 = math_muladd_f32(va2, vb3, vacc23);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);
      vacc32 = math_muladd_f32(va3, vb2, vacc32);
      vacc33 = math_muladd_f32(va3, vb3, vacc33);
    }

    const float vscale0 = ((const float*)w)[0];
    const float vscale1 = ((const float*)w)[1];
    const float vscale2 = ((const float*)w)[2];
    const float vscale3 = ((const float*)w)[3];
    w = (const float*) w + 4;
    vacc00 *= vscale0;
    vacc10 *= vscale0;
    vacc20 *= vscale0;
    vacc30 *= vscale0;
    vacc01 *= vscale1;
    vacc11 *= vscale1;
    vacc21 *= vscale1;
    vacc31 *= vscale1;
    vacc02 *= vscale2;
    vacc12 *= vscale2;
    vacc22 *= vscale2;
    vacc32 *= vscale2;
    vacc03 *= vscale3;
    vacc13 *= vscale3;
    vacc23 *= vscale3;
    vacc33 *= vscale3;
    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc02 = __builtin_wasm_max_f32(vacc02, vmin);
    vacc03 = __builtin_wasm_max_f32(vacc03, vmin);
    vacc10 = __builtin_wasm_max_f32(vacc10, vmin);
    vacc11 = __builtin_wasm_max_f32(vacc11, vmin);
    vacc12 = __builtin_wasm_max_f32(vacc12, vmin);
    vacc13 = __builtin_wasm_max_f32(vacc13, vmin);
    vacc20 = __builtin_wasm_max_f32(vacc20, vmin);
    vacc21 = __builtin_wasm_max_f32(vacc21, vmin);
    vacc22 = __builtin_wasm_max_f32(vacc22, vmin);
    vacc23 = __builtin_wasm_max_f32(vacc23, vmin);
    vacc30 = __builtin_wasm_max_f32(vacc30, vmin);
    vacc31 = __builtin_wasm_max_f32(vacc31, vmin);
    vacc32 = __builtin_wasm_max_f32(vacc32, vmin);
    vacc33 = __builtin_wasm_max_f32(vacc33, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc02 = __builtin_wasm_min_f32(vacc02, vmax);
    vacc03 = __builtin_wasm_min_f32(vacc03, vmax);
    vacc10 = __builtin_wasm_min_f32(vacc10, vmax);
    vacc11 = __builtin_wasm_min_f32(vacc11, vmax);
    vacc12 = __builtin_wasm_min_f32(vacc12, vmax);
    vacc13 = __builtin_wasm_min_f32(vacc13, vmax);
    vacc20 = __builtin_wasm_min_f32(vacc20, vmax);
    vacc21 = __builtin_wasm_min_f32(vacc21, vmax);
    vacc22 = __builtin_wasm_min_f32(vacc22, vmax);
    vacc23 = __builtin_wasm_min_f32(vacc23, vmax);
    vacc30 = __builtin_wasm_min_f32(vacc30, vmax);
    vacc31 = __builtin_wasm_min_f32(vacc31, vmax);
    vacc32 = __builtin_wasm_min_f32(vacc32, vmax);
    vacc33 = __builtin_wasm_min_f32(vacc33, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a3 = (const void*) ((uintptr_t) a3 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
        c1[0] = vacc10;
        c2[0] = vacc20;
        c3[0] = vacc30;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = ((const float*)w)[0];
    float vacc01 = ((const float*)w)[1];
    float vacc02 = ((const float*)w)[2];
    float vacc03 = ((const float*)w)[3];
    w = (const float*) w + 4;

    size_t k = kc;
    do {
      const float va0 = *a0++;

      const float vb0 = (float) ((const int8_t*) w)[0];
      const float vb1 = (float) ((const int8_t*) w)[1];
      const float vb2 = (float) ((const int8_t*) w)[2];
      const float vb3 = (float) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);

      k -= sizeof(float);
    } while (k != 0);

    const float vscale0 = ((const float*)w)[0];
    const float vscale1 = ((const float*)w)[1];
    const float vscale2 = ((const float*)w)[2];
    const float vscale3 = ((const float*)w)[3];
    w = (const float*) w + 4;
    vacc00 *= vscale0;
    vacc01 *= vscale1;
    vacc02 *= vscale2;
    vacc03 *= vscale3;
    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc02 = __builtin_wasm_max_f32(vacc02, vmin);
    vacc03 = __builtin_wasm_max_f32(vacc03, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc02 = __builtin_wasm_min_f32(vacc02, vmax);
    vacc03 = __builtin_wasm_min_f32(vacc03, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = ((const float*)w)[0];
    float vacc01 = ((const float*)w)[1];
    float vacc02 = ((const float*)w)[2];
    float vacc03 = ((const float*)w)[3];
    w = (const float*) w + 4;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const float vb0 = (float) ((const int8_t*) w)[0];
      const float vb1 = (float) ((const int8_t*) w)[1];
      const float vb2 = (float) ((const int8_t*) w)[2];
      const float vb3 = (float) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc12 = math_muladd_f32(va1, vb2, vacc12);
      vacc13 = math_muladd_f32(va1, vb3, vacc13);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc22 = math_muladd_f32(va2, vb2, vacc22);
      vacc23 = math_muladd_f32(va2, vb3, vacc23);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);
      vacc32 = math_muladd_f32(va3, vb2, vacc32);
      vacc33 = math_muladd_f32(va3, vb3, vacc33);

      k -= sizeof(float);
    } while (k != 0);

    const float vscale0 = ((const float*)w)[0];
    const float vscale1 = ((const float*)w)[1];
    const float vscale2 = ((const float*)w)[2];
    const float vscale3 = ((const float*)w)[3];
    w = (const float*) w + 4;
    vacc00 *= vscale0;
    vacc10 *= vscale0;
    vacc20 *= vscale0;
    vacc30 *= vscale0;
    vacc01 *= vscale1;
    vacc11 *= vscale1;
    vacc21 *= vscale1;
    vacc31 *= vscale1;
    vacc02 *= vscale2;
    vacc12 *= vscale2;
    vacc22 *= vscale2;
    vacc32 *= vscale2;
    vacc03 *= vscale3;
    vacc13 *= vscale3;
    vacc23 *= vscale3;
    vacc33 *= vscale3;
    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc02 = __builtin_wasm_max_f32(vacc02, vmin);
    vacc03 = __builtin_wasm_max_f32(vacc03, vmin);
    vacc10 = __builtin_wasm_max_f32(vacc10, vmin);
    vacc11 = __builtin_wasm_max_f32(vacc11, vmin);
    vacc12 = __builtin_wasm_max_f32(vacc12, vmin);
    vacc13 = __builtin_wasm_max_f32(vacc13, vmin);
    vacc20 = __builtin_wasm_max_f32(vacc20, vmin);
    vacc21 = __builtin_wasm_max_f32(vacc21, vmin);
    vacc22 = __builtin_wasm_max_f32(vacc22, vmin);
    vacc23 = __builtin_wasm_max_f32(vacc23, vmin);
    vacc30 = __builtin_wasm_max_f32(vacc30, vmin);
    vacc31 = __builtin_wasm_max_f32(vacc31, vmin);
    vacc32 = __builtin_wasm_max_f32(vacc32, vmin);
    vacc33 = __builtin_wasm_max_f32(vacc33, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc02 = __builtin_wasm_min_f32(vacc02, vmax);
    vacc03 = __builtin_wasm_min_f32(vacc03, vmax);
    vacc10 = __builtin_wasm_min_f32(vacc10, vmax);
    vacc11 = __builtin_wasm_min_f32(vacc11, vmax);
    vacc12 = __builtin_wasm_min_f32(vacc12, vmax);
    vacc13 = __builtin_wasm_min_f32(vacc13, vmax);
    vacc20 = __builtin_wasm_min_f32(vacc20, vmax);
    vacc21 = __builtin_wasm_min_f32(vacc21, vmax);
    vacc22 = __builtin_wasm_min_f32(vacc22, vmax);
    vacc23 = __builtin_wasm_min_f32(vacc23, vmax);
    vacc30 = __builtin_wasm_min_f32(vacc30, vmax);
    vacc31 = __builtin_wasm_min_f32(vacc31, vmax);
    vacc32 = __builtin_wasm_min_f32(vacc32, vmax);
    vacc33 = __builtin_wasm_min_f32(vacc33, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a3 = (const void*) ((uintptr_t) a3 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
        c1[0] = vacc10;
        c2[0] = vacc20;
        c3[0] = vacc30;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qs8_vcvt_ukernel__wasm_fmagic_u4(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float* i = input;
  const float vscale = params->scalar.scale;
  const float voutput_min_less_zero_point = (float) ((int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point);
  const float voutput_max_less_zero_point = (float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point);
  const float vmagic_bias = 12582912.0f;
  const int32_t vmagic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vx0 = i[0];
    float vx1 = i[1];
    float vx2 = i[2];
    float vx3 = i[3];
    i += 4;

    vx0 *= vscale;
    vx1 *= vscale;
    vx2 *= vscale;
    vx3 *= vscale;

    vx0 = __builtin_wasm_max_f32(vx0, voutput_min_less_zero_point);
    vx1 = __builtin_wasm_max_f32(vx1, voutput_min_less_zero_point);
    vx2 = __builtin_wasm_max_f32(vx2, voutput_min_less_zero_point);
    vx3 = __builtin_wasm_max_f32(vx3, voutput_min_less_zero_point);

    vx0 = __builtin_wasm_min_f32(vx0, voutput_max_less_zero_point);
    vx1 = __builtin_wasm_min_f32(vx1, voutput_max_less_zero_point);
    vx2 = __builtin_wasm_min_f32(vx2, voutput_max_less_zero_point);
    vx3 = __builtin_wasm_min_f32(vx3, voutput_max_less_zero_point);

    vx0 += vmagic_bias;
    vx1 += vmagic_bias;
    vx2 += vmagic_bias;
    vx3 += vmagic_bias;

    int32_t vy0 = (int32_t) float_as_uint32(vx0);
    int32_t vy1 = (int32_t) float_as_uint32(vx1);
    int32_t vy2 = (int32_t) float_as_uint32(vx2);
    int32_t vy3 = (int32_t) float_as_uint32(vx3);

    vy0 -= vmagic_bias_less_zero_point;
    vy1 -= vmagic_bias_less_zero_point;
    vy2 -= vmagic_bias_less_zero_point;
    vy3 -= vmagic_bias_less_zero_point;

    output[0] = (int8_t) vy0;
    output[1] = (int8_t) vy1;
    output[2] = (int8_t) vy2;
    output[3] = (int8_t) vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = *i++;
      vx *= vscale;
      vx = __builtin_wasm_max_f32(vx, voutput_min_less_zero_point);
      vx = __builtin_wasm_min_f32(vx, voutput_max_less_zero_point);
      vx += vmagic_bias;

      int32_t vy = (int32_t) float_as_uint32(vx);
      vy -= vmagic_bias_less_zero_point;

      *output++ = (int8_t) vy;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4(
    size_t batch,
    const float* input,
    uint8_t* output,
    const union xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float* i = input;
  const float vscale = params->scalar.scale;
  const float voutput_min_less_zero_point = (float) ((int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point);
  const float voutput_max_less_zero_point = (float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point);
  const float vmagic_bias = 12582912.0f;
  const int32_t vmagic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vx0 = i[0];
    float vx1 = i[1];
    float vx2 = i[2];
    float vx3 = i[3];
    i += 4;

    vx0 *= vscale;
    vx1 *= vscale;
    vx2 *= vscale;
    vx3 *= vscale;

    vx0 = __builtin_wasm_max_f32(vx0, voutput_min_less_zero_point);
    vx1 = __builtin_wasm_max_f32(vx1, voutput_min_less_zero_point);
    vx2 = __builtin_wasm_max_f32(vx2, voutput_min_less_zero_point);
    vx3 = __builtin_wasm_max_f32(vx3, voutput_min_less_zero_point);

    vx0 = __builtin_wasm_min_f32(vx0, voutput_max_less_zero_point);
    vx1 = __builtin_wasm_min_f32(vx1, voutput_max_less_zero_point);
    vx2 = __builtin_wasm_min_f32(vx2, voutput_max_less_zero_point);
    vx3 = __builtin_wasm_min_f32(vx3, voutput_max_less_zero_point);

    vx0 += vmagic_bias;
    vx1 += vmagic_bias;
    vx2 += vmagic_bias;
    vx3 += vmagic_bias;

    int32_t vy0 = (int32_t) float_as_uint32(vx0);
    int32_t vy1 = (int32_t) float_as_uint32(vx1);
    int32_t vy2 = (int32_t) float_as_uint32(vx2);
    int32_t vy3 = (int32_t) float_as_uint32(vx3);

    vy0 -= vmagic_bias_less_zero_point;
    vy1 -= vmagic_bias_less_zero_point;
    vy2 -= vmagic_bias_less_zero_point;
    vy3 -= vmagic_bias_less_zero_point;

    output[0] = (uint8_t) vy0;
    output[1] = (uint8_t) vy1;
    output[2] = (uint8_t) vy2;
    output[3] = (uint8_t) vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = *i++;
      vx *= vscale;
      vx = __builtin_wasm_max_f32(vx, voutput_min_less_zero_point);
      vx = __builtin_wasm_min_f32(vx, voutput_max_less_zero_point);
      vx += vmagic_bias;

      int32_t vy = (int32_t) float_as_uint32(vx);
      vy -= vmagic_bias_less_zero_point;

      *output++ = (uint8_t) vy;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_rminmax_ukernel__wasm_u4_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float* i = input;

  float vmin0 = *i;
  float vmax0 = *i;
  float vmin1 = vmin0;
  float vmax1 = vmax0;
  float vmin2 = vmin0;
  float vmax2 = vmax0;
  float vmin3 = vmin0;
  float vmax3 = vmax0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vt0 = i[0];
    const float vt1 = i[1];
    const float vt2 = i[2];
    const float vt3 = i[3];
    i += 4;

    vmin0 = __builtin_wasm_min_f32(vmin0, vt0);
    vmax0 = __builtin_wasm_max_f32(vmax0, vt0);
    vmin1 = __builtin_wasm_min_f32(vmin1, vt1);
    vmax1 = __builtin_wasm_max_f32(vmax1, vt1);
    vmin2 = __builtin_wasm_min_f32(vmin2, vt2);
    vmax2 = __builtin_wasm_max_f32(vmax2, vt2);
    vmin3 = __builtin_wasm_min_f32(vmin3, vt3);
    vmax3 = __builtin_wasm_max_f32(vmax3, vt3);
  }
  vmin0 = __builtin_wasm_min_f32(vmin0, vmin1);
  vmax0 = __builtin_wasm_max_f32(vmax0, vmax1);
  vmin2 = __builtin_wasm_min_f32(vmin2, vmin3);
  vmax2 = __builtin_wasm_max_f32(vmax2, vmax3);
  vmin0 = __builtin_wasm_min_f32(vmin0, vmin2);
  vmax0 = __builtin_wasm_max_f32(vmax0, vmax2);

  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vt = *i++;
      vmin0 = __builtin_wasm_min_f32(vmin0, vt);
      vmax0 = __builtin_wasm_max_f32(vmax0, vt);
      batch -= sizeof(float);
    } while (batch != 0);
  }
  output[0] = vmin0;
  output[1] = vmax0;
}

void xnn_f32_vadd_minmax_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = va0 + vb0;
    float vacc1 = va1 + vb1;
    float vacc2 = va2 + vb2;
    float vacc3 = va3 + vb3;
    float vacc4 = va4 + vb4;
    float vacc5 = va5 + vb5;
    float vacc6 = va6 + vb6;
    float vacc7 = va7 + vb7;


    vacc0 = __builtin_wasm_max_f32(vacc0, voutput_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, voutput_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, voutput_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, voutput_min);
    vacc4 = __builtin_wasm_max_f32(vacc4, voutput_min);
    vacc5 = __builtin_wasm_max_f32(vacc5, voutput_min);
    vacc6 = __builtin_wasm_max_f32(vacc6, voutput_min);
    vacc7 = __builtin_wasm_max_f32(vacc7, voutput_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, voutput_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, voutput_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, voutput_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, voutput_max);
    vacc4 = __builtin_wasm_min_f32(vacc4, voutput_max);
    vacc5 = __builtin_wasm_min_f32(vacc5, voutput_max);
    vacc6 = __builtin_wasm_min_f32(vacc6, voutput_max);
    vacc7 = __builtin_wasm_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = va + vb;
      vacc = __builtin_wasm_max_f32(vacc, voutput_min);
      vacc = __builtin_wasm_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vaddc_minmax_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = va0 + vb;
    float vacc1 = va1 + vb;
    float vacc2 = va2 + vb;
    float vacc3 = va3 + vb;
    float vacc4 = va4 + vb;
    float vacc5 = va5 + vb;
    float vacc6 = va6 + vb;
    float vacc7 = va7 + vb;


    vacc0 = __builtin_wasm_max_f32(vacc0, voutput_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, voutput_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, voutput_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, voutput_min);
    vacc4 = __builtin_wasm_max_f32(vacc4, voutput_min);
    vacc5 = __builtin_wasm_max_f32(vacc5, voutput_min);
    vacc6 = __builtin_wasm_max_f32(vacc6, voutput_min);
    vacc7 = __builtin_wasm_max_f32(vacc7, voutput_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, voutput_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, voutput_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, voutput_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, voutput_max);
    vacc4 = __builtin_wasm_min_f32(vacc4, voutput_max);
    vacc5 = __builtin_wasm_min_f32(vacc5, voutput_max);
    vacc6 = __builtin_wasm_min_f32(vacc6, voutput_max);
    vacc7 = __builtin_wasm_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = va + vb;
      vacc = __builtin_wasm_max_f32(vacc, voutput_min);
      vacc = __builtin_wasm_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vdiv_minmax_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = va0 / vb0;
    float vacc1 = va1 / vb1;
    float vacc2 = va2 / vb2;
    float vacc3 = va3 / vb3;
    float vacc4 = va4 / vb4;
    float vacc5 = va5 / vb5;
    float vacc6 = va6 / vb6;
    float vacc7 = va7 / vb7;


    vacc0 = __builtin_wasm_max_f32(vacc0, voutput_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, voutput_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, voutput_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, voutput_min);
    vacc4 = __builtin_wasm_max_f32(vacc4, voutput_min);
    vacc5 = __builtin_wasm_max_f32(vacc5, voutput_min);
    vacc6 = __builtin_wasm_max_f32(vacc6, voutput_min);
    vacc7 = __builtin_wasm_max_f32(vacc7, voutput_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, voutput_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, voutput_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, voutput_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, voutput_max);
    vacc4 = __builtin_wasm_min_f32(vacc4, voutput_max);
    vacc5 = __builtin_wasm_min_f32(vacc5, voutput_max);
    vacc6 = __builtin_wasm_min_f32(vacc6, voutput_max);
    vacc7 = __builtin_wasm_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = va / vb;
      vacc = __builtin_wasm_max_f32(vacc, voutput_min);
      vacc = __builtin_wasm_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vdivc_minmax_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = va0 / vb;
    float vacc1 = va1 / vb;
    float vacc2 = va2 / vb;
    float vacc3 = va3 / vb;
    float vacc4 = va4 / vb;
    float vacc5 = va5 / vb;
    float vacc6 = va6 / vb;
    float vacc7 = va7 / vb;


    vacc0 = __builtin_wasm_max_f32(vacc0, voutput_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, voutput_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, voutput_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, voutput_min);
    vacc4 = __builtin_wasm_max_f32(vacc4, voutput_min);
    vacc5 = __builtin_wasm_max_f32(vacc5, voutput_min);
    vacc6 = __builtin_wasm_max_f32(vacc6, voutput_min);
    vacc7 = __builtin_wasm_max_f32(vacc7, voutput_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, voutput_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, voutput_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, voutput_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, voutput_max);
    vacc4 = __builtin_wasm_min_f32(vacc4, voutput_max);
    vacc5 = __builtin_wasm_min_f32(vacc5, voutput_max);
    vacc6 = __builtin_wasm_min_f32(vacc6, voutput_max);
    vacc7 = __builtin_wasm_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = va / vb;
      vacc = __builtin_wasm_max_f32(vacc, voutput_min);
      vacc = __builtin_wasm_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vmax_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = __builtin_wasm_max_f32(va0, vb0);
    float vacc1 = __builtin_wasm_max_f32(va1, vb1);
    float vacc2 = __builtin_wasm_max_f32(va2, vb2);
    float vacc3 = __builtin_wasm_max_f32(va3, vb3);
    float vacc4 = __builtin_wasm_max_f32(va4, vb4);
    float vacc5 = __builtin_wasm_max_f32(va5, vb5);
    float vacc6 = __builtin_wasm_max_f32(va6, vb6);
    float vacc7 = __builtin_wasm_max_f32(va7, vb7);



    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = __builtin_wasm_max_f32(va, vb);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vmaxc_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = __builtin_wasm_max_f32(va0, vb);
    float vacc1 = __builtin_wasm_max_f32(va1, vb);
    float vacc2 = __builtin_wasm_max_f32(va2, vb);
    float vacc3 = __builtin_wasm_max_f32(va3, vb);
    float vacc4 = __builtin_wasm_max_f32(va4, vb);
    float vacc5 = __builtin_wasm_max_f32(va5, vb);
    float vacc6 = __builtin_wasm_max_f32(va6, vb);
    float vacc7 = __builtin_wasm_max_f32(va7, vb);



    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = __builtin_wasm_max_f32(va, vb);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vmin_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = __builtin_wasm_min_f32(va0, vb0);
    float vacc1 = __builtin_wasm_min_f32(va1, vb1);
    float vacc2 = __builtin_wasm_min_f32(va2, vb2);
    float vacc3 = __builtin_wasm_min_f32(va3, vb3);
    float vacc4 = __builtin_wasm_min_f32(va4, vb4);
    float vacc5 = __builtin_wasm_min_f32(va5, vb5);
    float vacc6 = __builtin_wasm_min_f32(va6, vb6);
    float vacc7 = __builtin_wasm_min_f32(va7, vb7);



    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = __builtin_wasm_min_f32(va, vb);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vminc_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = __builtin_wasm_min_f32(va0, vb);
    float vacc1 = __builtin_wasm_min_f32(va1, vb);
    float vacc2 = __builtin_wasm_min_f32(va2, vb);
    float vacc3 = __builtin_wasm_min_f32(va3, vb);
    float vacc4 = __builtin_wasm_min_f32(va4, vb);
    float vacc5 = __builtin_wasm_min_f32(va5, vb);
    float vacc6 = __builtin_wasm_min_f32(va6, vb);
    float vacc7 = __builtin_wasm_min_f32(va7, vb);



    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = __builtin_wasm_min_f32(va, vb);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vmul_minmax_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = va0 * vb0;
    float vacc1 = va1 * vb1;
    float vacc2 = va2 * vb2;
    float vacc3 = va3 * vb3;
    float vacc4 = va4 * vb4;
    float vacc5 = va5 * vb5;
    float vacc6 = va6 * vb6;
    float vacc7 = va7 * vb7;


    vacc0 = __builtin_wasm_max_f32(vacc0, voutput_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, voutput_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, voutput_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, voutput_min);
    vacc4 = __builtin_wasm_max_f32(vacc4, voutput_min);
    vacc5 = __builtin_wasm_max_f32(vacc5, voutput_min);
    vacc6 = __builtin_wasm_max_f32(vacc6, voutput_min);
    vacc7 = __builtin_wasm_max_f32(vacc7, voutput_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, voutput_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, voutput_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, voutput_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, voutput_max);
    vacc4 = __builtin_wasm_min_f32(vacc4, voutput_max);
    vacc5 = __builtin_wasm_min_f32(vacc5, voutput_max);
    vacc6 = __builtin_wasm_min_f32(vacc6, voutput_max);
    vacc7 = __builtin_wasm_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = va * vb;
      vacc = __builtin_wasm_max_f32(vacc, voutput_min);
      vacc = __builtin_wasm_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vmulc_minmax_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = va0 * vb;
    float vacc1 = va1 * vb;
    float vacc2 = va2 * vb;
    float vacc3 = va3 * vb;
    float vacc4 = va4 * vb;
    float vacc5 = va5 * vb;
    float vacc6 = va6 * vb;
    float vacc7 = va7 * vb;


    vacc0 = __builtin_wasm_max_f32(vacc0, voutput_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, voutput_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, voutput_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, voutput_min);
    vacc4 = __builtin_wasm_max_f32(vacc4, voutput_min);
    vacc5 = __builtin_wasm_max_f32(vacc5, voutput_min);
    vacc6 = __builtin_wasm_max_f32(vacc6, voutput_min);
    vacc7 = __builtin_wasm_max_f32(vacc7, voutput_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, voutput_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, voutput_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, voutput_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, voutput_max);
    vacc4 = __builtin_wasm_min_f32(vacc4, voutput_max);
    vacc5 = __builtin_wasm_min_f32(vacc5, voutput_max);
    vacc6 = __builtin_wasm_min_f32(vacc6, voutput_max);
    vacc7 = __builtin_wasm_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = va * vb;
      vacc = __builtin_wasm_max_f32(vacc, voutput_min);
      vacc = __builtin_wasm_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vrdivc_minmax_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = vb / va0;
    float vacc1 = vb / va1;
    float vacc2 = vb / va2;
    float vacc3 = vb / va3;
    float vacc4 = vb / va4;
    float vacc5 = vb / va5;
    float vacc6 = vb / va6;
    float vacc7 = vb / va7;


    vacc0 = __builtin_wasm_max_f32(vacc0, voutput_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, voutput_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, voutput_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, voutput_min);
    vacc4 = __builtin_wasm_max_f32(vacc4, voutput_min);
    vacc5 = __builtin_wasm_max_f32(vacc5, voutput_min);
    vacc6 = __builtin_wasm_max_f32(vacc6, voutput_min);
    vacc7 = __builtin_wasm_max_f32(vacc7, voutput_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, voutput_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, voutput_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, voutput_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, voutput_max);
    vacc4 = __builtin_wasm_min_f32(vacc4, voutput_max);
    vacc5 = __builtin_wasm_min_f32(vacc5, voutput_max);
    vacc6 = __builtin_wasm_min_f32(vacc6, voutput_max);
    vacc7 = __builtin_wasm_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = vb / va;
      vacc = __builtin_wasm_max_f32(vacc, voutput_min);
      vacc = __builtin_wasm_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vrsubc_minmax_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = vb - va0;
    float vacc1 = vb - va1;
    float vacc2 = vb - va2;
    float vacc3 = vb - va3;
    float vacc4 = vb - va4;
    float vacc5 = vb - va5;
    float vacc6 = vb - va6;
    float vacc7 = vb - va7;


    vacc0 = __builtin_wasm_max_f32(vacc0, voutput_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, voutput_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, voutput_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, voutput_min);
    vacc4 = __builtin_wasm_max_f32(vacc4, voutput_min);
    vacc5 = __builtin_wasm_max_f32(vacc5, voutput_min);
    vacc6 = __builtin_wasm_max_f32(vacc6, voutput_min);
    vacc7 = __builtin_wasm_max_f32(vacc7, voutput_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, voutput_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, voutput_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, voutput_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, voutput_max);
    vacc4 = __builtin_wasm_min_f32(vacc4, voutput_max);
    vacc5 = __builtin_wasm_min_f32(vacc5, voutput_max);
    vacc6 = __builtin_wasm_min_f32(vacc6, voutput_max);
    vacc7 = __builtin_wasm_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = vb - va;
      vacc = __builtin_wasm_max_f32(vacc, voutput_min);
      vacc = __builtin_wasm_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vsub_minmax_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = va0 - vb0;
    float vacc1 = va1 - vb1;
    float vacc2 = va2 - vb2;
    float vacc3 = va3 - vb3;
    float vacc4 = va4 - vb4;
    float vacc5 = va5 - vb5;
    float vacc6 = va6 - vb6;
    float vacc7 = va7 - vb7;


    vacc0 = __builtin_wasm_max_f32(vacc0, voutput_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, voutput_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, voutput_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, voutput_min);
    vacc4 = __builtin_wasm_max_f32(vacc4, voutput_min);
    vacc5 = __builtin_wasm_max_f32(vacc5, voutput_min);
    vacc6 = __builtin_wasm_max_f32(vacc6, voutput_min);
    vacc7 = __builtin_wasm_max_f32(vacc7, voutput_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, voutput_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, voutput_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, voutput_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, voutput_max);
    vacc4 = __builtin_wasm_min_f32(vacc4, voutput_max);
    vacc5 = __builtin_wasm_min_f32(vacc5, voutput_max);
    vacc6 = __builtin_wasm_min_f32(vacc6, voutput_max);
    vacc7 = __builtin_wasm_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = va - vb;
      vacc = __builtin_wasm_max_f32(vacc, voutput_min);
      vacc = __builtin_wasm_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vsubc_minmax_ukernel__wasm_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = va0 - vb;
    float vacc1 = va1 - vb;
    float vacc2 = va2 - vb;
    float vacc3 = va3 - vb;
    float vacc4 = va4 - vb;
    float vacc5 = va5 - vb;
    float vacc6 = va6 - vb;
    float vacc7 = va7 - vb;


    vacc0 = __builtin_wasm_max_f32(vacc0, voutput_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, voutput_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, voutput_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, voutput_min);
    vacc4 = __builtin_wasm_max_f32(vacc4, voutput_min);
    vacc5 = __builtin_wasm_max_f32(vacc5, voutput_min);
    vacc6 = __builtin_wasm_max_f32(vacc6, voutput_min);
    vacc7 = __builtin_wasm_max_f32(vacc7, voutput_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, voutput_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, voutput_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, voutput_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, voutput_max);
    vacc4 = __builtin_wasm_min_f32(vacc4, voutput_max);
    vacc5 = __builtin_wasm_min_f32(vacc5, voutput_max);
    vacc6 = __builtin_wasm_min_f32(vacc6, voutput_max);
    vacc7 = __builtin_wasm_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = va - vb;
      vacc = __builtin_wasm_max_f32(vacc, voutput_min);
      vacc = __builtin_wasm_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vclamp_ukernel__wasm_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vacc0 = input[0];
    float vacc1 = input[1];
    float vacc2 = input[2];
    float vacc3 = input[3];
    input += 4;

    vacc0 = __builtin_wasm_max_f32(vacc0, vy_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, vy_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, vy_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, vy_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, vy_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, vy_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, vy_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, vy_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vacc = *input++;
      vacc = __builtin_wasm_max_f32(vacc, vy_min);
      vacc = __builtin_wasm_min_f32(vacc, vy_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_velu_ukernel__wasm_rr2_p6_u6(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsat_cutoff = -0x1.154246p+4f;
  const float vmagic_bias = 0x1.8000FEp23f;
  const float vlog2e = 0x1.715476p+0f;
  const float vminus_ln2_hi = -0x1.62E440p-1f;
  const float vminus_ln2_lo = 0x1.0105C6p-21f;
  const float vc6 = 0x1.6b7338p-10f;
  const float vc5 = 0x1.12278Ep-7f;
  const float vc4 = 0x1.555716p-5f;
  const float vc3 = 0x1.5554B0p-3f;
  const float vc2 = 0x1.FFFFFEp-2f;
  const float vone = 1.0f;

  const float vprescale = params->scalar.prescale;
  const float valpha = params->scalar.alpha;
  const float vbeta = params->scalar.beta;

  for (; batch >= 6 * sizeof(float); batch -= 6 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    float vx2 = input[2];
    float vx3 = input[3];
    float vx4 = input[4];
    float vx5 = input[5];
    input += 6;

    const float vz0 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx0 * vprescale, vsat_cutoff), 0.0f);
    const float vz1 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx1 * vprescale, vsat_cutoff), 0.0f);
    const float vz2 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx2 * vprescale, vsat_cutoff), 0.0f);
    const float vz3 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx3 * vprescale, vsat_cutoff), 0.0f);
    const float vz4 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx4 * vprescale, vsat_cutoff), 0.0f);
    const float vz5 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx5 * vprescale, vsat_cutoff), 0.0f);

    float vn0 = vz0 * vlog2e + vmagic_bias;
    float vn1 = vz1 * vlog2e + vmagic_bias;
    float vn2 = vz2 * vlog2e + vmagic_bias;
    float vn3 = vz3 * vlog2e + vmagic_bias;
    float vn4 = vz4 * vlog2e + vmagic_bias;
    float vn5 = vz5 * vlog2e + vmagic_bias;

    float vs0 = uint32_as_float(float_as_uint32(vn0) << 23);
    vn0 -= vmagic_bias;
    float vs1 = uint32_as_float(float_as_uint32(vn1) << 23);
    vn1 -= vmagic_bias;
    float vs2 = uint32_as_float(float_as_uint32(vn2) << 23);
    vn2 -= vmagic_bias;
    float vs3 = uint32_as_float(float_as_uint32(vn3) << 23);
    vn3 -= vmagic_bias;
    float vs4 = uint32_as_float(float_as_uint32(vn4) << 23);
    vn4 -= vmagic_bias;
    float vs5 = uint32_as_float(float_as_uint32(vn5) << 23);
    vn5 -= vmagic_bias;

    float vt0 = vn0 * vminus_ln2_hi + vz0;
    float vt1 = vn1 * vminus_ln2_hi + vz1;
    float vt2 = vn2 * vminus_ln2_hi + vz2;
    float vt3 = vn3 * vminus_ln2_hi + vz3;
    float vt4 = vn4 * vminus_ln2_hi + vz4;
    float vt5 = vn5 * vminus_ln2_hi + vz5;

    vt0 = vn0 * vminus_ln2_lo + vt0;
    vt1 = vn1 * vminus_ln2_lo + vt1;
    vt2 = vn2 * vminus_ln2_lo + vt2;
    vt3 = vn3 * vminus_ln2_lo + vt3;
    vt4 = vn4 * vminus_ln2_lo + vt4;
    vt5 = vn5 * vminus_ln2_lo + vt5;


    float vp0 = vc6 * vt0 + vc5;
    float vp1 = vc6 * vt1 + vc5;
    float vp2 = vc6 * vt2 + vc5;
    float vp3 = vc6 * vt3 + vc5;
    float vp4 = vc6 * vt4 + vc5;
    float vp5 = vc6 * vt5 + vc5;

    vp0 = vp0 * vt0 + vc4;
    vp1 = vp1 * vt1 + vc4;
    vp2 = vp2 * vt2 + vc4;
    vp3 = vp3 * vt3 + vc4;
    vp4 = vp4 * vt4 + vc4;
    vp5 = vp5 * vt5 + vc4;

    vp0 = vp0 * vt0 + vc3;
    vp1 = vp1 * vt1 + vc3;
    vp2 = vp2 * vt2 + vc3;
    vp3 = vp3 * vt3 + vc3;
    vp4 = vp4 * vt4 + vc3;
    vp5 = vp5 * vt5 + vc3;

    vp0 = vp0 * vt0 + vc2;
    vp1 = vp1 * vt1 + vc2;
    vp2 = vp2 * vt2 + vc2;
    vp3 = vp3 * vt3 + vc2;
    vp4 = vp4 * vt4 + vc2;
    vp5 = vp5 * vt5 + vc2;

    vp0 *= vt0;
    vp1 *= vt1;
    vp2 *= vt2;
    vp3 *= vt3;
    vp4 *= vt4;
    vp5 *= vt5;

    vt0 *= vs0;
    vs0 -= vone;
    vt1 *= vs1;
    vs1 -= vone;
    vt2 *= vs2;
    vs2 -= vone;
    vt3 *= vs3;
    vs3 -= vone;
    vt4 *= vs4;
    vs4 -= vone;
    vt5 *= vs5;
    vs5 -= vone;

    vp0 = vp0 * vt0 + vt0;
    vp1 = vp1 * vt1 + vt1;
    vp2 = vp2 * vt2 + vt2;
    vp3 = vp3 * vt3 + vt3;
    vp4 = vp4 * vt4 + vt4;
    vp5 = vp5 * vt5 + vt5;

    const float ve0 = (vp0 + vs0) * valpha;
    float vy0 = __builtin_wasm_max_f32(vx0 * vbeta, 0.0f);
    const float ve1 = (vp1 + vs1) * valpha;
    float vy1 = __builtin_wasm_max_f32(vx1 * vbeta, 0.0f);
    const float ve2 = (vp2 + vs2) * valpha;
    float vy2 = __builtin_wasm_max_f32(vx2 * vbeta, 0.0f);
    const float ve3 = (vp3 + vs3) * valpha;
    float vy3 = __builtin_wasm_max_f32(vx3 * vbeta, 0.0f);
    const float ve4 = (vp4 + vs4) * valpha;
    float vy4 = __builtin_wasm_max_f32(vx4 * vbeta, 0.0f);
    const float ve5 = (vp5 + vs5) * valpha;
    float vy5 = __builtin_wasm_max_f32(vx5 * vbeta, 0.0f);

    vy0 += __builtin_wasm_min_f32(ve0, 0.0f);
    vy1 += __builtin_wasm_min_f32(ve1, 0.0f);
    vy2 += __builtin_wasm_min_f32(ve2, 0.0f);
    vy3 += __builtin_wasm_min_f32(ve3, 0.0f);
    vy4 += __builtin_wasm_min_f32(ve4, 0.0f);
    vy5 += __builtin_wasm_min_f32(ve5, 0.0f);

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output[4] = vy4;
    output[5] = vy5;
    output += 6;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = *input++;

      const float vz = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx * vprescale, vsat_cutoff), 0.0f);

      float vn = vz * vlog2e + vmagic_bias;
      float vs = uint32_as_float(float_as_uint32(vn) << 23);
      vn -= vmagic_bias;

      float vt = vn * vminus_ln2_hi + vz;
      vt = vn * vminus_ln2_lo + vt;


      float vp = vc6 * vt + vc5;
      vp = vp * vt + vc4;
      vp = vp * vt + vc3;
      vp = vp * vt + vc2;
      vp *= vt;

      vt *= vs;
      vs -= vone;
      vp = vp * vt + vt;
      const float ve = (vp + vs) * valpha;

      float vy = __builtin_wasm_max_f32(vx * vbeta, 0.0f);
      vy += __builtin_wasm_min_f32(ve, 0.0f);

      *output++ = vy;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vhswish_ukernel__wasm_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsixth = 0x1.555556p-3f;
  const float vthree = 3.0f;
  const float vsix = 6.0f;
  const float vzero = 0.0f;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    float vx2 = input[2];
    float vx3 = input[3];
    input += 4;

    float vacc0 = vx0 + vthree;
    vx0 *= vsixth;
    float vacc1 = vx1 + vthree;
    vx1 *= vsixth;
    float vacc2 = vx2 + vthree;
    vx2 *= vsixth;
    float vacc3 = vx3 + vthree;
    vx3 *= vsixth;

    vacc0 = __builtin_wasm_max_f32(vacc0, vzero);
    vacc1 = __builtin_wasm_max_f32(vacc1, vzero);
    vacc2 = __builtin_wasm_max_f32(vacc2, vzero);
    vacc3 = __builtin_wasm_max_f32(vacc3, vzero);

    vacc0 = __builtin_wasm_min_f32(vacc0, vsix);
    vacc1 = __builtin_wasm_min_f32(vacc1, vsix);
    vacc2 = __builtin_wasm_min_f32(vacc2, vsix);
    vacc3 = __builtin_wasm_min_f32(vacc3, vsix);

    vacc0 *= vx0;
    vacc1 *= vx1;
    vacc2 *= vx2;
    vacc3 *= vx3;

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = *input++;
      float vacc = vx + vthree;
      vx *= vsixth;
      vacc = __builtin_wasm_max_f32(vacc, vzero);
      vacc = __builtin_wasm_min_f32(vacc, vsix);
      vacc *= vx;
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    do {
      const float vscale = w[0];

      float vacc0 = *i0++;
      float vacc1 = *i1++;

      const float vbias = w[1];

      vacc0 = vacc0 * vscale + vbias;
      vacc1 = vacc1 * vscale + vbias;

      vacc0 = __builtin_wasm_max_f32(vacc0, vmin);
      vacc1 = __builtin_wasm_max_f32(vacc1, vmin);

      vacc0 = __builtin_wasm_min_f32(vacc0, vmax);
      vacc1 = __builtin_wasm_min_f32(vacc1, vmax);

      *o0++ = vacc0;
      *o1++ = vacc1;

      w += 2;
      c -= sizeof(float);
    } while (c != 0);
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_vrelu_ukernel__wasm_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vzero = 0.0f;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float vacc0 = input[0];
    float vacc1 = input[1];
    float vacc2 = input[2];
    float vacc3 = input[3];
    float vacc4 = input[4];
    float vacc5 = input[5];
    float vacc6 = input[6];
    float vacc7 = input[7];
    input += 8;

    vacc0 = __builtin_wasm_max_f32(vacc0, vzero);
    vacc1 = __builtin_wasm_max_f32(vacc1, vzero);
    vacc2 = __builtin_wasm_max_f32(vacc2, vzero);
    vacc3 = __builtin_wasm_max_f32(vacc3, vzero);
    vacc4 = __builtin_wasm_max_f32(vacc4, vzero);
    vacc5 = __builtin_wasm_max_f32(vacc5, vzero);
    vacc6 = __builtin_wasm_max_f32(vacc6, vzero);
    vacc7 = __builtin_wasm_max_f32(vacc7, vzero);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vacc = *input++;
      vacc = __builtin_wasm_max_f32(vacc, vzero);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsat_cutoff = params->scalar_expm1minus_rr1_p6h5.sat_cutoff;
  const float vminus_log2e = params->scalar_expm1minus_rr1_p6h5.minus_log2e;
  const float vmagic_bias = params->scalar_expm1minus_rr1_p6h5.magic_bias;
  const float vln2 = params->scalar_expm1minus_rr1_p6h5.ln2;
  const float vc6 = params->scalar_expm1minus_rr1_p6h5.c6;
  const float vc5 = params->scalar_expm1minus_rr1_p6h5.c5;
  const float vc4 = params->scalar_expm1minus_rr1_p6h5.c4;
  const float vc3 = params->scalar_expm1minus_rr1_p6h5.c3;
  const float vc2 = params->scalar_expm1minus_rr1_p6h5.c2;
  const float vminus_two = params->scalar_expm1minus_rr1_p6h5.minus_two;
  const float vone = params->scalar_expm1minus_rr1_p6h5.one;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    float vz0 = fabsf(vx0);
    float vz1 = fabsf(vx1);
    float vz2 = fabsf(vx2);
    float vz3 = fabsf(vx3);

    vz0 = __builtin_wasm_min_f32(vz0, vsat_cutoff);
    vz1 = __builtin_wasm_min_f32(vz1, vsat_cutoff);
    vz2 = __builtin_wasm_min_f32(vz2, vsat_cutoff);
    vz3 = __builtin_wasm_min_f32(vz3, vsat_cutoff);

    float vn0 = vz0 * vminus_log2e + vmagic_bias;
    float vn1 = vz1 * vminus_log2e + vmagic_bias;
    float vn2 = vz2 * vminus_log2e + vmagic_bias;
    float vn3 = vz3 * vminus_log2e + vmagic_bias;

    const uint32_t vb0 = float_as_uint32(vn0);
    vn0 -= vmagic_bias;
    const uint32_t vb1 = float_as_uint32(vn1);
    vn1 -= vmagic_bias;
    const uint32_t vb2 = float_as_uint32(vn2);
    vn2 -= vmagic_bias;
    const uint32_t vb3 = float_as_uint32(vn3);
    vn3 -= vmagic_bias;

    const uint32_t ve0 = vb0 << 23;
    const uint32_t ve1 = vb1 << 23;
    const uint32_t ve2 = vb2 << 23;
    const uint32_t ve3 = vb3 << 23;

    const float vt0 = vn0 * vln2 + vz0;
    const float vs0 = uint32_as_float(ve0);
    const float vt1 = vn1 * vln2 + vz1;
    const float vs1 = uint32_as_float(ve1);
    const float vt2 = vn2 * vln2 + vz2;
    const float vs2 = uint32_as_float(ve2);
    const float vt3 = vn3 * vln2 + vz3;
    const float vs3 = uint32_as_float(ve3);

    float vp0 = vc6 * vt0 + vc5;
    float vp1 = vc6 * vt1 + vc5;
    float vp2 = vc6 * vt2 + vc5;
    float vp3 = vc6 * vt3 + vc5;
    vp0 = vp0 * vt0 + vc4;
    vp1 = vp1 * vt1 + vc4;
    vp2 = vp2 * vt2 + vc4;
    vp3 = vp3 * vt3 + vc4;
    vp0 = vp0 * vt0 + vc3;
    vp1 = vp1 * vt1 + vc3;
    vp2 = vp2 * vt2 + vc3;
    vp3 = vp3 * vt3 + vc3;
    vp0 = vp0 * vt0 + vc2;
    vp1 = vp1 * vt1 + vc2;
    vp2 = vp2 * vt2 + vc2;
    vp3 = vp3 * vt3 + vc2;
    vp0 = vp0 * vt0 + vminus_two;
    vp1 = vp1 * vt1 + vminus_two;
    vp2 = vp2 * vt2 + vminus_two;
    vp3 = vp3 * vt3 + vminus_two;

    const float vts0 = vt0 * vs0;
    const float vsmo0 = vs0 - vone;
    const float vts1 = vt1 * vs1;
    const float vsmo1 = vs1 - vone;
    const float vts2 = vt2 * vs2;
    const float vsmo2 = vs2 - vone;
    const float vts3 = vt3 * vs3;
    const float vsmo3 = vs3 - vone;

    const float vemo0 = vp0 * vts0 + vsmo0;
    const float vemo1 = vp1 * vts1 + vsmo1;
    const float vemo2 = vp2 * vts2 + vsmo2;
    const float vemo3 = vp3 * vts3 + vsmo3;

    const float vepo0 = vemo0 - vminus_two;
    const float vepo1 = vemo1 - vminus_two;
    const float vepo2 = vemo2 - vminus_two;
    const float vepo3 = vemo3 - vminus_two;

    float vy0 = vemo0 / vepo0;
    float vy1 = vemo1 / vepo1;
    float vy2 = vemo2 / vepo2;
    float vy3 = vemo3 / vepo3;

    vy0 = copysignf(vy0, vx0);
    vy1 = copysignf(vy1, vx1);
    vy2 = copysignf(vy2, vx2);
    vy3 = copysignf(vy3, vx3);

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;

      float vz = fabsf(vx);

      vz = __builtin_wasm_min_f32(vz, vsat_cutoff);

      float vn = vz * vminus_log2e + vmagic_bias;

      const uint32_t vb = float_as_uint32(vn);
      vn -= vmagic_bias;

      const uint32_t ve = vb << 23;
      const float vs = uint32_as_float(ve);

      const float vt = vn * vln2 + vz;

      float vp = vc6 * vt + vc5;
      vp = vp * vt + vc4;
      vp = vp * vt + vc3;
      vp = vp * vt + vc2;
      vp = vp * vt + vminus_two;

      const float vts = vt * vs;
      const float vsmo = vs - vone;
      const float vemo = vp * vts + vsmo;

      const float vepo = vemo - vminus_two;

      float vy = vemo / vepo;

      vy = copysignf(vy, vx);

      *output++ = vy;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;

  kc = round_up_po2(kc, 2);
  do {
    const int32_t vksum0 = ((const int32_t*) w)[0];
    const int32_t vksum1 = ((const int32_t*) w)[1];
    const int32_t vksum2 = ((const int32_t*) w)[2];
    const int32_t vksum3 = ((const int32_t*) w)[3];
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    int32_t vacc0x2 = vksum2 * vinput_zero_point0;
    int32_t vacc0x3 = vksum3 * vinput_zero_point0;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    for (; k >= 2 * sizeof(uint8_t); k -= 2 * sizeof(uint8_t)) {
      const int32_t va0c0 = (int32_t) a0[0];
      const int32_t va0c1 = (int32_t) a0[1];
      a0 += 2;

      const uint8_t vbi0 = ((const uint8_t*) w)[0];
      const uint8_t vbi1 = ((const uint8_t*) w)[1];
      const uint8_t vbi2 = ((const uint8_t*) w)[2];
      const uint8_t vbi3 = ((const uint8_t*) w)[3];
      w = (const uint8_t*) w + 4;
      const int32_t vb0c0 = (int32_t) (int8_t) (vbi0 << 4);
      const int32_t vb0c1 = (int32_t) (int8_t) (vbi0 & 0xF0);
      const int32_t vb1c0 = (int32_t) (int8_t) (vbi1 << 4);
      const int32_t vb1c1 = (int32_t) (int8_t) (vbi1 & 0xF0);
      const int32_t vb2c0 = (int32_t) (int8_t) (vbi2 << 4);
      const int32_t vb2c1 = (int32_t) (int8_t) (vbi2 & 0xF0);
      const int32_t vb3c0 = (int32_t) (int8_t) (vbi3 << 4);
      const int32_t vb3c1 = (int32_t) (int8_t) (vbi3 & 0xF0);

      vacc0x0 += va0c0 * vb0c0;
      vacc0x1 += va0c0 * vb1c0;
      vacc0x2 += va0c0 * vb2c0;
      vacc0x3 += va0c0 * vb3c0;
      vacc0x0 += va0c1 * vb0c1;
      vacc0x1 += va0c1 * vb1c1;
      vacc0x2 += va0c1 * vb2c1;
      vacc0x3 += va0c1 * vb3c1;
    }

    float vout0x0 = (float) math_asr_s32(vacc0x0, 4);
    float vout0x1 = (float) math_asr_s32(vacc0x1, 4);
    float vout0x2 = (float) math_asr_s32(vacc0x2, 4);
    float vout0x3 = (float) math_asr_s32(vacc0x3, 4);

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;
    vout0x2 *= vinput_scale0;
    vout0x3 *= vinput_scale0;

    const float vfilter_output_scale0 = ((const float*) w)[0];
    vout0x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = ((const float*) w)[1];
    vout0x1 *= vfilter_output_scale1;
    const float vfilter_output_scale2 = ((const float*) w)[2];
    vout0x2 *= vfilter_output_scale2;
    const float vfilter_output_scale3 = ((const float*) w)[3];
    vout0x3 *= vfilter_output_scale3;

    const float vbias0 = ((const float*) w)[4];
    vout0x0 += vbias0;
    const float vbias1 = ((const float*) w)[5];
    vout0x1 += vbias1;
    const float vbias2 = ((const float*) w)[6];
    vout0x2 += vbias2;
    const float vbias3 = ((const float*) w)[7];
    vout0x3 += vbias3;

    w = (const float*) w + 8;

    const float voutput_min = params->scalar.min;
    vout0x0 = __builtin_wasm_max_f32(vout0x0, voutput_min);
    vout0x1 = __builtin_wasm_max_f32(vout0x1, voutput_min);
    vout0x2 = __builtin_wasm_max_f32(vout0x2, voutput_min);
    vout0x3 = __builtin_wasm_max_f32(vout0x3, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = __builtin_wasm_min_f32(vout0x0, voutput_max);
    vout0x1 = __builtin_wasm_min_f32(vout0x1, voutput_max);
    vout0x2 = __builtin_wasm_min_f32(vout0x2, voutput_max);
    vout0x3 = __builtin_wasm_min_f32(vout0x3, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  kc = round_up_po2(kc, 2);
  do {
    const int32_t vksum0 = ((const int32_t*) w)[0];
    const int32_t vksum1 = ((const int32_t*) w)[1];
    const int32_t vksum2 = ((const int32_t*) w)[2];
    const int32_t vksum3 = ((const int32_t*) w)[3];
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    int32_t vacc0x2 = vksum2 * vinput_zero_point0;
    int32_t vacc0x3 = vksum3 * vinput_zero_point0;
    const int32_t vinput_zero_point1 = quantization_params[1].zero_point;
    int32_t vacc1x0 = vksum0 * vinput_zero_point1;
    int32_t vacc1x1 = vksum1 * vinput_zero_point1;
    int32_t vacc1x2 = vksum2 * vinput_zero_point1;
    int32_t vacc1x3 = vksum3 * vinput_zero_point1;
    const int32_t vinput_zero_point2 = quantization_params[2].zero_point;
    int32_t vacc2x0 = vksum0 * vinput_zero_point2;
    int32_t vacc2x1 = vksum1 * vinput_zero_point2;
    int32_t vacc2x2 = vksum2 * vinput_zero_point2;
    int32_t vacc2x3 = vksum3 * vinput_zero_point2;
    const int32_t vinput_zero_point3 = quantization_params[3].zero_point;
    int32_t vacc3x0 = vksum0 * vinput_zero_point3;
    int32_t vacc3x1 = vksum1 * vinput_zero_point3;
    int32_t vacc3x2 = vksum2 * vinput_zero_point3;
    int32_t vacc3x3 = vksum3 * vinput_zero_point3;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    for (; k >= 2 * sizeof(uint8_t); k -= 2 * sizeof(uint8_t)) {
      const int32_t va0c0 = (int32_t) a0[0];
      const int32_t va0c1 = (int32_t) a0[1];
      a0 += 2;
      const int32_t va1c0 = (int32_t) a1[0];
      const int32_t va1c1 = (int32_t) a1[1];
      a1 += 2;
      const int32_t va2c0 = (int32_t) a2[0];
      const int32_t va2c1 = (int32_t) a2[1];
      a2 += 2;
      const int32_t va3c0 = (int32_t) a3[0];
      const int32_t va3c1 = (int32_t) a3[1];
      a3 += 2;

      const uint8_t vbi0 = ((const uint8_t*) w)[0];
      const uint8_t vbi1 = ((const uint8_t*) w)[1];
      const uint8_t vbi2 = ((const uint8_t*) w)[2];
      const uint8_t vbi3 = ((const uint8_t*) w)[3];
      w = (const uint8_t*) w + 4;
      const int32_t vb0c0 = (int32_t) (int8_t) (vbi0 << 4);
      const int32_t vb0c1 = (int32_t) (int8_t) (vbi0 & 0xF0);
      const int32_t vb1c0 = (int32_t) (int8_t) (vbi1 << 4);
      const int32_t vb1c1 = (int32_t) (int8_t) (vbi1 & 0xF0);
      const int32_t vb2c0 = (int32_t) (int8_t) (vbi2 << 4);
      const int32_t vb2c1 = (int32_t) (int8_t) (vbi2 & 0xF0);
      const int32_t vb3c0 = (int32_t) (int8_t) (vbi3 << 4);
      const int32_t vb3c1 = (int32_t) (int8_t) (vbi3 & 0xF0);

      vacc0x0 += va0c0 * vb0c0;
      vacc0x1 += va0c0 * vb1c0;
      vacc0x2 += va0c0 * vb2c0;
      vacc0x3 += va0c0 * vb3c0;
      vacc1x0 += va1c0 * vb0c0;
      vacc1x1 += va1c0 * vb1c0;
      vacc1x2 += va1c0 * vb2c0;
      vacc1x3 += va1c0 * vb3c0;
      vacc2x0 += va2c0 * vb0c0;
      vacc2x1 += va2c0 * vb1c0;
      vacc2x2 += va2c0 * vb2c0;
      vacc2x3 += va2c0 * vb3c0;
      vacc3x0 += va3c0 * vb0c0;
      vacc3x1 += va3c0 * vb1c0;
      vacc3x2 += va3c0 * vb2c0;
      vacc3x3 += va3c0 * vb3c0;
      vacc0x0 += va0c1 * vb0c1;
      vacc0x1 += va0c1 * vb1c1;
      vacc0x2 += va0c1 * vb2c1;
      vacc0x3 += va0c1 * vb3c1;
      vacc1x0 += va1c1 * vb0c1;
      vacc1x1 += va1c1 * vb1c1;
      vacc1x2 += va1c1 * vb2c1;
      vacc1x3 += va1c1 * vb3c1;
      vacc2x0 += va2c1 * vb0c1;
      vacc2x1 += va2c1 * vb1c1;
      vacc2x2 += va2c1 * vb2c1;
      vacc2x3 += va2c1 * vb3c1;
      vacc3x0 += va3c1 * vb0c1;
      vacc3x1 += va3c1 * vb1c1;
      vacc3x2 += va3c1 * vb2c1;
      vacc3x3 += va3c1 * vb3c1;
    }

    float vout0x0 = (float) math_asr_s32(vacc0x0, 4);
    float vout0x1 = (float) math_asr_s32(vacc0x1, 4);
    float vout0x2 = (float) math_asr_s32(vacc0x2, 4);
    float vout0x3 = (float) math_asr_s32(vacc0x3, 4);
    float vout1x0 = (float) math_asr_s32(vacc1x0, 4);
    float vout1x1 = (float) math_asr_s32(vacc1x1, 4);
    float vout1x2 = (float) math_asr_s32(vacc1x2, 4);
    float vout1x3 = (float) math_asr_s32(vacc1x3, 4);
    float vout2x0 = (float) math_asr_s32(vacc2x0, 4);
    float vout2x1 = (float) math_asr_s32(vacc2x1, 4);
    float vout2x2 = (float) math_asr_s32(vacc2x2, 4);
    float vout2x3 = (float) math_asr_s32(vacc2x3, 4);
    float vout3x0 = (float) math_asr_s32(vacc3x0, 4);
    float vout3x1 = (float) math_asr_s32(vacc3x1, 4);
    float vout3x2 = (float) math_asr_s32(vacc3x2, 4);
    float vout3x3 = (float) math_asr_s32(vacc3x3, 4);

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;
    vout0x2 *= vinput_scale0;
    vout0x3 *= vinput_scale0;
    const float vinput_scale1 = quantization_params[1].inv_scale;
    vout1x0 *= vinput_scale1;
    vout1x1 *= vinput_scale1;
    vout1x2 *= vinput_scale1;
    vout1x3 *= vinput_scale1;
    const float vinput_scale2 = quantization_params[2].inv_scale;
    vout2x0 *= vinput_scale2;
    vout2x1 *= vinput_scale2;
    vout2x2 *= vinput_scale2;
    vout2x3 *= vinput_scale2;
    const float vinput_scale3 = quantization_params[3].inv_scale;
    vout3x0 *= vinput_scale3;
    vout3x1 *= vinput_scale3;
    vout3x2 *= vinput_scale3;
    vout3x3 *= vinput_scale3;

    const float vfilter_output_scale0 = ((const float*) w)[0];
    vout0x0 *= vfilter_output_scale0;
    vout1x0 *= vfilter_output_scale0;
    vout2x0 *= vfilter_output_scale0;
    vout3x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = ((const float*) w)[1];
    vout0x1 *= vfilter_output_scale1;
    vout1x1 *= vfilter_output_scale1;
    vout2x1 *= vfilter_output_scale1;
    vout3x1 *= vfilter_output_scale1;
    const float vfilter_output_scale2 = ((const float*) w)[2];
    vout0x2 *= vfilter_output_scale2;
    vout1x2 *= vfilter_output_scale2;
    vout2x2 *= vfilter_output_scale2;
    vout3x2 *= vfilter_output_scale2;
    const float vfilter_output_scale3 = ((const float*) w)[3];
    vout0x3 *= vfilter_output_scale3;
    vout1x3 *= vfilter_output_scale3;
    vout2x3 *= vfilter_output_scale3;
    vout3x3 *= vfilter_output_scale3;

    const float vbias0 = ((const float*) w)[4];
    vout0x0 += vbias0;
    vout1x0 += vbias0;
    vout2x0 += vbias0;
    vout3x0 += vbias0;
    const float vbias1 = ((const float*) w)[5];
    vout0x1 += vbias1;
    vout1x1 += vbias1;
    vout2x1 += vbias1;
    vout3x1 += vbias1;
    const float vbias2 = ((const float*) w)[6];
    vout0x2 += vbias2;
    vout1x2 += vbias2;
    vout2x2 += vbias2;
    vout3x2 += vbias2;
    const float vbias3 = ((const float*) w)[7];
    vout0x3 += vbias3;
    vout1x3 += vbias3;
    vout2x3 += vbias3;
    vout3x3 += vbias3;

    w = (const float*) w + 8;

    const float voutput_min = params->scalar.min;
    vout0x0 = __builtin_wasm_max_f32(vout0x0, voutput_min);
    vout1x0 = __builtin_wasm_max_f32(vout1x0, voutput_min);
    vout2x0 = __builtin_wasm_max_f32(vout2x0, voutput_min);
    vout3x0 = __builtin_wasm_max_f32(vout3x0, voutput_min);
    vout0x1 = __builtin_wasm_max_f32(vout0x1, voutput_min);
    vout1x1 = __builtin_wasm_max_f32(vout1x1, voutput_min);
    vout2x1 = __builtin_wasm_max_f32(vout2x1, voutput_min);
    vout3x1 = __builtin_wasm_max_f32(vout3x1, voutput_min);
    vout0x2 = __builtin_wasm_max_f32(vout0x2, voutput_min);
    vout1x2 = __builtin_wasm_max_f32(vout1x2, voutput_min);
    vout2x2 = __builtin_wasm_max_f32(vout2x2, voutput_min);
    vout3x2 = __builtin_wasm_max_f32(vout3x2, voutput_min);
    vout0x3 = __builtin_wasm_max_f32(vout0x3, voutput_min);
    vout1x3 = __builtin_wasm_max_f32(vout1x3, voutput_min);
    vout2x3 = __builtin_wasm_max_f32(vout2x3, voutput_min);
    vout3x3 = __builtin_wasm_max_f32(vout3x3, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = __builtin_wasm_min_f32(vout0x0, voutput_max);
    vout1x0 = __builtin_wasm_min_f32(vout1x0, voutput_max);
    vout2x0 = __builtin_wasm_min_f32(vout2x0, voutput_max);
    vout3x0 = __builtin_wasm_min_f32(vout3x0, voutput_max);
    vout0x1 = __builtin_wasm_min_f32(vout0x1, voutput_max);
    vout1x1 = __builtin_wasm_min_f32(vout1x1, voutput_max);
    vout2x1 = __builtin_wasm_min_f32(vout2x1, voutput_max);
    vout3x1 = __builtin_wasm_min_f32(vout3x1, voutput_max);
    vout0x2 = __builtin_wasm_min_f32(vout0x2, voutput_max);
    vout1x2 = __builtin_wasm_min_f32(vout1x2, voutput_max);
    vout2x2 = __builtin_wasm_min_f32(vout2x2, voutput_max);
    vout3x2 = __builtin_wasm_min_f32(vout3x2, voutput_max);
    vout0x3 = __builtin_wasm_min_f32(vout0x3, voutput_max);
    vout1x3 = __builtin_wasm_min_f32(vout1x3, voutput_max);
    vout2x3 = __builtin_wasm_min_f32(vout2x3, voutput_max);
    vout3x3 = __builtin_wasm_min_f32(vout3x3, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;
      c1[0] = vout1x0;
      c1[1] = vout1x1;
      c1[2] = vout1x2;
      c1[3] = vout1x3;
      c2[0] = vout2x0;
      c2[1] = vout2x1;
      c2[2] = vout2x2;
      c2[3] = vout2x3;
      c3[0] = vout3x0;
      c3[1] = vout3x1;
      c3[2] = vout3x2;
      c3[3] = vout3x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = vout1x0;
        c1[1] = vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = vout2x0;
        c2[1] = vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c3[0] = vout3x0;
        c3[1] = vout3x1;
        vout3x0 = vout3x2;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = vout0x0;
        c1[0] = vout1x0;
        c2[0] = vout2x0;
        c3[0] = vout3x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;

  do {
    const int32_t vksum0 = ((const int32_t*) w)[0];
    const int32_t vksum1 = ((const int32_t*) w)[1];
    const int32_t vksum2 = ((const int32_t*) w)[2];
    const int32_t vksum3 = ((const int32_t*) w)[3];
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    int32_t vacc0x2 = vksum2 * vinput_zero_point0;
    int32_t vacc0x3 = vksum3 * vinput_zero_point0;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    float vout0x2 = (float) vacc0x2;
    float vout0x3 = (float) vacc0x3;

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;
    vout0x2 *= vinput_scale0;
    vout0x3 *= vinput_scale0;

    const float vfilter_output_scale0 = ((const float*) w)[0];
    vout0x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = ((const float*) w)[1];
    vout0x1 *= vfilter_output_scale1;
    const float vfilter_output_scale2 = ((const float*) w)[2];
    vout0x2 *= vfilter_output_scale2;
    const float vfilter_output_scale3 = ((const float*) w)[3];
    vout0x3 *= vfilter_output_scale3;

    const float vbias0 = ((const float*) w)[4];
    vout0x0 += vbias0;
    const float vbias1 = ((const float*) w)[5];
    vout0x1 += vbias1;
    const float vbias2 = ((const float*) w)[6];
    vout0x2 += vbias2;
    const float vbias3 = ((const float*) w)[7];
    vout0x3 += vbias3;

    w = (const float*) w + 8;

    const float voutput_min = params->scalar.min;
    vout0x0 = __builtin_wasm_max_f32(vout0x0, voutput_min);
    vout0x1 = __builtin_wasm_max_f32(vout0x1, voutput_min);
    vout0x2 = __builtin_wasm_max_f32(vout0x2, voutput_min);
    vout0x3 = __builtin_wasm_max_f32(vout0x3, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = __builtin_wasm_min_f32(vout0x0, voutput_max);
    vout0x1 = __builtin_wasm_min_f32(vout0x1, voutput_max);
    vout0x2 = __builtin_wasm_min_f32(vout0x2, voutput_max);
    vout0x3 = __builtin_wasm_min_f32(vout0x3, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    const int32_t vksum0 = ((const int32_t*) w)[0];
    const int32_t vksum1 = ((const int32_t*) w)[1];
    const int32_t vksum2 = ((const int32_t*) w)[2];
    const int32_t vksum3 = ((const int32_t*) w)[3];
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    int32_t vacc0x2 = vksum2 * vinput_zero_point0;
    int32_t vacc0x3 = vksum3 * vinput_zero_point0;
    const int32_t vinput_zero_point1 = quantization_params[1].zero_point;
    int32_t vacc1x0 = vksum0 * vinput_zero_point1;
    int32_t vacc1x1 = vksum1 * vinput_zero_point1;
    int32_t vacc1x2 = vksum2 * vinput_zero_point1;
    int32_t vacc1x3 = vksum3 * vinput_zero_point1;
    const int32_t vinput_zero_point2 = quantization_params[2].zero_point;
    int32_t vacc2x0 = vksum0 * vinput_zero_point2;
    int32_t vacc2x1 = vksum1 * vinput_zero_point2;
    int32_t vacc2x2 = vksum2 * vinput_zero_point2;
    int32_t vacc2x3 = vksum3 * vinput_zero_point2;
    const int32_t vinput_zero_point3 = quantization_params[3].zero_point;
    int32_t vacc3x0 = vksum0 * vinput_zero_point3;
    int32_t vacc3x1 = vksum1 * vinput_zero_point3;
    int32_t vacc3x2 = vksum2 * vinput_zero_point3;
    int32_t vacc3x3 = vksum3 * vinput_zero_point3;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;
      const int32_t va2 = (int32_t) *a2++;
      const int32_t va3 = (int32_t) *a3++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;
      vacc1x2 += va1 * vb2;
      vacc1x3 += va1 * vb3;
      vacc2x0 += va2 * vb0;
      vacc2x1 += va2 * vb1;
      vacc2x2 += va2 * vb2;
      vacc2x3 += va2 * vb3;
      vacc3x0 += va3 * vb0;
      vacc3x1 += va3 * vb1;
      vacc3x2 += va3 * vb2;
      vacc3x3 += va3 * vb3;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    float vout0x2 = (float) vacc0x2;
    float vout0x3 = (float) vacc0x3;
    float vout1x0 = (float) vacc1x0;
    float vout1x1 = (float) vacc1x1;
    float vout1x2 = (float) vacc1x2;
    float vout1x3 = (float) vacc1x3;
    float vout2x0 = (float) vacc2x0;
    float vout2x1 = (float) vacc2x1;
    float vout2x2 = (float) vacc2x2;
    float vout2x3 = (float) vacc2x3;
    float vout3x0 = (float) vacc3x0;
    float vout3x1 = (float) vacc3x1;
    float vout3x2 = (float) vacc3x2;
    float vout3x3 = (float) vacc3x3;

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;
    vout0x2 *= vinput_scale0;
    vout0x3 *= vinput_scale0;
    const float vinput_scale1 = quantization_params[1].inv_scale;
    vout1x0 *= vinput_scale1;
    vout1x1 *= vinput_scale1;
    vout1x2 *= vinput_scale1;
    vout1x3 *= vinput_scale1;
    const float vinput_scale2 = quantization_params[2].inv_scale;
    vout2x0 *= vinput_scale2;
    vout2x1 *= vinput_scale2;
    vout2x2 *= vinput_scale2;
    vout2x3 *= vinput_scale2;
    const float vinput_scale3 = quantization_params[3].inv_scale;
    vout3x0 *= vinput_scale3;
    vout3x1 *= vinput_scale3;
    vout3x2 *= vinput_scale3;
    vout3x3 *= vinput_scale3;

    const float vfilter_output_scale0 = ((const float*) w)[0];
    vout0x0 *= vfilter_output_scale0;
    vout1x0 *= vfilter_output_scale0;
    vout2x0 *= vfilter_output_scale0;
    vout3x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = ((const float*) w)[1];
    vout0x1 *= vfilter_output_scale1;
    vout1x1 *= vfilter_output_scale1;
    vout2x1 *= vfilter_output_scale1;
    vout3x1 *= vfilter_output_scale1;
    const float vfilter_output_scale2 = ((const float*) w)[2];
    vout0x2 *= vfilter_output_scale2;
    vout1x2 *= vfilter_output_scale2;
    vout2x2 *= vfilter_output_scale2;
    vout3x2 *= vfilter_output_scale2;
    const float vfilter_output_scale3 = ((const float*) w)[3];
    vout0x3 *= vfilter_output_scale3;
    vout1x3 *= vfilter_output_scale3;
    vout2x3 *= vfilter_output_scale3;
    vout3x3 *= vfilter_output_scale3;

    const float vbias0 = ((const float*) w)[4];
    vout0x0 += vbias0;
    vout1x0 += vbias0;
    vout2x0 += vbias0;
    vout3x0 += vbias0;
    const float vbias1 = ((const float*) w)[5];
    vout0x1 += vbias1;
    vout1x1 += vbias1;
    vout2x1 += vbias1;
    vout3x1 += vbias1;
    const float vbias2 = ((const float*) w)[6];
    vout0x2 += vbias2;
    vout1x2 += vbias2;
    vout2x2 += vbias2;
    vout3x2 += vbias2;
    const float vbias3 = ((const float*) w)[7];
    vout0x3 += vbias3;
    vout1x3 += vbias3;
    vout2x3 += vbias3;
    vout3x3 += vbias3;

    w = (const float*) w + 8;

    const float voutput_min = params->scalar.min;
    vout0x0 = __builtin_wasm_max_f32(vout0x0, voutput_min);
    vout1x0 = __builtin_wasm_max_f32(vout1x0, voutput_min);
    vout2x0 = __builtin_wasm_max_f32(vout2x0, voutput_min);
    vout3x0 = __builtin_wasm_max_f32(vout3x0, voutput_min);
    vout0x1 = __builtin_wasm_max_f32(vout0x1, voutput_min);
    vout1x1 = __builtin_wasm_max_f32(vout1x1, voutput_min);
    vout2x1 = __builtin_wasm_max_f32(vout2x1, voutput_min);
    vout3x1 = __builtin_wasm_max_f32(vout3x1, voutput_min);
    vout0x2 = __builtin_wasm_max_f32(vout0x2, voutput_min);
    vout1x2 = __builtin_wasm_max_f32(vout1x2, voutput_min);
    vout2x2 = __builtin_wasm_max_f32(vout2x2, voutput_min);
    vout3x2 = __builtin_wasm_max_f32(vout3x2, voutput_min);
    vout0x3 = __builtin_wasm_max_f32(vout0x3, voutput_min);
    vout1x3 = __builtin_wasm_max_f32(vout1x3, voutput_min);
    vout2x3 = __builtin_wasm_max_f32(vout2x3, voutput_min);
    vout3x3 = __builtin_wasm_max_f32(vout3x3, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = __builtin_wasm_min_f32(vout0x0, voutput_max);
    vout1x0 = __builtin_wasm_min_f32(vout1x0, voutput_max);
    vout2x0 = __builtin_wasm_min_f32(vout2x0, voutput_max);
    vout3x0 = __builtin_wasm_min_f32(vout3x0, voutput_max);
    vout0x1 = __builtin_wasm_min_f32(vout0x1, voutput_max);
    vout1x1 = __builtin_wasm_min_f32(vout1x1, voutput_max);
    vout2x1 = __builtin_wasm_min_f32(vout2x1, voutput_max);
    vout3x1 = __builtin_wasm_min_f32(vout3x1, voutput_max);
    vout0x2 = __builtin_wasm_min_f32(vout0x2, voutput_max);
    vout1x2 = __builtin_wasm_min_f32(vout1x2, voutput_max);
    vout2x2 = __builtin_wasm_min_f32(vout2x2, voutput_max);
    vout3x2 = __builtin_wasm_min_f32(vout3x2, voutput_max);
    vout0x3 = __builtin_wasm_min_f32(vout0x3, voutput_max);
    vout1x3 = __builtin_wasm_min_f32(vout1x3, voutput_max);
    vout2x3 = __builtin_wasm_min_f32(vout2x3, voutput_max);
    vout3x3 = __builtin_wasm_min_f32(vout3x3, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;
      c1[0] = vout1x0;
      c1[1] = vout1x1;
      c1[2] = vout1x2;
      c1[3] = vout1x3;
      c2[0] = vout2x0;
      c2[1] = vout2x1;
      c2[2] = vout2x2;
      c2[3] = vout2x3;
      c3[0] = vout3x0;
      c3[1] = vout3x1;
      c3[2] = vout3x2;
      c3[3] = vout3x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = vout1x0;
        c1[1] = vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = vout2x0;
        c2[1] = vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c3[0] = vout3x0;
        c3[1] = vout3x1;
        vout3x0 = vout3x2;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = vout0x0;
        c1[0] = vout1x0;
        c2[0] = vout2x0;
        c3[0] = vout3x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    const int32_t vksum0 = ((const int32_t*) w)[0];
    const int32_t vksum1 = ((const int32_t*) w)[1];
    const int32_t vksum2 = ((const int32_t*) w)[2];
    const int32_t vksum3 = ((const int32_t*) w)[3];
    const int32_t vinput_zero_point = quantization_params->zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point;
    int32_t vacc0x1 = vksum1 * vinput_zero_point;
    int32_t vacc0x2 = vksum2 * vinput_zero_point;
    int32_t vacc0x3 = vksum3 * vinput_zero_point;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
        w = (const void*) ((const int8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    float vout0x2 = (float) vacc0x2;
    float vout0x3 = (float) vacc0x3;

    const float vinput_scale = quantization_params->inv_scale;
    vout0x0 *= vinput_scale;
    vout0x1 *= vinput_scale;
    vout0x2 *= vinput_scale;
    vout0x3 *= vinput_scale;

    const float vfilter_output_scale0 = ((const float*) w)[0];
    vout0x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = ((const float*) w)[1];
    vout0x1 *= vfilter_output_scale1;
    const float vfilter_output_scale2 = ((const float*) w)[2];
    vout0x2 *= vfilter_output_scale2;
    const float vfilter_output_scale3 = ((const float*) w)[3];
    vout0x3 *= vfilter_output_scale3;

    const float vbias0 = ((const float*) w)[4];
    vout0x0 += vbias0;
    const float vbias1 = ((const float*) w)[5];
    vout0x1 += vbias1;
    const float vbias2 = ((const float*) w)[6];
    vout0x2 += vbias2;
    const float vbias3 = ((const float*) w)[7];
    vout0x3 += vbias3;

    w = (const float*) w + 8;

    const float voutput_min = params->scalar.min;
    vout0x0 = __builtin_wasm_max_f32(vout0x0, voutput_min);
    vout0x1 = __builtin_wasm_max_f32(vout0x1, voutput_min);
    vout0x2 = __builtin_wasm_max_f32(vout0x2, voutput_min);
    vout0x3 = __builtin_wasm_max_f32(vout0x3, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = __builtin_wasm_min_f32(vout0x0, voutput_max);
    vout0x1 = __builtin_wasm_min_f32(vout0x1, voutput_max);
    vout0x2 = __builtin_wasm_min_f32(vout0x2, voutput_max);
    vout0x3 = __builtin_wasm_min_f32(vout0x3, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    const int32_t vksum0 = ((const int32_t*) w)[0];
    const int32_t vksum1 = ((const int32_t*) w)[1];
    const int32_t vksum2 = ((const int32_t*) w)[2];
    const int32_t vksum3 = ((const int32_t*) w)[3];
    const int32_t vinput_zero_point = quantization_params->zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point;
    int32_t vacc0x1 = vksum1 * vinput_zero_point;
    int32_t vacc0x2 = vksum2 * vinput_zero_point;
    int32_t vacc0x3 = vksum3 * vinput_zero_point;
    int32_t vacc1x0 = vksum0 * vinput_zero_point;
    int32_t vacc1x1 = vksum1 * vinput_zero_point;
    int32_t vacc1x2 = vksum2 * vinput_zero_point;
    int32_t vacc1x3 = vksum3 * vinput_zero_point;
    int32_t vacc2x0 = vksum0 * vinput_zero_point;
    int32_t vacc2x1 = vksum1 * vinput_zero_point;
    int32_t vacc2x2 = vksum2 * vinput_zero_point;
    int32_t vacc2x3 = vksum3 * vinput_zero_point;
    int32_t vacc3x0 = vksum0 * vinput_zero_point;
    int32_t vacc3x1 = vksum1 * vinput_zero_point;
    int32_t vacc3x2 = vksum2 * vinput_zero_point;
    int32_t vacc3x3 = vksum3 * vinput_zero_point;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      a += 4;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;
        const int32_t va1 = (int32_t) *a1++;
        const int32_t va2 = (int32_t) *a2++;
        const int32_t va3 = (int32_t) *a3++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
        w = (const void*) ((const int8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;
        vacc1x0 += va1 * vb0;
        vacc1x1 += va1 * vb1;
        vacc1x2 += va1 * vb2;
        vacc1x3 += va1 * vb3;
        vacc2x0 += va2 * vb0;
        vacc2x1 += va2 * vb1;
        vacc2x2 += va2 * vb2;
        vacc2x3 += va2 * vb3;
        vacc3x0 += va3 * vb0;
        vacc3x1 += va3 * vb1;
        vacc3x2 += va3 * vb2;
        vacc3x3 += va3 * vb3;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    float vout0x2 = (float) vacc0x2;
    float vout0x3 = (float) vacc0x3;
    float vout1x0 = (float) vacc1x0;
    float vout1x1 = (float) vacc1x1;
    float vout1x2 = (float) vacc1x2;
    float vout1x3 = (float) vacc1x3;
    float vout2x0 = (float) vacc2x0;
    float vout2x1 = (float) vacc2x1;
    float vout2x2 = (float) vacc2x2;
    float vout2x3 = (float) vacc2x3;
    float vout3x0 = (float) vacc3x0;
    float vout3x1 = (float) vacc3x1;
    float vout3x2 = (float) vacc3x2;
    float vout3x3 = (float) vacc3x3;

    const float vinput_scale = quantization_params->inv_scale;
    vout0x0 *= vinput_scale;
    vout0x1 *= vinput_scale;
    vout0x2 *= vinput_scale;
    vout0x3 *= vinput_scale;
    vout1x0 *= vinput_scale;
    vout1x1 *= vinput_scale;
    vout1x2 *= vinput_scale;
    vout1x3 *= vinput_scale;
    vout2x0 *= vinput_scale;
    vout2x1 *= vinput_scale;
    vout2x2 *= vinput_scale;
    vout2x3 *= vinput_scale;
    vout3x0 *= vinput_scale;
    vout3x1 *= vinput_scale;
    vout3x2 *= vinput_scale;
    vout3x3 *= vinput_scale;

    const float vfilter_output_scale0 = ((const float*) w)[0];
    vout0x0 *= vfilter_output_scale0;
    vout1x0 *= vfilter_output_scale0;
    vout2x0 *= vfilter_output_scale0;
    vout3x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = ((const float*) w)[1];
    vout0x1 *= vfilter_output_scale1;
    vout1x1 *= vfilter_output_scale1;
    vout2x1 *= vfilter_output_scale1;
    vout3x1 *= vfilter_output_scale1;
    const float vfilter_output_scale2 = ((const float*) w)[2];
    vout0x2 *= vfilter_output_scale2;
    vout1x2 *= vfilter_output_scale2;
    vout2x2 *= vfilter_output_scale2;
    vout3x2 *= vfilter_output_scale2;
    const float vfilter_output_scale3 = ((const float*) w)[3];
    vout0x3 *= vfilter_output_scale3;
    vout1x3 *= vfilter_output_scale3;
    vout2x3 *= vfilter_output_scale3;
    vout3x3 *= vfilter_output_scale3;

    const float vbias0 = ((const float*) w)[4];
    vout0x0 += vbias0;
    vout1x0 += vbias0;
    vout2x0 += vbias0;
    vout3x0 += vbias0;
    const float vbias1 = ((const float*) w)[5];
    vout0x1 += vbias1;
    vout1x1 += vbias1;
    vout2x1 += vbias1;
    vout3x1 += vbias1;
    const float vbias2 = ((const float*) w)[6];
    vout0x2 += vbias2;
    vout1x2 += vbias2;
    vout2x2 += vbias2;
    vout3x2 += vbias2;
    const float vbias3 = ((const float*) w)[7];
    vout0x3 += vbias3;
    vout1x3 += vbias3;
    vout2x3 += vbias3;
    vout3x3 += vbias3;

    w = (const float*) w + 8;

    const float voutput_min = params->scalar.min;
    vout0x0 = __builtin_wasm_max_f32(vout0x0, voutput_min);
    vout1x0 = __builtin_wasm_max_f32(vout1x0, voutput_min);
    vout2x0 = __builtin_wasm_max_f32(vout2x0, voutput_min);
    vout3x0 = __builtin_wasm_max_f32(vout3x0, voutput_min);
    vout0x1 = __builtin_wasm_max_f32(vout0x1, voutput_min);
    vout1x1 = __builtin_wasm_max_f32(vout1x1, voutput_min);
    vout2x1 = __builtin_wasm_max_f32(vout2x1, voutput_min);
    vout3x1 = __builtin_wasm_max_f32(vout3x1, voutput_min);
    vout0x2 = __builtin_wasm_max_f32(vout0x2, voutput_min);
    vout1x2 = __builtin_wasm_max_f32(vout1x2, voutput_min);
    vout2x2 = __builtin_wasm_max_f32(vout2x2, voutput_min);
    vout3x2 = __builtin_wasm_max_f32(vout3x2, voutput_min);
    vout0x3 = __builtin_wasm_max_f32(vout0x3, voutput_min);
    vout1x3 = __builtin_wasm_max_f32(vout1x3, voutput_min);
    vout2x3 = __builtin_wasm_max_f32(vout2x3, voutput_min);
    vout3x3 = __builtin_wasm_max_f32(vout3x3, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = __builtin_wasm_min_f32(vout0x0, voutput_max);
    vout1x0 = __builtin_wasm_min_f32(vout1x0, voutput_max);
    vout2x0 = __builtin_wasm_min_f32(vout2x0, voutput_max);
    vout3x0 = __builtin_wasm_min_f32(vout3x0, voutput_max);
    vout0x1 = __builtin_wasm_min_f32(vout0x1, voutput_max);
    vout1x1 = __builtin_wasm_min_f32(vout1x1, voutput_max);
    vout2x1 = __builtin_wasm_min_f32(vout2x1, voutput_max);
    vout3x1 = __builtin_wasm_min_f32(vout3x1, voutput_max);
    vout0x2 = __builtin_wasm_min_f32(vout0x2, voutput_max);
    vout1x2 = __builtin_wasm_min_f32(vout1x2, voutput_max);
    vout2x2 = __builtin_wasm_min_f32(vout2x2, voutput_max);
    vout3x2 = __builtin_wasm_min_f32(vout3x2, voutput_max);
    vout0x3 = __builtin_wasm_min_f32(vout0x3, voutput_max);
    vout1x3 = __builtin_wasm_min_f32(vout1x3, voutput_max);
    vout2x3 = __builtin_wasm_min_f32(vout2x3, voutput_max);
    vout3x3 = __builtin_wasm_min_f32(vout3x3, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      c3[0] = vout3x0;
      c3[1] = vout3x1;
      c3[2] = vout3x2;
      c3[3] = vout3x3;
      c2[0] = vout2x0;
      c2[1] = vout2x1;
      c2[2] = vout2x2;
      c2[3] = vout2x3;
      c1[0] = vout1x0;
      c1[1] = vout1x1;
      c1[2] = vout1x2;
      c1[3] = vout1x3;
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = vout3x0;
        c3[1] = vout3x1;
        vout3x0 = vout3x2;
        c3 += 2;
        c2[0] = vout2x0;
        c2[1] = vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c1[0] = vout1x0;
        c1[1] = vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = vout3x0;
        c2[0] = vout2x0;
        c1[0] = vout1x0;
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_25p2c__wasm_fmagic(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vscale = params->fp32_scalar_fmagic.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) i3[0];
      const int32_t vi3x1 = (int32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7];

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) i4[0];
      const int32_t vi4x1 = (int32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9];

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) i5[0];
      const int32_t vi5x1 = (int32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      const int32_t vk5x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11];

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) i6[0];
      const int32_t vi6x1 = (int32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      const int32_t vk6x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13];

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) i7[0];
      const int32_t vi7x1 = (int32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      const int32_t vk7x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15];

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) i8[0];
      const int32_t vi8x1 = (int32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      const int32_t vk8x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17];

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      const int32_t vi9x0 = (int32_t) i9[0];
      const int32_t vi9x1 = (int32_t) i9[1];
      i9 += 2;

      const int32_t vk9x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18];
      const int32_t vk9x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[19];

      vacc0 += vi9x0 * vk9x0;
      vacc1 += vi9x1 * vk9x1;

      const int32_t vi10x0 = (int32_t) i10[0];
      const int32_t vi10x1 = (int32_t) i10[1];
      i10 += 2;

      const int32_t vk10x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20];
      const int32_t vk10x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[21];

      vacc0 += vi10x0 * vk10x0;
      vacc1 += vi10x1 * vk10x1;

      const int32_t vi11x0 = (int32_t) i11[0];
      const int32_t vi11x1 = (int32_t) i11[1];
      i11 += 2;

      const int32_t vk11x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22];
      const int32_t vk11x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[23];

      vacc0 += vi11x0 * vk11x0;
      vacc1 += vi11x1 * vk11x1;

      const int32_t vi12x0 = (int32_t) i12[0];
      const int32_t vi12x1 = (int32_t) i12[1];
      i12 += 2;

      const int32_t vk12x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24];
      const int32_t vk12x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[25];

      vacc0 += vi12x0 * vk12x0;
      vacc1 += vi12x1 * vk12x1;

      const int32_t vi13x0 = (int32_t) i13[0];
      const int32_t vi13x1 = (int32_t) i13[1];
      i13 += 2;

      const int32_t vk13x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26];
      const int32_t vk13x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[27];

      vacc0 += vi13x0 * vk13x0;
      vacc1 += vi13x1 * vk13x1;

      const int32_t vi14x0 = (int32_t) i14[0];
      const int32_t vi14x1 = (int32_t) i14[1];
      i14 += 2;

      const int32_t vk14x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28];
      const int32_t vk14x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[29];

      vacc0 += vi14x0 * vk14x0;
      vacc1 += vi14x1 * vk14x1;

      const int32_t vi15x0 = (int32_t) i15[0];
      const int32_t vi15x1 = (int32_t) i15[1];
      i15 += 2;

      const int32_t vk15x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30];
      const int32_t vk15x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[31];

      vacc0 += vi15x0 * vk15x0;
      vacc1 += vi15x1 * vk15x1;

      const int32_t vi16x0 = (int32_t) i16[0];
      const int32_t vi16x1 = (int32_t) i16[1];
      i16 += 2;

      const int32_t vk16x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32];
      const int32_t vk16x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[33];

      vacc0 += vi16x0 * vk16x0;
      vacc1 += vi16x1 * vk16x1;

      const int32_t vi17x0 = (int32_t) i17[0];
      const int32_t vi17x1 = (int32_t) i17[1];
      i17 += 2;

      const int32_t vk17x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34];
      const int32_t vk17x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[35];

      vacc0 += vi17x0 * vk17x0;
      vacc1 += vi17x1 * vk17x1;

      const int32_t vi18x0 = (int32_t) i18[0];
      const int32_t vi18x1 = (int32_t) i18[1];
      i18 += 2;

      const int32_t vk18x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36];
      const int32_t vk18x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[37];

      vacc0 += vi18x0 * vk18x0;
      vacc1 += vi18x1 * vk18x1;

      const int32_t vi19x0 = (int32_t) i19[0];
      const int32_t vi19x1 = (int32_t) i19[1];
      i19 += 2;

      const int32_t vk19x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38];
      const int32_t vk19x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[39];

      vacc0 += vi19x0 * vk19x0;
      vacc1 += vi19x1 * vk19x1;

      const int32_t vi20x0 = (int32_t) i20[0];
      const int32_t vi20x1 = (int32_t) i20[1];
      i20 += 2;

      const int32_t vk20x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40];
      const int32_t vk20x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[41];

      vacc0 += vi20x0 * vk20x0;
      vacc1 += vi20x1 * vk20x1;

      const int32_t vi21x0 = (int32_t) i21[0];
      const int32_t vi21x1 = (int32_t) i21[1];
      i21 += 2;

      const int32_t vk21x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42];
      const int32_t vk21x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[43];

      vacc0 += vi21x0 * vk21x0;
      vacc1 += vi21x1 * vk21x1;

      const int32_t vi22x0 = (int32_t) i22[0];
      const int32_t vi22x1 = (int32_t) i22[1];
      i22 += 2;

      const int32_t vk22x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44];
      const int32_t vk22x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[45];

      vacc0 += vi22x0 * vk22x0;
      vacc1 += vi22x1 * vk22x1;

      const int32_t vi23x0 = (int32_t) i23[0];
      const int32_t vi23x1 = (int32_t) i23[1];
      i23 += 2;

      const int32_t vk23x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46];
      const int32_t vk23x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[47];

      vacc0 += vi23x0 * vk23x0;
      vacc1 += vi23x1 * vk23x1;

      const int32_t vi24x0 = (int32_t) i24[0];
      const int32_t vi24x1 = (int32_t) i24[1];
      i24 += 2;

      const int32_t vk24x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48];
      const int32_t vk24x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[49];

      vacc0 += vi24x0 * vk24x0;
      vacc1 += vi24x1 * vk24x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 50 * sizeof(int8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      vfpacc0 *= vscale;
      vfpacc1 *= vscale;

      vfpacc0 = __builtin_wasm_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = __builtin_wasm_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = __builtin_wasm_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = __builtin_wasm_min_f32(vfpacc1, voutput_max_less_zero_point);

      vfpacc0 += vmagic_bias;
      vfpacc1 += vmagic_bias;

      int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
      int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) *i0;
      const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) *i1;
      const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) *i2;
      const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) *i3;
      const int32_t vk3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) *i4;
      const int32_t vk4 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) *i5;
      const int32_t vk5 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) *i6;
      const int32_t vk6 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) *i7;
      const int32_t vk7 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) *i8;
      const int32_t vk8 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      vacc += vi8 * vk8;
      const int32_t vi9 = (int32_t) *i9;
      const int32_t vk9 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18];
      vacc += vi9 * vk9;
      const int32_t vi10 = (int32_t) *i10;
      const int32_t vk10 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20];
      vacc += vi10 * vk10;
      const int32_t vi11 = (int32_t) *i11;
      const int32_t vk11 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22];
      vacc += vi11 * vk11;
      const int32_t vi12 = (int32_t) *i12;
      const int32_t vk12 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24];
      vacc += vi12 * vk12;
      const int32_t vi13 = (int32_t) *i13;
      const int32_t vk13 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26];
      vacc += vi13 * vk13;
      const int32_t vi14 = (int32_t) *i14;
      const int32_t vk14 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28];
      vacc += vi14 * vk14;
      const int32_t vi15 = (int32_t) *i15;
      const int32_t vk15 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30];
      vacc += vi15 * vk15;
      const int32_t vi16 = (int32_t) *i16;
      const int32_t vk16 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32];
      vacc += vi16 * vk16;
      const int32_t vi17 = (int32_t) *i17;
      const int32_t vk17 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34];
      vacc += vi17 * vk17;
      const int32_t vi18 = (int32_t) *i18;
      const int32_t vk18 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36];
      vacc += vi18 * vk18;
      const int32_t vi19 = (int32_t) *i19;
      const int32_t vk19 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38];
      vacc += vi19 * vk19;
      const int32_t vi20 = (int32_t) *i20;
      const int32_t vk20 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40];
      vacc += vi20 * vk20;
      const int32_t vi21 = (int32_t) *i21;
      const int32_t vk21 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42];
      vacc += vi21 * vk21;
      const int32_t vi22 = (int32_t) *i22;
      const int32_t vk22 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44];
      vacc += vi22 * vk22;
      const int32_t vi23 = (int32_t) *i23;
      const int32_t vk23 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46];
      vacc += vi23 * vk23;
      const int32_t vi24 = (int32_t) *i24;
      const int32_t vk24 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48];
      vacc += vi24 * vk24;

      float vfpacc = (float) vacc * vscale;

      vfpacc = __builtin_wasm_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = __builtin_wasm_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;

      *output++ = (int8_t) vout;
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_9p2c__wasm_fmagic(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vscale = params->fp32_scalar_fmagic.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) i3[0];
      const int32_t vi3x1 = (int32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7];

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) i4[0];
      const int32_t vi4x1 = (int32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9];

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) i5[0];
      const int32_t vi5x1 = (int32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      const int32_t vk5x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11];

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) i6[0];
      const int32_t vi6x1 = (int32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      const int32_t vk6x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13];

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) i7[0];
      const int32_t vi7x1 = (int32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      const int32_t vk7x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15];

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) i8[0];
      const int32_t vi8x1 = (int32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      const int32_t vk8x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17];

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 18 * sizeof(int8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      vfpacc0 *= vscale;
      vfpacc1 *= vscale;

      vfpacc0 = __builtin_wasm_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = __builtin_wasm_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = __builtin_wasm_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = __builtin_wasm_min_f32(vfpacc1, voutput_max_less_zero_point);

      vfpacc0 += vmagic_bias;
      vfpacc1 += vmagic_bias;

      int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
      int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) *i0;
      const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) *i1;
      const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) *i2;
      const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) *i3;
      const int32_t vk3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) *i4;
      const int32_t vk4 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) *i5;
      const int32_t vk5 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) *i6;
      const int32_t vk6 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) *i7;
      const int32_t vk7 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) *i8;
      const int32_t vk8 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      vacc += vi8 * vk8;

      float vfpacc = (float) vacc * vscale;

      vfpacc = __builtin_wasm_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = __builtin_wasm_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;

      *output++ = (int8_t) vout;
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p2c__wasm_fmagic(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) i3[0];
      const int32_t vi3x1 = (int32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7];

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) i4[0];
      const int32_t vi4x1 = (int32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9];

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) i5[0];
      const int32_t vi5x1 = (int32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      const int32_t vk5x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11];

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) i6[0];
      const int32_t vi6x1 = (int32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      const int32_t vk6x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13];

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) i7[0];
      const int32_t vi7x1 = (int32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      const int32_t vk7x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15];

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) i8[0];
      const int32_t vi8x1 = (int32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      const int32_t vk8x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17];

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      const int32_t vi9x0 = (int32_t) i9[0];
      const int32_t vi9x1 = (int32_t) i9[1];
      i9 += 2;

      const int32_t vk9x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18];
      const int32_t vk9x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[19];

      vacc0 += vi9x0 * vk9x0;
      vacc1 += vi9x1 * vk9x1;

      const int32_t vi10x0 = (int32_t) i10[0];
      const int32_t vi10x1 = (int32_t) i10[1];
      i10 += 2;

      const int32_t vk10x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20];
      const int32_t vk10x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[21];

      vacc0 += vi10x0 * vk10x0;
      vacc1 += vi10x1 * vk10x1;

      const int32_t vi11x0 = (int32_t) i11[0];
      const int32_t vi11x1 = (int32_t) i11[1];
      i11 += 2;

      const int32_t vk11x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22];
      const int32_t vk11x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[23];

      vacc0 += vi11x0 * vk11x0;
      vacc1 += vi11x1 * vk11x1;

      const int32_t vi12x0 = (int32_t) i12[0];
      const int32_t vi12x1 = (int32_t) i12[1];
      i12 += 2;

      const int32_t vk12x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24];
      const int32_t vk12x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[25];

      vacc0 += vi12x0 * vk12x0;
      vacc1 += vi12x1 * vk12x1;

      const int32_t vi13x0 = (int32_t) i13[0];
      const int32_t vi13x1 = (int32_t) i13[1];
      i13 += 2;

      const int32_t vk13x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26];
      const int32_t vk13x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[27];

      vacc0 += vi13x0 * vk13x0;
      vacc1 += vi13x1 * vk13x1;

      const int32_t vi14x0 = (int32_t) i14[0];
      const int32_t vi14x1 = (int32_t) i14[1];
      i14 += 2;

      const int32_t vk14x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28];
      const int32_t vk14x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[29];

      vacc0 += vi14x0 * vk14x0;
      vacc1 += vi14x1 * vk14x1;

      const int32_t vi15x0 = (int32_t) i15[0];
      const int32_t vi15x1 = (int32_t) i15[1];
      i15 += 2;

      const int32_t vk15x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30];
      const int32_t vk15x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[31];

      vacc0 += vi15x0 * vk15x0;
      vacc1 += vi15x1 * vk15x1;

      const int32_t vi16x0 = (int32_t) i16[0];
      const int32_t vi16x1 = (int32_t) i16[1];
      i16 += 2;

      const int32_t vk16x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32];
      const int32_t vk16x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[33];

      vacc0 += vi16x0 * vk16x0;
      vacc1 += vi16x1 * vk16x1;

      const int32_t vi17x0 = (int32_t) i17[0];
      const int32_t vi17x1 = (int32_t) i17[1];
      i17 += 2;

      const int32_t vk17x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34];
      const int32_t vk17x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[35];

      vacc0 += vi17x0 * vk17x0;
      vacc1 += vi17x1 * vk17x1;

      const int32_t vi18x0 = (int32_t) i18[0];
      const int32_t vi18x1 = (int32_t) i18[1];
      i18 += 2;

      const int32_t vk18x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36];
      const int32_t vk18x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[37];

      vacc0 += vi18x0 * vk18x0;
      vacc1 += vi18x1 * vk18x1;

      const int32_t vi19x0 = (int32_t) i19[0];
      const int32_t vi19x1 = (int32_t) i19[1];
      i19 += 2;

      const int32_t vk19x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38];
      const int32_t vk19x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[39];

      vacc0 += vi19x0 * vk19x0;
      vacc1 += vi19x1 * vk19x1;

      const int32_t vi20x0 = (int32_t) i20[0];
      const int32_t vi20x1 = (int32_t) i20[1];
      i20 += 2;

      const int32_t vk20x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40];
      const int32_t vk20x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[41];

      vacc0 += vi20x0 * vk20x0;
      vacc1 += vi20x1 * vk20x1;

      const int32_t vi21x0 = (int32_t) i21[0];
      const int32_t vi21x1 = (int32_t) i21[1];
      i21 += 2;

      const int32_t vk21x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42];
      const int32_t vk21x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[43];

      vacc0 += vi21x0 * vk21x0;
      vacc1 += vi21x1 * vk21x1;

      const int32_t vi22x0 = (int32_t) i22[0];
      const int32_t vi22x1 = (int32_t) i22[1];
      i22 += 2;

      const int32_t vk22x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44];
      const int32_t vk22x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[45];

      vacc0 += vi22x0 * vk22x0;
      vacc1 += vi22x1 * vk22x1;

      const int32_t vi23x0 = (int32_t) i23[0];
      const int32_t vi23x1 = (int32_t) i23[1];
      i23 += 2;

      const int32_t vk23x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46];
      const int32_t vk23x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[47];

      vacc0 += vi23x0 * vk23x0;
      vacc1 += vi23x1 * vk23x1;

      const int32_t vi24x0 = (int32_t) i24[0];
      const int32_t vi24x1 = (int32_t) i24[1];
      i24 += 2;

      const int32_t vk24x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48];
      const int32_t vk24x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[49];

      vacc0 += vi24x0 * vk24x0;
      vacc1 += vi24x1 * vk24x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 50 * sizeof(int8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      const float vscale0 = unaligned_indexed_load_f32(w, 0);
      const float vscale1 = unaligned_indexed_load_f32(w, 1);
      w = (const void*) ((const float*) w + 2);

      vfpacc0 *= vscale0;
      vfpacc1 *= vscale1;

      vfpacc0 = __builtin_wasm_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = __builtin_wasm_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = __builtin_wasm_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = __builtin_wasm_min_f32(vfpacc1, voutput_max_less_zero_point);

      vfpacc0 += vmagic_bias;
      vfpacc1 += vmagic_bias;

      int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
      int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) *i0;
      const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) *i1;
      const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) *i2;
      const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) *i3;
      const int32_t vk3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) *i4;
      const int32_t vk4 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) *i5;
      const int32_t vk5 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) *i6;
      const int32_t vk6 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) *i7;
      const int32_t vk7 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) *i8;
      const int32_t vk8 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      vacc += vi8 * vk8;
      const int32_t vi9 = (int32_t) *i9;
      const int32_t vk9 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18];
      vacc += vi9 * vk9;
      const int32_t vi10 = (int32_t) *i10;
      const int32_t vk10 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20];
      vacc += vi10 * vk10;
      const int32_t vi11 = (int32_t) *i11;
      const int32_t vk11 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22];
      vacc += vi11 * vk11;
      const int32_t vi12 = (int32_t) *i12;
      const int32_t vk12 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24];
      vacc += vi12 * vk12;
      const int32_t vi13 = (int32_t) *i13;
      const int32_t vk13 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26];
      vacc += vi13 * vk13;
      const int32_t vi14 = (int32_t) *i14;
      const int32_t vk14 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28];
      vacc += vi14 * vk14;
      const int32_t vi15 = (int32_t) *i15;
      const int32_t vk15 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30];
      vacc += vi15 * vk15;
      const int32_t vi16 = (int32_t) *i16;
      const int32_t vk16 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32];
      vacc += vi16 * vk16;
      const int32_t vi17 = (int32_t) *i17;
      const int32_t vk17 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34];
      vacc += vi17 * vk17;
      const int32_t vi18 = (int32_t) *i18;
      const int32_t vk18 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36];
      vacc += vi18 * vk18;
      const int32_t vi19 = (int32_t) *i19;
      const int32_t vk19 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38];
      vacc += vi19 * vk19;
      const int32_t vi20 = (int32_t) *i20;
      const int32_t vk20 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40];
      vacc += vi20 * vk20;
      const int32_t vi21 = (int32_t) *i21;
      const int32_t vk21 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42];
      vacc += vi21 * vk21;
      const int32_t vi22 = (int32_t) *i22;
      const int32_t vk22 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44];
      vacc += vi22 * vk22;
      const int32_t vi23 = (int32_t) *i23;
      const int32_t vk23 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46];
      vacc += vi23 * vk23;
      const int32_t vi24 = (int32_t) *i24;
      const int32_t vk24 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48];
      vacc += vi24 * vk24;

      const float vscale = unaligned_load_f32((const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 50 * sizeof(int8_t)));
      float vfpacc = (float) vacc * vscale;

      vfpacc = __builtin_wasm_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = __builtin_wasm_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;

      *output++ = (int8_t) vout;
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p2c__wasm_fmagic(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 6 * sizeof(int8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      const float vscale0 = unaligned_indexed_load_f32(w, 0);
      const float vscale1 = unaligned_indexed_load_f32(w, 1);
      w = (const void*) ((const float*) w + 2);

      vfpacc0 *= vscale0;
      vfpacc1 *= vscale1;

      vfpacc0 = __builtin_wasm_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = __builtin_wasm_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = __builtin_wasm_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = __builtin_wasm_min_f32(vfpacc1, voutput_max_less_zero_point);

      vfpacc0 += vmagic_bias;
      vfpacc1 += vmagic_bias;

      int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
      int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) *i0;
      const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) *i1;
      const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) *i2;
      const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      vacc += vi2 * vk2;

      const float vscale = unaligned_load_f32((const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 6 * sizeof(int8_t)));
      float vfpacc = (float) vacc * vscale;

      vfpacc = __builtin_wasm_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = __builtin_wasm_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;

      *output++ = (int8_t) vout;
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p2c__wasm_fmagic(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) i0[0];
      const int32_t vi0x1 = (int32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) i1[0];
      const int32_t vi1x1 = (int32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) i2[0];
      const int32_t vi2x1 = (int32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) i3[0];
      const int32_t vi3x1 = (int32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7];

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) i4[0];
      const int32_t vi4x1 = (int32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9];

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) i5[0];
      const int32_t vi5x1 = (int32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      const int32_t vk5x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11];

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) i6[0];
      const int32_t vi6x1 = (int32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      const int32_t vk6x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13];

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) i7[0];
      const int32_t vi7x1 = (int32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      const int32_t vk7x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15];

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) i8[0];
      const int32_t vi8x1 = (int32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      const int32_t vk8x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17];

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 18 * sizeof(int8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      const float vscale0 = unaligned_indexed_load_f32(w, 0);
      const float vscale1 = unaligned_indexed_load_f32(w, 1);
      w = (const void*) ((const float*) w + 2);

      vfpacc0 *= vscale0;
      vfpacc1 *= vscale1;

      vfpacc0 = __builtin_wasm_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = __builtin_wasm_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = __builtin_wasm_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = __builtin_wasm_min_f32(vfpacc1, voutput_max_less_zero_point);

      vfpacc0 += vmagic_bias;
      vfpacc1 += vmagic_bias;

      int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
      int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) *i0;
      const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) *i1;
      const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) *i2;
      const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) *i3;
      const int32_t vk3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) *i4;
      const int32_t vk4 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) *i5;
      const int32_t vk5 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) *i6;
      const int32_t vk6 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) *i7;
      const int32_t vk7 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) *i8;
      const int32_t vk8 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16];
      vacc += vi8 * vk8;

      const float vscale = unaligned_load_f32((const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 18 * sizeof(int8_t)));
      float vfpacc = (float) vacc * vscale;

      vfpacc = __builtin_wasm_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = __builtin_wasm_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;

      *output++ = (int8_t) vout;
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    const float vscale2 = ((const float*) w)[2];
    vfpacc0x2 *= vscale2;
    const float vscale3 = ((const float*) w)[3];
    vfpacc0x3 *= vscale3;
    w = (const void*) ((const float*) w + 4);

    const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
    vfpacc0x0 = __builtin_wasm_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = __builtin_wasm_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = __builtin_wasm_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = __builtin_wasm_max_f32(vfpacc0x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
    vfpacc0x0 = __builtin_wasm_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = __builtin_wasm_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = __builtin_wasm_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = __builtin_wasm_min_f32(vfpacc0x3, voutput_max_less_zero_point);

    const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc0x2 += vmagic_bias;
    vfpacc0x3 += vmagic_bias;

    const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0) - vmagic_bias_less_output_zero_point;
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1) - vmagic_bias_less_output_zero_point;
    int32_t vout0x2 = (int32_t) float_as_uint32(vfpacc0x2) - vmagic_bias_less_output_zero_point;
    int32_t vout0x3 = (int32_t) float_as_uint32(vfpacc0x3) - vmagic_bias_less_output_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    int32_t vacc3x0 = vacc0x0;
    int32_t vacc3x1 = vacc0x1;
    int32_t vacc3x2 = vacc0x2;
    int32_t vacc3x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;
      const int32_t va2 = (int32_t) *a2++;
      const int32_t va3 = (int32_t) *a3++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;
      vacc1x2 += va1 * vb2;
      vacc1x3 += va1 * vb3;
      vacc2x0 += va2 * vb0;
      vacc2x1 += va2 * vb1;
      vacc2x2 += va2 * vb2;
      vacc2x3 += va2 * vb3;
      vacc3x0 += va3 * vb0;
      vacc3x1 += va3 * vb1;
      vacc3x2 += va3 * vb2;
      vacc3x3 += va3 * vb3;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;
    float vfpacc3x0 = (float) vacc3x0;
    float vfpacc3x1 = (float) vacc3x1;
    float vfpacc3x2 = (float) vacc3x2;
    float vfpacc3x3 = (float) vacc3x3;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    vfpacc1x0 *= vscale0;
    vfpacc2x0 *= vscale0;
    vfpacc3x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    vfpacc1x1 *= vscale1;
    vfpacc2x1 *= vscale1;
    vfpacc3x1 *= vscale1;
    const float vscale2 = ((const float*) w)[2];
    vfpacc0x2 *= vscale2;
    vfpacc1x2 *= vscale2;
    vfpacc2x2 *= vscale2;
    vfpacc3x2 *= vscale2;
    const float vscale3 = ((const float*) w)[3];
    vfpacc0x3 *= vscale3;
    vfpacc1x3 *= vscale3;
    vfpacc2x3 *= vscale3;
    vfpacc3x3 *= vscale3;
    w = (const void*) ((const float*) w + 4);

    const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
    vfpacc0x0 = __builtin_wasm_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = __builtin_wasm_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = __builtin_wasm_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = __builtin_wasm_max_f32(vfpacc0x3, voutput_min_less_zero_point);
    vfpacc1x0 = __builtin_wasm_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = __builtin_wasm_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc1x2 = __builtin_wasm_max_f32(vfpacc1x2, voutput_min_less_zero_point);
    vfpacc1x3 = __builtin_wasm_max_f32(vfpacc1x3, voutput_min_less_zero_point);
    vfpacc2x0 = __builtin_wasm_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = __builtin_wasm_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc2x2 = __builtin_wasm_max_f32(vfpacc2x2, voutput_min_less_zero_point);
    vfpacc2x3 = __builtin_wasm_max_f32(vfpacc2x3, voutput_min_less_zero_point);
    vfpacc3x0 = __builtin_wasm_max_f32(vfpacc3x0, voutput_min_less_zero_point);
    vfpacc3x1 = __builtin_wasm_max_f32(vfpacc3x1, voutput_min_less_zero_point);
    vfpacc3x2 = __builtin_wasm_max_f32(vfpacc3x2, voutput_min_less_zero_point);
    vfpacc3x3 = __builtin_wasm_max_f32(vfpacc3x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
    vfpacc0x0 = __builtin_wasm_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = __builtin_wasm_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = __builtin_wasm_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = __builtin_wasm_min_f32(vfpacc0x3, voutput_max_less_zero_point);
    vfpacc1x0 = __builtin_wasm_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = __builtin_wasm_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc1x2 = __builtin_wasm_min_f32(vfpacc1x2, voutput_max_less_zero_point);
    vfpacc1x3 = __builtin_wasm_min_f32(vfpacc1x3, voutput_max_less_zero_point);
    vfpacc2x0 = __builtin_wasm_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = __builtin_wasm_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc2x2 = __builtin_wasm_min_f32(vfpacc2x2, voutput_max_less_zero_point);
    vfpacc2x3 = __builtin_wasm_min_f32(vfpacc2x3, voutput_max_less_zero_point);
    vfpacc3x0 = __builtin_wasm_min_f32(vfpacc3x0, voutput_max_less_zero_point);
    vfpacc3x1 = __builtin_wasm_min_f32(vfpacc3x1, voutput_max_less_zero_point);
    vfpacc3x2 = __builtin_wasm_min_f32(vfpacc3x2, voutput_max_less_zero_point);
    vfpacc3x3 = __builtin_wasm_min_f32(vfpacc3x3, voutput_max_less_zero_point);

    const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc0x2 += vmagic_bias;
    vfpacc0x3 += vmagic_bias;
    vfpacc1x0 += vmagic_bias;
    vfpacc1x1 += vmagic_bias;
    vfpacc1x2 += vmagic_bias;
    vfpacc1x3 += vmagic_bias;
    vfpacc2x0 += vmagic_bias;
    vfpacc2x1 += vmagic_bias;
    vfpacc2x2 += vmagic_bias;
    vfpacc2x3 += vmagic_bias;
    vfpacc3x0 += vmagic_bias;
    vfpacc3x1 += vmagic_bias;
    vfpacc3x2 += vmagic_bias;
    vfpacc3x3 += vmagic_bias;

    const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0) - vmagic_bias_less_output_zero_point;
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1) - vmagic_bias_less_output_zero_point;
    int32_t vout0x2 = (int32_t) float_as_uint32(vfpacc0x2) - vmagic_bias_less_output_zero_point;
    int32_t vout0x3 = (int32_t) float_as_uint32(vfpacc0x3) - vmagic_bias_less_output_zero_point;
    int32_t vout1x0 = (int32_t) float_as_uint32(vfpacc1x0) - vmagic_bias_less_output_zero_point;
    int32_t vout1x1 = (int32_t) float_as_uint32(vfpacc1x1) - vmagic_bias_less_output_zero_point;
    int32_t vout1x2 = (int32_t) float_as_uint32(vfpacc1x2) - vmagic_bias_less_output_zero_point;
    int32_t vout1x3 = (int32_t) float_as_uint32(vfpacc1x3) - vmagic_bias_less_output_zero_point;
    int32_t vout2x0 = (int32_t) float_as_uint32(vfpacc2x0) - vmagic_bias_less_output_zero_point;
    int32_t vout2x1 = (int32_t) float_as_uint32(vfpacc2x1) - vmagic_bias_less_output_zero_point;
    int32_t vout2x2 = (int32_t) float_as_uint32(vfpacc2x2) - vmagic_bias_less_output_zero_point;
    int32_t vout2x3 = (int32_t) float_as_uint32(vfpacc2x3) - vmagic_bias_less_output_zero_point;
    int32_t vout3x0 = (int32_t) float_as_uint32(vfpacc3x0) - vmagic_bias_less_output_zero_point;
    int32_t vout3x1 = (int32_t) float_as_uint32(vfpacc3x1) - vmagic_bias_less_output_zero_point;
    int32_t vout3x2 = (int32_t) float_as_uint32(vfpacc3x2) - vmagic_bias_less_output_zero_point;
    int32_t vout3x3 = (int32_t) float_as_uint32(vfpacc3x3) - vmagic_bias_less_output_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c1[2] = (int8_t) vout1x2;
      c1[3] = (int8_t) vout1x3;
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c2[2] = (int8_t) vout2x2;
      c2[3] = (int8_t) vout2x3;
      c3[0] = (int8_t) vout3x0;
      c3[1] = (int8_t) vout3x1;
      c3[2] = (int8_t) vout3x2;
      c3[3] = (int8_t) vout3x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = (int8_t) vout1x0;
        c1[1] = (int8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = (int8_t) vout2x0;
        c2[1] = (int8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c3[0] = (int8_t) vout3x0;
        c3[1] = (int8_t) vout3x1;
        vout3x0 = vout3x2;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
        c1[0] = (int8_t) vout1x0;
        c2[0] = (int8_t) vout2x0;
        c3[0] = (int8_t) vout3x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
        w = (const void*) ((const int8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    const float vscale2 = ((const float*) w)[2];
    vfpacc0x2 *= vscale2;
    const float vscale3 = ((const float*) w)[3];
    vfpacc0x3 *= vscale3;
    w = (const void*) ((const float*) w + 4);

    const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
    vfpacc0x0 = __builtin_wasm_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = __builtin_wasm_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = __builtin_wasm_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = __builtin_wasm_max_f32(vfpacc0x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
    vfpacc0x0 = __builtin_wasm_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = __builtin_wasm_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = __builtin_wasm_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = __builtin_wasm_min_f32(vfpacc0x3, voutput_max_less_zero_point);

    const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc0x2 += vmagic_bias;
    vfpacc0x3 += vmagic_bias;

    const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0) - vmagic_bias_less_output_zero_point;
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1) - vmagic_bias_less_output_zero_point;
    int32_t vout0x2 = (int32_t) float_as_uint32(vfpacc0x2) - vmagic_bias_less_output_zero_point;
    int32_t vout0x3 = (int32_t) float_as_uint32(vfpacc0x3) - vmagic_bias_less_output_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    int32_t vacc3x0 = vacc0x0;
    int32_t vacc3x1 = vacc0x1;
    int32_t vacc3x2 = vacc0x2;
    int32_t vacc3x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;
        const int32_t va1 = (int32_t) *a1++;
        const int32_t va2 = (int32_t) *a2++;
        const int32_t va3 = (int32_t) *a3++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
        w = (const void*) ((const int8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;
        vacc1x0 += va1 * vb0;
        vacc1x1 += va1 * vb1;
        vacc1x2 += va1 * vb2;
        vacc1x3 += va1 * vb3;
        vacc2x0 += va2 * vb0;
        vacc2x1 += va2 * vb1;
        vacc2x2 += va2 * vb2;
        vacc2x3 += va2 * vb3;
        vacc3x0 += va3 * vb0;
        vacc3x1 += va3 * vb1;
        vacc3x2 += va3 * vb2;
        vacc3x3 += va3 * vb3;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;
    float vfpacc3x0 = (float) vacc3x0;
    float vfpacc3x1 = (float) vacc3x1;
    float vfpacc3x2 = (float) vacc3x2;
    float vfpacc3x3 = (float) vacc3x3;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    vfpacc1x0 *= vscale0;
    vfpacc2x0 *= vscale0;
    vfpacc3x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    vfpacc1x1 *= vscale1;
    vfpacc2x1 *= vscale1;
    vfpacc3x1 *= vscale1;
    const float vscale2 = ((const float*) w)[2];
    vfpacc0x2 *= vscale2;
    vfpacc1x2 *= vscale2;
    vfpacc2x2 *= vscale2;
    vfpacc3x2 *= vscale2;
    const float vscale3 = ((const float*) w)[3];
    vfpacc0x3 *= vscale3;
    vfpacc1x3 *= vscale3;
    vfpacc2x3 *= vscale3;
    vfpacc3x3 *= vscale3;
    w = (const void*) ((const float*) w + 4);

    const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
    vfpacc0x0 = __builtin_wasm_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = __builtin_wasm_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = __builtin_wasm_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = __builtin_wasm_max_f32(vfpacc0x3, voutput_min_less_zero_point);
    vfpacc1x0 = __builtin_wasm_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = __builtin_wasm_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc1x2 = __builtin_wasm_max_f32(vfpacc1x2, voutput_min_less_zero_point);
    vfpacc1x3 = __builtin_wasm_max_f32(vfpacc1x3, voutput_min_less_zero_point);
    vfpacc2x0 = __builtin_wasm_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = __builtin_wasm_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc2x2 = __builtin_wasm_max_f32(vfpacc2x2, voutput_min_less_zero_point);
    vfpacc2x3 = __builtin_wasm_max_f32(vfpacc2x3, voutput_min_less_zero_point);
    vfpacc3x0 = __builtin_wasm_max_f32(vfpacc3x0, voutput_min_less_zero_point);
    vfpacc3x1 = __builtin_wasm_max_f32(vfpacc3x1, voutput_min_less_zero_point);
    vfpacc3x2 = __builtin_wasm_max_f32(vfpacc3x2, voutput_min_less_zero_point);
    vfpacc3x3 = __builtin_wasm_max_f32(vfpacc3x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
    vfpacc0x0 = __builtin_wasm_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = __builtin_wasm_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = __builtin_wasm_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = __builtin_wasm_min_f32(vfpacc0x3, voutput_max_less_zero_point);
    vfpacc1x0 = __builtin_wasm_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = __builtin_wasm_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc1x2 = __builtin_wasm_min_f32(vfpacc1x2, voutput_max_less_zero_point);
    vfpacc1x3 = __builtin_wasm_min_f32(vfpacc1x3, voutput_max_less_zero_point);
    vfpacc2x0 = __builtin_wasm_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = __builtin_wasm_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc2x2 = __builtin_wasm_min_f32(vfpacc2x2, voutput_max_less_zero_point);
    vfpacc2x3 = __builtin_wasm_min_f32(vfpacc2x3, voutput_max_less_zero_point);
    vfpacc3x0 = __builtin_wasm_min_f32(vfpacc3x0, voutput_max_less_zero_point);
    vfpacc3x1 = __builtin_wasm_min_f32(vfpacc3x1, voutput_max_less_zero_point);
    vfpacc3x2 = __builtin_wasm_min_f32(vfpacc3x2, voutput_max_less_zero_point);
    vfpacc3x3 = __builtin_wasm_min_f32(vfpacc3x3, voutput_max_less_zero_point);

    const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc0x2 += vmagic_bias;
    vfpacc0x3 += vmagic_bias;
    vfpacc1x0 += vmagic_bias;
    vfpacc1x1 += vmagic_bias;
    vfpacc1x2 += vmagic_bias;
    vfpacc1x3 += vmagic_bias;
    vfpacc2x0 += vmagic_bias;
    vfpacc2x1 += vmagic_bias;
    vfpacc2x2 += vmagic_bias;
    vfpacc2x3 += vmagic_bias;
    vfpacc3x0 += vmagic_bias;
    vfpacc3x1 += vmagic_bias;
    vfpacc3x2 += vmagic_bias;
    vfpacc3x3 += vmagic_bias;

    const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0) - vmagic_bias_less_output_zero_point;
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1) - vmagic_bias_less_output_zero_point;
    int32_t vout0x2 = (int32_t) float_as_uint32(vfpacc0x2) - vmagic_bias_less_output_zero_point;
    int32_t vout0x3 = (int32_t) float_as_uint32(vfpacc0x3) - vmagic_bias_less_output_zero_point;
    int32_t vout1x0 = (int32_t) float_as_uint32(vfpacc1x0) - vmagic_bias_less_output_zero_point;
    int32_t vout1x1 = (int32_t) float_as_uint32(vfpacc1x1) - vmagic_bias_less_output_zero_point;
    int32_t vout1x2 = (int32_t) float_as_uint32(vfpacc1x2) - vmagic_bias_less_output_zero_point;
    int32_t vout1x3 = (int32_t) float_as_uint32(vfpacc1x3) - vmagic_bias_less_output_zero_point;
    int32_t vout2x0 = (int32_t) float_as_uint32(vfpacc2x0) - vmagic_bias_less_output_zero_point;
    int32_t vout2x1 = (int32_t) float_as_uint32(vfpacc2x1) - vmagic_bias_less_output_zero_point;
    int32_t vout2x2 = (int32_t) float_as_uint32(vfpacc2x2) - vmagic_bias_less_output_zero_point;
    int32_t vout2x3 = (int32_t) float_as_uint32(vfpacc2x3) - vmagic_bias_less_output_zero_point;
    int32_t vout3x0 = (int32_t) float_as_uint32(vfpacc3x0) - vmagic_bias_less_output_zero_point;
    int32_t vout3x1 = (int32_t) float_as_uint32(vfpacc3x1) - vmagic_bias_less_output_zero_point;
    int32_t vout3x2 = (int32_t) float_as_uint32(vfpacc3x2) - vmagic_bias_less_output_zero_point;
    int32_t vout3x3 = (int32_t) float_as_uint32(vfpacc3x3) - vmagic_bias_less_output_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c3[0] = (int8_t) vout3x0;
      c3[1] = (int8_t) vout3x1;
      c3[2] = (int8_t) vout3x2;
      c3[3] = (int8_t) vout3x3;
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c2[2] = (int8_t) vout2x2;
      c2[3] = (int8_t) vout2x3;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c1[2] = (int8_t) vout1x2;
      c1[3] = (int8_t) vout1x3;
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = (int8_t) vout3x0;
        c3[1] = (int8_t) vout3x1;
        vout3x0 = vout3x2;
        c3 += 2;
        c2[0] = (int8_t) vout2x0;
        c2[1] = (int8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c1[0] = (int8_t) vout1x0;
        c1[1] = (int8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = (int8_t) vout3x0;
        c2[0] = (int8_t) vout2x0;
        c1[0] = (int8_t) vout1x0;
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_25p2c__wasm_fmagic(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vscale = params->fp32_scalar_fmagic.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
  const int32_t vkernel_zero_point = params->fp32_scalar_fmagic.kernel_zero_point;
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const uint8_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const uint8_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const uint8_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const uint8_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const uint8_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const uint8_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const uint8_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const uint8_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const uint8_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const uint8_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const uint8_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const uint8_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const uint8_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const uint8_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const uint8_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const uint8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) (uint32_t) i0[0];
      const int32_t vi0x1 = (int32_t) (uint32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0] - vkernel_zero_point;
      const int32_t vk0x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1] - vkernel_zero_point;

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) (uint32_t) i1[0];
      const int32_t vi1x1 = (int32_t) (uint32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2] - vkernel_zero_point;
      const int32_t vk1x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3] - vkernel_zero_point;

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) (uint32_t) i2[0];
      const int32_t vi2x1 = (int32_t) (uint32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4] - vkernel_zero_point;
      const int32_t vk2x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5] - vkernel_zero_point;

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) (uint32_t) i3[0];
      const int32_t vi3x1 = (int32_t) (uint32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6] - vkernel_zero_point;
      const int32_t vk3x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7] - vkernel_zero_point;

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) (uint32_t) i4[0];
      const int32_t vi4x1 = (int32_t) (uint32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8] - vkernel_zero_point;
      const int32_t vk4x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9] - vkernel_zero_point;

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) (uint32_t) i5[0];
      const int32_t vi5x1 = (int32_t) (uint32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10] - vkernel_zero_point;
      const int32_t vk5x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11] - vkernel_zero_point;

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) (uint32_t) i6[0];
      const int32_t vi6x1 = (int32_t) (uint32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12] - vkernel_zero_point;
      const int32_t vk6x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13] - vkernel_zero_point;

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) (uint32_t) i7[0];
      const int32_t vi7x1 = (int32_t) (uint32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14] - vkernel_zero_point;
      const int32_t vk7x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15] - vkernel_zero_point;

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) (uint32_t) i8[0];
      const int32_t vi8x1 = (int32_t) (uint32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16] - vkernel_zero_point;
      const int32_t vk8x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17] - vkernel_zero_point;

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      const int32_t vi9x0 = (int32_t) (uint32_t) i9[0];
      const int32_t vi9x1 = (int32_t) (uint32_t) i9[1];
      i9 += 2;

      const int32_t vk9x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18] - vkernel_zero_point;
      const int32_t vk9x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[19] - vkernel_zero_point;

      vacc0 += vi9x0 * vk9x0;
      vacc1 += vi9x1 * vk9x1;

      const int32_t vi10x0 = (int32_t) (uint32_t) i10[0];
      const int32_t vi10x1 = (int32_t) (uint32_t) i10[1];
      i10 += 2;

      const int32_t vk10x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20] - vkernel_zero_point;
      const int32_t vk10x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[21] - vkernel_zero_point;

      vacc0 += vi10x0 * vk10x0;
      vacc1 += vi10x1 * vk10x1;

      const int32_t vi11x0 = (int32_t) (uint32_t) i11[0];
      const int32_t vi11x1 = (int32_t) (uint32_t) i11[1];
      i11 += 2;

      const int32_t vk11x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22] - vkernel_zero_point;
      const int32_t vk11x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[23] - vkernel_zero_point;

      vacc0 += vi11x0 * vk11x0;
      vacc1 += vi11x1 * vk11x1;

      const int32_t vi12x0 = (int32_t) (uint32_t) i12[0];
      const int32_t vi12x1 = (int32_t) (uint32_t) i12[1];
      i12 += 2;

      const int32_t vk12x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24] - vkernel_zero_point;
      const int32_t vk12x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[25] - vkernel_zero_point;

      vacc0 += vi12x0 * vk12x0;
      vacc1 += vi12x1 * vk12x1;

      const int32_t vi13x0 = (int32_t) (uint32_t) i13[0];
      const int32_t vi13x1 = (int32_t) (uint32_t) i13[1];
      i13 += 2;

      const int32_t vk13x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26] - vkernel_zero_point;
      const int32_t vk13x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[27] - vkernel_zero_point;

      vacc0 += vi13x0 * vk13x0;
      vacc1 += vi13x1 * vk13x1;

      const int32_t vi14x0 = (int32_t) (uint32_t) i14[0];
      const int32_t vi14x1 = (int32_t) (uint32_t) i14[1];
      i14 += 2;

      const int32_t vk14x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28] - vkernel_zero_point;
      const int32_t vk14x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[29] - vkernel_zero_point;

      vacc0 += vi14x0 * vk14x0;
      vacc1 += vi14x1 * vk14x1;

      const int32_t vi15x0 = (int32_t) (uint32_t) i15[0];
      const int32_t vi15x1 = (int32_t) (uint32_t) i15[1];
      i15 += 2;

      const int32_t vk15x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30] - vkernel_zero_point;
      const int32_t vk15x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[31] - vkernel_zero_point;

      vacc0 += vi15x0 * vk15x0;
      vacc1 += vi15x1 * vk15x1;

      const int32_t vi16x0 = (int32_t) (uint32_t) i16[0];
      const int32_t vi16x1 = (int32_t) (uint32_t) i16[1];
      i16 += 2;

      const int32_t vk16x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32] - vkernel_zero_point;
      const int32_t vk16x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[33] - vkernel_zero_point;

      vacc0 += vi16x0 * vk16x0;
      vacc1 += vi16x1 * vk16x1;

      const int32_t vi17x0 = (int32_t) (uint32_t) i17[0];
      const int32_t vi17x1 = (int32_t) (uint32_t) i17[1];
      i17 += 2;

      const int32_t vk17x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34] - vkernel_zero_point;
      const int32_t vk17x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[35] - vkernel_zero_point;

      vacc0 += vi17x0 * vk17x0;
      vacc1 += vi17x1 * vk17x1;

      const int32_t vi18x0 = (int32_t) (uint32_t) i18[0];
      const int32_t vi18x1 = (int32_t) (uint32_t) i18[1];
      i18 += 2;

      const int32_t vk18x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36] - vkernel_zero_point;
      const int32_t vk18x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[37] - vkernel_zero_point;

      vacc0 += vi18x0 * vk18x0;
      vacc1 += vi18x1 * vk18x1;

      const int32_t vi19x0 = (int32_t) (uint32_t) i19[0];
      const int32_t vi19x1 = (int32_t) (uint32_t) i19[1];
      i19 += 2;

      const int32_t vk19x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38] - vkernel_zero_point;
      const int32_t vk19x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[39] - vkernel_zero_point;

      vacc0 += vi19x0 * vk19x0;
      vacc1 += vi19x1 * vk19x1;

      const int32_t vi20x0 = (int32_t) (uint32_t) i20[0];
      const int32_t vi20x1 = (int32_t) (uint32_t) i20[1];
      i20 += 2;

      const int32_t vk20x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40] - vkernel_zero_point;
      const int32_t vk20x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[41] - vkernel_zero_point;

      vacc0 += vi20x0 * vk20x0;
      vacc1 += vi20x1 * vk20x1;

      const int32_t vi21x0 = (int32_t) (uint32_t) i21[0];
      const int32_t vi21x1 = (int32_t) (uint32_t) i21[1];
      i21 += 2;

      const int32_t vk21x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42] - vkernel_zero_point;
      const int32_t vk21x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[43] - vkernel_zero_point;

      vacc0 += vi21x0 * vk21x0;
      vacc1 += vi21x1 * vk21x1;

      const int32_t vi22x0 = (int32_t) (uint32_t) i22[0];
      const int32_t vi22x1 = (int32_t) (uint32_t) i22[1];
      i22 += 2;

      const int32_t vk22x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44] - vkernel_zero_point;
      const int32_t vk22x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[45] - vkernel_zero_point;

      vacc0 += vi22x0 * vk22x0;
      vacc1 += vi22x1 * vk22x1;

      const int32_t vi23x0 = (int32_t) (uint32_t) i23[0];
      const int32_t vi23x1 = (int32_t) (uint32_t) i23[1];
      i23 += 2;

      const int32_t vk23x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46] - vkernel_zero_point;
      const int32_t vk23x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[47] - vkernel_zero_point;

      vacc0 += vi23x0 * vk23x0;
      vacc1 += vi23x1 * vk23x1;

      const int32_t vi24x0 = (int32_t) (uint32_t) i24[0];
      const int32_t vi24x1 = (int32_t) (uint32_t) i24[1];
      i24 += 2;

      const int32_t vk24x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48] - vkernel_zero_point;
      const int32_t vk24x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[49] - vkernel_zero_point;

      vacc0 += vi24x0 * vk24x0;
      vacc1 += vi24x1 * vk24x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 50 * sizeof(uint8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      vfpacc0 *= vscale;
      vfpacc1 *= vscale;

      vfpacc0 = __builtin_wasm_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = __builtin_wasm_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = __builtin_wasm_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = __builtin_wasm_min_f32(vfpacc1, voutput_max_less_zero_point);

      vfpacc0 += vmagic_bias;
      vfpacc1 += vmagic_bias;

      int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
      int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;

      output[0] = (uint8_t) vout0;
      output[1] = (uint8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) (uint32_t) *i0;
      const int32_t vk0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0] - vkernel_zero_point;
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) (uint32_t) *i1;
      const int32_t vk1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2] - vkernel_zero_point;
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) (uint32_t) *i2;
      const int32_t vk2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4] - vkernel_zero_point;
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) (uint32_t) *i3;
      const int32_t vk3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6] - vkernel_zero_point;
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) (uint32_t) *i4;
      const int32_t vk4 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8] - vkernel_zero_point;
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) (uint32_t) *i5;
      const int32_t vk5 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10] - vkernel_zero_point;
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) (uint32_t) *i6;
      const int32_t vk6 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12] - vkernel_zero_point;
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) (uint32_t) *i7;
      const int32_t vk7 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14] - vkernel_zero_point;
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) (uint32_t) *i8;
      const int32_t vk8 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16] - vkernel_zero_point;
      vacc += vi8 * vk8;
      const int32_t vi9 = (int32_t) (uint32_t) *i9;
      const int32_t vk9 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[18] - vkernel_zero_point;
      vacc += vi9 * vk9;
      const int32_t vi10 = (int32_t) (uint32_t) *i10;
      const int32_t vk10 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[20] - vkernel_zero_point;
      vacc += vi10 * vk10;
      const int32_t vi11 = (int32_t) (uint32_t) *i11;
      const int32_t vk11 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[22] - vkernel_zero_point;
      vacc += vi11 * vk11;
      const int32_t vi12 = (int32_t) (uint32_t) *i12;
      const int32_t vk12 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[24] - vkernel_zero_point;
      vacc += vi12 * vk12;
      const int32_t vi13 = (int32_t) (uint32_t) *i13;
      const int32_t vk13 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[26] - vkernel_zero_point;
      vacc += vi13 * vk13;
      const int32_t vi14 = (int32_t) (uint32_t) *i14;
      const int32_t vk14 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[28] - vkernel_zero_point;
      vacc += vi14 * vk14;
      const int32_t vi15 = (int32_t) (uint32_t) *i15;
      const int32_t vk15 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[30] - vkernel_zero_point;
      vacc += vi15 * vk15;
      const int32_t vi16 = (int32_t) (uint32_t) *i16;
      const int32_t vk16 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[32] - vkernel_zero_point;
      vacc += vi16 * vk16;
      const int32_t vi17 = (int32_t) (uint32_t) *i17;
      const int32_t vk17 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[34] - vkernel_zero_point;
      vacc += vi17 * vk17;
      const int32_t vi18 = (int32_t) (uint32_t) *i18;
      const int32_t vk18 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[36] - vkernel_zero_point;
      vacc += vi18 * vk18;
      const int32_t vi19 = (int32_t) (uint32_t) *i19;
      const int32_t vk19 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[38] - vkernel_zero_point;
      vacc += vi19 * vk19;
      const int32_t vi20 = (int32_t) (uint32_t) *i20;
      const int32_t vk20 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[40] - vkernel_zero_point;
      vacc += vi20 * vk20;
      const int32_t vi21 = (int32_t) (uint32_t) *i21;
      const int32_t vk21 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[42] - vkernel_zero_point;
      vacc += vi21 * vk21;
      const int32_t vi22 = (int32_t) (uint32_t) *i22;
      const int32_t vk22 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[44] - vkernel_zero_point;
      vacc += vi22 * vk22;
      const int32_t vi23 = (int32_t) (uint32_t) *i23;
      const int32_t vk23 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[46] - vkernel_zero_point;
      vacc += vi23 * vk23;
      const int32_t vi24 = (int32_t) (uint32_t) *i24;
      const int32_t vk24 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[48] - vkernel_zero_point;
      vacc += vi24 * vk24;

      float vfpacc = (float) vacc * vscale;

      vfpacc = __builtin_wasm_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = __builtin_wasm_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;

      *output++ = (uint8_t) vout;
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_9p2c__wasm_fmagic(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vscale = params->fp32_scalar_fmagic.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
  const int32_t vkernel_zero_point = params->fp32_scalar_fmagic.kernel_zero_point;
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 2; c -= 2) {
      int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
      int32_t vacc1 = unaligned_indexed_load_s32(w, 1);


      const int32_t vi0x0 = (int32_t) (uint32_t) i0[0];
      const int32_t vi0x1 = (int32_t) (uint32_t) i0[1];
      i0 += 2;

      const int32_t vk0x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0] - vkernel_zero_point;
      const int32_t vk0x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1] - vkernel_zero_point;

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;

      const int32_t vi1x0 = (int32_t) (uint32_t) i1[0];
      const int32_t vi1x1 = (int32_t) (uint32_t) i1[1];
      i1 += 2;

      const int32_t vk1x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2] - vkernel_zero_point;
      const int32_t vk1x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3] - vkernel_zero_point;

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;

      const int32_t vi2x0 = (int32_t) (uint32_t) i2[0];
      const int32_t vi2x1 = (int32_t) (uint32_t) i2[1];
      i2 += 2;

      const int32_t vk2x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4] - vkernel_zero_point;
      const int32_t vk2x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5] - vkernel_zero_point;

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;

      const int32_t vi3x0 = (int32_t) (uint32_t) i3[0];
      const int32_t vi3x1 = (int32_t) (uint32_t) i3[1];
      i3 += 2;

      const int32_t vk3x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6] - vkernel_zero_point;
      const int32_t vk3x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7] - vkernel_zero_point;

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;

      const int32_t vi4x0 = (int32_t) (uint32_t) i4[0];
      const int32_t vi4x1 = (int32_t) (uint32_t) i4[1];
      i4 += 2;

      const int32_t vk4x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8] - vkernel_zero_point;
      const int32_t vk4x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9] - vkernel_zero_point;

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;

      const int32_t vi5x0 = (int32_t) (uint32_t) i5[0];
      const int32_t vi5x1 = (int32_t) (uint32_t) i5[1];
      i5 += 2;

      const int32_t vk5x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10] - vkernel_zero_point;
      const int32_t vk5x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11] - vkernel_zero_point;

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;

      const int32_t vi6x0 = (int32_t) (uint32_t) i6[0];
      const int32_t vi6x1 = (int32_t) (uint32_t) i6[1];
      i6 += 2;

      const int32_t vk6x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12] - vkernel_zero_point;
      const int32_t vk6x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13] - vkernel_zero_point;

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;

      const int32_t vi7x0 = (int32_t) (uint32_t) i7[0];
      const int32_t vi7x1 = (int32_t) (uint32_t) i7[1];
      i7 += 2;

      const int32_t vk7x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14] - vkernel_zero_point;
      const int32_t vk7x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15] - vkernel_zero_point;

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;

      const int32_t vi8x0 = (int32_t) (uint32_t) i8[0];
      const int32_t vi8x1 = (int32_t) (uint32_t) i8[1];
      i8 += 2;

      const int32_t vk8x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16] - vkernel_zero_point;
      const int32_t vk8x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[17] - vkernel_zero_point;

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;

      w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 18 * sizeof(uint8_t));

      float vfpacc0 = (float) vacc0;
      float vfpacc1 = (float) vacc1;

      vfpacc0 *= vscale;
      vfpacc1 *= vscale;

      vfpacc0 = __builtin_wasm_max_f32(vfpacc0, voutput_min_less_zero_point);
      vfpacc1 = __builtin_wasm_max_f32(vfpacc1, voutput_min_less_zero_point);

      vfpacc0 = __builtin_wasm_min_f32(vfpacc0, voutput_max_less_zero_point);
      vfpacc1 = __builtin_wasm_min_f32(vfpacc1, voutput_max_less_zero_point);

      vfpacc0 += vmagic_bias;
      vfpacc1 += vmagic_bias;

      int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
      int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;

      output[0] = (uint8_t) vout0;
      output[1] = (uint8_t) vout1;
      output += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      int32_t vacc = unaligned_load_s32(w);

      const int32_t vi0 = (int32_t) (uint32_t) *i0;
      const int32_t vk0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0] - vkernel_zero_point;
      vacc += vi0 * vk0;
      const int32_t vi1 = (int32_t) (uint32_t) *i1;
      const int32_t vk1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2] - vkernel_zero_point;
      vacc += vi1 * vk1;
      const int32_t vi2 = (int32_t) (uint32_t) *i2;
      const int32_t vk2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4] - vkernel_zero_point;
      vacc += vi2 * vk2;
      const int32_t vi3 = (int32_t) (uint32_t) *i3;
      const int32_t vk3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6] - vkernel_zero_point;
      vacc += vi3 * vk3;
      const int32_t vi4 = (int32_t) (uint32_t) *i4;
      const int32_t vk4 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8] - vkernel_zero_point;
      vacc += vi4 * vk4;
      const int32_t vi5 = (int32_t) (uint32_t) *i5;
      const int32_t vk5 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10] - vkernel_zero_point;
      vacc += vi5 * vk5;
      const int32_t vi6 = (int32_t) (uint32_t) *i6;
      const int32_t vk6 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12] - vkernel_zero_point;
      vacc += vi6 * vk6;
      const int32_t vi7 = (int32_t) (uint32_t) *i7;
      const int32_t vk7 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14] - vkernel_zero_point;
      vacc += vi7 * vk7;
      const int32_t vi8 = (int32_t) (uint32_t) *i8;
      const int32_t vk8 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[16] - vkernel_zero_point;
      vacc += vi8 * vk8;

      float vfpacc = (float) vacc * vscale;

      vfpacc = __builtin_wasm_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = __builtin_wasm_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;

      *output++ = (uint8_t) vout;
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const uint8_t* a0 = a;
  uint8_t* c0 = c;

  const int32_t vb_zero_point = params->fp32_scalar_fmagic.kernel_zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) (uint32_t) *a0++;

      const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
      const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
      const int32_t vb2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vb_zero_point;
      const int32_t vb3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vb_zero_point;
      w = (const uint8_t*) w + 4;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;

      k -= sizeof(uint8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;

    const float vscale = params->fp32_scalar_fmagic.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
    vfpacc0x0 = __builtin_wasm_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = __builtin_wasm_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = __builtin_wasm_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = __builtin_wasm_max_f32(vfpacc0x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
    vfpacc0x0 = __builtin_wasm_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = __builtin_wasm_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = __builtin_wasm_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = __builtin_wasm_min_f32(vfpacc0x3, voutput_max_less_zero_point);

    const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc0x2 += vmagic_bias;
    vfpacc0x3 += vmagic_bias;

    const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0) - vmagic_bias_less_output_zero_point;
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1) - vmagic_bias_less_output_zero_point;
    int32_t vout0x2 = (int32_t) float_as_uint32(vfpacc0x2) - vmagic_bias_less_output_zero_point;
    int32_t vout0x3 = (int32_t) float_as_uint32(vfpacc0x3) - vmagic_bias_less_output_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c0[2] = (uint8_t) vout0x2;
      c0[3] = (uint8_t) vout0x3;

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (uint8_t) vout0x0;
        c0[1] = (uint8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (uint8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);

  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const int32_t vb_zero_point = params->fp32_scalar_fmagic.kernel_zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    int32_t vacc3x0 = vacc0x0;
    int32_t vacc3x1 = vacc0x1;
    int32_t vacc3x2 = vacc0x2;
    int32_t vacc3x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) (uint32_t) *a0++;
      const int32_t va1 = (int32_t) (uint32_t) *a1++;
      const int32_t va2 = (int32_t) (uint32_t) *a2++;
      const int32_t va3 = (int32_t) (uint32_t) *a3++;

      const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
      const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
      const int32_t vb2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vb_zero_point;
      const int32_t vb3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vb_zero_point;
      w = (const uint8_t*) w + 4;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;
      vacc1x2 += va1 * vb2;
      vacc1x3 += va1 * vb3;
      vacc2x0 += va2 * vb0;
      vacc2x1 += va2 * vb1;
      vacc2x2 += va2 * vb2;
      vacc2x3 += va2 * vb3;
      vacc3x0 += va3 * vb0;
      vacc3x1 += va3 * vb1;
      vacc3x2 += va3 * vb2;
      vacc3x3 += va3 * vb3;

      k -= sizeof(uint8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;
    float vfpacc3x0 = (float) vacc3x0;
    float vfpacc3x1 = (float) vacc3x1;
    float vfpacc3x2 = (float) vacc3x2;
    float vfpacc3x3 = (float) vacc3x3;

    const float vscale = params->fp32_scalar_fmagic.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;
    vfpacc1x2 *= vscale;
    vfpacc1x3 *= vscale;
    vfpacc2x0 *= vscale;
    vfpacc2x1 *= vscale;
    vfpacc2x2 *= vscale;
    vfpacc2x3 *= vscale;
    vfpacc3x0 *= vscale;
    vfpacc3x1 *= vscale;
    vfpacc3x2 *= vscale;
    vfpacc3x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
    vfpacc0x0 = __builtin_wasm_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = __builtin_wasm_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = __builtin_wasm_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = __builtin_wasm_max_f32(vfpacc0x3, voutput_min_less_zero_point);
    vfpacc1x0 = __builtin_wasm_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = __builtin_wasm_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc1x2 = __builtin_wasm_max_f32(vfpacc1x2, voutput_min_less_zero_point);
    vfpacc1x3 = __builtin_wasm_max_f32(vfpacc1x3, voutput_min_less_zero_point);
    vfpacc2x0 = __builtin_wasm_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = __builtin_wasm_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc2x2 = __builtin_wasm_max_f32(vfpacc2x2, voutput_min_less_zero_point);
    vfpacc2x3 = __builtin_wasm_max_f32(vfpacc2x3, voutput_min_less_zero_point);
    vfpacc3x0 = __builtin_wasm_max_f32(vfpacc3x0, voutput_min_less_zero_point);
    vfpacc3x1 = __builtin_wasm_max_f32(vfpacc3x1, voutput_min_less_zero_point);
    vfpacc3x2 = __builtin_wasm_max_f32(vfpacc3x2, voutput_min_less_zero_point);
    vfpacc3x3 = __builtin_wasm_max_f32(vfpacc3x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
    vfpacc0x0 = __builtin_wasm_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = __builtin_wasm_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = __builtin_wasm_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = __builtin_wasm_min_f32(vfpacc0x3, voutput_max_less_zero_point);
    vfpacc1x0 = __builtin_wasm_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = __builtin_wasm_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc1x2 = __builtin_wasm_min_f32(vfpacc1x2, voutput_max_less_zero_point);
    vfpacc1x3 = __builtin_wasm_min_f32(vfpacc1x3, voutput_max_less_zero_point);
    vfpacc2x0 = __builtin_wasm_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = __builtin_wasm_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc2x2 = __builtin_wasm_min_f32(vfpacc2x2, voutput_max_less_zero_point);
    vfpacc2x3 = __builtin_wasm_min_f32(vfpacc2x3, voutput_max_less_zero_point);
    vfpacc3x0 = __builtin_wasm_min_f32(vfpacc3x0, voutput_max_less_zero_point);
    vfpacc3x1 = __builtin_wasm_min_f32(vfpacc3x1, voutput_max_less_zero_point);
    vfpacc3x2 = __builtin_wasm_min_f32(vfpacc3x2, voutput_max_less_zero_point);
    vfpacc3x3 = __builtin_wasm_min_f32(vfpacc3x3, voutput_max_less_zero_point);

    const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc0x2 += vmagic_bias;
    vfpacc0x3 += vmagic_bias;
    vfpacc1x0 += vmagic_bias;
    vfpacc1x1 += vmagic_bias;
    vfpacc1x2 += vmagic_bias;
    vfpacc1x3 += vmagic_bias;
    vfpacc2x0 += vmagic_bias;
    vfpacc2x1 += vmagic_bias;
    vfpacc2x2 += vmagic_bias;
    vfpacc2x3 += vmagic_bias;
    vfpacc3x0 += vmagic_bias;
    vfpacc3x1 += vmagic_bias;
    vfpacc3x2 += vmagic_bias;
    vfpacc3x3 += vmagic_bias;

    const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0) - vmagic_bias_less_output_zero_point;
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1) - vmagic_bias_less_output_zero_point;
    int32_t vout0x2 = (int32_t) float_as_uint32(vfpacc0x2) - vmagic_bias_less_output_zero_point;
    int32_t vout0x3 = (int32_t) float_as_uint32(vfpacc0x3) - vmagic_bias_less_output_zero_point;
    int32_t vout1x0 = (int32_t) float_as_uint32(vfpacc1x0) - vmagic_bias_less_output_zero_point;
    int32_t vout1x1 = (int32_t) float_as_uint32(vfpacc1x1) - vmagic_bias_less_output_zero_point;
    int32_t vout1x2 = (int32_t) float_as_uint32(vfpacc1x2) - vmagic_bias_less_output_zero_point;
    int32_t vout1x3 = (int32_t) float_as_uint32(vfpacc1x3) - vmagic_bias_less_output_zero_point;
    int32_t vout2x0 = (int32_t) float_as_uint32(vfpacc2x0) - vmagic_bias_less_output_zero_point;
    int32_t vout2x1 = (int32_t) float_as_uint32(vfpacc2x1) - vmagic_bias_less_output_zero_point;
    int32_t vout2x2 = (int32_t) float_as_uint32(vfpacc2x2) - vmagic_bias_less_output_zero_point;
    int32_t vout2x3 = (int32_t) float_as_uint32(vfpacc2x3) - vmagic_bias_less_output_zero_point;
    int32_t vout3x0 = (int32_t) float_as_uint32(vfpacc3x0) - vmagic_bias_less_output_zero_point;
    int32_t vout3x1 = (int32_t) float_as_uint32(vfpacc3x1) - vmagic_bias_less_output_zero_point;
    int32_t vout3x2 = (int32_t) float_as_uint32(vfpacc3x2) - vmagic_bias_less_output_zero_point;
    int32_t vout3x3 = (int32_t) float_as_uint32(vfpacc3x3) - vmagic_bias_less_output_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c0[2] = (uint8_t) vout0x2;
      c0[3] = (uint8_t) vout0x3;
      c1[0] = (uint8_t) vout1x0;
      c1[1] = (uint8_t) vout1x1;
      c1[2] = (uint8_t) vout1x2;
      c1[3] = (uint8_t) vout1x3;
      c2[0] = (uint8_t) vout2x0;
      c2[1] = (uint8_t) vout2x1;
      c2[2] = (uint8_t) vout2x2;
      c2[3] = (uint8_t) vout2x3;
      c3[0] = (uint8_t) vout3x0;
      c3[1] = (uint8_t) vout3x1;
      c3[2] = (uint8_t) vout3x2;
      c3[3] = (uint8_t) vout3x3;

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint8_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint8_t*) ((uintptr_t) a3 - kc);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (uint8_t) vout0x0;
        c0[1] = (uint8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = (uint8_t) vout1x0;
        c1[1] = (uint8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = (uint8_t) vout2x0;
        c2[1] = (uint8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c3[0] = (uint8_t) vout3x0;
        c3[1] = (uint8_t) vout3x1;
        vout3x0 = vout3x2;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = (uint8_t) vout0x0;
        c1[0] = (uint8_t) vout1x0;
        c2[0] = (uint8_t) vout2x0;
        c3[0] = (uint8_t) vout3x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint8_t* c0 = c;

  const int32_t vb_zero_point = params->fp32_scalar_fmagic.kernel_zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) (uint32_t) *a0++;

        const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
        const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
        const int32_t vb2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vb_zero_point;
        const int32_t vb3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vb_zero_point;
        w = (const void*) ((const uint8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;

        k -= sizeof(uint8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;

    const float vscale = params->fp32_scalar_fmagic.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
    vfpacc0x0 = __builtin_wasm_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = __builtin_wasm_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = __builtin_wasm_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = __builtin_wasm_max_f32(vfpacc0x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
    vfpacc0x0 = __builtin_wasm_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = __builtin_wasm_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = __builtin_wasm_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = __builtin_wasm_min_f32(vfpacc0x3, voutput_max_less_zero_point);

    const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc0x2 += vmagic_bias;
    vfpacc0x3 += vmagic_bias;

    const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0) - vmagic_bias_less_output_zero_point;
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1) - vmagic_bias_less_output_zero_point;
    int32_t vout0x2 = (int32_t) float_as_uint32(vfpacc0x2) - vmagic_bias_less_output_zero_point;
    int32_t vout0x3 = (int32_t) float_as_uint32(vfpacc0x3) - vmagic_bias_less_output_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c0[2] = (uint8_t) vout0x2;
      c0[3] = (uint8_t) vout0x3;

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (uint8_t) vout0x0;
        c0[1] = (uint8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (uint8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const int32_t vb_zero_point = params->fp32_scalar_fmagic.kernel_zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    int32_t vacc3x0 = vacc0x0;
    int32_t vacc3x1 = vacc0x1;
    int32_t vacc3x2 = vacc0x2;
    int32_t vacc3x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint8_t* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint8_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint8_t* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const uint8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) (uint32_t) *a0++;
        const int32_t va1 = (int32_t) (uint32_t) *a1++;
        const int32_t va2 = (int32_t) (uint32_t) *a2++;
        const int32_t va3 = (int32_t) (uint32_t) *a3++;

        const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
        const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
        const int32_t vb2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vb_zero_point;
        const int32_t vb3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vb_zero_point;
        w = (const void*) ((const uint8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;
        vacc1x0 += va1 * vb0;
        vacc1x1 += va1 * vb1;
        vacc1x2 += va1 * vb2;
        vacc1x3 += va1 * vb3;
        vacc2x0 += va2 * vb0;
        vacc2x1 += va2 * vb1;
        vacc2x2 += va2 * vb2;
        vacc2x3 += va2 * vb3;
        vacc3x0 += va3 * vb0;
        vacc3x1 += va3 * vb1;
        vacc3x2 += va3 * vb2;
        vacc3x3 += va3 * vb3;

        k -= sizeof(uint8_t);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;
    float vfpacc3x0 = (float) vacc3x0;
    float vfpacc3x1 = (float) vacc3x1;
    float vfpacc3x2 = (float) vacc3x2;
    float vfpacc3x3 = (float) vacc3x3;

    const float vscale = params->fp32_scalar_fmagic.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;
    vfpacc1x2 *= vscale;
    vfpacc1x3 *= vscale;
    vfpacc2x0 *= vscale;
    vfpacc2x1 *= vscale;
    vfpacc2x2 *= vscale;
    vfpacc2x3 *= vscale;
    vfpacc3x0 *= vscale;
    vfpacc3x1 *= vscale;
    vfpacc3x2 *= vscale;
    vfpacc3x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
    vfpacc0x0 = __builtin_wasm_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = __builtin_wasm_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = __builtin_wasm_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = __builtin_wasm_max_f32(vfpacc0x3, voutput_min_less_zero_point);
    vfpacc1x0 = __builtin_wasm_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = __builtin_wasm_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc1x2 = __builtin_wasm_max_f32(vfpacc1x2, voutput_min_less_zero_point);
    vfpacc1x3 = __builtin_wasm_max_f32(vfpacc1x3, voutput_min_less_zero_point);
    vfpacc2x0 = __builtin_wasm_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = __builtin_wasm_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc2x2 = __builtin_wasm_max_f32(vfpacc2x2, voutput_min_less_zero_point);
    vfpacc2x3 = __builtin_wasm_max_f32(vfpacc2x3, voutput_min_less_zero_point);
    vfpacc3x0 = __builtin_wasm_max_f32(vfpacc3x0, voutput_min_less_zero_point);
    vfpacc3x1 = __builtin_wasm_max_f32(vfpacc3x1, voutput_min_less_zero_point);
    vfpacc3x2 = __builtin_wasm_max_f32(vfpacc3x2, voutput_min_less_zero_point);
    vfpacc3x3 = __builtin_wasm_max_f32(vfpacc3x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
    vfpacc0x0 = __builtin_wasm_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = __builtin_wasm_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = __builtin_wasm_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = __builtin_wasm_min_f32(vfpacc0x3, voutput_max_less_zero_point);
    vfpacc1x0 = __builtin_wasm_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = __builtin_wasm_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc1x2 = __builtin_wasm_min_f32(vfpacc1x2, voutput_max_less_zero_point);
    vfpacc1x3 = __builtin_wasm_min_f32(vfpacc1x3, voutput_max_less_zero_point);
    vfpacc2x0 = __builtin_wasm_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = __builtin_wasm_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc2x2 = __builtin_wasm_min_f32(vfpacc2x2, voutput_max_less_zero_point);
    vfpacc2x3 = __builtin_wasm_min_f32(vfpacc2x3, voutput_max_less_zero_point);
    vfpacc3x0 = __builtin_wasm_min_f32(vfpacc3x0, voutput_max_less_zero_point);
    vfpacc3x1 = __builtin_wasm_min_f32(vfpacc3x1, voutput_max_less_zero_point);
    vfpacc3x2 = __builtin_wasm_min_f32(vfpacc3x2, voutput_max_less_zero_point);
    vfpacc3x3 = __builtin_wasm_min_f32(vfpacc3x3, voutput_max_less_zero_point);

    const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc0x2 += vmagic_bias;
    vfpacc0x3 += vmagic_bias;
    vfpacc1x0 += vmagic_bias;
    vfpacc1x1 += vmagic_bias;
    vfpacc1x2 += vmagic_bias;
    vfpacc1x3 += vmagic_bias;
    vfpacc2x0 += vmagic_bias;
    vfpacc2x1 += vmagic_bias;
    vfpacc2x2 += vmagic_bias;
    vfpacc2x3 += vmagic_bias;
    vfpacc3x0 += vmagic_bias;
    vfpacc3x1 += vmagic_bias;
    vfpacc3x2 += vmagic_bias;
    vfpacc3x3 += vmagic_bias;

    const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0) - vmagic_bias_less_output_zero_point;
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1) - vmagic_bias_less_output_zero_point;
    int32_t vout0x2 = (int32_t) float_as_uint32(vfpacc0x2) - vmagic_bias_less_output_zero_point;
    int32_t vout0x3 = (int32_t) float_as_uint32(vfpacc0x3) - vmagic_bias_less_output_zero_point;
    int32_t vout1x0 = (int32_t) float_as_uint32(vfpacc1x0) - vmagic_bias_less_output_zero_point;
    int32_t vout1x1 = (int32_t) float_as_uint32(vfpacc1x1) - vmagic_bias_less_output_zero_point;
    int32_t vout1x2 = (int32_t) float_as_uint32(vfpacc1x2) - vmagic_bias_less_output_zero_point;
    int32_t vout1x3 = (int32_t) float_as_uint32(vfpacc1x3) - vmagic_bias_less_output_zero_point;
    int32_t vout2x0 = (int32_t) float_as_uint32(vfpacc2x0) - vmagic_bias_less_output_zero_point;
    int32_t vout2x1 = (int32_t) float_as_uint32(vfpacc2x1) - vmagic_bias_less_output_zero_point;
    int32_t vout2x2 = (int32_t) float_as_uint32(vfpacc2x2) - vmagic_bias_less_output_zero_point;
    int32_t vout2x3 = (int32_t) float_as_uint32(vfpacc2x3) - vmagic_bias_less_output_zero_point;
    int32_t vout3x0 = (int32_t) float_as_uint32(vfpacc3x0) - vmagic_bias_less_output_zero_point;
    int32_t vout3x1 = (int32_t) float_as_uint32(vfpacc3x1) - vmagic_bias_less_output_zero_point;
    int32_t vout3x2 = (int32_t) float_as_uint32(vfpacc3x2) - vmagic_bias_less_output_zero_point;
    int32_t vout3x3 = (int32_t) float_as_uint32(vfpacc3x3) - vmagic_bias_less_output_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c3[0] = (uint8_t) vout3x0;
      c3[1] = (uint8_t) vout3x1;
      c3[2] = (uint8_t) vout3x2;
      c3[3] = (uint8_t) vout3x3;
      c2[0] = (uint8_t) vout2x0;
      c2[1] = (uint8_t) vout2x1;
      c2[2] = (uint8_t) vout2x2;
      c2[3] = (uint8_t) vout2x3;
      c1[0] = (uint8_t) vout1x0;
      c1[1] = (uint8_t) vout1x1;
      c1[2] = (uint8_t) vout1x2;
      c1[3] = (uint8_t) vout1x3;
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c0[2] = (uint8_t) vout0x2;
      c0[3] = (uint8_t) vout0x3;

      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = (uint8_t) vout3x0;
        c3[1] = (uint8_t) vout3x1;
        vout3x0 = vout3x2;
        c3 += 2;
        c2[0] = (uint8_t) vout2x0;
        c2[1] = (uint8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c1[0] = (uint8_t) vout1x0;
        c1[1] = (uint8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c0[0] = (uint8_t) vout0x0;
        c0[1] = (uint8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = (uint8_t) vout3x0;
        c2[0] = (uint8_t) vout2x0;
        c1[0] = (uint8_t) vout1x0;
        c0[0] = (uint8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
