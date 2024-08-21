// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"
#include "xnnpack/conv.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/gemm.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/microparams.h"
#include "xnnpack/pavgpool.h"
#include "xnnpack/reduce.h"
#include "xnnpack/spmm.h"
#include "xnnpack/transpose.h"
#include "xnnpack/vbinary.h"
#include "xnnpack/vmulcaddc.h"
#include "xnnpack/vunary.h"


void xnn_f32_avgpool_minmax_ukernel_9p8x__sse_c4(
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
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const __m128 vscale = _mm_set1_ps(params->sse.scale);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  
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
      for (size_t c = 0; c < channels; c += 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vi8 = _mm_loadu_ps(i8);
        i8 += 4;

        const __m128 vsum01 = _mm_add_ps(vi0, vi1);
        const __m128 vsum23 = _mm_add_ps(vi2, vi3);
        const __m128 vsum45 = _mm_add_ps(vi4, vi5);
        const __m128 vsum67 = _mm_add_ps(vi6, vi7);
        const __m128 vsum018 = _mm_add_ps(vsum01, vi8);
        const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
        const __m128 vsum01678 = _mm_add_ps(vsum018, vsum67);
        const __m128 vsum = _mm_add_ps(vsum2345, vsum01678);

        _mm_store_ps(b, vsum); b += 4;
      }
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
      for (size_t c = 0; c < channels; c += 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vacc = _mm_load_ps(b);

        const __m128 vsum01 = _mm_add_ps(vi0, vi1);
        const __m128 vsum23 = _mm_add_ps(vi2, vi3);
        const __m128 vsum45 = _mm_add_ps(vi4, vi5);
        const __m128 vsum67 = _mm_add_ps(vi6, vi7);
        const __m128 vsum01a = _mm_add_ps(vsum01, vacc);
        const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
        const __m128 vsum0167a = _mm_add_ps(vsum01a, vsum67);
        const __m128 vsum = _mm_add_ps(vsum2345, vsum0167a);

        _mm_store_ps(b, vsum); b += 4;
      }
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
      while (c >= 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vacc = _mm_load_ps(b);
        b += 4;

        const __m128 vsum01 = _mm_add_ps(vi0, vi1);
        const __m128 vsum23 = _mm_add_ps(vi2, vi3);
        const __m128 vsum45 = _mm_add_ps(vi4, vi5);
        const __m128 vsum67 = _mm_add_ps(vi6, vi7);
        const __m128 vsum01a = _mm_add_ps(vsum01, vacc);
        const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
        const __m128 vsum0167a = _mm_add_ps(vsum01a, vsum67);
        const __m128 vsum = _mm_add_ps(vsum2345, vsum0167a);

        __m128 vout = _mm_mul_ps(vsum, vscale);
        vout = _mm_max_ps(vout, vmin);
        vout = _mm_min_ps(vout, vmax);

        _mm_storeu_ps(output, vout);
        output += 4;

        c -= 4;
      }
      if (c != 0) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        const __m128 vi1 = _mm_loadu_ps(i1);
        const __m128 vi2 = _mm_loadu_ps(i2);
        const __m128 vi3 = _mm_loadu_ps(i3);
        const __m128 vi4 = _mm_loadu_ps(i4);
        const __m128 vi5 = _mm_loadu_ps(i5);
        const __m128 vi6 = _mm_loadu_ps(i6);
        const __m128 vi7 = _mm_loadu_ps(i7);
        const __m128 vacc = _mm_load_ps(b);

        const __m128 vsum01 = _mm_add_ps(vi0, vi1);
        const __m128 vsum23 = _mm_add_ps(vi2, vi3);
        const __m128 vsum45 = _mm_add_ps(vi4, vi5);
        const __m128 vsum67 = _mm_add_ps(vi6, vi7);
        const __m128 vsum01a = _mm_add_ps(vsum01, vacc);
        const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
        const __m128 vsum0167a = _mm_add_ps(vsum01a, vsum67);
        const __m128 vsum = _mm_add_ps(vsum2345, vsum0167a);

        __m128 vout = _mm_mul_ps(vsum, vscale);
        vout = _mm_max_ps(vout, vmin);
        vout = _mm_min_ps(vout, vmax);

        if (c & 2) {
          _mm_storel_pi((__m64*) output, vout);
          vout = _mm_movehl_ps(vout, vout);
          output += 2;
        }
        if (c & 1) {
          _mm_store_ss(output, vout);
          output += 1;
        }
      }
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_avgpool_minmax_ukernel_9x__sse_c4(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const __m128 vscale = _mm_set1_ps(params->sse.scale);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

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
    while (c >= 4) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1 = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2 = _mm_loadu_ps(i2);
      i2 += 4;
      const __m128 vi3 = _mm_loadu_ps(i3);
      i3 += 4;
      const __m128 vi4 = _mm_loadu_ps(i4);
      i4 += 4;
      const __m128 vi5 = _mm_loadu_ps(i5);
      i5 += 4;
      const __m128 vi6 = _mm_loadu_ps(i6);
      i6 += 4;
      const __m128 vi7 = _mm_loadu_ps(i7);
      i7 += 4;
      const __m128 vi8 = _mm_loadu_ps(i8);
      i8 += 4;

      const __m128 vsum018 = _mm_add_ps(_mm_add_ps(vi0, vi1), vi8);
      const __m128 vsum23 = _mm_add_ps(vi2, vi3);
      const __m128 vsum45 = _mm_add_ps(vi4, vi5);
      const __m128 vsum67 = _mm_add_ps(vi6, vi7);

      const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
      const __m128 vsum01678 = _mm_add_ps(vsum018, vsum67);
      const __m128 vsum = _mm_add_ps(vsum2345, vsum01678);

      __m128 vout = _mm_mul_ps(vsum, vscale);
      vout = _mm_max_ps(vout, vmin);
      vout = _mm_min_ps(vout, vmax);

      _mm_storeu_ps(output, vout); output += 4;

      c -= 4;
    }
    if (c != 0) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      const __m128 vi1 = _mm_loadu_ps(i1);
      const __m128 vi2 = _mm_loadu_ps(i2);
      const __m128 vi3 = _mm_loadu_ps(i3);
      const __m128 vi4 = _mm_loadu_ps(i4);
      const __m128 vi5 = _mm_loadu_ps(i5);
      const __m128 vi6 = _mm_loadu_ps(i6);
      const __m128 vi7 = _mm_loadu_ps(i7);
      const __m128 vi8 = _mm_loadu_ps(i8);

      const __m128 vsum01 = _mm_add_ps(vi0, vi1);
      const __m128 vsum23 = _mm_add_ps(vi2, vi3);
      const __m128 vsum45 = _mm_add_ps(vi4, vi5);
      const __m128 vsum67 = _mm_add_ps(vi6, vi7);
      const __m128 vsum018 = _mm_add_ps(vsum01, vi8);
      const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
      const __m128 vsum01678 = _mm_add_ps(vsum018, vsum67);
      const __m128 vsum = _mm_add_ps(vsum2345, vsum01678);

      __m128 vout = _mm_mul_ps(vsum, vscale);
      vout = _mm_max_ps(vout, vmin);
      vout = _mm_min_ps(vout, vmax);

      if (c & 2) {
        _mm_storel_pi((__m64*) output, vout);
        vout = _mm_movehl_ps(vout, vout);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vout);
        output += 1;
      }
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_increment = round_down_po2(input_width, 4) * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * 4 - output_width * sizeof(float);

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  float* output0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);
  float* output1 = (float*) ((uintptr_t) output0 + output_height_stride);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 2) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    const size_t input_y4 = input_y2 + 2;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 > input_height) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 >= input_height) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(output_y + 2 > output_y_end) {
      output1 = output0;
    }

    const float* w = weights;
    size_t c = output_channels;
    float* o0c0 = output0;
    float* o1c0 = output1;
    float* o0c1 = (float*) ((uintptr_t) o0c0 + output_channel_stride);
    float* o1c1 = (float*) ((uintptr_t) o1c0 + output_channel_stride);
    float* o0c2 = (float*) ((uintptr_t) o0c1 + output_channel_stride);
    float* o1c2 = (float*) ((uintptr_t) o1c1 + output_channel_stride);
    float* o0c3 = (float*) ((uintptr_t) o0c2 + output_channel_stride);
    float* o1c3 = (float*) ((uintptr_t) o1c2 + output_channel_stride);
    do {
      if XNN_UNPREDICTABLE(c < 2) {
        o0c1 = o0c0;
        o1c1 = o1c0;
      }
      if XNN_UNPREDICTABLE(c <= 2) {
        o0c2 = o0c1;
        o1c2 = o1c1;
      }
      if XNN_UNPREDICTABLE(c < 4) {
        o0c3 = o0c2;
        o1c3 = o1c2;
      }

      // viMx0 = ( iM0c2, iM0c1, iM0c0, --- )
      __m128 vi0x0 = _mm_setzero_ps();
      __m128 vi1x0 = _mm_setzero_ps();
      __m128 vi2x0 = _mm_setzero_ps();
      __m128 vi3x0 = _mm_setzero_ps();
      __m128 vi4x0 = _mm_setzero_ps();

      size_t iw = input_width;
      for (; iw >= 4; iw -= 4) {
        __m128 vo0x0 = _mm_load_ps(w);
        __m128 vo1x0 = vo0x0;
        __m128 vo0x1 = vo0x0;
        __m128 vo1x1 = vo0x0;

        const __m128 vk00c0 = _mm_load_ps(w + 4);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        const __m128 vi0x1 = _mm_loadu_ps(i0); i0 += 4;
        const __m128 vi1x1 = _mm_loadu_ps(i1); i1 += 4;
        const __m128 vi2x1 = _mm_loadu_ps(i2); i2 += 4;
        const __m128 vi3x1 = _mm_loadu_ps(i3); i3 += 4;
        const __m128 vi4x1 = _mm_loadu_ps(i4); i4 += 4;

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk10c0 = _mm_load_ps(w + 8);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk20c0 = _mm_load_ps(w + 12);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk00c1 = _mm_load_ps(w + 16);

        // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
        const __m128 vi0x2 = _mm_loadu_ps(i0); i0 += 4;
        const __m128 vi1x2 = _mm_loadu_ps(i1); i1 += 4;
        const __m128 vi2x2 = _mm_loadu_ps(i2); i2 += 4;
        const __m128 vi3x2 = _mm_loadu_ps(i3); i3 += 4;
        const __m128 vi4x2 = _mm_loadu_ps(i4); i4 += 4;

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk10c1 = _mm_load_ps(w + 20);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk20c1 = _mm_load_ps(w + 24);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk00c2 = _mm_load_ps(w + 28);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk10c2 = _mm_load_ps(w + 32);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk20c2 = _mm_load_ps(w + 36);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk01c0 = _mm_load_ps(w + 40);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk11c0 = _mm_load_ps(w + 44);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk21c0 = _mm_load_ps(w + 48);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk01c1 = _mm_load_ps(w + 52);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk11c1 = _mm_load_ps(w + 56);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk21c1 = _mm_load_ps(w + 60);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk01c2 = _mm_load_ps(w + 64);

        // viMx3 = ( iM4c2, iM4c1, iM4c0, iM3c2 )
        const __m128 vi0x3 = _mm_loadu_ps(i0); i0 += 4;
        const __m128 vi1x3 = _mm_loadu_ps(i1); i1 += 4;
        const __m128 vi2x3 = _mm_loadu_ps(i2); i2 += 4;
        const __m128 vi3x3 = _mm_loadu_ps(i3); i3 += 4;
        const __m128 vi4x3 = _mm_loadu_ps(i4); i4 += 4;

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi0x3, vi0x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk11c2 = _mm_load_ps(w + 68);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi1x3, vi1x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi3x3, vi3x3, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk21c2 = _mm_load_ps(w + 72);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi4x3, vi4x3, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk02c0 = _mm_load_ps(w + 76);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi0x3, vi0x3, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk12c0 = _mm_load_ps(w + 80);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi1x3, vi1x3, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi3x3, vi3x3, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk22c0 = _mm_load_ps(w + 84);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi4x3, vi4x3, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk02c1 = _mm_load_ps(w + 88);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi0x3, vi0x3, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk12c1 = _mm_load_ps(w + 92);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi1x3, vi1x3, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi3x3, vi3x3, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk22c1 = _mm_load_ps(w + 96);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi4x3, vi4x3, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk02c2 = _mm_load_ps(w + 100);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi0x3, vi0x3, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk12c2 = _mm_load_ps(w + 104);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi1x3, vi1x3, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi3x3, vi3x3, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk22c2 = _mm_load_ps(w + 108);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi4x3, vi4x3, _MM_SHUFFLE(3, 3, 3, 3))));

        vi0x0 = vi0x3;
        vi1x0 = vi1x3;
        vi2x0 = vi2x3;
        vi3x0 = vi3x3;
        vi4x0 = vi4x3;

        vo0x0 = _mm_max_ps(vo0x0, vmin);
        vo1x0 = _mm_max_ps(vo1x0, vmin);
        vo0x1 = _mm_max_ps(vo0x1, vmin);
        vo1x1 = _mm_max_ps(vo1x1, vmin);

        vo0x0 = _mm_min_ps(vo0x0, vmax);
        vo1x0 = _mm_min_ps(vo1x0, vmax);
        vo0x1 = _mm_min_ps(vo0x1, vmax);
        vo1x1 = _mm_min_ps(vo1x1, vmax);

        const __m128 vo0c01 = _mm_unpacklo_ps(vo0x0, vo0x1);
        const __m128 vo0c23 = _mm_unpackhi_ps(vo0x0, vo0x1);
        const __m128 vo1c01 = _mm_unpacklo_ps(vo1x0, vo1x1);
        const __m128 vo1c23 = _mm_unpackhi_ps(vo1x0, vo1x1);

        // Always 2+ output width elements remaining
        _mm_storel_pi((__m64 *)o1c0, vo1c01); o1c0 += 2;
        _mm_storel_pi((__m64 *)o1c1, _mm_shuffle_ps(vo1c01, vo1c01, _MM_SHUFFLE(3, 2, 3, 2))); o1c1 += 2;
        _mm_storel_pi((__m64 *)o1c2, vo1c23); o1c2 += 2;
        _mm_storel_pi((__m64 *)o1c3, _mm_shuffle_ps(vo1c23, vo1c23, _MM_SHUFFLE(3, 2, 3, 2))); o1c3 += 2;

        _mm_storel_pi((__m64 *)o0c0, vo0c01); o0c0 += 2;
        _mm_storel_pi((__m64 *)o0c1, _mm_shuffle_ps(vo0c01, vo0c01, _MM_SHUFFLE(3, 2, 3, 2))); o0c1 += 2;
        _mm_storel_pi((__m64 *)o0c2, vo0c23); o0c2 += 2;
        _mm_storel_pi((__m64 *)o0c3, _mm_shuffle_ps(vo0c23, vo0c23, _MM_SHUFFLE(3, 2, 3, 2))); o0c3 += 2;
      }
      assert(iw < 4);
      if XNN_UNLIKELY(iw != 0) {
        __m128 vo0x0 = _mm_load_ps(w);
        __m128 vo1x0 = vo0x0;
        __m128 vo0x1 = vo0x0;
        __m128 vo1x1 = vo0x0;

        const __m128 vk00c0 = _mm_load_ps(w + 4);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        __m128 vi0x1 = _mm_loadu_ps(i0);
        __m128 vi1x1 = _mm_loadu_ps(i1);
        __m128 vi2x1 = _mm_loadu_ps(i2);
        __m128 vi3x1 = _mm_loadu_ps(i3);
        __m128 vi4x1 = _mm_loadu_ps(i4);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk10c0 = _mm_load_ps(w + 8);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk20c0 = _mm_load_ps(w + 12);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk00c1 = _mm_load_ps(w + 16);

        __m128 vi0x2 = _mm_setzero_ps();
        __m128 vi1x2 = _mm_setzero_ps();
        __m128 vi2x2 = _mm_setzero_ps();
        __m128 vi3x2 = _mm_setzero_ps();
        __m128 vi4x2 = _mm_setzero_ps();
        if (iw >= 2) {
          // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
          vi0x2 = _mm_loadu_ps(i0 + 4);
          vi1x2 = _mm_loadu_ps(i1 + 4);
          vi2x2 = _mm_loadu_ps(i2 + 4);
          vi3x2 = _mm_loadu_ps(i3 + 4);
          vi4x2 = _mm_loadu_ps(i4 + 4);
        }

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk10c1 = _mm_load_ps(w + 20);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk20c1 = _mm_load_ps(w + 24);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk00c2 = _mm_load_ps(w + 28);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk10c2 = _mm_load_ps(w + 32);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk20c2 = _mm_load_ps(w + 36);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk01c0 = _mm_load_ps(w + 40);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(0, 0, 0, 0))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(2, 2, 2, 2))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(2, 2, 2, 2))));
        }

        const __m128 vk11c0 = _mm_load_ps(w + 44);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(0, 0, 0, 0))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(2, 2, 2, 2))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(2, 2, 2, 2))));
        }

        const __m128 vk21c0 = _mm_load_ps(w + 48);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(0, 0, 0, 0))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(2, 2, 2, 2))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(2, 2, 2, 2))));
        }

        const __m128 vk01c1 = _mm_load_ps(w + 52);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk11c1 = _mm_load_ps(w + 56);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk21c1 = _mm_load_ps(w + 60);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk01c2 = _mm_load_ps(w + 64);

        __m128 vi0x3 = _mm_setzero_ps();
        __m128 vi1x3 = _mm_setzero_ps();
        __m128 vi2x3 = _mm_setzero_ps();
        __m128 vi3x3 = _mm_setzero_ps();
        __m128 vi4x3 = _mm_setzero_ps();
        if (iw > 2) {
          // viMx3 = ( 0.0, 0.0, 0.0, iM3c2 )
          vi0x3 = _mm_load_ss(i0 + 8);
          vi1x3 = _mm_load_ss(i1 + 8);
          vi2x3 = _mm_load_ss(i2 + 8);
          vi3x3 = _mm_load_ss(i3 + 8);
          vi4x3 = _mm_load_ss(i4 + 8);
        }

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi0x3, vi0x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk11c2 = _mm_load_ps(w + 68);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi1x3, vi1x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi3x3, vi3x3, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk21c2 = _mm_load_ps(w + 72);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi4x3, vi4x3, _MM_SHUFFLE(0, 0, 0, 0))));

        if (iw >= 2) {
          const __m128 vk02c0 = _mm_load_ps(w + 76);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));

          const __m128 vk12c0 = _mm_load_ps(w + 80);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(3, 3, 3, 3))));

          const __m128 vk22c0 = _mm_load_ps(w + 84);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(3, 3, 3, 3))));

          const __m128 vk02c1 = _mm_load_ps(w + 88);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(0, 0, 0, 0))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));

          const __m128 vk12c1 = _mm_load_ps(w + 92);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(0, 0, 0, 0))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(0, 0, 0, 0))));

          const __m128 vk22c1 = _mm_load_ps(w + 96);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(0, 0, 0, 0))));

          const __m128 vk02c2 = _mm_load_ps(w + 100);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(1, 1, 1, 1))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));

          const __m128 vk12c2 = _mm_load_ps(w + 104);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(1, 1, 1, 1))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(1, 1, 1, 1))));

          const __m128 vk22c2 = _mm_load_ps(w + 108);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(1, 1, 1, 1))));
        }

        vo0x0 = _mm_max_ps(vo0x0, vmin);
        vo1x0 = _mm_max_ps(vo1x0, vmin);
        vo0x1 = _mm_max_ps(vo0x1, vmin);
        vo1x1 = _mm_max_ps(vo1x1, vmin);

        vo0x0 = _mm_min_ps(vo0x0, vmax);
        vo1x0 = _mm_min_ps(vo1x0, vmax);
        vo0x1 = _mm_min_ps(vo0x1, vmax);
        vo1x1 = _mm_min_ps(vo1x1, vmax);

        if (iw == 3) {
          // Exactly 2 output width elements remaining
          const __m128 vo0c01 = _mm_unpacklo_ps(vo0x0, vo0x1);
          const __m128 vo0c23 = _mm_unpackhi_ps(vo0x0, vo0x1);
          const __m128 vo1c01 = _mm_unpacklo_ps(vo1x0, vo1x1);
          const __m128 vo1c23 = _mm_unpackhi_ps(vo1x0, vo1x1);

          _mm_storel_pi((__m64 *)o1c0, vo1c01); o1c0 += 2;
          _mm_storel_pi((__m64 *)o1c1, _mm_shuffle_ps(vo1c01, vo1c01, _MM_SHUFFLE(3, 2, 3, 2))); o1c1 += 2;
          _mm_storel_pi((__m64 *)o1c2, vo1c23); o1c2 += 2;
          _mm_storel_pi((__m64 *)o1c3, _mm_shuffle_ps(vo1c23, vo1c23, _MM_SHUFFLE(3, 2, 3, 2))); o1c3 += 2;

          _mm_storel_pi((__m64 *)o0c0, vo0c01); o0c0 += 2;
          _mm_storel_pi((__m64 *)o0c1, _mm_shuffle_ps(vo0c01, vo0c01, _MM_SHUFFLE(3, 2, 3, 2))); o0c1 += 2;
          _mm_storel_pi((__m64 *)o0c2, vo0c23); o0c2 += 2;
          _mm_storel_pi((__m64 *)o0c3, _mm_shuffle_ps(vo0c23, vo0c23, _MM_SHUFFLE(3, 2, 3, 2))); o0c3 += 2;
        } else {
          // Exactly 1 output width element remaining

          _mm_store_ss(o1c0, _mm_shuffle_ps(vo1x0, vo1x0, _MM_SHUFFLE(0, 0, 0, 0))); o1c0 += 1;
          _mm_store_ss(o1c1, _mm_shuffle_ps(vo1x0, vo1x0, _MM_SHUFFLE(1, 1, 1, 1))); o1c1 += 1;
          _mm_store_ss(o1c2, _mm_shuffle_ps(vo1x0, vo1x0, _MM_SHUFFLE(2, 2, 2, 2))); o1c2 += 1;
          _mm_store_ss(o1c3, _mm_shuffle_ps(vo1x0, vo1x0, _MM_SHUFFLE(3, 3, 3, 3))); o1c3 += 1;

          _mm_store_ss(o0c0, _mm_shuffle_ps(vo0x0, vo0x0, _MM_SHUFFLE(0, 0, 0, 0))); o0c0 += 1;
          _mm_store_ss(o0c1, _mm_shuffle_ps(vo0x0, vo0x0, _MM_SHUFFLE(1, 1, 1, 1))); o0c1 += 1;
          _mm_store_ss(o0c2, _mm_shuffle_ps(vo0x0, vo0x0, _MM_SHUFFLE(2, 2, 2, 2))); o0c2 += 1;
          _mm_store_ss(o0c3, _mm_shuffle_ps(vo0x0, vo0x0, _MM_SHUFFLE(3, 3, 3, 3))); o0c3 += 1;
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0c0 = (float*) ((uintptr_t) o0c0 + output_channel_increment);
      o0c1 = (float*) ((uintptr_t) o0c1 + output_channel_increment);
      o0c2 = (float*) ((uintptr_t) o0c2 + output_channel_increment);
      o0c3 = (float*) ((uintptr_t) o0c3 + output_channel_increment);
      o1c0 = (float*) ((uintptr_t) o1c0 + output_channel_increment);
      o1c1 = (float*) ((uintptr_t) o1c1 + output_channel_increment);
      o1c2 = (float*) ((uintptr_t) o1c2 + output_channel_increment);
      o1c3 = (float*) ((uintptr_t) o1c3 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_increment);
      i1 = (const float*) ((uintptr_t) i1 - input_width_increment);
      i2 = (const float*) ((uintptr_t) i2 - input_width_increment);
      i3 = (const float*) ((uintptr_t) i3 - input_width_increment);
      i4 = (const float*) ((uintptr_t) i4 - input_width_increment);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers forward to the next two rows
    output0 = (float*) ((uintptr_t) output1 + output_height_stride);
    output1 = (float*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next four rows
    i0 = i4;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
    i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  }
}

void xnn_f32_dwconv_minmax_ukernel_25p8c__sse(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
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
    for (; c >= 8; c -= 8) {
      __m128 vacc0123p0 = _mm_load_ps(w);
      __m128 vacc4567p0 = _mm_load_ps(w + 4);


      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
      i0 += 8;

      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      const __m128 vk0x4567 = _mm_load_ps(w + 12);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
      i1 += 8;

      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      const __m128 vk1x4567 = _mm_load_ps(w + 20);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      const __m128 vi2x4567 = _mm_loadu_ps(i2 + 4);
      i2 += 8;

      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      const __m128 vk2x4567 = _mm_load_ps(w + 28);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi2x4567, vk2x4567));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      const __m128 vi3x4567 = _mm_loadu_ps(i3 + 4);
      i3 += 8;

      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      const __m128 vk3x4567 = _mm_load_ps(w + 36);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi3x4567, vk3x4567));

      const __m128 vi4x0123 = _mm_loadu_ps(i4);
      const __m128 vi4x4567 = _mm_loadu_ps(i4 + 4);
      i4 += 8;

      const __m128 vk4x0123 = _mm_load_ps(w + 40);
      const __m128 vk4x4567 = _mm_load_ps(w + 44);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi4x4567, vk4x4567));

      const __m128 vi5x0123 = _mm_loadu_ps(i5);
      const __m128 vi5x4567 = _mm_loadu_ps(i5 + 4);
      i5 += 8;

      const __m128 vk5x0123 = _mm_load_ps(w + 48);
      const __m128 vk5x4567 = _mm_load_ps(w + 52);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi5x4567, vk5x4567));

      const __m128 vi6x0123 = _mm_loadu_ps(i6);
      const __m128 vi6x4567 = _mm_loadu_ps(i6 + 4);
      i6 += 8;

      const __m128 vk6x0123 = _mm_load_ps(w + 56);
      const __m128 vk6x4567 = _mm_load_ps(w + 60);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi6x4567, vk6x4567));

      const __m128 vi7x0123 = _mm_loadu_ps(i7);
      const __m128 vi7x4567 = _mm_loadu_ps(i7 + 4);
      i7 += 8;

      const __m128 vk7x0123 = _mm_load_ps(w + 64);
      const __m128 vk7x4567 = _mm_load_ps(w + 68);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi7x4567, vk7x4567));

      const __m128 vi8x0123 = _mm_loadu_ps(i8);
      const __m128 vi8x4567 = _mm_loadu_ps(i8 + 4);
      i8 += 8;

      const __m128 vk8x0123 = _mm_load_ps(w + 72);
      const __m128 vk8x4567 = _mm_load_ps(w + 76);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi8x4567, vk8x4567));

      const __m128 vi9x0123 = _mm_loadu_ps(i9);
      const __m128 vi9x4567 = _mm_loadu_ps(i9 + 4);
      i9 += 8;

      const __m128 vk9x0123 = _mm_load_ps(w + 80);
      const __m128 vk9x4567 = _mm_load_ps(w + 84);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi9x0123, vk9x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi9x4567, vk9x4567));

      const __m128 vi10x0123 = _mm_loadu_ps(i10);
      const __m128 vi10x4567 = _mm_loadu_ps(i10 + 4);
      i10 += 8;

      const __m128 vk10x0123 = _mm_load_ps(w + 88);
      const __m128 vk10x4567 = _mm_load_ps(w + 92);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi10x0123, vk10x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi10x4567, vk10x4567));

      const __m128 vi11x0123 = _mm_loadu_ps(i11);
      const __m128 vi11x4567 = _mm_loadu_ps(i11 + 4);
      i11 += 8;

      const __m128 vk11x0123 = _mm_load_ps(w + 96);
      const __m128 vk11x4567 = _mm_load_ps(w + 100);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi11x0123, vk11x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi11x4567, vk11x4567));

      const __m128 vi12x0123 = _mm_loadu_ps(i12);
      const __m128 vi12x4567 = _mm_loadu_ps(i12 + 4);
      i12 += 8;

      const __m128 vk12x0123 = _mm_load_ps(w + 104);
      const __m128 vk12x4567 = _mm_load_ps(w + 108);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi12x0123, vk12x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi12x4567, vk12x4567));

      const __m128 vi13x0123 = _mm_loadu_ps(i13);
      const __m128 vi13x4567 = _mm_loadu_ps(i13 + 4);
      i13 += 8;

      const __m128 vk13x0123 = _mm_load_ps(w + 112);
      const __m128 vk13x4567 = _mm_load_ps(w + 116);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi13x0123, vk13x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi13x4567, vk13x4567));

      const __m128 vi14x0123 = _mm_loadu_ps(i14);
      const __m128 vi14x4567 = _mm_loadu_ps(i14 + 4);
      i14 += 8;

      const __m128 vk14x0123 = _mm_load_ps(w + 120);
      const __m128 vk14x4567 = _mm_load_ps(w + 124);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi14x0123, vk14x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi14x4567, vk14x4567));

      const __m128 vi15x0123 = _mm_loadu_ps(i15);
      const __m128 vi15x4567 = _mm_loadu_ps(i15 + 4);
      i15 += 8;

      const __m128 vk15x0123 = _mm_load_ps(w + 128);
      const __m128 vk15x4567 = _mm_load_ps(w + 132);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi15x0123, vk15x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi15x4567, vk15x4567));

      const __m128 vi16x0123 = _mm_loadu_ps(i16);
      const __m128 vi16x4567 = _mm_loadu_ps(i16 + 4);
      i16 += 8;

      const __m128 vk16x0123 = _mm_load_ps(w + 136);
      const __m128 vk16x4567 = _mm_load_ps(w + 140);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi16x0123, vk16x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi16x4567, vk16x4567));

      const __m128 vi17x0123 = _mm_loadu_ps(i17);
      const __m128 vi17x4567 = _mm_loadu_ps(i17 + 4);
      i17 += 8;

      const __m128 vk17x0123 = _mm_load_ps(w + 144);
      const __m128 vk17x4567 = _mm_load_ps(w + 148);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi17x0123, vk17x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi17x4567, vk17x4567));

      const __m128 vi18x0123 = _mm_loadu_ps(i18);
      const __m128 vi18x4567 = _mm_loadu_ps(i18 + 4);
      i18 += 8;

      const __m128 vk18x0123 = _mm_load_ps(w + 152);
      const __m128 vk18x4567 = _mm_load_ps(w + 156);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi18x0123, vk18x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi18x4567, vk18x4567));

      const __m128 vi19x0123 = _mm_loadu_ps(i19);
      const __m128 vi19x4567 = _mm_loadu_ps(i19 + 4);
      i19 += 8;

      const __m128 vk19x0123 = _mm_load_ps(w + 160);
      const __m128 vk19x4567 = _mm_load_ps(w + 164);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi19x0123, vk19x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi19x4567, vk19x4567));

      const __m128 vi20x0123 = _mm_loadu_ps(i20);
      const __m128 vi20x4567 = _mm_loadu_ps(i20 + 4);
      i20 += 8;

      const __m128 vk20x0123 = _mm_load_ps(w + 168);
      const __m128 vk20x4567 = _mm_load_ps(w + 172);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi20x0123, vk20x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi20x4567, vk20x4567));

      const __m128 vi21x0123 = _mm_loadu_ps(i21);
      const __m128 vi21x4567 = _mm_loadu_ps(i21 + 4);
      i21 += 8;

      const __m128 vk21x0123 = _mm_load_ps(w + 176);
      const __m128 vk21x4567 = _mm_load_ps(w + 180);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi21x0123, vk21x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi21x4567, vk21x4567));

      const __m128 vi22x0123 = _mm_loadu_ps(i22);
      const __m128 vi22x4567 = _mm_loadu_ps(i22 + 4);
      i22 += 8;

      const __m128 vk22x0123 = _mm_load_ps(w + 184);
      const __m128 vk22x4567 = _mm_load_ps(w + 188);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi22x0123, vk22x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi22x4567, vk22x4567));

      const __m128 vi23x0123 = _mm_loadu_ps(i23);
      const __m128 vi23x4567 = _mm_loadu_ps(i23 + 4);
      i23 += 8;

      const __m128 vk23x0123 = _mm_load_ps(w + 192);
      const __m128 vk23x4567 = _mm_load_ps(w + 196);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi23x0123, vk23x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi23x4567, vk23x4567));

      const __m128 vi24x0123 = _mm_loadu_ps(i24);
      const __m128 vi24x4567 = _mm_loadu_ps(i24 + 4);
      i24 += 8;

      const __m128 vk24x0123 = _mm_load_ps(w + 200);
      const __m128 vk24x4567 = _mm_load_ps(w + 204);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi24x0123, vk24x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi24x4567, vk24x4567));

      w += 208;


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      __m128 vacc4567 = _mm_max_ps(vacc4567p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);
      vacc4567 = _mm_min_ps(vacc4567, vmax);

      _mm_storeu_ps(output, vacc0123);
      _mm_storeu_ps(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      __m128 vacc0123p0 = _mm_load_ps(w);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 += 4;

      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 += 4;

      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      i2 += 4;

      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      i3 += 4;

      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

      const __m128 vi4x0123 = _mm_loadu_ps(i4);
      i4 += 4;

      const __m128 vk4x0123 = _mm_load_ps(w + 40);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

      const __m128 vi5x0123 = _mm_loadu_ps(i5);
      i5 += 4;

      const __m128 vk5x0123 = _mm_load_ps(w + 48);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

      const __m128 vi6x0123 = _mm_loadu_ps(i6);
      i6 += 4;

      const __m128 vk6x0123 = _mm_load_ps(w + 56);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

      const __m128 vi7x0123 = _mm_loadu_ps(i7);
      i7 += 4;

      const __m128 vk7x0123 = _mm_load_ps(w + 64);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

      const __m128 vi8x0123 = _mm_loadu_ps(i8);
      i8 += 4;

      const __m128 vk8x0123 = _mm_load_ps(w + 72);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));

      const __m128 vi9x0123 = _mm_loadu_ps(i9);
      i9 += 4;

      const __m128 vk9x0123 = _mm_load_ps(w + 80);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi9x0123, vk9x0123));

      const __m128 vi10x0123 = _mm_loadu_ps(i10);
      i10 += 4;

      const __m128 vk10x0123 = _mm_load_ps(w + 88);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi10x0123, vk10x0123));

      const __m128 vi11x0123 = _mm_loadu_ps(i11);
      i11 += 4;

      const __m128 vk11x0123 = _mm_load_ps(w + 96);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi11x0123, vk11x0123));

      const __m128 vi12x0123 = _mm_loadu_ps(i12);
      i12 += 4;

      const __m128 vk12x0123 = _mm_load_ps(w + 104);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi12x0123, vk12x0123));

      const __m128 vi13x0123 = _mm_loadu_ps(i13);
      i13 += 4;

      const __m128 vk13x0123 = _mm_load_ps(w + 112);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi13x0123, vk13x0123));

      const __m128 vi14x0123 = _mm_loadu_ps(i14);
      i14 += 4;

      const __m128 vk14x0123 = _mm_load_ps(w + 120);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi14x0123, vk14x0123));

      const __m128 vi15x0123 = _mm_loadu_ps(i15);
      i15 += 4;

      const __m128 vk15x0123 = _mm_load_ps(w + 128);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi15x0123, vk15x0123));

      const __m128 vi16x0123 = _mm_loadu_ps(i16);
      i16 += 4;

      const __m128 vk16x0123 = _mm_load_ps(w + 136);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi16x0123, vk16x0123));

      const __m128 vi17x0123 = _mm_loadu_ps(i17);
      i17 += 4;

      const __m128 vk17x0123 = _mm_load_ps(w + 144);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi17x0123, vk17x0123));

      const __m128 vi18x0123 = _mm_loadu_ps(i18);
      i18 += 4;

      const __m128 vk18x0123 = _mm_load_ps(w + 152);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi18x0123, vk18x0123));

      const __m128 vi19x0123 = _mm_loadu_ps(i19);
      i19 += 4;

      const __m128 vk19x0123 = _mm_load_ps(w + 160);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi19x0123, vk19x0123));

      const __m128 vi20x0123 = _mm_loadu_ps(i20);
      i20 += 4;

      const __m128 vk20x0123 = _mm_load_ps(w + 168);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi20x0123, vk20x0123));

      const __m128 vi21x0123 = _mm_loadu_ps(i21);
      i21 += 4;

      const __m128 vk21x0123 = _mm_load_ps(w + 176);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi21x0123, vk21x0123));

      const __m128 vi22x0123 = _mm_loadu_ps(i22);
      i22 += 4;

      const __m128 vk22x0123 = _mm_load_ps(w + 184);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi22x0123, vk22x0123));

      const __m128 vi23x0123 = _mm_loadu_ps(i23);
      i23 += 4;

      const __m128 vk23x0123 = _mm_load_ps(w + 192);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi23x0123, vk23x0123));

      const __m128 vi24x0123 = _mm_loadu_ps(i24);
      i24 += 4;

      const __m128 vk24x0123 = _mm_load_ps(w + 200);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi24x0123, vk24x0123));

      w += 4;


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);

      _mm_storeu_ps(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128 vacc0123p0 = _mm_load_ps(w);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

      const __m128 vi4x0123 = _mm_loadu_ps(i4);
      const __m128 vk4x0123 = _mm_load_ps(w + 40);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

      const __m128 vi5x0123 = _mm_loadu_ps(i5);
      const __m128 vk5x0123 = _mm_load_ps(w + 48);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

      const __m128 vi6x0123 = _mm_loadu_ps(i6);
      const __m128 vk6x0123 = _mm_load_ps(w + 56);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

      const __m128 vi7x0123 = _mm_loadu_ps(i7);
      const __m128 vk7x0123 = _mm_load_ps(w + 64);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

      const __m128 vi8x0123 = _mm_loadu_ps(i8);
      const __m128 vk8x0123 = _mm_load_ps(w + 72);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));

      const __m128 vi9x0123 = _mm_loadu_ps(i9);
      const __m128 vk9x0123 = _mm_load_ps(w + 80);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi9x0123, vk9x0123));

      const __m128 vi10x0123 = _mm_loadu_ps(i10);
      const __m128 vk10x0123 = _mm_load_ps(w + 88);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi10x0123, vk10x0123));

      const __m128 vi11x0123 = _mm_loadu_ps(i11);
      const __m128 vk11x0123 = _mm_load_ps(w + 96);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi11x0123, vk11x0123));

      const __m128 vi12x0123 = _mm_loadu_ps(i12);
      const __m128 vk12x0123 = _mm_load_ps(w + 104);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi12x0123, vk12x0123));

      const __m128 vi13x0123 = _mm_loadu_ps(i13);
      const __m128 vk13x0123 = _mm_load_ps(w + 112);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi13x0123, vk13x0123));

      const __m128 vi14x0123 = _mm_loadu_ps(i14);
      const __m128 vk14x0123 = _mm_load_ps(w + 120);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi14x0123, vk14x0123));

      const __m128 vi15x0123 = _mm_loadu_ps(i15);
      const __m128 vk15x0123 = _mm_load_ps(w + 128);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi15x0123, vk15x0123));

      const __m128 vi16x0123 = _mm_loadu_ps(i16);
      const __m128 vk16x0123 = _mm_load_ps(w + 136);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi16x0123, vk16x0123));

      const __m128 vi17x0123 = _mm_loadu_ps(i17);
      const __m128 vk17x0123 = _mm_load_ps(w + 144);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi17x0123, vk17x0123));

      const __m128 vi18x0123 = _mm_loadu_ps(i18);
      const __m128 vk18x0123 = _mm_load_ps(w + 152);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi18x0123, vk18x0123));

      const __m128 vi19x0123 = _mm_loadu_ps(i19);
      const __m128 vk19x0123 = _mm_load_ps(w + 160);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi19x0123, vk19x0123));

      const __m128 vi20x0123 = _mm_loadu_ps(i20);
      const __m128 vk20x0123 = _mm_load_ps(w + 168);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi20x0123, vk20x0123));

      const __m128 vi21x0123 = _mm_loadu_ps(i21);
      const __m128 vk21x0123 = _mm_load_ps(w + 176);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi21x0123, vk21x0123));

      const __m128 vi22x0123 = _mm_loadu_ps(i22);
      const __m128 vk22x0123 = _mm_load_ps(w + 184);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi22x0123, vk22x0123));

      const __m128 vi23x0123 = _mm_loadu_ps(i23);
      const __m128 vk23x0123 = _mm_load_ps(w + 192);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi23x0123, vk23x0123));

      const __m128 vi24x0123 = _mm_loadu_ps(i24);
      const __m128 vk24x0123 = _mm_load_ps(w + 200);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi24x0123, vk24x0123));


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);

      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_3p8c__sse(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
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
    for (; c >= 8; c -= 8) {
      __m128 vacc0123p0 = _mm_load_ps(w);
      __m128 vacc4567p0 = _mm_load_ps(w + 4);


      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
      i0 += 8;

      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      const __m128 vk0x4567 = _mm_load_ps(w + 12);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
      i1 += 8;

      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      const __m128 vk1x4567 = _mm_load_ps(w + 20);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      const __m128 vi2x4567 = _mm_loadu_ps(i2 + 4);
      i2 += 8;

      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      const __m128 vk2x4567 = _mm_load_ps(w + 28);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi2x4567, vk2x4567));

      w += 32;


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      __m128 vacc4567 = _mm_max_ps(vacc4567p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);
      vacc4567 = _mm_min_ps(vacc4567, vmax);

      _mm_storeu_ps(output, vacc0123);
      _mm_storeu_ps(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      __m128 vacc0123p0 = _mm_load_ps(w);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 += 4;

      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 += 4;

      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      i2 += 4;

      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

      w += 4;


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);

      _mm_storeu_ps(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128 vacc0123p0 = _mm_load_ps(w);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);

      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_4p8c__sse(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
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
    for (; c >= 8; c -= 8) {
      __m128 vacc0123p0 = _mm_load_ps(w);
      __m128 vacc4567p0 = _mm_load_ps(w + 4);


      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
      i0 += 8;

      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      const __m128 vk0x4567 = _mm_load_ps(w + 12);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
      i1 += 8;

      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      const __m128 vk1x4567 = _mm_load_ps(w + 20);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      const __m128 vi2x4567 = _mm_loadu_ps(i2 + 4);
      i2 += 8;

      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      const __m128 vk2x4567 = _mm_load_ps(w + 28);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi2x4567, vk2x4567));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      const __m128 vi3x4567 = _mm_loadu_ps(i3 + 4);
      i3 += 8;

      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      const __m128 vk3x4567 = _mm_load_ps(w + 36);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi3x4567, vk3x4567));

      w += 40;


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      __m128 vacc4567 = _mm_max_ps(vacc4567p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);
      vacc4567 = _mm_min_ps(vacc4567, vmax);

      _mm_storeu_ps(output, vacc0123);
      _mm_storeu_ps(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      __m128 vacc0123p0 = _mm_load_ps(w);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 += 4;

      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 += 4;

      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      i2 += 4;

      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      i3 += 4;

      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

      w += 4;


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);

      _mm_storeu_ps(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128 vacc0123p0 = _mm_load_ps(w);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);

      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 8);

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* w = weights;

    // First pass to process 8 inputs.
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
      input += 8;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);
      for (; c >= 16; c -= 16) {
        __m128 vacc0123p0 = _mm_load_ps(w);
        __m128 vacc4567p0 = _mm_load_ps(w + 4);
        __m128 vacc89ABp0 = _mm_load_ps(w + 8);
        __m128 vaccCDEFp0 = _mm_load_ps(w + 12);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
        const __m128 vi0x89AB = _mm_loadu_ps(i0 + 8);
        const __m128 vi0xCDEF = _mm_loadu_ps(i0 + 12);
        i0 += 16;

        const __m128 vk0x0123 = _mm_load_ps(w + 16);
        const __m128 vk0x4567 = _mm_load_ps(w + 20);
        const __m128 vk0x89AB = _mm_load_ps(w + 24);
        const __m128 vk0xCDEF = _mm_load_ps(w + 28);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi0x89AB, vk0x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi0xCDEF, vk0xCDEF));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
        const __m128 vi1x89AB = _mm_loadu_ps(i1 + 8);
        const __m128 vi1xCDEF = _mm_loadu_ps(i1 + 12);
        i1 += 16;

        const __m128 vk1x0123 = _mm_load_ps(w + 32);
        const __m128 vk1x4567 = _mm_load_ps(w + 36);
        const __m128 vk1x89AB = _mm_load_ps(w + 40);
        const __m128 vk1xCDEF = _mm_load_ps(w + 44);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi1x89AB, vk1x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi1xCDEF, vk1xCDEF));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        const __m128 vi2x4567 = _mm_loadu_ps(i2 + 4);
        const __m128 vi2x89AB = _mm_loadu_ps(i2 + 8);
        const __m128 vi2xCDEF = _mm_loadu_ps(i2 + 12);
        i2 += 16;

        const __m128 vk2x0123 = _mm_load_ps(w + 48);
        const __m128 vk2x4567 = _mm_load_ps(w + 52);
        const __m128 vk2x89AB = _mm_load_ps(w + 56);
        const __m128 vk2xCDEF = _mm_load_ps(w + 60);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi2x4567, vk2x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi2x89AB, vk2x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi2xCDEF, vk2xCDEF));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        const __m128 vi3x4567 = _mm_loadu_ps(i3 + 4);
        const __m128 vi3x89AB = _mm_loadu_ps(i3 + 8);
        const __m128 vi3xCDEF = _mm_loadu_ps(i3 + 12);
        i3 += 16;

        const __m128 vk3x0123 = _mm_load_ps(w + 64);
        const __m128 vk3x4567 = _mm_load_ps(w + 68);
        const __m128 vk3x89AB = _mm_load_ps(w + 72);
        const __m128 vk3xCDEF = _mm_load_ps(w + 76);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi3x4567, vk3x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi3x89AB, vk3x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi3xCDEF, vk3xCDEF));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        const __m128 vi4x4567 = _mm_loadu_ps(i4 + 4);
        const __m128 vi4x89AB = _mm_loadu_ps(i4 + 8);
        const __m128 vi4xCDEF = _mm_loadu_ps(i4 + 12);
        i4 += 16;

        const __m128 vk4x0123 = _mm_load_ps(w + 80);
        const __m128 vk4x4567 = _mm_load_ps(w + 84);
        const __m128 vk4x89AB = _mm_load_ps(w + 88);
        const __m128 vk4xCDEF = _mm_load_ps(w + 92);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi4x4567, vk4x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi4x89AB, vk4x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi4xCDEF, vk4xCDEF));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        const __m128 vi5x4567 = _mm_loadu_ps(i5 + 4);
        const __m128 vi5x89AB = _mm_loadu_ps(i5 + 8);
        const __m128 vi5xCDEF = _mm_loadu_ps(i5 + 12);
        i5 += 16;

        const __m128 vk5x0123 = _mm_load_ps(w + 96);
        const __m128 vk5x4567 = _mm_load_ps(w + 100);
        const __m128 vk5x89AB = _mm_load_ps(w + 104);
        const __m128 vk5xCDEF = _mm_load_ps(w + 108);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi5x4567, vk5x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi5x89AB, vk5x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi5xCDEF, vk5xCDEF));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        const __m128 vi6x4567 = _mm_loadu_ps(i6 + 4);
        const __m128 vi6x89AB = _mm_loadu_ps(i6 + 8);
        const __m128 vi6xCDEF = _mm_loadu_ps(i6 + 12);
        i6 += 16;

        const __m128 vk6x0123 = _mm_load_ps(w + 112);
        const __m128 vk6x4567 = _mm_load_ps(w + 116);
        const __m128 vk6x89AB = _mm_load_ps(w + 120);
        const __m128 vk6xCDEF = _mm_load_ps(w + 124);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi6x4567, vk6x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi6x89AB, vk6x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi6xCDEF, vk6xCDEF));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        const __m128 vi7x4567 = _mm_loadu_ps(i7 + 4);
        const __m128 vi7x89AB = _mm_loadu_ps(i7 + 8);
        const __m128 vi7xCDEF = _mm_loadu_ps(i7 + 12);
        i7 += 16;

        const __m128 vk7x0123 = _mm_load_ps(w + 128);
        const __m128 vk7x4567 = _mm_load_ps(w + 132);
        const __m128 vk7x89AB = _mm_load_ps(w + 136);
        const __m128 vk7xCDEF = _mm_load_ps(w + 140);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi7x4567, vk7x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi7x89AB, vk7x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi7xCDEF, vk7xCDEF));

        w += 144;


        _mm_store_ps(b, vacc0123p0);
        _mm_store_ps(b + 4, vacc4567p0);
        _mm_store_ps(b + 8, vacc89ABp0);
        _mm_store_ps(b + 12, vaccCDEFp0);
        b += 16;
      }

      for (; c != 0; c -= 4) {
        __m128 vacc0123p0 = _mm_load_ps(w);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        i0 += 4;

        const __m128 vk0x0123 = _mm_load_ps(w + 4);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        i1 += 4;

        const __m128 vk1x0123 = _mm_load_ps(w + 8);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        i2 += 4;

        const __m128 vk2x0123 = _mm_load_ps(w + 12);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        i3 += 4;

        const __m128 vk3x0123 = _mm_load_ps(w + 16);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        i4 += 4;

        const __m128 vk4x0123 = _mm_load_ps(w + 20);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        i5 += 4;

        const __m128 vk5x0123 = _mm_load_ps(w + 24);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        i6 += 4;

        const __m128 vk6x0123 = _mm_load_ps(w + 28);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        i7 += 4;

        const __m128 vk7x0123 = _mm_load_ps(w + 32);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

        w += 36;


        _mm_store_ps(b, vacc0123p0);
        b += 4;
      }
    }

    // Middle pass to process 8 inputs in each iteration.
    for (size_t ks = kernel_size - 8; ks > 9; ks -= 8) {
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
      input += 8;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 16; c -= 16) {
        __m128 vacc0123p0 = _mm_load_ps(b);
        __m128 vacc4567p0 = _mm_load_ps(b + 4);
        __m128 vacc89ABp0 = _mm_load_ps(b + 8);
        __m128 vaccCDEFp0 = _mm_load_ps(b + 12);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
        const __m128 vi0x89AB = _mm_loadu_ps(i0 + 8);
        const __m128 vi0xCDEF = _mm_loadu_ps(i0 + 12);
        i0 += 16;

        const __m128 vk0x0123 = _mm_load_ps(w);
        const __m128 vk0x4567 = _mm_load_ps(w + 4);
        const __m128 vk0x89AB = _mm_load_ps(w + 8);
        const __m128 vk0xCDEF = _mm_load_ps(w + 12);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi0x89AB, vk0x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi0xCDEF, vk0xCDEF));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
        const __m128 vi1x89AB = _mm_loadu_ps(i1 + 8);
        const __m128 vi1xCDEF = _mm_loadu_ps(i1 + 12);
        i1 += 16;

        const __m128 vk1x0123 = _mm_load_ps(w + 16);
        const __m128 vk1x4567 = _mm_load_ps(w + 20);
        const __m128 vk1x89AB = _mm_load_ps(w + 24);
        const __m128 vk1xCDEF = _mm_load_ps(w + 28);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi1x89AB, vk1x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi1xCDEF, vk1xCDEF));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        const __m128 vi2x4567 = _mm_loadu_ps(i2 + 4);
        const __m128 vi2x89AB = _mm_loadu_ps(i2 + 8);
        const __m128 vi2xCDEF = _mm_loadu_ps(i2 + 12);
        i2 += 16;

        const __m128 vk2x0123 = _mm_load_ps(w + 32);
        const __m128 vk2x4567 = _mm_load_ps(w + 36);
        const __m128 vk2x89AB = _mm_load_ps(w + 40);
        const __m128 vk2xCDEF = _mm_load_ps(w + 44);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi2x4567, vk2x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi2x89AB, vk2x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi2xCDEF, vk2xCDEF));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        const __m128 vi3x4567 = _mm_loadu_ps(i3 + 4);
        const __m128 vi3x89AB = _mm_loadu_ps(i3 + 8);
        const __m128 vi3xCDEF = _mm_loadu_ps(i3 + 12);
        i3 += 16;

        const __m128 vk3x0123 = _mm_load_ps(w + 48);
        const __m128 vk3x4567 = _mm_load_ps(w + 52);
        const __m128 vk3x89AB = _mm_load_ps(w + 56);
        const __m128 vk3xCDEF = _mm_load_ps(w + 60);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi3x4567, vk3x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi3x89AB, vk3x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi3xCDEF, vk3xCDEF));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        const __m128 vi4x4567 = _mm_loadu_ps(i4 + 4);
        const __m128 vi4x89AB = _mm_loadu_ps(i4 + 8);
        const __m128 vi4xCDEF = _mm_loadu_ps(i4 + 12);
        i4 += 16;

        const __m128 vk4x0123 = _mm_load_ps(w + 64);
        const __m128 vk4x4567 = _mm_load_ps(w + 68);
        const __m128 vk4x89AB = _mm_load_ps(w + 72);
        const __m128 vk4xCDEF = _mm_load_ps(w + 76);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi4x4567, vk4x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi4x89AB, vk4x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi4xCDEF, vk4xCDEF));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        const __m128 vi5x4567 = _mm_loadu_ps(i5 + 4);
        const __m128 vi5x89AB = _mm_loadu_ps(i5 + 8);
        const __m128 vi5xCDEF = _mm_loadu_ps(i5 + 12);
        i5 += 16;

        const __m128 vk5x0123 = _mm_load_ps(w + 80);
        const __m128 vk5x4567 = _mm_load_ps(w + 84);
        const __m128 vk5x89AB = _mm_load_ps(w + 88);
        const __m128 vk5xCDEF = _mm_load_ps(w + 92);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi5x4567, vk5x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi5x89AB, vk5x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi5xCDEF, vk5xCDEF));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        const __m128 vi6x4567 = _mm_loadu_ps(i6 + 4);
        const __m128 vi6x89AB = _mm_loadu_ps(i6 + 8);
        const __m128 vi6xCDEF = _mm_loadu_ps(i6 + 12);
        i6 += 16;

        const __m128 vk6x0123 = _mm_load_ps(w + 96);
        const __m128 vk6x4567 = _mm_load_ps(w + 100);
        const __m128 vk6x89AB = _mm_load_ps(w + 104);
        const __m128 vk6xCDEF = _mm_load_ps(w + 108);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi6x4567, vk6x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi6x89AB, vk6x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi6xCDEF, vk6xCDEF));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        const __m128 vi7x4567 = _mm_loadu_ps(i7 + 4);
        const __m128 vi7x89AB = _mm_loadu_ps(i7 + 8);
        const __m128 vi7xCDEF = _mm_loadu_ps(i7 + 12);
        i7 += 16;

        const __m128 vk7x0123 = _mm_load_ps(w + 112);
        const __m128 vk7x4567 = _mm_load_ps(w + 116);
        const __m128 vk7x89AB = _mm_load_ps(w + 120);
        const __m128 vk7xCDEF = _mm_load_ps(w + 124);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi7x4567, vk7x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi7x89AB, vk7x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi7xCDEF, vk7xCDEF));

        w += 128;


        _mm_store_ps(b, vacc0123p0);
        _mm_store_ps(b + 4, vacc4567p0);
        _mm_store_ps(b + 8, vacc89ABp0);
        _mm_store_ps(b + 12, vaccCDEFp0);
        b += 16;
      }

      for (; c != 0; c -= 4) {
        __m128 vacc0123p0 = _mm_load_ps(b);


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        i0 += 4;

        const __m128 vk0x0123 = _mm_load_ps(w);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        i1 += 4;

        const __m128 vk1x0123 = _mm_load_ps(w + 4);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        i2 += 4;

        const __m128 vk2x0123 = _mm_load_ps(w + 8);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        i3 += 4;

        const __m128 vk3x0123 = _mm_load_ps(w + 12);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        i4 += 4;

        const __m128 vk4x0123 = _mm_load_ps(w + 16);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        i5 += 4;

        const __m128 vk5x0123 = _mm_load_ps(w + 20);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        i6 += 4;

        const __m128 vk6x0123 = _mm_load_ps(w + 24);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        i7 += 4;

        const __m128 vk7x0123 = _mm_load_ps(w + 28);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

        w += 32;


        _mm_store_ps(b, vacc0123p0);
        b += 4;
      }
    }

    // Last pass to process up to 9 inputs.
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

      size_t c = channels;
      for (; c >= 16; c -= 16) {
        __m128 vacc0123p0 = _mm_load_ps(b);
        __m128 vacc4567p0 = _mm_load_ps(b + 4);
        __m128 vacc89ABp0 = _mm_load_ps(b + 8);
        __m128 vaccCDEFp0 = _mm_load_ps(b + 12);
        b += 16;


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
        const __m128 vi0x89AB = _mm_loadu_ps(i0 + 8);
        const __m128 vi0xCDEF = _mm_loadu_ps(i0 + 12);
        i0 += 16;

        __m128 vk0x0123 = _mm_load_ps(w);
        __m128 vk0x4567 = _mm_load_ps(w + 4);
        __m128 vk0x89AB = _mm_load_ps(w + 8);
        __m128 vk0xCDEF = _mm_load_ps(w + 12);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi0x89AB, vk0x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi0xCDEF, vk0xCDEF));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
        const __m128 vi1x89AB = _mm_loadu_ps(i1 + 8);
        const __m128 vi1xCDEF = _mm_loadu_ps(i1 + 12);
        i1 += 16;

        __m128 vk1x0123 = _mm_load_ps(w + 16);
        __m128 vk1x4567 = _mm_load_ps(w + 20);
        __m128 vk1x89AB = _mm_load_ps(w + 24);
        __m128 vk1xCDEF = _mm_load_ps(w + 28);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi1x89AB, vk1x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi1xCDEF, vk1xCDEF));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        const __m128 vi2x4567 = _mm_loadu_ps(i2 + 4);
        const __m128 vi2x89AB = _mm_loadu_ps(i2 + 8);
        const __m128 vi2xCDEF = _mm_loadu_ps(i2 + 12);
        i2 += 16;

        __m128 vk2x0123 = _mm_load_ps(w + 32);
        __m128 vk2x4567 = _mm_load_ps(w + 36);
        __m128 vk2x89AB = _mm_load_ps(w + 40);
        __m128 vk2xCDEF = _mm_load_ps(w + 44);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi2x4567, vk2x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi2x89AB, vk2x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi2xCDEF, vk2xCDEF));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        const __m128 vi3x4567 = _mm_loadu_ps(i3 + 4);
        const __m128 vi3x89AB = _mm_loadu_ps(i3 + 8);
        const __m128 vi3xCDEF = _mm_loadu_ps(i3 + 12);
        i3 += 16;

        __m128 vk3x0123 = _mm_load_ps(w + 48);
        __m128 vk3x4567 = _mm_load_ps(w + 52);
        __m128 vk3x89AB = _mm_load_ps(w + 56);
        __m128 vk3xCDEF = _mm_load_ps(w + 60);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi3x4567, vk3x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi3x89AB, vk3x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi3xCDEF, vk3xCDEF));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        const __m128 vi4x4567 = _mm_loadu_ps(i4 + 4);
        const __m128 vi4x89AB = _mm_loadu_ps(i4 + 8);
        const __m128 vi4xCDEF = _mm_loadu_ps(i4 + 12);
        i4 += 16;

        __m128 vk4x0123 = _mm_load_ps(w + 64);
        __m128 vk4x4567 = _mm_load_ps(w + 68);
        __m128 vk4x89AB = _mm_load_ps(w + 72);
        __m128 vk4xCDEF = _mm_load_ps(w + 76);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi4x4567, vk4x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi4x89AB, vk4x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi4xCDEF, vk4xCDEF));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        const __m128 vi5x4567 = _mm_loadu_ps(i5 + 4);
        const __m128 vi5x89AB = _mm_loadu_ps(i5 + 8);
        const __m128 vi5xCDEF = _mm_loadu_ps(i5 + 12);
        i5 += 16;

        __m128 vk5x0123 = _mm_load_ps(w + 80);
        __m128 vk5x4567 = _mm_load_ps(w + 84);
        __m128 vk5x89AB = _mm_load_ps(w + 88);
        __m128 vk5xCDEF = _mm_load_ps(w + 92);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi5x4567, vk5x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi5x89AB, vk5x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi5xCDEF, vk5xCDEF));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        const __m128 vi6x4567 = _mm_loadu_ps(i6 + 4);
        const __m128 vi6x89AB = _mm_loadu_ps(i6 + 8);
        const __m128 vi6xCDEF = _mm_loadu_ps(i6 + 12);
        i6 += 16;

        __m128 vk6x0123 = _mm_load_ps(w + 96);
        __m128 vk6x4567 = _mm_load_ps(w + 100);
        __m128 vk6x89AB = _mm_load_ps(w + 104);
        __m128 vk6xCDEF = _mm_load_ps(w + 108);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi6x4567, vk6x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi6x89AB, vk6x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi6xCDEF, vk6xCDEF));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        const __m128 vi7x4567 = _mm_loadu_ps(i7 + 4);
        const __m128 vi7x89AB = _mm_loadu_ps(i7 + 8);
        const __m128 vi7xCDEF = _mm_loadu_ps(i7 + 12);
        i7 += 16;

        __m128 vk7x0123 = _mm_load_ps(w + 112);
        __m128 vk7x4567 = _mm_load_ps(w + 116);
        __m128 vk7x89AB = _mm_load_ps(w + 120);
        __m128 vk7xCDEF = _mm_load_ps(w + 124);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi7x4567, vk7x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi7x89AB, vk7x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi7xCDEF, vk7xCDEF));

        const __m128 vi8x0123 = _mm_loadu_ps(i8);
        const __m128 vi8x4567 = _mm_loadu_ps(i8 + 4);
        const __m128 vi8x89AB = _mm_loadu_ps(i8 + 8);
        const __m128 vi8xCDEF = _mm_loadu_ps(i8 + 12);
        i8 += 16;

        __m128 vk8x0123 = _mm_load_ps(w + 128);
        __m128 vk8x4567 = _mm_load_ps(w + 132);
        __m128 vk8x89AB = _mm_load_ps(w + 136);
        __m128 vk8xCDEF = _mm_load_ps(w + 140);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));
        vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi8x4567, vk8x4567));
        vacc89ABp0 = _mm_add_ps(vacc89ABp0, _mm_mul_ps(vi8x89AB, vk8x89AB));
        vaccCDEFp0 = _mm_add_ps(vaccCDEFp0, _mm_mul_ps(vi8xCDEF, vk8xCDEF));

        w += 144;


        __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
        __m128 vacc4567 = _mm_max_ps(vacc4567p0, vmin);
        __m128 vacc89AB = _mm_max_ps(vacc89ABp0, vmin);
        __m128 vaccCDEF = _mm_max_ps(vaccCDEFp0, vmin);

        vacc0123 = _mm_min_ps(vacc0123, vmax);
        vacc4567 = _mm_min_ps(vacc4567, vmax);
        vacc89AB = _mm_min_ps(vacc89AB, vmax);
        vaccCDEF = _mm_min_ps(vaccCDEF, vmax);

        _mm_storeu_ps(output, vacc0123);
        _mm_storeu_ps(output + 4, vacc4567);
        _mm_storeu_ps(output + 8, vacc89AB);
        _mm_storeu_ps(output + 12, vaccCDEF);
        output += 16;
      }


      for (; c >= 4; c -= 4) {
        __m128 vacc0123p0 = _mm_load_ps(b);
        b += 4;


        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        i0 += 4;

        __m128 vk0x0123 = _mm_load_ps(w);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        i1 += 4;

        __m128 vk1x0123 = _mm_load_ps(w + 4);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        i2 += 4;

        __m128 vk2x0123 = _mm_load_ps(w + 8);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        i3 += 4;

        __m128 vk3x0123 = _mm_load_ps(w + 12);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        i4 += 4;

        __m128 vk4x0123 = _mm_load_ps(w + 16);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        i5 += 4;

        __m128 vk5x0123 = _mm_load_ps(w + 20);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        i6 += 4;

        __m128 vk6x0123 = _mm_load_ps(w + 24);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        i7 += 4;

        __m128 vk7x0123 = _mm_load_ps(w + 28);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

        const __m128 vi8x0123 = _mm_loadu_ps(i8);
        i8 += 4;

        __m128 vk8x0123 = _mm_load_ps(w + 32);

        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));

        w += 36;



        __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);

        vacc0123 = _mm_min_ps(vacc0123, vmax);

        _mm_storeu_ps(output, vacc0123);
        output += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        __m128 vacc0123p0 = _mm_load_ps(b);

        const __m128 vi0x0123 = _mm_loadu_ps(i0);
        __m128 vk0x0123 = _mm_load_ps(w);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

        const __m128 vi1x0123 = _mm_loadu_ps(i1);
        __m128 vk1x0123 = _mm_load_ps(w + 4);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

        const __m128 vi2x0123 = _mm_loadu_ps(i2);
        __m128 vk2x0123 = _mm_load_ps(w + 8);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

        const __m128 vi3x0123 = _mm_loadu_ps(i3);
        __m128 vk3x0123 = _mm_load_ps(w + 12);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

        const __m128 vi4x0123 = _mm_loadu_ps(i4);
        __m128 vk4x0123 = _mm_load_ps(w + 16);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

        const __m128 vi5x0123 = _mm_loadu_ps(i5);
        __m128 vk5x0123 = _mm_load_ps(w + 20);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

        const __m128 vi6x0123 = _mm_loadu_ps(i6);
        __m128 vk6x0123 = _mm_load_ps(w + 24);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

        const __m128 vi7x0123 = _mm_loadu_ps(i7);
        __m128 vk7x0123 = _mm_load_ps(w + 28);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

        const __m128 vi8x0123 = _mm_loadu_ps(i8);
        __m128 vk8x0123 = _mm_load_ps(w + 32);
        vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));


        __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
        vacc0123 = _mm_min_ps(vacc0123, vmax);

        if (c & 2) {
          _mm_storel_pi((__m64*) output, vacc0123);
          vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
          output += 2;
        }
        if (c & 1) {
          _mm_store_ss(output, vacc0123);
          output += 1;
        }
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_9p8c__sse(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
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
    for (; c >= 8; c -= 8) {
      __m128 vacc0123p0 = _mm_load_ps(w);
      __m128 vacc4567p0 = _mm_load_ps(w + 4);


      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
      i0 += 8;

      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      const __m128 vk0x4567 = _mm_load_ps(w + 12);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi0x4567, vk0x4567));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
      i1 += 8;

      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      const __m128 vk1x4567 = _mm_load_ps(w + 20);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi1x4567, vk1x4567));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      const __m128 vi2x4567 = _mm_loadu_ps(i2 + 4);
      i2 += 8;

      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      const __m128 vk2x4567 = _mm_load_ps(w + 28);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi2x4567, vk2x4567));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      const __m128 vi3x4567 = _mm_loadu_ps(i3 + 4);
      i3 += 8;

      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      const __m128 vk3x4567 = _mm_load_ps(w + 36);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi3x4567, vk3x4567));

      const __m128 vi4x0123 = _mm_loadu_ps(i4);
      const __m128 vi4x4567 = _mm_loadu_ps(i4 + 4);
      i4 += 8;

      const __m128 vk4x0123 = _mm_load_ps(w + 40);
      const __m128 vk4x4567 = _mm_load_ps(w + 44);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi4x4567, vk4x4567));

      const __m128 vi5x0123 = _mm_loadu_ps(i5);
      const __m128 vi5x4567 = _mm_loadu_ps(i5 + 4);
      i5 += 8;

      const __m128 vk5x0123 = _mm_load_ps(w + 48);
      const __m128 vk5x4567 = _mm_load_ps(w + 52);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi5x4567, vk5x4567));

      const __m128 vi6x0123 = _mm_loadu_ps(i6);
      const __m128 vi6x4567 = _mm_loadu_ps(i6 + 4);
      i6 += 8;

      const __m128 vk6x0123 = _mm_load_ps(w + 56);
      const __m128 vk6x4567 = _mm_load_ps(w + 60);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi6x4567, vk6x4567));

      const __m128 vi7x0123 = _mm_loadu_ps(i7);
      const __m128 vi7x4567 = _mm_loadu_ps(i7 + 4);
      i7 += 8;

      const __m128 vk7x0123 = _mm_load_ps(w + 64);
      const __m128 vk7x4567 = _mm_load_ps(w + 68);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi7x4567, vk7x4567));

      const __m128 vi8x0123 = _mm_loadu_ps(i8);
      const __m128 vi8x4567 = _mm_loadu_ps(i8 + 4);
      i8 += 8;

      const __m128 vk8x0123 = _mm_load_ps(w + 72);
      const __m128 vk8x4567 = _mm_load_ps(w + 76);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));
      vacc4567p0 = _mm_add_ps(vacc4567p0, _mm_mul_ps(vi8x4567, vk8x4567));

      w += 80;


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      __m128 vacc4567 = _mm_max_ps(vacc4567p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);
      vacc4567 = _mm_min_ps(vacc4567, vmax);

      _mm_storeu_ps(output, vacc0123);
      _mm_storeu_ps(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      __m128 vacc0123p0 = _mm_load_ps(w);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 += 4;

      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 += 4;

      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      i2 += 4;

      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      i3 += 4;

      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

      const __m128 vi4x0123 = _mm_loadu_ps(i4);
      i4 += 4;

      const __m128 vk4x0123 = _mm_load_ps(w + 40);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

      const __m128 vi5x0123 = _mm_loadu_ps(i5);
      i5 += 4;

      const __m128 vk5x0123 = _mm_load_ps(w + 48);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

      const __m128 vi6x0123 = _mm_loadu_ps(i6);
      i6 += 4;

      const __m128 vk6x0123 = _mm_load_ps(w + 56);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

      const __m128 vi7x0123 = _mm_loadu_ps(i7);
      i7 += 4;

      const __m128 vk7x0123 = _mm_load_ps(w + 64);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

      const __m128 vi8x0123 = _mm_loadu_ps(i8);
      i8 += 4;

      const __m128 vk8x0123 = _mm_load_ps(w + 72);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));

      w += 4;


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);

      _mm_storeu_ps(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128 vacc0123p0 = _mm_load_ps(w);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vk0x0123 = _mm_load_ps(w + 8);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi0x0123, vk0x0123));

      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vk1x0123 = _mm_load_ps(w + 16);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi1x0123, vk1x0123));

      const __m128 vi2x0123 = _mm_loadu_ps(i2);
      const __m128 vk2x0123 = _mm_load_ps(w + 24);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi2x0123, vk2x0123));

      const __m128 vi3x0123 = _mm_loadu_ps(i3);
      const __m128 vk3x0123 = _mm_load_ps(w + 32);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi3x0123, vk3x0123));

      const __m128 vi4x0123 = _mm_loadu_ps(i4);
      const __m128 vk4x0123 = _mm_load_ps(w + 40);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi4x0123, vk4x0123));

      const __m128 vi5x0123 = _mm_loadu_ps(i5);
      const __m128 vk5x0123 = _mm_load_ps(w + 48);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi5x0123, vk5x0123));

      const __m128 vi6x0123 = _mm_loadu_ps(i6);
      const __m128 vk6x0123 = _mm_load_ps(w + 56);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi6x0123, vk6x0123));

      const __m128 vi7x0123 = _mm_loadu_ps(i7);
      const __m128 vk7x0123 = _mm_load_ps(w + 64);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi7x0123, vk7x0123));

      const __m128 vi8x0123 = _mm_loadu_ps(i8);
      const __m128 vk8x0123 = _mm_load_ps(w + 72);
      vacc0123p0 = _mm_add_ps(vacc0123p0, _mm_mul_ps(vi8x0123, vk8x0123));


      __m128 vacc0123 = _mm_max_ps(vacc0123p0, vmin);
      vacc0123 = _mm_min_ps(vacc0123, vmax);

      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_2x4_acc2(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 1);

  const __m128 vmask = _mm_load_ps((const float*) params->sse_stride1.mask);
  const __m128 vmax = _mm_set1_ps(params->sse_stride1.max);
  const __m128 vmin = _mm_set1_ps(params->sse_stride1.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128 vbias = _mm_load1_ps(weights);
  const __m128 vk00 = _mm_load1_ps(weights + 1);
  const __m128 vk01 = _mm_load1_ps(weights + 2);
  const __m128 vk02 = _mm_load1_ps(weights + 3);
  const __m128 vk10 = _mm_load1_ps(weights + 4);
  const __m128 vk11 = _mm_load1_ps(weights + 5);
  const __m128 vk12 = _mm_load1_ps(weights + 6);
  const __m128 vk20 = _mm_load1_ps(weights + 7);
  const __m128 vk21 = _mm_load1_ps(weights + 8);
  const __m128 vk22 = _mm_load1_ps(weights + 9);

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i3 = zero;
    }

    // vi0x3012 = ( vi02, vi01, vi{M}0, vi{M}3 )
    __m128 vi0x3012 = _mm_setzero_ps();
    // vi1x3012 = ( vi12, vi11, vi{M}0, vi{M}3 )
    __m128 vi1x3012 = _mm_setzero_ps();
    // vi2x3012 = ( vi22, vi21, vi{M}0, vi{M}3 )
    __m128 vi2x3012 = _mm_setzero_ps();
    // vi3x3012 = ( vi32, vi31, vi{M}0, vi{M}3 )
    __m128 vi3x3012 = _mm_setzero_ps();

    __m128 vi0x4567 = _mm_loadu_ps(i0);
    i0 += 4;
    __m128 vi1x4567 = _mm_loadu_ps(i1);
    i1 += 4;
    __m128 vi2x4567 = _mm_loadu_ps(i2);
    i2 += 4;
    __m128 vi3x4567 = _mm_loadu_ps(i3);
    i3 += 4;

    size_t w = input_width;
    for (; w > 4 * sizeof(float); w -= 4 * sizeof(float)) {
      // vi0x89AB = ( vi0B, vi0A, vi09, vi08 )
      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      i0 += 4;
      // vi1x89AB = ( vi1B, vi1A, vi19, vi18 )
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      i1 += 4;
      // vi2x89AB = ( vi2B, vi2A, vi29, vi28 )
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      i2 += 4;
      // vi3x89AB = ( vi3B, vi3A, vi39, vi38 )
      const __m128 vi3x89AB = _mm_loadu_ps(i3);
      i3 += 4;

      // vi0x7456 = ( vi06, vi05, vi04, vi07 )
      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi1x7456 = ( vi16, vi15, vi14, vi17 )
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi2x7456 = ( vi26, vi25, vi24, vi27 )
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi3x7456 = ( vi36, vi35, vi34, vi37 )
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk01));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk01));
      __m128 vo0p1 = _mm_mul_ps(vi1x4567, vk11);
      __m128 vo1p1 = _mm_mul_ps(vi2x4567, vk11);
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk21));

      // vi0x3456 = ( vi06, vi05, vi04, vi03 )
      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      // vi1x3456 = ( vi16, vi15, vi14, vi13 )
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      // vi2x3456 = ( vi26, vi25, vi24, vi23 )
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      // vi3x3456 = ( vi36, vi35, vi34, vi33 )
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);

      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi0x3456, vk00));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi1x3456, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk10));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi2x3456, vk20));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi3x3456, vk20));

      vi0x3012 = vi0x7456;
      vi1x3012 = vi1x7456;
      vi2x3012 = vi2x7456;
      vi3x3012 = vi3x7456;

      // vi0x8567 = ( vi07, vi06, vi05, vi08 )
      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vi0x89AB);
      // vi1x8567 = ( vi17, vi16, vi15, vi18 )
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vi1x89AB);
      // vi2x8567 = ( vi27, vi26, vi25, vi28 )
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vi2x89AB);
      // vi3x8567 = ( vi37, vi36, vi35, vi38 )
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vi3x89AB);

      // vi0x5678 = ( vi08, vi07, vi06, vi05 )
      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi1x5678 = ( vi18, vi17, vi16, vi15 )
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi2x5678 = ( vi28, vi27, vi26, vi25 )
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi3x5678 = ( vi38, vi37, vi36, vi35 )
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk02));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk02));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x5678, vk12));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi2x5678, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk22));

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;

      vo0p0 = _mm_add_ps(vo0p0, vo0p1);
      vo1p0 = _mm_add_ps(vo1p0, vo1p1);

      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);

      _mm_storeu_ps(o1, vo1);
      o1 += 4;
      _mm_storeu_ps(o0, vo0);
      o0 += 4;
    }
    // Always process the last block of 1..4 pixels.
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      vi0x4567 = _mm_and_ps(vmask, vi0x4567);
      vi1x4567 = _mm_and_ps(vmask, vi1x4567);
      vi2x4567 = _mm_and_ps(vmask, vi2x4567);
      vi3x4567 = _mm_and_ps(vmask, vi3x4567);

      // vi0x7456 = ( vi06, vi05, vi04, vi07 )
      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi1x7456 = ( vi16, vi15, vi14, vi17 )
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi2x7456 = ( vi26, vi25, vi24, vi27 )
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi3x7456 = ( vi36, vi35, vi34, vi37 )
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk01));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk01));
      __m128 vo0p1 = _mm_mul_ps(vi1x4567, vk11);
      __m128 vo1p1 = _mm_mul_ps(vi2x4567, vk11);
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk21));

      // vi0x3456 = ( vi06, vi05, vi04, vi03 )
      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      // vi1x3456 = ( vi16, vi15, vi14, vi13 )
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      // vi2x3456 = ( vi26, vi25, vi24, vi23 )
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      // vi3x3456 = ( vi36, vi35, vi34, vi33 )
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);

      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi0x3456, vk00));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi1x3456, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk10));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi2x3456, vk20));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi3x3456, vk20));

      const __m128 vzero = _mm_setzero_ps();
      // vi0x8567 = ( vi07, vi06, vi05, 0.0 )
      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vzero);
      // vi1x8567 = ( vi17, vi16, vi15, 0.0 )
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vzero);
      // vi2x8567 = ( vi27, vi26, vi25, 0.0 )
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vzero);
      // vi3x8567 = ( vi37, vi36, vi35, 0.0 )
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vzero);

      // vi0x5678 = ( vi08, vi07, vi06, vi05 )
      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi1x5678 = ( vi18, vi17, vi16, vi15 )
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi2x5678 = ( vi28, vi27, vi26, vi25 )
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi3x5678 = ( vi38, vi37, vi36, vi35 )
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk02));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk02));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x5678, vk12));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi2x5678, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk22));

      vo0p0 = _mm_add_ps(vo0p0, vo0p1);
      vo1p0 = _mm_add_ps(vo1p0, vo1p1);

      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);

      if XNN_LIKELY(w == 4 * sizeof(float)) {
        _mm_storeu_ps(o1, vo1);
        o1 += 4;
        _mm_storeu_ps(o0, vo0);
        o0 += 4;
      } else {
        if (w & (2 * sizeof(float))) {
          _mm_storel_pi((__m64*) o1, vo1);
          o1 += 2;
          _mm_storel_pi((__m64*) o0, vo0);
          o0 += 2;

          vo0 = _mm_movehl_ps(vo0, vo0);
          vo1 = _mm_movehl_ps(vo1, vo1);
        }
        if (w & (1 * sizeof(float))) {
          _mm_store_ss(o1, vo1);
          o1 += 1;
          _mm_store_ss(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_decrement);
    i1 = (const float*) ((uintptr_t) i3 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);

    o0 = o1;
    o1 = (float*) ((uintptr_t) o0 + input_width);

    output_height = doz(output_height, 2);
  } while (output_height != 0);
}

void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4_acc3(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top >= 0);
  assert(padding_top <= 1);

  const __m128 vmask_even = _mm_load_ps((const float*) params->sse_stride2.mask_even);
  const __m128 vmask_odd  = _mm_load_ps((const float*) params->sse_stride2.mask_odd);
  const __m128 vmax = _mm_set1_ps(params->sse_stride2.max);
  const __m128 vmin = _mm_set1_ps(params->sse_stride2.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128 vbias = _mm_load1_ps(weights);
  const __m128 vk00 = _mm_load1_ps(weights + 1);
  const __m128 vk01 = _mm_load1_ps(weights + 2);
  const __m128 vk02 = _mm_load1_ps(weights + 3);
  const __m128 vk10 = _mm_load1_ps(weights + 4);
  const __m128 vk11 = _mm_load1_ps(weights + 5);
  const __m128 vk12 = _mm_load1_ps(weights + 6);
  const __m128 vk20 = _mm_load1_ps(weights + 7);
  const __m128 vk21 = _mm_load1_ps(weights + 8);
  const __m128 vk22 = _mm_load1_ps(weights + 9);

  const size_t input_decrement = round_down_po2(input_width, 4 /* SIMD output width */ * 2 /* subsampling */ * sizeof(float));

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

    __m128 vi0x7531 = _mm_setzero_ps();
    __m128 vi1x7531 = _mm_setzero_ps();
    __m128 vi2x7531 = _mm_setzero_ps();

    size_t w = input_width;
    for (; w >= 8 * sizeof(float); w -= 8 * sizeof(float)) {
      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      const __m128 vi0xCDEF = _mm_loadu_ps(i0 + 4);
      i0 += 8;
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      const __m128 vi1xCDEF = _mm_loadu_ps(i1 + 4);
      i1 += 8;
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      const __m128 vi2xCDEF = _mm_loadu_ps(i2 + 4);
      i2 += 8;

      const __m128 vi0x8ACE = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi0x9BDF = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi1x8ACE = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi1x9BDF = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi2x8ACE = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi2x9BDF = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(3, 1, 3, 1));

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x8ACE, vk01));
      __m128 vo0p1 = _mm_mul_ps(vi1x8ACE, vk11);
      __m128 vo0p2 = _mm_mul_ps(vi2x8ACE, vk21);

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x9BDF, vk02));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x9BDF, vk12));
      vo0p2 = _mm_add_ps(vo0p2, _mm_mul_ps(vi2x9BDF, vk22));

      const __m128 vi0x79BD = _mm_move_ss(vi0xF9BD, vi0x7531);
      const __m128 vi1x79BD = _mm_move_ss(vi1xF9BD, vi1x7531);
      const __m128 vi2x79BD = _mm_move_ss(vi2xF9BD, vi2x7531);

      vi0x7531 = vi0xF9BD;
      vi1x7531 = vi1xF9BD;
      vi2x7531 = vi2xF9BD;

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x79BD, vk00));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x79BD, vk10));
      vo0p2 = _mm_add_ps(vo0p2, _mm_mul_ps(vi2x79BD, vk20));

      vo0p0 = _mm_add_ps(vo0p0, vo0p1);
      vo0p0 = _mm_add_ps(vo0p0, vo0p2);

      __m128 vo0 = _mm_max_ps(vo0p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);

      _mm_storeu_ps(o0, vo0);
      o0 += 4;
    }
    // Potentially process the last block of 0..7 pixels.
    assert(w < 8 * sizeof(float));
    if XNN_LIKELY(w != 0) {
      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      const __m128 vi0xCDEF = _mm_loadu_ps(i0 + 4);
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      const __m128 vi1xCDEF = _mm_loadu_ps(i1 + 4);
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      const __m128 vi2xCDEF = _mm_loadu_ps(i2 + 4);

      const __m128 vi0x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi0x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi1x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi1x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi2x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi2x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x8ACE, vk01));
      __m128 vo0p1 = _mm_mul_ps(vi1x8ACE, vk11);
      __m128 vo0p2 = _mm_mul_ps(vi2x8ACE, vk21);

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x9BDF, vk02));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x9BDF, vk12));
      vo0p2 = _mm_add_ps(vo0p2, _mm_mul_ps(vi2x9BDF, vk22));

      const __m128 vi0x79BD = _mm_move_ss(vi0xF9BD, vi0x7531);
      const __m128 vi1x79BD = _mm_move_ss(vi1xF9BD, vi1x7531);
      const __m128 vi2x79BD = _mm_move_ss(vi2xF9BD, vi2x7531);

      vi0x7531 = vi0xF9BD;
      vi1x7531 = vi1xF9BD;
      vi2x7531 = vi2xF9BD;

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x79BD, vk00));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x79BD, vk10));
      vo0p2 = _mm_add_ps(vo0p2, _mm_mul_ps(vi2x79BD, vk20));

      vo0p0 = _mm_add_ps(vo0p0, vo0p1);
      vo0p0 = _mm_add_ps(vo0p0, vo0p2);

      __m128 vo0 = _mm_max_ps(vo0p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);

      if (w == 7 * sizeof(float)) {
        _mm_storeu_ps(o0, vo0);
        o0 += 4;
      } else {
        w += 1 * sizeof(float);
        if (w & (4 * sizeof(float))) {
          _mm_storel_pi((__m64*) o0, vo0);
          o0 += 2;

          vo0 = _mm_movehl_ps(vo0, vo0);
        }
        if (w & (2 * sizeof(float))) {
          _mm_store_ss(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_decrement);
    i1 = (const float*) ((uintptr_t) i0 + input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}

void xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_4x4(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 2);

  const __m128 vmask = _mm_load_ps((const float*) params->sse_stride1.mask);
  const __m128 vmax = _mm_set1_ps(params->sse_stride1.max);
  const __m128 vmin = _mm_set1_ps(params->sse_stride1.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128 vbias = _mm_load1_ps(weights);
  const __m128 vk00 = _mm_load1_ps(weights + 1);
  const __m128 vk01 = _mm_load1_ps(weights + 2);
  const __m128 vk02 = _mm_load1_ps(weights + 3);
  const __m128 vk03 = _mm_load1_ps(weights + 4);
  const __m128 vk04 = _mm_load1_ps(weights + 5);
  const __m128 vk10 = _mm_load1_ps(weights + 6);
  const __m128 vk11 = _mm_load1_ps(weights + 7);
  const __m128 vk12 = _mm_load1_ps(weights + 8);
  const __m128 vk13 = _mm_load1_ps(weights + 9);
  const __m128 vk14 = _mm_load1_ps(weights + 10);
  const __m128 vk20 = _mm_load1_ps(weights + 11);
  const __m128 vk21 = _mm_load1_ps(weights + 12);
  const __m128 vk22 = _mm_load1_ps(weights + 13);
  const __m128 vk23 = _mm_load1_ps(weights + 14);
  const __m128 vk24 = _mm_load1_ps(weights + 15);
  const __m128 vk30 = _mm_load1_ps(weights + 16);
  const __m128 vk31 = _mm_load1_ps(weights + 17);
  const __m128 vk32 = _mm_load1_ps(weights + 18);
  const __m128 vk33 = _mm_load1_ps(weights + 19);
  const __m128 vk34 = _mm_load1_ps(weights + 20);
  const __m128 vk40 = _mm_load1_ps(weights + 21);
  const __m128 vk41 = _mm_load1_ps(weights + 22);
  const __m128 vk42 = _mm_load1_ps(weights + 23);
  const __m128 vk43 = _mm_load1_ps(weights + 24);
  const __m128 vk44 = _mm_load1_ps(weights + 25);

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = zero;
  const float* i2 = input;
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);
  float* o3 = (float*) ((uintptr_t) o2 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i4 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i5 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(output_height < 5) {
      i6 = zero;
    }
    if XNN_UNPREDICTABLE(output_height < 6) {
      i7 = zero;
    }

    __m128 vi0x3012 = _mm_setzero_ps();
    __m128 vi1x3012 = _mm_setzero_ps();
    __m128 vi2x3012 = _mm_setzero_ps();
    __m128 vi3x3012 = _mm_setzero_ps();
    __m128 vi4x3012 = _mm_setzero_ps();
    __m128 vi5x3012 = _mm_setzero_ps();
    __m128 vi6x3012 = _mm_setzero_ps();
    __m128 vi7x3012 = _mm_setzero_ps();

    __m128 vi0x4567 = _mm_loadu_ps(i0);
    i0 += 4;
    __m128 vi1x4567 = _mm_loadu_ps(i1);
    i1 += 4;
    __m128 vi2x4567 = _mm_loadu_ps(i2);
    i2 += 4;
    __m128 vi3x4567 = _mm_loadu_ps(i3);
    i3 += 4;
    __m128 vi4x4567 = _mm_loadu_ps(i4);
    i4 += 4;
    __m128 vi5x4567 = _mm_loadu_ps(i5);
    i5 += 4;
    __m128 vi6x4567 = _mm_loadu_ps(i6);
    i6 += 4;
    __m128 vi7x4567 = _mm_loadu_ps(i7);
    i7 += 4;

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 4 * sizeof(float)) {
      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk02));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk02));
      __m128 vo2p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x4567, vk02));
      __m128 vo3p0 = _mm_add_ps(vbias, _mm_mul_ps(vi3x4567, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x4567, vk12));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x4567, vk12));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x4567, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk22));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x4567, vk22));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x4567, vk22));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x4567, vk32));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x4567, vk32));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x4567, vk32));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x4567, vk32));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x4567, vk42));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x4567, vk42));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x4567, vk42));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x4567, vk42));

      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4x7456 = _mm_shuffle_ps(vi4x4567, vi4x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5x7456 = _mm_shuffle_ps(vi5x4567, vi5x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6x7456 = _mm_shuffle_ps(vi6x4567, vi6x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi7x7456 = _mm_shuffle_ps(vi7x4567, vi7x4567, _MM_SHUFFLE(2, 1, 0, 3));

      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      i2 += 4;
      const __m128 vi3x89AB = _mm_loadu_ps(i3);
      i3 += 4;
      const __m128 vi4x89AB = _mm_loadu_ps(i4);
      i4 += 4;
      const __m128 vi5x89AB = _mm_loadu_ps(i5);
      i5 += 4;
      const __m128 vi6x89AB = _mm_loadu_ps(i6);
      i6 += 4;
      const __m128 vi7x89AB = _mm_loadu_ps(i7);
      i7 += 4;

      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);
      const __m128 vi4x3456 = _mm_move_ss(vi4x7456, vi4x3012);
      const __m128 vi5x3456 = _mm_move_ss(vi5x7456, vi5x3012);
      const __m128 vi6x3456 = _mm_move_ss(vi6x7456, vi6x3012);
      const __m128 vi7x3456 = _mm_move_ss(vi7x7456, vi7x3012);

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk01));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x3456, vk01));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x3456, vk01));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x3456, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk11));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x3456, vk11));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x3456, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x3456, vk21));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x3456, vk21));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x3456, vk21));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x3456, vk31));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x3456, vk31));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x3456, vk31));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x3456, vk31));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x3456, vk41));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x3456, vk41));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x3456, vk41));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x3456, vk41));

      const __m128 vi0x2345 = _mm_shuffle_ps(vi0x3012, vi0x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi0x3012 = vi0x7456;
      const __m128 vi1x2345 = _mm_shuffle_ps(vi1x3012, vi1x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi1x3012 = vi1x7456;
      const __m128 vi2x2345 = _mm_shuffle_ps(vi2x3012, vi2x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi2x3012 = vi2x7456;
      const __m128 vi3x2345 = _mm_shuffle_ps(vi3x3012, vi3x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi3x3012 = vi3x7456;
      const __m128 vi4x2345 = _mm_shuffle_ps(vi4x3012, vi4x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi4x3012 = vi4x7456;
      const __m128 vi5x2345 = _mm_shuffle_ps(vi5x3012, vi5x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi5x3012 = vi5x7456;
      const __m128 vi6x2345 = _mm_shuffle_ps(vi6x3012, vi6x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi6x3012 = vi6x7456;
      const __m128 vi7x2345 = _mm_shuffle_ps(vi7x3012, vi7x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi7x3012 = vi7x7456;

      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vi0x89AB);
      vi0x4567 = vi0x89AB;
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vi1x89AB);
      vi1x4567 = vi1x89AB;
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vi2x89AB);
      vi2x4567 = vi2x89AB;
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vi3x89AB);
      vi3x4567 = vi3x89AB;
      const __m128 vi4x8567 = _mm_move_ss(vi4x4567, vi4x89AB);
      vi4x4567 = vi4x89AB;
      const __m128 vi5x8567 = _mm_move_ss(vi5x4567, vi5x89AB);
      vi5x4567 = vi5x89AB;
      const __m128 vi6x8567 = _mm_move_ss(vi6x4567, vi6x89AB);
      vi6x4567 = vi6x89AB;
      const __m128 vi7x8567 = _mm_move_ss(vi7x4567, vi7x89AB);
      vi7x4567 = vi7x89AB;

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x2345, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x2345, vk00));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x2345, vk00));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x2345, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x2345, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x2345, vk10));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x2345, vk10));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x2345, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x2345, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x2345, vk20));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x2345, vk20));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x2345, vk20));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x2345, vk30));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x2345, vk30));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x2345, vk30));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x2345, vk30));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x2345, vk40));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x2345, vk40));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x2345, vk40));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x2345, vk40));

      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi4x5678 = _mm_shuffle_ps(vi4x8567, vi4x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi5x5678 = _mm_shuffle_ps(vi5x8567, vi5x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi6x5678 = _mm_shuffle_ps(vi6x8567, vi6x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi7x5678 = _mm_shuffle_ps(vi7x8567, vi7x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk03));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk03));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x5678, vk03));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x5678, vk03));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk13));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x5678, vk13));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x5678, vk13));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x5678, vk13));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk23));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk23));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x5678, vk23));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x5678, vk23));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x5678, vk33));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x5678, vk33));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x5678, vk33));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x5678, vk33));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x5678, vk43));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x5678, vk43));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x5678, vk43));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x5678, vk43));

      const __m128 vi0x6789 = _mm_shuffle_ps(vi0x5678, vi0x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi1x6789 = _mm_shuffle_ps(vi1x5678, vi1x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi2x6789 = _mm_shuffle_ps(vi2x5678, vi2x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi3x6789 = _mm_shuffle_ps(vi3x5678, vi3x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi4x6789 = _mm_shuffle_ps(vi4x5678, vi4x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi5x6789 = _mm_shuffle_ps(vi5x5678, vi5x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi6x6789 = _mm_shuffle_ps(vi6x5678, vi6x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi7x6789 = _mm_shuffle_ps(vi7x5678, vi7x89AB, _MM_SHUFFLE(1, 0, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x6789, vk04));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x6789, vk04));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x6789, vk04));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x6789, vk04));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x6789, vk14));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x6789, vk14));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x6789, vk14));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x6789, vk14));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x6789, vk24));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x6789, vk24));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x6789, vk24));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x6789, vk24));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x6789, vk34));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x6789, vk34));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x6789, vk34));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x6789, vk34));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x6789, vk44));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x6789, vk44));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x6789, vk44));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x6789, vk44));


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);
      __m128 vo2 = _mm_max_ps(vo2p0, vmin);
      __m128 vo3 = _mm_max_ps(vo3p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);
      vo2 = _mm_min_ps(vo2, vmax);
      vo3 = _mm_min_ps(vo3, vmax);

      _mm_storeu_ps(o3, vo3);
      o3 += 4;
      _mm_storeu_ps(o2, vo2);
      o2 += 4;
      _mm_storeu_ps(o1, vo1);
      o1 += 4;
      _mm_storeu_ps(o0, vo0);
      o0 += 4;
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(w > 4 * sizeof(float)) {
      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk02));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk02));
      __m128 vo2p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x4567, vk02));
      __m128 vo3p0 = _mm_add_ps(vbias, _mm_mul_ps(vi3x4567, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x4567, vk12));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x4567, vk12));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x4567, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk22));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x4567, vk22));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x4567, vk22));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x4567, vk32));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x4567, vk32));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x4567, vk32));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x4567, vk32));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x4567, vk42));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x4567, vk42));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x4567, vk42));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x4567, vk42));

      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4x7456 = _mm_shuffle_ps(vi4x4567, vi4x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5x7456 = _mm_shuffle_ps(vi5x4567, vi5x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6x7456 = _mm_shuffle_ps(vi6x4567, vi6x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi7x7456 = _mm_shuffle_ps(vi7x4567, vi7x4567, _MM_SHUFFLE(2, 1, 0, 3));

      const __m128 vi0x89AB = _mm_and_ps(_mm_loadu_ps(i0), vmask);
      i0 += 4;
      const __m128 vi1x89AB = _mm_and_ps(_mm_loadu_ps(i1), vmask);
      i1 += 4;
      const __m128 vi2x89AB = _mm_and_ps(_mm_loadu_ps(i2), vmask);
      i2 += 4;
      const __m128 vi3x89AB = _mm_and_ps(_mm_loadu_ps(i3), vmask);
      i3 += 4;
      const __m128 vi4x89AB = _mm_and_ps(_mm_loadu_ps(i4), vmask);
      i4 += 4;
      const __m128 vi5x89AB = _mm_and_ps(_mm_loadu_ps(i5), vmask);
      i5 += 4;
      const __m128 vi6x89AB = _mm_and_ps(_mm_loadu_ps(i6), vmask);
      i6 += 4;
      const __m128 vi7x89AB = _mm_and_ps(_mm_loadu_ps(i7), vmask);
      i7 += 4;

      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);
      const __m128 vi4x3456 = _mm_move_ss(vi4x7456, vi4x3012);
      const __m128 vi5x3456 = _mm_move_ss(vi5x7456, vi5x3012);
      const __m128 vi6x3456 = _mm_move_ss(vi6x7456, vi6x3012);
      const __m128 vi7x3456 = _mm_move_ss(vi7x7456, vi7x3012);

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk01));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x3456, vk01));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x3456, vk01));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x3456, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk11));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x3456, vk11));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x3456, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x3456, vk21));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x3456, vk21));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x3456, vk21));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x3456, vk31));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x3456, vk31));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x3456, vk31));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x3456, vk31));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x3456, vk41));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x3456, vk41));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x3456, vk41));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x3456, vk41));

      const __m128 vi0x2345 = _mm_shuffle_ps(vi0x3012, vi0x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi0x3012 = vi0x7456;
      const __m128 vi1x2345 = _mm_shuffle_ps(vi1x3012, vi1x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi1x3012 = vi1x7456;
      const __m128 vi2x2345 = _mm_shuffle_ps(vi2x3012, vi2x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi2x3012 = vi2x7456;
      const __m128 vi3x2345 = _mm_shuffle_ps(vi3x3012, vi3x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi3x3012 = vi3x7456;
      const __m128 vi4x2345 = _mm_shuffle_ps(vi4x3012, vi4x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi4x3012 = vi4x7456;
      const __m128 vi5x2345 = _mm_shuffle_ps(vi5x3012, vi5x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi5x3012 = vi5x7456;
      const __m128 vi6x2345 = _mm_shuffle_ps(vi6x3012, vi6x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi6x3012 = vi6x7456;
      const __m128 vi7x2345 = _mm_shuffle_ps(vi7x3012, vi7x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi7x3012 = vi7x7456;

      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vi0x89AB);
      vi0x4567 = vi0x89AB;
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vi1x89AB);
      vi1x4567 = vi1x89AB;
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vi2x89AB);
      vi2x4567 = vi2x89AB;
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vi3x89AB);
      vi3x4567 = vi3x89AB;
      const __m128 vi4x8567 = _mm_move_ss(vi4x4567, vi4x89AB);
      vi4x4567 = vi4x89AB;
      const __m128 vi5x8567 = _mm_move_ss(vi5x4567, vi5x89AB);
      vi5x4567 = vi5x89AB;
      const __m128 vi6x8567 = _mm_move_ss(vi6x4567, vi6x89AB);
      vi6x4567 = vi6x89AB;
      const __m128 vi7x8567 = _mm_move_ss(vi7x4567, vi7x89AB);
      vi7x4567 = vi7x89AB;

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x2345, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x2345, vk00));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x2345, vk00));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x2345, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x2345, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x2345, vk10));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x2345, vk10));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x2345, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x2345, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x2345, vk20));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x2345, vk20));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x2345, vk20));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x2345, vk30));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x2345, vk30));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x2345, vk30));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x2345, vk30));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x2345, vk40));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x2345, vk40));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x2345, vk40));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x2345, vk40));

      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi4x5678 = _mm_shuffle_ps(vi4x8567, vi4x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi5x5678 = _mm_shuffle_ps(vi5x8567, vi5x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi6x5678 = _mm_shuffle_ps(vi6x8567, vi6x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi7x5678 = _mm_shuffle_ps(vi7x8567, vi7x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk03));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk03));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x5678, vk03));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x5678, vk03));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk13));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x5678, vk13));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x5678, vk13));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x5678, vk13));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk23));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk23));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x5678, vk23));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x5678, vk23));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x5678, vk33));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x5678, vk33));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x5678, vk33));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x5678, vk33));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x5678, vk43));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x5678, vk43));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x5678, vk43));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x5678, vk43));

      const __m128 vi0x6789 = _mm_shuffle_ps(vi0x5678, vi0x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi1x6789 = _mm_shuffle_ps(vi1x5678, vi1x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi2x6789 = _mm_shuffle_ps(vi2x5678, vi2x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi3x6789 = _mm_shuffle_ps(vi3x5678, vi3x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi4x6789 = _mm_shuffle_ps(vi4x5678, vi4x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi5x6789 = _mm_shuffle_ps(vi5x5678, vi5x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi6x6789 = _mm_shuffle_ps(vi6x5678, vi6x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi7x6789 = _mm_shuffle_ps(vi7x5678, vi7x89AB, _MM_SHUFFLE(1, 0, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x6789, vk04));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x6789, vk04));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x6789, vk04));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x6789, vk04));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x6789, vk14));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x6789, vk14));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x6789, vk14));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x6789, vk14));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x6789, vk24));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x6789, vk24));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x6789, vk24));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x6789, vk24));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x6789, vk34));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x6789, vk34));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x6789, vk34));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x6789, vk34));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x6789, vk44));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x6789, vk44));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x6789, vk44));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x6789, vk44));


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);
      __m128 vo2 = _mm_max_ps(vo2p0, vmin);
      __m128 vo3 = _mm_max_ps(vo3p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);
      vo2 = _mm_min_ps(vo2, vmax);
      vo3 = _mm_min_ps(vo3, vmax);

      _mm_storeu_ps(o3, vo3);
      o3 += 4;
      _mm_storeu_ps(o2, vo2);
      o2 += 4;
      _mm_storeu_ps(o1, vo1);
      o1 += 4;
      _mm_storeu_ps(o0, vo0);
      o0 += 4;

      w -= 4 * sizeof(float);
    }
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      vi0x4567 = _mm_and_ps(vi0x4567, vmask);
      vi1x4567 = _mm_and_ps(vi1x4567, vmask);
      vi2x4567 = _mm_and_ps(vi2x4567, vmask);
      vi3x4567 = _mm_and_ps(vi3x4567, vmask);
      vi4x4567 = _mm_and_ps(vi4x4567, vmask);
      vi5x4567 = _mm_and_ps(vi5x4567, vmask);
      vi6x4567 = _mm_and_ps(vi6x4567, vmask);
      vi7x4567 = _mm_and_ps(vi7x4567, vmask);

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk02));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk02));
      __m128 vo2p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x4567, vk02));
      __m128 vo3p0 = _mm_add_ps(vbias, _mm_mul_ps(vi3x4567, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x4567, vk12));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x4567, vk12));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x4567, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk22));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x4567, vk22));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x4567, vk22));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x4567, vk32));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x4567, vk32));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x4567, vk32));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x4567, vk32));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x4567, vk42));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x4567, vk42));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x4567, vk42));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x4567, vk42));

      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4x7456 = _mm_shuffle_ps(vi4x4567, vi4x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5x7456 = _mm_shuffle_ps(vi5x4567, vi5x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6x7456 = _mm_shuffle_ps(vi6x4567, vi6x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi7x7456 = _mm_shuffle_ps(vi7x4567, vi7x4567, _MM_SHUFFLE(2, 1, 0, 3));

      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);
      const __m128 vi4x3456 = _mm_move_ss(vi4x7456, vi4x3012);
      const __m128 vi5x3456 = _mm_move_ss(vi5x7456, vi5x3012);
      const __m128 vi6x3456 = _mm_move_ss(vi6x7456, vi6x3012);
      const __m128 vi7x3456 = _mm_move_ss(vi7x7456, vi7x3012);

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk01));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x3456, vk01));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x3456, vk01));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x3456, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk11));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x3456, vk11));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x3456, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x3456, vk21));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x3456, vk21));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x3456, vk21));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x3456, vk31));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x3456, vk31));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x3456, vk31));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x3456, vk31));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x3456, vk41));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x3456, vk41));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x3456, vk41));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x3456, vk41));

      const __m128 vi0x2345 = _mm_shuffle_ps(vi0x3012, vi0x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1x2345 = _mm_shuffle_ps(vi1x3012, vi1x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2x2345 = _mm_shuffle_ps(vi2x3012, vi2x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3x2345 = _mm_shuffle_ps(vi3x3012, vi3x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4x2345 = _mm_shuffle_ps(vi4x3012, vi4x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5x2345 = _mm_shuffle_ps(vi5x3012, vi5x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6x2345 = _mm_shuffle_ps(vi6x3012, vi6x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi7x2345 = _mm_shuffle_ps(vi7x3012, vi7x7456, _MM_SHUFFLE(2, 1, 0, 3));

      const __m128 vzero = _mm_setzero_ps();
      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vzero);
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vzero);
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vzero);
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vzero);
      const __m128 vi4x8567 = _mm_move_ss(vi4x4567, vzero);
      const __m128 vi5x8567 = _mm_move_ss(vi5x4567, vzero);
      const __m128 vi6x8567 = _mm_move_ss(vi6x4567, vzero);
      const __m128 vi7x8567 = _mm_move_ss(vi7x4567, vzero);

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x2345, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x2345, vk00));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x2345, vk00));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x2345, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x2345, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x2345, vk10));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x2345, vk10));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x2345, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x2345, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x2345, vk20));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x2345, vk20));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x2345, vk20));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x2345, vk30));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x2345, vk30));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x2345, vk30));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x2345, vk30));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x2345, vk40));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x2345, vk40));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x2345, vk40));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x2345, vk40));

      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi4x5678 = _mm_shuffle_ps(vi4x8567, vi4x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi5x5678 = _mm_shuffle_ps(vi5x8567, vi5x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi6x5678 = _mm_shuffle_ps(vi6x8567, vi6x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi7x5678 = _mm_shuffle_ps(vi7x8567, vi7x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk03));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk03));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x5678, vk03));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x5678, vk03));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk13));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x5678, vk13));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x5678, vk13));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x5678, vk13));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk23));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk23));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x5678, vk23));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x5678, vk23));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x5678, vk33));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x5678, vk33));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x5678, vk33));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x5678, vk33));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x5678, vk43));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x5678, vk43));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x5678, vk43));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x5678, vk43));

      const __m128 vi0x6789 = _mm_shuffle_ps(vi0x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi1x6789 = _mm_shuffle_ps(vi1x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi2x6789 = _mm_shuffle_ps(vi2x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi3x6789 = _mm_shuffle_ps(vi3x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi4x6789 = _mm_shuffle_ps(vi4x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi5x6789 = _mm_shuffle_ps(vi5x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi6x6789 = _mm_shuffle_ps(vi6x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi7x6789 = _mm_shuffle_ps(vi7x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x6789, vk04));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x6789, vk04));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x6789, vk04));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x6789, vk04));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x6789, vk14));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x6789, vk14));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x6789, vk14));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x6789, vk14));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x6789, vk24));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x6789, vk24));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x6789, vk24));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x6789, vk24));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x6789, vk34));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x6789, vk34));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x6789, vk34));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x6789, vk34));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x6789, vk44));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x6789, vk44));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x6789, vk44));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x6789, vk44));


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);
      __m128 vo2 = _mm_max_ps(vo2p0, vmin);
      __m128 vo3 = _mm_max_ps(vo3p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);
      vo2 = _mm_min_ps(vo2, vmax);
      vo3 = _mm_min_ps(vo3, vmax);

      if XNN_LIKELY(w & (4 * sizeof(float))) {
        _mm_storeu_ps(o3, vo3);
        o3 += 4;
        _mm_storeu_ps(o2, vo2);
        o2 += 4;
        _mm_storeu_ps(o1, vo1);
        o1 += 4;
        _mm_storeu_ps(o0, vo0);
        o0 += 4;
      } else {
        if (w & (2 * sizeof(float))) {
          _mm_storel_pi((__m64*) o3, vo3);
          o3 += 2;
          _mm_storel_pi((__m64*) o2, vo2);
          o2 += 2;
          _mm_storel_pi((__m64*) o1, vo1);
          o1 += 2;
          _mm_storel_pi((__m64*) o0, vo0);
          o0 += 2;

          vo0 = _mm_movehl_ps(vo0, vo0);
          vo1 = _mm_movehl_ps(vo1, vo1);
          vo2 = _mm_movehl_ps(vo2, vo2);
          vo3 = _mm_movehl_ps(vo3, vo3);
        }
        if (w & (1 * sizeof(float))) {
          _mm_store_ss(o3, vo3);
          o3 += 1;
          _mm_store_ss(o2, vo2);
          o2 += 1;
          _mm_store_ss(o1, vo1);
          o1 += 1;
          _mm_store_ss(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i4 - input_decrement);
    i1 = (const float*) ((uintptr_t) i5 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);

    o0 = o3;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);
    o3 = (float*) ((uintptr_t) o2 + input_width);

    output_height = doz(output_height, 4);
  } while (output_height != 0);
}

void xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_2x4(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const __m128 vmask_even = _mm_load_ps((const float*) params->sse_stride2.mask_even);
  const __m128 vmask_odd  = _mm_load_ps((const float*) params->sse_stride2.mask_odd);
  const __m128 vmax = _mm_set1_ps(params->sse_stride2.max);
  const __m128 vmin = _mm_set1_ps(params->sse_stride2.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128 vbias = _mm_load1_ps(weights);
  const __m128 vk00 = _mm_load1_ps(weights + 1);
  const __m128 vk01 = _mm_load1_ps(weights + 2);
  const __m128 vk02 = _mm_load1_ps(weights + 3);
  const __m128 vk03 = _mm_load1_ps(weights + 4);
  const __m128 vk04 = _mm_load1_ps(weights + 5);
  const __m128 vk10 = _mm_load1_ps(weights + 6);
  const __m128 vk11 = _mm_load1_ps(weights + 7);
  const __m128 vk12 = _mm_load1_ps(weights + 8);
  const __m128 vk13 = _mm_load1_ps(weights + 9);
  const __m128 vk14 = _mm_load1_ps(weights + 10);
  const __m128 vk20 = _mm_load1_ps(weights + 11);
  const __m128 vk21 = _mm_load1_ps(weights + 12);
  const __m128 vk22 = _mm_load1_ps(weights + 13);
  const __m128 vk23 = _mm_load1_ps(weights + 14);
  const __m128 vk24 = _mm_load1_ps(weights + 15);
  const __m128 vk30 = _mm_load1_ps(weights + 16);
  const __m128 vk31 = _mm_load1_ps(weights + 17);
  const __m128 vk32 = _mm_load1_ps(weights + 18);
  const __m128 vk33 = _mm_load1_ps(weights + 19);
  const __m128 vk34 = _mm_load1_ps(weights + 20);
  const __m128 vk40 = _mm_load1_ps(weights + 21);
  const __m128 vk41 = _mm_load1_ps(weights + 22);
  const __m128 vk42 = _mm_load1_ps(weights + 23);
  const __m128 vk43 = _mm_load1_ps(weights + 24);
  const __m128 vk44 = _mm_load1_ps(weights + 25);

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_up_po2(input_width, 8 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);

  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(float)) / 2, sizeof(float));

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + output_width);

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i4 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 8) {
      i5 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 9) {
      i6 = zero;
    }

    __m128 vi0x6024 = _mm_setzero_ps();
    __m128 vi1x6024 = _mm_setzero_ps();
    __m128 vi2x6024 = _mm_setzero_ps();
    __m128 vi3x6024 = _mm_setzero_ps();
    __m128 vi4x6024 = _mm_setzero_ps();
    __m128 vi5x6024 = _mm_setzero_ps();
    __m128 vi6x6024 = _mm_setzero_ps();

    __m128 vi0x7135 = _mm_setzero_ps();
    __m128 vi1x7135 = _mm_setzero_ps();
    __m128 vi2x7135 = _mm_setzero_ps();
    __m128 vi3x7135 = _mm_setzero_ps();
    __m128 vi4x7135 = _mm_setzero_ps();
    __m128 vi5x7135 = _mm_setzero_ps();
    __m128 vi6x7135 = _mm_setzero_ps();

    const __m128 vi0x89AB = _mm_loadu_ps(i0);
    const __m128 vi0xCDEF = _mm_loadu_ps(i0 + 4);
    i0 += 8;
    const __m128 vi1x89AB = _mm_loadu_ps(i1);
    const __m128 vi1xCDEF = _mm_loadu_ps(i1 + 4);
    i1 += 8;
    const __m128 vi2x89AB = _mm_loadu_ps(i2);
    const __m128 vi2xCDEF = _mm_loadu_ps(i2 + 4);
    i2 += 8;
    const __m128 vi3x89AB = _mm_loadu_ps(i3);
    const __m128 vi3xCDEF = _mm_loadu_ps(i3 + 4);
    i3 += 8;
    const __m128 vi4x89AB = _mm_loadu_ps(i4);
    const __m128 vi4xCDEF = _mm_loadu_ps(i4 + 4);
    i4 += 8;
    const __m128 vi5x89AB = _mm_loadu_ps(i5);
    const __m128 vi5xCDEF = _mm_loadu_ps(i5 + 4);
    i5 += 8;
    const __m128 vi6x89AB = _mm_loadu_ps(i6);
    const __m128 vi6xCDEF = _mm_loadu_ps(i6 + 4);
    i6 += 8;

    __m128 vi0x8ACE = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi0x9BDF = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi1x8ACE = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi1x9BDF = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi2x8ACE = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi2x9BDF = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi3x8ACE = _mm_shuffle_ps(vi3x89AB, vi3xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi3x9BDF = _mm_shuffle_ps(vi3x89AB, vi3xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi4x8ACE = _mm_shuffle_ps(vi4x89AB, vi4xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi4x9BDF = _mm_shuffle_ps(vi4x89AB, vi4xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi5x8ACE = _mm_shuffle_ps(vi5x89AB, vi5xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi5x9BDF = _mm_shuffle_ps(vi5x89AB, vi5xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi6x8ACE = _mm_shuffle_ps(vi6x89AB, vi6xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi6x9BDF = _mm_shuffle_ps(vi6x89AB, vi6xCDEF, _MM_SHUFFLE(3, 1, 3, 1));

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 8 * sizeof(float)) {
      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x8ACE, vk02));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x8ACE, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x8ACE, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x8ACE, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x8ACE, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x8ACE, vk22));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x8ACE, vk32));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x8ACE, vk32));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x8ACE, vk42));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x8ACE, vk42));

      const __m128 vi0xE8AC = _mm_shuffle_ps(vi0x8ACE, vi0x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xE8AC = _mm_shuffle_ps(vi1x8ACE, vi1x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xE8AC = _mm_shuffle_ps(vi2x8ACE, vi2x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3xE8AC = _mm_shuffle_ps(vi3x8ACE, vi3x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4xE8AC = _mm_shuffle_ps(vi4x8ACE, vi4x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5xE8AC = _mm_shuffle_ps(vi5x8ACE, vi5x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6xE8AC = _mm_shuffle_ps(vi6x8ACE, vi6x8ACE, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x9BDF, vk03));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x9BDF, vk03));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x9BDF, vk13));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x9BDF, vk13));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x9BDF, vk23));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x9BDF, vk23));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x9BDF, vk33));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x9BDF, vk33));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x9BDF, vk43));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x9BDF, vk43));

      const __m128 vi0x68AC = _mm_move_ss(vi0xE8AC, vi0x6024);
      vi0x6024 = vi0xE8AC;
      const __m128 vi1x68AC = _mm_move_ss(vi1xE8AC, vi1x6024);
      vi1x6024 = vi1xE8AC;
      const __m128 vi2x68AC = _mm_move_ss(vi2xE8AC, vi2x6024);
      vi2x6024 = vi2xE8AC;
      const __m128 vi3x68AC = _mm_move_ss(vi3xE8AC, vi3x6024);
      vi3x6024 = vi3xE8AC;
      const __m128 vi4x68AC = _mm_move_ss(vi4xE8AC, vi4x6024);
      vi4x6024 = vi4xE8AC;
      const __m128 vi5x68AC = _mm_move_ss(vi5xE8AC, vi5x6024);
      vi5x6024 = vi5xE8AC;
      const __m128 vi6x68AC = _mm_move_ss(vi6xE8AC, vi6x6024);
      vi6x6024 = vi6xE8AC;

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3xF9BD = _mm_shuffle_ps(vi3x9BDF, vi3x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4xF9BD = _mm_shuffle_ps(vi4x9BDF, vi4x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5xF9BD = _mm_shuffle_ps(vi5x9BDF, vi5x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6xF9BD = _mm_shuffle_ps(vi6x9BDF, vi6x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x68AC, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x68AC, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x68AC, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x68AC, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x68AC, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x68AC, vk20));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x68AC, vk30));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x68AC, vk30));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x68AC, vk40));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x68AC, vk40));

      const __m128 vi0xGHIJ = _mm_loadu_ps(i0);
      const __m128 vi0xKLMN = _mm_loadu_ps(i0 + 4);
      i0 += 8;
      const __m128 vi1xGHIJ = _mm_loadu_ps(i1);
      const __m128 vi1xKLMN = _mm_loadu_ps(i1 + 4);
      i1 += 8;
      const __m128 vi2xGHIJ = _mm_loadu_ps(i2);
      const __m128 vi2xKLMN = _mm_loadu_ps(i2 + 4);
      i2 += 8;
      const __m128 vi3xGHIJ = _mm_loadu_ps(i3);
      const __m128 vi3xKLMN = _mm_loadu_ps(i3 + 4);
      i3 += 8;
      const __m128 vi4xGHIJ = _mm_loadu_ps(i4);
      const __m128 vi4xKLMN = _mm_loadu_ps(i4 + 4);
      i4 += 8;
      const __m128 vi5xGHIJ = _mm_loadu_ps(i5);
      const __m128 vi5xKLMN = _mm_loadu_ps(i5 + 4);
      i5 += 8;
      const __m128 vi6xGHIJ = _mm_loadu_ps(i6);
      const __m128 vi6xKLMN = _mm_loadu_ps(i6 + 4);
      i6 += 8;

      const __m128 vi0x79BD = _mm_move_ss(vi0xF9BD, vi0x7135);
      vi0x7135 = vi0xF9BD;
      const __m128 vi1x79BD = _mm_move_ss(vi1xF9BD, vi1x7135);
      vi1x7135 = vi1xF9BD;
      const __m128 vi2x79BD = _mm_move_ss(vi2xF9BD, vi2x7135);
      vi2x7135 = vi2xF9BD;
      const __m128 vi3x79BD = _mm_move_ss(vi3xF9BD, vi3x7135);
      vi3x7135 = vi3xF9BD;
      const __m128 vi4x79BD = _mm_move_ss(vi4xF9BD, vi4x7135);
      vi4x7135 = vi4xF9BD;
      const __m128 vi5x79BD = _mm_move_ss(vi5xF9BD, vi5x7135);
      vi5x7135 = vi5xF9BD;
      const __m128 vi6x79BD = _mm_move_ss(vi6xF9BD, vi6x7135);
      vi6x7135 = vi6xF9BD;

      const __m128 vi0xGIKM = _mm_shuffle_ps(vi0xGHIJ, vi0xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi0xHJLN = _mm_shuffle_ps(vi0xGHIJ, vi0xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi0x9BDF = vi0xHJLN;
      const __m128 vi1xGIKM = _mm_shuffle_ps(vi1xGHIJ, vi1xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi1xHJLN = _mm_shuffle_ps(vi1xGHIJ, vi1xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi1x9BDF = vi1xHJLN;
      const __m128 vi2xGIKM = _mm_shuffle_ps(vi2xGHIJ, vi2xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi2xHJLN = _mm_shuffle_ps(vi2xGHIJ, vi2xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi2x9BDF = vi2xHJLN;
      const __m128 vi3xGIKM = _mm_shuffle_ps(vi3xGHIJ, vi3xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi3xHJLN = _mm_shuffle_ps(vi3xGHIJ, vi3xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi3x9BDF = vi3xHJLN;
      const __m128 vi4xGIKM = _mm_shuffle_ps(vi4xGHIJ, vi4xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi4xHJLN = _mm_shuffle_ps(vi4xGHIJ, vi4xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi4x9BDF = vi4xHJLN;
      const __m128 vi5xGIKM = _mm_shuffle_ps(vi5xGHIJ, vi5xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi5xHJLN = _mm_shuffle_ps(vi5xGHIJ, vi5xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi5x9BDF = vi5xHJLN;
      const __m128 vi6xGIKM = _mm_shuffle_ps(vi6xGHIJ, vi6xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi6xHJLN = _mm_shuffle_ps(vi6xGHIJ, vi6xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi6x9BDF = vi6xHJLN;

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x79BD, vk01));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x79BD, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x79BD, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x79BD, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x79BD, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x79BD, vk21));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x79BD, vk31));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x79BD, vk31));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x79BD, vk41));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x79BD, vk41));

      const __m128 vi0xGACE = _mm_move_ss(vi0x8ACE, vi0xGIKM);
      vi0x8ACE = vi0xGIKM;
      const __m128 vi1xGACE = _mm_move_ss(vi1x8ACE, vi1xGIKM);
      vi1x8ACE = vi1xGIKM;
      const __m128 vi2xGACE = _mm_move_ss(vi2x8ACE, vi2xGIKM);
      vi2x8ACE = vi2xGIKM;
      const __m128 vi3xGACE = _mm_move_ss(vi3x8ACE, vi3xGIKM);
      vi3x8ACE = vi3xGIKM;
      const __m128 vi4xGACE = _mm_move_ss(vi4x8ACE, vi4xGIKM);
      vi4x8ACE = vi4xGIKM;
      const __m128 vi5xGACE = _mm_move_ss(vi5x8ACE, vi5xGIKM);
      vi5x8ACE = vi5xGIKM;
      const __m128 vi6xGACE = _mm_move_ss(vi6x8ACE, vi6xGIKM);
      vi6x8ACE = vi6xGIKM;

      const __m128 vi0xACEG = _mm_shuffle_ps(vi0xGACE, vi0xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi1xACEG = _mm_shuffle_ps(vi1xGACE, vi1xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi2xACEG = _mm_shuffle_ps(vi2xGACE, vi2xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi3xACEG = _mm_shuffle_ps(vi3xGACE, vi3xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi4xACEG = _mm_shuffle_ps(vi4xGACE, vi4xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi5xACEG = _mm_shuffle_ps(vi5xGACE, vi5xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi6xACEG = _mm_shuffle_ps(vi6xGACE, vi6xGACE, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0xACEG, vk04));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2xACEG, vk04));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1xACEG, vk14));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3xACEG, vk14));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2xACEG, vk24));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4xACEG, vk24));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3xACEG, vk34));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5xACEG, vk34));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4xACEG, vk44));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6xACEG, vk44));


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);

      _mm_storeu_ps(o1, vo1);
      o1 += 4;
      _mm_storeu_ps(o0, vo0);
      o0 += 4;
    }
    // Last block has 1-8 pixels to process.
    assert(w <= 8 * sizeof(float));
    assert(w >= 1 * sizeof(float));
    {
      vi0x8ACE = _mm_and_ps(vi0x8ACE, vmask_even);
      vi0x9BDF = _mm_and_ps(vi0x9BDF, vmask_odd);
      vi1x8ACE = _mm_and_ps(vi1x8ACE, vmask_even);
      vi1x9BDF = _mm_and_ps(vi1x9BDF, vmask_odd);
      vi2x8ACE = _mm_and_ps(vi2x8ACE, vmask_even);
      vi2x9BDF = _mm_and_ps(vi2x9BDF, vmask_odd);
      vi3x8ACE = _mm_and_ps(vi3x8ACE, vmask_even);
      vi3x9BDF = _mm_and_ps(vi3x9BDF, vmask_odd);
      vi4x8ACE = _mm_and_ps(vi4x8ACE, vmask_even);
      vi4x9BDF = _mm_and_ps(vi4x9BDF, vmask_odd);
      vi5x8ACE = _mm_and_ps(vi5x8ACE, vmask_even);
      vi5x9BDF = _mm_and_ps(vi5x9BDF, vmask_odd);
      vi6x8ACE = _mm_and_ps(vi6x8ACE, vmask_even);
      vi6x9BDF = _mm_and_ps(vi6x9BDF, vmask_odd);

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x8ACE, vk02));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x8ACE, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x8ACE, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x8ACE, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x8ACE, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x8ACE, vk22));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x8ACE, vk32));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x8ACE, vk32));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x8ACE, vk42));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x8ACE, vk42));

      const __m128 vi0xE8AC = _mm_shuffle_ps(vi0x8ACE, vi0x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xE8AC = _mm_shuffle_ps(vi1x8ACE, vi1x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xE8AC = _mm_shuffle_ps(vi2x8ACE, vi2x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3xE8AC = _mm_shuffle_ps(vi3x8ACE, vi3x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4xE8AC = _mm_shuffle_ps(vi4x8ACE, vi4x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5xE8AC = _mm_shuffle_ps(vi5x8ACE, vi5x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6xE8AC = _mm_shuffle_ps(vi6x8ACE, vi6x8ACE, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x9BDF, vk03));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x9BDF, vk03));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x9BDF, vk13));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x9BDF, vk13));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x9BDF, vk23));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x9BDF, vk23));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x9BDF, vk33));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x9BDF, vk33));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x9BDF, vk43));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x9BDF, vk43));

      const __m128 vi0x68AC = _mm_move_ss(vi0xE8AC, vi0x6024);
      const __m128 vi1x68AC = _mm_move_ss(vi1xE8AC, vi1x6024);
      const __m128 vi2x68AC = _mm_move_ss(vi2xE8AC, vi2x6024);
      const __m128 vi3x68AC = _mm_move_ss(vi3xE8AC, vi3x6024);
      const __m128 vi4x68AC = _mm_move_ss(vi4xE8AC, vi4x6024);
      const __m128 vi5x68AC = _mm_move_ss(vi5xE8AC, vi5x6024);
      const __m128 vi6x68AC = _mm_move_ss(vi6xE8AC, vi6x6024);

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3xF9BD = _mm_shuffle_ps(vi3x9BDF, vi3x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4xF9BD = _mm_shuffle_ps(vi4x9BDF, vi4x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5xF9BD = _mm_shuffle_ps(vi5x9BDF, vi5x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6xF9BD = _mm_shuffle_ps(vi6x9BDF, vi6x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x68AC, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x68AC, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x68AC, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x68AC, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x68AC, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x68AC, vk20));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x68AC, vk30));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x68AC, vk30));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x68AC, vk40));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x68AC, vk40));

      const __m128 vi0x79BD = _mm_move_ss(vi0xF9BD, vi0x7135);
      const __m128 vi1x79BD = _mm_move_ss(vi1xF9BD, vi1x7135);
      const __m128 vi2x79BD = _mm_move_ss(vi2xF9BD, vi2x7135);
      const __m128 vi3x79BD = _mm_move_ss(vi3xF9BD, vi3x7135);
      const __m128 vi4x79BD = _mm_move_ss(vi4xF9BD, vi4x7135);
      const __m128 vi5x79BD = _mm_move_ss(vi5xF9BD, vi5x7135);
      const __m128 vi6x79BD = _mm_move_ss(vi6xF9BD, vi6x7135);

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x79BD, vk01));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x79BD, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x79BD, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x79BD, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x79BD, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x79BD, vk21));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x79BD, vk31));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x79BD, vk31));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x79BD, vk41));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x79BD, vk41));

      const __m128 vzero = _mm_setzero_ps();
      const __m128 vi0xGACE = _mm_move_ss(vi0x8ACE, vzero);
      const __m128 vi1xGACE = _mm_move_ss(vi1x8ACE, vzero);
      const __m128 vi2xGACE = _mm_move_ss(vi2x8ACE, vzero);
      const __m128 vi3xGACE = _mm_move_ss(vi3x8ACE, vzero);
      const __m128 vi4xGACE = _mm_move_ss(vi4x8ACE, vzero);
      const __m128 vi5xGACE = _mm_move_ss(vi5x8ACE, vzero);
      const __m128 vi6xGACE = _mm_move_ss(vi6x8ACE, vzero);

      const __m128 vi0xACEG = _mm_shuffle_ps(vi0xGACE, vi0xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi1xACEG = _mm_shuffle_ps(vi1xGACE, vi1xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi2xACEG = _mm_shuffle_ps(vi2xGACE, vi2xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi3xACEG = _mm_shuffle_ps(vi3xGACE, vi3xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi4xACEG = _mm_shuffle_ps(vi4xGACE, vi4xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi5xACEG = _mm_shuffle_ps(vi5xGACE, vi5xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi6xACEG = _mm_shuffle_ps(vi6xGACE, vi6xGACE, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0xACEG, vk04));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2xACEG, vk04));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1xACEG, vk14));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3xACEG, vk14));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2xACEG, vk24));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4xACEG, vk24));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3xACEG, vk34));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5xACEG, vk34));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4xACEG, vk44));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6xACEG, vk44));


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);

      size_t w_tmp = (w + 1 * sizeof(float)) / (2 * sizeof(float));
      if XNN_LIKELY(w_tmp >= 4) {
        _mm_storeu_ps(o1, vo1);
        o1 += 4;
        _mm_storeu_ps(o0, vo0);
        o0 += 4;
      } else {
        if (w_tmp & 2) {
          _mm_storel_pi((__m64*) o1, vo1);
          o1 += 2;
          _mm_storel_pi((__m64*) o0, vo0);
          o0 += 2;

          vo0 = _mm_movehl_ps(vo0, vo0);
          vo1 = _mm_movehl_ps(vo1, vo1);
        }
        if (w_tmp & 1) {
          _mm_store_ss(o1, vo1);
          o1 += 1;
          _mm_store_ss(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i4 - input_decrement);
    i1 = (const float*) ((uintptr_t) i5 - input_decrement);
    i2 = (const float*) ((uintptr_t) i6 - input_decrement);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);

    o0 = o1;
    o1 = (float*) ((uintptr_t) o0 + output_width);

    output_height = doz(output_height, 2);
    padded_input_height = doz(padded_input_height, 4);
  } while (output_height != 0);
}

void xnn_f32_gavgpool_cw_ukernel__sse_u4(
    size_t elements,
    size_t channels,
    const float* input,
    float* output,
    const union xnn_f32_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(elements != 0);
  assert(elements % sizeof(float) == 0);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + elements);
  const float* i2 = (const float*) ((uintptr_t) i1 + elements);
  const float* i3 = (const float*) ((uintptr_t) i2 + elements);

  const __m128 vmask = _mm_load_ps((const float*) params->sse.mask);
  const __m128 vmultiplier = _mm_load_ps(params->sse.multiplier);
  const __m128 voutput_min = _mm_load_ps(params->sse.output_min);
  const __m128 voutput_max = _mm_load_ps(params->sse.output_max);

  while (channels >= 4) {
    __m128 vsum0 = _mm_setzero_ps();
    __m128 vsum1 = _mm_setzero_ps();
    __m128 vsum2 = _mm_setzero_ps();
    __m128 vsum3 = _mm_setzero_ps();
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1 = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2 = _mm_loadu_ps(i2);
      i2 += 4;
      const __m128 vi3 = _mm_loadu_ps(i3);
      i3 += 4;

      vsum0 = _mm_add_ps(vsum0, vi0);
      vsum1 = _mm_add_ps(vsum1, vi1);
      vsum2 = _mm_add_ps(vsum2, vi2);
      vsum3 = _mm_add_ps(vsum3, vi3);
      n -= 4 * sizeof(float);
    }

    if XNN_UNLIKELY(n != 0) {
      const __m128 vi0 = _mm_and_ps(_mm_loadu_ps(i0), vmask);
      i0 = (const float*) ((uintptr_t) i0 + n);
      const __m128 vi1 = _mm_and_ps(_mm_loadu_ps(i1), vmask);
      i1 = (const float*) ((uintptr_t) i1 + n);
      const __m128 vi2 = _mm_and_ps(_mm_loadu_ps(i2), vmask);
      i2 = (const float*) ((uintptr_t) i2 + n);
      const __m128 vi3 = _mm_and_ps(_mm_loadu_ps(i3), vmask);
      i3 = (const float*) ((uintptr_t) i3 + n);

      vsum0 = _mm_add_ps(vsum0, vi0);
      vsum1 = _mm_add_ps(vsum1, vi1);
      vsum2 = _mm_add_ps(vsum2, vi2);
      vsum3 = _mm_add_ps(vsum3, vi3);
    }

    // Having exactly 4 rows makes this work out nicely as we end up with
    // the 4 totals in 4 different lanes of the same vector.
    const __m128 vsum01 = _mm_add_ps(_mm_unpacklo_ps(vsum0, vsum1), _mm_unpackhi_ps(vsum0, vsum1));
    const __m128 vsum23 = _mm_add_ps(_mm_unpacklo_ps(vsum2, vsum3), _mm_unpackhi_ps(vsum2, vsum3));
    const __m128 vsum = _mm_add_ps(_mm_movelh_ps(vsum01, vsum23), _mm_movehl_ps(vsum23, vsum01));
    __m128 vout = _mm_mul_ps(vsum, vmultiplier);

    vout = _mm_max_ps(vout, voutput_min);
    vout = _mm_min_ps(vout, voutput_max);

    _mm_storeu_ps(output, vout);
    output += 4;
    i0 = i3;
    i1 = (const float*) ((uintptr_t) i0 + elements);
    i2 = (const float*) ((uintptr_t) i1 + elements);
    i3 = (const float*) ((uintptr_t) i2 + elements);
    channels -= 4;
  }

  while (channels != 0) {
    __m128 vsum = _mm_setzero_ps();
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      i0 += 4;
      vsum = _mm_add_ps(vsum, vi0);
      n -= 4 * sizeof(float);
    }

    if XNN_UNLIKELY(n != 0) {
      __m128 vi0 = _mm_and_ps(_mm_loadu_ps(i0), vmask);
      i0 = (const float*) ((uintptr_t) i0 + n);
      vsum = _mm_add_ps(vsum, vi0);
    }

    vsum = _mm_add_ps(vsum, _mm_movehl_ps(vsum, vsum));
    vsum = _mm_add_ss(vsum, _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(3, 2, 1, 1)));

    __m128 vout = _mm_mul_ss(vsum, vmultiplier);

    vout = _mm_max_ss(vout, voutput_min);
    vout = _mm_min_ss(vout, voutput_max);

    _mm_store_ss(output, vout);
    output += 1;
    channels -= 1;
  }
}

void xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* buffer,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
  const size_t packed_channels = round_up_po2(channels, 4);
  const size_t input_increment = 7 * input_stride - packed_channels * sizeof(float);

  float* b = buffer;
  for (size_t c = 0; c < channels; c += 4) {
    const __m128 vi0 = _mm_loadu_ps(i0);
    i0 += 4;
    const __m128 vi1 = _mm_loadu_ps(i1);
    i1 += 4;
    const __m128 vi2 = _mm_loadu_ps(i2);
    i2 += 4;
    const __m128 vi3 = _mm_loadu_ps(i3);
    i3 += 4;
    const __m128 vi4 = _mm_loadu_ps(i4);
    i4 += 4;
    const __m128 vi5 = _mm_loadu_ps(i5);
    i5 += 4;
    const __m128 vi6 = _mm_loadu_ps(i6);
    i6 += 4;

    const __m128 vsum01 = _mm_add_ps(vi0, vi1);
    const __m128 vsum23 = _mm_add_ps(vi2, vi3);
    const __m128 vsum45 = _mm_add_ps(vi4, vi5);

    const __m128 vsum016 = _mm_add_ps(vsum01, vi6);
    const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);

    const __m128 vsum = _mm_add_ps(vsum016, vsum2345);

    _mm_store_ps(b, vsum); b += 4;
  }
  for (rows -= 7; rows > 7; rows -= 7) {
    b = buffer;

    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_increment);
    i3 = (const float*) ((uintptr_t) i3 + input_increment);
    i4 = (const float*) ((uintptr_t) i4 + input_increment);
    i5 = (const float*) ((uintptr_t) i5 + input_increment);
    i6 = (const float*) ((uintptr_t) i6 + input_increment);

    for (size_t c = 0; c < channels; c += 4) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1 = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2 = _mm_loadu_ps(i2);
      i2 += 4;
      const __m128 vi3 = _mm_loadu_ps(i3);
      i3 += 4;
      const __m128 vi4 = _mm_loadu_ps(i4);
      i4 += 4;
      const __m128 vi5 = _mm_loadu_ps(i5);
      i5 += 4;
      const __m128 vi6 = _mm_loadu_ps(i6);
      i6 += 4;
      const __m128 vacc = _mm_load_ps(b);

      const __m128 vsum01 = _mm_add_ps(vi0, vi1);
      const __m128 vsum23 = _mm_add_ps(vi2, vi3);
      const __m128 vsum45 = _mm_add_ps(vi4, vi5);
      const __m128 vsum6a = _mm_add_ps(vi6, vacc);

      const __m128 vsum0123 = _mm_add_ps(vsum01, vsum23);
      const __m128 vsum456a = _mm_add_ps(vsum45, vsum6a);

      const __m128 vsum = _mm_add_ps(vsum0123, vsum456a);

      _mm_store_ps(b, vsum); b += 4;
    }
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
  const __m128 vscale = _mm_set1_ps(params->sse.scale);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  b = buffer;
  while (channels >= 4) {
    const __m128 vi0 = _mm_loadu_ps(i0);
    i0 += 4;
    const __m128 vi1 = _mm_loadu_ps(i1);
    i1 += 4;
    const __m128 vi2 = _mm_loadu_ps(i2);
    i2 += 4;
    const __m128 vi3 = _mm_loadu_ps(i3);
    i3 += 4;
    const __m128 vi4 = _mm_loadu_ps(i4);
    i4 += 4;
    const __m128 vi5 = _mm_loadu_ps(i5);
    i5 += 4;
    const __m128 vi6 = _mm_loadu_ps(i6);
    i6 += 4;
    const __m128 vacc = _mm_load_ps(b);
    b += 4;

    const __m128 vsum01 = _mm_add_ps(vi0, vi1);
    const __m128 vsum23 = _mm_add_ps(vi2, vi3);
    const __m128 vsum45 = _mm_add_ps(vi4, vi5);
    const __m128 vsum6a = _mm_add_ps(vi6, vacc);

    const __m128 vsum0123 = _mm_add_ps(vsum01, vsum23);
    const __m128 vsum456a = _mm_add_ps(vsum45, vsum6a);

    const __m128 vsum = _mm_add_ps(vsum0123, vsum456a);

    __m128 vout = _mm_mul_ps(vsum, vscale);
    vout = _mm_max_ps(vout, vmin);
    vout = _mm_min_ps(vout, vmax);

    _mm_storeu_ps(output, vout);
    output += 4;

    channels -= 4;
  }
  if (channels != 0) {
    const __m128 vi0 = _mm_loadu_ps(i0);
    const __m128 vi1 = _mm_loadu_ps(i1);
    const __m128 vi2 = _mm_loadu_ps(i2);
    const __m128 vi3 = _mm_loadu_ps(i3);
    const __m128 vi4 = _mm_loadu_ps(i4);
    const __m128 vi5 = _mm_loadu_ps(i5);
    const __m128 vi6 = _mm_loadu_ps(i6);
    const __m128 vacc = _mm_loadu_ps(b);

    const __m128 vsum01 = _mm_add_ps(vi0, vi1);
    const __m128 vsum23 = _mm_add_ps(vi2, vi3);
    const __m128 vsum45 = _mm_add_ps(vi4, vi5);
    const __m128 vsum6a = _mm_add_ps(vi6, vacc);

    const __m128 vsum0123 = _mm_add_ps(vsum01, vsum23);
    const __m128 vsum456a = _mm_add_ps(vsum45, vsum6a);

    const __m128 vsum = _mm_add_ps(vsum0123, vsum456a);

    __m128 vout = _mm_mul_ps(vsum, vscale);
    vout = _mm_max_ps(vout, vmin);
    vout = _mm_min_ps(vout, vmax);

    if (channels & 2) {
      _mm_storel_pi((__m64*) output, vout);
      vout = _mm_movehl_ps(vout, vout);
      output += 2;
    }
    if (channels & 1) {
      _mm_store_ss(output, vout);
    }
  }
}

void xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
  const __m128 vscale = _mm_set1_ps(params->sse.scale);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  while (channels >= 4) {
    const __m128 vi0 = _mm_loadu_ps(i0);
    i0 += 4;
    const __m128 vi1 = _mm_loadu_ps(i1);
    i1 += 4;
    const __m128 vi2 = _mm_loadu_ps(i2);
    i2 += 4;
    const __m128 vi3 = _mm_loadu_ps(i3);
    i3 += 4;
    const __m128 vi4 = _mm_loadu_ps(i4);
    i4 += 4;
    const __m128 vi5 = _mm_loadu_ps(i5);
    i5 += 4;
    const __m128 vi6 = _mm_loadu_ps(i6);
    i6 += 4;

    const __m128 vsum01 = _mm_add_ps(vi0, vi1);
    const __m128 vsum23 = _mm_add_ps(vi2, vi3);
    const __m128 vsum45 = _mm_add_ps(vi4, vi5);

    const __m128 vsum016 = _mm_add_ps(vsum01, vi6);
    const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);

    const __m128 vsum = _mm_add_ps(vsum016, vsum2345);

    __m128 vout = _mm_mul_ps(vsum, vscale);
    vout = _mm_max_ps(vout, vmin);
    vout = _mm_min_ps(vout, vmax);

    _mm_storeu_ps(output, vout);
    output += 4;

    channels -= 4;
  }
  if (channels != 0) {
    const __m128 vi0 = _mm_loadu_ps(i0);
    const __m128 vi1 = _mm_loadu_ps(i1);
    const __m128 vi2 = _mm_loadu_ps(i2);
    const __m128 vi3 = _mm_loadu_ps(i3);
    const __m128 vi4 = _mm_loadu_ps(i4);
    const __m128 vi5 = _mm_loadu_ps(i5);
    const __m128 vi6 = _mm_loadu_ps(i6);

    const __m128 vsum01 = _mm_add_ps(vi0, vi1);
    const __m128 vsum23 = _mm_add_ps(vi2, vi3);
    const __m128 vsum45 = _mm_add_ps(vi4, vi5);

    const __m128 vsum016 = _mm_add_ps(vsum01, vi6);
    const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);

    const __m128 vsum = _mm_add_ps(vsum016, vsum2345);

    __m128 vout = _mm_mul_ps(vsum, vscale);
    vout = _mm_max_ps(vout, vmin);
    vout = _mm_min_ps(vout, vmax);

    if (channels & 2) {
      _mm_storel_pi((__m64*) output, vout);
      vout = _mm_movehl_ps(vout, vout);
      output += 2;
    }
    if (channels & 1) {
      _mm_store_ss(output, vout);
    }
  }
}

void xnn_f32_gemm_minmax_ukernel_1x8__sse_load1(
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

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_load_ps(w + 0);
    __m128 vacc0x4567 = _mm_load_ps(w + 4);
    w += 8;

    size_t k = kc;
    do {
      const __m128 va0 = _mm_load1_ps(a0);
      a0 += 1;

      const __m128 vb0123 = _mm_load_ps(w);
      const __m128 vb4567 = _mm_load_ps(w + 4);
      w += 8;

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567));

      k -= sizeof(float);
    } while (k != 0);

    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x2c4__sse(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0c4 = _mm_load_ss(w);
    __m128 vacc0x1c4 = _mm_load_ss(w + 1);
    __m128 vacc1x0c4 = vacc0x0c4;
    __m128 vacc1x1c4 = vacc0x1c4;
    __m128 vacc2x0c4 = vacc0x0c4;
    __m128 vacc2x1c4 = vacc0x1c4;
    __m128 vacc3x0c4 = vacc0x0c4;
    __m128 vacc3x1c4 = vacc0x1c4;
    w += 2;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const __m128 va0 = _mm_loadu_ps(a0);
      a0 += 4;
      const __m128 va1 = _mm_loadu_ps(a1);
      a1 += 4;
      const __m128 va2 = _mm_loadu_ps(a2);
      a2 += 4;
      const __m128 va3 = _mm_loadu_ps(a3);
      a3 += 4;

      const __m128 vb0 = _mm_loadu_ps(w);
      const __m128 vb1 = _mm_loadu_ps(w + 4);
      w += 8;

      vacc0x0c4 = _mm_add_ps(vacc0x0c4, _mm_mul_ps(va0, vb0));
      vacc0x1c4 = _mm_add_ps(vacc0x1c4, _mm_mul_ps(va0, vb1));
      vacc1x0c4 = _mm_add_ps(vacc1x0c4, _mm_mul_ps(va1, vb0));
      vacc1x1c4 = _mm_add_ps(vacc1x1c4, _mm_mul_ps(va1, vb1));
      vacc2x0c4 = _mm_add_ps(vacc2x0c4, _mm_mul_ps(va2, vb0));
      vacc2x1c4 = _mm_add_ps(vacc2x1c4, _mm_mul_ps(va2, vb1));
      vacc3x0c4 = _mm_add_ps(vacc3x0c4, _mm_mul_ps(va3, vb0));
      vacc3x1c4 = _mm_add_ps(vacc3x1c4, _mm_mul_ps(va3, vb1));
    }
    if XNN_UNLIKELY(k != 0) {
      const __m128 va0 = _mm_loadu_ps(a0);
      a0 = (const float*) ((uintptr_t) a0 + k);
      const __m128 va1 = _mm_loadu_ps(a1);
      a1 = (const float*) ((uintptr_t) a1 + k);
      const __m128 va2 = _mm_loadu_ps(a2);
      a2 = (const float*) ((uintptr_t) a2 + k);
      const __m128 va3 = _mm_loadu_ps(a3);
      a3 = (const float*) ((uintptr_t) a3 + k);

      const __m128 vb0 = _mm_loadu_ps(w);
      const __m128 vb1 = _mm_loadu_ps(w + 4);
      w += 8;

      const __m128 vmask0 = _mm_cmpeq_ps(_mm_setzero_ps(), vb0);
      const __m128 vmask1 = _mm_cmpeq_ps(_mm_setzero_ps(), vb1);

      vacc0x0c4 = _mm_add_ps(vacc0x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va0), vb0));
      vacc0x1c4 = _mm_add_ps(vacc0x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va0), vb1));
      vacc1x0c4 = _mm_add_ps(vacc1x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va1), vb0));
      vacc1x1c4 = _mm_add_ps(vacc1x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va1), vb1));
      vacc2x0c4 = _mm_add_ps(vacc2x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va2), vb0));
      vacc2x1c4 = _mm_add_ps(vacc2x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va2), vb1));
      vacc3x0c4 = _mm_add_ps(vacc3x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va3), vb0));
      vacc3x1c4 = _mm_add_ps(vacc3x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va3), vb1));
    }

    const __m128 vacc0x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc0x0c4, vacc0x1c4), _mm_unpackhi_ps(vacc0x0c4, vacc0x1c4));
    const __m128 vacc1x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc1x0c4, vacc1x1c4), _mm_unpackhi_ps(vacc1x0c4, vacc1x1c4));
    const __m128 vacc2x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc2x0c4, vacc2x1c4), _mm_unpackhi_ps(vacc2x0c4, vacc2x1c4));
    const __m128 vacc3x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc3x0c4, vacc3x1c4), _mm_unpackhi_ps(vacc3x0c4, vacc3x1c4));

    __m128 vacc01x01 = _mm_add_ps(_mm_movelh_ps(vacc0x01c2, vacc1x01c2), _mm_movehl_ps(vacc1x01c2, vacc0x01c2));
    __m128 vacc23x01 = _mm_add_ps(_mm_movelh_ps(vacc2x01c2, vacc3x01c2), _mm_movehl_ps(vacc3x01c2, vacc2x01c2));

    vacc01x01 = _mm_min_ps(vacc01x01, vmax);
    vacc23x01 = _mm_min_ps(vacc23x01, vmax);

    vacc01x01 = _mm_max_ps(vacc01x01, vmin);
    vacc23x01 = _mm_max_ps(vacc23x01, vmin);

    if XNN_LIKELY(nc >= 2) {
      _mm_storel_pi((__m64*) c0, vacc01x01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a0 = (const float*) ((uintptr_t) a0 - kc);
      _mm_storeh_pi((__m64*) c1, vacc01x01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      _mm_storel_pi((__m64*) c2, vacc23x01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      _mm_storeh_pi((__m64*) c3, vacc23x01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 2;
    } else {
      assert(nc == 1);
      _mm_store_ss(c0, vacc01x01);
      _mm_store_ss(c1, _mm_movehl_ps(vacc01x01, vacc01x01));
      _mm_store_ss(c2, vacc23x01);
      _mm_store_ss(c3, _mm_movehl_ps(vacc23x01, vacc23x01));

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x8__sse_load1(
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

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_load_ps(w + 0);
    __m128 vacc0x4567 = _mm_load_ps(w + 4);
    __m128 vacc1x0123 = vacc0x0123;
    __m128 vacc1x4567 = vacc0x4567;
    __m128 vacc2x0123 = vacc0x0123;
    __m128 vacc2x4567 = vacc0x4567;
    __m128 vacc3x0123 = vacc0x0123;
    __m128 vacc3x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    do {
      const __m128 va0 = _mm_load1_ps(a0);
      a0 += 1;
      const __m128 va1 = _mm_load1_ps(a1);
      a1 += 1;
      const __m128 va2 = _mm_load1_ps(a2);
      a2 += 1;
      const __m128 va3 = _mm_load1_ps(a3);
      a3 += 1;

      const __m128 vb0123 = _mm_load_ps(w);
      const __m128 vb4567 = _mm_load_ps(w + 4);
      w += 8;

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1, vb0123));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2, vb0123));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3, vb0123));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1, vb4567));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2, vb4567));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3, vb4567));

      k -= sizeof(float);
    } while (k != 0);

    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc1x0123 = _mm_min_ps(vacc1x0123, vmax);
    vacc2x0123 = _mm_min_ps(vacc2x0123, vmax);
    vacc3x0123 = _mm_min_ps(vacc3x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);
    vacc1x4567 = _mm_min_ps(vacc1x4567, vmax);
    vacc2x4567 = _mm_min_ps(vacc2x4567, vmax);
    vacc3x4567 = _mm_min_ps(vacc3x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc1x0123 = _mm_max_ps(vacc1x0123, vmin);
    vacc2x0123 = _mm_max_ps(vacc2x0123, vmin);
    vacc3x0123 = _mm_max_ps(vacc3x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);
    vacc1x4567 = _mm_max_ps(vacc1x4567, vmin);
    vacc2x4567 = _mm_max_ps(vacc2x4567, vmin);
    vacc3x4567 = _mm_max_ps(vacc3x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_ps(c1, vacc1x0123);
      _mm_storeu_ps(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_ps(c2, vacc2x0123);
      _mm_storeu_ps(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_ps(c3, vacc3x0123);
      _mm_storeu_ps(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c3, vacc3x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c3, vacc3x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_ibilinear_chw_ukernel__sse_p8(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t input_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(float) == 0);

  do {
    const float** i = input;
    const float* w = weights;
    size_t p = output_pixels;
    for (; p >= 8; p -= 8) {
      const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
      const float* itl2 = (const float*) ((uintptr_t) i[4] + input_offset);
      const float* ibl2 = (const float*) ((uintptr_t) i[5] + input_offset);
      const float* itl3 = (const float*) ((uintptr_t) i[6] + input_offset);
      const float* ibl3 = (const float*) ((uintptr_t) i[7] + input_offset);
      const float* itl4 = (const float*) ((uintptr_t) i[8] + input_offset);
      const float* ibl4 = (const float*) ((uintptr_t) i[9] + input_offset);
      const float* itl5 = (const float*) ((uintptr_t) i[10] + input_offset);
      const float* ibl5 = (const float*) ((uintptr_t) i[11] + input_offset);
      const float* itl6 = (const float*) ((uintptr_t) i[12] + input_offset);
      const float* ibl6 = (const float*) ((uintptr_t) i[13] + input_offset);
      const float* itl7 = (const float*) ((uintptr_t) i[14] + input_offset);
      const float* ibl7 = (const float*) ((uintptr_t) i[15] + input_offset);
      i += 2 * 8;

      const __m128 vw0123p0 = _mm_loadu_ps(w + 0);
      const __m128 vw0123p1 = _mm_loadu_ps(w + 4);
      const __m128 vw4567p0 = _mm_loadu_ps(w + 8);
      const __m128 vw4567p1 = _mm_loadu_ps(w + 12);
      w += 2 * 8;

      const __m128 vtltr0 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl0);
      const __m128 vblbr0 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl0);
      const __m128 vtltr2 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl2);
      const __m128 vblbr2 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl2);
      const __m128 vtltr4 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl4);
      const __m128 vblbr4 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl4);
      const __m128 vtltr6 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl6);
      const __m128 vblbr6 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl6);

      const __m128 valphah0123 = _mm_shuffle_ps(vw0123p0, vw0123p1, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 valphav0123 = _mm_shuffle_ps(vw0123p0, vw0123p1, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 valphah4567 = _mm_shuffle_ps(vw4567p0, vw4567p1, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 valphav4567 = _mm_shuffle_ps(vw4567p0, vw4567p1, _MM_SHUFFLE(3, 1, 3, 1));

      const __m128 vtltr01 = _mm_loadh_pi(vtltr0, (const __m64*) itl1);
      const __m128 vblbr01 = _mm_loadh_pi(vblbr0, (const __m64*) ibl1);
      const __m128 vtltr23 = _mm_loadh_pi(vtltr2, (const __m64*) itl3);
      const __m128 vblbr23 = _mm_loadh_pi(vblbr2, (const __m64*) ibl3);
      const __m128 vtltr45 = _mm_loadh_pi(vtltr4, (const __m64*) itl5);
      const __m128 vblbr45 = _mm_loadh_pi(vblbr4, (const __m64*) ibl5);
      const __m128 vtltr67 = _mm_loadh_pi(vtltr6, (const __m64*) itl7);
      const __m128 vblbr67 = _mm_loadh_pi(vblbr6, (const __m64*) ibl7);

      const __m128 vldrd01 = _mm_sub_ps(vblbr01, vtltr01);
      const __m128 vldrd23 = _mm_sub_ps(vblbr23, vtltr23);
      const __m128 vldrd45 = _mm_sub_ps(vblbr45, vtltr45);
      const __m128 vldrd67 = _mm_sub_ps(vblbr67, vtltr67);

      const __m128 vld0123 = _mm_shuffle_ps(vldrd01, vldrd23, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vrd0123 = _mm_shuffle_ps(vldrd01, vldrd23, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vld4567 = _mm_shuffle_ps(vldrd45, vldrd67, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vrd4567 = _mm_shuffle_ps(vldrd45, vldrd67, _MM_SHUFFLE(3, 1, 3, 1));

      const __m128 vtl0123 = _mm_shuffle_ps(vtltr01, vtltr23, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vtr0123 = _mm_shuffle_ps(vtltr01, vtltr23, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vtl4567 = _mm_shuffle_ps(vtltr45, vtltr67, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vtr4567 = _mm_shuffle_ps(vtltr45, vtltr67, _MM_SHUFFLE(3, 1, 3, 1));

      const __m128 vl0123 = _mm_add_ps(vtl0123, _mm_mul_ps(vld0123, valphav0123));
      const __m128 vr0123 = _mm_add_ps(vtr0123, _mm_mul_ps(vrd0123, valphav0123));
      const __m128 vl4567 = _mm_add_ps(vtl4567, _mm_mul_ps(vld4567, valphav4567));
      const __m128 vr4567 = _mm_add_ps(vtr4567, _mm_mul_ps(vrd4567, valphav4567));

      const __m128 vd0123 = _mm_sub_ps(vr0123, vl0123);
      const __m128 vd4567 = _mm_sub_ps(vr4567, vl4567);

      const __m128 vo0123 = _mm_add_ps(vl0123, _mm_mul_ps(vd0123, valphah0123));
      const __m128 vo4567 = _mm_add_ps(vl4567, _mm_mul_ps(vd4567, valphah4567));

      _mm_storeu_ps(output + 0, vo0123);
      _mm_storeu_ps(output + 4, vo4567);
      output += 8;
    }

    for (; p >= 4; p -= 4) {
      const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
      const float* itl2 = (const float*) ((uintptr_t) i[4] + input_offset);
      const float* ibl2 = (const float*) ((uintptr_t) i[5] + input_offset);
      const float* itl3 = (const float*) ((uintptr_t) i[6] + input_offset);
      const float* ibl3 = (const float*) ((uintptr_t) i[7] + input_offset);
      i += 8;

      const __m128 vw0 = _mm_loadu_ps(w);
      const __m128 vw1 = _mm_loadu_ps(w + 4);
      w += 8;

      const __m128 vtltr0 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl0);
      const __m128 vblbr0 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl0);
      const __m128 vtltr2 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl2);
      const __m128 vblbr2 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl2);

      const __m128 valphah = _mm_shuffle_ps(vw0, vw1, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 valphav = _mm_shuffle_ps(vw0, vw1, _MM_SHUFFLE(3, 1, 3, 1));

      const __m128 vtltr01 = _mm_loadh_pi(vtltr0, (const __m64*) itl1);
      const __m128 vblbr01 = _mm_loadh_pi(vblbr0, (const __m64*) ibl1);
      const __m128 vtltr23 = _mm_loadh_pi(vtltr2, (const __m64*) itl3);
      const __m128 vblbr23 = _mm_loadh_pi(vblbr2, (const __m64*) ibl3);

      const __m128 vldrd01 = _mm_sub_ps(vblbr01, vtltr01);
      const __m128 vldrd23 = _mm_sub_ps(vblbr23, vtltr23);

      const __m128 vld = _mm_shuffle_ps(vldrd01, vldrd23, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vrd = _mm_shuffle_ps(vldrd01, vldrd23, _MM_SHUFFLE(3, 1, 3, 1));

      const __m128 vtl = _mm_shuffle_ps(vtltr01, vtltr23, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vtr = _mm_shuffle_ps(vtltr01, vtltr23, _MM_SHUFFLE(3, 1, 3, 1));

      const __m128 vl = _mm_add_ps(vtl, _mm_mul_ps(vld, valphav));
      const __m128 vr = _mm_add_ps(vtr, _mm_mul_ps(vrd, valphav));

      const __m128 vd = _mm_sub_ps(vr, vl);
      const __m128 vo = _mm_add_ps(vl, _mm_mul_ps(vd, valphah));

      _mm_storeu_ps(output, vo);
      output += 4;
    }

    if XNN_UNLIKELY(p != 0) {
      if (p & 2) {
        const __m128 vw = _mm_loadu_ps(w);
        w += 4;

        const __m128 valphah = _mm_shuffle_ps(vw, vw, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 valphav = _mm_shuffle_ps(vw, vw, _MM_SHUFFLE(3, 1, 3, 1));

        const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
        const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
        const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
        i += 4;

        const __m128 vtltr = _mm_loadh_pi(_mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl0), (const __m64*) itl1);
        const __m128 vblbr = _mm_loadh_pi(_mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl0), (const __m64*) ibl1);

        const __m128 vldrd = _mm_sub_ps(vblbr, vtltr);
        const __m128 vld = _mm_shuffle_ps(vldrd, vldrd, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 vrd = _mm_shuffle_ps(vldrd, vldrd, _MM_SHUFFLE(3, 1, 3, 1));

        const __m128 vtl = _mm_shuffle_ps(vtltr, vtltr, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 vtr = _mm_shuffle_ps(vtltr, vtltr, _MM_SHUFFLE(3, 1, 3, 1));

        const __m128 vl = _mm_add_ps(vtl, _mm_mul_ps(vld, valphav));
        const __m128 vr = _mm_add_ps(vtr, _mm_mul_ps(vrd, valphav));

        const __m128 vd = _mm_sub_ps(vr, vl);
        const __m128 vo = _mm_add_ps(vl, _mm_mul_ps(vd, valphah));

        _mm_storel_pi((__m64*) output, vo);
        output += 2;
      }

      if (p & 1) {
        // We are computing the following formula:
        //   result = (1 - alpha_h) * (1 - alpha_v) * top_left +
        //                 alpha_h  * (1 - alpha_v) * top_right +
        //            (1 - alpha_h) *      alpha_v  * bottom_left +
        //                 alpha_h  *      alpha_v  * bottom_right.
        //
        // Rearranging gives
        //   result =    left + alpha_h * (right        - left),
        // where
        //   left =  top_left + alpha_v * (bottom_left  - top_left),
        //  right = top_right + alpha_v * (bottom_right - top_right).

        const float alphah = *w;
        const __m128 valphav = _mm_load_ps1(w + 1);
        w += 2;

        const float* itl = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl = (const float*) ((uintptr_t) i[1] + input_offset);
        i += 2;

        const __m128 vtltr = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) itl);
        const __m128 vblbr = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) ibl);

        // Compute at once
        //    left_diff = bottom_left  - top_left
        //   right_diff = bottom_right - top_right
        const __m128 vldrd = _mm_sub_ps(vblbr, vtltr);
        const __m128 vlr = _mm_add_ps(vtltr, _mm_mul_ps(vldrd, valphav));

        // Extract them and compute the result.
        const float l = _mm_cvtss_f32(vlr);
        const float r = _mm_cvtss_f32(_mm_shuffle_ps(vlr, vlr, 1));

        *output++ = l + alphah * (r - l);
      }
    }

    input_offset += input_increment;
  } while (--channels != 0);
}

void xnn_f32_ibilinear_ukernel__sse_c8(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  do {
    const float* i0 = (const float*) ((uintptr_t) input[0] + input_offset);
    const float* i1 = (const float*) ((uintptr_t) input[1] + input_offset);
    const float* i2 = (const float*) ((uintptr_t) input[2] + input_offset);
    const float* i3 = (const float*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    __m128 valphahv = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) weights);
    valphahv = _mm_unpacklo_ps(valphahv, valphahv);
    const __m128 valphah = _mm_movelh_ps(valphahv, valphahv);
    const __m128 valphav = _mm_movehl_ps(valphahv, valphahv);
    weights += 2;

    size_t c = channels;
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const __m128 vtl0123 = _mm_loadu_ps(i0);
      const __m128 vtr0123 = _mm_loadu_ps(i1);
      const __m128 vbl0123 = _mm_loadu_ps(i2);
      const __m128 vbr0123 = _mm_loadu_ps(i3);
      const __m128 vtl4567 = _mm_loadu_ps(i0 + 4);
      const __m128 vtr4567 = _mm_loadu_ps(i1 + 4);
      const __m128 vbl4567 = _mm_loadu_ps(i2 + 4);
      const __m128 vbr4567 = _mm_loadu_ps(i3 + 4);
      i0 += 8;
      i1 += 8;
      i2 += 8;
      i3 += 8;

      const __m128 vtd0123 = _mm_sub_ps(vtr0123, vtl0123);
      const __m128 vbd0123 = _mm_sub_ps(vbr0123, vbl0123);
      const __m128 vtd4567 = _mm_sub_ps(vtr4567, vtl4567);
      const __m128 vbd4567 = _mm_sub_ps(vbr4567, vbl4567);

      const __m128 vt0123 = _mm_add_ps(vtl0123, _mm_mul_ps(vtd0123, valphah));
      const __m128 vb0123 = _mm_add_ps(vbl0123, _mm_mul_ps(vbd0123, valphah));
      const __m128 vt4567 = _mm_add_ps(vtl4567, _mm_mul_ps(vtd4567, valphah));
      const __m128 vb4567 = _mm_add_ps(vbl4567, _mm_mul_ps(vbd4567, valphah));

      const __m128 vd0123 = _mm_sub_ps(vb0123, vt0123);
      const __m128 vd4567 = _mm_sub_ps(vb4567, vt4567);

      const __m128 vo0123 = _mm_add_ps(vt0123, _mm_mul_ps(vd0123, valphav));
      const __m128 vo4567 = _mm_add_ps(vt4567, _mm_mul_ps(vd4567, valphav));

      _mm_storeu_ps(output, vo0123);
      _mm_storeu_ps(output + 4, vo4567);
      output += 8;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const __m128 vtl0123 = _mm_loadu_ps(i0);
      const __m128 vtr0123 = _mm_loadu_ps(i1);
      const __m128 vbl0123 = _mm_loadu_ps(i2);
      const __m128 vbr0123 = _mm_loadu_ps(i3);
      i0 += 4;
      i1 += 4;
      i2 += 4;
      i3 += 4;

      const __m128 vtd0123 = _mm_sub_ps(vtr0123, vtl0123);
      const __m128 vbd0123 = _mm_sub_ps(vbr0123, vbl0123);

      const __m128 vt0123 = _mm_add_ps(vtl0123, _mm_mul_ps(vtd0123, valphah));
      const __m128 vb0123 = _mm_add_ps(vbl0123, _mm_mul_ps(vbd0123, valphah));

      const __m128 vd0123 = _mm_sub_ps(vb0123, vt0123);

      const __m128 vo0123 = _mm_add_ps(vt0123, _mm_mul_ps(vd0123, valphav));

      _mm_storeu_ps(output, vo0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m128 vtl0123 = _mm_loadu_ps(i0);
      const __m128 vtr0123 = _mm_loadu_ps(i1);
      const __m128 vbl0123 = _mm_loadu_ps(i2);
      const __m128 vbr0123 = _mm_loadu_ps(i3);

      const __m128 vtd0123 = _mm_sub_ps(vtr0123, vtl0123);
      const __m128 vbd0123 = _mm_sub_ps(vbr0123, vbl0123);

      const __m128 vt0123 = _mm_add_ps(vtl0123, _mm_mul_ps(vtd0123, valphah));
      const __m128 vb0123 = _mm_add_ps(vbl0123, _mm_mul_ps(vbd0123, valphah));

      const __m128 vd0123 = _mm_sub_ps(vb0123, vt0123);

      __m128 vo0123 = _mm_add_ps(vt0123, _mm_mul_ps(vd0123, valphav));

      if (c & (2 * sizeof(float))) {
        _mm_storel_pi((__m64*) output, vo0123);
        vo0123 = _mm_movehl_ps(vo0123, vo0123);
        output += 2;
      }
      if (c & (1 * sizeof(float))) {
        _mm_store_ss(output, vo0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x8__sse_load1(
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

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_load_ps(w);
    __m128 vacc0x4567 = _mm_load_ps(w + 4);
    w += 8;

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
        const __m128 vb0123 = _mm_load_ps(w);
        const __m128 vb4567 = _mm_load_ps(w + 4);
        w += 8;

        const __m128 va0 = _mm_load1_ps(a0);
        a0 += 1;

        vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123));
        vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567));
        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x2c4__sse(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0c4 = _mm_load_ss(w);
    __m128 vacc0x1c4 = _mm_load_ss(w + 1);
    __m128 vacc1x0c4 = vacc0x0c4;
    __m128 vacc1x1c4 = vacc0x1c4;
    __m128 vacc2x0c4 = vacc0x0c4;
    __m128 vacc2x1c4 = vacc0x1c4;
    __m128 vacc3x0c4 = vacc0x0c4;
    __m128 vacc3x1c4 = vacc0x1c4;
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
      for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
        const __m128 va0 = _mm_loadu_ps(a0);
        a0 += 4;
        const __m128 va1 = _mm_loadu_ps(a1);
        a1 += 4;
        const __m128 va2 = _mm_loadu_ps(a2);
        a2 += 4;
        const __m128 va3 = _mm_loadu_ps(a3);
        a3 += 4;

        const __m128 vb0 = _mm_loadu_ps(w);
        const __m128 vb1 = _mm_loadu_ps(w + 4);
        w += 8;

        vacc0x0c4 = _mm_add_ps(vacc0x0c4, _mm_mul_ps(va0, vb0));
        vacc0x1c4 = _mm_add_ps(vacc0x1c4, _mm_mul_ps(va0, vb1));
        vacc1x0c4 = _mm_add_ps(vacc1x0c4, _mm_mul_ps(va1, vb0));
        vacc1x1c4 = _mm_add_ps(vacc1x1c4, _mm_mul_ps(va1, vb1));
        vacc2x0c4 = _mm_add_ps(vacc2x0c4, _mm_mul_ps(va2, vb0));
        vacc2x1c4 = _mm_add_ps(vacc2x1c4, _mm_mul_ps(va2, vb1));
        vacc3x0c4 = _mm_add_ps(vacc3x0c4, _mm_mul_ps(va3, vb0));
        vacc3x1c4 = _mm_add_ps(vacc3x1c4, _mm_mul_ps(va3, vb1));
      }
      if XNN_UNLIKELY(k != 0) {
        const __m128 va0 = _mm_loadu_ps(a0);
        const __m128 va1 = _mm_loadu_ps(a1);
        const __m128 va2 = _mm_loadu_ps(a2);
        const __m128 va3 = _mm_loadu_ps(a3);

        const __m128 vb0 = _mm_loadu_ps(w);
        const __m128 vb1 = _mm_loadu_ps(w + 4);
        w += 8;

        const __m128 vmask0 = _mm_cmpeq_ps(_mm_setzero_ps(), vb0);
        const __m128 vmask1 = _mm_cmpeq_ps(_mm_setzero_ps(), vb1);

        vacc0x0c4 = _mm_add_ps(vacc0x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va0), vb0));
        vacc0x1c4 = _mm_add_ps(vacc0x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va0), vb1));
        vacc1x0c4 = _mm_add_ps(vacc1x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va1), vb0));
        vacc1x1c4 = _mm_add_ps(vacc1x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va1), vb1));
        vacc2x0c4 = _mm_add_ps(vacc2x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va2), vb0));
        vacc2x1c4 = _mm_add_ps(vacc2x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va2), vb1));
        vacc3x0c4 = _mm_add_ps(vacc3x0c4, _mm_mul_ps(_mm_andnot_ps(vmask0, va3), vb0));
        vacc3x1c4 = _mm_add_ps(vacc3x1c4, _mm_mul_ps(_mm_andnot_ps(vmask1, va3), vb1));
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m128 vacc0x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc0x0c4, vacc0x1c4), _mm_unpackhi_ps(vacc0x0c4, vacc0x1c4));
    const __m128 vacc1x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc1x0c4, vacc1x1c4), _mm_unpackhi_ps(vacc1x0c4, vacc1x1c4));
    const __m128 vacc2x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc2x0c4, vacc2x1c4), _mm_unpackhi_ps(vacc2x0c4, vacc2x1c4));
    const __m128 vacc3x01c2 = _mm_add_ps(_mm_unpacklo_ps(vacc3x0c4, vacc3x1c4), _mm_unpackhi_ps(vacc3x0c4, vacc3x1c4));

    __m128 vacc01x01 = _mm_add_ps(_mm_movelh_ps(vacc0x01c2, vacc1x01c2), _mm_movehl_ps(vacc1x01c2, vacc0x01c2));
    __m128 vacc23x01 = _mm_add_ps(_mm_movelh_ps(vacc2x01c2, vacc3x01c2), _mm_movehl_ps(vacc3x01c2, vacc2x01c2));

    vacc01x01 = _mm_min_ps(vacc01x01, vmax);
    vacc23x01 = _mm_min_ps(vacc23x01, vmax);

    vacc01x01 = _mm_max_ps(vacc01x01, vmin);
    vacc23x01 = _mm_max_ps(vacc23x01, vmin);

    if XNN_LIKELY(nc >= 2) {
      _mm_storeh_pi((__m64*) c3, vacc23x01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm_storel_pi((__m64*) c2, vacc23x01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm_storeh_pi((__m64*) c1, vacc01x01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm_storel_pi((__m64*) c0, vacc01x01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      assert(nc == 1);
      _mm_store_ss(c3, _mm_movehl_ps(vacc23x01, vacc23x01));
      _mm_store_ss(c2, vacc23x01);
      _mm_store_ss(c1, _mm_movehl_ps(vacc01x01, vacc01x01));
      _mm_store_ss(c0, vacc01x01);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x8__sse_load1(
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

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_load_ps(w);
    __m128 vacc0x4567 = _mm_load_ps(w + 4);
    __m128 vacc1x0123 = vacc0x0123;
    __m128 vacc1x4567 = vacc0x4567;
    __m128 vacc2x0123 = vacc0x0123;
    __m128 vacc2x4567 = vacc0x4567;
    __m128 vacc3x0123 = vacc0x0123;
    __m128 vacc3x4567 = vacc0x4567;
    w += 8;

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
        const __m128 vb0123 = _mm_load_ps(w);
        const __m128 vb4567 = _mm_load_ps(w + 4);
        w += 8;

        const __m128 va0 = _mm_load1_ps(a0);
        a0 += 1;
        const __m128 va1 = _mm_load1_ps(a1);
        a1 += 1;
        const __m128 va2 = _mm_load1_ps(a2);
        a2 += 1;
        const __m128 va3 = _mm_load1_ps(a3);
        a3 += 1;

        vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123));
        vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567));
        vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1, vb0123));
        vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1, vb4567));
        vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2, vb0123));
        vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2, vb4567));
        vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3, vb0123));
        vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3, vb4567));
        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc1x0123 = _mm_min_ps(vacc1x0123, vmax);
    vacc2x0123 = _mm_min_ps(vacc2x0123, vmax);
    vacc3x0123 = _mm_min_ps(vacc3x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);
    vacc1x4567 = _mm_min_ps(vacc1x4567, vmax);
    vacc2x4567 = _mm_min_ps(vacc2x4567, vmax);
    vacc3x4567 = _mm_min_ps(vacc3x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc1x0123 = _mm_max_ps(vacc1x0123, vmin);
    vacc2x0123 = _mm_max_ps(vacc2x0123, vmin);
    vacc3x0123 = _mm_max_ps(vacc3x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);
    vacc1x4567 = _mm_max_ps(vacc1x4567, vmin);
    vacc2x4567 = _mm_max_ps(vacc2x4567, vmin);
    vacc3x4567 = _mm_max_ps(vacc3x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c3, vacc3x0123);
      _mm_storeu_ps(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_ps(c2, vacc2x0123);
      _mm_storeu_ps(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_ps(c1, vacc1x0123);
      _mm_storeu_ps(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(voutput_max);
  XNN_FORCE_REALIZATION(voutput_min);

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
      for (; c >= 4; c -= 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vi8 = _mm_loadu_ps(i8);
        i8 += 4;

        const __m128 vmax018 = _mm_max_ps(_mm_max_ps(vi0, vi1), vi8);
        const __m128 vmax23 = _mm_max_ps(vi2, vi3);
        const __m128 vmax45 = _mm_max_ps(vi4, vi5);
        const __m128 vmax67 = _mm_max_ps(vi6, vi7);

        const __m128 vmax2345 = _mm_max_ps(vmax23, vmax45);
        const __m128 vmax01678 = _mm_max_ps(vmax018, vmax67);
        const __m128 vmax = _mm_max_ps(vmax2345, vmax01678);
        const __m128 vout = _mm_max_ps(_mm_min_ps(vmax, voutput_max), voutput_min);

        _mm_storeu_ps(o, vout);
        o += 4;
      }
      if (c != 0) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vi8 = _mm_loadu_ps(i8);
        i8 += 4;

        const __m128 vmax018 = _mm_max_ps(_mm_max_ps(vi0, vi1), vi8);
        const __m128 vmax23 = _mm_max_ps(vi2, vi3);
        const __m128 vmax45 = _mm_max_ps(vi4, vi5);
        const __m128 vmax67 = _mm_max_ps(vi6, vi7);

        const __m128 vmax2345 = _mm_max_ps(vmax23, vmax45);
        const __m128 vmax01678 = _mm_max_ps(vmax018, vmax67);
        const __m128 vmax = _mm_max_ps(vmax2345, vmax01678);
        __m128 vout = _mm_max_ps(_mm_min_ps(vmax, voutput_max), voutput_min);

        if (c & 2) {
          _mm_storel_pi((__m64*) o, vout);
          o += 2;
          vout = _mm_movehl_ps(vout, vout);
        }
        if (c & 1) {
          _mm_store_ss(o, vout);
          o += 1;
        }
      }
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
      for (; c >= 4; c -= 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vo = _mm_loadu_ps(o);

        const __m128 vmax01 = _mm_max_ps(_mm_max_ps(vi0, vi1), vo);
        const __m128 vmax23 = _mm_max_ps(vi2, vi3);
        const __m128 vmax45 = _mm_max_ps(vi4, vi5);
        const __m128 vmax67 = _mm_max_ps(vi6, vi7);

        const __m128 vmax2345 = _mm_max_ps(vmax23, vmax45);
        const __m128 vmax0167 = _mm_max_ps(vmax01, vmax67);
        const __m128 vmax = _mm_max_ps(vmax2345, vmax0167);
        const __m128 vout = _mm_max_ps(_mm_min_ps(vmax, voutput_max), voutput_min);

        _mm_storeu_ps(o, vout);
        o += 4;
      }
      if (c != 0) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        const __m128 vi1 = _mm_loadu_ps(i1);
        const __m128 vi2 = _mm_loadu_ps(i2);
        const __m128 vi3 = _mm_loadu_ps(i3);
        const __m128 vi4 = _mm_loadu_ps(i4);
        const __m128 vi5 = _mm_loadu_ps(i5);
        const __m128 vi6 = _mm_loadu_ps(i6);
        const __m128 vi7 = _mm_loadu_ps(i7);
        const __m128 vo = _mm_loadu_ps(o);

        const __m128 vmax01 = _mm_max_ps(_mm_max_ps(vi0, vi1), vo);
        const __m128 vmax23 = _mm_max_ps(vi2, vi3);
        const __m128 vmax45 = _mm_max_ps(vi4, vi5);
        const __m128 vmax67 = _mm_max_ps(vi6, vi7);

        const __m128 vmax2345 = _mm_max_ps(vmax23, vmax45);
        const __m128 vmax0167 = _mm_max_ps(vmax01, vmax67);
        const __m128 vmax = _mm_max_ps(vmax2345, vmax0167);
        __m128 vout = _mm_max_ps(_mm_min_ps(vmax, voutput_max), voutput_min);

        if (c & 2) {
          _mm_storel_pi((__m64*) o, vout);
          o += 2;
          vout = _mm_movehl_ps(vout, vout);
        }
        if (c & 1) {
          _mm_store_ss(o, vout);
          o += 1;
        }
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_pavgpool_minmax_ukernel_9p8x__sse_c4(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_max);
  XNN_FORCE_REALIZATION(voutput_min);

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
      for (size_t c = 0; c < channels; c += 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vi8 = _mm_loadu_ps(i8);
        i8 += 4;

        const __m128 vsum01 = _mm_add_ps(vi0, vi1);
        const __m128 vsum23 = _mm_add_ps(vi2, vi3);
        const __m128 vsum45 = _mm_add_ps(vi4, vi5);
        const __m128 vsum67 = _mm_add_ps(vi6, vi7);
        const __m128 vsum018 = _mm_add_ps(vsum01, vi8);
        const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
        const __m128 vsum01678 = _mm_add_ps(vsum018, vsum67);
        const __m128 vsum = _mm_add_ps(vsum2345, vsum01678);

        _mm_store_ps(b, vsum); b += 4;
      }
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
      for (size_t c = 0; c < channels; c += 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vacc = _mm_load_ps(b);

        const __m128 vsum01 = _mm_add_ps(vi0, vi1);
        const __m128 vsum23 = _mm_add_ps(vi2, vi3);
        const __m128 vsum45 = _mm_add_ps(vi4, vi5);
        const __m128 vsum67 = _mm_add_ps(vi6, vi7);
        const __m128 vsum01a = _mm_add_ps(vsum01, vacc);
        const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
        const __m128 vsum0167a = _mm_add_ps(vsum01a, vsum67);
        const __m128 vsum = _mm_add_ps(vsum2345, vsum0167a);

        _mm_store_ps(b, vsum); b += 4;
      }
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

      const __m128 vmultiplier = _mm_load1_ps(multiplier);
      multiplier += 1;

      size_t c = channels;
      float* b = buffer;
      while (c >= 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vacc = _mm_load_ps(b);
        b += 4;

        const __m128 vsum01 = _mm_add_ps(vi0, vi1);
        const __m128 vsum23 = _mm_add_ps(vi2, vi3);
        const __m128 vsum45 = _mm_add_ps(vi4, vi5);
        const __m128 vsum67 = _mm_add_ps(vi6, vi7);
        const __m128 vsum01a = _mm_add_ps(vsum01, vacc);
        const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
        const __m128 vsum0167a = _mm_add_ps(vsum01a, vsum67);
        const __m128 vsum = _mm_add_ps(vsum2345, vsum0167a);

        __m128 vout = _mm_mul_ps(vsum, vmultiplier);
        vout = _mm_max_ps(vout, voutput_min);
        vout = _mm_min_ps(vout, voutput_max);

        _mm_storeu_ps(output, vout);
        output += 4;

        c -= 4;
      }
      if (c != 0) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        const __m128 vi1 = _mm_loadu_ps(i1);
        const __m128 vi2 = _mm_loadu_ps(i2);
        const __m128 vi3 = _mm_loadu_ps(i3);
        const __m128 vi4 = _mm_loadu_ps(i4);
        const __m128 vi5 = _mm_loadu_ps(i5);
        const __m128 vi6 = _mm_loadu_ps(i6);
        const __m128 vi7 = _mm_loadu_ps(i7);
        const __m128 vacc = _mm_load_ps(b);

        const __m128 vsum01 = _mm_add_ps(vi0, vi1);
        const __m128 vsum23 = _mm_add_ps(vi2, vi3);
        const __m128 vsum45 = _mm_add_ps(vi4, vi5);
        const __m128 vsum67 = _mm_add_ps(vi6, vi7);
        const __m128 vsum01a = _mm_add_ps(vsum01, vacc);
        const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
        const __m128 vsum0167a = _mm_add_ps(vsum01a, vsum67);
        const __m128 vsum = _mm_add_ps(vsum2345, vsum0167a);

        __m128 vout = _mm_mul_ps(vsum, vmultiplier);
        vout = _mm_max_ps(vout, voutput_min);
        vout = _mm_min_ps(vout, voutput_max);

        if (c & 2) {
          _mm_storel_pi((__m64*) output, vout);
          vout = _mm_movehl_ps(vout, vout);
          output += 2;
        }
        if (c & 1) {
          _mm_store_ss(output, vout);
          output += 1;
        }
      }
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_pavgpool_minmax_ukernel_9x__sse_c4(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_max);
  XNN_FORCE_REALIZATION(voutput_min);

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

    const __m128 vmultiplier = _mm_load1_ps(multiplier);
    multiplier += 1;

    size_t c = channels;
    while (c >= 4) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1 = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2 = _mm_loadu_ps(i2);
      i2 += 4;
      const __m128 vi3 = _mm_loadu_ps(i3);
      i3 += 4;
      const __m128 vi4 = _mm_loadu_ps(i4);
      i4 += 4;
      const __m128 vi5 = _mm_loadu_ps(i5);
      i5 += 4;
      const __m128 vi6 = _mm_loadu_ps(i6);
      i6 += 4;
      const __m128 vi7 = _mm_loadu_ps(i7);
      i7 += 4;
      const __m128 vi8 = _mm_loadu_ps(i8);
      i8 += 4;

      const __m128 vsum018 = _mm_add_ps(_mm_add_ps(vi0, vi1), vi8);
      const __m128 vsum23 = _mm_add_ps(vi2, vi3);
      const __m128 vsum45 = _mm_add_ps(vi4, vi5);
      const __m128 vsum67 = _mm_add_ps(vi6, vi7);

      const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
      const __m128 vsum01678 = _mm_add_ps(vsum018, vsum67);
      const __m128 vsum = _mm_add_ps(vsum2345, vsum01678);

      __m128 vout = _mm_mul_ps(vsum, vmultiplier);
      vout = _mm_max_ps(vout, voutput_min);
      vout = _mm_min_ps(vout, voutput_max);

      _mm_storeu_ps(output, vout); output += 4;

      c -= 4;
    }
    if (c != 0) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      const __m128 vi1 = _mm_loadu_ps(i1);
      const __m128 vi2 = _mm_loadu_ps(i2);
      const __m128 vi3 = _mm_loadu_ps(i3);
      const __m128 vi4 = _mm_loadu_ps(i4);
      const __m128 vi5 = _mm_loadu_ps(i5);
      const __m128 vi6 = _mm_loadu_ps(i6);
      const __m128 vi7 = _mm_loadu_ps(i7);
      const __m128 vi8 = _mm_loadu_ps(i8);

      const __m128 vsum01 = _mm_add_ps(vi0, vi1);
      const __m128 vsum23 = _mm_add_ps(vi2, vi3);
      const __m128 vsum45 = _mm_add_ps(vi4, vi5);
      const __m128 vsum67 = _mm_add_ps(vi6, vi7);
      const __m128 vsum018 = _mm_add_ps(vsum01, vi8);
      const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
      const __m128 vsum01678 = _mm_add_ps(vsum018, vsum67);
      const __m128 vsum = _mm_add_ps(vsum2345, vsum01678);

      __m128 vout = _mm_mul_ps(vsum, vmultiplier);
      vout = _mm_max_ps(vout, voutput_min);
      vout = _mm_min_ps(vout, voutput_max);

      if (c & 2) {
        _mm_storel_pi((__m64*) output, vout);
        vout = _mm_movehl_ps(vout, vout);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vout);
        output += 1;
      }
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_rdsum_ukernel_7p7x__sse_c16(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128 vmin = _mm_set1_ps(params->scalar.min);
  const __m128 vmax = _mm_set1_ps(params->scalar.max);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 16; channels -= 16) {
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);

    __m128 vacc0 = _mm_setzero_ps();
    __m128 vacc1 = _mm_setzero_ps();
    __m128 vacc2 = _mm_setzero_ps();
    __m128 vacc3 = _mm_setzero_ps();

    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      __m128 vin0;
      __m128 vin1;
      __m128 vin2;
      __m128 vin3;
      vin0 = _mm_loadu_ps(&i0[0]);
      vin1 = _mm_loadu_ps(&i0[4]);
      vin2 = _mm_loadu_ps(&i0[8]);
      vin3 = _mm_loadu_ps(&i0[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i1[0]);
      vin1 = _mm_loadu_ps(&i1[4]);
      vin2 = _mm_loadu_ps(&i1[8]);
      vin3 = _mm_loadu_ps(&i1[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i2[0]);
      vin1 = _mm_loadu_ps(&i2[4]);
      vin2 = _mm_loadu_ps(&i2[8]);
      vin3 = _mm_loadu_ps(&i2[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i3[0]);
      vin1 = _mm_loadu_ps(&i3[4]);
      vin2 = _mm_loadu_ps(&i3[8]);
      vin3 = _mm_loadu_ps(&i3[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i4[0]);
      vin1 = _mm_loadu_ps(&i4[4]);
      vin2 = _mm_loadu_ps(&i4[8]);
      vin3 = _mm_loadu_ps(&i4[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i5[0]);
      vin1 = _mm_loadu_ps(&i5[4]);
      vin2 = _mm_loadu_ps(&i5[8]);
      vin3 = _mm_loadu_ps(&i5[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i6[0]);
      vin1 = _mm_loadu_ps(&i6[4]);
      vin2 = _mm_loadu_ps(&i6[8]);
      vin3 = _mm_loadu_ps(&i6[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = _mm_mul_ps(vacc0, vscale);
    vacc0 = _mm_max_ps(vacc0, vmin);
    vacc0 = _mm_min_ps(vacc0, vmax);
    vacc1 = _mm_mul_ps(vacc1, vscale);
    vacc1 = _mm_max_ps(vacc1, vmin);
    vacc1 = _mm_min_ps(vacc1, vmax);
    vacc2 = _mm_mul_ps(vacc2, vscale);
    vacc2 = _mm_max_ps(vacc2, vmin);
    vacc2 = _mm_min_ps(vacc2, vmax);
    vacc3 = _mm_mul_ps(vacc3, vscale);
    vacc3 = _mm_max_ps(vacc3, vmin);
    vacc3 = _mm_min_ps(vacc3, vmax);

    const float* o = output;
    __m128 vo0 = _mm_loadu_ps(o); o += 4;
    __m128 vo1 = _mm_loadu_ps(o); o += 4;
    __m128 vo2 = _mm_loadu_ps(o); o += 4;
    __m128 vo3 = _mm_loadu_ps(o); o += 4;
    vacc0 = _mm_add_ps(vo0, vacc0);
    vacc1 = _mm_add_ps(vo1, vacc1);
    vacc2 = _mm_add_ps(vo2, vacc2);
    vacc3 = _mm_add_ps(vo3, vacc3);
    _mm_storeu_ps(output, vacc0); output += 4;
    _mm_storeu_ps(output, vacc1); output += 4;
    _mm_storeu_ps(output, vacc2); output += 4;
    _mm_storeu_ps(output, vacc3); output += 4;

    input = (const float*) ((uintptr_t) input + 16 * sizeof(float));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);
    __m128 vacc[4];
    vacc[0] = _mm_setzero_ps();
    vacc[1] = _mm_setzero_ps();
    vacc[2] = _mm_setzero_ps();
    vacc[3] = _mm_setzero_ps();

    size_t num_chunks = round_up_po2(channels, 4) >> 2;
    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      for (int i = 0; i < num_chunks; ++i) {
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i0[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i1[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i2[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i3[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i4[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i5[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i6[i*4]), vacc[i]);
      }
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    for (int i = 0; i < num_chunks; ++i) {
      vacc[i] = _mm_mul_ps(vacc[i], vscale);
      vacc[i] = _mm_max_ps(vacc[i], vmin);
      vacc[i] = _mm_min_ps(vacc[i], vmax);
    }

    __m128 vo[4];
    const float* o = output;
    for (int i = 0; i < channels >> 2; ++i) {
      vo[i] = _mm_loadu_ps(o); o += 4;
    }
    for (int i = 0; i < channels >> 2; ++i) {
      vacc[i] = _mm_add_ps(vo[i], vacc[i]);
    }
    for (int i = 0; i < channels >> 2; ++i) {
      _mm_storeu_ps(output, vacc[i]); output += 4;
    }
    const size_t pos = channels >> 2;
    channels &= 0x3;
    __m128 vout = vacc[pos];
    if (channels & 2) {
      __m128 vo = _mm_loadl_pi(vscale, (__m64*) output);
      _mm_storel_pi((__m64*) output, _mm_add_ps(vo, vout));
      vout = _mm_movehl_ps(vout, vout);
      output += 2;
    }
    if (channels & 1) {
      __m128 vo = _mm_load_ss(output);
      _mm_store_ss(output, _mm_add_ps(vo, vout));
    }
  }
}

void xnn_f32_rmax_ukernel__sse_u16_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m128 vmax0 = _mm_load_ss(input);
  vmax0 = _mm_shuffle_ps(vmax0, vmax0, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 vmax1 = vmax0;
  __m128 vmax2 = vmax0;
  __m128 vmax3 = vmax0;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m128 vt0 = _mm_loadu_ps(input);
    const __m128 vt1 = _mm_loadu_ps(input + 4);
    const __m128 vt2 = _mm_loadu_ps(input + 8);
    const __m128 vt3 = _mm_loadu_ps(input + 12);
    input += 16;

    vmax0 = _mm_max_ps(vmax0, vt0);
    vmax1 = _mm_max_ps(vmax1, vt1);
    vmax2 = _mm_max_ps(vmax2, vt2);
    vmax3 = _mm_max_ps(vmax3, vt3);
  }
  vmax0 = _mm_max_ps(vmax0, vmax1);
  vmax2 = _mm_max_ps(vmax2, vmax3);
  vmax0 = _mm_max_ps(vmax0, vmax2);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vt = _mm_loadu_ps(input);
    input += 4;

    vmax0 = _mm_max_ps(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m128 vt = _mm_load_ss(input);
      input += 1;
      vmax0 = _mm_max_ss(vmax0, vt);
      batch -= sizeof(float);
    } while (batch != 0);
  }
  vmax0 = _mm_max_ps(vmax0, _mm_movehl_ps(vmax0, vmax0));
  vmax0 = _mm_max_ss(vmax0, _mm_shuffle_ps(vmax0, vmax0, _MM_SHUFFLE(1, 1, 1, 1)));
  _mm_store_ss(output, vmax0);
}

void xnn_f32_rminmax_ukernel__sse_u16_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m128 vmin0 = _mm_load_ss(input);
  vmin0 = _mm_shuffle_ps(vmin0, vmin0, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 vmax0 = vmin0;
  __m128 vmin1 = vmin0;
  __m128 vmax1 = vmax0;
  __m128 vmin2 = vmin0;
  __m128 vmax2 = vmax0;
  __m128 vmin3 = vmin0;
  __m128 vmax3 = vmax0;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m128 vt0 = _mm_loadu_ps(input);
    const __m128 vt1 = _mm_loadu_ps(input + 4);
    const __m128 vt2 = _mm_loadu_ps(input + 8);
    const __m128 vt3 = _mm_loadu_ps(input + 12);
    input += 16;

    vmin0 = _mm_min_ps(vmin0, vt0);
    vmax0 = _mm_max_ps(vmax0, vt0);
    vmin1 = _mm_min_ps(vmin1, vt1);
    vmax1 = _mm_max_ps(vmax1, vt1);
    vmin2 = _mm_min_ps(vmin2, vt2);
    vmax2 = _mm_max_ps(vmax2, vt2);
    vmin3 = _mm_min_ps(vmin3, vt3);
    vmax3 = _mm_max_ps(vmax3, vt3);
  }
  vmin0 = _mm_min_ps(vmin0, vmin1);
  vmax0 = _mm_max_ps(vmax0, vmax1);
  vmin2 = _mm_min_ps(vmin2, vmin3);
  vmax2 = _mm_max_ps(vmax2, vmax3);
  vmin0 = _mm_min_ps(vmin0, vmin2);
  vmax0 = _mm_max_ps(vmax0, vmax2);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vt = _mm_loadu_ps(input);
    input += 4;

    vmin0 = _mm_min_ps(vmin0, vt);
    vmax0 = _mm_max_ps(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m128 vt = _mm_load_ss(input);
      input += 1;
      vmin0 = _mm_min_ss(vmin0, vt);
      vmax0 = _mm_max_ss(vmax0, vt);
      batch -= sizeof(float);
    } while (batch != 0);
  }
  vmin0 = _mm_min_ps(vmin0, _mm_movehl_ps(vmin0, vmin0));
  vmax0 = _mm_max_ps(vmax0, _mm_movehl_ps(vmax0, vmax0));
  vmin0 = _mm_min_ss(vmin0, _mm_shuffle_ps(vmin0, vmin0, _MM_SHUFFLE(1, 1, 1, 1)));
  vmax0 = _mm_max_ss(vmax0, _mm_shuffle_ps(vmax0, vmax0, _MM_SHUFFLE(1, 1, 1, 1)));
  _mm_store_ss(output, vmin0);
  _mm_store_ss(output + 1, vmax0);
}

void xnn_f32_rsum_ukernel__sse_u16_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m128 vacc0 = _mm_setzero_ps();
  __m128 vacc1 = _mm_setzero_ps();
  __m128 vacc2 = _mm_setzero_ps();
  __m128 vacc3 = _mm_setzero_ps();
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m128 vt0 = _mm_loadu_ps(input);
    const __m128 vt1 = _mm_loadu_ps(input + 4);
    const __m128 vt2 = _mm_loadu_ps(input + 8);
    const __m128 vt3 = _mm_loadu_ps(input + 12);
    input += 16;

    vacc0 = _mm_add_ps(vacc0, vt0);
    vacc1 = _mm_add_ps(vacc1, vt1);
    vacc2 = _mm_add_ps(vacc2, vt2);
    vacc3 = _mm_add_ps(vacc3, vt3);
  }
  vacc0 = _mm_add_ps(vacc0, vacc1);
  vacc2 = _mm_add_ps(vacc2, vacc3);
  vacc0 = _mm_add_ps(vacc0, vacc2);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vt = _mm_loadu_ps(input);
    input += 4;

    vacc0 = _mm_add_ps(vacc0, vt);
  }
  vacc0 = _mm_add_ps(vacc0, _mm_movehl_ps(vacc0, vacc0));
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m128 vt = _mm_load_ss(input);
      input += 1;
      vacc0 = _mm_add_ss(vacc0, vt);
      batch -= sizeof(float);
    } while (batch != 0);
  }
  vacc0 = _mm_add_ss(vacc0, _mm_shuffle_ps(vacc0, vacc0, _MM_SHUFFLE(1, 1, 1, 1)));
  vacc0 = _mm_mul_ss(vacc0, _mm_load_ss(&params->scalar.scale));
  vacc0 = _mm_max_ss(vacc0, _mm_load_ss(&params->scalar.min));
  vacc0 = _mm_min_ss(vacc0, _mm_load_ss(&params->scalar.max));
  *output += _mm_cvtss_f32(vacc0);
}

void xnn_f32_spmm_minmax_ukernel_32x1__sse(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  size_t output_decrement = output_stride * nc - 32 * sizeof(float);
  while XNN_LIKELY(mc >= 32 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      __m128 vacc0123 = _mm_load1_ps(w); w += 1;
      __m128 vacc4567 = vacc0123;
      __m128 vacc89AB = vacc0123;
      __m128 vaccCDEF = vacc0123;
      __m128 vaccGHIJ = vacc0123;
      __m128 vaccKLMN = vacc0123;
      __m128 vaccOPQR = vacc0123;
      __m128 vaccSTUV = vacc0123;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const __m128 vi0123 = _mm_loadu_ps(input);
          const __m128 vi4567 = _mm_loadu_ps(input + 4);
          const __m128 vi89AB = _mm_loadu_ps(input + 8);
          const __m128 viCDEF = _mm_loadu_ps(input + 12);
          const __m128 viGHIJ = _mm_loadu_ps(input + 16);
          const __m128 viKLMN = _mm_loadu_ps(input + 20);
          const __m128 viOPQR = _mm_loadu_ps(input + 24);
          const __m128 viSTUV = _mm_loadu_ps(input + 28);
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const __m128 vw = _mm_load1_ps(w); w += 1;
          vacc0123 = _mm_add_ps(vacc0123, _mm_mul_ps(vi0123, vw));
          vacc4567 = _mm_add_ps(vacc4567, _mm_mul_ps(vi4567, vw));
          vacc89AB = _mm_add_ps(vacc89AB, _mm_mul_ps(vi89AB, vw));
          vaccCDEF = _mm_add_ps(vaccCDEF, _mm_mul_ps(viCDEF, vw));
          vaccGHIJ = _mm_add_ps(vaccGHIJ, _mm_mul_ps(viGHIJ, vw));
          vaccKLMN = _mm_add_ps(vaccKLMN, _mm_mul_ps(viKLMN, vw));
          vaccOPQR = _mm_add_ps(vaccOPQR, _mm_mul_ps(viOPQR, vw));
          vaccSTUV = _mm_add_ps(vaccSTUV, _mm_mul_ps(viSTUV, vw));
        } while (--nnz != 0);
      }
      __m128 vout0123 = _mm_min_ps(vacc0123, vmax);
      __m128 vout4567 = _mm_min_ps(vacc4567, vmax);
      __m128 vout89AB = _mm_min_ps(vacc89AB, vmax);
      __m128 voutCDEF = _mm_min_ps(vaccCDEF, vmax);
      __m128 voutGHIJ = _mm_min_ps(vaccGHIJ, vmax);
      __m128 voutKLMN = _mm_min_ps(vaccKLMN, vmax);
      __m128 voutOPQR = _mm_min_ps(vaccOPQR, vmax);
      __m128 voutSTUV = _mm_min_ps(vaccSTUV, vmax);
      vout0123 = _mm_max_ps(vout0123, vmin);
      vout4567 = _mm_max_ps(vout4567, vmin);
      vout89AB = _mm_max_ps(vout89AB, vmin);
      voutCDEF = _mm_max_ps(voutCDEF, vmin);
      voutGHIJ = _mm_max_ps(voutGHIJ, vmin);
      voutKLMN = _mm_max_ps(voutKLMN, vmin);
      voutOPQR = _mm_max_ps(voutOPQR, vmin);
      voutSTUV = _mm_max_ps(voutSTUV, vmin);
      _mm_storeu_ps(output, vout0123);
      _mm_storeu_ps(output + 4, vout4567);
      _mm_storeu_ps(output + 8, vout89AB);
      _mm_storeu_ps(output + 12, voutCDEF);
      _mm_storeu_ps(output + 16, voutGHIJ);
      _mm_storeu_ps(output + 20, voutKLMN);
      _mm_storeu_ps(output + 24, voutOPQR);
      _mm_storeu_ps(output + 28, voutSTUV);
      output = (float*restrict) ((uintptr_t) output + output_stride);
    } while (--n != 0);
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 32;
    mc -= 32 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        __m128 vacc0123 = _mm_load1_ps(w); w += 1;
        __m128 vacc4567 = vacc0123;
        __m128 vacc89AB = vacc0123;
        __m128 vaccCDEF = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const __m128 vi0123 = _mm_loadu_ps(input);
            const __m128 vi4567 = _mm_loadu_ps(input + 4);
            const __m128 vi89AB = _mm_loadu_ps(input + 8);
            const __m128 viCDEF = _mm_loadu_ps(input + 12);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const __m128 vw = _mm_load1_ps(w); w += 1;
            vacc0123 = _mm_add_ps(vacc0123, _mm_mul_ps(vi0123, vw));
            vacc4567 = _mm_add_ps(vacc4567, _mm_mul_ps(vi4567, vw));
            vacc89AB = _mm_add_ps(vacc89AB, _mm_mul_ps(vi89AB, vw));
            vaccCDEF = _mm_add_ps(vaccCDEF, _mm_mul_ps(viCDEF, vw));
          } while (--nnz != 0);
        }
        __m128 vout0123 = _mm_min_ps(vacc0123, vmax);
        __m128 vout4567 = _mm_min_ps(vacc4567, vmax);
        __m128 vout89AB = _mm_min_ps(vacc89AB, vmax);
        __m128 voutCDEF = _mm_min_ps(vaccCDEF, vmax);
        vout0123 = _mm_max_ps(vout0123, vmin);
        vout4567 = _mm_max_ps(vout4567, vmin);
        vout89AB = _mm_max_ps(vout89AB, vmin);
        voutCDEF = _mm_max_ps(voutCDEF, vmin);
        _mm_storeu_ps(output, vout0123);
        _mm_storeu_ps(output + 4, vout4567);
        _mm_storeu_ps(output + 8, vout89AB);
        _mm_storeu_ps(output + 12, voutCDEF);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        __m128 vacc0123 = _mm_load1_ps(w); w += 1;
        __m128 vacc4567 = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const __m128 vi0123 = _mm_loadu_ps(input);
            const __m128 vi4567 = _mm_loadu_ps(input + 4);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const __m128 vw = _mm_load1_ps(w); w += 1;
            vacc0123 = _mm_add_ps(vacc0123, _mm_mul_ps(vi0123, vw));
            vacc4567 = _mm_add_ps(vacc4567, _mm_mul_ps(vi4567, vw));
          } while (--nnz != 0);
        }
        __m128 vout0123 = _mm_min_ps(vacc0123, vmax);
        __m128 vout4567 = _mm_min_ps(vacc4567, vmax);
        vout0123 = _mm_max_ps(vout0123, vmin);
        vout4567 = _mm_max_ps(vout4567, vmin);
        _mm_storeu_ps(output, vout0123);
        _mm_storeu_ps(output + 4, vout4567);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 8;
    }
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        __m128 vacc0123 = _mm_load1_ps(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const __m128 vi0123 = _mm_loadu_ps(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const __m128 vw = _mm_load1_ps(w); w += 1;
            vacc0123 = _mm_add_ps(vacc0123, _mm_mul_ps(vi0123, vw));
          } while (--nnz != 0);
        }
        __m128 vout0123 = _mm_min_ps(vacc0123, vmax);
        vout0123 = _mm_max_ps(vout0123, vmin);
        _mm_storeu_ps(output, vout0123);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        __m128 vacc01 = _mm_load_ss(w); w += 1;
        vacc01 = _mm_unpacklo_ps(vacc01, vacc01);
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const __m128 vi01 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            __m128 vw = _mm_load_ss(w); w += 1;
            vw = _mm_unpacklo_ps(vw, vw);
            vacc01 = _mm_add_ps(vacc01, _mm_mul_ps(vi01, vw));
          } while (--nnz != 0);
        }
        __m128 vout01 = _mm_min_ps(vacc01, vmax);
        vout01 = _mm_max_ps(vout01, vmin);
        _mm_storel_pi((__m64*) output, vout01);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        __m128 vacc0 = _mm_load_ss(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const __m128 vi0 = _mm_load_ss(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const __m128 vw = _mm_load_ss(w); w += 1;
            vacc0 = _mm_add_ss(vacc0, _mm_mul_ss(vi0, vw));
          } while (--nnz != 0);
        }
        __m128 vout0 = _mm_min_ss(vacc0, vmax);
        vout0 = _mm_max_ss(vout0, vmin);
        _mm_store_ss(output, vout0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}

void xnn_f32_vadd_minmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    const __m128 vb0 = _mm_loadu_ps(input_b);
    const __m128 vb1 = _mm_loadu_ps(input_b + 4);
    input_b += 8;

    __m128 vacc0 = _mm_add_ps(va0, vb0);
    __m128 vacc1 = _mm_add_ps(va1, vb1);


    vacc0 = _mm_max_ps(vacc0, voutput_min);
    vacc1 = _mm_max_ps(vacc1, voutput_min);

    vacc0 = _mm_min_ps(vacc0, voutput_max);
    vacc1 = _mm_min_ps(vacc1, voutput_max);

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    const __m128 vb = _mm_loadu_ps(input_b);
    input_b += 4;

    __m128 vacc = _mm_add_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);
    const __m128 vb = _mm_loadu_ps(input_b);

    __m128 vacc = _mm_add_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vaddc_minmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);
  const __m128 vb = _mm_load1_ps(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vacc0 = _mm_add_ps(va0, vb);
    __m128 vacc1 = _mm_add_ps(va1, vb);


    vacc0 = _mm_max_ps(vacc0, voutput_min);
    vacc1 = _mm_max_ps(vacc1, voutput_min);

    vacc0 = _mm_min_ps(vacc0, voutput_max);
    vacc1 = _mm_min_ps(vacc1, voutput_max);

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vacc = _mm_add_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);

    __m128 vacc = _mm_add_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vdiv_minmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    const __m128 vb0 = _mm_loadu_ps(input_b);
    const __m128 vb1 = _mm_loadu_ps(input_b + 4);
    input_b += 8;

    __m128 vacc0 = _mm_div_ps(va0, vb0);
    __m128 vacc1 = _mm_div_ps(va1, vb1);


    vacc0 = _mm_max_ps(vacc0, voutput_min);
    vacc1 = _mm_max_ps(vacc1, voutput_min);

    vacc0 = _mm_min_ps(vacc0, voutput_max);
    vacc1 = _mm_min_ps(vacc1, voutput_max);

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    const __m128 vb = _mm_loadu_ps(input_b);
    input_b += 4;

    __m128 vacc = _mm_div_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);
    const __m128 vb = _mm_loadu_ps(input_b);

    __m128 vacc = _mm_div_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vdivc_minmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);
  const __m128 vb = _mm_load1_ps(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vacc0 = _mm_div_ps(va0, vb);
    __m128 vacc1 = _mm_div_ps(va1, vb);


    vacc0 = _mm_max_ps(vacc0, voutput_min);
    vacc1 = _mm_max_ps(vacc1, voutput_min);

    vacc0 = _mm_min_ps(vacc0, voutput_max);
    vacc1 = _mm_min_ps(vacc1, voutput_max);

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vacc = _mm_div_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);

    __m128 vacc = _mm_div_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    const __m128 vb0 = _mm_loadu_ps(input_b);
    const __m128 vb1 = _mm_loadu_ps(input_b + 4);
    input_b += 8;

    __m128 vacc0 = _mm_max_ps(va0, vb0);
    __m128 vacc1 = _mm_max_ps(va1, vb1);



    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    const __m128 vb = _mm_loadu_ps(input_b);
    input_b += 4;

    __m128 vacc = _mm_max_ps(va, vb);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);
    const __m128 vb = _mm_loadu_ps(input_b);

    __m128 vacc = _mm_max_ps(va, vb);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vmaxc_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 vb = _mm_load1_ps(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vacc0 = _mm_max_ps(va0, vb);
    __m128 vacc1 = _mm_max_ps(va1, vb);



    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vacc = _mm_max_ps(va, vb);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);

    __m128 vacc = _mm_max_ps(va, vb);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vmin_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    const __m128 vb0 = _mm_loadu_ps(input_b);
    const __m128 vb1 = _mm_loadu_ps(input_b + 4);
    input_b += 8;

    __m128 vacc0 = _mm_min_ps(va0, vb0);
    __m128 vacc1 = _mm_min_ps(va1, vb1);



    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    const __m128 vb = _mm_loadu_ps(input_b);
    input_b += 4;

    __m128 vacc = _mm_min_ps(va, vb);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);
    const __m128 vb = _mm_loadu_ps(input_b);

    __m128 vacc = _mm_min_ps(va, vb);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vminc_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 vb = _mm_load1_ps(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vacc0 = _mm_min_ps(va0, vb);
    __m128 vacc1 = _mm_min_ps(va1, vb);



    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vacc = _mm_min_ps(va, vb);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);

    __m128 vacc = _mm_min_ps(va, vb);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vmul_minmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    const __m128 vb0 = _mm_loadu_ps(input_b);
    const __m128 vb1 = _mm_loadu_ps(input_b + 4);
    input_b += 8;

    __m128 vacc0 = _mm_mul_ps(va0, vb0);
    __m128 vacc1 = _mm_mul_ps(va1, vb1);


    vacc0 = _mm_max_ps(vacc0, voutput_min);
    vacc1 = _mm_max_ps(vacc1, voutput_min);

    vacc0 = _mm_min_ps(vacc0, voutput_max);
    vacc1 = _mm_min_ps(vacc1, voutput_max);

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    const __m128 vb = _mm_loadu_ps(input_b);
    input_b += 4;

    __m128 vacc = _mm_mul_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);
    const __m128 vb = _mm_loadu_ps(input_b);

    __m128 vacc = _mm_mul_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vmulc_minmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);
  const __m128 vb = _mm_load1_ps(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vacc0 = _mm_mul_ps(va0, vb);
    __m128 vacc1 = _mm_mul_ps(va1, vb);


    vacc0 = _mm_max_ps(vacc0, voutput_min);
    vacc1 = _mm_max_ps(vacc1, voutput_min);

    vacc0 = _mm_min_ps(vacc0, voutput_max);
    vacc1 = _mm_min_ps(vacc1, voutput_max);

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vacc = _mm_mul_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);

    __m128 vacc = _mm_mul_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vrdivc_minmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);
  const __m128 vb = _mm_load1_ps(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vacc0 = _mm_div_ps(vb, va0);
    __m128 vacc1 = _mm_div_ps(vb, va1);


    vacc0 = _mm_max_ps(vacc0, voutput_min);
    vacc1 = _mm_max_ps(vacc1, voutput_min);

    vacc0 = _mm_min_ps(vacc0, voutput_max);
    vacc1 = _mm_min_ps(vacc1, voutput_max);

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vacc = _mm_div_ps(vb, va);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);

    __m128 vacc = _mm_div_ps(vb, va);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vrsubc_minmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);
  const __m128 vb = _mm_load1_ps(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vacc0 = _mm_sub_ps(vb, va0);
    __m128 vacc1 = _mm_sub_ps(vb, va1);


    vacc0 = _mm_max_ps(vacc0, voutput_min);
    vacc1 = _mm_max_ps(vacc1, voutput_min);

    vacc0 = _mm_min_ps(vacc0, voutput_max);
    vacc1 = _mm_min_ps(vacc1, voutput_max);

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vacc = _mm_sub_ps(vb, va);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);

    __m128 vacc = _mm_sub_ps(vb, va);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vsqrdiff_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    const __m128 vb0 = _mm_loadu_ps(input_b);
    const __m128 vb1 = _mm_loadu_ps(input_b + 4);
    input_b += 8;

    __m128 vacc0 = _mm_sub_ps(va0, vb0);
    __m128 vacc1 = _mm_sub_ps(va1, vb1);

    vacc0 = _mm_mul_ps(vacc0, vacc0);
    vacc1 = _mm_mul_ps(vacc1, vacc1);


    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    const __m128 vb = _mm_loadu_ps(input_b);
    input_b += 4;

    __m128 vacc = _mm_sub_ps(va, vb);
    vacc = _mm_mul_ps(vacc, vacc);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);
    const __m128 vb = _mm_loadu_ps(input_b);

    __m128 vacc = _mm_sub_ps(va, vb);
    vacc = _mm_mul_ps(vacc, vacc);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vsqrdiffc_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 vb = _mm_load1_ps(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vacc0 = _mm_sub_ps(va0, vb);
    __m128 vacc1 = _mm_sub_ps(va1, vb);

    vacc0 = _mm_mul_ps(vacc0, vacc0);
    vacc1 = _mm_mul_ps(vacc1, vacc1);


    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vacc = _mm_sub_ps(va, vb);
    vacc = _mm_mul_ps(vacc, vacc);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);

    __m128 vacc = _mm_sub_ps(va, vb);
    vacc = _mm_mul_ps(vacc, vacc);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vsub_minmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    const __m128 vb0 = _mm_loadu_ps(input_b);
    const __m128 vb1 = _mm_loadu_ps(input_b + 4);
    input_b += 8;

    __m128 vacc0 = _mm_sub_ps(va0, vb0);
    __m128 vacc1 = _mm_sub_ps(va1, vb1);


    vacc0 = _mm_max_ps(vacc0, voutput_min);
    vacc1 = _mm_max_ps(vacc1, voutput_min);

    vacc0 = _mm_min_ps(vacc0, voutput_max);
    vacc1 = _mm_min_ps(vacc1, voutput_max);

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    const __m128 vb = _mm_loadu_ps(input_b);
    input_b += 4;

    __m128 vacc = _mm_sub_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);
    const __m128 vb = _mm_loadu_ps(input_b);

    __m128 vacc = _mm_sub_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vsubc_minmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);
  const __m128 vb = _mm_load1_ps(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vacc0 = _mm_sub_ps(va0, vb);
    __m128 vacc1 = _mm_sub_ps(va1, vb);


    vacc0 = _mm_max_ps(vacc0, voutput_min);
    vacc1 = _mm_max_ps(vacc1, voutput_min);

    vacc0 = _mm_min_ps(vacc0, voutput_max);
    vacc1 = _mm_min_ps(vacc1, voutput_max);

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vacc = _mm_sub_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);

    __m128 vacc = _mm_sub_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vclamp_ukernel__sse_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vy_min = _mm_set1_ps(params->sse.min);
  const __m128 vy_max = _mm_set1_ps(params->sse.max);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m128 vacc0123 = _mm_loadu_ps(input);
    __m128 vacc4567 = _mm_loadu_ps(input + 4);
    input += 8;

    vacc0123 = _mm_max_ps(vacc0123, vy_min);
    vacc4567 = _mm_max_ps(vacc4567, vy_min);

    vacc0123 = _mm_min_ps(vacc0123, vy_max);
    vacc4567 = _mm_min_ps(vacc4567, vy_max);

    _mm_storeu_ps(output, vacc0123);
    _mm_storeu_ps(output + 4, vacc4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    __m128 vacc = _mm_loadu_ps(input);
    input += 4;

    vacc = _mm_max_ps(vacc, vy_min);
    vacc = _mm_min_ps(vacc, vy_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128 vacc = _mm_loadu_ps(input);
    vacc = _mm_max_ps(vacc, vy_min);
    vacc = _mm_min_ps(vacc, vy_max);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vcmul_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float* ar = input_a;
  const float* ai = (const float*) ((uintptr_t) input_a + batch);
  const float* br = input_b;
  const float* bi = (const float*) ((uintptr_t) input_b + batch);
  float* or = output;
  float* oi = (float*) ((uintptr_t) output + batch);
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0r = _mm_loadu_ps(ar);
    const __m128 va0i = _mm_loadu_ps(ai);
    const __m128 vb0r = _mm_loadu_ps(br);
    const __m128 vb0i = _mm_loadu_ps(bi);
    const __m128 va1r = _mm_loadu_ps(ar + 4);
    const __m128 va1i = _mm_loadu_ps(ai + 4);
    const __m128 vb1r = _mm_loadu_ps(br + 4);
    const __m128 vb1i = _mm_loadu_ps(bi + 4);
    ar += 8;
    ai += 8;
    br += 8;
    bi += 8;

    __m128 vacc0r = _mm_mul_ps(va0r, vb0r);
    __m128 vacc0i = _mm_mul_ps(va0r, vb0i);
    __m128 vacc1r = _mm_mul_ps(va1r, vb1r);
    __m128 vacc1i = _mm_mul_ps(va1r, vb1i);

    vacc0r = _mm_sub_ps(vacc0r, _mm_mul_ps(va0i, vb0i));
    vacc0i = _mm_add_ps(vacc0i, _mm_mul_ps(va0i, vb0r));
    vacc1r = _mm_sub_ps(vacc1r, _mm_mul_ps(va1i, vb1i));
    vacc1i = _mm_add_ps(vacc1i, _mm_mul_ps(va1i, vb1r));

    _mm_storeu_ps(or, vacc0r);
    _mm_storeu_ps(oi, vacc0i);
    _mm_storeu_ps(or + 4, vacc1r);
    _mm_storeu_ps(oi + 4, vacc1i);
    or += 8;
    oi += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 var = _mm_loadu_ps(ar);
    ar += 4;
    const __m128 vai = _mm_loadu_ps(ai);
    ai += 4;
    const __m128 vbr = _mm_loadu_ps(br);
    br += 4;
    const __m128 vbi = _mm_loadu_ps(bi);
    bi += 4;

    __m128 vaccr = _mm_mul_ps(var, vbr);
    __m128 vacci = _mm_mul_ps(var, vbi);

    vaccr = _mm_sub_ps(vaccr, _mm_mul_ps(vai, vbi));
    vacci = _mm_add_ps(vacci, _mm_mul_ps(vai, vbr));

    _mm_storeu_ps(or, vaccr);
    or += 4;
    _mm_storeu_ps(oi, vacci);
    oi += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 var = _mm_loadu_ps(ar);
    ar += 4;
    const __m128 vai = _mm_loadu_ps(ai);
    ai += 4;
    const __m128 vbr = _mm_loadu_ps(br);
    br += 4;
    const __m128 vbi = _mm_loadu_ps(bi);
    bi += 4;

    __m128 vaccr = _mm_mul_ps(var, vbr);
    __m128 vacci = _mm_mul_ps(var, vbi);

    vaccr = _mm_sub_ps(vaccr, _mm_mul_ps(vai, vbi));
    vacci = _mm_add_ps(vacci, _mm_mul_ps(vai, vbr));

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) or, vaccr);
      or += 2;
      _mm_storel_pi((__m64*) oi, vacci);
      oi += 2;
      vaccr = _mm_movehl_ps(vaccr, vaccr);
      vacci = _mm_movehl_ps(vacci, vacci);
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(or, vaccr);
      _mm_store_ss(oi, vacci);
    }
  }
}

void xnn_f32_vhswish_ukernel__sse_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsixth = _mm_set1_ps(0x1.555556p-3f);
  const __m128 vhalf = _mm_set1_ps(0.5f);
  const __m128 vone = _mm_set1_ps(1.0f);
  const __m128 vzero = _mm_setzero_ps();

  XNN_FORCE_REALIZATION(vsixth);
  XNN_FORCE_REALIZATION(vhalf);
  XNN_FORCE_REALIZATION(vone);
  // XNN_FORCE_REALIZATION(vzero);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    __m128 vacc0123 = _mm_mul_ps(vx0123, vsixth);
    __m128 vacc4567 = _mm_mul_ps(vx4567, vsixth);

    vacc0123 = _mm_add_ps(vacc0123, vhalf);
    vacc4567 = _mm_add_ps(vacc4567, vhalf);

    vacc0123 = _mm_max_ps(vacc0123, vzero);
    vacc4567 = _mm_max_ps(vacc4567, vzero);

    vacc0123 = _mm_min_ps(vacc0123, vone);
    vacc4567 = _mm_min_ps(vacc4567, vone);

    vacc0123 = _mm_mul_ps(vacc0123, vx0123);
    vacc4567 = _mm_mul_ps(vacc4567, vx4567);

    _mm_storeu_ps(output, vacc0123);
    _mm_storeu_ps(output + 4, vacc4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    input += 4;
    __m128 vacc0123 = _mm_mul_ps(vx0123, vsixth);
    vacc0123 = _mm_add_ps(vacc0123, vhalf);
    vacc0123 = _mm_max_ps(vacc0123, vzero);
    vacc0123 = _mm_min_ps(vacc0123, vone);
    vacc0123 = _mm_mul_ps(vacc0123, vx0123);
    _mm_storeu_ps(output, vacc0123);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    __m128 vacc0123 = _mm_mul_ps(vx0123, vsixth);
    vacc0123 = _mm_add_ps(vacc0123, vhalf);
    vacc0123 = _mm_max_ps(vacc0123, vzero);
    vacc0123 = _mm_min_ps(vacc0123, vone);
    vacc0123 = _mm_mul_ps(vacc0123, vx0123);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc0123);
      vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc0123);
    }
  }
}

void xnn_f32_vlrelu_ukernel__sse_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vslope = _mm_set1_ps(params->scalar.slope);
  const __m128 vzero = _mm_setzero_ps();
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m128 vx0123 = _mm_loadu_ps(input);
    __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    __m128 vacc0123 = _mm_max_ps(_mm_setzero_ps(), vx0123);
    vx0123 = _mm_min_ps(vx0123, vzero);
    __m128 vacc4567 = _mm_max_ps(_mm_setzero_ps(), vx4567);
    vx4567 = _mm_min_ps(vx4567, vzero);

    vacc0123 = _mm_add_ps(vacc0123, _mm_mul_ps(vx0123, vslope));
    vacc4567 = _mm_add_ps(vacc4567, _mm_mul_ps(vx4567, vslope));

    _mm_storeu_ps(output, vacc0123);
    _mm_storeu_ps(output + 4, vacc4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    __m128 vx = _mm_loadu_ps(input);
    input += 4;

    __m128 vacc = _mm_max_ps(_mm_setzero_ps(), vx);
    vx = _mm_min_ps(vx, vzero);
    vacc = _mm_add_ps(vacc, _mm_mul_ps(vx, vslope));

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128 vx = _mm_loadu_ps(input);

    __m128 vacc = _mm_max_ps(_mm_setzero_ps(), vx);
    vx = _mm_min_ps(vx, vzero);
    vacc = _mm_add_ps(vacc, _mm_mul_ps(vx, vslope));

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const __m128 vscale0123 = _mm_load_ps(w);

      __m128 vacc0x0123 = _mm_loadu_ps(i0);
      i0 += 4;
      __m128 vacc1x0123 = _mm_loadu_ps(i1);
      i1 += 4;

      vacc0x0123 = _mm_mul_ps(vacc0x0123, vscale0123);
      vacc1x0123 = _mm_mul_ps(vacc1x0123, vscale0123);

      const __m128 vbias0123 = _mm_load_ps(w + 4);

      vacc0x0123 = _mm_add_ps(vacc0x0123, vbias0123);
      vacc1x0123 = _mm_add_ps(vacc1x0123, vbias0123);

      vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
      vacc1x0123 = _mm_max_ps(vacc1x0123, vmin);

      vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
      vacc1x0123 = _mm_min_ps(vacc1x0123, vmax);

      _mm_storeu_ps(o0, vacc0x0123);
      o0 += 4;
      _mm_storeu_ps(o1, vacc1x0123);
      o1 += 4;

      w += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m128 vscale0123 = _mm_load_ps(w);

      __m128 vacc0x0123 = _mm_loadu_ps(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      __m128 vacc1x0123 = _mm_loadu_ps(i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      vacc0x0123 = _mm_mul_ps(vacc0x0123, vscale0123);
      vacc1x0123 = _mm_mul_ps(vacc1x0123, vscale0123);

      const __m128 vbias0123 = _mm_load_ps(w + 4);

      vacc0x0123 = _mm_add_ps(vacc0x0123, vbias0123);
      vacc1x0123 = _mm_add_ps(vacc1x0123, vbias0123);

      vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
      vacc1x0123 = _mm_max_ps(vacc1x0123, vmin);

      vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
      vacc1x0123 = _mm_min_ps(vacc1x0123, vmax);

      if (c & (2 * sizeof(float))) {
        _mm_storel_pi((__m64*) o0, vacc0x0123);
        _mm_storel_pi((__m64*) o1, vacc1x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(float))) {
        _mm_store_ss(o0, vacc0x0123);
        _mm_store_ss(o1, vacc1x0123);

        o0 += 1;
        o1 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_vrsqrt_ukernel__sse_rsqrt_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const __m128 vthree = _mm_set1_ps(3.0f);
  const __m128 vhalf = _mm_set1_ps(0.5f);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    // Generate the initial 12-bit approximation.
    const __m128 vt0_0123 = _mm_rsqrt_ps(vx0123);
    const __m128 vt0_4567 = _mm_rsqrt_ps(vx4567);

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1_0123 = _mm_mul_ps(vt0_0123, vt0_0123);
    const __m128 vt1_4567 = _mm_mul_ps(vt0_4567, vt0_4567);
    const __m128 vt2_0123 = _mm_mul_ps(vx0123, vt1_0123);
    const __m128 vt2_4567 = _mm_mul_ps(vx4567, vt1_4567);
    const __m128 vt3_0123 = _mm_sub_ps(vthree, vt2_0123);
    const __m128 vt3_4567 = _mm_sub_ps(vthree, vt2_4567);
    const __m128 vt4_0123 = _mm_mul_ps(vhalf, vt0_0123);
    const __m128 vt4_4567 = _mm_mul_ps(vhalf, vt0_4567);
    const __m128 vy0123 = _mm_mul_ps(vt3_0123, vt4_0123);
    const __m128 vy4567 = _mm_mul_ps(vt3_4567, vt4_4567);

    // Store the results.
    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    // Generate the initial 12-bit approximation.
    const __m128 vt0 = _mm_rsqrt_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1 = _mm_mul_ps(vt0, vt0);
    const __m128 vt2 = _mm_mul_ps(vx, vt1);
    const __m128 vt3 = _mm_sub_ps(vthree, vt2);
    const __m128 vt4 = _mm_mul_ps(vhalf, vt0);
    const __m128 vy = _mm_mul_ps(vt3, vt4);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);

    // Generate the initial 12-bit approximation.
    const __m128 vt0 = _mm_rsqrt_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1 = _mm_mul_ps(vt0, vt0);
    const __m128 vt2 = _mm_mul_ps(vx, vt1);
    const __m128 vt3 = _mm_sub_ps(vthree, vt2);
    const __m128 vt4 = _mm_mul_ps(vhalf, vt0);
    __m128 vy = _mm_mul_ps(vt3, vt4);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}

void xnn_f32_vsqrt_ukernel__sse_rsqrt_u12(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const __m128 vthree = _mm_set1_ps(3.0f);
  const __m128 vhalf = _mm_set1_ps(0.5f);

  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    const __m128 vx0 = _mm_loadu_ps(input);
    const __m128 vx1 = _mm_loadu_ps(input + 4);
    const __m128 vx2 = _mm_loadu_ps(input + 8);
    input += 12;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m128 vmask0 = _mm_cmpeq_ps(vx0, _mm_setzero_ps());
    const __m128 vmask1 = _mm_cmpeq_ps(vx1, _mm_setzero_ps());
    const __m128 vmask2 = _mm_cmpeq_ps(vx2, _mm_setzero_ps());

    // Generate the initial 12-bit approximation.
    const __m128 vt0_0 = _mm_rsqrt_ps(vx0);
    const __m128 vt0_1 = _mm_rsqrt_ps(vx1);
    const __m128 vt0_2 = _mm_rsqrt_ps(vx2);

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1_0 = _mm_mul_ps(vt0_0, vt0_0);
    const __m128 vt1_1 = _mm_mul_ps(vt0_1, vt0_1);
    const __m128 vt1_2 = _mm_mul_ps(vt0_2, vt0_2);
    const __m128 vt2_0 = _mm_mul_ps(vx0, vt1_0);
    const __m128 vt2_1 = _mm_mul_ps(vx1, vt1_1);
    const __m128 vt2_2 = _mm_mul_ps(vx2, vt1_2);
    const __m128 vt3_0 = _mm_sub_ps(vthree, vt2_0);
    const __m128 vt3_1 = _mm_sub_ps(vthree, vt2_1);
    const __m128 vt3_2 = _mm_sub_ps(vthree, vt2_2);
    const __m128 vt4_0 = _mm_mul_ps(vhalf, vt0_0);
    const __m128 vt4_1 = _mm_mul_ps(vhalf, vt0_1);
    const __m128 vt4_2 = _mm_mul_ps(vhalf, vt0_2);
    const __m128 vt5_0 = _mm_mul_ps(vt3_0, vt4_0);
    const __m128 vt5_1 = _mm_mul_ps(vt3_1, vt4_1);
    const __m128 vt5_2 = _mm_mul_ps(vt3_2, vt4_2);
    const __m128 vt6_0 = _mm_andnot_ps(vmask0, vt5_0);
    const __m128 vt6_1 = _mm_andnot_ps(vmask1, vt5_1);
    const __m128 vt6_2 = _mm_andnot_ps(vmask2, vt5_2);
    const __m128 vy0 = _mm_mul_ps(vx0, vt6_0);
    const __m128 vy1 = _mm_mul_ps(vx1, vt6_1);
    const __m128 vy2 = _mm_mul_ps(vx2, vt6_2);

    // Store the results.
    _mm_storeu_ps(output, vy0);
    _mm_storeu_ps(output + 4, vy1);
    _mm_storeu_ps(output + 8, vy2);
    output += 12;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    // Generate the initial 12-bit approximation.
    const __m128 vt0 = _mm_rsqrt_ps(vx);

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m128 vmask = _mm_cmpeq_ps(vx, _mm_setzero_ps());

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1 = _mm_mul_ps(vt0, vt0);
    const __m128 vt2 = _mm_mul_ps(vx, vt1);
    const __m128 vt3 = _mm_sub_ps(vthree, vt2);
    const __m128 vt4 = _mm_mul_ps(vhalf, vt0);
    const __m128 vt5 = _mm_mul_ps(vt3, vt4);
    const __m128 vt6 = _mm_andnot_ps(vmask, vt5);
    const __m128 vy = _mm_mul_ps(vx, vt6);
    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);
    // Generate the initial 12-bit approximation.
    const __m128 vt0 = _mm_rsqrt_ps(vx);

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m128 vmask = _mm_cmpeq_ps(vx, _mm_setzero_ps());

    // Do a single Newton-Raphson step as described above.
    const __m128 vt1 = _mm_mul_ps(vt0, vt0);
    const __m128 vt2 = _mm_mul_ps(vx, vt1);
    const __m128 vt3 = _mm_sub_ps(vthree, vt2);
    const __m128 vt4 = _mm_mul_ps(vhalf, vt0);
    const __m128 vt5 = _mm_mul_ps(vt3, vt4);
    const __m128 vt6 = _mm_andnot_ps(vmask, vt5);
    __m128 vy = _mm_mul_ps(vx, vt6);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}

void xnn_x32_transposec_ukernel__4x4_sse(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * sizeof(float);
  const size_t input_vreset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_vreset = tile_height * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);
  const size_t input_offset = tile_height * input_stride;

  const float* i0 = (const float*) input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);

  float* o0 = (float*) output;
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);
  float* o2 = (float*) ((uintptr_t) o1 + output_stride);
  float* o3 = (float*) ((uintptr_t) o2 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 2) {
      o2 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 4) {
      o3 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 4; bh -= 4) {
      __m128 v0 = _mm_loadu_ps(i0);
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      __m128 v1 = _mm_loadu_ps(i1);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      __m128 v2 = _mm_loadu_ps(i2);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      __m128 v3 = _mm_loadu_ps(i3);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);

      _MM_TRANSPOSE4_PS(v0, v1, v2, v3);

      _mm_storeu_ps(o3, v3);
      o3 = (float*) ((uintptr_t) o3 + tile_wbytes);
      _mm_storeu_ps(o2, v2);
      o2 = (float*) ((uintptr_t) o2 + tile_wbytes);
      _mm_storeu_ps(o1, v1);
      o1 = (float*) ((uintptr_t) o1 + tile_wbytes);
      _mm_storeu_ps(o0, v0);
      o0 = (float*) ((uintptr_t) o0 + tile_wbytes);
    }

    if (bh != 0) {
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      __m128 v0 = _mm_loadu_ps(i0);
      __m128 v1 = _mm_loadu_ps(i1);
      __m128 v2 = _mm_loadu_ps(i2);
      __m128 v3 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(v0, v1, v2, v3);

      if (bh & 2) {
        _mm_storel_pi((__m64*) o3, v3);
        o3 += 2;
        _mm_storel_pi((__m64*) o2, v2);
        o2 += 2;
        _mm_storel_pi((__m64*) o1, v1);
        o1 += 2;
        _mm_storel_pi((__m64*) o0, v0);
        o0 += 2;
        v0 = _mm_movehl_ps(v0, v0);
        v1 = _mm_movehl_ps(v1, v1);
        v2 = _mm_movehl_ps(v2, v2);
        v3 = _mm_movehl_ps(v3, v3);
      }
      if (bh & 1) {
        _mm_store_ss(o3, v3);
        _mm_store_ss(o2, v2);
        _mm_store_ss(o1, v1);
        _mm_store_ss(o0, v0);
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_vreset);
    i1 = (const float*) ((uintptr_t) i0 + input_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_stride);
    o0 = (float*) ((uintptr_t) o0 + output_vreset);
    o1 = (float*) ((uintptr_t) o1 + output_vreset);
    o2 = (float*) ((uintptr_t) o2 + output_vreset);
    o3 = (float*) ((uintptr_t) o3 + output_vreset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
