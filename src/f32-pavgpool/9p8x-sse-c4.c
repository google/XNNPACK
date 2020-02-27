// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/pavgpool.h>


void xnn_f32_pavgpool_ukernel_9p8x__sse_c4(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    const float* zero,
    const float* multiplier,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const __m128 voutput_min = _mm_load_ps(params->sse.min);
  const __m128 voutput_max = _mm_load_ps(params->sse.max);

  do {
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
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;

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
      if (k <= 2) {
        i2 = zero;
      }
      if (k < 4) {
        i3 = zero;
      }
      if (k <= 4) {
        i4 = zero;
      }
      if (k < 6) {
        i5 = zero;
      }
      if (k <= 6) {
        i6 = zero;
      }
      if (k != 8) {
        i7 = zero;
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
