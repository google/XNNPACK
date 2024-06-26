// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/argmaxpool.h"


void xnn_f32_argmaxpool_ukernel_4x__sse2_c4(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements <= 4);
  assert(channels != 0);

  do {
    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    i0 = (const float*) ((uintptr_t) i0 + input_offset);
    i1 = (const float*) ((uintptr_t) i1 + input_offset);
    i2 = (const float*) ((uintptr_t) i2 + input_offset);
    i3 = (const float*) ((uintptr_t) i3 + input_offset);
    if (pooling_elements < 2) {
      i1 = i0;
    }
    if (pooling_elements <= 2) {
      i2 = i0;
    }
    if (pooling_elements != 4) {
      i3 = i0;
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

      __m128 vmax = vi0;
      __m128i vidx = _mm_setzero_si128();

      const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
      vmax = _mm_max_ps(vi1, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, _mm_set1_epi32(1)));

      const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
      vmax = _mm_max_ps(vi2, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, _mm_set1_epi32(2)));

      const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
      vmax = _mm_max_ps(vi3, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, _mm_set1_epi32(3)));

      _mm_storeu_ps(output, vmax);
      output += 4;
      _mm_storeu_si128((__m128i*) index, vidx);
      index += 4;
    }
    if (c != 0) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      const __m128 vi1 = _mm_loadu_ps(i1);
      const __m128 vi2 = _mm_loadu_ps(i2);
      const __m128 vi3 = _mm_loadu_ps(i3);

      __m128 vmax = vi0;
      __m128i vidx = _mm_setzero_si128();

      const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
      vmax = _mm_max_ps(vi1, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, _mm_set1_epi32(1)));

      const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
      vmax = _mm_max_ps(vi2, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, _mm_set1_epi32(2)));

      const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
      vmax = _mm_max_ps(vi3, vmax);
      vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, _mm_set1_epi32(3)));

      if (c & 2) {
        _mm_storel_pi((__m64*) output, vmax);
        _mm_storel_epi64((__m128i*) index, vidx);
        vmax = _mm_movehl_ps(vmax, vmax);
        vidx = _mm_unpackhi_epi64(vidx, vidx);
        output += 2;
        index += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vmax);
        *index = (uint32_t) _mm_cvtsi128_si32(vidx);
        output += 1;
        index += 1;
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
