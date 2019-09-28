/*
 * Copyright 2019 Google LLC
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/argmaxpool.h>


void xnn_f32_argmaxpool_ukernel_up4__sse2(
    size_t n,
    size_t ks,
    size_t kc,
    const float** input,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(ks != 0);
  assert(ks <= 4);
  assert(kc != 0);

  const __m128 voutput_max = _mm_load_ps(params->sse.max);
  const __m128 voutput_min = _mm_load_ps(params->sse.min);
  do {
    float* o = output;
    uint32_t* i = index;

    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    if (ks < 2) {
      i1 = i0;
    }
    if (ks <= 2) {
      i2 = i0;
    }
    if (ks != 4) {
      i3 = i0;
    }

    size_t k = kc;
    for (; k >= 4; k -= 4) {
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

      const __m128 vout = _mm_max_ps(_mm_min_ps(vmax, voutput_max), voutput_min);

      _mm_storeu_ps(o, vout);
      o += 4;
      _mm_storeu_si128((__m128i*) i, vidx);
      i += 4;
    }
    if (k != 0) {
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

      __m128 vout = _mm_max_ps(_mm_min_ps(vmax, voutput_max), voutput_min);

      if (k & 2) {
        _mm_store_sd((double*) o, _mm_castps_pd(vout));
        _mm_storel_epi64((__m128i*) i, vidx);
        vout = _mm_movehl_ps(vout, vout);
        vidx = _mm_unpackhi_epi64(vidx, vidx);
        o += 2;
        i += 2;
      }
      if (k & 1) {
        _mm_store_ss(o, vout);
        *i = (uint32_t) _mm_cvtsi128_si32(vidx);
        o += 1;
        i += 1;
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
    index = (uint32_t*) i;
  } while (--n != 0);
}
