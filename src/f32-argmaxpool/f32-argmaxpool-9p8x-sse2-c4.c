// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/argmaxpool.h"


void xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* accumulation_buffer,
    uint32_t* index_buffer,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements > 9);
  assert(channels != 0);

  do {
    {
      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;

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

        const __m128i vm4 = _mm_castps_si128(_mm_cmpgt_ps(vi4, vmax));
        vmax = _mm_max_ps(vi4, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm4, vidx), _mm_and_si128(vm4, _mm_set1_epi32(4)));

        const __m128i vm5 = _mm_castps_si128(_mm_cmpgt_ps(vi5, vmax));
        vmax = _mm_max_ps(vi5, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm5, vidx), _mm_and_si128(vm5, _mm_set1_epi32(5)));

        const __m128i vm6 = _mm_castps_si128(_mm_cmpgt_ps(vi6, vmax));
        vmax = _mm_max_ps(vi6, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm6, vidx), _mm_and_si128(vm6, _mm_set1_epi32(6)));

        const __m128i vm7 = _mm_castps_si128(_mm_cmpgt_ps(vi7, vmax));
        vmax = _mm_max_ps(vi7, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm7, vidx), _mm_and_si128(vm7, _mm_set1_epi32(7)));

        const __m128i vm8 = _mm_castps_si128(_mm_cmpgt_ps(vi8, vmax));
        vmax = _mm_max_ps(vi8, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm8, vidx), _mm_and_si128(vm8, _mm_set1_epi32(8)));

        _mm_store_ps(ab, vmax);
        ab += 4;
        _mm_store_si128((__m128i*) ib, vidx);
        ib += 4;
      }
    }
    const __m128i v1 = _mm_set1_epi32(1);
    const __m128i v8 = _mm_set1_epi32(8);
    __m128i vidx0 = _mm_add_epi32(v1, v8);

    size_t k = pooling_elements;
    for (k -= 9; k > 8; k -= 8) {
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

      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;

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

        __m128 vmax = _mm_load_ps(ab);
        __m128i vidx = _mm_load_si128((const __m128i*) ib);

        const __m128i vm0 = _mm_castps_si128(_mm_cmpgt_ps(vi0, vmax));
        vmax = _mm_max_ps(vi0, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm0, vidx), _mm_and_si128(vm0, vidx0));

        const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
        const __m128i vidx1 = _mm_add_epi32(vidx0, v1);
        vmax = _mm_max_ps(vi1, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, vidx1));

        const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
        const __m128i vidx2 = _mm_add_epi32(vidx1, v1);
        vmax = _mm_max_ps(vi2, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, vidx2));

        const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
        const __m128i vidx3 = _mm_add_epi32(vidx2, v1);
        vmax = _mm_max_ps(vi3, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, vidx3));

        const __m128i vm4 = _mm_castps_si128(_mm_cmpgt_ps(vi4, vmax));
        const __m128i vidx4 = _mm_add_epi32(vidx3, v1);
        vmax = _mm_max_ps(vi4, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm4, vidx), _mm_and_si128(vm4, vidx4));

        const __m128i vm5 = _mm_castps_si128(_mm_cmpgt_ps(vi5, vmax));
        const __m128i vidx5 = _mm_add_epi32(vidx4, v1);
        vmax = _mm_max_ps(vi5, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm5, vidx), _mm_and_si128(vm5, vidx5));

        const __m128i vm6 = _mm_castps_si128(_mm_cmpgt_ps(vi6, vmax));
        const __m128i vidx6 = _mm_add_epi32(vidx5, v1);
        vmax = _mm_max_ps(vi6, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm6, vidx), _mm_and_si128(vm6, vidx6));

        const __m128i vm7 = _mm_castps_si128(_mm_cmpgt_ps(vi7, vmax));
        const __m128i vidx7 = _mm_add_epi32(vidx6, v1);
        vmax = _mm_max_ps(vi7, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm7, vidx), _mm_and_si128(vm7, vidx7));

        _mm_store_ps(ab, vmax);
        ab += 4;
        _mm_store_si128((__m128i*) ib, vidx);
        ib += 4;
      }
      vidx0 = _mm_add_epi32(vidx0, v8);
    }

    float* o = output;
    uint32_t* i = index;
    {
      const float* i0 = input[0];
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      input = (const float**) ((uintptr_t) input + input_increment);
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
      if (k != 8) {
        i7 = i0;
      }

      size_t c = channels;
      float* ab = accumulation_buffer;
      uint32_t* ib = index_buffer;
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

        __m128 vmax = _mm_load_ps(ab);
        ab += 4;
        __m128i vidx = _mm_load_si128((const __m128i*) ib);
        ib += 4;

        const __m128i vm0 = _mm_castps_si128(_mm_cmpgt_ps(vi0, vmax));
        vmax = _mm_max_ps(vi0, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm0, vidx), _mm_and_si128(vm0, vidx0));

        const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
        const __m128i vidx1 = _mm_add_epi32(vidx0, v1);
        vmax = _mm_max_ps(vi1, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, vidx1));

        const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
        const __m128i vidx2 = _mm_add_epi32(vidx1, v1);
        vmax = _mm_max_ps(vi2, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, vidx2));

        const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
        const __m128i vidx3 = _mm_add_epi32(vidx2, v1);
        vmax = _mm_max_ps(vi3, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, vidx3));

        const __m128i vm4 = _mm_castps_si128(_mm_cmpgt_ps(vi4, vmax));
        const __m128i vidx4 = _mm_add_epi32(vidx3, v1);
        vmax = _mm_max_ps(vi4, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm4, vidx), _mm_and_si128(vm4, vidx4));

        const __m128i vm5 = _mm_castps_si128(_mm_cmpgt_ps(vi5, vmax));
        const __m128i vidx5 = _mm_add_epi32(vidx4, v1);
        vmax = _mm_max_ps(vi5, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm5, vidx), _mm_and_si128(vm5, vidx5));

        const __m128i vm6 = _mm_castps_si128(_mm_cmpgt_ps(vi6, vmax));
        const __m128i vidx6 = _mm_add_epi32(vidx5, v1);
        vmax = _mm_max_ps(vi6, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm6, vidx), _mm_and_si128(vm6, vidx6));

        const __m128i vm7 = _mm_castps_si128(_mm_cmpgt_ps(vi7, vmax));
        const __m128i vidx7 = _mm_add_epi32(vidx6, v1);
        vmax = _mm_max_ps(vi7, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm7, vidx), _mm_and_si128(vm7, vidx7));

        _mm_storeu_ps(o, vmax);
        o += 4;
        _mm_storeu_si128((__m128i*) i, vidx);
        i += 4;
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

        __m128 vmax = _mm_load_ps(ab);
        __m128i vidx = _mm_load_si128((const __m128i*) ib);

        const __m128i vm0 = _mm_castps_si128(_mm_cmpgt_ps(vi0, vmax));
        vmax = _mm_max_ps(vi0, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm0, vidx), _mm_and_si128(vm0, vidx0));

        const __m128i vm1 = _mm_castps_si128(_mm_cmpgt_ps(vi1, vmax));
        const __m128i vidx1 = _mm_add_epi32(vidx0, v1);
        vmax = _mm_max_ps(vi1, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm1, vidx), _mm_and_si128(vm1, vidx1));

        const __m128i vm2 = _mm_castps_si128(_mm_cmpgt_ps(vi2, vmax));
        const __m128i vidx2 = _mm_add_epi32(vidx1, v1);
        vmax = _mm_max_ps(vi2, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm2, vidx), _mm_and_si128(vm2, vidx2));

        const __m128i vm3 = _mm_castps_si128(_mm_cmpgt_ps(vi3, vmax));
        const __m128i vidx3 = _mm_add_epi32(vidx2, v1);
        vmax = _mm_max_ps(vi3, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm3, vidx), _mm_and_si128(vm3, vidx3));

        const __m128i vm4 = _mm_castps_si128(_mm_cmpgt_ps(vi4, vmax));
        const __m128i vidx4 = _mm_add_epi32(vidx3, v1);
        vmax = _mm_max_ps(vi4, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm4, vidx), _mm_and_si128(vm4, vidx4));

        const __m128i vm5 = _mm_castps_si128(_mm_cmpgt_ps(vi5, vmax));
        const __m128i vidx5 = _mm_add_epi32(vidx4, v1);
        vmax = _mm_max_ps(vi5, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm5, vidx), _mm_and_si128(vm5, vidx5));

        const __m128i vm6 = _mm_castps_si128(_mm_cmpgt_ps(vi6, vmax));
        const __m128i vidx6 = _mm_add_epi32(vidx5, v1);
        vmax = _mm_max_ps(vi6, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm6, vidx), _mm_and_si128(vm6, vidx6));

        const __m128i vm7 = _mm_castps_si128(_mm_cmpgt_ps(vi7, vmax));
        const __m128i vidx7 = _mm_add_epi32(vidx6, v1);
        vmax = _mm_max_ps(vi7, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm7, vidx), _mm_and_si128(vm7, vidx7));

        if (c & 2) {
          _mm_storel_pi((__m64*) o, vmax);
          _mm_storel_epi64((__m128i*) i, vidx);
          vmax = _mm_movehl_ps(vmax, vmax);
          vidx = _mm_unpackhi_epi64(vidx, vidx);
          o += 2;
          i += 2;
        }
        if (c & 1) {
          _mm_store_ss(o, vmax);
          *i = (uint32_t) _mm_cvtsi128_si32(vidx);
          o += 1;
          i += 1;
        }
      }
    }

    output = (float*) ((uintptr_t) o + output_increment);
    index = (uint32_t*) i;
  } while (--output_pixels != 0);
}
