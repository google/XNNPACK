// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/argmaxpool.h"
#include "src/xnnpack/simd/f32-sse2.h"

static XNN_INLINE __m128i
xnn_load_tail_safe_u32(const uint32_t* input, size_t num_elements) {
  return _mm_castps_si128(xnn_load_tail_safe_f32((const float*) input, num_elements));
}

void xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4(
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
  assert(channels != 0);

  const __m128i v1 = _mm_set1_epi32(1);
  XNN_FORCE_REALIZATION(v1);

  do {
    // Accumulators start out null, after each pass the accumulator is set to
    // the output.
    const float* ab = NULL;
    const uint32_t* ib = NULL;
    const float** id = input;

    __m128i vidx0 = _mm_set1_epi32(0);
    __m128i vidx8;

    assert(!ab);
    assert(!ib);

    ptrdiff_t k = pooling_elements;
    for (; k > 0; k -= 9) {
      const float* i0 = *id++;
      const float* i1 = 1 < k ? *id++ : i0;
      const float* i2 = 2 < k ? *id++ : i0;
      const float* i3 = 3 < k ? *id++ : i0;
      const float* i4 = 4 < k ? *id++ : i0;
      const float* i5 = 5 < k ? *id++ : i0;
      const float* i6 = 6 < k ? *id++ : i0;
      const float* i7 = 7 < k ? *id++ : i0;
      const float* i8 = 8 < k ? *id++ : i0;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);

      float* o = output;
      uint32_t* i = index;
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

        __m128 vmax;
        __m128i vidx;
        if (ab) {
          vmax = _mm_loadu_ps(ab); ab += 4;
          vidx = _mm_loadu_si128((const __m128i*) ib); ib += 4;

          const __m128i vm0 = _mm_castps_si128(_mm_cmpgt_ps(vi0, vmax));
          vmax = _mm_max_ps(vi0, vmax);
          vidx = _mm_or_si128(_mm_andnot_si128(vm0, vidx), _mm_and_si128(vm0, vidx0));
        } else {
          vmax = vi0;
          vidx = vidx0;
        }

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

        const __m128i vm8 = _mm_castps_si128(_mm_cmpgt_ps(vi8, vmax));
        vidx8 = _mm_add_epi32(vidx7, v1);
        vmax = _mm_max_ps(vi8, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm8, vidx), _mm_and_si128(vm8, vidx8));

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
        const __m128 vi8 = _mm_loadu_ps(i8);

        __m128 vmax;
        __m128i vidx;
        if (ab) {
          vmax = xnn_load_tail_safe_f32(ab, c);
          vidx = xnn_load_tail_safe_u32(ib, c);

          const __m128i vm0 = _mm_castps_si128(_mm_cmpgt_ps(vi0, vmax));
          vmax = _mm_max_ps(vi0, vmax);
          vidx = _mm_or_si128(_mm_andnot_si128(vm0, vidx), _mm_and_si128(vm0, vidx0));
        } else {
          vmax = vi0;
          vidx = vidx0;
        }

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

        const __m128i vm8 = _mm_castps_si128(_mm_cmpgt_ps(vi8, vmax));
        vidx8 = _mm_add_epi32(vidx7, v1);
        vmax = _mm_max_ps(vi8, vmax);
        vidx = _mm_or_si128(_mm_andnot_si128(vm8, vidx), _mm_and_si128(vm8, vidx8));

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
      vidx0 = _mm_add_epi32(vidx8, v1);
      ab = output;
      ib = index;
    }

    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) output + output_increment);
    index += channels;
  } while (--output_pixels != 0);
}
