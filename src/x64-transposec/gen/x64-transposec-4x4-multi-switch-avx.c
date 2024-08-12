// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <immintrin.h>

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"
#include "xnnpack/unaligned.h"

void xnn_x64_transposec_ukernel__4x4_multi_switch_avx(
    const uint64_t* input,
    uint64_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  static const int64_t mask_table[7] = {-1, -1, -1, -1, 0, 0, 0};

  assert(block_width == 1 || output_stride >= block_height * sizeof(double));
  assert(block_height == 1 || input_stride >= block_width * sizeof(double));

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_hbytes = tile_height * sizeof(double);
  const size_t tile_wbytes = tile_width * sizeof(double);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(double);

  const double* i0 = (const double*) input;
  const double* i1 = (const double*) ((uintptr_t) i0 + input_stride);
  const double* i2 = (const double*) ((uintptr_t) i1 + input_stride);
  const double* i3 = (const double*) ((uintptr_t) i2 + input_stride);
  double* o = (double*) output;
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 3);
    const size_t oN_stride = rem * output_stride;

    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[rem ^ 3]));

    size_t bh = block_height;
    for (; bh >= 4; bh -= 4) {
      const __m256d v2_0 = _mm256_maskload_pd(i0, vmask);
      i0 = (double*) ((uintptr_t) i0 + input_offset);
      const __m256d v2_1 = _mm256_maskload_pd(i1, vmask);
      i1 = (double*) ((uintptr_t) i1 + input_offset);
      const __m256d v2_2 = _mm256_maskload_pd(i2, vmask);
      i2 = (double*) ((uintptr_t) i2 + input_offset);
      const __m256d v2_3 = _mm256_maskload_pd(i3, vmask);
      i3 = (double*) ((uintptr_t) i3 + input_offset);

      const __m256d v1_0 =  _mm256_unpacklo_pd(v2_0, v2_1);
      const __m256d v1_1 = _mm256_unpackhi_pd(v2_0, v2_1);
      const __m256d v1_2 =  _mm256_unpacklo_pd(v2_2, v2_3);
      const __m256d v1_3 = _mm256_unpackhi_pd(v2_2, v2_3);


      double* oN = (double*) ((uintptr_t) o + oN_stride);
      switch (rem) {
        default:
        case 3: {
          const __m256d v0_3 = _mm256_permute2f128_pd(v1_1, v1_3, 0x31);
          _mm256_storeu_pd(oN, v0_3);
          oN = (double*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 2: {
          const __m256d v0_2 = _mm256_permute2f128_pd(v1_0, v1_2, 0x31);
          _mm256_storeu_pd(oN, v0_2);
          oN = (double*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 1: {
          const __m256d v0_1 = _mm256_insertf128_pd(v1_1, _mm256_castpd256_pd128(v1_3), 1);
          _mm256_storeu_pd( oN, v0_1);
        }
        XNN_FALLTHROUGH
        case 0: {
          const __m256d v0_0 = _mm256_insertf128_pd(v1_0, _mm256_castpd256_pd128(v1_2), 1);
          _mm256_storeu_pd(o, v0_0);
          o = (double*) ((uintptr_t) o + tile_hbytes);
        }
      }
    }
    if (bh != 0) {
      const __m256d v2_0 = _mm256_maskload_pd(i0, vmask);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m256d v2_1 = _mm256_maskload_pd(i1, vmask);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      const __m256d v2_2 = _mm256_maskload_pd(i2, vmask);
      const __m256d v2_3 = _mm256_undefined_pd();

      const __m256d v1_0 =  _mm256_unpacklo_pd(v2_0, v2_1);
      const __m256d v1_1 = _mm256_unpackhi_pd(v2_0, v2_1);
      const __m256d v1_2 =  _mm256_unpacklo_pd(v2_2, v2_3);
      const __m256d v1_3 = _mm256_unpackhi_pd(v2_2, v2_3);

      __m256d v0_0 = _mm256_insertf128_pd(v1_0, _mm256_castpd256_pd128(v1_2), 1);
      __m256d v0_2 = _mm256_permute2f128_pd(v1_0, v1_2, 0x31);
      __m256d v0_1 = _mm256_insertf128_pd(v1_1, _mm256_castpd256_pd128(v1_3), 1);
      __m256d v0_3 = _mm256_permute2f128_pd(v1_1, v1_3, 0x31);

      __m128d v0_0_lo = _mm256_castpd256_pd128(v0_0);
      __m128d v0_1_lo = _mm256_castpd256_pd128(v0_1);
      __m128d v0_2_lo = _mm256_castpd256_pd128(v0_2);
      __m128d v0_3_lo = _mm256_castpd256_pd128(v0_3);

      if (bh & 2) {
        double* oN = (double*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 3:
            _mm_storeu_pd(oN, v0_3_lo);
             v0_3_lo = _mm256_extractf128_pd(v0_3, 1);
            oN = (double*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            _mm_storeu_pd(oN, v0_2_lo);
             v0_2_lo = _mm256_extractf128_pd(v0_2, 1);
            oN = (double*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            _mm_storeu_pd(oN, v0_1_lo);
            v0_1_lo = _mm256_extractf128_pd(v0_1, 1);
            XNN_FALLTHROUGH
          case 0:
            _mm_storeu_pd(o, v0_0_lo);
            v0_0_lo = _mm256_extractf128_pd(v0_0, 1);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 2;
      }

      if (bh & 1) {
        double* oN = (double*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 3:
            _mm_storel_pd(oN, v0_3_lo);
            oN = (double*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            _mm_storel_pd(oN, v0_2_lo);
            oN = (double*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            _mm_storel_pd(oN, v0_1_lo);
            XNN_FALLTHROUGH
          case 0:
            _mm_storel_pd(o, v0_0_lo);
            break;
          default:
            XNN_UNREACHABLE;
        }
      }
    }

    i0 = (const double*) ((uintptr_t) i0 + input_reset);
    i1 = (const double*) ((uintptr_t) i0 + input_stride);
    i2 = (const double*) ((uintptr_t) i1 + input_stride);
    i3 = (const double*) ((uintptr_t) i2 + input_stride);
    o = (double*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
