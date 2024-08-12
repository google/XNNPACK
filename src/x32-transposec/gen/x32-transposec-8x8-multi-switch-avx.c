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

void xnn_x32_transposec_ukernel__8x8_multi_switch_avx(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  static const int32_t mask_table[15] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  assert(block_width == 1 || output_stride >= block_height * sizeof(float));
  assert(block_height == 1 || input_stride >= block_width * sizeof(float));

  const size_t tile_height = 8;
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(float);
  const size_t tile_wbytes = tile_width * sizeof(float);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(float);

  const float* i0 = (const float*) input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_stride);
  float* o = (float*) output;
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 7);
    const size_t oN_stride = rem * output_stride;

    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[rem ^ 7]));

    size_t bh = block_height;
    for (; bh >= 8; bh -= 8) {
      const __m256 v3_0 = _mm256_maskload_ps(i0, vmask);
      i0 = (float*) ((uintptr_t) i0 + input_offset);
      const __m256 v3_1 = _mm256_maskload_ps(i1, vmask);
      i1 = (float*) ((uintptr_t) i1 + input_offset);
      const __m256 v3_2 = _mm256_maskload_ps(i2, vmask);
      i2 = (float*) ((uintptr_t) i2 + input_offset);
      const __m256 v3_3 = _mm256_maskload_ps(i3, vmask);
      i3 = (float*) ((uintptr_t) i3 + input_offset);
      const __m256 v3_4 = _mm256_maskload_ps(i4, vmask);
      i4 = (float*) ((uintptr_t) i4 + input_offset);
      const __m256 v3_5 = _mm256_maskload_ps(i5, vmask);
      i5 = (float*) ((uintptr_t) i5 + input_offset);
      const __m256 v3_6 = _mm256_maskload_ps(i6, vmask);
      i6 = (float*) ((uintptr_t) i6 + input_offset);
      const __m256 v3_7 = _mm256_maskload_ps(i7, vmask);
      i7 = (float*) ((uintptr_t) i7 + input_offset);

      const __m256 v2_0 =  _mm256_unpacklo_ps(v3_0, v3_2);
      const __m256 v2_1 = _mm256_unpackhi_ps(v3_0, v3_2);
      const __m256 v2_2 =  _mm256_unpacklo_ps(v3_1, v3_3);
      const __m256 v2_3 = _mm256_unpackhi_ps(v3_1, v3_3);
      const __m256 v2_4 =  _mm256_unpacklo_ps(v3_4, v3_6);
      const __m256 v2_5 = _mm256_unpackhi_ps(v3_4, v3_6);
      const __m256 v2_6 =  _mm256_unpacklo_ps(v3_5, v3_7);
      const __m256 v2_7 = _mm256_unpackhi_ps(v3_5, v3_7);
      const __m256 v1_0 =  _mm256_unpacklo_ps(v2_0, v2_2);
      const __m256 v1_1 = _mm256_unpackhi_ps(v2_0, v2_2);
      const __m256 v1_2 =  _mm256_unpacklo_ps(v2_1, v2_3);
      const __m256 v1_3 = _mm256_unpackhi_ps(v2_1, v2_3);
      const __m256 v1_4 =  _mm256_unpacklo_ps(v2_4, v2_6);
      const __m256 v1_5 = _mm256_unpackhi_ps(v2_4, v2_6);
      const __m256 v1_6 =  _mm256_unpacklo_ps(v2_5, v2_7);
      const __m256 v1_7 = _mm256_unpackhi_ps(v2_5, v2_7);


      float* oN = (float*) ((uintptr_t) o + oN_stride);
      switch (rem) {
        default:
        case 7: {
          const __m256 v0_7 = _mm256_permute2f128_ps(v1_3, v1_7, 0x31);
          _mm256_storeu_ps(oN, v0_7);
          oN = (float*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 6: {
          const __m256 v0_6 = _mm256_permute2f128_ps(v1_2, v1_6, 0x31);
          _mm256_storeu_ps(oN, v0_6);
          oN = (float*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 5: {
          const __m256 v0_5 = _mm256_permute2f128_ps(v1_1, v1_5, 0x31);
          _mm256_storeu_ps(oN, v0_5);
          oN = (float*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 4: {
          const __m256 v0_4 = _mm256_permute2f128_ps(v1_0, v1_4, 0x31);
          _mm256_storeu_ps(oN, v0_4);
          oN = (float*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 3: {
          const __m256 v0_3 = _mm256_insertf128_ps(v1_3, _mm256_castps256_ps128(v1_7), 1);
          _mm256_storeu_ps(oN, v0_3);
          oN = (float*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 2: {
          const __m256 v0_2 = _mm256_insertf128_ps(v1_2, _mm256_castps256_ps128(v1_6), 1);
          _mm256_storeu_ps(oN, v0_2);
          oN = (float*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 1: {
          const __m256 v0_1 = _mm256_insertf128_ps(v1_1, _mm256_castps256_ps128(v1_5), 1);
          _mm256_storeu_ps( oN, v0_1);
        }
        XNN_FALLTHROUGH
        case 0: {
          const __m256 v0_0 = _mm256_insertf128_ps(v1_0, _mm256_castps256_ps128(v1_4), 1);
          _mm256_storeu_ps(o, v0_0);
          o = (float*) ((uintptr_t) o + tile_hbytes);
        }
      }
    }
    if (bh != 0) {
      const __m256 v3_0 = _mm256_maskload_ps(i0, vmask);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m256 v3_1 = _mm256_maskload_ps(i1, vmask);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      const __m256 v3_2 = _mm256_maskload_ps(i2, vmask);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i0;
      }
      const __m256 v3_3 = _mm256_maskload_ps(i3, vmask);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i0;
      }
      const __m256 v3_4 = _mm256_maskload_ps(i4, vmask);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i0;
      }
      const __m256 v3_5 = _mm256_maskload_ps(i5, vmask);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i0;
      }
      const __m256 v3_6 = _mm256_maskload_ps(i6, vmask);
      const __m256 v3_7 = _mm256_undefined_ps();

      const __m256 v2_0 =  _mm256_unpacklo_ps(v3_0, v3_2);
      const __m256 v2_1 = _mm256_unpackhi_ps(v3_0, v3_2);
      const __m256 v2_2 =  _mm256_unpacklo_ps(v3_1, v3_3);
      const __m256 v2_3 = _mm256_unpackhi_ps(v3_1, v3_3);
      const __m256 v2_4 =  _mm256_unpacklo_ps(v3_4, v3_6);
      const __m256 v2_5 = _mm256_unpackhi_ps(v3_4, v3_6);
      const __m256 v2_6 =  _mm256_unpacklo_ps(v3_5, v3_7);
      const __m256 v2_7 = _mm256_unpackhi_ps(v3_5, v3_7);
      const __m256 v1_0 =  _mm256_unpacklo_ps(v2_0, v2_2);
      const __m256 v1_1 = _mm256_unpackhi_ps(v2_0, v2_2);
      const __m256 v1_2 =  _mm256_unpacklo_ps(v2_1, v2_3);
      const __m256 v1_3 = _mm256_unpackhi_ps(v2_1, v2_3);
      const __m256 v1_4 =  _mm256_unpacklo_ps(v2_4, v2_6);
      const __m256 v1_5 = _mm256_unpackhi_ps(v2_4, v2_6);
      const __m256 v1_6 =  _mm256_unpacklo_ps(v2_5, v2_7);
      const __m256 v1_7 = _mm256_unpackhi_ps(v2_5, v2_7);

      __m256 v0_0 = _mm256_insertf128_ps(v1_0, _mm256_castps256_ps128(v1_4), 1);
      __m256 v0_4 = _mm256_permute2f128_ps(v1_0, v1_4, 0x31);
      __m256 v0_1 = _mm256_insertf128_ps(v1_1, _mm256_castps256_ps128(v1_5), 1);
      __m256 v0_5 = _mm256_permute2f128_ps(v1_1, v1_5, 0x31);
      __m256 v0_2 = _mm256_insertf128_ps(v1_2, _mm256_castps256_ps128(v1_6), 1);
      __m256 v0_6 = _mm256_permute2f128_ps(v1_2, v1_6, 0x31);
      __m256 v0_3 = _mm256_insertf128_ps(v1_3, _mm256_castps256_ps128(v1_7), 1);
      __m256 v0_7 = _mm256_permute2f128_ps(v1_3, v1_7, 0x31);

      __m128 v0_0_lo = _mm256_castps256_ps128(v0_0);
      __m128 v0_1_lo = _mm256_castps256_ps128(v0_1);
      __m128 v0_2_lo = _mm256_castps256_ps128(v0_2);
      __m128 v0_3_lo = _mm256_castps256_ps128(v0_3);
      __m128 v0_4_lo = _mm256_castps256_ps128(v0_4);
      __m128 v0_5_lo = _mm256_castps256_ps128(v0_5);
      __m128 v0_6_lo = _mm256_castps256_ps128(v0_6);
      __m128 v0_7_lo = _mm256_castps256_ps128(v0_7);

      if (bh & 4) {
        float* oN = (float*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            _mm_storeu_ps(oN, v0_7_lo);
             v0_7_lo = _mm256_extractf128_ps(v0_7, 1);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            _mm_storeu_ps(oN, v0_6_lo);
             v0_6_lo = _mm256_extractf128_ps(v0_6, 1);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            _mm_storeu_ps(oN, v0_5_lo);
             v0_5_lo = _mm256_extractf128_ps(v0_5, 1);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            _mm_storeu_ps(oN, v0_4_lo);
             v0_4_lo = _mm256_extractf128_ps(v0_4, 1);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            _mm_storeu_ps(oN, v0_3_lo);
             v0_3_lo = _mm256_extractf128_ps(v0_3, 1);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            _mm_storeu_ps(oN, v0_2_lo);
             v0_2_lo = _mm256_extractf128_ps(v0_2, 1);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            _mm_storeu_ps(oN, v0_1_lo);
            v0_1_lo = _mm256_extractf128_ps(v0_1, 1);
            XNN_FALLTHROUGH
          case 0:
            _mm_storeu_ps(o, v0_0_lo);
            v0_0_lo = _mm256_extractf128_ps(v0_0, 1);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 4;
      }

      if (bh & 2) {
        float* oN = (float*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            _mm_storel_pi((__m64*) oN, v0_7_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            _mm_storel_pi((__m64*) oN, v0_6_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            _mm_storel_pi((__m64*) oN, v0_5_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            _mm_storel_pi((__m64*) oN, v0_4_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            _mm_storel_pi((__m64*) oN, v0_3_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            _mm_storel_pi((__m64*) oN, v0_2_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            _mm_storel_pi((__m64*) oN, v0_1_lo);
            XNN_FALLTHROUGH
          case 0:
            _mm_storel_pi((__m64*) o, v0_0_lo);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 2;
        v0_0_lo = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v0_0_lo), _mm_castps_pd(v0_0_lo)));
        v0_1_lo = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v0_1_lo), _mm_castps_pd(v0_1_lo)));
        v0_2_lo = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v0_2_lo), _mm_castps_pd(v0_2_lo)));
        v0_3_lo = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v0_3_lo), _mm_castps_pd(v0_3_lo)));
        v0_4_lo = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v0_4_lo), _mm_castps_pd(v0_4_lo)));
        v0_5_lo = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v0_5_lo), _mm_castps_pd(v0_5_lo)));
        v0_6_lo = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v0_6_lo), _mm_castps_pd(v0_6_lo)));
        v0_7_lo = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v0_7_lo), _mm_castps_pd(v0_7_lo)));
      }
      if (bh & 1) {
        float* oN = (float*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            _mm_store_ss(oN, v0_7_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            _mm_store_ss(oN, v0_6_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            _mm_store_ss(oN, v0_5_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            _mm_store_ss(oN, v0_4_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            _mm_store_ss(oN, v0_3_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            _mm_store_ss(oN, v0_2_lo);
            oN = (float*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            _mm_store_ss(oN, v0_1_lo);
            XNN_FALLTHROUGH
          case 0:
            _mm_store_ss(o, v0_0_lo);
            break;
          default:
            XNN_UNREACHABLE;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i0 + input_reset);
    i1 = (const float*) ((uintptr_t) i0 + input_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_stride);
    i4 = (const float*) ((uintptr_t) i3 + input_stride);
    i5 = (const float*) ((uintptr_t) i4 + input_stride);
    i6 = (const float*) ((uintptr_t) i5 + input_stride);
    i7 = (const float*) ((uintptr_t) i6 + input_stride);
    o = (float*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
