// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-gio-avxvnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/packw.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_packw_gemm_gio_ukernel_x8c8__avxvnni(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const int8_t* weights,
  const int32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params) XNN_OOB_READS
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 8);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;

  const __m256i vone = _mm256_set1_epi8(1);
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 0): 0);
  __m256i vzeropoint = _mm256_set1_epi32((int32_t) izp);

  do {
    // NC main loop multiple of 8
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 8; n -= 8) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        const __m256i vb0 = _mm256_loadu_si256((const __m256i*) (b + 0));
        _mm256_storeu_si256((__m256i*) (out + 0), vb0);
        b += 8;
      } else {
        _mm256_storeu_si256((__m256i*) (out + 0), _mm256_setzero_si256());
      }
      out += 8 * sizeof(int32_t);

      const int8_t* w1 = w0 + k_stride;
      const int8_t* w2 = w1 + k_stride;
      const int8_t* w3 = w2 + k_stride;
      const int8_t* w4 = w3 + k_stride;
      const int8_t* w5 = w4 + k_stride;
      const int8_t* w6 = w5 + k_stride;
      const int8_t* w7 = w6 + k_stride;

      __m256i vacc0 = _mm256_setzero_si256();
      __m256i vacc4 = _mm256_setzero_si256();

      size_t k = kc;

      // KC main loop multiple of 8x8
      for (; k >= 8; k -= 8) {
        __m128i v0x01234567 = _mm_loadu_si64(w0);
        __m128i v1x01234567 = _mm_loadu_si64(w1);
        __m128i v2x01234567 = _mm_loadu_si64(w2);
        __m128i v3x01234567 = _mm_loadu_si64(w3);
        __m128i v4x01234567 = _mm_loadu_si64(w4);
        __m128i v5x01234567 = _mm_loadu_si64(w5);
        __m128i v6x01234567 = _mm_loadu_si64(w6);
        __m128i v7x01234567 = _mm_loadu_si64(w7);

        __m128i v01x01234567 = _mm_unpacklo_epi8(v0x01234567, v1x01234567);
        __m128i v23x01234567 = _mm_unpacklo_epi8(v2x01234567, v3x01234567);
        __m128i v45x01234567 = _mm_unpacklo_epi8(v4x01234567, v5x01234567);
        __m128i v67x01234567 = _mm_unpacklo_epi8(v6x01234567, v7x01234567);

        __m128i v0123x0123 = _mm_unpacklo_epi16(v01x01234567, v23x01234567);
        __m128i v0123x4567 = _mm_unpackhi_epi16(v01x01234567, v23x01234567);
        __m128i v4567x0123 = _mm_unpacklo_epi16(v45x01234567, v67x01234567);
        __m128i v4567x4567 = _mm_unpackhi_epi16(v45x01234567, v67x01234567);

        __m128i v01234567x01 = _mm_unpacklo_epi32(v0123x0123, v4567x0123);
        __m128i v01234567x23 = _mm_unpackhi_epi32(v0123x0123, v4567x0123);
        __m128i v01234567x45 = _mm_unpacklo_epi32(v0123x4567, v4567x4567);
        __m128i v01234567x67 = _mm_unpackhi_epi32(v0123x4567, v4567x4567);

        __m256i v0 = _mm256_inserti128_si256(_mm256_castsi128_si256(v01234567x01), v01234567x23, 1);
        __m256i v4 = _mm256_inserti128_si256(_mm256_castsi128_si256(v01234567x45), v01234567x67, 1);

        vacc0 = _mm256_dpbusd_avx_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_avx_epi32(vacc4, vone, v4);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);

        w0 += 8 * k_stride;
        w1 += 8 * k_stride;
        w2 += 8 * k_stride;
        w3 += 8 * k_stride;
        w4 += 8 * k_stride;
        w5 += 8 * k_stride;
        w6 += 8 * k_stride;
        w7 += 8 * k_stride;
        out += 64;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);

        __m128i vzero = _mm_setzero_pd();
        __m128i v0x01234567 = _mm_loadu_si64(w0);
        __m128i v1x01234567 = vzero;
        if (1 < k) {
          v1x01234567 = _mm_loadu_si64(w1);
        }
        __m128i v2x01234567 = vzero;
        if (2 < k) {
          v2x01234567 = _mm_loadu_si64(w2);
        }
        __m128i v3x01234567 = vzero;
        if (3 < k) {
          v3x01234567 = _mm_loadu_si64(w3);
        }
        __m128i v4x01234567 = vzero;
        if (4 < k) {
          v4x01234567 = _mm_loadu_si64(w4);
        }
        __m128i v5x01234567 = vzero;
        if (5 < k) {
          v5x01234567 = _mm_loadu_si64(w5);
        }
        __m128i v6x01234567 = vzero;
        if (6 < k) {
          v6x01234567 = _mm_loadu_si64(w6);
        }
        __m128i v7x01234567 = vzero;
        if (7 < k) {
          v7x01234567 = _mm_loadu_si64(w7);
        }

        __m128i v01x01234567 = _mm_unpacklo_epi8(v0x01234567, v1x01234567);
        __m128i v23x01234567 = _mm_unpacklo_epi8(v2x01234567, v3x01234567);
        __m128i v45x01234567 = _mm_unpacklo_epi8(v4x01234567, v5x01234567);
        __m128i v67x01234567 = _mm_unpacklo_epi8(v6x01234567, v7x01234567);

        __m128i v0123x0123 = _mm_unpacklo_epi16(v01x01234567, v23x01234567);
        __m128i v0123x4567 = _mm_unpackhi_epi16(v01x01234567, v23x01234567);
        __m128i v4567x0123 = _mm_unpacklo_epi16(v45x01234567, v67x01234567);
        __m128i v4567x4567 = _mm_unpackhi_epi16(v45x01234567, v67x01234567);

        __m128i v01234567x01 = _mm_unpacklo_epi32(v0123x0123, v4567x0123);
        __m128i v01234567x23 = _mm_unpackhi_epi32(v0123x0123, v4567x0123);
        __m128i v01234567x45 = _mm_unpacklo_epi32(v0123x4567, v4567x4567);
        __m128i v01234567x67 = _mm_unpackhi_epi32(v0123x4567, v4567x4567);

        __m256i v0 = _mm256_inserti128_si256(_mm256_castsi128_si256(v01234567x01), v01234567x23, 1);
        __m256i v4 = _mm256_inserti128_si256(_mm256_castsi128_si256(v01234567x45), v01234567x67, 1);

        vacc0 = _mm256_dpbusd_avx_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_avx_epi32(vacc4, vone, v4);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);

        w0 += 8 * k_stride;
        w1 += 8 * k_stride;
        w2 += 8 * k_stride;
        w3 += 8 * k_stride;
        w4 += 8 * k_stride;
        w5 += 8 * k_stride;
        w6 += 8 * k_stride;
        w7 += 8 * k_stride;
        out += 64;
      }

      __m256i vksum0 = _mm256_hadd_epi32(vacc0, vacc4);
      vksum0 = _mm256_permute4x64_epi64(vksum0, _MM_SHUFFLE(3, 1, 2, 0));
      vksum0 = _mm256_mullo_epi32(vksum0, vzeropoint);
      __m256i vpack0 =  _mm256_loadu_si256((const __m256i*) (packed_b + 0));
      vpack0 = _mm256_sub_epi32(vpack0, vksum0);
      _mm256_storeu_si256((__m256i *) (packed_b + 0), vpack0);
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w0 - kc * k_stride + 8;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *((int32_t*) out) = *b++;
          out += sizeof(int32_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *((int32_t*) out) = 0;
          out += sizeof(int32_t);
        } while (--nb != 0);
      }
      out += (8 - n) * sizeof(int32_t);

     const int8_t* w1 = w0 + k_stride;
     const int8_t* w2 = w1 + k_stride;
     const int8_t* w3 = w2 + k_stride;
     const int8_t* w4 = w3 + k_stride;
     const int8_t* w5 = w4 + k_stride;
     const int8_t* w6 = w5 + k_stride;
     const int8_t* w7 = w6 + k_stride;

     uint32_t ksum0 = 0;
     uint32_t ksum1 = 0;
     uint32_t ksum2 = 0;
     uint32_t ksum3 = 0;
     uint32_t ksum4 = 0;
     uint32_t ksum5 = 0;
     uint32_t ksum6 = 0;

     // KC main loop multiple of 8x8
     size_t k = kc;
     for (; k >= 8; k -= 8) {
       const int8_t v0x0 = w0[0];
       const int8_t v1x0 = w1[0];
       const int8_t v2x0 = w2[0];
       const int8_t v3x0 = w3[0];
       const int8_t v4x0 = w4[0];
       const int8_t v5x0 = w5[0];
       const int8_t v6x0 = w6[0];
       const int8_t v7x0 = w7[0];
       ksum0 += (uint32_t) v0x0;
       ksum0 += (uint32_t) v1x0;
       ksum0 += (uint32_t) v2x0;
       ksum0 += (uint32_t) v3x0;
       ksum0 += (uint32_t) v4x0;
       ksum0 += (uint32_t) v5x0;
       ksum0 += (uint32_t) v6x0;
       ksum0 += (uint32_t) v7x0;
       out[0] = v0x0;
       out[1] = v1x0;
       out[2] = v2x0;
       out[3] = v3x0;
       out[4] = v4x0;
       out[5] = v5x0;
       out[6] = v6x0;
       out[7] = v7x0;
       const int8_t v0x1 = w0[1];
       const int8_t v1x1 = w1[1];
       const int8_t v2x1 = w2[1];
       const int8_t v3x1 = w3[1];
       const int8_t v4x1 = w4[1];
       const int8_t v5x1 = w5[1];
       const int8_t v6x1 = w6[1];
       const int8_t v7x1 = w7[1];
       ksum1 += (uint32_t) v0x1;
       ksum1 += (uint32_t) v1x1;
       ksum1 += (uint32_t) v2x1;
       ksum1 += (uint32_t) v3x1;
       ksum1 += (uint32_t) v4x1;
       ksum1 += (uint32_t) v5x1;
       ksum1 += (uint32_t) v6x1;
       ksum1 += (uint32_t) v7x1;
       out[8] = v0x1;
       out[9] = v1x1;
       out[10] = v2x1;
       out[11] = v3x1;
       out[12] = v4x1;
       out[13] = v5x1;
       out[14] = v6x1;
       out[15] = v7x1;
       const int8_t v0x2 = w0[2];
       const int8_t v1x2 = w1[2];
       const int8_t v2x2 = w2[2];
       const int8_t v3x2 = w3[2];
       const int8_t v4x2 = w4[2];
       const int8_t v5x2 = w5[2];
       const int8_t v6x2 = w6[2];
       const int8_t v7x2 = w7[2];
       ksum2 += (uint32_t) v0x2;
       ksum2 += (uint32_t) v1x2;
       ksum2 += (uint32_t) v2x2;
       ksum2 += (uint32_t) v3x2;
       ksum2 += (uint32_t) v4x2;
       ksum2 += (uint32_t) v5x2;
       ksum2 += (uint32_t) v6x2;
       ksum2 += (uint32_t) v7x2;
       out[16] = v0x2;
       out[17] = v1x2;
       out[18] = v2x2;
       out[19] = v3x2;
       out[20] = v4x2;
       out[21] = v5x2;
       out[22] = v6x2;
       out[23] = v7x2;
       const int8_t v0x3 = w0[3];
       const int8_t v1x3 = w1[3];
       const int8_t v2x3 = w2[3];
       const int8_t v3x3 = w3[3];
       const int8_t v4x3 = w4[3];
       const int8_t v5x3 = w5[3];
       const int8_t v6x3 = w6[3];
       const int8_t v7x3 = w7[3];
       ksum3 += (uint32_t) v0x3;
       ksum3 += (uint32_t) v1x3;
       ksum3 += (uint32_t) v2x3;
       ksum3 += (uint32_t) v3x3;
       ksum3 += (uint32_t) v4x3;
       ksum3 += (uint32_t) v5x3;
       ksum3 += (uint32_t) v6x3;
       ksum3 += (uint32_t) v7x3;
       out[24] = v0x3;
       out[25] = v1x3;
       out[26] = v2x3;
       out[27] = v3x3;
       out[28] = v4x3;
       out[29] = v5x3;
       out[30] = v6x3;
       out[31] = v7x3;
       const int8_t v0x4 = w0[4];
       const int8_t v1x4 = w1[4];
       const int8_t v2x4 = w2[4];
       const int8_t v3x4 = w3[4];
       const int8_t v4x4 = w4[4];
       const int8_t v5x4 = w5[4];
       const int8_t v6x4 = w6[4];
       const int8_t v7x4 = w7[4];
       ksum4 += (uint32_t) v0x4;
       ksum4 += (uint32_t) v1x4;
       ksum4 += (uint32_t) v2x4;
       ksum4 += (uint32_t) v3x4;
       ksum4 += (uint32_t) v4x4;
       ksum4 += (uint32_t) v5x4;
       ksum4 += (uint32_t) v6x4;
       ksum4 += (uint32_t) v7x4;
       out[32] = v0x4;
       out[33] = v1x4;
       out[34] = v2x4;
       out[35] = v3x4;
       out[36] = v4x4;
       out[37] = v5x4;
       out[38] = v6x4;
       out[39] = v7x4;
       const int8_t v0x5 = w0[5];
       const int8_t v1x5 = w1[5];
       const int8_t v2x5 = w2[5];
       const int8_t v3x5 = w3[5];
       const int8_t v4x5 = w4[5];
       const int8_t v5x5 = w5[5];
       const int8_t v6x5 = w6[5];
       const int8_t v7x5 = w7[5];
       ksum5 += (uint32_t) v0x5;
       ksum5 += (uint32_t) v1x5;
       ksum5 += (uint32_t) v2x5;
       ksum5 += (uint32_t) v3x5;
       ksum5 += (uint32_t) v4x5;
       ksum5 += (uint32_t) v5x5;
       ksum5 += (uint32_t) v6x5;
       ksum5 += (uint32_t) v7x5;
       out[40] = v0x5;
       out[41] = v1x5;
       out[42] = v2x5;
       out[43] = v3x5;
       out[44] = v4x5;
       out[45] = v5x5;
       out[46] = v6x5;
       out[47] = v7x5;
       const int8_t v0x6 = w0[6];
       const int8_t v1x6 = w1[6];
       const int8_t v2x6 = w2[6];
       const int8_t v3x6 = w3[6];
       const int8_t v4x6 = w4[6];
       const int8_t v5x6 = w5[6];
       const int8_t v6x6 = w6[6];
       const int8_t v7x6 = w7[6];
       ksum6 += (uint32_t) v0x6;
       ksum6 += (uint32_t) v1x6;
       ksum6 += (uint32_t) v2x6;
       ksum6 += (uint32_t) v3x6;
       ksum6 += (uint32_t) v4x6;
       ksum6 += (uint32_t) v5x6;
       ksum6 += (uint32_t) v6x6;
       ksum6 += (uint32_t) v7x6;
       out[48] = v0x6;
       out[49] = v1x6;
       out[50] = v2x6;
       out[51] = v3x6;
       out[52] = v4x6;
       out[53] = v5x6;
       out[54] = v6x6;
       out[55] = v7x6;
       w0 += 8 * k_stride;
       w1 += 8 * k_stride;
       w2 += 8 * k_stride;
       w3 += 8 * k_stride;
       w4 += 8 * k_stride;
       w5 += 8 * k_stride;
       w6 += 8 * k_stride;
       w7 += 8 * k_stride;
       out += 64;
     }

     // KC remainder of 1..7
     if (k != 0) {
       assert(k >= 1 && k <= 7);
       const int8_t v0x0 = w0[0];
       ksum0 += (uint32_t) v0x0;
       out[0] = v0x0;
       if (1 < k) {
         const int8_t v1x0 = w1[0];
         ksum0 += (uint32_t) v1x0;
         out[1] = v1x0;
       }
       if (2 < k) {
         const int8_t v2x0 = w2[0];
         ksum0 += (uint32_t) v2x0;
         out[2] = v2x0;
       }
       if (3 < k) {
         const int8_t v3x0 = w3[0];
         ksum0 += (uint32_t) v3x0;
         out[3] = v3x0;
       }
       if (4 < k) {
         const int8_t v4x0 = w4[0];
         ksum0 += (uint32_t) v4x0;
         out[4] = v4x0;
       }
       if (5 < k) {
         const int8_t v5x0 = w5[0];
         ksum0 += (uint32_t) v5x0;
         out[5] = v5x0;
       }
       if (6 < k) {
         const int8_t v6x0 = w6[0];
         ksum0 += (uint32_t) v6x0;
         out[6] = v6x0;
       }
       if (7 < k) {
         const int8_t v7x0 = w7[0];
         ksum0 += (uint32_t) v7x0;
         out[7] = v7x0;
       }
       const int8_t v0x1 = w0[1];
       ksum1 += (uint32_t) v0x1;
       out[8] = v0x1;
       if (1 < k) {
         const int8_t v1x1 = w1[1];
         ksum1 += (uint32_t) v1x1;
         out[9] = v1x1;
       }
       if (2 < k) {
         const int8_t v2x1 = w2[1];
         ksum1 += (uint32_t) v2x1;
         out[10] = v2x1;
       }
       if (3 < k) {
         const int8_t v3x1 = w3[1];
         ksum1 += (uint32_t) v3x1;
         out[11] = v3x1;
       }
       if (4 < k) {
         const int8_t v4x1 = w4[1];
         ksum1 += (uint32_t) v4x1;
         out[12] = v4x1;
       }
       if (5 < k) {
         const int8_t v5x1 = w5[1];
         ksum1 += (uint32_t) v5x1;
         out[13] = v5x1;
       }
       if (6 < k) {
         const int8_t v6x1 = w6[1];
         ksum1 += (uint32_t) v6x1;
         out[14] = v6x1;
       }
       if (7 < k) {
         const int8_t v7x1 = w7[1];
         ksum1 += (uint32_t) v7x1;
         out[15] = v7x1;
       }
       const int8_t v0x2 = w0[2];
       ksum2 += (uint32_t) v0x2;
       out[16] = v0x2;
       if (1 < k) {
         const int8_t v1x2 = w1[2];
         ksum2 += (uint32_t) v1x2;
         out[17] = v1x2;
       }
       if (2 < k) {
         const int8_t v2x2 = w2[2];
         ksum2 += (uint32_t) v2x2;
         out[18] = v2x2;
       }
       if (3 < k) {
         const int8_t v3x2 = w3[2];
         ksum2 += (uint32_t) v3x2;
         out[19] = v3x2;
       }
       if (4 < k) {
         const int8_t v4x2 = w4[2];
         ksum2 += (uint32_t) v4x2;
         out[20] = v4x2;
       }
       if (5 < k) {
         const int8_t v5x2 = w5[2];
         ksum2 += (uint32_t) v5x2;
         out[21] = v5x2;
       }
       if (6 < k) {
         const int8_t v6x2 = w6[2];
         ksum2 += (uint32_t) v6x2;
         out[22] = v6x2;
       }
       if (7 < k) {
         const int8_t v7x2 = w7[2];
         ksum2 += (uint32_t) v7x2;
         out[23] = v7x2;
       }
       const int8_t v0x3 = w0[3];
       ksum3 += (uint32_t) v0x3;
       out[24] = v0x3;
       if (1 < k) {
         const int8_t v1x3 = w1[3];
         ksum3 += (uint32_t) v1x3;
         out[25] = v1x3;
       }
       if (2 < k) {
         const int8_t v2x3 = w2[3];
         ksum3 += (uint32_t) v2x3;
         out[26] = v2x3;
       }
       if (3 < k) {
         const int8_t v3x3 = w3[3];
         ksum3 += (uint32_t) v3x3;
         out[27] = v3x3;
       }
       if (4 < k) {
         const int8_t v4x3 = w4[3];
         ksum3 += (uint32_t) v4x3;
         out[28] = v4x3;
       }
       if (5 < k) {
         const int8_t v5x3 = w5[3];
         ksum3 += (uint32_t) v5x3;
         out[29] = v5x3;
       }
       if (6 < k) {
         const int8_t v6x3 = w6[3];
         ksum3 += (uint32_t) v6x3;
         out[30] = v6x3;
       }
       if (7 < k) {
         const int8_t v7x3 = w7[3];
         ksum3 += (uint32_t) v7x3;
         out[31] = v7x3;
       }
       const int8_t v0x4 = w0[4];
       ksum4 += (uint32_t) v0x4;
       out[32] = v0x4;
       if (1 < k) {
         const int8_t v1x4 = w1[4];
         ksum4 += (uint32_t) v1x4;
         out[33] = v1x4;
       }
       if (2 < k) {
         const int8_t v2x4 = w2[4];
         ksum4 += (uint32_t) v2x4;
         out[34] = v2x4;
       }
       if (3 < k) {
         const int8_t v3x4 = w3[4];
         ksum4 += (uint32_t) v3x4;
         out[35] = v3x4;
       }
       if (4 < k) {
         const int8_t v4x4 = w4[4];
         ksum4 += (uint32_t) v4x4;
         out[36] = v4x4;
       }
       if (5 < k) {
         const int8_t v5x4 = w5[4];
         ksum4 += (uint32_t) v5x4;
         out[37] = v5x4;
       }
       if (6 < k) {
         const int8_t v6x4 = w6[4];
         ksum4 += (uint32_t) v6x4;
         out[38] = v6x4;
       }
       if (7 < k) {
         const int8_t v7x4 = w7[4];
         ksum4 += (uint32_t) v7x4;
         out[39] = v7x4;
       }
       const int8_t v0x5 = w0[5];
       ksum5 += (uint32_t) v0x5;
       out[40] = v0x5;
       if (1 < k) {
         const int8_t v1x5 = w1[5];
         ksum5 += (uint32_t) v1x5;
         out[41] = v1x5;
       }
       if (2 < k) {
         const int8_t v2x5 = w2[5];
         ksum5 += (uint32_t) v2x5;
         out[42] = v2x5;
       }
       if (3 < k) {
         const int8_t v3x5 = w3[5];
         ksum5 += (uint32_t) v3x5;
         out[43] = v3x5;
       }
       if (4 < k) {
         const int8_t v4x5 = w4[5];
         ksum5 += (uint32_t) v4x5;
         out[44] = v4x5;
       }
       if (5 < k) {
         const int8_t v5x5 = w5[5];
         ksum5 += (uint32_t) v5x5;
         out[45] = v5x5;
       }
       if (6 < k) {
         const int8_t v6x5 = w6[5];
         ksum5 += (uint32_t) v6x5;
         out[46] = v6x5;
       }
       if (7 < k) {
         const int8_t v7x5 = w7[5];
         ksum5 += (uint32_t) v7x5;
         out[47] = v7x5;
       }
       const int8_t v0x6 = w0[6];
       ksum6 += (uint32_t) v0x6;
       out[48] = v0x6;
       if (1 < k) {
         const int8_t v1x6 = w1[6];
         ksum6 += (uint32_t) v1x6;
         out[49] = v1x6;
       }
       if (2 < k) {
         const int8_t v2x6 = w2[6];
         ksum6 += (uint32_t) v2x6;
         out[50] = v2x6;
       }
       if (3 < k) {
         const int8_t v3x6 = w3[6];
         ksum6 += (uint32_t) v3x6;
         out[51] = v3x6;
       }
       if (4 < k) {
         const int8_t v4x6 = w4[6];
         ksum6 += (uint32_t) v4x6;
         out[52] = v4x6;
       }
       if (5 < k) {
         const int8_t v5x6 = w5[6];
         ksum6 += (uint32_t) v5x6;
         out[53] = v5x6;
       }
       if (6 < k) {
         const int8_t v6x6 = w6[6];
         ksum6 += (uint32_t) v6x6;
         out[54] = v6x6;
       }
       if (7 < k) {
         const int8_t v7x6 = w7[6];
         ksum6 += (uint32_t) v7x6;
         out[55] = v7x6;
       }
       w0 += k * k_stride;
       w1 += k * k_stride;
       w2 += k * k_stride;
       w3 += k * k_stride;
       w4 += k * k_stride;
       w5 += k * k_stride;
       w6 += k * k_stride;
       w7 += k * k_stride;
       out += 64;
     }

     packed_b[0] -= ksum0 * izp;
     packed_b[1] -= ksum1 * izp;
     packed_b[2] -= ksum2 * izp;
     packed_b[3] -= ksum3 * izp;
     packed_b[4] -= ksum4 * izp;
     packed_b[5] -= ksum5 * izp;
     packed_b[6] -= ksum6 * izp;
     out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights += nc * kc;
  } while (--g != 0);
}
