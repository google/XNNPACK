// clang-format off
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

#include "src/xnnpack/packw.h"
#include "src/xnnpack/unaligned.h"

XNN_INLINE static uint64_t safe_load_u64(const void* src, size_t n) {
  uint64_t value = 0;
  const uint8_t* bytes = (const uint8_t*)src;
  for (size_t i = 0; i < n; ++i) {
    value |= (uint64_t)bytes[i] << (i * 8);
  }
  return value;
}

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
  const void* params)
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

        __m128i vzero = _mm_setzero_si128();
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

     __m256i vacc0 = _mm256_setzero_si256();
     __m256i vacc4 = _mm256_setzero_si256();

     size_t k = kc;

     // KC main loop multiple of 8x8
     for (; k >= 8; k -= 8) {
       __m128i v0x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w0, n));
       __m128i v1x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w1, n));
       __m128i v2x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w2, n));
       __m128i v3x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w3, n));
       __m128i v4x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w4, n));
       __m128i v5x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w5, n));
       __m128i v6x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w6, n));
       __m128i v7x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w7, n));

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

       __m128i vzero = _mm_setzero_si128();
       __m128i v0x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w0, n));
       __m128i v1x01234567 = vzero;
       if (1 < k) {
         v1x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w1, n));
       }
       __m128i v2x01234567 = vzero;
       if (2 < k) {
         v2x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w2, n));
       }
       __m128i v3x01234567 = vzero;
       if (3 < k) {
         v3x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w3, n));
       }
       __m128i v4x01234567 = vzero;
       if (4 < k) {
         v4x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w4, n));
       }
       __m128i v5x01234567 = vzero;
       if (5 < k) {
         v5x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w5, n));
       }
       __m128i v6x01234567 = vzero;
       if (6 < k) {
         v6x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w6, n));
       }
       __m128i v7x01234567 = vzero;
       if (7 < k) {
         v7x01234567 = _mm_set1_epi64x((int64_t) safe_load_u64(w7, n));
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

     __m256i vksum0 = _mm256_hadd_epi32(vacc0, vacc4);
     vksum0 = _mm256_permute4x64_epi64(vksum0, _MM_SHUFFLE(3, 1, 2, 0));
     vksum0 = _mm256_mullo_epi32(vksum0, vzeropoint);
     __m256i vpack0 =  _mm256_loadu_si256((const __m256i*) (packed_b + 0));
     vpack0 = _mm256_sub_epi32(vpack0, vksum0);
     _mm256_storeu_si256((__m256i *) (packed_b + 0), vpack0);
     out = (int8_t*) ((uintptr_t) out + extra_bytes);
     w0 = w0 - kc * k_stride + 8;
    }

    weights += nc * kc;
  } while (--g != 0);
}
