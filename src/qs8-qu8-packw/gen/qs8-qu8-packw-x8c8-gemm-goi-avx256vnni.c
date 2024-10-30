// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-avxvnni.c.in
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


void xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x8c8__avx256vnni(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
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

  const __m256i vone = _mm256_set1_epi8(1);
  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 128): 128);
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
      out += 8 * sizeof(uint32_t);

      const int8_t* w1 = w0 + kc;
      const int8_t* w2 = w1 + kc;
      const int8_t* w3 = w2 + kc;
      const int8_t* w4 = w3 + kc;
      const int8_t* w5 = w4 + kc;
      const int8_t* w6 = w5 + kc;
      const int8_t* w7 = w6 + kc;

      __m256i vacc0 = _mm256_setzero_si256();
      __m256i vacc4 = _mm256_setzero_si256();

      size_t k = kc;
      // KC main loop multiple of 8x32
      for (; k >= 32; k -= 32) {
        const __m256i v0_0123 = _mm256_loadu_si256((const __m256i*) w0);
        const __m256i v1_0123 = _mm256_loadu_si256((const __m256i*) w1);
        const __m256i v2_0123 = _mm256_loadu_si256((const __m256i*) w2);
        const __m256i v3_0123 = _mm256_loadu_si256((const __m256i*) w3);
        const __m256i v4_0123 = _mm256_loadu_si256((const __m256i*) w4);
        const __m256i v5_0123 = _mm256_loadu_si256((const __m256i*) w5);
        const __m256i v6_0123 = _mm256_loadu_si256((const __m256i*) w6);
        const __m256i v7_0123 = _mm256_loadu_si256((const __m256i*) w7);

        const __m256i v01_02 = _mm256_unpacklo_epi64(v0_0123, v1_0123);
        const __m256i v01_13 = _mm256_unpackhi_epi64(v0_0123, v1_0123);
        const __m256i v23_02 = _mm256_unpacklo_epi64(v2_0123, v3_0123);
        const __m256i v23_13 = _mm256_unpackhi_epi64(v2_0123, v3_0123);
        const __m256i v45_02 = _mm256_unpacklo_epi64(v4_0123, v5_0123);
        const __m256i v45_13 = _mm256_unpackhi_epi64(v4_0123, v5_0123);
        const __m256i v67_02 = _mm256_unpacklo_epi64(v6_0123, v7_0123);
        const __m256i v67_13 = _mm256_unpackhi_epi64(v6_0123, v7_0123);


        const __m256i v0_0 = _mm256_permute2f128_si256(v01_02, v23_02, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v0_1 = _mm256_permute2f128_si256(v01_13, v23_13, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v0_2 = _mm256_permute2f128_si256(v01_02, v23_02, _MM_SHUFFLE(0, 3, 0, 1));
        const __m256i v0_3 = _mm256_permute2f128_si256(v01_13, v23_13, _MM_SHUFFLE(0, 3, 0, 1));
        const __m256i v4_0 = _mm256_permute2f128_si256(v45_02, v67_02, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v4_1 = _mm256_permute2f128_si256(v45_13, v67_13, _MM_SHUFFLE(0, 2, 0, 0));
        const __m256i v4_2 = _mm256_permute2f128_si256(v45_02, v67_02, _MM_SHUFFLE(0, 3, 0, 1));
        const __m256i v4_3 = _mm256_permute2f128_si256(v45_13, v67_13, _MM_SHUFFLE(0, 3, 0, 1));

        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0_0);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0_1);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0_2);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0_3);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, v4_0);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, v4_1);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, v4_2);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, v4_3);

        _mm256_storeu_si256((__m256i *)&out[0],  v0_0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4_0);
        _mm256_storeu_si256((__m256i *)&out[64],  v0_1);
        _mm256_storeu_si256((__m256i *)&out[96],  v4_1);
        _mm256_storeu_si256((__m256i *)&out[128],  v0_2);
        _mm256_storeu_si256((__m256i *)&out[160],  v4_2);
        _mm256_storeu_si256((__m256i *)&out[192],  v0_3);
        _mm256_storeu_si256((__m256i *)&out[224],  v4_3);

        w0 += 32;
        w1 += 32;
        w2 += 32;
        w3 += 32;
        w4 += 32;
        w5 += 32;
        w6 += 32;
        w7 += 32;
        out += 256;
      }

      // KC main loop multiple of 8x8
      for (; k >= 8; k -= 8) {
        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);

        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, v4);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        out += 64;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);

        const __m256i vmask = _mm256_srli_epi64(_mm256_set1_epi32(-1), (8 - k) * sizeof(int8_t) * 8);

        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);
        v0 = _mm256_and_si256(v0, vmask);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);
        v4 = _mm256_and_si256(v4, vmask);

        w0 += k;
        w1 += k;
        w2 += k;
        w3 += k;
        w4 += k;
        w5 += k;
        w6 += k;
        w7 += k;

        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, v4);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);

        out += 64;
      }

      __m256i vksum0 = _mm256_hadd_epi32(vacc0, vacc4);
      vksum0 = _mm256_permute4x64_epi64(vksum0, _MM_SHUFFLE(3, 1, 2, 0));
      vksum0 = _mm256_mullo_epi32(vksum0, vzeropoint);
      __m256i vpack0 =  _mm256_loadu_si256((const __m256i*) (packed_b + 0));
      vpack0 = _mm256_sub_epi32(vpack0, vksum0);
      _mm256_storeu_si256((__m256i *) (packed_b + 0), vpack0);
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 7);

      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *((uint32_t*) out) = *b++;
          out += sizeof(uint32_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *((uint32_t*) out) = 0;
          out += sizeof(uint32_t);
        } while (--nb != 0);
      }
      out += (8 - n) * sizeof(uint32_t);

      const int8_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int8_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int8_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int8_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int8_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int8_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }

      __m256i vacc0 = _mm256_setzero_si256();
      __m256i vacc4 = _mm256_setzero_si256();

      // KC main loop multiple of 8x8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);

        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, v4);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        out += 64;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);

        const __m256i vmask = _mm256_srli_epi64(_mm256_set1_epi32(-1), (8 - k) * sizeof(int8_t) * 8);

        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);
        v0 = _mm256_and_si256(v0, vmask);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);
        v4 = _mm256_and_si256(v4, vmask);

        w0 += k;
        w1 += k;
        w2 += k;
        w3 += k;
        w4 += k;
        w5 += k;
        w6 += k;
        w7 += k;

        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, v4);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);

        out += 64;
      }

      __m256i vksum0 = _mm256_hadd_epi32(vacc0, vacc4);
      vksum0 = _mm256_permute4x64_epi64(vksum0, _MM_SHUFFLE(3, 1, 2, 0));
      vksum0 = _mm256_mullo_epi32(vksum0, vzeropoint);
      __m256i vpack0 =  _mm256_loadu_si256((const __m256i*) (packed_b + 0));
      vpack0 = _mm256_sub_epi32(vpack0, vksum0);
      _mm256_storeu_si256((__m256i *) (packed_b + 0), vpack0);
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights += nc * kc;
  } while (--g != 0);
}
