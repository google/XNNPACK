// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <emmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/unaligned.h"


void xnn_qs8_qc4w_packw_gemm_goi_ukernel_x4c8__sse2(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* weights,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 4);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);
  assert(kc % 2 == 0);

  const size_t mock_kc = kc >> 1; // kc in bytes
  const uint32_t izp = (uint32_t) params->input_zero_point + 0;
  const uint32_t kernel_zero_point = (uint32_t) params->kernel_zero_point;

  uint8_t* out = (uint8_t*) packed_weights;
  const int32_t* b = bias;

  const __m128i vmask = _mm_set1_epi8((char)0xF0);
  const __m128i vkernel_zero_point = _mm_set1_epi8((char)(kernel_zero_point * 0x11));


  do {
    // NC main loop multiple of 4
    const uint8_t* w0 = weights;
    size_t n = nc;
    for (; n >= 4; n -= 4) {
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < 4; ++i) {
          packed_b[i] = b[i] * 16;
        }
        b += 4;
      } else {
        for (size_t i = 0; i < 4; ++i) {
          packed_b[i] = 0;
        }
      }
      out += 4 * sizeof(int32_t);

      const uint8_t* w1 = w0 + mock_kc;
      const uint8_t* w2 = w1 + mock_kc;
      const uint8_t* w3 = w2 + mock_kc;

      __m128i vacc0 = _mm_setzero_si128();
      __m128i vacc1 = _mm_setzero_si128();
      __m128i vacc2 = _mm_setzero_si128();
      __m128i vacc3 = _mm_setzero_si128();

      // KC main loop multiple of 16 bytes (32 elements = 4 * KR)
      size_t k = mock_kc;
      for (; k >= 16; k -= 16) {
        __m128i v0 = _mm_loadu_si128((const __m128i*) w0);
        if (kernel_zero_point != 0) {
          v0 = _mm_xor_si128(v0, vkernel_zero_point);
        }
        w0 += 16;
        __m128i v1 = _mm_loadu_si128((const __m128i*) w1);
        if (kernel_zero_point != 0) {
          v1 = _mm_xor_si128(v1, vkernel_zero_point);
        }
        w1 += 16;
        __m128i v2 = _mm_loadu_si128((const __m128i*) w2);
        if (kernel_zero_point != 0) {
          v2 = _mm_xor_si128(v2, vkernel_zero_point);
        }
        w2 += 16;
        __m128i v3 = _mm_loadu_si128((const __m128i*) w3);
        if (kernel_zero_point != 0) {
          v3 = _mm_xor_si128(v3, vkernel_zero_point);
        }
        w3 += 16;

        const __m128i vs0 = _mm_shuffle_epi32(v0, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt0 = _mm_slli_epi32(vs0, 4);
        const __m128i vl0 = _mm_and_si128(vt0, vmask);
        const __m128i vh0 = _mm_and_si128(vs0, vmask);
        const __m128i v01_0 = _mm_unpacklo_epi8(vl0, vh0);
        const __m128i v23_0 = _mm_unpackhi_epi8(vl0, vh0);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_0, v01_0), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_0, v01_0), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc0 = _mm_add_epi32(vacc0, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_0, v23_0), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_0, v23_0), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc0 = _mm_add_epi32(vacc0, v_sum32);
        }

        const __m128i vl01_0 = _mm_srli_epi32(v01_0, 4);
        __m128i vout0 = _mm_or_si128(vl01_0, v23_0);
        const __m128i vs1 = _mm_shuffle_epi32(v1, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt1 = _mm_slli_epi32(vs1, 4);
        const __m128i vl1 = _mm_and_si128(vt1, vmask);
        const __m128i vh1 = _mm_and_si128(vs1, vmask);
        const __m128i v01_1 = _mm_unpacklo_epi8(vl1, vh1);
        const __m128i v23_1 = _mm_unpackhi_epi8(vl1, vh1);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_1, v01_1), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_1, v01_1), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc1 = _mm_add_epi32(vacc1, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_1, v23_1), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_1, v23_1), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc1 = _mm_add_epi32(vacc1, v_sum32);
        }

        const __m128i vl01_1 = _mm_srli_epi32(v01_1, 4);
        __m128i vout1 = _mm_or_si128(vl01_1, v23_1);
        const __m128i vs2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt2 = _mm_slli_epi32(vs2, 4);
        const __m128i vl2 = _mm_and_si128(vt2, vmask);
        const __m128i vh2 = _mm_and_si128(vs2, vmask);
        const __m128i v01_2 = _mm_unpacklo_epi8(vl2, vh2);
        const __m128i v23_2 = _mm_unpackhi_epi8(vl2, vh2);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_2, v01_2), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_2, v01_2), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc2 = _mm_add_epi32(vacc2, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_2, v23_2), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_2, v23_2), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc2 = _mm_add_epi32(vacc2, v_sum32);
        }

        const __m128i vl01_2 = _mm_srli_epi32(v01_2, 4);
        __m128i vout2 = _mm_or_si128(vl01_2, v23_2);
        const __m128i vs3 = _mm_shuffle_epi32(v3, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt3 = _mm_slli_epi32(vs3, 4);
        const __m128i vl3 = _mm_and_si128(vt3, vmask);
        const __m128i vh3 = _mm_and_si128(vs3, vmask);
        const __m128i v01_3 = _mm_unpacklo_epi8(vl3, vh3);
        const __m128i v23_3 = _mm_unpackhi_epi8(vl3, vh3);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_3, v01_3), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_3, v01_3), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc3 = _mm_add_epi32(vacc3, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_3, v23_3), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_3, v23_3), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc3 = _mm_add_epi32(vacc3, v_sum32);
        }

        const __m128i vl01_3 = _mm_srli_epi32(v01_3, 4);
        __m128i vout3 = _mm_or_si128(vl01_3, v23_3);

        const __m128i vout01_0 = _mm_unpacklo_epi64(vout0, vout1);
        const __m128i vout23_0 = _mm_unpackhi_epi64(vout0, vout1);
        _mm_storeu_si128((__m128i*) &out[0 * 8], vout01_0);
        _mm_storeu_si128((__m128i*) &out[(4 + 0) * 8], vout23_0);
        const __m128i vout01_2 = _mm_unpacklo_epi64(vout2, vout3);
        const __m128i vout23_2 = _mm_unpackhi_epi64(vout2, vout3);
        _mm_storeu_si128((__m128i*) &out[2 * 8], vout01_2);
        _mm_storeu_si128((__m128i*) &out[(4 + 2) * 8], vout23_2);

        out += 4 * 16;
      }

      // KC remainder of 1..15 bytes
      if (k != 0) {
        uint8_t temp_w0[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w0[i] = w0[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w0[i] = kernel_zero_point * 0x11;
        }
        __m128i v0 = _mm_loadu_si128((const __m128i*) temp_w0);
        if (kernel_zero_point != 0) {
          v0 = _mm_xor_si128(v0, vkernel_zero_point);
        }
        w0 += k;
        uint8_t temp_w1[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w1[i] = w1[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w1[i] = kernel_zero_point * 0x11;
        }
        __m128i v1 = _mm_loadu_si128((const __m128i*) temp_w1);
        if (kernel_zero_point != 0) {
          v1 = _mm_xor_si128(v1, vkernel_zero_point);
        }
        w1 += k;
        uint8_t temp_w2[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w2[i] = w2[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w2[i] = kernel_zero_point * 0x11;
        }
        __m128i v2 = _mm_loadu_si128((const __m128i*) temp_w2);
        if (kernel_zero_point != 0) {
          v2 = _mm_xor_si128(v2, vkernel_zero_point);
        }
        w2 += k;
        uint8_t temp_w3[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w3[i] = w3[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w3[i] = kernel_zero_point * 0x11;
        }
        __m128i v3 = _mm_loadu_si128((const __m128i*) temp_w3);
        if (kernel_zero_point != 0) {
          v3 = _mm_xor_si128(v3, vkernel_zero_point);
        }
        w3 += k;

        const __m128i vs0 = _mm_shuffle_epi32(v0, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt0 = _mm_slli_epi32(vs0, 4);
        const __m128i vl0 = _mm_and_si128(vt0, vmask);
        const __m128i vh0 = _mm_and_si128(vs0, vmask);
        const __m128i v01_0 = _mm_unpacklo_epi8(vl0, vh0);
        const __m128i v23_0 = _mm_unpackhi_epi8(vl0, vh0);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_0, v01_0), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_0, v01_0), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc0 = _mm_add_epi32(vacc0, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_0, v23_0), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_0, v23_0), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc0 = _mm_add_epi32(vacc0, v_sum32);
        }

        const __m128i vl01_0 = _mm_srli_epi32(v01_0, 4);
        __m128i vout0 = _mm_or_si128(vl01_0, v23_0);
        const __m128i vs1 = _mm_shuffle_epi32(v1, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt1 = _mm_slli_epi32(vs1, 4);
        const __m128i vl1 = _mm_and_si128(vt1, vmask);
        const __m128i vh1 = _mm_and_si128(vs1, vmask);
        const __m128i v01_1 = _mm_unpacklo_epi8(vl1, vh1);
        const __m128i v23_1 = _mm_unpackhi_epi8(vl1, vh1);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_1, v01_1), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_1, v01_1), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc1 = _mm_add_epi32(vacc1, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_1, v23_1), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_1, v23_1), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc1 = _mm_add_epi32(vacc1, v_sum32);
        }

        const __m128i vl01_1 = _mm_srli_epi32(v01_1, 4);
        __m128i vout1 = _mm_or_si128(vl01_1, v23_1);
        const __m128i vs2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt2 = _mm_slli_epi32(vs2, 4);
        const __m128i vl2 = _mm_and_si128(vt2, vmask);
        const __m128i vh2 = _mm_and_si128(vs2, vmask);
        const __m128i v01_2 = _mm_unpacklo_epi8(vl2, vh2);
        const __m128i v23_2 = _mm_unpackhi_epi8(vl2, vh2);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_2, v01_2), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_2, v01_2), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc2 = _mm_add_epi32(vacc2, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_2, v23_2), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_2, v23_2), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc2 = _mm_add_epi32(vacc2, v_sum32);
        }

        const __m128i vl01_2 = _mm_srli_epi32(v01_2, 4);
        __m128i vout2 = _mm_or_si128(vl01_2, v23_2);
        const __m128i vs3 = _mm_shuffle_epi32(v3, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt3 = _mm_slli_epi32(vs3, 4);
        const __m128i vl3 = _mm_and_si128(vt3, vmask);
        const __m128i vh3 = _mm_and_si128(vs3, vmask);
        const __m128i v01_3 = _mm_unpacklo_epi8(vl3, vh3);
        const __m128i v23_3 = _mm_unpackhi_epi8(vl3, vh3);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_3, v01_3), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_3, v01_3), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc3 = _mm_add_epi32(vacc3, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_3, v23_3), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_3, v23_3), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc3 = _mm_add_epi32(vacc3, v_sum32);
        }

        const __m128i vl01_3 = _mm_srli_epi32(v01_3, 4);
        __m128i vout3 = _mm_or_si128(vl01_3, v23_3);

        const __m128i vout01_0 = _mm_unpacklo_epi64(vout0, vout1);
        _mm_storeu_si128((__m128i*) &out[0 * 8], vout01_0);
        const __m128i vout01_2 = _mm_unpacklo_epi64(vout2, vout3);
        _mm_storeu_si128((__m128i*) &out[2 * 8], vout01_2);

        if (k > 8) {
          const __m128i vout23_0 = _mm_unpackhi_epi64(vout0, vout1);
          _mm_storeu_si128((__m128i*) &out[(4 + 0) * 8], vout23_0);
          const __m128i vout23_2 = _mm_unpackhi_epi64(vout2, vout3);
          _mm_storeu_si128((__m128i*) &out[(4 + 2) * 8], vout23_2);
          out += 4 * 16;
        } else {
          out += 4 * 8;
        }
      }

      int32_t ksum0 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc0, _mm_srli_si128(vacc0, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum0 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum1 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc1, _mm_srli_si128(vacc1, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum1 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum2 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc2, _mm_srli_si128(vacc2, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum2 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum3 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc3, _mm_srli_si128(vacc3, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum3 = _mm_cvtsi128_si32(sum1);
      }

      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      packed_b[2] -= ksum2 * izp;
      packed_b[3] -= ksum3 * izp;
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w3;
    }

    // NC remainder (1..3)
    if XNN_UNLIKELY(n != 0) {
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          packed_b[i] = b[i] * 16;
        }
        b += n;
      } else {
        for (size_t i = 0; i < n; ++i) {
          packed_b[i] = 0;
        }
      }
      out += 4 * sizeof(int32_t);

      // Clamp weight pointers
      const uint8_t* w1 = w0 + mock_kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint8_t* w2 = w1 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }

      __m128i vacc0 = _mm_setzero_si128();
      __m128i vacc1 = _mm_setzero_si128();
      __m128i vacc2 = _mm_setzero_si128();

      size_t k = mock_kc;
      for (; k >= 16; k -= 16) {
        __m128i v0 = _mm_loadu_si128((const __m128i*) w0);
        if (kernel_zero_point != 0) {
          v0 = _mm_xor_si128(v0, vkernel_zero_point);
        }
        w0 += 16;
        __m128i v1 = _mm_loadu_si128((const __m128i*) w1);
        if (kernel_zero_point != 0) {
          v1 = _mm_xor_si128(v1, vkernel_zero_point);
        }
        w1 += 16;
        __m128i v2 = _mm_loadu_si128((const __m128i*) w2);
        if (kernel_zero_point != 0) {
          v2 = _mm_xor_si128(v2, vkernel_zero_point);
        }
        w2 += 16;

        const __m128i vs0 = _mm_shuffle_epi32(v0, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt0 = _mm_slli_epi32(vs0, 4);
        const __m128i vl0 = _mm_and_si128(vt0, vmask);
        const __m128i vh0 = _mm_and_si128(vs0, vmask);
        const __m128i v01_0 = _mm_unpacklo_epi8(vl0, vh0);
        const __m128i v23_0 = _mm_unpackhi_epi8(vl0, vh0);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_0, v01_0), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_0, v01_0), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc0 = _mm_add_epi32(vacc0, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_0, v23_0), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_0, v23_0), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc0 = _mm_add_epi32(vacc0, v_sum32);
        }

        const __m128i vl01_0 = _mm_srli_epi32(v01_0, 4);
        __m128i vout0 = _mm_or_si128(vl01_0, v23_0);
        const __m128i vs1 = _mm_shuffle_epi32(v1, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt1 = _mm_slli_epi32(vs1, 4);
        const __m128i vl1 = _mm_and_si128(vt1, vmask);
        const __m128i vh1 = _mm_and_si128(vs1, vmask);
        const __m128i v01_1 = _mm_unpacklo_epi8(vl1, vh1);
        const __m128i v23_1 = _mm_unpackhi_epi8(vl1, vh1);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_1, v01_1), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_1, v01_1), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc1 = _mm_add_epi32(vacc1, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_1, v23_1), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_1, v23_1), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc1 = _mm_add_epi32(vacc1, v_sum32);
        }

        const __m128i vl01_1 = _mm_srli_epi32(v01_1, 4);
        __m128i vout1 = _mm_or_si128(vl01_1, v23_1);
        const __m128i vs2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt2 = _mm_slli_epi32(vs2, 4);
        const __m128i vl2 = _mm_and_si128(vt2, vmask);
        const __m128i vh2 = _mm_and_si128(vs2, vmask);
        const __m128i v01_2 = _mm_unpacklo_epi8(vl2, vh2);
        const __m128i v23_2 = _mm_unpackhi_epi8(vl2, vh2);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_2, v01_2), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_2, v01_2), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc2 = _mm_add_epi32(vacc2, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_2, v23_2), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_2, v23_2), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc2 = _mm_add_epi32(vacc2, v_sum32);
        }

        const __m128i vl01_2 = _mm_srli_epi32(v01_2, 4);
        __m128i vout2 = _mm_or_si128(vl01_2, v23_2);

        const __m128i vout3 = _mm_setzero_si128();

        const __m128i vout01_0 = _mm_unpacklo_epi64(vout0, vout1);
        const __m128i vout23_0 = _mm_unpackhi_epi64(vout0, vout1);
        _mm_storeu_si128((__m128i*) &out[0 * 8], vout01_0);
        _mm_storeu_si128((__m128i*) &out[(4 + 0) * 8], vout23_0);
        const __m128i vout01_2 = _mm_unpacklo_epi64(vout2, vout3);
        const __m128i vout23_2 = _mm_unpackhi_epi64(vout2, vout3);
        _mm_storeu_si128((__m128i*) &out[2 * 8], vout01_2);
        _mm_storeu_si128((__m128i*) &out[(4 + 2) * 8], vout23_2);

        out += 4 * 16;
      }

      if (k != 0) {
        uint8_t temp_w0[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w0[i] = w0[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w0[i] = kernel_zero_point * 0x11;
        }
        __m128i v0 = _mm_loadu_si128((const __m128i*) temp_w0);
        if (kernel_zero_point != 0) {
          v0 = _mm_xor_si128(v0, vkernel_zero_point);
        }
        w0 += k;
        uint8_t temp_w1[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w1[i] = w1[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w1[i] = kernel_zero_point * 0x11;
        }
        __m128i v1 = _mm_loadu_si128((const __m128i*) temp_w1);
        if (kernel_zero_point != 0) {
          v1 = _mm_xor_si128(v1, vkernel_zero_point);
        }
        w1 += k;
        uint8_t temp_w2[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w2[i] = w2[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w2[i] = kernel_zero_point * 0x11;
        }
        __m128i v2 = _mm_loadu_si128((const __m128i*) temp_w2);
        if (kernel_zero_point != 0) {
          v2 = _mm_xor_si128(v2, vkernel_zero_point);
        }
        w2 += k;

        const __m128i vs0 = _mm_shuffle_epi32(v0, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt0 = _mm_slli_epi32(vs0, 4);
        const __m128i vl0 = _mm_and_si128(vt0, vmask);
        const __m128i vh0 = _mm_and_si128(vs0, vmask);
        const __m128i v01_0 = _mm_unpacklo_epi8(vl0, vh0);
        const __m128i v23_0 = _mm_unpackhi_epi8(vl0, vh0);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_0, v01_0), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_0, v01_0), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc0 = _mm_add_epi32(vacc0, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_0, v23_0), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_0, v23_0), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc0 = _mm_add_epi32(vacc0, v_sum32);
        }

        const __m128i vl01_0 = _mm_srli_epi32(v01_0, 4);
        __m128i vout0 = _mm_or_si128(vl01_0, v23_0);
        const __m128i vs1 = _mm_shuffle_epi32(v1, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt1 = _mm_slli_epi32(vs1, 4);
        const __m128i vl1 = _mm_and_si128(vt1, vmask);
        const __m128i vh1 = _mm_and_si128(vs1, vmask);
        const __m128i v01_1 = _mm_unpacklo_epi8(vl1, vh1);
        const __m128i v23_1 = _mm_unpackhi_epi8(vl1, vh1);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_1, v01_1), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_1, v01_1), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc1 = _mm_add_epi32(vacc1, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_1, v23_1), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_1, v23_1), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc1 = _mm_add_epi32(vacc1, v_sum32);
        }

        const __m128i vl01_1 = _mm_srli_epi32(v01_1, 4);
        __m128i vout1 = _mm_or_si128(vl01_1, v23_1);
        const __m128i vs2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt2 = _mm_slli_epi32(vs2, 4);
        const __m128i vl2 = _mm_and_si128(vt2, vmask);
        const __m128i vh2 = _mm_and_si128(vs2, vmask);
        const __m128i v01_2 = _mm_unpacklo_epi8(vl2, vh2);
        const __m128i v23_2 = _mm_unpackhi_epi8(vl2, vh2);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_2, v01_2), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_2, v01_2), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc2 = _mm_add_epi32(vacc2, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_2, v23_2), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_2, v23_2), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc2 = _mm_add_epi32(vacc2, v_sum32);
        }

        const __m128i vl01_2 = _mm_srli_epi32(v01_2, 4);
        __m128i vout2 = _mm_or_si128(vl01_2, v23_2);

        const __m128i vout3 = _mm_setzero_si128();

        const __m128i vout01_0 = _mm_unpacklo_epi64(vout0, vout1);
        _mm_storeu_si128((__m128i*) &out[0 * 8], vout01_0);
        const __m128i vout01_2 = _mm_unpacklo_epi64(vout2, vout3);
        _mm_storeu_si128((__m128i*) &out[2 * 8], vout01_2);

        if (k > 8) {
          const __m128i vout23_0 = _mm_unpackhi_epi64(vout0, vout1);
          _mm_storeu_si128((__m128i*) &out[(4 + 0) * 8], vout23_0);
          const __m128i vout23_2 = _mm_unpackhi_epi64(vout2, vout3);
          _mm_storeu_si128((__m128i*) &out[(4 + 2) * 8], vout23_2);
          out += 4 * 16;
        } else {
          out += 4 * 8;
        }
      }

      int32_t ksum0 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc0, _mm_srli_si128(vacc0, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum0 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum1 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc1, _mm_srli_si128(vacc1, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum1 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum2 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc2, _mm_srli_si128(vacc2, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum2 = _mm_cvtsi128_si32(sum1);
      }

      if (0 < n) {
        packed_b[0] -= ksum0 * izp;
      }
      if (1 < n) {
        packed_b[1] -= ksum1 * izp;
      }
      if (2 < n) {
        packed_b[2] -= ksum2 * izp;
      }
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights = (const uint8_t*)((intptr_t) weights + nc * kc);
  } while (--g != 0);
}
