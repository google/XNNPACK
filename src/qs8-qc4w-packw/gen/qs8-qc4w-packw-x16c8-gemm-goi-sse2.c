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


void xnn_qs8_qc4w_packw_gemm_goi_ukernel_x16c8__sse2(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t n_stride,
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
  assert(nr == 16);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);

  const size_t mock_kc = kc >> 1; // kc in bytes
  // Rows are byte aligned, so when KC is odd the last byte of each row holds
  // 1 weight and its high nibble is padding that is replaced by the kernel
  // zero point.
  const size_t kc_odd = kc & 1;
  const uint32_t izp = (uint32_t) params->input_zero_point + 0;
  const uint32_t kernel_zero_point = (uint32_t) params->kernel_zero_point;

  uint8_t* out = (uint8_t*) packed_weights;
  const int32_t* b = bias;

  const __m128i vmask = _mm_set1_epi8((char)0xF0);
  const __m128i vkernel_zero_point = _mm_set1_epi8((char)(kernel_zero_point * 0x11));


  do {
    // NC main loop multiple of 16
    const uint8_t* w0 = weights;
    size_t n = nc;
    for (; n >= 16; n -= 16) {
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < 16; ++i) {
          packed_b[i] = b[i] * 16;
        }
        b += 16;
      } else {
        for (size_t i = 0; i < 16; ++i) {
          packed_b[i] = 0;
        }
      }
      out += 16 * sizeof(int32_t);

      const uint8_t* w1 = w0 + (n_stride >> 1);
      const uint8_t* w2 = w1 + (n_stride >> 1);
      const uint8_t* w3 = w2 + (n_stride >> 1);
      const uint8_t* w4 = w3 + (n_stride >> 1);
      const uint8_t* w5 = w4 + (n_stride >> 1);
      const uint8_t* w6 = w5 + (n_stride >> 1);
      const uint8_t* w7 = w6 + (n_stride >> 1);
      const uint8_t* w8 = w7 + (n_stride >> 1);
      const uint8_t* w9 = w8 + (n_stride >> 1);
      const uint8_t* w10 = w9 + (n_stride >> 1);
      const uint8_t* w11 = w10 + (n_stride >> 1);
      const uint8_t* w12 = w11 + (n_stride >> 1);
      const uint8_t* w13 = w12 + (n_stride >> 1);
      const uint8_t* w14 = w13 + (n_stride >> 1);
      const uint8_t* w15 = w14 + (n_stride >> 1);

      __m128i vacc0 = _mm_setzero_si128();
      __m128i vacc1 = _mm_setzero_si128();
      __m128i vacc2 = _mm_setzero_si128();
      __m128i vacc3 = _mm_setzero_si128();
      __m128i vacc4 = _mm_setzero_si128();
      __m128i vacc5 = _mm_setzero_si128();
      __m128i vacc6 = _mm_setzero_si128();
      __m128i vacc7 = _mm_setzero_si128();
      __m128i vacc8 = _mm_setzero_si128();
      __m128i vacc9 = _mm_setzero_si128();
      __m128i vacc10 = _mm_setzero_si128();
      __m128i vacc11 = _mm_setzero_si128();
      __m128i vacc12 = _mm_setzero_si128();
      __m128i vacc13 = _mm_setzero_si128();
      __m128i vacc14 = _mm_setzero_si128();
      __m128i vacc15 = _mm_setzero_si128();

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
        __m128i v4 = _mm_loadu_si128((const __m128i*) w4);
        if (kernel_zero_point != 0) {
          v4 = _mm_xor_si128(v4, vkernel_zero_point);
        }
        w4 += 16;
        __m128i v5 = _mm_loadu_si128((const __m128i*) w5);
        if (kernel_zero_point != 0) {
          v5 = _mm_xor_si128(v5, vkernel_zero_point);
        }
        w5 += 16;
        __m128i v6 = _mm_loadu_si128((const __m128i*) w6);
        if (kernel_zero_point != 0) {
          v6 = _mm_xor_si128(v6, vkernel_zero_point);
        }
        w6 += 16;
        __m128i v7 = _mm_loadu_si128((const __m128i*) w7);
        if (kernel_zero_point != 0) {
          v7 = _mm_xor_si128(v7, vkernel_zero_point);
        }
        w7 += 16;
        __m128i v8 = _mm_loadu_si128((const __m128i*) w8);
        if (kernel_zero_point != 0) {
          v8 = _mm_xor_si128(v8, vkernel_zero_point);
        }
        w8 += 16;
        __m128i v9 = _mm_loadu_si128((const __m128i*) w9);
        if (kernel_zero_point != 0) {
          v9 = _mm_xor_si128(v9, vkernel_zero_point);
        }
        w9 += 16;
        __m128i v10 = _mm_loadu_si128((const __m128i*) w10);
        if (kernel_zero_point != 0) {
          v10 = _mm_xor_si128(v10, vkernel_zero_point);
        }
        w10 += 16;
        __m128i v11 = _mm_loadu_si128((const __m128i*) w11);
        if (kernel_zero_point != 0) {
          v11 = _mm_xor_si128(v11, vkernel_zero_point);
        }
        w11 += 16;
        __m128i v12 = _mm_loadu_si128((const __m128i*) w12);
        if (kernel_zero_point != 0) {
          v12 = _mm_xor_si128(v12, vkernel_zero_point);
        }
        w12 += 16;
        __m128i v13 = _mm_loadu_si128((const __m128i*) w13);
        if (kernel_zero_point != 0) {
          v13 = _mm_xor_si128(v13, vkernel_zero_point);
        }
        w13 += 16;
        __m128i v14 = _mm_loadu_si128((const __m128i*) w14);
        if (kernel_zero_point != 0) {
          v14 = _mm_xor_si128(v14, vkernel_zero_point);
        }
        w14 += 16;
        __m128i v15 = _mm_loadu_si128((const __m128i*) w15);
        if (kernel_zero_point != 0) {
          v15 = _mm_xor_si128(v15, vkernel_zero_point);
        }
        w15 += 16;

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
        const __m128i vs4 = _mm_shuffle_epi32(v4, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt4 = _mm_slli_epi32(vs4, 4);
        const __m128i vl4 = _mm_and_si128(vt4, vmask);
        const __m128i vh4 = _mm_and_si128(vs4, vmask);
        const __m128i v01_4 = _mm_unpacklo_epi8(vl4, vh4);
        const __m128i v23_4 = _mm_unpackhi_epi8(vl4, vh4);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_4, v01_4), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_4, v01_4), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc4 = _mm_add_epi32(vacc4, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_4, v23_4), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_4, v23_4), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc4 = _mm_add_epi32(vacc4, v_sum32);
        }

        const __m128i vl01_4 = _mm_srli_epi32(v01_4, 4);
        __m128i vout4 = _mm_or_si128(vl01_4, v23_4);
        const __m128i vs5 = _mm_shuffle_epi32(v5, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt5 = _mm_slli_epi32(vs5, 4);
        const __m128i vl5 = _mm_and_si128(vt5, vmask);
        const __m128i vh5 = _mm_and_si128(vs5, vmask);
        const __m128i v01_5 = _mm_unpacklo_epi8(vl5, vh5);
        const __m128i v23_5 = _mm_unpackhi_epi8(vl5, vh5);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_5, v01_5), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_5, v01_5), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc5 = _mm_add_epi32(vacc5, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_5, v23_5), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_5, v23_5), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc5 = _mm_add_epi32(vacc5, v_sum32);
        }

        const __m128i vl01_5 = _mm_srli_epi32(v01_5, 4);
        __m128i vout5 = _mm_or_si128(vl01_5, v23_5);
        const __m128i vs6 = _mm_shuffle_epi32(v6, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt6 = _mm_slli_epi32(vs6, 4);
        const __m128i vl6 = _mm_and_si128(vt6, vmask);
        const __m128i vh6 = _mm_and_si128(vs6, vmask);
        const __m128i v01_6 = _mm_unpacklo_epi8(vl6, vh6);
        const __m128i v23_6 = _mm_unpackhi_epi8(vl6, vh6);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_6, v01_6), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_6, v01_6), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc6 = _mm_add_epi32(vacc6, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_6, v23_6), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_6, v23_6), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc6 = _mm_add_epi32(vacc6, v_sum32);
        }

        const __m128i vl01_6 = _mm_srli_epi32(v01_6, 4);
        __m128i vout6 = _mm_or_si128(vl01_6, v23_6);
        const __m128i vs7 = _mm_shuffle_epi32(v7, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt7 = _mm_slli_epi32(vs7, 4);
        const __m128i vl7 = _mm_and_si128(vt7, vmask);
        const __m128i vh7 = _mm_and_si128(vs7, vmask);
        const __m128i v01_7 = _mm_unpacklo_epi8(vl7, vh7);
        const __m128i v23_7 = _mm_unpackhi_epi8(vl7, vh7);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_7, v01_7), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_7, v01_7), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc7 = _mm_add_epi32(vacc7, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_7, v23_7), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_7, v23_7), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc7 = _mm_add_epi32(vacc7, v_sum32);
        }

        const __m128i vl01_7 = _mm_srli_epi32(v01_7, 4);
        __m128i vout7 = _mm_or_si128(vl01_7, v23_7);
        const __m128i vs8 = _mm_shuffle_epi32(v8, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt8 = _mm_slli_epi32(vs8, 4);
        const __m128i vl8 = _mm_and_si128(vt8, vmask);
        const __m128i vh8 = _mm_and_si128(vs8, vmask);
        const __m128i v01_8 = _mm_unpacklo_epi8(vl8, vh8);
        const __m128i v23_8 = _mm_unpackhi_epi8(vl8, vh8);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_8, v01_8), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_8, v01_8), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc8 = _mm_add_epi32(vacc8, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_8, v23_8), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_8, v23_8), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc8 = _mm_add_epi32(vacc8, v_sum32);
        }

        const __m128i vl01_8 = _mm_srli_epi32(v01_8, 4);
        __m128i vout8 = _mm_or_si128(vl01_8, v23_8);
        const __m128i vs9 = _mm_shuffle_epi32(v9, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt9 = _mm_slli_epi32(vs9, 4);
        const __m128i vl9 = _mm_and_si128(vt9, vmask);
        const __m128i vh9 = _mm_and_si128(vs9, vmask);
        const __m128i v01_9 = _mm_unpacklo_epi8(vl9, vh9);
        const __m128i v23_9 = _mm_unpackhi_epi8(vl9, vh9);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_9, v01_9), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_9, v01_9), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc9 = _mm_add_epi32(vacc9, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_9, v23_9), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_9, v23_9), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc9 = _mm_add_epi32(vacc9, v_sum32);
        }

        const __m128i vl01_9 = _mm_srli_epi32(v01_9, 4);
        __m128i vout9 = _mm_or_si128(vl01_9, v23_9);
        const __m128i vs10 = _mm_shuffle_epi32(v10, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt10 = _mm_slli_epi32(vs10, 4);
        const __m128i vl10 = _mm_and_si128(vt10, vmask);
        const __m128i vh10 = _mm_and_si128(vs10, vmask);
        const __m128i v01_10 = _mm_unpacklo_epi8(vl10, vh10);
        const __m128i v23_10 = _mm_unpackhi_epi8(vl10, vh10);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_10, v01_10), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_10, v01_10), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc10 = _mm_add_epi32(vacc10, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_10, v23_10), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_10, v23_10), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc10 = _mm_add_epi32(vacc10, v_sum32);
        }

        const __m128i vl01_10 = _mm_srli_epi32(v01_10, 4);
        __m128i vout10 = _mm_or_si128(vl01_10, v23_10);
        const __m128i vs11 = _mm_shuffle_epi32(v11, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt11 = _mm_slli_epi32(vs11, 4);
        const __m128i vl11 = _mm_and_si128(vt11, vmask);
        const __m128i vh11 = _mm_and_si128(vs11, vmask);
        const __m128i v01_11 = _mm_unpacklo_epi8(vl11, vh11);
        const __m128i v23_11 = _mm_unpackhi_epi8(vl11, vh11);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_11, v01_11), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_11, v01_11), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc11 = _mm_add_epi32(vacc11, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_11, v23_11), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_11, v23_11), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc11 = _mm_add_epi32(vacc11, v_sum32);
        }

        const __m128i vl01_11 = _mm_srli_epi32(v01_11, 4);
        __m128i vout11 = _mm_or_si128(vl01_11, v23_11);
        const __m128i vs12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt12 = _mm_slli_epi32(vs12, 4);
        const __m128i vl12 = _mm_and_si128(vt12, vmask);
        const __m128i vh12 = _mm_and_si128(vs12, vmask);
        const __m128i v01_12 = _mm_unpacklo_epi8(vl12, vh12);
        const __m128i v23_12 = _mm_unpackhi_epi8(vl12, vh12);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_12, v01_12), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_12, v01_12), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc12 = _mm_add_epi32(vacc12, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_12, v23_12), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_12, v23_12), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc12 = _mm_add_epi32(vacc12, v_sum32);
        }

        const __m128i vl01_12 = _mm_srli_epi32(v01_12, 4);
        __m128i vout12 = _mm_or_si128(vl01_12, v23_12);
        const __m128i vs13 = _mm_shuffle_epi32(v13, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt13 = _mm_slli_epi32(vs13, 4);
        const __m128i vl13 = _mm_and_si128(vt13, vmask);
        const __m128i vh13 = _mm_and_si128(vs13, vmask);
        const __m128i v01_13 = _mm_unpacklo_epi8(vl13, vh13);
        const __m128i v23_13 = _mm_unpackhi_epi8(vl13, vh13);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_13, v01_13), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_13, v01_13), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc13 = _mm_add_epi32(vacc13, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_13, v23_13), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_13, v23_13), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc13 = _mm_add_epi32(vacc13, v_sum32);
        }

        const __m128i vl01_13 = _mm_srli_epi32(v01_13, 4);
        __m128i vout13 = _mm_or_si128(vl01_13, v23_13);
        const __m128i vs14 = _mm_shuffle_epi32(v14, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt14 = _mm_slli_epi32(vs14, 4);
        const __m128i vl14 = _mm_and_si128(vt14, vmask);
        const __m128i vh14 = _mm_and_si128(vs14, vmask);
        const __m128i v01_14 = _mm_unpacklo_epi8(vl14, vh14);
        const __m128i v23_14 = _mm_unpackhi_epi8(vl14, vh14);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_14, v01_14), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_14, v01_14), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc14 = _mm_add_epi32(vacc14, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_14, v23_14), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_14, v23_14), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc14 = _mm_add_epi32(vacc14, v_sum32);
        }

        const __m128i vl01_14 = _mm_srli_epi32(v01_14, 4);
        __m128i vout14 = _mm_or_si128(vl01_14, v23_14);
        const __m128i vs15 = _mm_shuffle_epi32(v15, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt15 = _mm_slli_epi32(vs15, 4);
        const __m128i vl15 = _mm_and_si128(vt15, vmask);
        const __m128i vh15 = _mm_and_si128(vs15, vmask);
        const __m128i v01_15 = _mm_unpacklo_epi8(vl15, vh15);
        const __m128i v23_15 = _mm_unpackhi_epi8(vl15, vh15);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_15, v01_15), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_15, v01_15), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc15 = _mm_add_epi32(vacc15, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_15, v23_15), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_15, v23_15), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc15 = _mm_add_epi32(vacc15, v_sum32);
        }

        const __m128i vl01_15 = _mm_srli_epi32(v01_15, 4);
        __m128i vout15 = _mm_or_si128(vl01_15, v23_15);

        const __m128i vout01_0 = _mm_unpacklo_epi64(vout0, vout1);
        const __m128i vout23_0 = _mm_unpackhi_epi64(vout0, vout1);
        _mm_storeu_si128((__m128i*) &out[0 * 8], vout01_0);
        _mm_storeu_si128((__m128i*) &out[(16 + 0) * 8], vout23_0);
        const __m128i vout01_2 = _mm_unpacklo_epi64(vout2, vout3);
        const __m128i vout23_2 = _mm_unpackhi_epi64(vout2, vout3);
        _mm_storeu_si128((__m128i*) &out[2 * 8], vout01_2);
        _mm_storeu_si128((__m128i*) &out[(16 + 2) * 8], vout23_2);
        const __m128i vout01_4 = _mm_unpacklo_epi64(vout4, vout5);
        const __m128i vout23_4 = _mm_unpackhi_epi64(vout4, vout5);
        _mm_storeu_si128((__m128i*) &out[4 * 8], vout01_4);
        _mm_storeu_si128((__m128i*) &out[(16 + 4) * 8], vout23_4);
        const __m128i vout01_6 = _mm_unpacklo_epi64(vout6, vout7);
        const __m128i vout23_6 = _mm_unpackhi_epi64(vout6, vout7);
        _mm_storeu_si128((__m128i*) &out[6 * 8], vout01_6);
        _mm_storeu_si128((__m128i*) &out[(16 + 6) * 8], vout23_6);
        const __m128i vout01_8 = _mm_unpacklo_epi64(vout8, vout9);
        const __m128i vout23_8 = _mm_unpackhi_epi64(vout8, vout9);
        _mm_storeu_si128((__m128i*) &out[8 * 8], vout01_8);
        _mm_storeu_si128((__m128i*) &out[(16 + 8) * 8], vout23_8);
        const __m128i vout01_10 = _mm_unpacklo_epi64(vout10, vout11);
        const __m128i vout23_10 = _mm_unpackhi_epi64(vout10, vout11);
        _mm_storeu_si128((__m128i*) &out[10 * 8], vout01_10);
        _mm_storeu_si128((__m128i*) &out[(16 + 10) * 8], vout23_10);
        const __m128i vout01_12 = _mm_unpacklo_epi64(vout12, vout13);
        const __m128i vout23_12 = _mm_unpackhi_epi64(vout12, vout13);
        _mm_storeu_si128((__m128i*) &out[12 * 8], vout01_12);
        _mm_storeu_si128((__m128i*) &out[(16 + 12) * 8], vout23_12);
        const __m128i vout01_14 = _mm_unpacklo_epi64(vout14, vout15);
        const __m128i vout23_14 = _mm_unpackhi_epi64(vout14, vout15);
        _mm_storeu_si128((__m128i*) &out[14 * 8], vout01_14);
        _mm_storeu_si128((__m128i*) &out[(16 + 14) * 8], vout23_14);

        out += 16 * 16;
      }

      // KC remainder of 1..16 bytes, including the odd KC byte
      if (k != 0 || kc_odd != 0) {
        assert(k + kc_odd >= 1 && k + kc_odd <= 16);
        uint8_t temp_w0[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w0[i] = w0[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w0[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w0[k] = (uint8_t) ((w0[k] & 0x0F) | (kernel_zero_point << 4));
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
        if (kc_odd) {
          temp_w1[k] = (uint8_t) ((w1[k] & 0x0F) | (kernel_zero_point << 4));
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
        if (kc_odd) {
          temp_w2[k] = (uint8_t) ((w2[k] & 0x0F) | (kernel_zero_point << 4));
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
        if (kc_odd) {
          temp_w3[k] = (uint8_t) ((w3[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v3 = _mm_loadu_si128((const __m128i*) temp_w3);
        if (kernel_zero_point != 0) {
          v3 = _mm_xor_si128(v3, vkernel_zero_point);
        }
        w3 += k;
        uint8_t temp_w4[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w4[i] = w4[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w4[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w4[k] = (uint8_t) ((w4[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v4 = _mm_loadu_si128((const __m128i*) temp_w4);
        if (kernel_zero_point != 0) {
          v4 = _mm_xor_si128(v4, vkernel_zero_point);
        }
        w4 += k;
        uint8_t temp_w5[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w5[i] = w5[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w5[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w5[k] = (uint8_t) ((w5[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v5 = _mm_loadu_si128((const __m128i*) temp_w5);
        if (kernel_zero_point != 0) {
          v5 = _mm_xor_si128(v5, vkernel_zero_point);
        }
        w5 += k;
        uint8_t temp_w6[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w6[i] = w6[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w6[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w6[k] = (uint8_t) ((w6[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v6 = _mm_loadu_si128((const __m128i*) temp_w6);
        if (kernel_zero_point != 0) {
          v6 = _mm_xor_si128(v6, vkernel_zero_point);
        }
        w6 += k;
        uint8_t temp_w7[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w7[i] = w7[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w7[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w7[k] = (uint8_t) ((w7[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v7 = _mm_loadu_si128((const __m128i*) temp_w7);
        if (kernel_zero_point != 0) {
          v7 = _mm_xor_si128(v7, vkernel_zero_point);
        }
        w7 += k;
        uint8_t temp_w8[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w8[i] = w8[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w8[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w8[k] = (uint8_t) ((w8[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v8 = _mm_loadu_si128((const __m128i*) temp_w8);
        if (kernel_zero_point != 0) {
          v8 = _mm_xor_si128(v8, vkernel_zero_point);
        }
        w8 += k;
        uint8_t temp_w9[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w9[i] = w9[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w9[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w9[k] = (uint8_t) ((w9[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v9 = _mm_loadu_si128((const __m128i*) temp_w9);
        if (kernel_zero_point != 0) {
          v9 = _mm_xor_si128(v9, vkernel_zero_point);
        }
        w9 += k;
        uint8_t temp_w10[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w10[i] = w10[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w10[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w10[k] = (uint8_t) ((w10[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v10 = _mm_loadu_si128((const __m128i*) temp_w10);
        if (kernel_zero_point != 0) {
          v10 = _mm_xor_si128(v10, vkernel_zero_point);
        }
        w10 += k;
        uint8_t temp_w11[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w11[i] = w11[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w11[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w11[k] = (uint8_t) ((w11[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v11 = _mm_loadu_si128((const __m128i*) temp_w11);
        if (kernel_zero_point != 0) {
          v11 = _mm_xor_si128(v11, vkernel_zero_point);
        }
        w11 += k;
        uint8_t temp_w12[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w12[i] = w12[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w12[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w12[k] = (uint8_t) ((w12[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v12 = _mm_loadu_si128((const __m128i*) temp_w12);
        if (kernel_zero_point != 0) {
          v12 = _mm_xor_si128(v12, vkernel_zero_point);
        }
        w12 += k;
        uint8_t temp_w13[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w13[i] = w13[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w13[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w13[k] = (uint8_t) ((w13[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v13 = _mm_loadu_si128((const __m128i*) temp_w13);
        if (kernel_zero_point != 0) {
          v13 = _mm_xor_si128(v13, vkernel_zero_point);
        }
        w13 += k;
        uint8_t temp_w14[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w14[i] = w14[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w14[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w14[k] = (uint8_t) ((w14[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v14 = _mm_loadu_si128((const __m128i*) temp_w14);
        if (kernel_zero_point != 0) {
          v14 = _mm_xor_si128(v14, vkernel_zero_point);
        }
        w14 += k;
        uint8_t temp_w15[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w15[i] = w15[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w15[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w15[k] = (uint8_t) ((w15[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v15 = _mm_loadu_si128((const __m128i*) temp_w15);
        if (kernel_zero_point != 0) {
          v15 = _mm_xor_si128(v15, vkernel_zero_point);
        }
        w15 += k;

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
        const __m128i vs4 = _mm_shuffle_epi32(v4, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt4 = _mm_slli_epi32(vs4, 4);
        const __m128i vl4 = _mm_and_si128(vt4, vmask);
        const __m128i vh4 = _mm_and_si128(vs4, vmask);
        const __m128i v01_4 = _mm_unpacklo_epi8(vl4, vh4);
        const __m128i v23_4 = _mm_unpackhi_epi8(vl4, vh4);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_4, v01_4), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_4, v01_4), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc4 = _mm_add_epi32(vacc4, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_4, v23_4), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_4, v23_4), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc4 = _mm_add_epi32(vacc4, v_sum32);
        }

        const __m128i vl01_4 = _mm_srli_epi32(v01_4, 4);
        __m128i vout4 = _mm_or_si128(vl01_4, v23_4);
        const __m128i vs5 = _mm_shuffle_epi32(v5, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt5 = _mm_slli_epi32(vs5, 4);
        const __m128i vl5 = _mm_and_si128(vt5, vmask);
        const __m128i vh5 = _mm_and_si128(vs5, vmask);
        const __m128i v01_5 = _mm_unpacklo_epi8(vl5, vh5);
        const __m128i v23_5 = _mm_unpackhi_epi8(vl5, vh5);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_5, v01_5), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_5, v01_5), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc5 = _mm_add_epi32(vacc5, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_5, v23_5), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_5, v23_5), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc5 = _mm_add_epi32(vacc5, v_sum32);
        }

        const __m128i vl01_5 = _mm_srli_epi32(v01_5, 4);
        __m128i vout5 = _mm_or_si128(vl01_5, v23_5);
        const __m128i vs6 = _mm_shuffle_epi32(v6, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt6 = _mm_slli_epi32(vs6, 4);
        const __m128i vl6 = _mm_and_si128(vt6, vmask);
        const __m128i vh6 = _mm_and_si128(vs6, vmask);
        const __m128i v01_6 = _mm_unpacklo_epi8(vl6, vh6);
        const __m128i v23_6 = _mm_unpackhi_epi8(vl6, vh6);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_6, v01_6), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_6, v01_6), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc6 = _mm_add_epi32(vacc6, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_6, v23_6), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_6, v23_6), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc6 = _mm_add_epi32(vacc6, v_sum32);
        }

        const __m128i vl01_6 = _mm_srli_epi32(v01_6, 4);
        __m128i vout6 = _mm_or_si128(vl01_6, v23_6);
        const __m128i vs7 = _mm_shuffle_epi32(v7, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt7 = _mm_slli_epi32(vs7, 4);
        const __m128i vl7 = _mm_and_si128(vt7, vmask);
        const __m128i vh7 = _mm_and_si128(vs7, vmask);
        const __m128i v01_7 = _mm_unpacklo_epi8(vl7, vh7);
        const __m128i v23_7 = _mm_unpackhi_epi8(vl7, vh7);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_7, v01_7), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_7, v01_7), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc7 = _mm_add_epi32(vacc7, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_7, v23_7), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_7, v23_7), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc7 = _mm_add_epi32(vacc7, v_sum32);
        }

        const __m128i vl01_7 = _mm_srli_epi32(v01_7, 4);
        __m128i vout7 = _mm_or_si128(vl01_7, v23_7);
        const __m128i vs8 = _mm_shuffle_epi32(v8, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt8 = _mm_slli_epi32(vs8, 4);
        const __m128i vl8 = _mm_and_si128(vt8, vmask);
        const __m128i vh8 = _mm_and_si128(vs8, vmask);
        const __m128i v01_8 = _mm_unpacklo_epi8(vl8, vh8);
        const __m128i v23_8 = _mm_unpackhi_epi8(vl8, vh8);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_8, v01_8), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_8, v01_8), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc8 = _mm_add_epi32(vacc8, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_8, v23_8), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_8, v23_8), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc8 = _mm_add_epi32(vacc8, v_sum32);
        }

        const __m128i vl01_8 = _mm_srli_epi32(v01_8, 4);
        __m128i vout8 = _mm_or_si128(vl01_8, v23_8);
        const __m128i vs9 = _mm_shuffle_epi32(v9, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt9 = _mm_slli_epi32(vs9, 4);
        const __m128i vl9 = _mm_and_si128(vt9, vmask);
        const __m128i vh9 = _mm_and_si128(vs9, vmask);
        const __m128i v01_9 = _mm_unpacklo_epi8(vl9, vh9);
        const __m128i v23_9 = _mm_unpackhi_epi8(vl9, vh9);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_9, v01_9), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_9, v01_9), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc9 = _mm_add_epi32(vacc9, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_9, v23_9), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_9, v23_9), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc9 = _mm_add_epi32(vacc9, v_sum32);
        }

        const __m128i vl01_9 = _mm_srli_epi32(v01_9, 4);
        __m128i vout9 = _mm_or_si128(vl01_9, v23_9);
        const __m128i vs10 = _mm_shuffle_epi32(v10, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt10 = _mm_slli_epi32(vs10, 4);
        const __m128i vl10 = _mm_and_si128(vt10, vmask);
        const __m128i vh10 = _mm_and_si128(vs10, vmask);
        const __m128i v01_10 = _mm_unpacklo_epi8(vl10, vh10);
        const __m128i v23_10 = _mm_unpackhi_epi8(vl10, vh10);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_10, v01_10), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_10, v01_10), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc10 = _mm_add_epi32(vacc10, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_10, v23_10), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_10, v23_10), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc10 = _mm_add_epi32(vacc10, v_sum32);
        }

        const __m128i vl01_10 = _mm_srli_epi32(v01_10, 4);
        __m128i vout10 = _mm_or_si128(vl01_10, v23_10);
        const __m128i vs11 = _mm_shuffle_epi32(v11, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt11 = _mm_slli_epi32(vs11, 4);
        const __m128i vl11 = _mm_and_si128(vt11, vmask);
        const __m128i vh11 = _mm_and_si128(vs11, vmask);
        const __m128i v01_11 = _mm_unpacklo_epi8(vl11, vh11);
        const __m128i v23_11 = _mm_unpackhi_epi8(vl11, vh11);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_11, v01_11), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_11, v01_11), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc11 = _mm_add_epi32(vacc11, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_11, v23_11), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_11, v23_11), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc11 = _mm_add_epi32(vacc11, v_sum32);
        }

        const __m128i vl01_11 = _mm_srli_epi32(v01_11, 4);
        __m128i vout11 = _mm_or_si128(vl01_11, v23_11);
        const __m128i vs12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt12 = _mm_slli_epi32(vs12, 4);
        const __m128i vl12 = _mm_and_si128(vt12, vmask);
        const __m128i vh12 = _mm_and_si128(vs12, vmask);
        const __m128i v01_12 = _mm_unpacklo_epi8(vl12, vh12);
        const __m128i v23_12 = _mm_unpackhi_epi8(vl12, vh12);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_12, v01_12), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_12, v01_12), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc12 = _mm_add_epi32(vacc12, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_12, v23_12), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_12, v23_12), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc12 = _mm_add_epi32(vacc12, v_sum32);
        }

        const __m128i vl01_12 = _mm_srli_epi32(v01_12, 4);
        __m128i vout12 = _mm_or_si128(vl01_12, v23_12);
        const __m128i vs13 = _mm_shuffle_epi32(v13, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt13 = _mm_slli_epi32(vs13, 4);
        const __m128i vl13 = _mm_and_si128(vt13, vmask);
        const __m128i vh13 = _mm_and_si128(vs13, vmask);
        const __m128i v01_13 = _mm_unpacklo_epi8(vl13, vh13);
        const __m128i v23_13 = _mm_unpackhi_epi8(vl13, vh13);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_13, v01_13), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_13, v01_13), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc13 = _mm_add_epi32(vacc13, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_13, v23_13), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_13, v23_13), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc13 = _mm_add_epi32(vacc13, v_sum32);
        }

        const __m128i vl01_13 = _mm_srli_epi32(v01_13, 4);
        __m128i vout13 = _mm_or_si128(vl01_13, v23_13);
        const __m128i vs14 = _mm_shuffle_epi32(v14, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt14 = _mm_slli_epi32(vs14, 4);
        const __m128i vl14 = _mm_and_si128(vt14, vmask);
        const __m128i vh14 = _mm_and_si128(vs14, vmask);
        const __m128i v01_14 = _mm_unpacklo_epi8(vl14, vh14);
        const __m128i v23_14 = _mm_unpackhi_epi8(vl14, vh14);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_14, v01_14), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_14, v01_14), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc14 = _mm_add_epi32(vacc14, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_14, v23_14), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_14, v23_14), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc14 = _mm_add_epi32(vacc14, v_sum32);
        }

        const __m128i vl01_14 = _mm_srli_epi32(v01_14, 4);
        __m128i vout14 = _mm_or_si128(vl01_14, v23_14);
        const __m128i vs15 = _mm_shuffle_epi32(v15, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt15 = _mm_slli_epi32(vs15, 4);
        const __m128i vl15 = _mm_and_si128(vt15, vmask);
        const __m128i vh15 = _mm_and_si128(vs15, vmask);
        const __m128i v01_15 = _mm_unpacklo_epi8(vl15, vh15);
        const __m128i v23_15 = _mm_unpackhi_epi8(vl15, vh15);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_15, v01_15), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_15, v01_15), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc15 = _mm_add_epi32(vacc15, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_15, v23_15), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_15, v23_15), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc15 = _mm_add_epi32(vacc15, v_sum32);
        }

        const __m128i vl01_15 = _mm_srli_epi32(v01_15, 4);
        __m128i vout15 = _mm_or_si128(vl01_15, v23_15);

        const __m128i vout01_0 = _mm_unpacklo_epi64(vout0, vout1);
        _mm_storeu_si128((__m128i*) &out[0 * 8], vout01_0);
        const __m128i vout01_2 = _mm_unpacklo_epi64(vout2, vout3);
        _mm_storeu_si128((__m128i*) &out[2 * 8], vout01_2);
        const __m128i vout01_4 = _mm_unpacklo_epi64(vout4, vout5);
        _mm_storeu_si128((__m128i*) &out[4 * 8], vout01_4);
        const __m128i vout01_6 = _mm_unpacklo_epi64(vout6, vout7);
        _mm_storeu_si128((__m128i*) &out[6 * 8], vout01_6);
        const __m128i vout01_8 = _mm_unpacklo_epi64(vout8, vout9);
        _mm_storeu_si128((__m128i*) &out[8 * 8], vout01_8);
        const __m128i vout01_10 = _mm_unpacklo_epi64(vout10, vout11);
        _mm_storeu_si128((__m128i*) &out[10 * 8], vout01_10);
        const __m128i vout01_12 = _mm_unpacklo_epi64(vout12, vout13);
        _mm_storeu_si128((__m128i*) &out[12 * 8], vout01_12);
        const __m128i vout01_14 = _mm_unpacklo_epi64(vout14, vout15);
        _mm_storeu_si128((__m128i*) &out[14 * 8], vout01_14);

        if (k + kc_odd > 8) {
          const __m128i vout23_0 = _mm_unpackhi_epi64(vout0, vout1);
          _mm_storeu_si128((__m128i*) &out[(16 + 0) * 8], vout23_0);
          const __m128i vout23_2 = _mm_unpackhi_epi64(vout2, vout3);
          _mm_storeu_si128((__m128i*) &out[(16 + 2) * 8], vout23_2);
          const __m128i vout23_4 = _mm_unpackhi_epi64(vout4, vout5);
          _mm_storeu_si128((__m128i*) &out[(16 + 4) * 8], vout23_4);
          const __m128i vout23_6 = _mm_unpackhi_epi64(vout6, vout7);
          _mm_storeu_si128((__m128i*) &out[(16 + 6) * 8], vout23_6);
          const __m128i vout23_8 = _mm_unpackhi_epi64(vout8, vout9);
          _mm_storeu_si128((__m128i*) &out[(16 + 8) * 8], vout23_8);
          const __m128i vout23_10 = _mm_unpackhi_epi64(vout10, vout11);
          _mm_storeu_si128((__m128i*) &out[(16 + 10) * 8], vout23_10);
          const __m128i vout23_12 = _mm_unpackhi_epi64(vout12, vout13);
          _mm_storeu_si128((__m128i*) &out[(16 + 12) * 8], vout23_12);
          const __m128i vout23_14 = _mm_unpackhi_epi64(vout14, vout15);
          _mm_storeu_si128((__m128i*) &out[(16 + 14) * 8], vout23_14);
          out += 16 * 16;
        } else {
          out += 16 * 8;
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
      int32_t ksum4 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc4, _mm_srli_si128(vacc4, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum4 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum5 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc5, _mm_srli_si128(vacc5, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum5 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum6 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc6, _mm_srli_si128(vacc6, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum6 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum7 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc7, _mm_srli_si128(vacc7, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum7 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum8 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc8, _mm_srli_si128(vacc8, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum8 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum9 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc9, _mm_srli_si128(vacc9, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum9 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum10 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc10, _mm_srli_si128(vacc10, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum10 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum11 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc11, _mm_srli_si128(vacc11, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum11 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum12 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc12, _mm_srli_si128(vacc12, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum12 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum13 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc13, _mm_srli_si128(vacc13, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum13 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum14 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc14, _mm_srli_si128(vacc14, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum14 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum15 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc15, _mm_srli_si128(vacc15, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum15 = _mm_cvtsi128_si32(sum1);
      }

      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      packed_b[2] -= ksum2 * izp;
      packed_b[3] -= ksum3 * izp;
      packed_b[4] -= ksum4 * izp;
      packed_b[5] -= ksum5 * izp;
      packed_b[6] -= ksum6 * izp;
      packed_b[7] -= ksum7 * izp;
      packed_b[8] -= ksum8 * izp;
      packed_b[9] -= ksum9 * izp;
      packed_b[10] -= ksum10 * izp;
      packed_b[11] -= ksum11 * izp;
      packed_b[12] -= ksum12 * izp;
      packed_b[13] -= ksum13 * izp;
      packed_b[14] -= ksum14 * izp;
      packed_b[15] -= ksum15 * izp;
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w15 + (n_stride >> 1) - mock_kc;
    }

    // NC remainder (1..15)
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
      out += 16 * sizeof(int32_t);

      // Clamp weight pointers
      const uint8_t* w1 = w0 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint8_t* w2 = w1 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint8_t* w3 = w2 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint8_t* w4 = w3 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint8_t* w5 = w4 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint8_t* w6 = w5 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint8_t* w7 = w6 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint8_t* w8 = w7 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint8_t* w9 = w8 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint8_t* w10 = w9 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint8_t* w11 = w10 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint8_t* w12 = w11 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint8_t* w13 = w12 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint8_t* w14 = w13 + (n_stride >> 1);
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }

      __m128i vacc0 = _mm_setzero_si128();
      __m128i vacc1 = _mm_setzero_si128();
      __m128i vacc2 = _mm_setzero_si128();
      __m128i vacc3 = _mm_setzero_si128();
      __m128i vacc4 = _mm_setzero_si128();
      __m128i vacc5 = _mm_setzero_si128();
      __m128i vacc6 = _mm_setzero_si128();
      __m128i vacc7 = _mm_setzero_si128();
      __m128i vacc8 = _mm_setzero_si128();
      __m128i vacc9 = _mm_setzero_si128();
      __m128i vacc10 = _mm_setzero_si128();
      __m128i vacc11 = _mm_setzero_si128();
      __m128i vacc12 = _mm_setzero_si128();
      __m128i vacc13 = _mm_setzero_si128();
      __m128i vacc14 = _mm_setzero_si128();

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
        __m128i v4 = _mm_loadu_si128((const __m128i*) w4);
        if (kernel_zero_point != 0) {
          v4 = _mm_xor_si128(v4, vkernel_zero_point);
        }
        w4 += 16;
        __m128i v5 = _mm_loadu_si128((const __m128i*) w5);
        if (kernel_zero_point != 0) {
          v5 = _mm_xor_si128(v5, vkernel_zero_point);
        }
        w5 += 16;
        __m128i v6 = _mm_loadu_si128((const __m128i*) w6);
        if (kernel_zero_point != 0) {
          v6 = _mm_xor_si128(v6, vkernel_zero_point);
        }
        w6 += 16;
        __m128i v7 = _mm_loadu_si128((const __m128i*) w7);
        if (kernel_zero_point != 0) {
          v7 = _mm_xor_si128(v7, vkernel_zero_point);
        }
        w7 += 16;
        __m128i v8 = _mm_loadu_si128((const __m128i*) w8);
        if (kernel_zero_point != 0) {
          v8 = _mm_xor_si128(v8, vkernel_zero_point);
        }
        w8 += 16;
        __m128i v9 = _mm_loadu_si128((const __m128i*) w9);
        if (kernel_zero_point != 0) {
          v9 = _mm_xor_si128(v9, vkernel_zero_point);
        }
        w9 += 16;
        __m128i v10 = _mm_loadu_si128((const __m128i*) w10);
        if (kernel_zero_point != 0) {
          v10 = _mm_xor_si128(v10, vkernel_zero_point);
        }
        w10 += 16;
        __m128i v11 = _mm_loadu_si128((const __m128i*) w11);
        if (kernel_zero_point != 0) {
          v11 = _mm_xor_si128(v11, vkernel_zero_point);
        }
        w11 += 16;
        __m128i v12 = _mm_loadu_si128((const __m128i*) w12);
        if (kernel_zero_point != 0) {
          v12 = _mm_xor_si128(v12, vkernel_zero_point);
        }
        w12 += 16;
        __m128i v13 = _mm_loadu_si128((const __m128i*) w13);
        if (kernel_zero_point != 0) {
          v13 = _mm_xor_si128(v13, vkernel_zero_point);
        }
        w13 += 16;
        __m128i v14 = _mm_loadu_si128((const __m128i*) w14);
        if (kernel_zero_point != 0) {
          v14 = _mm_xor_si128(v14, vkernel_zero_point);
        }
        w14 += 16;

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
        const __m128i vs4 = _mm_shuffle_epi32(v4, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt4 = _mm_slli_epi32(vs4, 4);
        const __m128i vl4 = _mm_and_si128(vt4, vmask);
        const __m128i vh4 = _mm_and_si128(vs4, vmask);
        const __m128i v01_4 = _mm_unpacklo_epi8(vl4, vh4);
        const __m128i v23_4 = _mm_unpackhi_epi8(vl4, vh4);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_4, v01_4), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_4, v01_4), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc4 = _mm_add_epi32(vacc4, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_4, v23_4), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_4, v23_4), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc4 = _mm_add_epi32(vacc4, v_sum32);
        }

        const __m128i vl01_4 = _mm_srli_epi32(v01_4, 4);
        __m128i vout4 = _mm_or_si128(vl01_4, v23_4);
        const __m128i vs5 = _mm_shuffle_epi32(v5, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt5 = _mm_slli_epi32(vs5, 4);
        const __m128i vl5 = _mm_and_si128(vt5, vmask);
        const __m128i vh5 = _mm_and_si128(vs5, vmask);
        const __m128i v01_5 = _mm_unpacklo_epi8(vl5, vh5);
        const __m128i v23_5 = _mm_unpackhi_epi8(vl5, vh5);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_5, v01_5), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_5, v01_5), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc5 = _mm_add_epi32(vacc5, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_5, v23_5), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_5, v23_5), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc5 = _mm_add_epi32(vacc5, v_sum32);
        }

        const __m128i vl01_5 = _mm_srli_epi32(v01_5, 4);
        __m128i vout5 = _mm_or_si128(vl01_5, v23_5);
        const __m128i vs6 = _mm_shuffle_epi32(v6, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt6 = _mm_slli_epi32(vs6, 4);
        const __m128i vl6 = _mm_and_si128(vt6, vmask);
        const __m128i vh6 = _mm_and_si128(vs6, vmask);
        const __m128i v01_6 = _mm_unpacklo_epi8(vl6, vh6);
        const __m128i v23_6 = _mm_unpackhi_epi8(vl6, vh6);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_6, v01_6), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_6, v01_6), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc6 = _mm_add_epi32(vacc6, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_6, v23_6), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_6, v23_6), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc6 = _mm_add_epi32(vacc6, v_sum32);
        }

        const __m128i vl01_6 = _mm_srli_epi32(v01_6, 4);
        __m128i vout6 = _mm_or_si128(vl01_6, v23_6);
        const __m128i vs7 = _mm_shuffle_epi32(v7, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt7 = _mm_slli_epi32(vs7, 4);
        const __m128i vl7 = _mm_and_si128(vt7, vmask);
        const __m128i vh7 = _mm_and_si128(vs7, vmask);
        const __m128i v01_7 = _mm_unpacklo_epi8(vl7, vh7);
        const __m128i v23_7 = _mm_unpackhi_epi8(vl7, vh7);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_7, v01_7), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_7, v01_7), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc7 = _mm_add_epi32(vacc7, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_7, v23_7), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_7, v23_7), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc7 = _mm_add_epi32(vacc7, v_sum32);
        }

        const __m128i vl01_7 = _mm_srli_epi32(v01_7, 4);
        __m128i vout7 = _mm_or_si128(vl01_7, v23_7);
        const __m128i vs8 = _mm_shuffle_epi32(v8, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt8 = _mm_slli_epi32(vs8, 4);
        const __m128i vl8 = _mm_and_si128(vt8, vmask);
        const __m128i vh8 = _mm_and_si128(vs8, vmask);
        const __m128i v01_8 = _mm_unpacklo_epi8(vl8, vh8);
        const __m128i v23_8 = _mm_unpackhi_epi8(vl8, vh8);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_8, v01_8), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_8, v01_8), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc8 = _mm_add_epi32(vacc8, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_8, v23_8), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_8, v23_8), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc8 = _mm_add_epi32(vacc8, v_sum32);
        }

        const __m128i vl01_8 = _mm_srli_epi32(v01_8, 4);
        __m128i vout8 = _mm_or_si128(vl01_8, v23_8);
        const __m128i vs9 = _mm_shuffle_epi32(v9, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt9 = _mm_slli_epi32(vs9, 4);
        const __m128i vl9 = _mm_and_si128(vt9, vmask);
        const __m128i vh9 = _mm_and_si128(vs9, vmask);
        const __m128i v01_9 = _mm_unpacklo_epi8(vl9, vh9);
        const __m128i v23_9 = _mm_unpackhi_epi8(vl9, vh9);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_9, v01_9), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_9, v01_9), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc9 = _mm_add_epi32(vacc9, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_9, v23_9), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_9, v23_9), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc9 = _mm_add_epi32(vacc9, v_sum32);
        }

        const __m128i vl01_9 = _mm_srli_epi32(v01_9, 4);
        __m128i vout9 = _mm_or_si128(vl01_9, v23_9);
        const __m128i vs10 = _mm_shuffle_epi32(v10, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt10 = _mm_slli_epi32(vs10, 4);
        const __m128i vl10 = _mm_and_si128(vt10, vmask);
        const __m128i vh10 = _mm_and_si128(vs10, vmask);
        const __m128i v01_10 = _mm_unpacklo_epi8(vl10, vh10);
        const __m128i v23_10 = _mm_unpackhi_epi8(vl10, vh10);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_10, v01_10), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_10, v01_10), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc10 = _mm_add_epi32(vacc10, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_10, v23_10), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_10, v23_10), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc10 = _mm_add_epi32(vacc10, v_sum32);
        }

        const __m128i vl01_10 = _mm_srli_epi32(v01_10, 4);
        __m128i vout10 = _mm_or_si128(vl01_10, v23_10);
        const __m128i vs11 = _mm_shuffle_epi32(v11, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt11 = _mm_slli_epi32(vs11, 4);
        const __m128i vl11 = _mm_and_si128(vt11, vmask);
        const __m128i vh11 = _mm_and_si128(vs11, vmask);
        const __m128i v01_11 = _mm_unpacklo_epi8(vl11, vh11);
        const __m128i v23_11 = _mm_unpackhi_epi8(vl11, vh11);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_11, v01_11), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_11, v01_11), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc11 = _mm_add_epi32(vacc11, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_11, v23_11), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_11, v23_11), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc11 = _mm_add_epi32(vacc11, v_sum32);
        }

        const __m128i vl01_11 = _mm_srli_epi32(v01_11, 4);
        __m128i vout11 = _mm_or_si128(vl01_11, v23_11);
        const __m128i vs12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt12 = _mm_slli_epi32(vs12, 4);
        const __m128i vl12 = _mm_and_si128(vt12, vmask);
        const __m128i vh12 = _mm_and_si128(vs12, vmask);
        const __m128i v01_12 = _mm_unpacklo_epi8(vl12, vh12);
        const __m128i v23_12 = _mm_unpackhi_epi8(vl12, vh12);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_12, v01_12), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_12, v01_12), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc12 = _mm_add_epi32(vacc12, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_12, v23_12), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_12, v23_12), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc12 = _mm_add_epi32(vacc12, v_sum32);
        }

        const __m128i vl01_12 = _mm_srli_epi32(v01_12, 4);
        __m128i vout12 = _mm_or_si128(vl01_12, v23_12);
        const __m128i vs13 = _mm_shuffle_epi32(v13, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt13 = _mm_slli_epi32(vs13, 4);
        const __m128i vl13 = _mm_and_si128(vt13, vmask);
        const __m128i vh13 = _mm_and_si128(vs13, vmask);
        const __m128i v01_13 = _mm_unpacklo_epi8(vl13, vh13);
        const __m128i v23_13 = _mm_unpackhi_epi8(vl13, vh13);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_13, v01_13), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_13, v01_13), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc13 = _mm_add_epi32(vacc13, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_13, v23_13), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_13, v23_13), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc13 = _mm_add_epi32(vacc13, v_sum32);
        }

        const __m128i vl01_13 = _mm_srli_epi32(v01_13, 4);
        __m128i vout13 = _mm_or_si128(vl01_13, v23_13);
        const __m128i vs14 = _mm_shuffle_epi32(v14, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt14 = _mm_slli_epi32(vs14, 4);
        const __m128i vl14 = _mm_and_si128(vt14, vmask);
        const __m128i vh14 = _mm_and_si128(vs14, vmask);
        const __m128i v01_14 = _mm_unpacklo_epi8(vl14, vh14);
        const __m128i v23_14 = _mm_unpackhi_epi8(vl14, vh14);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_14, v01_14), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_14, v01_14), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc14 = _mm_add_epi32(vacc14, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_14, v23_14), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_14, v23_14), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc14 = _mm_add_epi32(vacc14, v_sum32);
        }

        const __m128i vl01_14 = _mm_srli_epi32(v01_14, 4);
        __m128i vout14 = _mm_or_si128(vl01_14, v23_14);

        const __m128i vout15 = _mm_setzero_si128();

        const __m128i vout01_0 = _mm_unpacklo_epi64(vout0, vout1);
        const __m128i vout23_0 = _mm_unpackhi_epi64(vout0, vout1);
        _mm_storeu_si128((__m128i*) &out[0 * 8], vout01_0);
        _mm_storeu_si128((__m128i*) &out[(16 + 0) * 8], vout23_0);
        const __m128i vout01_2 = _mm_unpacklo_epi64(vout2, vout3);
        const __m128i vout23_2 = _mm_unpackhi_epi64(vout2, vout3);
        _mm_storeu_si128((__m128i*) &out[2 * 8], vout01_2);
        _mm_storeu_si128((__m128i*) &out[(16 + 2) * 8], vout23_2);
        const __m128i vout01_4 = _mm_unpacklo_epi64(vout4, vout5);
        const __m128i vout23_4 = _mm_unpackhi_epi64(vout4, vout5);
        _mm_storeu_si128((__m128i*) &out[4 * 8], vout01_4);
        _mm_storeu_si128((__m128i*) &out[(16 + 4) * 8], vout23_4);
        const __m128i vout01_6 = _mm_unpacklo_epi64(vout6, vout7);
        const __m128i vout23_6 = _mm_unpackhi_epi64(vout6, vout7);
        _mm_storeu_si128((__m128i*) &out[6 * 8], vout01_6);
        _mm_storeu_si128((__m128i*) &out[(16 + 6) * 8], vout23_6);
        const __m128i vout01_8 = _mm_unpacklo_epi64(vout8, vout9);
        const __m128i vout23_8 = _mm_unpackhi_epi64(vout8, vout9);
        _mm_storeu_si128((__m128i*) &out[8 * 8], vout01_8);
        _mm_storeu_si128((__m128i*) &out[(16 + 8) * 8], vout23_8);
        const __m128i vout01_10 = _mm_unpacklo_epi64(vout10, vout11);
        const __m128i vout23_10 = _mm_unpackhi_epi64(vout10, vout11);
        _mm_storeu_si128((__m128i*) &out[10 * 8], vout01_10);
        _mm_storeu_si128((__m128i*) &out[(16 + 10) * 8], vout23_10);
        const __m128i vout01_12 = _mm_unpacklo_epi64(vout12, vout13);
        const __m128i vout23_12 = _mm_unpackhi_epi64(vout12, vout13);
        _mm_storeu_si128((__m128i*) &out[12 * 8], vout01_12);
        _mm_storeu_si128((__m128i*) &out[(16 + 12) * 8], vout23_12);
        const __m128i vout01_14 = _mm_unpacklo_epi64(vout14, vout15);
        const __m128i vout23_14 = _mm_unpackhi_epi64(vout14, vout15);
        _mm_storeu_si128((__m128i*) &out[14 * 8], vout01_14);
        _mm_storeu_si128((__m128i*) &out[(16 + 14) * 8], vout23_14);

        out += 16 * 16;
      }

      if (k != 0 || kc_odd != 0) {
        assert(k + kc_odd >= 1 && k + kc_odd <= 16);
        uint8_t temp_w0[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w0[i] = w0[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w0[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w0[k] = (uint8_t) ((w0[k] & 0x0F) | (kernel_zero_point << 4));
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
        if (kc_odd) {
          temp_w1[k] = (uint8_t) ((w1[k] & 0x0F) | (kernel_zero_point << 4));
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
        if (kc_odd) {
          temp_w2[k] = (uint8_t) ((w2[k] & 0x0F) | (kernel_zero_point << 4));
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
        if (kc_odd) {
          temp_w3[k] = (uint8_t) ((w3[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v3 = _mm_loadu_si128((const __m128i*) temp_w3);
        if (kernel_zero_point != 0) {
          v3 = _mm_xor_si128(v3, vkernel_zero_point);
        }
        w3 += k;
        uint8_t temp_w4[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w4[i] = w4[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w4[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w4[k] = (uint8_t) ((w4[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v4 = _mm_loadu_si128((const __m128i*) temp_w4);
        if (kernel_zero_point != 0) {
          v4 = _mm_xor_si128(v4, vkernel_zero_point);
        }
        w4 += k;
        uint8_t temp_w5[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w5[i] = w5[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w5[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w5[k] = (uint8_t) ((w5[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v5 = _mm_loadu_si128((const __m128i*) temp_w5);
        if (kernel_zero_point != 0) {
          v5 = _mm_xor_si128(v5, vkernel_zero_point);
        }
        w5 += k;
        uint8_t temp_w6[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w6[i] = w6[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w6[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w6[k] = (uint8_t) ((w6[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v6 = _mm_loadu_si128((const __m128i*) temp_w6);
        if (kernel_zero_point != 0) {
          v6 = _mm_xor_si128(v6, vkernel_zero_point);
        }
        w6 += k;
        uint8_t temp_w7[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w7[i] = w7[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w7[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w7[k] = (uint8_t) ((w7[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v7 = _mm_loadu_si128((const __m128i*) temp_w7);
        if (kernel_zero_point != 0) {
          v7 = _mm_xor_si128(v7, vkernel_zero_point);
        }
        w7 += k;
        uint8_t temp_w8[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w8[i] = w8[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w8[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w8[k] = (uint8_t) ((w8[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v8 = _mm_loadu_si128((const __m128i*) temp_w8);
        if (kernel_zero_point != 0) {
          v8 = _mm_xor_si128(v8, vkernel_zero_point);
        }
        w8 += k;
        uint8_t temp_w9[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w9[i] = w9[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w9[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w9[k] = (uint8_t) ((w9[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v9 = _mm_loadu_si128((const __m128i*) temp_w9);
        if (kernel_zero_point != 0) {
          v9 = _mm_xor_si128(v9, vkernel_zero_point);
        }
        w9 += k;
        uint8_t temp_w10[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w10[i] = w10[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w10[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w10[k] = (uint8_t) ((w10[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v10 = _mm_loadu_si128((const __m128i*) temp_w10);
        if (kernel_zero_point != 0) {
          v10 = _mm_xor_si128(v10, vkernel_zero_point);
        }
        w10 += k;
        uint8_t temp_w11[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w11[i] = w11[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w11[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w11[k] = (uint8_t) ((w11[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v11 = _mm_loadu_si128((const __m128i*) temp_w11);
        if (kernel_zero_point != 0) {
          v11 = _mm_xor_si128(v11, vkernel_zero_point);
        }
        w11 += k;
        uint8_t temp_w12[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w12[i] = w12[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w12[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w12[k] = (uint8_t) ((w12[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v12 = _mm_loadu_si128((const __m128i*) temp_w12);
        if (kernel_zero_point != 0) {
          v12 = _mm_xor_si128(v12, vkernel_zero_point);
        }
        w12 += k;
        uint8_t temp_w13[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w13[i] = w13[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w13[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w13[k] = (uint8_t) ((w13[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v13 = _mm_loadu_si128((const __m128i*) temp_w13);
        if (kernel_zero_point != 0) {
          v13 = _mm_xor_si128(v13, vkernel_zero_point);
        }
        w13 += k;
        uint8_t temp_w14[16];
        for (size_t i = 0; i < k; ++i) {
          temp_w14[i] = w14[i];
        }
        for (size_t i = k; i < 16; ++i) {
          temp_w14[i] = kernel_zero_point * 0x11;
        }
        if (kc_odd) {
          temp_w14[k] = (uint8_t) ((w14[k] & 0x0F) | (kernel_zero_point << 4));
        }
        __m128i v14 = _mm_loadu_si128((const __m128i*) temp_w14);
        if (kernel_zero_point != 0) {
          v14 = _mm_xor_si128(v14, vkernel_zero_point);
        }
        w14 += k;

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
        const __m128i vs4 = _mm_shuffle_epi32(v4, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt4 = _mm_slli_epi32(vs4, 4);
        const __m128i vl4 = _mm_and_si128(vt4, vmask);
        const __m128i vh4 = _mm_and_si128(vs4, vmask);
        const __m128i v01_4 = _mm_unpacklo_epi8(vl4, vh4);
        const __m128i v23_4 = _mm_unpackhi_epi8(vl4, vh4);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_4, v01_4), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_4, v01_4), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc4 = _mm_add_epi32(vacc4, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_4, v23_4), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_4, v23_4), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc4 = _mm_add_epi32(vacc4, v_sum32);
        }

        const __m128i vl01_4 = _mm_srli_epi32(v01_4, 4);
        __m128i vout4 = _mm_or_si128(vl01_4, v23_4);
        const __m128i vs5 = _mm_shuffle_epi32(v5, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt5 = _mm_slli_epi32(vs5, 4);
        const __m128i vl5 = _mm_and_si128(vt5, vmask);
        const __m128i vh5 = _mm_and_si128(vs5, vmask);
        const __m128i v01_5 = _mm_unpacklo_epi8(vl5, vh5);
        const __m128i v23_5 = _mm_unpackhi_epi8(vl5, vh5);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_5, v01_5), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_5, v01_5), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc5 = _mm_add_epi32(vacc5, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_5, v23_5), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_5, v23_5), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc5 = _mm_add_epi32(vacc5, v_sum32);
        }

        const __m128i vl01_5 = _mm_srli_epi32(v01_5, 4);
        __m128i vout5 = _mm_or_si128(vl01_5, v23_5);
        const __m128i vs6 = _mm_shuffle_epi32(v6, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt6 = _mm_slli_epi32(vs6, 4);
        const __m128i vl6 = _mm_and_si128(vt6, vmask);
        const __m128i vh6 = _mm_and_si128(vs6, vmask);
        const __m128i v01_6 = _mm_unpacklo_epi8(vl6, vh6);
        const __m128i v23_6 = _mm_unpackhi_epi8(vl6, vh6);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_6, v01_6), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_6, v01_6), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc6 = _mm_add_epi32(vacc6, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_6, v23_6), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_6, v23_6), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc6 = _mm_add_epi32(vacc6, v_sum32);
        }

        const __m128i vl01_6 = _mm_srli_epi32(v01_6, 4);
        __m128i vout6 = _mm_or_si128(vl01_6, v23_6);
        const __m128i vs7 = _mm_shuffle_epi32(v7, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt7 = _mm_slli_epi32(vs7, 4);
        const __m128i vl7 = _mm_and_si128(vt7, vmask);
        const __m128i vh7 = _mm_and_si128(vs7, vmask);
        const __m128i v01_7 = _mm_unpacklo_epi8(vl7, vh7);
        const __m128i v23_7 = _mm_unpackhi_epi8(vl7, vh7);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_7, v01_7), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_7, v01_7), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc7 = _mm_add_epi32(vacc7, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_7, v23_7), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_7, v23_7), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc7 = _mm_add_epi32(vacc7, v_sum32);
        }

        const __m128i vl01_7 = _mm_srli_epi32(v01_7, 4);
        __m128i vout7 = _mm_or_si128(vl01_7, v23_7);
        const __m128i vs8 = _mm_shuffle_epi32(v8, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt8 = _mm_slli_epi32(vs8, 4);
        const __m128i vl8 = _mm_and_si128(vt8, vmask);
        const __m128i vh8 = _mm_and_si128(vs8, vmask);
        const __m128i v01_8 = _mm_unpacklo_epi8(vl8, vh8);
        const __m128i v23_8 = _mm_unpackhi_epi8(vl8, vh8);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_8, v01_8), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_8, v01_8), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc8 = _mm_add_epi32(vacc8, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_8, v23_8), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_8, v23_8), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc8 = _mm_add_epi32(vacc8, v_sum32);
        }

        const __m128i vl01_8 = _mm_srli_epi32(v01_8, 4);
        __m128i vout8 = _mm_or_si128(vl01_8, v23_8);
        const __m128i vs9 = _mm_shuffle_epi32(v9, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt9 = _mm_slli_epi32(vs9, 4);
        const __m128i vl9 = _mm_and_si128(vt9, vmask);
        const __m128i vh9 = _mm_and_si128(vs9, vmask);
        const __m128i v01_9 = _mm_unpacklo_epi8(vl9, vh9);
        const __m128i v23_9 = _mm_unpackhi_epi8(vl9, vh9);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_9, v01_9), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_9, v01_9), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc9 = _mm_add_epi32(vacc9, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_9, v23_9), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_9, v23_9), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc9 = _mm_add_epi32(vacc9, v_sum32);
        }

        const __m128i vl01_9 = _mm_srli_epi32(v01_9, 4);
        __m128i vout9 = _mm_or_si128(vl01_9, v23_9);
        const __m128i vs10 = _mm_shuffle_epi32(v10, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt10 = _mm_slli_epi32(vs10, 4);
        const __m128i vl10 = _mm_and_si128(vt10, vmask);
        const __m128i vh10 = _mm_and_si128(vs10, vmask);
        const __m128i v01_10 = _mm_unpacklo_epi8(vl10, vh10);
        const __m128i v23_10 = _mm_unpackhi_epi8(vl10, vh10);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_10, v01_10), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_10, v01_10), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc10 = _mm_add_epi32(vacc10, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_10, v23_10), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_10, v23_10), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc10 = _mm_add_epi32(vacc10, v_sum32);
        }

        const __m128i vl01_10 = _mm_srli_epi32(v01_10, 4);
        __m128i vout10 = _mm_or_si128(vl01_10, v23_10);
        const __m128i vs11 = _mm_shuffle_epi32(v11, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt11 = _mm_slli_epi32(vs11, 4);
        const __m128i vl11 = _mm_and_si128(vt11, vmask);
        const __m128i vh11 = _mm_and_si128(vs11, vmask);
        const __m128i v01_11 = _mm_unpacklo_epi8(vl11, vh11);
        const __m128i v23_11 = _mm_unpackhi_epi8(vl11, vh11);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_11, v01_11), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_11, v01_11), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc11 = _mm_add_epi32(vacc11, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_11, v23_11), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_11, v23_11), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc11 = _mm_add_epi32(vacc11, v_sum32);
        }

        const __m128i vl01_11 = _mm_srli_epi32(v01_11, 4);
        __m128i vout11 = _mm_or_si128(vl01_11, v23_11);
        const __m128i vs12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt12 = _mm_slli_epi32(vs12, 4);
        const __m128i vl12 = _mm_and_si128(vt12, vmask);
        const __m128i vh12 = _mm_and_si128(vs12, vmask);
        const __m128i v01_12 = _mm_unpacklo_epi8(vl12, vh12);
        const __m128i v23_12 = _mm_unpackhi_epi8(vl12, vh12);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_12, v01_12), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_12, v01_12), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc12 = _mm_add_epi32(vacc12, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_12, v23_12), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_12, v23_12), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc12 = _mm_add_epi32(vacc12, v_sum32);
        }

        const __m128i vl01_12 = _mm_srli_epi32(v01_12, 4);
        __m128i vout12 = _mm_or_si128(vl01_12, v23_12);
        const __m128i vs13 = _mm_shuffle_epi32(v13, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt13 = _mm_slli_epi32(vs13, 4);
        const __m128i vl13 = _mm_and_si128(vt13, vmask);
        const __m128i vh13 = _mm_and_si128(vs13, vmask);
        const __m128i v01_13 = _mm_unpacklo_epi8(vl13, vh13);
        const __m128i v23_13 = _mm_unpackhi_epi8(vl13, vh13);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_13, v01_13), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_13, v01_13), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc13 = _mm_add_epi32(vacc13, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_13, v23_13), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_13, v23_13), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc13 = _mm_add_epi32(vacc13, v_sum32);
        }

        const __m128i vl01_13 = _mm_srli_epi32(v01_13, 4);
        __m128i vout13 = _mm_or_si128(vl01_13, v23_13);
        const __m128i vs14 = _mm_shuffle_epi32(v14, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vt14 = _mm_slli_epi32(vs14, 4);
        const __m128i vl14 = _mm_and_si128(vt14, vmask);
        const __m128i vh14 = _mm_and_si128(vs14, vmask);
        const __m128i v01_14 = _mm_unpacklo_epi8(vl14, vh14);
        const __m128i v23_14 = _mm_unpackhi_epi8(vl14, vh14);

        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v01_14, v01_14), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v01_14, v01_14), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc14 = _mm_add_epi32(vacc14, v_sum32);
        }
        {
          const __m128i v_lo = _mm_srai_epi16(_mm_unpacklo_epi8(v23_14, v23_14), 8);
          const __m128i v_hi = _mm_srai_epi16(_mm_unpackhi_epi8(v23_14, v23_14), 8);
          const __m128i v_sum16 = _mm_add_epi16(v_lo, v_hi);
          const __m128i v_sum32 = _mm_madd_epi16(v_sum16, _mm_set1_epi16(1));
          vacc14 = _mm_add_epi32(vacc14, v_sum32);
        }

        const __m128i vl01_14 = _mm_srli_epi32(v01_14, 4);
        __m128i vout14 = _mm_or_si128(vl01_14, v23_14);

        const __m128i vout15 = _mm_setzero_si128();

        const __m128i vout01_0 = _mm_unpacklo_epi64(vout0, vout1);
        _mm_storeu_si128((__m128i*) &out[0 * 8], vout01_0);
        const __m128i vout01_2 = _mm_unpacklo_epi64(vout2, vout3);
        _mm_storeu_si128((__m128i*) &out[2 * 8], vout01_2);
        const __m128i vout01_4 = _mm_unpacklo_epi64(vout4, vout5);
        _mm_storeu_si128((__m128i*) &out[4 * 8], vout01_4);
        const __m128i vout01_6 = _mm_unpacklo_epi64(vout6, vout7);
        _mm_storeu_si128((__m128i*) &out[6 * 8], vout01_6);
        const __m128i vout01_8 = _mm_unpacklo_epi64(vout8, vout9);
        _mm_storeu_si128((__m128i*) &out[8 * 8], vout01_8);
        const __m128i vout01_10 = _mm_unpacklo_epi64(vout10, vout11);
        _mm_storeu_si128((__m128i*) &out[10 * 8], vout01_10);
        const __m128i vout01_12 = _mm_unpacklo_epi64(vout12, vout13);
        _mm_storeu_si128((__m128i*) &out[12 * 8], vout01_12);
        const __m128i vout01_14 = _mm_unpacklo_epi64(vout14, vout15);
        _mm_storeu_si128((__m128i*) &out[14 * 8], vout01_14);

        if (k + kc_odd > 8) {
          const __m128i vout23_0 = _mm_unpackhi_epi64(vout0, vout1);
          _mm_storeu_si128((__m128i*) &out[(16 + 0) * 8], vout23_0);
          const __m128i vout23_2 = _mm_unpackhi_epi64(vout2, vout3);
          _mm_storeu_si128((__m128i*) &out[(16 + 2) * 8], vout23_2);
          const __m128i vout23_4 = _mm_unpackhi_epi64(vout4, vout5);
          _mm_storeu_si128((__m128i*) &out[(16 + 4) * 8], vout23_4);
          const __m128i vout23_6 = _mm_unpackhi_epi64(vout6, vout7);
          _mm_storeu_si128((__m128i*) &out[(16 + 6) * 8], vout23_6);
          const __m128i vout23_8 = _mm_unpackhi_epi64(vout8, vout9);
          _mm_storeu_si128((__m128i*) &out[(16 + 8) * 8], vout23_8);
          const __m128i vout23_10 = _mm_unpackhi_epi64(vout10, vout11);
          _mm_storeu_si128((__m128i*) &out[(16 + 10) * 8], vout23_10);
          const __m128i vout23_12 = _mm_unpackhi_epi64(vout12, vout13);
          _mm_storeu_si128((__m128i*) &out[(16 + 12) * 8], vout23_12);
          const __m128i vout23_14 = _mm_unpackhi_epi64(vout14, vout15);
          _mm_storeu_si128((__m128i*) &out[(16 + 14) * 8], vout23_14);
          out += 16 * 16;
        } else {
          out += 16 * 8;
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
      int32_t ksum4 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc4, _mm_srli_si128(vacc4, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum4 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum5 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc5, _mm_srli_si128(vacc5, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum5 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum6 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc6, _mm_srli_si128(vacc6, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum6 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum7 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc7, _mm_srli_si128(vacc7, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum7 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum8 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc8, _mm_srli_si128(vacc8, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum8 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum9 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc9, _mm_srli_si128(vacc9, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum9 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum10 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc10, _mm_srli_si128(vacc10, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum10 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum11 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc11, _mm_srli_si128(vacc11, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum11 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum12 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc12, _mm_srli_si128(vacc12, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum12 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum13 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc13, _mm_srli_si128(vacc13, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum13 = _mm_cvtsi128_si32(sum1);
      }
      int32_t ksum14 = 0;
      {
        __m128i sum2 = _mm_add_epi32(vacc14, _mm_srli_si128(vacc14, 8));
        __m128i sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
        ksum14 = _mm_cvtsi128_si32(sum1);
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
      if (3 < n) {
        packed_b[3] -= ksum3 * izp;
      }
      if (4 < n) {
        packed_b[4] -= ksum4 * izp;
      }
      if (5 < n) {
        packed_b[5] -= ksum5 * izp;
      }
      if (6 < n) {
        packed_b[6] -= ksum6 * izp;
      }
      if (7 < n) {
        packed_b[7] -= ksum7 * izp;
      }
      if (8 < n) {
        packed_b[8] -= ksum8 * izp;
      }
      if (9 < n) {
        packed_b[9] -= ksum9 * izp;
      }
      if (10 < n) {
        packed_b[10] -= ksum10 * izp;
      }
      if (11 < n) {
        packed_b[11] -= ksum11 * izp;
      }
      if (12 < n) {
        packed_b[12] -= ksum12 * izp;
      }
      if (13 < n) {
        packed_b[13] -= ksum13 * izp;
      }
      if (14 < n) {
        packed_b[14] -= ksum14 * izp;
      }
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights = (const uint8_t*)((intptr_t) weights + nc * (n_stride >> 1));
  } while (--g != 0);
}
