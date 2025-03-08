// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-avx512amx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#if defined(__has_feature)
  #if __has_feature(memory_sanitizer)
    #include <sanitizer/msan_interface.h>
  #endif
#endif

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/unaligned.h"


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x32c4__avx512amx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

// AMX is only available for __x86_64__
#if XNN_ARCH_X86_64

  // Define tile config data structure
  struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[8];
    uint16_t reserved_1[8];
    uint8_t rows[8];
    uint8_t reserved_2[8];
  };

  XNN_ALIGN(64) struct __tile_config tile_data = {0};
  XNN_ALIGN(64) int32_t res[2][1 * 16];

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const size_t kremainder = (kc & 63) ? (kc & 63) : 64;

  tile_data.palette_id = 1;
  tile_data.rows[0] = mr;              // tmm0 = res[0]
  tile_data.rows[1] = mr;              // tmm1 = res[1]
  tile_data.rows[2] = mr;              // tmm2 = res[2]
  tile_data.rows[3] = mr;              // tmm3 = res[3]
  tile_data.rows[4] = mr;              // tmm4 = input
  tile_data.rows[5] = 16;              // tmm5 = weights
  tile_data.rows[6] = mr;              // tmm6 = input remainder
  tile_data.rows[7] = kremainder >> 2; // tmm7 = weights remainder
  tile_data.colsb[0] = 64;          // tmm0 = res[0]
  tile_data.colsb[1] = 64;          // tmm1 = res[1]
  tile_data.colsb[2] = 64;          // tmm2 = res[2]
  tile_data.colsb[3] = 64;          // tmm3 = res[3]
  tile_data.colsb[4] = 64;          // tmm4 = input
  tile_data.colsb[5] = 64;          // tmm5 = weights
  tile_data.colsb[6] = kremainder;  // tmm6 = input remainder
  tile_data.colsb[7] = 64;          // tmm7 = weights remainder
  _tile_loadconfig(&tile_data);

  int8_t* c0 = c;

  const __m512 voutput_max_less_zero_point = _mm512_set1_ps((int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point);
  const __m512i voutput_zero_point = _mm512_set1_epi32(params->fp32_scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->fp32_scalar.output_min);
  // XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  // XNN_FORCE_REALIZATION(voutput_zero_point);
  // XNN_FORCE_REALIZATION(voutput_min);

  do {
    const __m512i vksum0123456789ABCDEF = _mm512_load_epi32((const int32_t*) w + 0);
    const __m512i vksumGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int32_t*) w + 16);
    w = (const int32_t*) w + 32;

    // Zero tile accumulator
    _tile_zero(0);
    _tile_zero(1);

    size_t k = kc;
    while (k >= 64 * sizeof(int8_t)) {
      _tile_loadd(4, a, a_stride);
      a += 64;
      _tile_loadd(5, (const int8_t*) w + 0, 128);
      _tile_dpbssd(0, 4, 5);
      _tile_loadd(5, (const int8_t*) w + 64, 128);
      _tile_dpbssd(1, 4, 5);

      w = (const int8_t*) w + 2048;
      k -= 64 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      _tile_loadd(6, a, a_stride);
      a += kremainder;
      _tile_loadd(7, (const int8_t*) w + 0, 128);
      _tile_dpbssd(0, 6, 7);
      _tile_loadd(7, (const int8_t*) w + 64, 128);
      _tile_dpbssd(1, 6, 7);

      w = (const int8_t*) w + kremainder * 32;
      k -= kremainder * sizeof(int8_t);
    }

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 tile at a time (16 registers)
    _tile_stored(0, &res[0][0], 64);
    _tile_stored(1, &res[1][0], 64);

    // TODO: Fix msan for AMX
    #if defined(__has_feature)
      #if __has_feature(memory_sanitizer)
        __msan_unpoison(res, sizeof(res));
      #endif
    #endif

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 row at a time.
    // Add tile to bias
    __m512i vacc0x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(&res[0][0] + 0));
    __m512i vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(&res[1][0] + 0));

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled0xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc0xGHIJKLMNOPQRSTUV);

    const __m512 vscale0123456789ABCDEF = _mm512_load_ps((const float*) w + 0);
    const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_load_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled0xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled0xGHIJKLMNOPQRSTUV);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, voutput_zero_point);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc0xGHIJKLMNOPQRSTUV, voutput_zero_point);

    __m128i vout0x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc0x0123456789ABCDEF);
    __m128i vout0xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc0xGHIJKLMNOPQRSTUV);

    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);
    vout0xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout0xGHIJKLMNOPQRSTUV, voutput_min);

    if XNN_LIKELY(nc >= 32) {
      _mm_storeu_si128((__m128i*) (c0 + 0), vout0x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c0 + 16), vout0xGHIJKLMNOPQRSTUV);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a -= kc;
      nc -= 32;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));

      _mm_mask_storeu_epi8(c0 + 0, vmask0, vout0x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c0 + 16, vmask1, vout0xGHIJKLMNOPQRSTUV);
      nc = 0;
    }
  } while (nc != 0);
  // Release tile config
  _tile_release();
  #endif  // defined(__x86_64__)
}
