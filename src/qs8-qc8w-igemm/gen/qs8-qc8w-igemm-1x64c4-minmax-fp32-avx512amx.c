// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c4-avx512amx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#if defined(__has_feature)
  #if __has_feature(memory_sanitizer)
    #include <sanitizer/msan_interface.h>
  #endif
#endif

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x64c4__avx512amx(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params)
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
  XNN_ALIGN(64) int32_t res[4][1 * 16];
  XNN_ALIGN(64) int32_t vintile[1 * 16];

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const size_t kremainder = (kc & 63) ? (kc & 63) : 64;
  const __mmask16 kremainder_mask = _cvtu32_mask16((UINT32_C(1) << (kremainder >> 2)) - 1);

  // Load tile configuration
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
    const __m512i vksum0 = _mm512_loadu_epi32((const int32_t*) w + 0);
    const __m512i vksum1 = _mm512_loadu_epi32((const int32_t*) w + 16);
    const __m512i vksum2 = _mm512_loadu_epi32((const int32_t*) w + 32);
    const __m512i vksum3 = _mm512_loadu_epi32((const int32_t*) w + 48);
    w = (const int32_t*) w + 64;

    // Zero tile accumulator
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      {
        while (k >= 64 * sizeof(int8_t)) {
          _tile_loadd(4, a0, 64);   // Directly load input for mr=1
          a0 += 64;
          _tile_loadd(5, (const int8_t*) w + 0, 256);
          _tile_dpbssd(0, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 64, 256);
          _tile_dpbssd(1, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 128, 256);
          _tile_dpbssd(2, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 192, 256);
          _tile_dpbssd(3, 4, 5);

          w = (const int8_t*) w + 4096;
          k -= 64 * sizeof(int8_t);
        }
      }

      if XNN_UNLIKELY(k != 0) {
        const __m512i vin0 = _mm512_maskz_loadu_epi32(kremainder_mask, a0);
        a0 += kremainder;
        _mm512_store_epi32(vintile + 0, vin0);
        _tile_loadd(6, vintile, 64);
        _tile_loadd(7, (const int8_t*) w + 0, 256);
        _tile_dpbssd(0, 6, 7);
        _tile_loadd(7, (const int8_t*) w + 64, 256);
        _tile_dpbssd(1, 6, 7);
        _tile_loadd(7, (const int8_t*) w + 128, 256);
        _tile_dpbssd(2, 6, 7);
        _tile_loadd(7, (const int8_t*) w + 192, 256);
        _tile_dpbssd(3, 6, 7);

        w = (const int8_t*) w + kremainder * 64;
        k -= kremainder * sizeof(int8_t);
      }

      p -= 1 * sizeof(void*);
    } while (p != 0);

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 tile at a time (16 registers)
    _tile_stored(0, &res[0][0], 64);
    _tile_stored(1, &res[1][0], 64);
    _tile_stored(2, &res[2][0], 64);
    _tile_stored(3, &res[3][0], 64);

    // TODO: Fix msan for AMX
    #if defined(__has_feature)
      #if __has_feature(memory_sanitizer)
        __msan_unpoison(res, sizeof(res));
      #endif
    #endif

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 row at a time.
    // Add tile to bias
    __m512i vacc0x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 0));
    __m512i vacc0x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 0));
    __m512i vacc0x2 = _mm512_add_epi32(vksum2, _mm512_load_epi32(&res[2][0] + 0));
    __m512i vacc0x3 = _mm512_add_epi32(vksum3, _mm512_load_epi32(&res[3][0] + 0));

    __m512 vscaled0x0 = _mm512_cvtepi32_ps(vacc0x0);
    __m512 vscaled0x1 = _mm512_cvtepi32_ps(vacc0x1);
    __m512 vscaled0x2 = _mm512_cvtepi32_ps(vacc0x2);
    __m512 vscaled0x3 = _mm512_cvtepi32_ps(vacc0x3);

    const __m512 vscale0 = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vscale1 = _mm512_loadu_ps((const float*) w + 16);
    const __m512 vscale2 = _mm512_loadu_ps((const float*) w + 32);
    const __m512 vscale3 = _mm512_loadu_ps((const float*) w + 48);
    w = (const int32_t*) w + 64;

    vscaled0x0 = _mm512_mul_ps(vscaled0x0, vscale0);
    vscaled0x1 = _mm512_mul_ps(vscaled0x1, vscale1);
    vscaled0x2 = _mm512_mul_ps(vscaled0x2, vscale2);
    vscaled0x3 = _mm512_mul_ps(vscaled0x3, vscale3);

    vscaled0x0 = _mm512_min_ps(vscaled0x0, voutput_max_less_zero_point);
    vscaled0x1 = _mm512_min_ps(vscaled0x1, voutput_max_less_zero_point);
    vscaled0x2 = _mm512_min_ps(vscaled0x2, voutput_max_less_zero_point);
    vscaled0x3 = _mm512_min_ps(vscaled0x3, voutput_max_less_zero_point);

    vacc0x0 = _mm512_cvtps_epi32(vscaled0x0);
    vacc0x1 = _mm512_cvtps_epi32(vscaled0x1);
    vacc0x2 = _mm512_cvtps_epi32(vscaled0x2);
    vacc0x3 = _mm512_cvtps_epi32(vscaled0x3);

    vacc0x0 = _mm512_add_epi32(vacc0x0, voutput_zero_point);
    vacc0x1 = _mm512_add_epi32(vacc0x1, voutput_zero_point);
    vacc0x2 = _mm512_add_epi32(vacc0x2, voutput_zero_point);
    vacc0x3 = _mm512_add_epi32(vacc0x3, voutput_zero_point);

    __m128i vout0x0 = _mm512_cvtsepi32_epi8(vacc0x0);
    __m128i vout0x1 = _mm512_cvtsepi32_epi8(vacc0x1);
    __m128i vout0x2 = _mm512_cvtsepi32_epi8(vacc0x2);
    __m128i vout0x3 = _mm512_cvtsepi32_epi8(vacc0x3);

    vout0x0 = _mm_max_epi8(vout0x0, voutput_min);
    vout0x1 = _mm_max_epi8(vout0x1, voutput_min);
    vout0x2 = _mm_max_epi8(vout0x2, voutput_min);
    vout0x3 = _mm_max_epi8(vout0x3, voutput_min);

    if XNN_LIKELY(nc >= 64) {
      _mm_storeu_si128((__m128i*) (c0 + 0), vout0x0);
      _mm_storeu_si128((__m128i*) (c0 + 16), vout0x1);
      _mm_storeu_si128((__m128i*) (c0 + 32), vout0x2);
      _mm_storeu_si128((__m128i*) (c0 + 48), vout0x3);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 64;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
      const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));

      _mm_mask_storeu_epi8(c0 + 0, vmask0, vout0x0);
      _mm_mask_storeu_epi8(c0 + 16, vmask1, vout0x1);
      _mm_mask_storeu_epi8(c0 + 32, vmask2, vout0x2);
      _mm_mask_storeu_epi8(c0 + 48, vmask3, vout0x3);
      nc = 0;
    }
  } while (nc != 0);

  // Release tile config
  _tile_release();
  #endif  // defined(__x86_64__)
}
