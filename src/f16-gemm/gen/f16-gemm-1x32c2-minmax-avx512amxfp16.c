// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/c2-avx512amxfp16.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
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
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_f16_gemm_minmax_ukernel_1x32c2__avx512amxfp16(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_float16* restrict a,
    size_t a_stride,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_float16) == 0);
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
  XNN_ALIGN(64) float res[2][1 * 16];

  const size_t kremainder = (kc & 63) ? (kc & 63) : 64;
  const size_t kremainder_colsb = round_up_po2(kremainder, 2 * sizeof(xnn_float16));
  const __mmask32 kremainder_mask = _cvtu32_mask32((uint32_t) ((UINT64_C(1) << (kremainder >> XNN_LOG2_SIZEOF_FLOAT16)) - 1));

  tile_data.palette_id = 1;
  tile_data.rows[0] = 1;              // tmm0 = res[0]
  tile_data.rows[1] = 1;              // tmm1 = res[1]
  tile_data.rows[2] = 1;              // tmm2 = res[2]
  tile_data.rows[3] = 1;              // tmm3 = res[3]
  tile_data.rows[4] = 1;              // tmm4 = input
  tile_data.rows[5] = 16;                    // tmm5 = weights
  tile_data.rows[6] = 1;              // tmm6 = input remainder
  tile_data.rows[7] = kremainder_colsb >> 2; // tmm7 = weights remainder
  tile_data.colsb[0] = 64;                   // tmm0 = res[0]
  tile_data.colsb[1] = 64;                   // tmm1 = res[1]
  tile_data.colsb[2] = 64;                   // tmm2 = res[2]
  tile_data.colsb[3] = 64;                   // tmm3 = res[3]
  tile_data.colsb[4] = 64;                   // tmm4 = input
  tile_data.colsb[5] = 64;                   // tmm5 = weights
  tile_data.colsb[6] = kremainder_colsb;     // tmm6 = input remainder
  tile_data.colsb[7] = 64;                   // tmm7 = weights remainder
  _tile_loadconfig(&tile_data);

  const __m512 voutput_min = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m512 voutput_max = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.max));

  do {
    const __m512 vbias0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) ((const xnn_float16*) w + 0)));
    const __m512 vbias1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) ((const xnn_float16*) w + 16)));
    w = (const xnn_float16*) w + 32;

    // Zero tile accumulator
    _tile_zero(0);
    _tile_zero(1);

    size_t k = kc;
    while (k >= 32 * sizeof(xnn_float16)) {
      _tile_loadd(4, a, 64);
      a += 32;
      _tile_loadd(5, (const int8_t*) w + 0, 128);
      _tile_dpfp16ps(0, 4, 5);
      if XNN_LIKELY(nc > 16) {
        _tile_loadd(5, (const int8_t*) w + 64, 128);
        _tile_dpfp16ps(1, 4, 5);
      }

      w = (const xnn_float16*) ((const int8_t*) w + 2048);
      k -= 32 * sizeof(xnn_float16);
    }

    if XNN_UNLIKELY(k != 0) {
      const __m512i vin0 = _mm512_maskz_loadu_epi16(kremainder_mask, a);
      a = (const xnn_float16*) ((uintptr_t) a + kremainder);
      _mm512_store_epi32(&res[0][0], vin0);
      _tile_loadd(6, &res[0][0], 64);
      _tile_loadd(7, (const int8_t*) w + 0, 128);
      _tile_dpfp16ps(0, 6, 7);
      if XNN_LIKELY(nc > 16) {
        _tile_loadd(7, (const int8_t*) w + 64, 128);
        _tile_dpfp16ps(1, 6, 7);
      }

      w = (const xnn_float16*) ((const int8_t*) w + kremainder_colsb * 32);
      k -= kremainder;
    }

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 tile at a time (16 registers)
    _tile_stored(0, &res[0][0], 64);
    if XNN_LIKELY(nc > 16) {
      _tile_stored(1, &res[1][0], 64);
    }

    // TODO: Fix msan for AMX
    #if defined(__has_feature)
      #if __has_feature(memory_sanitizer)
        __msan_unpoison(res, sizeof(res));
      #endif
    #endif

    xnn_float16* c0 = c;
    {
      if XNN_LIKELY(nc >= 32) {
        __m512 vacc0 = _mm512_add_ps(_mm512_load_ps(&res[0][0]), vbias0);
        __m512 vacc1 = _mm512_add_ps(_mm512_load_ps(&res[1][0]), vbias1);
        vacc0 = _mm512_max_ps(vacc0, voutput_min);
        vacc1 = _mm512_max_ps(vacc1, voutput_min);
        vacc0 = _mm512_min_ps(vacc0, voutput_max);
        vacc1 = _mm512_min_ps(vacc1, voutput_max);
        __m256i vfp16out0 = _mm512_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT);
        __m256i vfp16out1 = _mm512_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*) (c0 + 0), vfp16out0);
        _mm256_storeu_si256((__m256i*) (c0 + 16), vfp16out1);
      } else {
        __m512 vacc0 = _mm512_add_ps(_mm512_load_ps(&res[0][0]), vbias0);
        vacc0 = _mm512_max_ps(vacc0, voutput_min);
        vacc0 = _mm512_min_ps(vacc0, voutput_max);
        const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
        __m256i vfp16out0 = _mm512_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT);
        _mm256_mask_storeu_epi16(c0 + 0, vmask0, vfp16out0);
        if (nc > 16) {
        __m512 vacc1 = _mm512_add_ps(_mm512_load_ps(&res[1][0]), vbias1);
        vacc1 = _mm512_max_ps(vacc1, voutput_min);
        vacc1 = _mm512_min_ps(vacc1, voutput_max);
        const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
        __m256i vfp16out1 = _mm512_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT);
        _mm256_mask_storeu_epi16(c0 + 16, vmask1, vfp16out1);
        }
      }
    }
    if XNN_LIKELY(nc >= 32) {
      c = (xnn_float16*) ((uintptr_t) c + cn_stride);
      a = (const xnn_float16*) ((uintptr_t) a - kc);
      nc -= 32;
    } else {
      nc = 0;
    }
  } while (nc != 0);
  // Release tile config
  _tile_release();
  #endif  // defined(__x86_64__)
}
