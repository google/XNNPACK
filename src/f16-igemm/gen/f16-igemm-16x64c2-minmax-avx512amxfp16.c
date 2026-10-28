// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-igemm/c2-avx512amxfp16.c.in
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
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_f16_igemm_minmax_ukernel_16x64c2__avx512amxfp16(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const xnn_float16** restrict a,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const xnn_float16* zero,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 16);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_float16) == 0);
  assert(ks != 0);
  assert(ks % (16 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(xnn_float16) == 0);
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
  XNN_ALIGN(64) float res[4][16 * 16];
  XNN_ALIGN(64) int32_t vintile[2][16 * 16];

  const size_t kremainder = (kc & 63) ? (kc & 63) : 64;
  const size_t kremainder_colsb = round_up_po2(kremainder, 2 * sizeof(xnn_float16));
  const __mmask32 kremainder_mask = _cvtu32_mask32((uint32_t) ((UINT64_C(1) << (kremainder >> XNN_LOG2_SIZEOF_FLOAT16)) - 1));

  // Load tile configuration
  tile_data.palette_id = 1;
  tile_data.rows[0] = mr;              // tmm0 = res[0]
  tile_data.rows[1] = mr;              // tmm1 = res[1]
  tile_data.rows[2] = mr;              // tmm2 = res[2]
  tile_data.rows[3] = mr;              // tmm3 = res[3]
  tile_data.rows[4] = mr;              // tmm4 = input
  tile_data.rows[5] = 16;                    // tmm5 = weights
  tile_data.rows[6] = mr;              // tmm6 = input remainder
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
    const __m512 vbias2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) ((const xnn_float16*) w + 32)));
    const __m512 vbias3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) ((const xnn_float16*) w + 48)));
    w = (const xnn_float16*) w + 64;

    // Zero tile accumulator
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);

    size_t p = ks;
    do {
      const xnn_float16* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const xnn_float16*) ((uintptr_t) a0 + a_offset);
      }

      size_t k = kc;
      if (mr == 1) {
        a += 16;
        while (k >= 32 * sizeof(xnn_float16)) {
          _tile_loadd(4, a0, 64);
          a0 += 32;
          _tile_loadd(5, (const int8_t*) w + 0, 256);
          _tile_dpfp16ps(0, 4, 5);
          if XNN_LIKELY(nc > 16) {
            _tile_loadd(5, (const int8_t*) w + 64, 256);
            _tile_dpfp16ps(1, 4, 5);
          }
          if XNN_LIKELY(nc > 32) {
            _tile_loadd(5, (const int8_t*) w + 128, 256);
            _tile_dpfp16ps(2, 4, 5);
          }
          if XNN_LIKELY(nc > 48) {
            _tile_loadd(5, (const int8_t*) w + 192, 256);
            _tile_dpfp16ps(3, 4, 5);
          }

          w = (const xnn_float16*) ((const int8_t*) w + 4096);
          k -= 32 * sizeof(xnn_float16);
        }
        if XNN_UNLIKELY(k != 0) {
          const __m512i vin0 = _mm512_maskz_loadu_epi16(kremainder_mask, a0);
          _mm512_store_epi32(&vintile[0][0], vin0);
          _tile_loadd(6, &vintile[0][0], 64);
          _tile_loadd(7, (const int8_t*) w + 0, 256);
          _tile_dpfp16ps(0, 6, 7);
          if XNN_LIKELY(nc > 16) {
            _tile_loadd(7, (const int8_t*) w + 64, 256);
            _tile_dpfp16ps(1, 6, 7);
          }
          if XNN_LIKELY(nc > 32) {
            _tile_loadd(7, (const int8_t*) w + 128, 256);
            _tile_dpfp16ps(2, 6, 7);
          }
          if XNN_LIKELY(nc > 48) {
            _tile_loadd(7, (const int8_t*) w + 192, 256);
            _tile_dpfp16ps(3, 6, 7);
          }

          w = (const xnn_float16*) ((const int8_t*) w + kremainder_colsb * 64);
          k -= kremainder;
        }
      } else {
        const xnn_float16* restrict a1 = a[1];
        assert(a1 != NULL);
        if XNN_UNPREDICTABLE(a1 != zero) {
          a1 = (const xnn_float16*) ((uintptr_t) a1 + a_offset);
        }
        const xnn_float16* restrict a2 = a[2];
        assert(a2 != NULL);
        if XNN_UNPREDICTABLE(a2 != zero) {
          a2 = (const xnn_float16*) ((uintptr_t) a2 + a_offset);
        }
        const xnn_float16* restrict a3 = a[3];
        assert(a3 != NULL);
        if XNN_UNPREDICTABLE(a3 != zero) {
          a3 = (const xnn_float16*) ((uintptr_t) a3 + a_offset);
        }
        const xnn_float16* restrict a4 = a[4];
        assert(a4 != NULL);
        if XNN_UNPREDICTABLE(a4 != zero) {
          a4 = (const xnn_float16*) ((uintptr_t) a4 + a_offset);
        }
        const xnn_float16* restrict a5 = a[5];
        assert(a5 != NULL);
        if XNN_UNPREDICTABLE(a5 != zero) {
          a5 = (const xnn_float16*) ((uintptr_t) a5 + a_offset);
        }
        const xnn_float16* restrict a6 = a[6];
        assert(a6 != NULL);
        if XNN_UNPREDICTABLE(a6 != zero) {
          a6 = (const xnn_float16*) ((uintptr_t) a6 + a_offset);
        }
        const xnn_float16* restrict a7 = a[7];
        assert(a7 != NULL);
        if XNN_UNPREDICTABLE(a7 != zero) {
          a7 = (const xnn_float16*) ((uintptr_t) a7 + a_offset);
        }
        const xnn_float16* restrict a8 = a[8];
        assert(a8 != NULL);
        if XNN_UNPREDICTABLE(a8 != zero) {
          a8 = (const xnn_float16*) ((uintptr_t) a8 + a_offset);
        }
        const xnn_float16* restrict a9 = a[9];
        assert(a9 != NULL);
        if XNN_UNPREDICTABLE(a9 != zero) {
          a9 = (const xnn_float16*) ((uintptr_t) a9 + a_offset);
        }
        const xnn_float16* restrict a10 = a[10];
        assert(a10 != NULL);
        if XNN_UNPREDICTABLE(a10 != zero) {
          a10 = (const xnn_float16*) ((uintptr_t) a10 + a_offset);
        }
        const xnn_float16* restrict a11 = a[11];
        assert(a11 != NULL);
        if XNN_UNPREDICTABLE(a11 != zero) {
          a11 = (const xnn_float16*) ((uintptr_t) a11 + a_offset);
        }
        const xnn_float16* restrict a12 = a[12];
        assert(a12 != NULL);
        if XNN_UNPREDICTABLE(a12 != zero) {
          a12 = (const xnn_float16*) ((uintptr_t) a12 + a_offset);
        }
        const xnn_float16* restrict a13 = a[13];
        assert(a13 != NULL);
        if XNN_UNPREDICTABLE(a13 != zero) {
          a13 = (const xnn_float16*) ((uintptr_t) a13 + a_offset);
        }
        const xnn_float16* restrict a14 = a[14];
        assert(a14 != NULL);
        if XNN_UNPREDICTABLE(a14 != zero) {
          a14 = (const xnn_float16*) ((uintptr_t) a14 + a_offset);
        }
        const xnn_float16* restrict a15 = a[15];
        assert(a15 != NULL);
        if XNN_UNPREDICTABLE(a15 != zero) {
          a15 = (const xnn_float16*) ((uintptr_t) a15 + a_offset);
        }
        a += 16;

        size_t buf_idx = 0;
        while (k >= 32 * sizeof(xnn_float16)) {
          const __m512i vin0 = _mm512_loadu_epi32(a0);
          a0 += 32;
          _mm512_store_epi32(&vintile[buf_idx][0], vin0);
          const __m512i vin1 = _mm512_loadu_epi32(a1);
          a1 += 32;
          _mm512_store_epi32(&vintile[buf_idx][16], vin1);
          const __m512i vin2 = _mm512_loadu_epi32(a2);
          a2 += 32;
          _mm512_store_epi32(&vintile[buf_idx][32], vin2);
          const __m512i vin3 = _mm512_loadu_epi32(a3);
          a3 += 32;
          _mm512_store_epi32(&vintile[buf_idx][48], vin3);
          const __m512i vin4 = _mm512_loadu_epi32(a4);
          a4 += 32;
          _mm512_store_epi32(&vintile[buf_idx][64], vin4);
          const __m512i vin5 = _mm512_loadu_epi32(a5);
          a5 += 32;
          _mm512_store_epi32(&vintile[buf_idx][80], vin5);
          const __m512i vin6 = _mm512_loadu_epi32(a6);
          a6 += 32;
          _mm512_store_epi32(&vintile[buf_idx][96], vin6);
          const __m512i vin7 = _mm512_loadu_epi32(a7);
          a7 += 32;
          _mm512_store_epi32(&vintile[buf_idx][112], vin7);
          const __m512i vin8 = _mm512_loadu_epi32(a8);
          a8 += 32;
          _mm512_store_epi32(&vintile[buf_idx][128], vin8);
          const __m512i vin9 = _mm512_loadu_epi32(a9);
          a9 += 32;
          _mm512_store_epi32(&vintile[buf_idx][144], vin9);
          const __m512i vin10 = _mm512_loadu_epi32(a10);
          a10 += 32;
          _mm512_store_epi32(&vintile[buf_idx][160], vin10);
          const __m512i vin11 = _mm512_loadu_epi32(a11);
          a11 += 32;
          _mm512_store_epi32(&vintile[buf_idx][176], vin11);
          const __m512i vin12 = _mm512_loadu_epi32(a12);
          a12 += 32;
          _mm512_store_epi32(&vintile[buf_idx][192], vin12);
          const __m512i vin13 = _mm512_loadu_epi32(a13);
          a13 += 32;
          _mm512_store_epi32(&vintile[buf_idx][208], vin13);
          const __m512i vin14 = _mm512_loadu_epi32(a14);
          a14 += 32;
          _mm512_store_epi32(&vintile[buf_idx][224], vin14);
          const __m512i vin15 = _mm512_loadu_epi32(a15);
          a15 += 32;
          _mm512_store_epi32(&vintile[buf_idx][240], vin15);
          _tile_loadd(4, &vintile[buf_idx][0], 64);
          buf_idx ^= 1;
          _tile_loadd(5, (const int8_t*) w + 0, 256);
          _tile_dpfp16ps(0, 4, 5);
          if XNN_LIKELY(nc > 16) {
            _tile_loadd(5, (const int8_t*) w + 64, 256);
            _tile_dpfp16ps(1, 4, 5);
          }
          if XNN_LIKELY(nc > 32) {
            _tile_loadd(5, (const int8_t*) w + 128, 256);
            _tile_dpfp16ps(2, 4, 5);
          }
          if XNN_LIKELY(nc > 48) {
            _tile_loadd(5, (const int8_t*) w + 192, 256);
            _tile_dpfp16ps(3, 4, 5);
          }

          w = (const xnn_float16*) ((const int8_t*) w + 4096);
          k -= 32 * sizeof(xnn_float16);
        }
        if XNN_UNLIKELY(k != 0) {
          const __m512i vin0 = _mm512_maskz_loadu_epi16(kremainder_mask, a0);
          a0 = (const xnn_float16*) ((uintptr_t) a0 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][0], vin0);
          const __m512i vin1 = _mm512_maskz_loadu_epi16(kremainder_mask, a1);
          a1 = (const xnn_float16*) ((uintptr_t) a1 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][16], vin1);
          const __m512i vin2 = _mm512_maskz_loadu_epi16(kremainder_mask, a2);
          a2 = (const xnn_float16*) ((uintptr_t) a2 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][32], vin2);
          const __m512i vin3 = _mm512_maskz_loadu_epi16(kremainder_mask, a3);
          a3 = (const xnn_float16*) ((uintptr_t) a3 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][48], vin3);
          const __m512i vin4 = _mm512_maskz_loadu_epi16(kremainder_mask, a4);
          a4 = (const xnn_float16*) ((uintptr_t) a4 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][64], vin4);
          const __m512i vin5 = _mm512_maskz_loadu_epi16(kremainder_mask, a5);
          a5 = (const xnn_float16*) ((uintptr_t) a5 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][80], vin5);
          const __m512i vin6 = _mm512_maskz_loadu_epi16(kremainder_mask, a6);
          a6 = (const xnn_float16*) ((uintptr_t) a6 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][96], vin6);
          const __m512i vin7 = _mm512_maskz_loadu_epi16(kremainder_mask, a7);
          a7 = (const xnn_float16*) ((uintptr_t) a7 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][112], vin7);
          const __m512i vin8 = _mm512_maskz_loadu_epi16(kremainder_mask, a8);
          a8 = (const xnn_float16*) ((uintptr_t) a8 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][128], vin8);
          const __m512i vin9 = _mm512_maskz_loadu_epi16(kremainder_mask, a9);
          a9 = (const xnn_float16*) ((uintptr_t) a9 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][144], vin9);
          const __m512i vin10 = _mm512_maskz_loadu_epi16(kremainder_mask, a10);
          a10 = (const xnn_float16*) ((uintptr_t) a10 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][160], vin10);
          const __m512i vin11 = _mm512_maskz_loadu_epi16(kremainder_mask, a11);
          a11 = (const xnn_float16*) ((uintptr_t) a11 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][176], vin11);
          const __m512i vin12 = _mm512_maskz_loadu_epi16(kremainder_mask, a12);
          a12 = (const xnn_float16*) ((uintptr_t) a12 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][192], vin12);
          const __m512i vin13 = _mm512_maskz_loadu_epi16(kremainder_mask, a13);
          a13 = (const xnn_float16*) ((uintptr_t) a13 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][208], vin13);
          const __m512i vin14 = _mm512_maskz_loadu_epi16(kremainder_mask, a14);
          a14 = (const xnn_float16*) ((uintptr_t) a14 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][224], vin14);
          const __m512i vin15 = _mm512_maskz_loadu_epi16(kremainder_mask, a15);
          a15 = (const xnn_float16*) ((uintptr_t) a15 + kremainder);
          _mm512_store_epi32(&vintile[buf_idx][240], vin15);
          _tile_loadd(6, &vintile[buf_idx][0], 64);
          _tile_loadd(7, (const int8_t*) w + 0, 256);
          _tile_dpfp16ps(0, 6, 7);
          if XNN_LIKELY(nc > 16) {
            _tile_loadd(7, (const int8_t*) w + 64, 256);
            _tile_dpfp16ps(1, 6, 7);
          }
          if XNN_LIKELY(nc > 32) {
            _tile_loadd(7, (const int8_t*) w + 128, 256);
            _tile_dpfp16ps(2, 6, 7);
          }
          if XNN_LIKELY(nc > 48) {
            _tile_loadd(7, (const int8_t*) w + 192, 256);
            _tile_dpfp16ps(3, 6, 7);
          }

          w = (const xnn_float16*) ((const int8_t*) w + kremainder_colsb * 64);
          k -= kremainder;
        }
      }

      p -= 16 * sizeof(void*);
    } while (p != 0);

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 tile at a time (16 registers)
    _tile_stored(0, &res[0][0], 64);
    if XNN_LIKELY(nc > 16) {
      _tile_stored(1, &res[1][0], 64);
    }
    if XNN_LIKELY(nc > 32) {
      _tile_stored(2, &res[2][0], 64);
    }
    if XNN_LIKELY(nc > 48) {
      _tile_stored(3, &res[3][0], 64);
    }

    // TODO: Fix msan for AMX
    #if defined(__has_feature)
      #if __has_feature(memory_sanitizer)
        __msan_unpoison(res, sizeof(res));
      #endif
    #endif

    xnn_float16* c0 = c;
    for (size_t m = 0; m < mr; ++m)
    {
      if XNN_LIKELY(nc >= 64) {
        __m512 vacc0 = _mm512_add_ps(_mm512_load_ps(&res[0][0] + m * 16), vbias0);
        __m512 vacc1 = _mm512_add_ps(_mm512_load_ps(&res[1][0] + m * 16), vbias1);
        __m512 vacc2 = _mm512_add_ps(_mm512_load_ps(&res[2][0] + m * 16), vbias2);
        __m512 vacc3 = _mm512_add_ps(_mm512_load_ps(&res[3][0] + m * 16), vbias3);
        vacc0 = _mm512_max_ps(vacc0, voutput_min);
        vacc1 = _mm512_max_ps(vacc1, voutput_min);
        vacc2 = _mm512_max_ps(vacc2, voutput_min);
        vacc3 = _mm512_max_ps(vacc3, voutput_min);
        vacc0 = _mm512_min_ps(vacc0, voutput_max);
        vacc1 = _mm512_min_ps(vacc1, voutput_max);
        vacc2 = _mm512_min_ps(vacc2, voutput_max);
        vacc3 = _mm512_min_ps(vacc3, voutput_max);
        __m256i vfp16out0 = _mm512_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT);
        __m256i vfp16out1 = _mm512_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT);
        __m256i vfp16out2 = _mm512_cvtps_ph(vacc2, _MM_FROUND_TO_NEAREST_INT);
        __m256i vfp16out3 = _mm512_cvtps_ph(vacc3, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*) (c0 + 0), vfp16out0);
        _mm256_storeu_si256((__m256i*) (c0 + 16), vfp16out1);
        _mm256_storeu_si256((__m256i*) (c0 + 32), vfp16out2);
        _mm256_storeu_si256((__m256i*) (c0 + 48), vfp16out3);
      } else {
        __m512 vacc0 = _mm512_add_ps(_mm512_load_ps(&res[0][0] + m * 16), vbias0);
        vacc0 = _mm512_max_ps(vacc0, voutput_min);
        vacc0 = _mm512_min_ps(vacc0, voutput_max);
        const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
        __m256i vfp16out0 = _mm512_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT);
        _mm256_mask_storeu_epi16(c0 + 0, vmask0, vfp16out0);
        if (nc > 16) {
        __m512 vacc1 = _mm512_add_ps(_mm512_load_ps(&res[1][0] + m * 16), vbias1);
        vacc1 = _mm512_max_ps(vacc1, voutput_min);
        vacc1 = _mm512_min_ps(vacc1, voutput_max);
        const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
        __m256i vfp16out1 = _mm512_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT);
        _mm256_mask_storeu_epi16(c0 + 16, vmask1, vfp16out1);
        }
        if (nc > 32) {
        __m512 vacc2 = _mm512_add_ps(_mm512_load_ps(&res[2][0] + m * 16), vbias2);
        vacc2 = _mm512_max_ps(vacc2, voutput_min);
        vacc2 = _mm512_min_ps(vacc2, voutput_max);
        const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
        __m256i vfp16out2 = _mm512_cvtps_ph(vacc2, _MM_FROUND_TO_NEAREST_INT);
        _mm256_mask_storeu_epi16(c0 + 32, vmask2, vfp16out2);
        }
        if (nc > 48) {
        __m512 vacc3 = _mm512_add_ps(_mm512_load_ps(&res[3][0] + m * 16), vbias3);
        vacc3 = _mm512_max_ps(vacc3, voutput_min);
        vacc3 = _mm512_min_ps(vacc3, voutput_max);
        const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));
        __m256i vfp16out3 = _mm512_cvtps_ph(vacc3, _MM_FROUND_TO_NEAREST_INT);
        _mm256_mask_storeu_epi16(c0 + 48, vmask3, vfp16out3);
        }
      }
      c0 = (xnn_float16*) ((uintptr_t) c0 + cm_stride);
    }
    if XNN_LIKELY(nc >= 64) {
      c = (xnn_float16*) ((uintptr_t) c + cn_stride);
      a = (const xnn_float16** restrict) ((uintptr_t) a - ks);
      nc -= 64;
    } else {
      nc = 0;
    }
  } while (nc != 0);

  // Release tile config
  _tile_release();
  #endif  // defined(__x86_64__)
}
