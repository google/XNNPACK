// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c4-avx512amx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>

// Define tile config data structure
typedef struct __tile_config {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[8];
  uint16_t reserved_1[8];
  uint8_t rows[8];
  uint8_t reserved_2[8];
} __tilecfg;

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512amx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

// TODO: amxintrin.h only provide intrinsics for __x86_64__
// Update if amxintrin changes
#if defined(__x86_64__) && defined(__AMX_TILE__)
  int32_t res[1 * 16];

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  size_t kremainder = kc & 63;
  if (kremainder == 0) {  // zero is invalid config
    kremainder = 64;
  }

  // Load tile configuration
  __tilecfg tile_data = {0};
  tile_data.palette_id = 1;
  tile_data.rows[0] = mr; // tmm0 = res
  tile_data.rows[1] = mr; // tmm1 = input
  tile_data.rows[2] = 16; // tmm2 = weights
  tile_data.rows[3] = mr; // tmm3 = input remainder
  tile_data.rows[4] = kremainder >> 2; // tmm4 = weights remainder
  tile_data.colsb[0] = 64;
  tile_data.colsb[1] = 64;
  tile_data.colsb[2] = 64;
  tile_data.colsb[3] = kremainder;
  tile_data.colsb[4] = 64;
  _tile_loadconfig(&tile_data);

  //const int8_t* a0 = a;
  int8_t* c0 = c;

  const __m512 voutput_max_less_zero_point = _mm512_set1_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_set1_epi32(params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);

  do {
    __m512i vacc0123456789ABCDEF = _mm512_load_epi32(w);
    w = (const int32_t*) w + 16;

   // Zero tile accumulator
   _tile_zero(0);  // tmm0 is accumulator

    size_t k = kc;
    while (k >= 64 * sizeof(int8_t)) {
      _tile_stream_loadd(1, a, a_stride);
      _tile_stream_loadd(2, w, 64);

      // Multiply tiles
      _tile_dpbssd (0, 1, 2);

      a += 64;
      w = (const int8_t*) w + 1024;
      k -= 64 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      _tile_stream_loadd(3, a, a_stride);
      _tile_stream_loadd(4, w, 64);

      // Multiply tiles
      _tile_dpbssd (0, 3, 4);

      a += kremainder;
      w = (const int8_t*) w + kremainder * 16;
      k -= kremainder * sizeof(int8_t);
    }

    // Add tile to bias
    _tile_stored(0, res, 64);
    __m512i vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_load_epi32(res + 0));

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const float*) w + 16;
    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale012345678ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, voutput_zero_point);

    __m128i vout0x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc0x0123456789ABCDEF);

    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t*) ((uintptr_t) a - kc);
      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - UINT32_C(1));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
  // Release tile config
  _tile_release();
#endif  // defined(__x86_64__) && defined(__AMX_TILE__)
}
