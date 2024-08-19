// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx16c4-avx512vnni.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/prefetch.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x16c4__avx512vnni_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 10);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    c7 = c6;
  }
  float* c8 = (float*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    c8 = c7;
  }
  float* c9 = (float*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 10) {
    c9 = c8;
  }

  const __m512i vsign_mask = _mm512_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m512i vinput_zero_point = _mm512_set1_epi32((int) quantization_params->zero_point + 128);
  const __m512 vinput_inv_scale = _mm512_set1_ps(quantization_params->inv_scale);
  const __m512 voutput_min = _mm512_set1_ps(params->avx512vnni.min);
  const __m512 voutput_max = _mm512_set1_ps(params->avx512vnni.max);
  do {
    const __m512i vksum0123456789ABCDEF = _mm512_load_epi32(w);
    __m512i vacc0x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point);
    __m512i vacc1x0x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc1x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x1x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc2x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x2x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc3x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x3x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc4x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x4x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc5x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x5x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc6x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x6x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc7x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x7x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc8x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x8x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc9x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x9x0123456789ABCDEF = _mm512_setzero_epi32();
    w = (const int32_t*) w + 16;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      } else {
        a4 = zero_data;
      }
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      } else {
        a5 = zero_data;
      }
      const int8_t* restrict a6 = a[6];
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const int8_t*) ((uintptr_t) a6 + a_offset);
      } else {
        a6 = zero_data;
      }
      const int8_t* restrict a7 = a[7];
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const int8_t*) ((uintptr_t) a7 + a_offset);
      } else {
        a7 = zero_data;
      }
      const int8_t* restrict a8 = a[8];
      if XNN_UNPREDICTABLE(a8 != zero) {
        a8 = (const int8_t*) ((uintptr_t) a8 + a_offset);
      } else {
        a8 = zero_data;
      }
      const int8_t* restrict a9 = a[9];
      if XNN_UNPREDICTABLE(a9 != zero) {
        a9 = (const int8_t*) ((uintptr_t) a9 + a_offset);
      } else {
        a9 = zero_data;
      }
      a += 10;

      size_t k = kc;
      while (k >= 8 * sizeof(int8_t)) {
        const __m512i va0x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a0)), vsign_mask);
        const __m512i va0x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a0 + 4)), vsign_mask);
        a0 += 8;
        const __m512i va1x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a1)), vsign_mask);
        const __m512i va1x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a1 + 4)), vsign_mask);
        a1 += 8;
        const __m512i va2x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a2)), vsign_mask);
        const __m512i va2x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a2 + 4)), vsign_mask);
        a2 += 8;
        const __m512i va3x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a3)), vsign_mask);
        const __m512i va3x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a3 + 4)), vsign_mask);
        a3 += 8;
        const __m512i va4x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a4)), vsign_mask);
        const __m512i va4x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a4 + 4)), vsign_mask);
        a4 += 8;
        const __m512i va5x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a5)), vsign_mask);
        const __m512i va5x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a5 + 4)), vsign_mask);
        a5 += 8;
        const __m512i va6x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a6)), vsign_mask);
        const __m512i va6x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a6 + 4)), vsign_mask);
        a6 += 8;
        const __m512i va7x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a7)), vsign_mask);
        const __m512i va7x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a7 + 4)), vsign_mask);
        a7 += 8;
        const __m512i va8x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a8)), vsign_mask);
        const __m512i va8x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a8 + 4)), vsign_mask);
        a8 += 8;
        const __m512i va9x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a9)), vsign_mask);
        const __m512i va9x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a9 + 4)), vsign_mask);
        a9 += 8;

        const __m512i vb0123456789ABCDEFx0123 = _mm512_load_si512(w);
        const __m512i vb0123456789ABCDEFx4567 = _mm512_load_si512((const int8_t*) w + 64);
        xnn_prefetch_to_l1((const int8_t*) w + 896);

        vacc0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc0x0123456789ABCDEF, va0x0123, vb0123456789ABCDEFx0123);
        vacc1x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x0123456789ABCDEF, va1x0123, vb0123456789ABCDEFx0123);
        vacc2x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc2x0123456789ABCDEF, va2x0123, vb0123456789ABCDEFx0123);
        vacc3x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc3x0123456789ABCDEF, va3x0123, vb0123456789ABCDEFx0123);
        vacc4x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc4x0123456789ABCDEF, va4x0123, vb0123456789ABCDEFx0123);
        vacc5x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc5x0123456789ABCDEF, va5x0123, vb0123456789ABCDEFx0123);
        vacc6x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc6x0123456789ABCDEF, va6x0123, vb0123456789ABCDEFx0123);
        vacc7x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc7x0123456789ABCDEF, va7x0123, vb0123456789ABCDEFx0123);
        vacc8x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc8x0123456789ABCDEF, va8x0123, vb0123456789ABCDEFx0123);
        vacc9x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc9x0123456789ABCDEF, va9x0123, vb0123456789ABCDEFx0123);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        vacc1x0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x0x0123456789ABCDEF, va0x4567, vb0123456789ABCDEFx4567);
        vacc1x1x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x1x0123456789ABCDEF, va1x4567, vb0123456789ABCDEFx4567);
        vacc1x2x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x2x0123456789ABCDEF, va2x4567, vb0123456789ABCDEFx4567);
        vacc1x3x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x3x0123456789ABCDEF, va3x4567, vb0123456789ABCDEFx4567);
        vacc1x4x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x4x0123456789ABCDEF, va4x4567, vb0123456789ABCDEFx4567);
        vacc1x5x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x5x0123456789ABCDEF, va5x4567, vb0123456789ABCDEFx4567);
        vacc1x6x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x6x0123456789ABCDEF, va6x4567, vb0123456789ABCDEFx4567);
        vacc1x7x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x7x0123456789ABCDEF, va7x4567, vb0123456789ABCDEFx4567);
        vacc1x8x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x8x0123456789ABCDEF, va8x4567, vb0123456789ABCDEFx4567);
        vacc1x9x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x9x0123456789ABCDEF, va9x4567, vb0123456789ABCDEFx4567);

        w = (const int8_t*) w + 128;
        k -= 8 * sizeof(int8_t);
      }

      if (k != 0) {
        const __m512i va0x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a0)), vsign_mask);
        a0 += 4;
        const __m512i va1x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a1)), vsign_mask);
        a1 += 4;
        const __m512i va2x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a2)), vsign_mask);
        a2 += 4;
        const __m512i va3x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a3)), vsign_mask);
        a3 += 4;
        const __m512i va4x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a4)), vsign_mask);
        a4 += 4;
        const __m512i va5x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a5)), vsign_mask);
        a5 += 4;
        const __m512i va6x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a6)), vsign_mask);
        a6 += 4;
        const __m512i va7x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a7)), vsign_mask);
        a7 += 4;
        const __m512i va8x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a8)), vsign_mask);
        a8 += 4;
        const __m512i va9x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a9)), vsign_mask);
        a9 += 4;

        const __m512i vb0123456789ABCDEF = _mm512_load_si512(w);
        xnn_prefetch_to_l1((const int8_t*) w + 960);

        vacc0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc0x0123456789ABCDEF, va0x0123, vb0123456789ABCDEF);
        vacc1x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x0123456789ABCDEF, va1x0123, vb0123456789ABCDEF);
        vacc2x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc2x0123456789ABCDEF, va2x0123, vb0123456789ABCDEF);
        vacc3x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc3x0123456789ABCDEF, va3x0123, vb0123456789ABCDEF);
        vacc4x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc4x0123456789ABCDEF, va4x0123, vb0123456789ABCDEF);
        vacc5x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc5x0123456789ABCDEF, va5x0123, vb0123456789ABCDEF);
        vacc6x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc6x0123456789ABCDEF, va6x0123, vb0123456789ABCDEF);
        vacc7x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc7x0123456789ABCDEF, va7x0123, vb0123456789ABCDEF);
        vacc8x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc8x0123456789ABCDEF, va8x0123, vb0123456789ABCDEF);
        vacc9x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc9x0123456789ABCDEF, va9x0123, vb0123456789ABCDEF);

        w = (const int8_t*) w + 64;
        k -= 4 * sizeof(int8_t);
      }

      p -= 10 * sizeof(void*);
    } while (p != 0);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, vacc1x0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc1x0123456789ABCDEF, vacc1x1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc2x0123456789ABCDEF, vacc1x2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_add_epi32(vacc3x0123456789ABCDEF, vacc1x3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_add_epi32(vacc4x0123456789ABCDEF, vacc1x4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_add_epi32(vacc5x0123456789ABCDEF, vacc1x5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_add_epi32(vacc6x0123456789ABCDEF, vacc1x6x0123456789ABCDEF);
    vacc7x0123456789ABCDEF = _mm512_add_epi32(vacc7x0123456789ABCDEF, vacc1x7x0123456789ABCDEF);
    vacc8x0123456789ABCDEF = _mm512_add_epi32(vacc8x0123456789ABCDEF, vacc1x8x0123456789ABCDEF);
    vacc9x0123456789ABCDEF = _mm512_add_epi32(vacc9x0123456789ABCDEF, vacc1x9x0123456789ABCDEF);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);
    __m512 vscaled5x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc5x0123456789ABCDEF);
    __m512 vscaled6x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc6x0123456789ABCDEF);
    __m512 vscaled7x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc7x0123456789ABCDEF);
    __m512 vscaled8x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc8x0123456789ABCDEF);
    __m512 vscaled9x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc9x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vinput_inv_scale);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vinput_inv_scale);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vinput_inv_scale);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vinput_inv_scale);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vinput_inv_scale);
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, vinput_inv_scale);
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, vinput_inv_scale);
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, vinput_inv_scale);
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, vinput_inv_scale);
    vscaled9x0123456789ABCDEF = _mm512_mul_ps(vscaled9x0123456789ABCDEF, vinput_inv_scale);

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w);
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_fmadd_ps(vscaled1x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_fmadd_ps(vscaled2x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled3x0123456789ABCDEF = _mm512_fmadd_ps(vscaled3x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled4x0123456789ABCDEF = _mm512_fmadd_ps(vscaled4x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled5x0123456789ABCDEF = _mm512_fmadd_ps(vscaled5x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled6x0123456789ABCDEF = _mm512_fmadd_ps(vscaled6x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled7x0123456789ABCDEF = _mm512_fmadd_ps(vscaled7x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled8x0123456789ABCDEF = _mm512_fmadd_ps(vscaled8x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled9x0123456789ABCDEF = _mm512_fmadd_ps(vscaled9x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);
    vscaled1x0123456789ABCDEF = _mm512_max_ps(vscaled1x0123456789ABCDEF, voutput_min);
    vscaled2x0123456789ABCDEF = _mm512_max_ps(vscaled2x0123456789ABCDEF, voutput_min);
    vscaled3x0123456789ABCDEF = _mm512_max_ps(vscaled3x0123456789ABCDEF, voutput_min);
    vscaled4x0123456789ABCDEF = _mm512_max_ps(vscaled4x0123456789ABCDEF, voutput_min);
    vscaled5x0123456789ABCDEF = _mm512_max_ps(vscaled5x0123456789ABCDEF, voutput_min);
    vscaled6x0123456789ABCDEF = _mm512_max_ps(vscaled6x0123456789ABCDEF, voutput_min);
    vscaled7x0123456789ABCDEF = _mm512_max_ps(vscaled7x0123456789ABCDEF, voutput_min);
    vscaled8x0123456789ABCDEF = _mm512_max_ps(vscaled8x0123456789ABCDEF, voutput_min);
    vscaled9x0123456789ABCDEF = _mm512_max_ps(vscaled9x0123456789ABCDEF, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max);
    vscaled7x0123456789ABCDEF = _mm512_min_ps(vscaled7x0123456789ABCDEF, voutput_max);
    vscaled8x0123456789ABCDEF = _mm512_min_ps(vscaled8x0123456789ABCDEF, voutput_max);
    vscaled9x0123456789ABCDEF = _mm512_min_ps(vscaled9x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c9, vscaled9x0123456789ABCDEF);
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      _mm512_storeu_ps(c8, vscaled8x0123456789ABCDEF);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      _mm512_storeu_ps(c7, vscaled7x0123456789ABCDEF);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm512_storeu_ps(c6, vscaled6x0123456789ABCDEF);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5, vscaled5x0123456789ABCDEF);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c4, vscaled4x0123456789ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3, vscaled3x0123456789ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2, vscaled2x0123456789ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1, vscaled1x0123456789ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0, vscaled0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c9, vmask, vscaled9x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c8, vmask, vscaled8x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c7, vmask, vscaled7x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c6, vmask, vscaled6x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c5, vmask, vscaled5x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c4, vmask, vscaled4x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c3, vmask, vscaled3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2, vmask, vscaled2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1, vmask, vscaled1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c0, vmask, vscaled0x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
