// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-igemm/avx512bf16-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"


// Packed weights layout, per block of 32 output channels:
//   float bias[32];
//   for each kernel element, for each pair of input channels:
//     uint16_t kernel[32][2];  // bf16
//
// Each pair of input channels is broadcast from the input rows and multiplied
// with the pairs of weights using vdpbf16ps.
void xnn_bf16_f32_igemm_minmax_ukernel_5x32c2__avx512bf16_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const xnn_bfloat16** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const xnn_bfloat16* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_bfloat16) == 0);
  assert(ks != 0);
  assert(ks % (5 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(xnn_bfloat16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

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

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  do {
    __m512 vacc0x0 = _mm512_loadu_ps((const float*) w + 0);
    __m512 vacc0x1 = _mm512_loadu_ps((const float*) w + 16);
    __m512 vacc1x0 = vacc0x0;
    __m512 vacc1x1 = vacc0x1;
    __m512 vacc2x0 = vacc0x0;
    __m512 vacc2x1 = vacc0x1;
    __m512 vacc3x0 = vacc0x0;
    __m512 vacc3x1 = vacc0x1;
    __m512 vacc4x0 = vacc0x0;
    __m512 vacc4x1 = vacc0x1;
    w = (const float*) w + 32;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != (const uint16_t*) zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint16_t* restrict a1 = (const uint16_t*) a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != (const uint16_t*) zero) {
        a1 = (const uint16_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint16_t* restrict a2 = (const uint16_t*) a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != (const uint16_t*) zero) {
        a2 = (const uint16_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint16_t* restrict a3 = (const uint16_t*) a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != (const uint16_t*) zero) {
        a3 = (const uint16_t*) ((uintptr_t) a3 + a_offset);
      }
      const uint16_t* restrict a4 = (const uint16_t*) a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != (const uint16_t*) zero) {
        a4 = (const uint16_t*) ((uintptr_t) a4 + a_offset);
      }
      a += 5;

      size_t k = kc;
      for (; k >= 2 * sizeof(xnn_bfloat16); k -= 2 * sizeof(xnn_bfloat16)) {
        const __m512i vb0 = _mm512_loadu_si512((const uint16_t*) w + 0);
        const __m512i vb1 = _mm512_loadu_si512((const uint16_t*) w + 32);
        w = (const uint16_t*) w + 64;

        const __m512i va0 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
        a0 += 2;
        vacc0x0 = _mm512_dpbf16_ps(vacc0x0, (__m512bh) va0, (__m512bh) vb0);
        vacc0x1 = _mm512_dpbf16_ps(vacc0x1, (__m512bh) va0, (__m512bh) vb1);
        const __m512i va1 = _mm512_set1_epi32((int) unaligned_load_u32(a1));
        a1 += 2;
        vacc1x0 = _mm512_dpbf16_ps(vacc1x0, (__m512bh) va1, (__m512bh) vb0);
        vacc1x1 = _mm512_dpbf16_ps(vacc1x1, (__m512bh) va1, (__m512bh) vb1);
        const __m512i va2 = _mm512_set1_epi32((int) unaligned_load_u32(a2));
        a2 += 2;
        vacc2x0 = _mm512_dpbf16_ps(vacc2x0, (__m512bh) va2, (__m512bh) vb0);
        vacc2x1 = _mm512_dpbf16_ps(vacc2x1, (__m512bh) va2, (__m512bh) vb1);
        const __m512i va3 = _mm512_set1_epi32((int) unaligned_load_u32(a3));
        a3 += 2;
        vacc3x0 = _mm512_dpbf16_ps(vacc3x0, (__m512bh) va3, (__m512bh) vb0);
        vacc3x1 = _mm512_dpbf16_ps(vacc3x1, (__m512bh) va3, (__m512bh) vb1);
        const __m512i va4 = _mm512_set1_epi32((int) unaligned_load_u32(a4));
        a4 += 2;
        vacc4x0 = _mm512_dpbf16_ps(vacc4x0, (__m512bh) va4, (__m512bh) vb0);
        vacc4x1 = _mm512_dpbf16_ps(vacc4x1, (__m512bh) va4, (__m512bh) vb1);
      }
      if XNN_UNLIKELY(k != 0) {
        // Odd number of input channels: the packed weights are zero-padded,
        // but the input is not, so only load the last input channel.
        const __m512i vb0 = _mm512_loadu_si512((const uint16_t*) w + 0);
        const __m512i vb1 = _mm512_loadu_si512((const uint16_t*) w + 32);
        w = (const uint16_t*) w + 64;

        const __m512i va0 = _mm512_set1_epi32((int) (uint32_t) a0[0]);
        vacc0x0 = _mm512_dpbf16_ps(vacc0x0, (__m512bh) va0, (__m512bh) vb0);
        vacc0x1 = _mm512_dpbf16_ps(vacc0x1, (__m512bh) va0, (__m512bh) vb1);
        const __m512i va1 = _mm512_set1_epi32((int) (uint32_t) a1[0]);
        vacc1x0 = _mm512_dpbf16_ps(vacc1x0, (__m512bh) va1, (__m512bh) vb0);
        vacc1x1 = _mm512_dpbf16_ps(vacc1x1, (__m512bh) va1, (__m512bh) vb1);
        const __m512i va2 = _mm512_set1_epi32((int) (uint32_t) a2[0]);
        vacc2x0 = _mm512_dpbf16_ps(vacc2x0, (__m512bh) va2, (__m512bh) vb0);
        vacc2x1 = _mm512_dpbf16_ps(vacc2x1, (__m512bh) va2, (__m512bh) vb1);
        const __m512i va3 = _mm512_set1_epi32((int) (uint32_t) a3[0]);
        vacc3x0 = _mm512_dpbf16_ps(vacc3x0, (__m512bh) va3, (__m512bh) vb0);
        vacc3x1 = _mm512_dpbf16_ps(vacc3x1, (__m512bh) va3, (__m512bh) vb1);
        const __m512i va4 = _mm512_set1_epi32((int) (uint32_t) a4[0]);
        vacc4x0 = _mm512_dpbf16_ps(vacc4x0, (__m512bh) va4, (__m512bh) vb0);
        vacc4x1 = _mm512_dpbf16_ps(vacc4x1, (__m512bh) va4, (__m512bh) vb1);
      }
      p -= 5 * sizeof(void*);
    } while (p != 0);

    vacc0x0 = _mm512_max_ps(vmin, vacc0x0);
    vacc0x1 = _mm512_max_ps(vmin, vacc0x1);
    vacc1x0 = _mm512_max_ps(vmin, vacc1x0);
    vacc1x1 = _mm512_max_ps(vmin, vacc1x1);
    vacc2x0 = _mm512_max_ps(vmin, vacc2x0);
    vacc2x1 = _mm512_max_ps(vmin, vacc2x1);
    vacc3x0 = _mm512_max_ps(vmin, vacc3x0);
    vacc3x1 = _mm512_max_ps(vmin, vacc3x1);
    vacc4x0 = _mm512_max_ps(vmin, vacc4x0);
    vacc4x1 = _mm512_max_ps(vmin, vacc4x1);
    vacc0x0 = _mm512_min_ps(vmax, vacc0x0);
    vacc0x1 = _mm512_min_ps(vmax, vacc0x1);
    vacc1x0 = _mm512_min_ps(vmax, vacc1x0);
    vacc1x1 = _mm512_min_ps(vmax, vacc1x1);
    vacc2x0 = _mm512_min_ps(vmax, vacc2x0);
    vacc2x1 = _mm512_min_ps(vmax, vacc2x1);
    vacc3x0 = _mm512_min_ps(vmax, vacc3x0);
    vacc3x1 = _mm512_min_ps(vmax, vacc3x1);
    vacc4x0 = _mm512_min_ps(vmax, vacc4x0);
    vacc4x1 = _mm512_min_ps(vmax, vacc4x1);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c4 + 0, vacc4x0);
      _mm512_storeu_ps(c4 + 16, vacc4x1);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3 + 0, vacc3x0);
      _mm512_storeu_ps(c3 + 16, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2 + 0, vacc2x0);
      _mm512_storeu_ps(c2 + 16, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1 + 0, vacc1x0);
      _mm512_storeu_ps(c1 + 16, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0 + 0, vacc0x0);
      _mm512_storeu_ps(c0 + 16, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const xnn_bfloat16**restrict) ((uintptr_t) a - ks);
      nc -= 32;
    } else {
      if (nc & 16) {
        _mm512_storeu_ps(c4, vacc4x0);
        vacc4x0 = vacc4x1;
        c4 += 16;
        _mm512_storeu_ps(c3, vacc3x0);
        vacc3x0 = vacc3x1;
        c3 += 16;
        _mm512_storeu_ps(c2, vacc2x0);
        vacc2x0 = vacc2x1;
        c2 += 16;
        _mm512_storeu_ps(c1, vacc1x0);
        vacc1x0 = vacc1x1;
        c1 += 16;
        _mm512_storeu_ps(c0, vacc0x0);
        vacc0x0 = vacc0x1;
        c0 += 16;
        nc -= 16;
      }
      if (nc != 0) {
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << nc) - 1));
        _mm512_mask_storeu_ps(c4, vmask, vacc4x0);
        _mm512_mask_storeu_ps(c3, vmask, vacc3x0);
        _mm512_mask_storeu_ps(c2, vmask, vacc2x0);
        _mm512_mask_storeu_ps(c1, vmask, vacc1x0);
        _mm512_mask_storeu_ps(c0, vmask, vacc0x0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
