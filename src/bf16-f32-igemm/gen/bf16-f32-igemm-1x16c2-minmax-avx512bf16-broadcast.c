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


// Packed weights layout, per block of 16 output channels:
//   float bias[16];
//   for each kernel element, for each pair of input channels:
//     uint16_t kernel[16][2];  // bf16
//
// Each pair of input channels is broadcast from the input rows and multiplied
// with the pairs of weights using vdpbf16ps.
void xnn_bf16_f32_igemm_minmax_ukernel_1x16c2__avx512bf16_broadcast(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_bfloat16) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(xnn_bfloat16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  do {
    __m512 vacc0x0 = _mm512_loadu_ps((const float*) w + 0);
    w = (const float*) w + 16;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != (const uint16_t*) zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      for (; k >= 2 * sizeof(xnn_bfloat16); k -= 2 * sizeof(xnn_bfloat16)) {
        const __m512i vb0 = _mm512_loadu_si512((const uint16_t*) w + 0);
        w = (const uint16_t*) w + 32;

        const __m512i va0 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
        a0 += 2;
        vacc0x0 = _mm512_dpbf16_ps(vacc0x0, (__m512bh) va0, (__m512bh) vb0);
      }
      if XNN_UNLIKELY(k != 0) {
        // Odd number of input channels: the packed weights are zero-padded,
        // but the input is not, so only load the last input channel.
        const __m512i vb0 = _mm512_loadu_si512((const uint16_t*) w + 0);
        w = (const uint16_t*) w + 32;

        const __m512i va0 = _mm512_set1_epi32((int) (uint32_t) a0[0]);
        vacc0x0 = _mm512_dpbf16_ps(vacc0x0, (__m512bh) va0, (__m512bh) vb0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0 = _mm512_max_ps(vmin, vacc0x0);
    vacc0x0 = _mm512_min_ps(vmax, vacc0x0);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0 + 0, vacc0x0);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const xnn_bfloat16**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc != 0) {
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << nc) - 1));
        _mm512_mask_storeu_ps(c0, vmask, vacc0x0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
