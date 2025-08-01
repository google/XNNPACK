// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/avx-shuffle4.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/gemm.h"


void xnn_f32_gemm_minmax_ukernel_6x16s4__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }
  const __m256 vmin = _mm256_set1_ps(params->scalar.min);
  const __m256 vmax = _mm256_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w + 0);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc4x01234567 = vacc0x01234567;
    __m256 vacc4x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc5x01234567 = vacc0x01234567;
    __m256 vacc5x89ABCDEF = vacc0x89ABCDEF;
    w += 16;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 += 4;
      __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
      a1 += 4;
      __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
      a2 += 4;
      __m256 va3 = _mm256_broadcast_ps((const __m128*) a3);
      a3 += 4;
      __m256 va4 = _mm256_broadcast_ps((const __m128*) a4);
      a4 += 4;
      __m256 va5 = _mm256_broadcast_ps((const __m128*) a5);
      a5 += 4;

      const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
      const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c0, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c0, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c0, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c0, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(va4, vb01234567c0, vacc4x01234567);
      vacc5x01234567 = _mm256_fmadd_ps(va5, vb01234567c0, vacc5x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc0, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc0, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc0, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc0, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(va4, vb89ABCDEFc0, vacc4x89ABCDEF);
      vacc5x89ABCDEF = _mm256_fmadd_ps(va5, vb89ABCDEFc0, vacc5x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      va4 = _mm256_permute_ps(va4, _MM_SHUFFLE(0, 3, 2, 1));
      va5 = _mm256_permute_ps(va5, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
      const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c1, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c1, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c1, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c1, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(va4, vb01234567c1, vacc4x01234567);
      vacc5x01234567 = _mm256_fmadd_ps(va5, vb01234567c1, vacc5x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc1, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc1, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc1, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc1, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(va4, vb89ABCDEFc1, vacc4x89ABCDEF);
      vacc5x89ABCDEF = _mm256_fmadd_ps(va5, vb89ABCDEFc1, vacc5x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      va4 = _mm256_permute_ps(va4, _MM_SHUFFLE(0, 3, 2, 1));
      va5 = _mm256_permute_ps(va5, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
      const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c2, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c2, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c2, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c2, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(va4, vb01234567c2, vacc4x01234567);
      vacc5x01234567 = _mm256_fmadd_ps(va5, vb01234567c2, vacc5x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc2, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc2, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc2, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc2, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(va4, vb89ABCDEFc2, vacc4x89ABCDEF);
      vacc5x89ABCDEF = _mm256_fmadd_ps(va5, vb89ABCDEFc2, vacc5x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      va4 = _mm256_permute_ps(va4, _MM_SHUFFLE(0, 3, 2, 1));
      va5 = _mm256_permute_ps(va5, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
      const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c3, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c3, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c3, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c3, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(va4, vb01234567c3, vacc4x01234567);
      vacc5x01234567 = _mm256_fmadd_ps(va5, vb01234567c3, vacc5x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc3, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc3, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc3, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc3, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(va4, vb89ABCDEFc3, vacc4x89ABCDEF);
      vacc5x89ABCDEF = _mm256_fmadd_ps(va5, vb89ABCDEFc3, vacc5x89ABCDEF);


      w += 64;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 = (const float*) ((uintptr_t) a0 + k);
      __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
      a1 = (const float*) ((uintptr_t) a1 + k);
      __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
      a2 = (const float*) ((uintptr_t) a2 + k);
      __m256 va3 = _mm256_broadcast_ps((const __m128*) a3);
      a3 = (const float*) ((uintptr_t) a3 + k);
      __m256 va4 = _mm256_broadcast_ps((const __m128*) a4);
      a4 = (const float*) ((uintptr_t) a4 + k);
      __m256 va5 = _mm256_broadcast_ps((const __m128*) a5);
      a5 = (const float*) ((uintptr_t) a5 + k);

      const __m256 vzero = _mm256_setzero_ps();
      const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
      const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va4, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc4x01234567);
      vacc5x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va5, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc5x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va4, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc4x89ABCDEF);
      vacc5x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va5, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc5x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      va4 = _mm256_permute_ps(va4, _MM_SHUFFLE(0, 3, 2, 1));
      va5 = _mm256_permute_ps(va5, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
      const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va4, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc4x01234567);
      vacc5x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va5, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc5x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va4, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc4x89ABCDEF);
      vacc5x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va5, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc5x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      va4 = _mm256_permute_ps(va4, _MM_SHUFFLE(0, 3, 2, 1));
      va5 = _mm256_permute_ps(va5, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
      const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va4, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc4x01234567);
      vacc5x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va5, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc5x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va4, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc4x89ABCDEF);
      vacc5x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va5, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc5x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      va4 = _mm256_permute_ps(va4, _MM_SHUFFLE(0, 3, 2, 1));
      va5 = _mm256_permute_ps(va5, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
      const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va4, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc4x01234567);
      vacc5x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va5, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc5x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va4, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc4x89ABCDEF);
      vacc5x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va5, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc5x89ABCDEF);


      w += 64;
    }

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc2x01234567 = _mm256_max_ps(vmin, vacc2x01234567);
    vacc3x01234567 = _mm256_max_ps(vmin, vacc3x01234567);
    vacc4x01234567 = _mm256_max_ps(vmin, vacc4x01234567);
    vacc5x01234567 = _mm256_max_ps(vmin, vacc5x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_max_ps(vmin, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_max_ps(vmin, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_max_ps(vmin, vacc3x89ABCDEF);
    vacc4x89ABCDEF = _mm256_max_ps(vmin, vacc4x89ABCDEF);
    vacc5x89ABCDEF = _mm256_max_ps(vmin, vacc5x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc2x01234567 = _mm256_min_ps(vmax, vacc2x01234567);
    vacc3x01234567 = _mm256_min_ps(vmax, vacc3x01234567);
    vacc4x01234567 = _mm256_min_ps(vmax, vacc4x01234567);
    vacc5x01234567 = _mm256_min_ps(vmax, vacc5x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_min_ps(vmax, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_min_ps(vmax, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_min_ps(vmax, vacc3x89ABCDEF);
    vacc4x89ABCDEF = _mm256_min_ps(vmax, vacc4x89ABCDEF);
    vacc5x89ABCDEF = _mm256_min_ps(vmax, vacc5x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c2, vacc2x01234567);
      _mm256_storeu_ps(c2 + 8, vacc2x89ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c3, vacc3x01234567);
      _mm256_storeu_ps(c3 + 8, vacc3x89ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_ps(c4, vacc4x01234567);
      _mm256_storeu_ps(c4 + 8, vacc4x89ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm256_storeu_ps(c5, vacc5x01234567);
      _mm256_storeu_ps(c5 + 8, vacc5x89ABCDEF);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c3, vacc3x01234567);
        _mm256_storeu_ps(c4, vacc4x01234567);
        _mm256_storeu_ps(c5, vacc5x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc3x01234567 = vacc3x89ABCDEF;
        vacc4x01234567 = vacc4x89ABCDEF;
        vacc5x01234567 = vacc5x89ABCDEF;

        c0 += 8;
        c1 += 8;
        c2 += 8;
        c3 += 8;
        c4 += 8;
        c5 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc4x0123 = _mm256_castps256_ps128(vacc4x01234567);
      __m128 vacc5x0123 = _mm256_castps256_ps128(vacc5x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c4, vacc4x0123);
        _mm_storeu_ps(c5, vacc5x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc4x0123 = _mm256_extractf128_ps(vacc4x01234567, 1);
        vacc5x0123 = _mm256_extractf128_ps(vacc5x01234567, 1);

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
        c4 += 4;
        c5 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c4, vacc4x0123);
        _mm_storel_pi((__m64*) c5, vacc5x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc4x0123 = _mm_movehl_ps(vacc4x0123, vacc4x0123);
        vacc5x0123 = _mm_movehl_ps(vacc5x0123, vacc5x0123);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
        c4 += 2;
        c5 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c4, vacc4x0123);
        _mm_store_ss(c5, vacc5x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
