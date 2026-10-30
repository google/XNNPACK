// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__sme2_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__sme2_get_nr() {
  return 2 * svcntsw();
}

#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za")
static void pf16_gemm_32x32c2_sme2_impl(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    const void* restrict w_fp16,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    float min,
    float max)
{
#ifdef PREFETCH_W_MATRIX
#if defined(__APPLE__)
  const size_t prfm_min_w_size = (16 * 1024 * 1024);
#else
  const size_t prfm_min_w_size = (1024 * 1024);
#endif
#endif
  const size_t vl = svcntw();
  const size_t nr = 2 * vl;
  cn_stride = nr * cn_stride; //nr
  const uint16_t* w  = (const uint16_t*) w_fp16;
  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;


  svbool_t pg_16_all    = svptrue_b16();
  svcount_t pcnt_16_all = svptrue_c16();

  svfloat16_t f16_one = svdup_f16(1.0f);
  svfloat16_t vmin = svdup_f16(min);
  svfloat16_t vmax = svdup_f16(max);

#ifdef PREFETCH_W_MATRIX
  size_t kc_packed = (kc + 1) & ~1;
  size_t nc_rup_nr = ((nc + nr - 1) / nr) * nr;
  size_t total_w_size = ((kc_packed * nc_rup_nr) + nc_rup_nr) * sizeof(uint16_t);
#endif


  if (mr <= vl) { //m<=vl, (Implem. ZA 1x4 tiling) mr here is mean m, not the tiling mr:16

    size_t kc_packed = (kc + 1) & ~1;
    size_t w_offset_nr = nr * (kc_packed + 1);
    const uint16_t* w0 = w;
    const uint16_t* w1 = (nc > nr) ? w0 + w_offset_nr : w0;

#ifdef PREFETCH_W_MATRIX
    const size_t kPrefetchDistance = 32;
    const size_t kBytesPerKLoop = 16 * vl;
    const size_t kCacheLineSize = 64;
    const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;
    const size_t kPrfmStride = kBytesPerKLoop / 4;
    const size_t kTrailerPrfmStride = 4 * vl;

    size_t block_size = w_offset_nr * sizeof(uint16_t);
    bool need_prfm_0 = (block_size >= kPrefetchDistance * kBytesPerKLoop) && (total_w_size > prfm_min_w_size);
    bool need_prfm_1 = (nc > nr) && need_prfm_0;

    size_t prfm_acc_w_size_0 = 0;
    size_t prfm_acc_w_size_1 = 0;

    const int8_t* prfm_ptr_0 = (const int8_t*) w0;
    const int8_t* prfm_ptr_1 = (const int8_t*) w1;

    if (need_prfm_0) {
      prfm_acc_w_size_0 = kPrefetchDistance * kBytesPerKLoop;
      int i = 0;
      for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
        __builtin_prefetch(prfm_ptr_0 + kCacheLineSize * i, 0, 2);
      }
      prfm_ptr_0 += kCacheLineSize * i;
    }
    if (need_prfm_1) {
      prfm_acc_w_size_1 = kPrefetchDistance * kBytesPerKLoop;
      int i = 0;
      for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
        __builtin_prefetch(prfm_ptr_1 + kCacheLineSize * i, 0, 2);
      }
      prfm_ptr_1 += kCacheLineSize * i;
    }
#endif

    do {
      svbool_t pg_32_next_nr32_bias = (nc > nr) ? svptrue_b32() : svpfalse_b();
      svfloat16_t f16_bias_0_0 = svreinterpret_f16(svld1uh_u32(svptrue_b32(), w0));
      svfloat16_t f16_bias_0_1 = svreinterpret_f16(svld1uh_u32(svptrue_b32(), w0 + vl)); w0 += nr;
      svfloat16_t f16_bias_1_0 = svreinterpret_f16(svld1uh_u32(pg_32_next_nr32_bias, w1));
      svfloat16_t f16_bias_1_1 = svreinterpret_f16(svld1uh_u32(pg_32_next_nr32_bias, w1 + vl)); w1 += nr;

      svzero_za();
      svmopa_za32_f16_m(0, pg_16_all, pg_16_all, f16_one, f16_bias_0_0);
      svmopa_za32_f16_m(1, pg_16_all, pg_16_all, f16_one, f16_bias_0_1);
      svmopa_za32_f16_m(2, pg_16_all, pg_16_all, f16_one, f16_bias_1_0);
      svmopa_za32_f16_m(3, pg_16_all, pg_16_all, f16_one, f16_bias_1_1);

      size_t k = kc;
      a0 = (const uint16_t*) a;

      svcount_t pcnt_16_next_nr32 = (nc > nr) ? svptrue_c16() : svpfalse_c();

      while (k >= 4) {
        // W matrix
        svfloat16x4_t vb_x4_0 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, w0)); w0 += 4*nr;
        svfloat16x4_t vb_x4_1 = svreinterpret_f16(svld1_u16_x4(pcnt_16_next_nr32, w1)); w1 += 4*nr;

        svfloat16_t vb0 = svget4(vb_x4_0, 0);
        svfloat16_t vb1 = svget4(vb_x4_0, 1);
        svfloat16_t vb2 = svget4(vb_x4_0, 2);
        svfloat16_t vb3 = svget4(vb_x4_0, 3);

        svfloat16_t vb4 = svget4(vb_x4_1, 0);
        svfloat16_t vb5 = svget4(vb_x4_1, 1);
        svfloat16_t vb6 = svget4(vb_x4_1, 2);
        svfloat16_t vb7 = svget4(vb_x4_1, 3);

        // A matrix
        svfloat16x4_t va_x4 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, a0)); a0 += nr*4;
        svfloat16_t va0 = svget4(va_x4, 0);
        svfloat16_t va2 = svget4(va_x4, 2);

#ifdef PREFETCH_W_MATRIX
        if (need_prfm_0 && (prfm_acc_w_size_0 < (block_size - kBytesPerKLoop))) {
          prfm_acc_w_size_0 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_0 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 3 * kPrfmStride, 0, 2);
          prfm_ptr_0 += kBytesPerKLoop;
        }
        if (need_prfm_1 && (prfm_acc_w_size_1 < (block_size - kBytesPerKLoop))) {
          prfm_acc_w_size_1 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_1 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 3 * kPrfmStride, 0, 2);
          prfm_ptr_1 += kBytesPerKLoop;
        }
#endif

        // Accumulate
        svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va0, vb0);
        svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va0, vb1);
        svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va0, vb4);
        svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va0, vb5);

        svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va2, vb2);
        svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va2, vb3);
        svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va2, vb6);
        svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va2, vb7);

        k -= 4;
      }

      while (k >= 2) {
        svfloat16x2_t vb_x2_0 = svreinterpret_f16(svld1_u16_x2(pcnt_16_all, w0)); w0 += 2*nr;
        svfloat16x2_t vb_x2_1 = svreinterpret_f16(svld1_u16_x2(pcnt_16_next_nr32, w1)); w1 += 2*nr;

        svfloat16_t vb0 = svget2(vb_x2_0, 0);
        svfloat16_t vb1 = svget2(vb_x2_0, 1);
        svfloat16_t vb4 = svget2(vb_x2_1, 0);
        svfloat16_t vb5 = svget2(vb_x2_1, 1);

        svfloat16x2_t va_x2 = svreinterpret_f16(svld1_u16_x2(pcnt_16_all, a0)); a0 += nr*2;
        svfloat16_t va0 = svget2(va_x2, 0);

        svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va0, vb0);
        svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va0, vb1);
        svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va0, vb4);
        svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va0, vb5);

        k -= 2;
      }

      if (k != 0) {
        svfloat16x2_t vb_x2_0 = svreinterpret_f16(svld1_u16_x2(pcnt_16_all, w0)); w0 += 2*nr;
        svfloat16x2_t vb_x2_1 = svreinterpret_f16(svld1_u16_x2(pcnt_16_next_nr32, w1)); w1 += 2*nr;

        svfloat16_t vb0 = svget2(vb_x2_0, 0);
        svfloat16_t vb1 = svget2(vb_x2_0, 1);
        svfloat16_t vb4 = svget2(vb_x2_1, 0);
        svfloat16_t vb5 = svget2(vb_x2_1, 1);

        svfloat16x2_t va_x2 = svreinterpret_f16(svld1_u16_x2(pcnt_16_all, a0)); a0 += nr*2;
        svfloat16_t va0 = svget2(va_x2, 0);

        svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va0, vb0);
        svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va0, vb1);
        svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va0, vb4);
        svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va0, vb5);

        k = 0;
      }

#ifdef PREFETCH_W_MATRIX
      if (need_prfm_0) {
        if (prfm_acc_w_size_0 <= (block_size - 3 * (4 * vl))) {
          prfm_acc_w_size_0 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_0 <= (block_size - 2 * (4 * vl))) {
          prfm_acc_w_size_0 += 2 * (4 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 2 * (4 * vl);
        } else if (prfm_acc_w_size_0 <= (block_size - 1 * (4 * vl))) {
          prfm_acc_w_size_0 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 1 * (4 * vl);
        }
      }
      if (need_prfm_1) {
        if (prfm_acc_w_size_1 <= (block_size - 3 * (4 * vl))) {
          prfm_acc_w_size_1 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_1 <= (block_size - 2 * (4 * vl))) {
          prfm_acc_w_size_1 += 2 * (4 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 2 * (4 * vl);
        } else if (prfm_acc_w_size_1 <= (block_size - 1 * (4 * vl))) {
          prfm_acc_w_size_1 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 1 * (4 * vl);
        }
      }
#endif

      uint16_t* c_inc = c0;
      if (nc != 0) {
        svcount_t pg_16_len_x2 = svwhilelt_c16_u64(0, nc, 2);
        size_t mr_idx = 0;
        for (size_t i = 0; i < mr; i++) {
          svfloat16_t f16_01 = svcvt_f16_f32_x2(svreinterpret_f32( svread_hor_za8_u8_vg2(0, mr_idx) ));
          svfloat16_t f16_23 = svcvt_f16_f32_x2(svreinterpret_f32( svread_hor_za8_u8_vg2(0, mr_idx + 2) ));
          svfloat16x2_t vout = svclamp(svcreate2(f16_01, f16_23), vmin, vmax);

          svst1_u16_x2(pg_16_len_x2, c_inc, svreinterpret_u16(vout));

          c_inc = (uint16_t*) ((uintptr_t)c_inc + cm_stride);
          mr_idx += 4;
        }

        if (nc > 2 * nr) {
          c0 = (uint16_t*)((uintptr_t)c0 + cn_stride * 2);
          w0 = w1;
          nc -= 2 * nr;
          w1 = (nc > nr) ? w0 + w_offset_nr : w0;
#ifdef PREFETCH_W_MATRIX
          need_prfm_1 = (nc > nr) && need_prfm_0;
          if (need_prfm_0) {
            prfm_ptr_0 = (const int8_t*) w0;
            prfm_acc_w_size_0 = kPrefetchDistance * kBytesPerKLoop;
            int i = 0;
            for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
              __builtin_prefetch(prfm_ptr_0 + kCacheLineSize * i, 0, 2);
            }
            prfm_ptr_0 += kCacheLineSize * i;
          }
          if (need_prfm_1) {
            prfm_ptr_1 = (const int8_t*) w1;
            prfm_acc_w_size_1 = kPrefetchDistance * kBytesPerKLoop;
            int i = 0;
            for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
              __builtin_prefetch(prfm_ptr_1 + kCacheLineSize * i, 0, 2);
            }
            prfm_ptr_1 += kCacheLineSize * i;
          }
#endif
        } else {
          nc = 0;
        }
      }

    } while (nc != 0);

  } else if (nc <= vl) { // mr > vl && nc <= vl

#ifdef PREFETCH_W_MATRIX
    const size_t kPrefetchDistance = 16;
    const size_t kBytesPerKLoop = 32 * vl;
    const size_t kCacheLineSize = 64;
    const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;
    const size_t kPrfmStride = kBytesPerKLoop / 8;
    const size_t kTrailerPrfmStride = 8 * vl;

    bool need_prfm = (total_w_size >= kPrefetchDistance * kBytesPerKLoop) && (total_w_size > prfm_min_w_size);
    size_t prfm_acc_w_size = 0;
    const int8_t* prfm_ptr = (const int8_t*) w;

    if (need_prfm) {
      prfm_acc_w_size = kPrefetchDistance * kBytesPerKLoop;
      int i = 0;
      for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
        __builtin_prefetch(prfm_ptr + kCacheLineSize * i, 0, 2);
      }
      prfm_ptr += kCacheLineSize * i;
    }
#endif

    svbool_t pg_32_all = svptrue_b32();
    svfloat16_t f16_bias_0 = svreinterpret_f16(svld1uh_u32(pg_32_all, w)); w = (const uint16_t*) w + nr;

    svzero_za();
    svmopa_za32_f16_m(0, pg_16_all, pg_16_all, f16_one, f16_bias_0);
    svmopa_za32_f16_m(2, pg_16_all, pg_16_all, f16_one, f16_bias_0);

    size_t k = (kc + 1) & ~1;
    a0 = (const uint16_t*) a;

    if (k >= 8) {
      size_t loop_cnt = k / 8;
      k = k % 8;

      svfloat16x4_t va_x4_0 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, a0)); a0 += 4 * nr;
      svfloat16_t vb0 = svreinterpret_f16(svld1_u16(pg_16_all, w));
      svfloat16_t vb2 = svreinterpret_f16(svld1_u16(pg_16_all, w + 2 * nr)); w = (const uint16_t*) w + 4 * nr;

      svfloat16x4_t va_x4_1 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, a0)); a0 += 4 * nr;
      svfloat16_t vb4 = svreinterpret_f16(svld1_u16(pg_16_all, w));
      svfloat16_t vb6 = svreinterpret_f16(svld1_u16(pg_16_all, w + 2 * nr)); w = (const uint16_t*) w + 4 * nr;

      svfloat16_t va0 = svget4(va_x4_0, 0);
      svfloat16_t va1 = svget4(va_x4_0, 1);
      svfloat16_t va2 = svget4(va_x4_0, 2);
      svfloat16_t va3 = svget4(va_x4_0, 3);

      svfloat16_t va4 = svget4(va_x4_1, 0);
      svfloat16_t va5 = svget4(va_x4_1, 1);
      svfloat16_t va6 = svget4(va_x4_1, 2);
      svfloat16_t va7 = svget4(va_x4_1, 3);

#ifdef PREFETCH_W_MATRIX
      if (need_prfm && (prfm_acc_w_size < (total_w_size - kBytesPerKLoop))) {
        prfm_acc_w_size += kBytesPerKLoop;
        __builtin_prefetch(prfm_ptr + 0 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 1 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 2 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 3 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 4 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 5 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 6 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 7 * kPrfmStride, 0, 2);
        prfm_ptr += kBytesPerKLoop;
      }
#endif

      while (loop_cnt > 1) {
#ifdef PREFETCH_W_MATRIX
        if (need_prfm && (prfm_acc_w_size < (total_w_size - kBytesPerKLoop))) {
          prfm_acc_w_size += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 3 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 4 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 5 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 6 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 7 * kPrfmStride, 0, 2);
          prfm_ptr += kBytesPerKLoop;
        }
#endif
        svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va0, vb0);
        svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va1, vb0);
        svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va2, vb2);
        svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va3, vb2);

        va_x4_0 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, a0)); a0 += 4 * nr;
        vb0 = svreinterpret_f16(svld1_u16(pg_16_all, w));
        vb2 = svreinterpret_f16(svld1_u16(pg_16_all, w + 2 * nr)); w = (const uint16_t*) w + 4 * nr;

        svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va4, vb4);
        svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va5, vb4);
        svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va6, vb6);
        svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va7, vb6);

        va_x4_1 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, a0)); a0 += 4 * nr;
        vb4 = svreinterpret_f16(svld1_u16(pg_16_all, w));
        vb6 = svreinterpret_f16(svld1_u16(pg_16_all, w + 2 * nr)); w = (const uint16_t*) w + 4 * nr;

        va0 = svget4(va_x4_0, 0);
        va1 = svget4(va_x4_0, 1);
        va2 = svget4(va_x4_0, 2);
        va3 = svget4(va_x4_0, 3);

        va4 = svget4(va_x4_1, 0);
        va5 = svget4(va_x4_1, 1);
        va6 = svget4(va_x4_1, 2);
        va7 = svget4(va_x4_1, 3);

        loop_cnt--;
      }

      svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va0, vb0);
      svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va1, vb0);
      svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va2, vb2);
      svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va3, vb2);

      svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va4, vb4);
      svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va5, vb4);
      svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va6, vb6);
      svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va7, vb6);
    }

    while (k >= 2) {
      svfloat16_t vb0_0 = svreinterpret_f16(svld1_u16(pg_16_all, w)); w = (const uint16_t*) w + 2 * nr;
      svfloat16x2_t va_x2 = svreinterpret_f16(svld1_u16_x2(pcnt_16_all, a0)); a0 += nr * 2;
      svfloat16_t va0 = svget2(va_x2, 0);
      svfloat16_t va1 = svget2(va_x2, 1);

      svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va0, vb0_0);
      svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va1, vb0_0);

      k -= 2;
    }

#ifdef PREFETCH_W_MATRIX
    if (need_prfm) {
      if (prfm_acc_w_size <= (total_w_size - 3 * (8 * vl))) {
        prfm_acc_w_size += 3 * (8 * vl);
        __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 1 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 2 * kTrailerPrfmStride, 0, 2);
        prfm_ptr += 3 * (8 * vl);
      } else if (prfm_acc_w_size <= (total_w_size - 2 * (8 * vl))) {
        prfm_acc_w_size += 2 * (8 * vl);
        __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 1 * kTrailerPrfmStride, 0, 2);
        prfm_ptr += 2 * (8 * vl);
      } else if (prfm_acc_w_size <= (total_w_size - 1 * (8 * vl))) {
        prfm_acc_w_size += 1 * (8 * vl);
        __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
        prfm_ptr += 1 * (8 * vl);
      }
    }
#endif

    uint16_t* c_inc_top = c0;
    uint16_t* c_inc_bot = (uint16_t*) ((uintptr_t)c0 + vl * cm_stride);
    svbool_t pg_16_len = svwhilelt_b16_u64(0, nc);

    size_t mr_idx = 0;
    for (size_t i = 0; i < vl; i++) {
      svfloat32x2_t za_top = svreinterpret_f32(svread_hor_za8_u8_vg2(0, mr_idx));
      svfloat32x2_t za_bot = svreinterpret_f32(svread_hor_za8_u8_vg2(0, mr_idx + 2));

      svfloat32_t sum_top = svadd_f32_x(pg_32_all, svget2(za_top, 0), svget2(za_top, 1));
      svfloat32_t sum_bot = svadd_f32_x(pg_32_all, svget2(za_bot, 0), svget2(za_bot, 1));

      svfloat16_t f16_top = svcvt_f16_f32_x2(svcreate2(sum_top, svget2(za_top, 1)));
      svfloat16_t f16_bot = svcvt_f16_f32_x2(svcreate2(sum_bot, svget2(za_bot, 1)));

      svfloat16x2_t vout_tmp = svclamp(svcreate2(f16_top, f16_bot), vmin, vmax);

      svst1_u16(pg_16_len, c_inc_top, svreinterpret_u16(svget2(vout_tmp, 0)));
      c_inc_top = (uint16_t*) ((uintptr_t)c_inc_top + cm_stride);

      if (i + vl < mr) {
        svst1_u16(pg_16_len, c_inc_bot, svreinterpret_u16(svget2(vout_tmp, 1)));
        c_inc_bot = (uint16_t*) ((uintptr_t)c_inc_bot + cm_stride);
      }

      mr_idx += 4;
    }

  } else { // mr > vl && nc > vl

#ifdef PREFETCH_W_MATRIX
    const size_t kPrefetchDistance = 16;
    const size_t kBytesPerKLoop = 32 * vl;
    const size_t kCacheLineSize = 64;
    const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;
    const size_t kPrfmStride = kBytesPerKLoop / 8;
    const size_t kTrailerPrfmStride = 8 * vl;

    bool need_prfm = (total_w_size >= kPrefetchDistance * kBytesPerKLoop) && (total_w_size > prfm_min_w_size);
    size_t prfm_acc_w_size = 0;
    const int8_t* prfm_ptr = (const int8_t*) w;

    if (need_prfm) {
      prfm_acc_w_size = kPrefetchDistance * kBytesPerKLoop;
      int i = 0;
      for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
        __builtin_prefetch(prfm_ptr + kCacheLineSize * i, 0, 2);
      }
      prfm_ptr += kCacheLineSize * i;
    }
#endif

    do { //nc x32 loops
      svfloat16_t f16_bias_0 = svreinterpret_f16(svld1uh_u32(svptrue_b32(), w));
      svfloat16_t f16_bias_1 = svreinterpret_f16(svld1uh_u32(svptrue_b32(), w + vl)); w = (const uint16_t*) w + nr;

      svzero_za();
      svmopa_za32_f16_m(0, pg_16_all, pg_16_all, f16_one, f16_bias_0);
      svmopa_za32_f16_m(1, pg_16_all, pg_16_all, f16_one, f16_bias_1);
      svmopa_za32_f16_m(2, pg_16_all, pg_16_all, f16_one, f16_bias_0);
      svmopa_za32_f16_m(3, pg_16_all, pg_16_all, f16_one, f16_bias_1);

      size_t k = (kc + 1) & ~1;
      a0 = (const uint16_t*) a;

      // main loop unrolled to 8 (interleaved ld1/fmopa)
      if (k >= 8) {
        size_t loop_cnt = k / 8;
        k = k % 8;

        // Initial loads
        svfloat16x4_t va_x4_0 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, a0)); a0 += 4 * nr;
        svfloat16x4_t vb_x4_0 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, w));  w = (const uint16_t*) w + 4 * nr;

        svfloat16x4_t va_x4_1 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, a0)); a0 += 4 * nr;
        svfloat16x4_t vb_x4_1 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, w));  w = (const uint16_t*) w + 4 * nr;

        svfloat16_t va0 = svget4(va_x4_0, 0);
        svfloat16_t va1 = svget4(va_x4_0, 1);
        svfloat16_t va2 = svget4(va_x4_0, 2);
        svfloat16_t va3 = svget4(va_x4_0, 3);

        svfloat16_t vb0 = svget4(vb_x4_0, 0);
        svfloat16_t vb1 = svget4(vb_x4_0, 1);
        svfloat16_t vb2 = svget4(vb_x4_0, 2);
        svfloat16_t vb3 = svget4(vb_x4_0, 3);

        svfloat16_t va4 = svget4(va_x4_1, 0);
        svfloat16_t va5 = svget4(va_x4_1, 1);
        svfloat16_t va6 = svget4(va_x4_1, 2);
        svfloat16_t va7 = svget4(va_x4_1, 3);

        svfloat16_t vb4 = svget4(vb_x4_1, 0);
        svfloat16_t vb5 = svget4(vb_x4_1, 1);
        svfloat16_t vb6 = svget4(vb_x4_1, 2);
        svfloat16_t vb7 = svget4(vb_x4_1, 3);

#ifdef PREFETCH_W_MATRIX
        if (need_prfm && (prfm_acc_w_size < (total_w_size - kBytesPerKLoop))) {
          prfm_acc_w_size += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 3 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 4 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 5 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 6 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 7 * kPrfmStride, 0, 2);
          prfm_ptr += kBytesPerKLoop;
        }
#endif

        while (loop_cnt > 1) {
#ifdef PREFETCH_W_MATRIX
          if (need_prfm && (prfm_acc_w_size < (total_w_size - kBytesPerKLoop))) {
            prfm_acc_w_size += kBytesPerKLoop;
            __builtin_prefetch(prfm_ptr + 0 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 1 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 2 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 3 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 4 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 5 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 6 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 7 * kPrfmStride, 0, 2);
            prfm_ptr += kBytesPerKLoop;
          }
#endif
          // Compute K0..3
          svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va0, vb0);
          svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va0, vb1);
          svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va1, vb0);
          svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va1, vb1);
          svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va2, vb2);
          svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va2, vb3);
          svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va3, vb2);
          svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va3, vb3);

          // Load next K0..3
          va_x4_0 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, a0)); a0 += 4 * nr;
          vb_x4_0 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, w));  w = (const uint16_t*) w + 4 * nr;

          // Compute K4..7
          svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va4, vb4);
          svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va4, vb5);
          svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va5, vb4);
          svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va5, vb5);
          svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va6, vb6);
          svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va6, vb7);
          svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va7, vb6);
          svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va7, vb7);

          // Load next K4..7
          va_x4_1 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, a0)); a0 += 4 * nr;
          vb_x4_1 = svreinterpret_f16(svld1_u16_x4(pcnt_16_all, w));  w = (const uint16_t*) w + 4 * nr;

          // Update vectors for next iteration
          va0 = svget4(va_x4_0, 0);
          va1 = svget4(va_x4_0, 1);
          va2 = svget4(va_x4_0, 2);
          va3 = svget4(va_x4_0, 3);
          vb0 = svget4(vb_x4_0, 0);
          vb1 = svget4(vb_x4_0, 1);
          vb2 = svget4(vb_x4_0, 2);
          vb3 = svget4(vb_x4_0, 3);

          va4 = svget4(va_x4_1, 0);
          va5 = svget4(va_x4_1, 1);
          va6 = svget4(va_x4_1, 2);
          va7 = svget4(va_x4_1, 3);
          vb4 = svget4(vb_x4_1, 0);
          vb5 = svget4(vb_x4_1, 1);
          vb6 = svget4(vb_x4_1, 2);
          vb7 = svget4(vb_x4_1, 3);

          loop_cnt--;
        }

        // Last iteration (tail) of K>=8
        svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va0, vb0);
        svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va0, vb1);
        svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va1, vb0);
        svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va1, vb1);
        svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va2, vb2);
        svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va2, vb3);
        svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va3, vb2);
        svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va3, vb3);

        svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va4, vb4);
        svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va4, vb5);
        svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va5, vb4);
        svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va5, vb5);
        svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va6, vb6);
        svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va6, vb7);
        svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va7, vb6);
        svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va7, vb7);
      }

      while (k >= 2) {
        //W matrix
        svfloat16x2_t vb_x2 = svreinterpret_f16(svld1_u16_x2(pcnt_16_all, w)); w = (const uint16_t*) w + 2*nr;
        svfloat16_t vb0_0 = svget2(vb_x2, 0);
        svfloat16_t vb0_1 = svget2(vb_x2, 1);

        //A matrix: fisrt 32 mr x 2 k cols
        svfloat16x2_t va_x2 = svreinterpret_f16(svld1_u16_x2(pcnt_16_all, a0)); a0 += nr*2;
        svfloat16_t va0 = svget2(va_x2, 0);
        svfloat16_t va1 = svget2(va_x2, 1);

        svmopa_za32_f16_m(0, pg_16_all, pg_16_all, va0, vb0_0);
        svmopa_za32_f16_m(1, pg_16_all, pg_16_all, va0, vb0_1);
        svmopa_za32_f16_m(2, pg_16_all, pg_16_all, va1, vb0_0);
        svmopa_za32_f16_m(3, pg_16_all, pg_16_all, va1, vb0_1);

        k -= 2;
      }

      uint16_t* c_inc_top = c0;
      uint16_t* c_inc_bot = (uint16_t*) ((uintptr_t)c0 + vl * cm_stride);

      svbool_t pg_16_len = svwhilelt_b16_u64(0, (nc >= nr) ? nr : nc);

      size_t mr_idx = 0;
      size_t loop_iters = (mr > vl) ? vl : mr;

      for (size_t i = 0; i < loop_iters; i++) {
          svfloat32x2_t top_pair = svreinterpret_f32( svread_hor_za8_u8_vg2(0, mr_idx) );
          svfloat16_t f16_top = svcvt_f16_f32_x2(top_pair);

          svfloat32x2_t bot_pair = svreinterpret_f32( svread_hor_za8_u8_vg2(0, mr_idx + 2) );
          svfloat16_t f16_bot = svcvt_f16_f32_x2(bot_pair);

          svfloat16x2_t vout_tmp = svclamp(svcreate2(f16_top, f16_bot), vmin, vmax);
          svfloat16_t vout_top = svget2(vout_tmp, 0);
          svfloat16_t vout_bot = svget2(vout_tmp, 1);

          svst1_u16(pg_16_len, c_inc_top, svreinterpret_u16(vout_top));
          c_inc_top = (uint16_t*) ((uintptr_t)c_inc_top + cm_stride);

          //Store the bottom row (if within bounds)
          if (i + vl < mr) {
              svst1_u16(pg_16_len, c_inc_bot, svreinterpret_u16(vout_bot));
              c_inc_bot = (uint16_t*) ((uintptr_t)c_inc_bot + cm_stride);
          }

          mr_idx += 4;
      }

      if (nc > nr) {
          c0 = (uint16_t*)((uintptr_t)c0 + cn_stride);
          nc -= nr;
      } else {
          nc = 0;
      }

    } while (nc != 0);

#ifdef PREFETCH_W_MATRIX
    if (need_prfm) {
      if (prfm_acc_w_size <= (total_w_size - 3 * (8 * vl))) {
        prfm_acc_w_size += 3 * (8 * vl);
        __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 1 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 2 * kTrailerPrfmStride, 0, 2);
        prfm_ptr += 3 * (8 * vl);
      } else if (prfm_acc_w_size <= (total_w_size - 2 * (8 * vl))) {
        prfm_acc_w_size += 2 * (8 * vl);
        __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 1 * kTrailerPrfmStride, 0, 2);
        prfm_ptr += 2 * (8 * vl);
      } else if (prfm_acc_w_size <= (total_w_size - 1 * (8 * vl))) {
        prfm_acc_w_size += 1 * (8 * vl);
        __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
        prfm_ptr += 1 * (8 * vl);
      }
    }
#endif

  } // end of mr>vl

}

void xnn_pf16_gemm_minmax_ukernel_32x32c2__sme2(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    const void* restrict w_fp16,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params)
{
  if XNN_UNLIKELY(mr == 0 || nc == 0) {
    return;
  }

  pf16_gemm_32x32c2_sme2_impl(mr, nc, kc / sizeof(xnn_float16), a, w_fp16, c, cm_stride, sizeof(xnn_float16),
      xnn_float16_to_float(params->scalar.min), xnn_float16_to_float(params->scalar.max));

}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
