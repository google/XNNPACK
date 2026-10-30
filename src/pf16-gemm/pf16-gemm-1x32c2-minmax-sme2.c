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
#include "src/xnnpack/math.h"

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__sme2_get_mr() {
  return 1;
}

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__sme2_get_nr() {
  return 2 * svcntsw();
}

#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za")
static void pf16_gemm_1x32c2_sme2_impl(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w_fp16,
    void* c,
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
  const uint16_t* w  = (const uint16_t*) w_fp16;
  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;
  const size_t vl = svcntw();
  const size_t nr = 2 * vl;

  size_t kc_packed = (kc + 1) & ~1;
  const size_t stream_stride = nr + kc_packed * nr;

#ifdef PREFETCH_W_MATRIX
  size_t nc_rup_nr = ((nc + nr - 1) / nr) * nr;
  size_t total_w_size = ((kc_packed * nc_rup_nr) + nc_rup_nr) * sizeof(uint16_t);
  bool need_prfm = (total_w_size > prfm_min_w_size);
  size_t w_block_bytes = stream_stride * sizeof(uint16_t);
#endif

  svbool_t pg_16_msk_8 = svwhilelt_b16_u64(0, 8);

  while (nc != 0) {
    size_t blocks_processed = (nc + nr - 1) / nr;
    if (blocks_processed > 8) blocks_processed = 8;

    // Branch 1: Merged (blocks_processed <= 1) and (blocks_processed <= 2) -> N = 32 or N = 64
    if (blocks_processed <= 2) {
      svcount_t pcnt_0 = svwhilelt_c16_u64(0 * 2, nc * 2, 2);
      svcount_t pcnt_1 = svwhilelt_c16_u64(nr * 2, nc * 2, 2);

      svbool_t pg_0 = svwhilelt_b16_u64(0, nc);
      svbool_t pg_1 = svwhilelt_b16_u64(nr, nc);

      const uint16_t* w0 = w;
      const uint16_t* w1 = (nc > nr) ? w0 + stream_stride : w0;
      w = w + blocks_processed * stream_stride;

      const uint16_t* wb0 = w0;
      const uint16_t* wb1 = w1;

      // Initialize the ZA0 accumulator group with the bias widened to FP32, so
      // that it is added before the FP32 -> FP16 narrowing. Lanes with n >= nc
      // are zero.
      svfloat32_t vbias_0_lo = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(0, nc), w0)));
      svfloat32_t vbias_0_hi = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(vl, nc), w0 + vl))); w0 += nr;
      svfloat32_t vbias_1_lo = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(nr, nc), w1)));
      svfloat32_t vbias_1_hi = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(nr + vl, nc), w1 + vl))); w1 += nr;

      svzero_za();
      svwrite_za32_f32_vg1x4(0, svcreate4(vbias_0_lo, vbias_0_hi, vbias_1_lo, vbias_1_hi));
      size_t k = kc;
      a0 = (const uint16_t*) a;

#ifdef PREFETCH_W_MATRIX
      const size_t kPrefetchDistance = 16;
      const size_t kBytesPerKLoop = 32 * nr;
      const size_t kCacheLineSize = 64;
      const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;
      const size_t kPrfmStride = 4 * vl;
      const size_t kTrailerPrfmStride = 4 * vl;

      bool need_prfm_0 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 0);
      bool need_prfm_1 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > nr);

      size_t prfm_acc_w_size_0 = 0;
      size_t prfm_acc_w_size_1 = 0;

      const int8_t* prfm_ptr_0 = (const int8_t*) wb0;
      const int8_t* prfm_ptr_1 = (const int8_t*) wb1;

      if (need_prfm_0) {
        prfm_acc_w_size_0 = kPrefetchDistance * kBytesPerKLoop;
        int step = 0;
        for (; step < (kPrefetchDistance * kCacheLinesPerLoop); step++) {
          __builtin_prefetch(prfm_ptr_0 + kCacheLineSize * step, 0, 2);
        }
        prfm_ptr_0 += kCacheLineSize * step;
      }
      if (need_prfm_1) {
        prfm_acc_w_size_1 = kPrefetchDistance * kBytesPerKLoop;
        int step = 0;
        for (; step < (kPrefetchDistance * kCacheLinesPerLoop); step++) {
          __builtin_prefetch(prfm_ptr_1 + kCacheLineSize * step, 0, 2);
        }
        prfm_ptr_1 += kCacheLineSize * step;
      }
#endif

      while (k >= 16) {
        svfloat16_t va0 = svreinterpret_f16(svld1rq_u16(pg_16_msk_8, a0));
        svfloat16_t va1 = svreinterpret_f16(svld1rq_u16(pg_16_msk_8, a0 + 8));
        a0 += 16;

        // Group 1: Steps 0, 1, 2, 3 (va0)
        svfloat16x2_t vb0_0 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb1_0 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        svfloat16x2_t vb0_1 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb1_1 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        svfloat16x2_t vb0_2 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb1_2 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        svfloat16x2_t vb0_3 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb1_3 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;

        svdot_lane_za32_f16_vg1x4(0, svcreate4(svget2(vb0_0, 0), svget2(vb0_0, 1), svget2(vb1_0, 0), svget2(vb1_0, 1)), va0, 0); // ZA0
        svdot_lane_za32_f16_vg1x4(1, svcreate4(svget2(vb0_1, 0), svget2(vb0_1, 1), svget2(vb1_1, 0), svget2(vb1_1, 1)), va0, 1); // ZA1
        svdot_lane_za32_f16_vg1x4(2, svcreate4(svget2(vb0_2, 0), svget2(vb0_2, 1), svget2(vb1_2, 0), svget2(vb1_2, 1)), va0, 2); // ZA2
        svdot_lane_za32_f16_vg1x4(3, svcreate4(svget2(vb0_3, 0), svget2(vb0_3, 1), svget2(vb1_3, 0), svget2(vb1_3, 1)), va0, 3); // ZA3

        // Group 2: Steps 4, 5, 6, 7 (va1)
        svfloat16x2_t vb0_4 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb1_4 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        svfloat16x2_t vb0_5 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb1_5 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        svfloat16x2_t vb0_6 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb1_6 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        svfloat16x2_t vb0_7 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb1_7 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;

        svdot_lane_za32_f16_vg1x4(0, svcreate4(svget2(vb0_4, 0), svget2(vb0_4, 1), svget2(vb1_4, 0), svget2(vb1_4, 1)), va1, 0); // ZA0
        svdot_lane_za32_f16_vg1x4(1, svcreate4(svget2(vb0_5, 0), svget2(vb0_5, 1), svget2(vb1_5, 0), svget2(vb1_5, 1)), va1, 1); // ZA1
        svdot_lane_za32_f16_vg1x4(2, svcreate4(svget2(vb0_6, 0), svget2(vb0_6, 1), svget2(vb1_6, 0), svget2(vb1_6, 1)), va1, 2); // ZA2
        svdot_lane_za32_f16_vg1x4(3, svcreate4(svget2(vb0_7, 0), svget2(vb0_7, 1), svget2(vb1_7, 0), svget2(vb1_7, 1)), va1, 3); // ZA3

#ifdef PREFETCH_W_MATRIX
        if (need_prfm_0 && (prfm_acc_w_size_0 < (w_block_bytes - (32 * nr)))) {
          prfm_acc_w_size_0 += 32 * nr;
          __builtin_prefetch(prfm_ptr_0 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 3 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 4 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 5 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 6 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 7 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 8 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 9 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 10 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 11 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 12 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 13 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 14 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 15 * kPrfmStride, 0, 2);
          prfm_ptr_0 += 32 * nr;
        }
        if (need_prfm_1 && (prfm_acc_w_size_1 < (w_block_bytes - (32 * nr)))) {
          prfm_acc_w_size_1 += 32 * nr;
          __builtin_prefetch(prfm_ptr_1 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 3 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 4 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 5 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 6 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 7 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 8 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 9 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 10 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 11 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 12 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 13 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 14 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 15 * kPrfmStride, 0, 2);
          prfm_ptr_1 += 32 * nr;
        }
#endif

        k -= 16;
      }

      while (k >= 8) {
        svfloat16_t va0 = svreinterpret_f16(svld1rq_u16(pg_16_msk_8, a0)); a0 += 8;

        svfloat16x2_t vb_s0_0 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb_s1_0 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        svfloat16x2_t vb_s0_1 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb_s1_1 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        svfloat16x2_t vb_s0_2 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb_s1_2 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        svfloat16x2_t vb_s0_3 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb_s1_3 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;

        // Round-robin across ZA0, ZA1, ZA2, ZA3:
        svdot_lane_za32_f16_vg1x4(0, svcreate4(svget2(vb_s0_0, 0), svget2(vb_s0_0, 1), svget2(vb_s1_0, 0), svget2(vb_s1_0, 1)), va0, 0); // ZA0
        svdot_lane_za32_f16_vg1x4(1, svcreate4(svget2(vb_s0_1, 0), svget2(vb_s0_1, 1), svget2(vb_s1_1, 0), svget2(vb_s1_1, 1)), va0, 1); // ZA1
        svdot_lane_za32_f16_vg1x4(2, svcreate4(svget2(vb_s0_2, 0), svget2(vb_s0_2, 1), svget2(vb_s1_2, 0), svget2(vb_s1_2, 1)), va0, 2); // ZA2
        svdot_lane_za32_f16_vg1x4(3, svcreate4(svget2(vb_s0_3, 0), svget2(vb_s0_3, 1), svget2(vb_s1_3, 0), svget2(vb_s1_3, 1)), va0, 3); // ZA3

#ifdef PREFETCH_W_MATRIX
        if (need_prfm_0 && (prfm_acc_w_size_0 < (w_block_bytes - (16 * nr)))) {
          prfm_acc_w_size_0 += 16 * nr;
          __builtin_prefetch(prfm_ptr_0 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 3 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 4 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 5 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 6 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 7 * kPrfmStride, 0, 2);
          prfm_ptr_0 += 16 * nr;
        }
        if (need_prfm_1 && (prfm_acc_w_size_1 < (w_block_bytes - (16 * nr)))) {
          prfm_acc_w_size_1 += 16 * nr;
          __builtin_prefetch(prfm_ptr_1 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 3 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 4 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 5 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 6 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 7 * kPrfmStride, 0, 2);
          prfm_ptr_1 += 16 * nr;
        }
#endif

        k -= 8;
      }

      if XNN_UNLIKELY(k != 0) {
        svfloat16_t va0;
        size_t len = k;
        k = 0;
        va0 = svreinterpret_f16(svld1rq_u16(svwhilelt_b16_u64(0, len), a0));
        a0 += len;

        bool l1_active = (len > 2);
        bool l2_active = (len > 4);
        bool l3_active = (len > 6);

        // Lane 0 -> ZA0
        svcount_t pcnt_l0_0 = svwhilelt_c16_u64(0 * 2, nc * 2, 2);
        svcount_t pcnt_l0_1 = svwhilelt_c16_u64(nr * 2, nc * 2, 2);
        svfloat16x2_t vb_s0 = svreinterpret_f16(svld1_u16_x2(pcnt_l0_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb_s1 = svreinterpret_f16(svld1_u16_x2(pcnt_l0_1, w1)); w1 += 2 * nr;
        svdot_lane_za32_f16_vg1x4(0, svcreate4(svget2(vb_s0, 0), svget2(vb_s0, 1), svget2(vb_s1, 0), svget2(vb_s1, 1)), va0, 0);

        // Lane 1 -> ZA1
        svcount_t pcnt_l1_0 = svwhilelt_c16_u64(0 * 2, l1_active ? nc * 2 : 0, 2);
        svcount_t pcnt_l1_1 = svwhilelt_c16_u64(nr * 2, l1_active ? nc * 2 : 0, 2);
        vb_s0 = svreinterpret_f16(svld1_u16_x2(pcnt_l1_0, w0)); if (l1_active) w0 += 2 * nr;
        vb_s1 = svreinterpret_f16(svld1_u16_x2(pcnt_l1_1, w1)); if (l1_active) w1 += 2 * nr;
        svdot_lane_za32_f16_vg1x4(1, svcreate4(svget2(vb_s0, 0), svget2(vb_s0, 1), svget2(vb_s1, 0), svget2(vb_s1, 1)), va0, 1);

        // Lane 2 -> ZA2
        svcount_t pcnt_l2_0 = svwhilelt_c16_u64(0 * 2, l2_active ? nc * 2 : 0, 2);
        svcount_t pcnt_l2_1 = svwhilelt_c16_u64(nr * 2, l2_active ? nc * 2 : 0, 2);
        vb_s0 = svreinterpret_f16(svld1_u16_x2(pcnt_l2_0, w0)); if (l2_active) w0 += 2 * nr;
        vb_s1 = svreinterpret_f16(svld1_u16_x2(pcnt_l2_1, w1)); if (l2_active) w1 += 2 * nr;
        svdot_lane_za32_f16_vg1x4(2, svcreate4(svget2(vb_s0, 0), svget2(vb_s0, 1), svget2(vb_s1, 0), svget2(vb_s1, 1)), va0, 2);

        // Lane 3 -> ZA3
        svcount_t pcnt_l3_0 = svwhilelt_c16_u64(0 * 2, l3_active ? nc * 2 : 0, 2);
        svcount_t pcnt_l3_1 = svwhilelt_c16_u64(nr * 2, l3_active ? nc * 2 : 0, 2);
        vb_s0 = svreinterpret_f16(svld1_u16_x2(pcnt_l3_0, w0)); if (l3_active) w0 += 2 * nr;
        vb_s1 = svreinterpret_f16(svld1_u16_x2(pcnt_l3_1, w1)); if (l3_active) w1 += 2 * nr;
        svdot_lane_za32_f16_vg1x4(3, svcreate4(svget2(vb_s0, 0), svget2(vb_s0, 1), svget2(vb_s1, 0), svget2(vb_s1, 1)), va0, 3);

      }

#ifdef PREFETCH_W_MATRIX
      if (need_prfm_0) {
        if (prfm_acc_w_size_0 <= (w_block_bytes - 7 * (4 * vl))) {
          prfm_acc_w_size_0 += 7 * (4 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 3 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 4 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 5 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 6 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 7 * (4 * vl);
        } else if (prfm_acc_w_size_0 <= (w_block_bytes - 5 * (4 * vl))) {
          prfm_acc_w_size_0 += 5 * (4 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 3 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 4 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 5 * (4 * vl);
        } else if (prfm_acc_w_size_0 <= (w_block_bytes - 3 * (4 * vl))) {
          prfm_acc_w_size_0 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_0 <= (w_block_bytes - 1 * (4 * vl))) {
          prfm_acc_w_size_0 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 1 * (4 * vl);
        }
      }
      if (need_prfm_1) {
        if (prfm_acc_w_size_1 <= (w_block_bytes - 7 * (4 * vl))) {
          prfm_acc_w_size_1 += 7 * (4 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 3 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 4 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 5 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 6 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 7 * (4 * vl);
        } else if (prfm_acc_w_size_1 <= (w_block_bytes - 5 * (4 * vl))) {
          prfm_acc_w_size_1 += 5 * (4 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 3 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 4 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 5 * (4 * vl);
        } else if (prfm_acc_w_size_1 <= (w_block_bytes - 3 * (4 * vl))) {
          prfm_acc_w_size_1 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_1 <= (w_block_bytes - 1 * (4 * vl))) {
          prfm_acc_w_size_1 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 1 * (4 * vl);
        }
      }
#endif

      svfloat16_t vmin = svdup_f16(min);
      svfloat16_t vmax = svdup_f16(max);
      svfloat32x4_t za0_vacco_x4 = svread_za32_f32_vg1x4(0);
      svfloat32x4_t za1_vacco_x4 = svread_za32_f32_vg1x4(1);
      svfloat32x4_t za2_vacco_x4 = svread_za32_f32_vg1x4(2);
      svfloat32x4_t za3_vacco_x4 = svread_za32_f32_vg1x4(3);
      svfloat32_t sum0 = svadd_f32_x(svptrue_b32(), svget4(za0_vacco_x4, 0), svget4(za1_vacco_x4, 0));
      sum0 = svadd_f32_x(svptrue_b32(), sum0, svget4(za2_vacco_x4, 0));
      sum0 = svadd_f32_x(svptrue_b32(), sum0, svget4(za3_vacco_x4, 0));
      svfloat32_t sum1 = svadd_f32_x(svptrue_b32(), svget4(za0_vacco_x4, 1), svget4(za1_vacco_x4, 1));
      sum1 = svadd_f32_x(svptrue_b32(), sum1, svget4(za2_vacco_x4, 1));
      sum1 = svadd_f32_x(svptrue_b32(), sum1, svget4(za3_vacco_x4, 1));
      svfloat32_t sum2 = svadd_f32_x(svptrue_b32(), svget4(za0_vacco_x4, 2), svget4(za1_vacco_x4, 2));
      sum2 = svadd_f32_x(svptrue_b32(), sum2, svget4(za2_vacco_x4, 2));
      sum2 = svadd_f32_x(svptrue_b32(), sum2, svget4(za3_vacco_x4, 2));
      svfloat32_t sum3 = svadd_f32_x(svptrue_b32(), svget4(za0_vacco_x4, 3), svget4(za1_vacco_x4, 3));
      sum3 = svadd_f32_x(svptrue_b32(), sum3, svget4(za2_vacco_x4, 3));
      sum3 = svadd_f32_x(svptrue_b32(), sum3, svget4(za3_vacco_x4, 3));
      svfloat32x2_t part01_0 = svcreate2(sum0, sum1);
      svfloat32x2_t part01_1 = svcreate2(sum2, sum3);
      svfloat16_t f16_out_0 = svcvt_f16_f32_x2(part01_0);
      svfloat16_t f16_out_1 = svcvt_f16_f32_x2(part01_1);
      f16_out_0 = svmax_f16_x(pg_0, f16_out_0, vmin);
      f16_out_0 = svmin_f16_x(pg_0, f16_out_0, vmax);
      svst1_u16(pg_0, c0, svreinterpret_u16(f16_out_0));

      if (nc > nr) {
        f16_out_1 = svmax_f16_x(pg_1, f16_out_1, vmin);
        f16_out_1 = svmin_f16_x(pg_1, f16_out_1, vmax);
        svst1_u16(pg_1, c0 + nr, svreinterpret_u16(f16_out_1));
      }

      c0 = (uint16_t*) ((uintptr_t) c0 + blocks_processed * nr * cn_stride);
      if (nc >= blocks_processed * nr) {
        nc -= blocks_processed * nr;
      } else {
        nc = 0;
      }

    } else {
      // Branch 2: Merged (blocks_processed <= 4) and (blocks_processed <= 8) -> N = 128 or N = 256
      svcount_t pcnt_0 = svwhilelt_c16_u64(0 * 2, nc * 2, 2);
      svcount_t pcnt_1 = svwhilelt_c16_u64(nr * 2, nc * 2, 2);
      svcount_t pcnt_2 = svwhilelt_c16_u64((2 * nr) * 2, nc * 2, 2);
      svcount_t pcnt_3 = svwhilelt_c16_u64((3 * nr) * 2, nc * 2, 2);
      svcount_t pcnt_4 = svwhilelt_c16_u64((4 * nr) * 2, nc * 2, 2);
      svcount_t pcnt_5 = svwhilelt_c16_u64((5 * nr) * 2, nc * 2, 2);
      svcount_t pcnt_6 = svwhilelt_c16_u64((6 * nr) * 2, nc * 2, 2);
      svcount_t pcnt_7 = svwhilelt_c16_u64((7 * nr) * 2, nc * 2, 2);

      svbool_t pg_0 = svwhilelt_b16_u64(0, nc);
      svbool_t pg_1 = svwhilelt_b16_u64(nr, nc);
      svbool_t pg_2 = svwhilelt_b16_u64(2 * nr, nc);
      svbool_t pg_3 = svwhilelt_b16_u64(3 * nr, nc);
      svbool_t pg_4 = svwhilelt_b16_u64(4 * nr, nc);
      svbool_t pg_5 = svwhilelt_b16_u64(5 * nr, nc);
      svbool_t pg_6 = svwhilelt_b16_u64(6 * nr, nc);
      svbool_t pg_7 = svwhilelt_b16_u64(7 * nr, nc);

      const uint16_t* w0 = w;
      const uint16_t* w1 = w0 + stream_stride;
      const uint16_t* w2 = w1 + stream_stride;
      const uint16_t* w3 = (nc > 3 * nr)  ? w2 + stream_stride : w0;
      const uint16_t* w4 = (nc > 4 * nr) ? w3 + stream_stride : w0;
      const uint16_t* w5 = (nc > 5 * nr) ? w4 + stream_stride : w0;
      const uint16_t* w6 = (nc > 6 * nr) ? w5 + stream_stride : w0;
      const uint16_t* w7 = (nc > 7 * nr) ? w6 + stream_stride : w0;
      w = w + blocks_processed * stream_stride;

      const uint16_t* wb0 = w0;
      const uint16_t* wb1 = w1;
      const uint16_t* wb2 = w2;
      const uint16_t* wb3 = w3;
      const uint16_t* wb4 = w4;
      const uint16_t* wb5 = w5;
      const uint16_t* wb6 = w6;
      const uint16_t* wb7 = w7;

      // Initialize ZA group g (blocks 2g and 2g + 1) with the bias widened to
      // FP32, so that it is added before the FP32 -> FP16 narrowing. Lanes with
      // n >= nc are zero (blocks past nc load nothing).
      svfloat32_t vbias_0_lo = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(0 * nr, nc), w0)));
      svfloat32_t vbias_0_hi = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(0 * nr + vl, nc), w0 + vl))); w0 += nr;
      svfloat32_t vbias_1_lo = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(1 * nr, nc), w1)));
      svfloat32_t vbias_1_hi = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(1 * nr + vl, nc), w1 + vl))); w1 += nr;
      svfloat32_t vbias_2_lo = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(2 * nr, nc), w2)));
      svfloat32_t vbias_2_hi = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(2 * nr + vl, nc), w2 + vl))); w2 += nr;
      svfloat32_t vbias_3_lo = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(3 * nr, nc), w3)));
      svfloat32_t vbias_3_hi = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(3 * nr + vl, nc), w3 + vl))); w3 += nr;
      svfloat32_t vbias_4_lo = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(4 * nr, nc), w4)));
      svfloat32_t vbias_4_hi = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(4 * nr + vl, nc), w4 + vl))); w4 += nr;
      svfloat32_t vbias_5_lo = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(5 * nr, nc), w5)));
      svfloat32_t vbias_5_hi = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(5 * nr + vl, nc), w5 + vl))); w5 += nr;
      svfloat32_t vbias_6_lo = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(6 * nr, nc), w6)));
      svfloat32_t vbias_6_hi = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(6 * nr + vl, nc), w6 + vl))); w6 += nr;
      svfloat32_t vbias_7_lo = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(7 * nr, nc), w7)));
      svfloat32_t vbias_7_hi = svcvt_f32_f16_x(svptrue_b32(), svreinterpret_f16(svld1uh_u32(svwhilelt_b32_u64(7 * nr + vl, nc), w7 + vl))); w7 += nr;

      svzero_za();
      svwrite_za32_f32_vg1x4(0, svcreate4(vbias_0_lo, vbias_0_hi, vbias_1_lo, vbias_1_hi));
      svwrite_za32_f32_vg1x4(1, svcreate4(vbias_2_lo, vbias_2_hi, vbias_3_lo, vbias_3_hi));
      svwrite_za32_f32_vg1x4(2, svcreate4(vbias_4_lo, vbias_4_hi, vbias_5_lo, vbias_5_hi));
      svwrite_za32_f32_vg1x4(3, svcreate4(vbias_6_lo, vbias_6_hi, vbias_7_lo, vbias_7_hi));
      size_t k = kc;
      a0 = (const uint16_t*) a;

#ifdef PREFETCH_W_MATRIX
      const size_t kPrefetchDistance = 16;
      const size_t kBytesPerKLoop = 8 * nr;
      const size_t kCacheLineSize = 64;
      const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;
      const size_t kPrfmStride = 4 * vl;
      const size_t kTrailerPrfmStride = 4 * vl;

      bool need_prfm_0 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 0);
      bool need_prfm_1 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 1 * nr);
      bool need_prfm_2 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 2 * nr);
      bool need_prfm_3 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 3 * nr);
      bool need_prfm_4 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 4 * nr);
      bool need_prfm_5 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 5 * nr);
      bool need_prfm_6 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 6 * nr);
      bool need_prfm_7 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 7 * nr);

      size_t prfm_acc_w_size_0 = 0;
      size_t prfm_acc_w_size_1 = 0;
      size_t prfm_acc_w_size_2 = 0;
      size_t prfm_acc_w_size_3 = 0;
      size_t prfm_acc_w_size_4 = 0;
      size_t prfm_acc_w_size_5 = 0;
      size_t prfm_acc_w_size_6 = 0;
      size_t prfm_acc_w_size_7 = 0;

      const int8_t* prfm_ptr_0 = (const int8_t*) wb0;
      const int8_t* prfm_ptr_1 = (const int8_t*) wb1;
      const int8_t* prfm_ptr_2 = (const int8_t*) wb2;
      const int8_t* prfm_ptr_3 = (const int8_t*) wb3;
      const int8_t* prfm_ptr_4 = (const int8_t*) wb4;
      const int8_t* prfm_ptr_5 = (const int8_t*) wb5;
      const int8_t* prfm_ptr_6 = (const int8_t*) wb6;
      const int8_t* prfm_ptr_7 = (const int8_t*) wb7;

      if (need_prfm_0) {
        prfm_acc_w_size_0 = kPrefetchDistance * kBytesPerKLoop;
        int step = 0;
        for (; step < (kPrefetchDistance * kCacheLinesPerLoop); step++) {
          __builtin_prefetch(prfm_ptr_0 + kCacheLineSize * step, 0, 2);
        }
        prfm_ptr_0 += kCacheLineSize * step;
      }
      if (need_prfm_1) {
        prfm_acc_w_size_1 = kPrefetchDistance * kBytesPerKLoop;
        int step = 0;
        for (; step < (kPrefetchDistance * kCacheLinesPerLoop); step++) {
          __builtin_prefetch(prfm_ptr_1 + kCacheLineSize * step, 0, 2);
        }
        prfm_ptr_1 += kCacheLineSize * step;
      }
      if (need_prfm_2) {
        prfm_acc_w_size_2 = kPrefetchDistance * kBytesPerKLoop;
        int step = 0;
        for (; step < (kPrefetchDistance * kCacheLinesPerLoop); step++) {
          __builtin_prefetch(prfm_ptr_2 + kCacheLineSize * step, 0, 2);
        }
        prfm_ptr_2 += kCacheLineSize * step;
      }
      if (need_prfm_3) {
        prfm_acc_w_size_3 = kPrefetchDistance * kBytesPerKLoop;
        int step = 0;
        for (; step < (kPrefetchDistance * kCacheLinesPerLoop); step++) {
          __builtin_prefetch(prfm_ptr_3 + kCacheLineSize * step, 0, 2);
        }
        prfm_ptr_3 += kCacheLineSize * step;
      }
      if (need_prfm_4) {
        prfm_acc_w_size_4 = kPrefetchDistance * kBytesPerKLoop;
        int step = 0;
        for (; step < (kPrefetchDistance * kCacheLinesPerLoop); step++) {
          __builtin_prefetch(prfm_ptr_4 + kCacheLineSize * step, 0, 2);
        }
        prfm_ptr_4 += kCacheLineSize * step;
      }
      if (need_prfm_5) {
        prfm_acc_w_size_5 = kPrefetchDistance * kBytesPerKLoop;
        int step = 0;
        for (; step < (kPrefetchDistance * kCacheLinesPerLoop); step++) {
          __builtin_prefetch(prfm_ptr_5 + kCacheLineSize * step, 0, 2);
        }
        prfm_ptr_5 += kCacheLineSize * step;
      }
      if (need_prfm_6) {
        prfm_acc_w_size_6 = kPrefetchDistance * kBytesPerKLoop;
        int step = 0;
        for (; step < (kPrefetchDistance * kCacheLinesPerLoop); step++) {
          __builtin_prefetch(prfm_ptr_6 + kCacheLineSize * step, 0, 2);
        }
        prfm_ptr_6 += kCacheLineSize * step;
      }
      if (need_prfm_7) {
        prfm_acc_w_size_7 = kPrefetchDistance * kBytesPerKLoop;
        int step = 0;
        for (; step < (kPrefetchDistance * kCacheLinesPerLoop); step++) {
          __builtin_prefetch(prfm_ptr_7 + kCacheLineSize * step, 0, 2);
        }
        prfm_ptr_7 += kCacheLineSize * step;
      }
#endif

      svbool_t pg_16_msk_4 = svwhilelt_b16_u64(0, 4);
      while (k >= 4) {
        svfloat16_t va0 = svreinterpret_f16(svld1rq_u16(pg_16_msk_4, a0)); a0 += 4;

        svfloat16x2_t vb_s0 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb_s1 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        svfloat16x2_t vb_s2 = svreinterpret_f16(svld1_u16_x2(pcnt_2, w2)); w2 += 2 * nr;
        svfloat16x2_t vb_s3 = svreinterpret_f16(svld1_u16_x2(pcnt_3, w3)); w3 += 2 * nr;
        svfloat16x2_t vb_s4 = svreinterpret_f16(svld1_u16_x2(pcnt_4, w4)); w4 += 2 * nr;
        svfloat16x2_t vb_s5 = svreinterpret_f16(svld1_u16_x2(pcnt_5, w5)); w5 += 2 * nr;
        svfloat16x2_t vb_s6 = svreinterpret_f16(svld1_u16_x2(pcnt_6, w6)); w6 += 2 * nr;
        svfloat16x2_t vb_s7 = svreinterpret_f16(svld1_u16_x2(pcnt_7, w7)); w7 += 2 * nr;

        svdot_lane_za32_f16_vg1x4(0, svcreate4(svget2(vb_s0, 0), svget2(vb_s0, 1), svget2(vb_s1, 0), svget2(vb_s1, 1)), va0, 0);
        svdot_lane_za32_f16_vg1x4(1, svcreate4(svget2(vb_s2, 0), svget2(vb_s2, 1), svget2(vb_s3, 0), svget2(vb_s3, 1)), va0, 0);
        svdot_lane_za32_f16_vg1x4(2, svcreate4(svget2(vb_s4, 0), svget2(vb_s4, 1), svget2(vb_s5, 0), svget2(vb_s5, 1)), va0, 0);
        svdot_lane_za32_f16_vg1x4(3, svcreate4(svget2(vb_s6, 0), svget2(vb_s6, 1), svget2(vb_s7, 0), svget2(vb_s7, 1)), va0, 0);

        vb_s0 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        vb_s1 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        vb_s2 = svreinterpret_f16(svld1_u16_x2(pcnt_2, w2)); w2 += 2 * nr;
        vb_s3 = svreinterpret_f16(svld1_u16_x2(pcnt_3, w3)); w3 += 2 * nr;
        vb_s4 = svreinterpret_f16(svld1_u16_x2(pcnt_4, w4)); w4 += 2 * nr;
        vb_s5 = svreinterpret_f16(svld1_u16_x2(pcnt_5, w5)); w5 += 2 * nr;
        vb_s6 = svreinterpret_f16(svld1_u16_x2(pcnt_6, w6)); w6 += 2 * nr;
        vb_s7 = svreinterpret_f16(svld1_u16_x2(pcnt_7, w7)); w7 += 2 * nr;

        svdot_lane_za32_f16_vg1x4(0, svcreate4(svget2(vb_s0, 0), svget2(vb_s0, 1), svget2(vb_s1, 0), svget2(vb_s1, 1)), va0, 1);
        svdot_lane_za32_f16_vg1x4(1, svcreate4(svget2(vb_s2, 0), svget2(vb_s2, 1), svget2(vb_s3, 0), svget2(vb_s3, 1)), va0, 1);
        svdot_lane_za32_f16_vg1x4(2, svcreate4(svget2(vb_s4, 0), svget2(vb_s4, 1), svget2(vb_s5, 0), svget2(vb_s5, 1)), va0, 1);
        svdot_lane_za32_f16_vg1x4(3, svcreate4(svget2(vb_s6, 0), svget2(vb_s6, 1), svget2(vb_s7, 0), svget2(vb_s7, 1)), va0, 1);

#ifdef PREFETCH_W_MATRIX
        if (need_prfm_0 && (prfm_acc_w_size_0 < (w_block_bytes - kBytesPerKLoop))) {
          prfm_acc_w_size_0 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_0 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 3 * kPrfmStride, 0, 2);
          prfm_ptr_0 += kBytesPerKLoop;
        }
        if (need_prfm_1 && (prfm_acc_w_size_1 < (w_block_bytes - kBytesPerKLoop))) {
          prfm_acc_w_size_1 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_1 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 3 * kPrfmStride, 0, 2);
          prfm_ptr_1 += kBytesPerKLoop;
        }
        if (need_prfm_2 && (prfm_acc_w_size_2 < (w_block_bytes - kBytesPerKLoop))) {
          prfm_acc_w_size_2 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_2 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_2 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_2 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_2 + 3 * kPrfmStride, 0, 2);
          prfm_ptr_2 += kBytesPerKLoop;
        }
        if (need_prfm_3 && (prfm_acc_w_size_3 < (w_block_bytes - kBytesPerKLoop))) {
          prfm_acc_w_size_3 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_3 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_3 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_3 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_3 + 3 * kPrfmStride, 0, 2);
          prfm_ptr_3 += kBytesPerKLoop;
        }
        if (need_prfm_4 && (prfm_acc_w_size_4 < (w_block_bytes - kBytesPerKLoop))) {
          prfm_acc_w_size_4 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_4 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_4 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_4 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_4 + 3 * kPrfmStride, 0, 2);
          prfm_ptr_4 += kBytesPerKLoop;
        }
        if (need_prfm_5 && (prfm_acc_w_size_5 < (w_block_bytes - kBytesPerKLoop))) {
          prfm_acc_w_size_5 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_5 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_5 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_5 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_5 + 3 * kPrfmStride, 0, 2);
          prfm_ptr_5 += kBytesPerKLoop;
        }
        if (need_prfm_6 && (prfm_acc_w_size_6 < (w_block_bytes - kBytesPerKLoop))) {
          prfm_acc_w_size_6 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_6 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_6 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_6 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_6 + 3 * kPrfmStride, 0, 2);
          prfm_ptr_6 += kBytesPerKLoop;
        }
        if (need_prfm_7 && (prfm_acc_w_size_7 < (w_block_bytes - kBytesPerKLoop))) {
          prfm_acc_w_size_7 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_7 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_7 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_7 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_7 + 3 * kPrfmStride, 0, 2);
          prfm_ptr_7 += kBytesPerKLoop;
        }
#endif

        k -= 4;
      }

      if XNN_UNLIKELY(k != 0) {
        svfloat16_t va0;
        size_t len = k;
        k = 0;
        va0 = svreinterpret_f16(svld1rq_u16(svwhilelt_b16_u64(0, len), a0));
        a0 += len;

        bool k2_active = (len > 2);

        // Lane 0: K=0,1 (always active for len >= 1)
        svfloat16x2_t vb_s0 = svreinterpret_f16(svld1_u16_x2(pcnt_0, w0)); w0 += 2 * nr;
        svfloat16x2_t vb_s1 = svreinterpret_f16(svld1_u16_x2(pcnt_1, w1)); w1 += 2 * nr;
        svfloat16x2_t vb_s2 = svreinterpret_f16(svld1_u16_x2(pcnt_2, w2)); w2 += 2 * nr;
        svfloat16x2_t vb_s3 = svreinterpret_f16(svld1_u16_x2(pcnt_3, w3)); w3 += 2 * nr;
        svfloat16x2_t vb_s4 = svreinterpret_f16(svld1_u16_x2(pcnt_4, w4)); w4 += 2 * nr;
        svfloat16x2_t vb_s5 = svreinterpret_f16(svld1_u16_x2(pcnt_5, w5)); w5 += 2 * nr;
        svfloat16x2_t vb_s6 = svreinterpret_f16(svld1_u16_x2(pcnt_6, w6)); w6 += 2 * nr;
        svfloat16x2_t vb_s7 = svreinterpret_f16(svld1_u16_x2(pcnt_7, w7)); w7 += 2 * nr;

        svdot_lane_za32_f16_vg1x4(0, svcreate4(svget2(vb_s0, 0), svget2(vb_s0, 1), svget2(vb_s1, 0), svget2(vb_s1, 1)), va0, 0);
        svdot_lane_za32_f16_vg1x4(1, svcreate4(svget2(vb_s2, 0), svget2(vb_s2, 1), svget2(vb_s3, 0), svget2(vb_s3, 1)), va0, 0);
        svdot_lane_za32_f16_vg1x4(2, svcreate4(svget2(vb_s4, 0), svget2(vb_s4, 1), svget2(vb_s5, 0), svget2(vb_s5, 1)), va0, 0);
        svdot_lane_za32_f16_vg1x4(3, svcreate4(svget2(vb_s6, 0), svget2(vb_s6, 1), svget2(vb_s7, 0), svget2(vb_s7, 1)), va0, 0);

        // Lane 1: K=2,3 (active when len > 2)
        svcount_t pcnt_k2_0 = svwhilelt_c16_u64(0 * 2, k2_active ? nc * 2 : 0, 2);
        svcount_t pcnt_k2_1 = svwhilelt_c16_u64(nr * 2, k2_active ? nc * 2 : 0, 2);
        svcount_t pcnt_k2_2 = svwhilelt_c16_u64((2 * nr) * 2, k2_active ? nc * 2 : 0, 2);
        svcount_t pcnt_k2_3 = svwhilelt_c16_u64((3 * nr) * 2, (nc > 3 * nr && k2_active) ? nc * 2 : 0, 2);
        svcount_t pcnt_k2_4 = svwhilelt_c16_u64((4 * nr) * 2, (nc > 4 * nr && k2_active) ? nc * 2 : 0, 2);
        svcount_t pcnt_k2_5 = svwhilelt_c16_u64((5 * nr) * 2, (nc > 5 * nr && k2_active) ? nc * 2 : 0, 2);
        svcount_t pcnt_k2_6 = svwhilelt_c16_u64((6 * nr) * 2, (nc > 6 * nr && k2_active) ? nc * 2 : 0, 2);
        svcount_t pcnt_k2_7 = svwhilelt_c16_u64((7 * nr) * 2, (nc > 7 * nr && k2_active) ? nc * 2 : 0, 2);

        vb_s0 = svreinterpret_f16(svld1_u16_x2(pcnt_k2_0, w0));
        vb_s1 = svreinterpret_f16(svld1_u16_x2(pcnt_k2_1, w1));
        vb_s2 = svreinterpret_f16(svld1_u16_x2(pcnt_k2_2, w2));
        vb_s3 = svreinterpret_f16(svld1_u16_x2(pcnt_k2_3, w3));
        vb_s4 = svreinterpret_f16(svld1_u16_x2(pcnt_k2_4, w4));
        vb_s5 = svreinterpret_f16(svld1_u16_x2(pcnt_k2_5, w5));
        vb_s6 = svreinterpret_f16(svld1_u16_x2(pcnt_k2_6, w6));
        vb_s7 = svreinterpret_f16(svld1_u16_x2(pcnt_k2_7, w7));

        svdot_lane_za32_f16_vg1x4(0, svcreate4(svget2(vb_s0, 0), svget2(vb_s0, 1), svget2(vb_s1, 0), svget2(vb_s1, 1)), va0, 1);
        svdot_lane_za32_f16_vg1x4(1, svcreate4(svget2(vb_s2, 0), svget2(vb_s2, 1), svget2(vb_s3, 0), svget2(vb_s3, 1)), va0, 1);
        svdot_lane_za32_f16_vg1x4(2, svcreate4(svget2(vb_s4, 0), svget2(vb_s4, 1), svget2(vb_s5, 0), svget2(vb_s5, 1)), va0, 1);
        svdot_lane_za32_f16_vg1x4(3, svcreate4(svget2(vb_s6, 0), svget2(vb_s6, 1), svget2(vb_s7, 0), svget2(vb_s7, 1)), va0, 1);

      }

#ifdef PREFETCH_W_MATRIX
      if (need_prfm_0) {
        if (prfm_acc_w_size_0 <= (w_block_bytes - 3 * (4 * vl))) {
          prfm_acc_w_size_0 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_0 <= (w_block_bytes - 1 * (4 * vl))) {
          prfm_acc_w_size_0 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 1 * (4 * vl);
        }
      }
      if (need_prfm_1) {
        if (prfm_acc_w_size_1 <= (w_block_bytes - 3 * (4 * vl))) {
          prfm_acc_w_size_1 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_1 <= (w_block_bytes - 1 * (4 * vl))) {
          prfm_acc_w_size_1 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 1 * (4 * vl);
        }
      }
      if (need_prfm_2) {
        if (prfm_acc_w_size_2 <= (w_block_bytes - 3 * (4 * vl))) {
          prfm_acc_w_size_2 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_2 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_2 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_2 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_2 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_2 <= (w_block_bytes - 1 * (4 * vl))) {
          prfm_acc_w_size_2 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_2 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_2 += 1 * (4 * vl);
        }
      }
      if (need_prfm_3) {
        if (prfm_acc_w_size_3 <= (w_block_bytes - 3 * (4 * vl))) {
          prfm_acc_w_size_3 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_3 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_3 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_3 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_3 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_3 <= (w_block_bytes - 1 * (4 * vl))) {
          prfm_acc_w_size_3 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_3 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_3 += 1 * (4 * vl);
        }
      }
      if (need_prfm_4) {
        if (prfm_acc_w_size_4 <= (w_block_bytes - 3 * (4 * vl))) {
          prfm_acc_w_size_4 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_4 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_4 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_4 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_4 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_4 <= (w_block_bytes - 1 * (4 * vl))) {
          prfm_acc_w_size_4 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_4 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_4 += 1 * (4 * vl);
        }
      }
      if (need_prfm_5) {
        if (prfm_acc_w_size_5 <= (w_block_bytes - 3 * (4 * vl))) {
          prfm_acc_w_size_5 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_5 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_5 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_5 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_5 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_5 <= (w_block_bytes - 1 * (4 * vl))) {
          prfm_acc_w_size_5 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_5 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_5 += 1 * (4 * vl);
        }
      }
      if (need_prfm_6) {
        if (prfm_acc_w_size_6 <= (w_block_bytes - 3 * (4 * vl))) {
          prfm_acc_w_size_6 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_6 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_6 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_6 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_6 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_6 <= (w_block_bytes - 1 * (4 * vl))) {
          prfm_acc_w_size_6 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_6 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_6 += 1 * (4 * vl);
        }
      }
      if (need_prfm_7) {
        if (prfm_acc_w_size_7 <= (w_block_bytes - 3 * (4 * vl))) {
          prfm_acc_w_size_7 += 3 * (4 * vl);
          __builtin_prefetch(prfm_ptr_7 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_7 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_7 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_7 += 3 * (4 * vl);
        } else if (prfm_acc_w_size_7 <= (w_block_bytes - 1 * (4 * vl))) {
          prfm_acc_w_size_7 += 1 * (4 * vl);
          __builtin_prefetch(prfm_ptr_7 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_7 += 1 * (4 * vl);
        }
      }
#endif

      svfloat16_t vmin = svdup_f16(min);
      svfloat16_t vmax = svdup_f16(max);

      svfloat32x4_t za0_vacco_x4 = svread_za32_f32_vg1x4(0);
      svfloat32x2_t part01_0 = svcreate2(svget4(za0_vacco_x4, 0), svget4(za0_vacco_x4, 1));
      svfloat32x2_t part01_1 = svcreate2(svget4(za0_vacco_x4, 2), svget4(za0_vacco_x4, 3));
      svfloat16_t f16_out_0 = svcvt_f16_f32_x2(part01_0);
      svfloat16_t f16_out_1 = svcvt_f16_f32_x2(part01_1);
      f16_out_0 = svmax_f16_x(pg_0, f16_out_0, vmin);
      f16_out_0 = svmin_f16_x(pg_0, f16_out_0, vmax);
      svst1_u16(pg_0, c0, svreinterpret_u16(f16_out_0));

      if (nc > nr) {
        f16_out_1 = svmax_f16_x(pg_1, f16_out_1, vmin);
        f16_out_1 = svmin_f16_x(pg_1, f16_out_1, vmax);
        svst1_u16(pg_1, c0 + nr, svreinterpret_u16(f16_out_1));
      }

      svfloat32x4_t za1_vacco_x4 = svread_za32_f32_vg1x4(1);
      svfloat32x2_t part23_0 = svcreate2(svget4(za1_vacco_x4, 0), svget4(za1_vacco_x4, 1));
      svfloat32x2_t part23_1 = svcreate2(svget4(za1_vacco_x4, 2), svget4(za1_vacco_x4, 3));
      svfloat16_t f16_out_2 = svcvt_f16_f32_x2(part23_0);
      svfloat16_t f16_out_3 = svcvt_f16_f32_x2(part23_1);
      if (nc > 2 * nr) {
        f16_out_2 = svmax_f16_x(pg_2, f16_out_2, vmin);
        f16_out_2 = svmin_f16_x(pg_2, f16_out_2, vmax);
        svst1_u16(pg_2, c0 + 2 * nr, svreinterpret_u16(f16_out_2));
      }

      if (nc > 3 * nr) {
        f16_out_3 = svmax_f16_x(pg_3, f16_out_3, vmin);
        f16_out_3 = svmin_f16_x(pg_3, f16_out_3, vmax);
        svst1_u16(pg_3, c0 + 3 * nr, svreinterpret_u16(f16_out_3));
      }

      svfloat32x4_t za2_vacco_x4 = svread_za32_f32_vg1x4(2);
      svfloat32x2_t part45_0 = svcreate2(svget4(za2_vacco_x4, 0), svget4(za2_vacco_x4, 1));
      svfloat32x2_t part45_1 = svcreate2(svget4(za2_vacco_x4, 2), svget4(za2_vacco_x4, 3));
      svfloat16_t f16_out_4 = svcvt_f16_f32_x2(part45_0);
      svfloat16_t f16_out_5 = svcvt_f16_f32_x2(part45_1);
      if (nc > 4 * nr) {
        f16_out_4 = svmax_f16_x(pg_4, f16_out_4, vmin);
        f16_out_4 = svmin_f16_x(pg_4, f16_out_4, vmax);
        svst1_u16(pg_4, c0 + 4 * nr, svreinterpret_u16(f16_out_4));
      }

      if (nc > 5 * nr) {
        f16_out_5 = svmax_f16_x(pg_5, f16_out_5, vmin);
        f16_out_5 = svmin_f16_x(pg_5, f16_out_5, vmax);
        svst1_u16(pg_5, c0 + 5 * nr, svreinterpret_u16(f16_out_5));
      }

      svfloat32x4_t za3_vacco_x4 = svread_za32_f32_vg1x4(3);
      svfloat32x2_t part67_0 = svcreate2(svget4(za3_vacco_x4, 0), svget4(za3_vacco_x4, 1));
      svfloat32x2_t part67_1 = svcreate2(svget4(za3_vacco_x4, 2), svget4(za3_vacco_x4, 3));
      svfloat16_t f16_out_6 = svcvt_f16_f32_x2(part67_0);
      svfloat16_t f16_out_7 = svcvt_f16_f32_x2(part67_1);
      if (nc > 6 * nr) {
        f16_out_6 = svmax_f16_x(pg_6, f16_out_6, vmin);
        f16_out_6 = svmin_f16_x(pg_6, f16_out_6, vmax);
        svst1_u16(pg_6, c0 + 6 * nr, svreinterpret_u16(f16_out_6));
      }

      if (nc > 7 * nr) {
        f16_out_7 = svmax_f16_x(pg_7, f16_out_7, vmin);
        f16_out_7 = svmin_f16_x(pg_7, f16_out_7, vmax);
        svst1_u16(pg_7, c0 + 7 * nr, svreinterpret_u16(f16_out_7));
      }

      c0 = (uint16_t*) ((uintptr_t) c0 + blocks_processed * nr * cn_stride);
      if (nc >= blocks_processed * nr) {
        nc -= blocks_processed * nr;
      } else {
        nc = 0;
      }
    }
  }
}

void xnn_pf16_gemm_minmax_ukernel_1x32c2__sme2(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w_fp16,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* params)
{
  if XNN_UNLIKELY(mr == 0 || nc == 0) {
    return;
  }
  pf16_gemm_1x32c2_sme2_impl(
      mr, nc, kc / sizeof(xnn_float16), a, w_fp16, c, cm_stride, sizeof(xnn_float16),
      xnn_float16_to_float(params->scalar.min),
      xnn_float16_to_float(params->scalar.max));
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
