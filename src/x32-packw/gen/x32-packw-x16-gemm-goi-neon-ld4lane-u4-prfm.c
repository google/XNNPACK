// Auto-generated file. Do not edit!
//   Template: src/x32-packw/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include "xnnpack/packw.h"
#include "xnnpack/prefetch.h"


void xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  uint32x4x4_t vtmp0123x0123;
  vtmp0123x0123.val[0] = vdupq_n_u32(0);
  vtmp0123x0123.val[1] = vdupq_n_u32(0);
  vtmp0123x0123.val[2] = vdupq_n_u32(0);
  vtmp0123x0123.val[3] = vdupq_n_u32(0);
  uint32x4x4_t vtmp0123x4567;
  vtmp0123x4567.val[0] = vdupq_n_u32(0);
  vtmp0123x4567.val[1] = vdupq_n_u32(0);
  vtmp0123x4567.val[2] = vdupq_n_u32(0);
  vtmp0123x4567.val[3] = vdupq_n_u32(0);
  uint32x4x4_t vtmp0123x89AB;
  vtmp0123x89AB.val[0] = vdupq_n_u32(0);
  vtmp0123x89AB.val[1] = vdupq_n_u32(0);
  vtmp0123x89AB.val[2] = vdupq_n_u32(0);
  vtmp0123x89AB.val[3] = vdupq_n_u32(0);
  uint32x4x4_t vtmp0123xCDEF;
  vtmp0123xCDEF.val[0] = vdupq_n_u32(0);
  vtmp0123xCDEF.val[1] = vdupq_n_u32(0);
  vtmp0123xCDEF.val[2] = vdupq_n_u32(0);
  vtmp0123xCDEF.val[3] = vdupq_n_u32(0);

  do {
    // NC main loop multiple of 16
    const uint32_t* w0 = weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(bias != NULL) {
        uint32x4_t vb0 = vld1q_u32(bias); bias += 4;
        uint32x4_t vb4 = vld1q_u32(bias); bias += 4;
        uint32x4_t vb8 = vld1q_u32(bias); bias += 4;
        uint32x4_t vb12 = vld1q_u32(bias); bias += 4;
        vst1q_u32(packed_weights, vb0); packed_weights += 4;
        vst1q_u32(packed_weights, vb4); packed_weights += 4;
        vst1q_u32(packed_weights, vb8); packed_weights += 4;
        vst1q_u32(packed_weights, vb12); packed_weights += 4;
      } else {
        const uint32x4_t vzero = vmovq_n_u32(0);
        vst1q_u32(packed_weights, vzero); packed_weights += 4;
        vst1q_u32(packed_weights, vzero); packed_weights += 4;
        vst1q_u32(packed_weights, vzero); packed_weights += 4;
        vst1q_u32(packed_weights, vzero); packed_weights += 4;
      }

      const uint32_t* w1 = w0 + kc;
      const uint32_t* w2 = w1 + kc;
      const uint32_t* w3 = w2 + kc;
      const uint32_t* w4 = w3 + kc;
      const uint32_t* w5 = w4 + kc;
      const uint32_t* w6 = w5 + kc;
      const uint32_t* w7 = w6 + kc;
      const uint32_t* w8 = w7 + kc;
      const uint32_t* w9 = w8 + kc;
      const uint32_t* w10 = w9 + kc;
      const uint32_t* w11 = w10 + kc;
      const uint32_t* w12 = w11 + kc;
      const uint32_t* w13 = w12 + kc;
      const uint32_t* w14 = w13 + kc;
      const uint32_t* w15 = w14 + kc;
      xnn_prefetch_to_l1((const int8_t*) w0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) w1);
      xnn_prefetch_to_l1((const int8_t*) w1 + 64);
      xnn_prefetch_to_l1((const int8_t*) w2);
      xnn_prefetch_to_l1((const int8_t*) w2 + 64);
      xnn_prefetch_to_l1((const int8_t*) w3);
      xnn_prefetch_to_l1((const int8_t*) w3 + 64);
      xnn_prefetch_to_l1((const int8_t*) w4);
      xnn_prefetch_to_l1((const int8_t*) w4 + 64);
      xnn_prefetch_to_l1((const int8_t*) w5);
      xnn_prefetch_to_l1((const int8_t*) w5 + 64);
      xnn_prefetch_to_l1((const int8_t*) w6);
      xnn_prefetch_to_l1((const int8_t*) w6 + 64);
      xnn_prefetch_to_l1((const int8_t*) w7);
      xnn_prefetch_to_l1((const int8_t*) w7 + 64);
      xnn_prefetch_to_l1((const int8_t*) w8);
      xnn_prefetch_to_l1((const int8_t*) w8 + 64);
      xnn_prefetch_to_l1((const int8_t*) w9);
      xnn_prefetch_to_l1((const int8_t*) w9 + 64);
      xnn_prefetch_to_l1((const int8_t*) w10);
      xnn_prefetch_to_l1((const int8_t*) w10 + 64);
      xnn_prefetch_to_l1((const int8_t*) w11);
      xnn_prefetch_to_l1((const int8_t*) w11 + 64);
      xnn_prefetch_to_l1((const int8_t*) w12);
      xnn_prefetch_to_l1((const int8_t*) w12 + 64);
      xnn_prefetch_to_l1((const int8_t*) w13);
      xnn_prefetch_to_l1((const int8_t*) w13 + 64);
      xnn_prefetch_to_l1((const int8_t*) w14);
      xnn_prefetch_to_l1((const int8_t*) w14 + 64);
      xnn_prefetch_to_l1((const int8_t*) w15);
      xnn_prefetch_to_l1((const int8_t*) w15 + 64);

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        vtmp0123x0123 = vld4q_lane_u32(w0, vtmp0123x0123, 0); w0 += 4;
        vtmp0123x0123 = vld4q_lane_u32(w1, vtmp0123x0123, 1); w1 += 4;
        vtmp0123x0123 = vld4q_lane_u32(w2, vtmp0123x0123, 2); w2 += 4;
        vtmp0123x0123 = vld4q_lane_u32(w3, vtmp0123x0123, 3); w3 += 4;
        vtmp0123x4567 = vld4q_lane_u32(w4, vtmp0123x4567, 0); w4 += 4;
        vtmp0123x4567 = vld4q_lane_u32(w5, vtmp0123x4567, 1); w5 += 4;
        vtmp0123x4567 = vld4q_lane_u32(w6, vtmp0123x4567, 2); w6 += 4;
        vtmp0123x4567 = vld4q_lane_u32(w7, vtmp0123x4567, 3); w7 += 4;
        vtmp0123x89AB = vld4q_lane_u32(w8, vtmp0123x89AB, 0); w8 += 4;
        vtmp0123x89AB = vld4q_lane_u32(w9, vtmp0123x89AB, 1); w9 += 4;
        vtmp0123x89AB = vld4q_lane_u32(w10, vtmp0123x89AB, 2); w10 += 4;
        vtmp0123x89AB = vld4q_lane_u32(w11, vtmp0123x89AB, 3); w11 += 4;
        vtmp0123xCDEF = vld4q_lane_u32(w12, vtmp0123xCDEF, 0); w12 += 4;
        vtmp0123xCDEF = vld4q_lane_u32(w13, vtmp0123xCDEF, 1); w13 += 4;
        vtmp0123xCDEF = vld4q_lane_u32(w14, vtmp0123xCDEF, 2); w14 += 4;
        vtmp0123xCDEF = vld4q_lane_u32(w15, vtmp0123xCDEF, 3); w15 += 4;
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        xnn_prefetch_to_l1((const int8_t*) w15 + 128);
        vst1q_u32(packed_weights, vtmp0123x0123.val[0]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x4567.val[0]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x89AB.val[0]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123xCDEF.val[0]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x0123.val[1]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x4567.val[1]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x89AB.val[1]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123xCDEF.val[1]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x0123.val[2]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x4567.val[2]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x89AB.val[2]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123xCDEF.val[2]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x0123.val[3]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x4567.val[3]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x89AB.val[3]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123xCDEF.val[3]); packed_weights += 4;
      }

      // KC remainder of 1..3
      // Same as main loop but ld1, ld2 or ld3
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          // KC remainder of 1
          case 1:
          {
            uint32x4_t vtmp0x0123 = vdupq_n_u32(0);
            uint32x4_t vtmp0x4567 = vdupq_n_u32(0);
            uint32x4_t vtmp0x89AB = vdupq_n_u32(0);
            uint32x4_t vtmp0xCDEF = vdupq_n_u32(0);
            vtmp0x0123 = vld1q_lane_u32(w0, vtmp0x0123, 0); w0 += 1;
            vtmp0x0123 = vld1q_lane_u32(w1, vtmp0x0123, 1); w1 += 1;
            vtmp0x0123 = vld1q_lane_u32(w2, vtmp0x0123, 2); w2 += 1;
            vtmp0x0123 = vld1q_lane_u32(w3, vtmp0x0123, 3); w3 += 1;
            vtmp0x4567 = vld1q_lane_u32(w4, vtmp0x4567, 0); w4 += 1;
            vtmp0x4567 = vld1q_lane_u32(w5, vtmp0x4567, 1); w5 += 1;
            vtmp0x4567 = vld1q_lane_u32(w6, vtmp0x4567, 2); w6 += 1;
            vtmp0x4567 = vld1q_lane_u32(w7, vtmp0x4567, 3); w7 += 1;
            vtmp0x89AB = vld1q_lane_u32(w8, vtmp0x89AB, 0); w8 += 1;
            vtmp0x89AB = vld1q_lane_u32(w9, vtmp0x89AB, 1); w9 += 1;
            vtmp0x89AB = vld1q_lane_u32(w10, vtmp0x89AB, 2); w10 += 1;
            vtmp0x89AB = vld1q_lane_u32(w11, vtmp0x89AB, 3); w11 += 1;
            vtmp0xCDEF = vld1q_lane_u32(w12, vtmp0xCDEF, 0); w12 += 1;
            vtmp0xCDEF = vld1q_lane_u32(w13, vtmp0xCDEF, 1); w13 += 1;
            vtmp0xCDEF = vld1q_lane_u32(w14, vtmp0xCDEF, 2); w14 += 1;
            vtmp0xCDEF = vld1q_lane_u32(w15, vtmp0xCDEF, 3); w15 += 1;
            vst1q_u32(packed_weights, vtmp0x0123); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp0x4567); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp0x89AB); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp0xCDEF); packed_weights += 4;
            break;
          }
          // KC remainder of 2
          case 2:
          {
            uint32x4x2_t vtmp01x0123;
            vtmp01x0123.val[0] = vdupq_n_u32(0);
            vtmp01x0123.val[1] = vdupq_n_u32(0);
            uint32x4x2_t vtmp01x4567;
            vtmp01x4567.val[0] = vdupq_n_u32(0);
            vtmp01x4567.val[1] = vdupq_n_u32(0);
            uint32x4x2_t vtmp01x89AB;
            vtmp01x89AB.val[0] = vdupq_n_u32(0);
            vtmp01x89AB.val[1] = vdupq_n_u32(0);
            uint32x4x2_t vtmp01xCDEF;
            vtmp01xCDEF.val[0] = vdupq_n_u32(0);
            vtmp01xCDEF.val[1] = vdupq_n_u32(0);
            vtmp01x0123 = vld2q_lane_u32(w0, vtmp01x0123, 0); w0 += 2;
            vtmp01x0123 = vld2q_lane_u32(w1, vtmp01x0123, 1); w1 += 2;
            vtmp01x0123 = vld2q_lane_u32(w2, vtmp01x0123, 2); w2 += 2;
            vtmp01x0123 = vld2q_lane_u32(w3, vtmp01x0123, 3); w3 += 2;
            vtmp01x4567 = vld2q_lane_u32(w4, vtmp01x4567, 0); w4 += 2;
            vtmp01x4567 = vld2q_lane_u32(w5, vtmp01x4567, 1); w5 += 2;
            vtmp01x4567 = vld2q_lane_u32(w6, vtmp01x4567, 2); w6 += 2;
            vtmp01x4567 = vld2q_lane_u32(w7, vtmp01x4567, 3); w7 += 2;
            vtmp01x89AB = vld2q_lane_u32(w8, vtmp01x89AB, 0); w8 += 2;
            vtmp01x89AB = vld2q_lane_u32(w9, vtmp01x89AB, 1); w9 += 2;
            vtmp01x89AB = vld2q_lane_u32(w10, vtmp01x89AB, 2); w10 += 2;
            vtmp01x89AB = vld2q_lane_u32(w11, vtmp01x89AB, 3); w11 += 2;
            vtmp01xCDEF = vld2q_lane_u32(w12, vtmp01xCDEF, 0); w12 += 2;
            vtmp01xCDEF = vld2q_lane_u32(w13, vtmp01xCDEF, 1); w13 += 2;
            vtmp01xCDEF = vld2q_lane_u32(w14, vtmp01xCDEF, 2); w14 += 2;
            vtmp01xCDEF = vld2q_lane_u32(w15, vtmp01xCDEF, 3); w15 += 2;
            vst1q_u32(packed_weights, vtmp01x0123.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01x4567.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01x89AB.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01xCDEF.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01x0123.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01x4567.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01x89AB.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01xCDEF.val[1]); packed_weights += 4;
            break;
          }
          // KC remainder of 3
          case 3:
          {
            uint32x4x3_t vtmp012x0123;
            vtmp012x0123.val[0] = vdupq_n_u32(0);
            vtmp012x0123.val[1] = vdupq_n_u32(0);
            vtmp012x0123.val[2] = vdupq_n_u32(0);
            uint32x4x3_t vtmp012x4567;
            vtmp012x4567.val[0] = vdupq_n_u32(0);
            vtmp012x4567.val[1] = vdupq_n_u32(0);
            vtmp012x4567.val[2] = vdupq_n_u32(0);
            uint32x4x3_t vtmp012x89AB;
            vtmp012x89AB.val[0] = vdupq_n_u32(0);
            vtmp012x89AB.val[1] = vdupq_n_u32(0);
            vtmp012x89AB.val[2] = vdupq_n_u32(0);
            uint32x4x3_t vtmp012xCDEF;
            vtmp012xCDEF.val[0] = vdupq_n_u32(0);
            vtmp012xCDEF.val[1] = vdupq_n_u32(0);
            vtmp012xCDEF.val[2] = vdupq_n_u32(0);
            vtmp012x0123 = vld3q_lane_u32(w0, vtmp012x0123, 0); w0 += 3;
            vtmp012x0123 = vld3q_lane_u32(w1, vtmp012x0123, 1); w1 += 3;
            vtmp012x0123 = vld3q_lane_u32(w2, vtmp012x0123, 2); w2 += 3;
            vtmp012x0123 = vld3q_lane_u32(w3, vtmp012x0123, 3); w3 += 3;
            vtmp012x4567 = vld3q_lane_u32(w4, vtmp012x4567, 0); w4 += 3;
            vtmp012x4567 = vld3q_lane_u32(w5, vtmp012x4567, 1); w5 += 3;
            vtmp012x4567 = vld3q_lane_u32(w6, vtmp012x4567, 2); w6 += 3;
            vtmp012x4567 = vld3q_lane_u32(w7, vtmp012x4567, 3); w7 += 3;
            vtmp012x89AB = vld3q_lane_u32(w8, vtmp012x89AB, 0); w8 += 3;
            vtmp012x89AB = vld3q_lane_u32(w9, vtmp012x89AB, 1); w9 += 3;
            vtmp012x89AB = vld3q_lane_u32(w10, vtmp012x89AB, 2); w10 += 3;
            vtmp012x89AB = vld3q_lane_u32(w11, vtmp012x89AB, 3); w11 += 3;
            vtmp012xCDEF = vld3q_lane_u32(w12, vtmp012xCDEF, 0); w12 += 3;
            vtmp012xCDEF = vld3q_lane_u32(w13, vtmp012xCDEF, 1); w13 += 3;
            vtmp012xCDEF = vld3q_lane_u32(w14, vtmp012xCDEF, 2); w14 += 3;
            vtmp012xCDEF = vld3q_lane_u32(w15, vtmp012xCDEF, 3); w15 += 3;
            vst1q_u32(packed_weights, vtmp012x0123.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x4567.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x89AB.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012xCDEF.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x0123.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x4567.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x89AB.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012xCDEF.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x0123.val[2]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x4567.val[2]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x89AB.val[2]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012xCDEF.val[2]); packed_weights += 4;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      if XNN_LIKELY(bias != NULL) {
        size_t nb = n;
        do {
          *packed_weights++  = *bias++;
        } while (--nb != 0);
        packed_weights += (16 - n);
      } else {
        const uint32x4_t vzero = vmovq_n_u32(0);
        vst1q_u32(packed_weights, vzero); packed_weights += 4;
        vst1q_u32(packed_weights, vzero); packed_weights += 4;
        vst1q_u32(packed_weights, vzero); packed_weights += 4;
        vst1q_u32(packed_weights, vzero); packed_weights += 4;
      }

      // NR remainder has less than 16 rows so last row is not loaded
      const uint32_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint32_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint32_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint32_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint32_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint32_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint32_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint32_t* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint32_t* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint32_t* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint32_t* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint32_t* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint32_t* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint32_t* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        vtmp0123x0123 = vld4q_lane_u32(w0, vtmp0123x0123, 0); w0 += 4;
        vtmp0123x0123 = vld4q_lane_u32(w1, vtmp0123x0123, 1); w1 += 4;
        vtmp0123x0123 = vld4q_lane_u32(w2, vtmp0123x0123, 2); w2 += 4;
        vtmp0123x0123 = vld4q_lane_u32(w3, vtmp0123x0123, 3); w3 += 4;
        vtmp0123x4567 = vld4q_lane_u32(w4, vtmp0123x4567, 0); w4 += 4;
        vtmp0123x4567 = vld4q_lane_u32(w5, vtmp0123x4567, 1); w5 += 4;
        vtmp0123x4567 = vld4q_lane_u32(w6, vtmp0123x4567, 2); w6 += 4;
        vtmp0123x4567 = vld4q_lane_u32(w7, vtmp0123x4567, 3); w7 += 4;
        vtmp0123x89AB = vld4q_lane_u32(w8, vtmp0123x89AB, 0); w8 += 4;
        vtmp0123x89AB = vld4q_lane_u32(w9, vtmp0123x89AB, 1); w9 += 4;
        vtmp0123x89AB = vld4q_lane_u32(w10, vtmp0123x89AB, 2); w10 += 4;
        vtmp0123x89AB = vld4q_lane_u32(w11, vtmp0123x89AB, 3); w11 += 4;
        vtmp0123xCDEF = vld4q_lane_u32(w12, vtmp0123xCDEF, 0); w12 += 4;
        vtmp0123xCDEF = vld4q_lane_u32(w13, vtmp0123xCDEF, 1); w13 += 4;
        vtmp0123xCDEF = vld4q_lane_u32(w14, vtmp0123xCDEF, 2); w14 += 4;
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        vst1q_u32(packed_weights, vtmp0123x0123.val[0]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x4567.val[0]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x89AB.val[0]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123xCDEF.val[0]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x0123.val[1]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x4567.val[1]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x89AB.val[1]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123xCDEF.val[1]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x0123.val[2]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x4567.val[2]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x89AB.val[2]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123xCDEF.val[2]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x0123.val[3]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x4567.val[3]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123x89AB.val[3]); packed_weights += 4;
        vst1q_u32(packed_weights, vtmp0123xCDEF.val[3]); packed_weights += 4;
      }

      // KC remainder of 1..3
      // Same as main loop but ld1, ld2 or ld3
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          // KC remainder of 1
          case 1:
          {
            uint32x4_t vtmp0x0123 = vdupq_n_u32(0);
            uint32x4_t vtmp0x4567 = vdupq_n_u32(0);
            uint32x4_t vtmp0x89AB = vdupq_n_u32(0);
            uint32x4_t vtmp0xCDEF = vdupq_n_u32(0);
            vtmp0x0123 = vld1q_lane_u32(w0, vtmp0x0123, 0);
            vtmp0x0123 = vld1q_lane_u32(w1, vtmp0x0123, 1);
            vtmp0x0123 = vld1q_lane_u32(w2, vtmp0x0123, 2);
            vtmp0x0123 = vld1q_lane_u32(w3, vtmp0x0123, 3);
            vtmp0x4567 = vld1q_lane_u32(w4, vtmp0x4567, 0);
            vtmp0x4567 = vld1q_lane_u32(w5, vtmp0x4567, 1);
            vtmp0x4567 = vld1q_lane_u32(w6, vtmp0x4567, 2);
            vtmp0x4567 = vld1q_lane_u32(w7, vtmp0x4567, 3);
            vtmp0x89AB = vld1q_lane_u32(w8, vtmp0x89AB, 0);
            vtmp0x89AB = vld1q_lane_u32(w9, vtmp0x89AB, 1);
            vtmp0x89AB = vld1q_lane_u32(w10, vtmp0x89AB, 2);
            vtmp0x89AB = vld1q_lane_u32(w11, vtmp0x89AB, 3);
            vtmp0xCDEF = vld1q_lane_u32(w12, vtmp0xCDEF, 0);
            vtmp0xCDEF = vld1q_lane_u32(w13, vtmp0xCDEF, 1);
            vtmp0xCDEF = vld1q_lane_u32(w14, vtmp0xCDEF, 2);
            vst1q_u32(packed_weights, vtmp0x0123); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp0x4567); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp0x89AB); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp0xCDEF); packed_weights += 4;
            break;
          }
          // KC remainder of 2
          case 2:
          {
            uint32x4x2_t vtmp01x0123;
            vtmp01x0123.val[0] = vdupq_n_u32(0);
            vtmp01x0123.val[1] = vdupq_n_u32(0);
            uint32x4x2_t vtmp01x4567;
            vtmp01x4567.val[0] = vdupq_n_u32(0);
            vtmp01x4567.val[1] = vdupq_n_u32(0);
            uint32x4x2_t vtmp01x89AB;
            vtmp01x89AB.val[0] = vdupq_n_u32(0);
            vtmp01x89AB.val[1] = vdupq_n_u32(0);
            uint32x4x2_t vtmp01xCDEF;
            vtmp01xCDEF.val[0] = vdupq_n_u32(0);
            vtmp01xCDEF.val[1] = vdupq_n_u32(0);
            vtmp01x0123 = vld2q_lane_u32(w0, vtmp01x0123, 0);
            vtmp01x0123 = vld2q_lane_u32(w1, vtmp01x0123, 1);
            vtmp01x0123 = vld2q_lane_u32(w2, vtmp01x0123, 2);
            vtmp01x0123 = vld2q_lane_u32(w3, vtmp01x0123, 3);
            vtmp01x4567 = vld2q_lane_u32(w4, vtmp01x4567, 0);
            vtmp01x4567 = vld2q_lane_u32(w5, vtmp01x4567, 1);
            vtmp01x4567 = vld2q_lane_u32(w6, vtmp01x4567, 2);
            vtmp01x4567 = vld2q_lane_u32(w7, vtmp01x4567, 3);
            vtmp01x89AB = vld2q_lane_u32(w8, vtmp01x89AB, 0);
            vtmp01x89AB = vld2q_lane_u32(w9, vtmp01x89AB, 1);
            vtmp01x89AB = vld2q_lane_u32(w10, vtmp01x89AB, 2);
            vtmp01x89AB = vld2q_lane_u32(w11, vtmp01x89AB, 3);
            vtmp01xCDEF = vld2q_lane_u32(w12, vtmp01xCDEF, 0);
            vtmp01xCDEF = vld2q_lane_u32(w13, vtmp01xCDEF, 1);
            vtmp01xCDEF = vld2q_lane_u32(w14, vtmp01xCDEF, 2);
            vst1q_u32(packed_weights, vtmp01x0123.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01x4567.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01x89AB.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01xCDEF.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01x0123.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01x4567.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01x89AB.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp01xCDEF.val[1]); packed_weights += 4;
            break;
          }
          // KC remainder of 3
          case 3:
          {
            uint32x4x3_t vtmp012x0123;
            vtmp012x0123.val[0] = vdupq_n_u32(0);
            vtmp012x0123.val[1] = vdupq_n_u32(0);
            vtmp012x0123.val[2] = vdupq_n_u32(0);
            uint32x4x3_t vtmp012x4567;
            vtmp012x4567.val[0] = vdupq_n_u32(0);
            vtmp012x4567.val[1] = vdupq_n_u32(0);
            vtmp012x4567.val[2] = vdupq_n_u32(0);
            uint32x4x3_t vtmp012x89AB;
            vtmp012x89AB.val[0] = vdupq_n_u32(0);
            vtmp012x89AB.val[1] = vdupq_n_u32(0);
            vtmp012x89AB.val[2] = vdupq_n_u32(0);
            uint32x4x3_t vtmp012xCDEF;
            vtmp012xCDEF.val[0] = vdupq_n_u32(0);
            vtmp012xCDEF.val[1] = vdupq_n_u32(0);
            vtmp012xCDEF.val[2] = vdupq_n_u32(0);
            vtmp012x0123 = vld3q_lane_u32(w0, vtmp012x0123, 0);
            vtmp012x0123 = vld3q_lane_u32(w1, vtmp012x0123, 1);
            vtmp012x0123 = vld3q_lane_u32(w2, vtmp012x0123, 2);
            vtmp012x0123 = vld3q_lane_u32(w3, vtmp012x0123, 3);
            vtmp012x4567 = vld3q_lane_u32(w4, vtmp012x4567, 0);
            vtmp012x4567 = vld3q_lane_u32(w5, vtmp012x4567, 1);
            vtmp012x4567 = vld3q_lane_u32(w6, vtmp012x4567, 2);
            vtmp012x4567 = vld3q_lane_u32(w7, vtmp012x4567, 3);
            vtmp012x89AB = vld3q_lane_u32(w8, vtmp012x89AB, 0);
            vtmp012x89AB = vld3q_lane_u32(w9, vtmp012x89AB, 1);
            vtmp012x89AB = vld3q_lane_u32(w10, vtmp012x89AB, 2);
            vtmp012x89AB = vld3q_lane_u32(w11, vtmp012x89AB, 3);
            vtmp012xCDEF = vld3q_lane_u32(w12, vtmp012xCDEF, 0);
            vtmp012xCDEF = vld3q_lane_u32(w13, vtmp012xCDEF, 1);
            vtmp012xCDEF = vld3q_lane_u32(w14, vtmp012xCDEF, 2);
            vst1q_u32(packed_weights, vtmp012x0123.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x4567.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x89AB.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012xCDEF.val[0]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x0123.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x4567.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x89AB.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012xCDEF.val[1]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x0123.val[2]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x4567.val[2]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012x89AB.val[2]); packed_weights += 4;
            vst1q_u32(packed_weights, vtmp012xCDEF.val[2]); packed_weights += 4;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
    }

    weights += nc * kc;
  } while (--g != 0);
}
