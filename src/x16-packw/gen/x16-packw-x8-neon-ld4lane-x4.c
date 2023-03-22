// Auto-generated file. Do not edit!
//   Template: src/x16-packw/neon.c.in
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

#include <xnnpack/packw.h>


void xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* weights,
  const uint16_t* bias,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 8);   // This kernel is for NR=8
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint16x4x4_t vtmp0123x0123;
  vtmp0123x0123.val[0] = vdup_n_u16(0);
  vtmp0123x0123.val[1] = vdup_n_u16(0);
  vtmp0123x0123.val[2] = vdup_n_u16(0);
  vtmp0123x0123.val[3] = vdup_n_u16(0);
  uint16x4x4_t vtmp0123x4567;
  vtmp0123x4567.val[0] = vdup_n_u16(0);
  vtmp0123x4567.val[1] = vdup_n_u16(0);
  vtmp0123x4567.val[2] = vdup_n_u16(0);
  vtmp0123x4567.val[3] = vdup_n_u16(0);

  do {
    // NC main loop multiple of 8
    const uint16_t* w0 = weights;
    size_t n = nc;

    for (; n >= 8; n -= 8) {
      if XNN_LIKELY(bias != NULL) {
        uint16x4_t vb0 = vld1_u16(bias); bias += 4;
        uint16x4_t vb4 = vld1_u16(bias); bias += 4;
        vst1_u16(packed_weights, vb0); packed_weights += 4;
        vst1_u16(packed_weights, vb4); packed_weights += 4;
      } else {
        packed_weights += 8;
      }

      const uint16_t* w1 = w0 + kc;
      const uint16_t* w2 = w1 + kc;
      const uint16_t* w3 = w2 + kc;
      const uint16_t* w4 = w3 + kc;
      const uint16_t* w5 = w4 + kc;
      const uint16_t* w6 = w5 + kc;
      const uint16_t* w7 = w6 + kc;

      // KC main loop multiple of 8x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        vtmp0123x0123 = vld4_lane_u16(w0, vtmp0123x0123, 0); w0 += 4;
        vtmp0123x4567 = vld4_lane_u16(w4, vtmp0123x4567, 0); w4 += 4;
        vtmp0123x0123 = vld4_lane_u16(w1, vtmp0123x0123, 1); w1 += 4;
        vtmp0123x4567 = vld4_lane_u16(w5, vtmp0123x4567, 1); w5 += 4;
        vtmp0123x0123 = vld4_lane_u16(w2, vtmp0123x0123, 2); w2 += 4;
        vtmp0123x4567 = vld4_lane_u16(w6, vtmp0123x4567, 2); w6 += 4;
        vtmp0123x0123 = vld4_lane_u16(w3, vtmp0123x0123, 3); w3 += 4;
        vtmp0123x4567 = vld4_lane_u16(w7, vtmp0123x4567, 3); w7 += 4;
        vst1_u16(packed_weights, vtmp0123x0123.val[0]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x4567.val[0]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x0123.val[1]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x4567.val[1]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x0123.val[2]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x4567.val[2]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x0123.val[3]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x4567.val[3]); packed_weights += 4;
      }

      // KC remainder of 1..3
      // Same as main loop but ld1, ld2 or ld3
      if XNN_UNLIKELY(k != 0) {
        switch (k) {
          // KC remainder of 8x1
          case 1:
          {
            uint16x4_t vtmp0x0123 = vdup_n_u16(0);
            uint16x4_t vtmp0x4567 = vdup_n_u16(0);

            vtmp0x0123 = vld1_lane_u16(w0, vtmp0x0123, 0); w0 += 1;
            vtmp0x4567 = vld1_lane_u16(w4, vtmp0x4567, 0); w4 += 1;
            vtmp0x0123 = vld1_lane_u16(w1, vtmp0x0123, 1); w1 += 1;
            vtmp0x4567 = vld1_lane_u16(w5, vtmp0x4567, 1); w5 += 1;
            vtmp0x0123 = vld1_lane_u16(w2, vtmp0x0123, 2); w2 += 1;
            vtmp0x4567 = vld1_lane_u16(w6, vtmp0x4567, 2); w6 += 1;
            vtmp0x0123 = vld1_lane_u16(w3, vtmp0x0123, 3); w3 += 1;
            vtmp0x4567 = vld1_lane_u16(w7, vtmp0x4567, 3); w7 += 1;

            vst1_u16(packed_weights, vtmp0x0123); packed_weights += 4;
            vst1_u16(packed_weights, vtmp0x4567); packed_weights += 4;
            break;
          }
          // KC remainder of 8x2
          case 2:
          {
            uint16x4x2_t vtmp01x0123;
            vtmp01x0123.val[0] = vdup_n_u16(0);
            vtmp01x0123.val[1] = vdup_n_u16(0);
            uint16x4x2_t vtmp01x4567;
            vtmp01x4567.val[0] = vdup_n_u16(0);
            vtmp01x4567.val[1] = vdup_n_u16(0);

            vtmp01x0123 = vld2_lane_u16(w0, vtmp01x0123, 0); w0 += 2;
            vtmp01x4567 = vld2_lane_u16(w4, vtmp01x4567, 0); w4 += 2;
            vtmp01x0123 = vld2_lane_u16(w1, vtmp01x0123, 1); w1 += 2;
            vtmp01x4567 = vld2_lane_u16(w5, vtmp01x4567, 1); w5 += 2;
            vtmp01x0123 = vld2_lane_u16(w2, vtmp01x0123, 2); w2 += 2;
            vtmp01x4567 = vld2_lane_u16(w6, vtmp01x4567, 2); w6 += 2;
            vtmp01x0123 = vld2_lane_u16(w3, vtmp01x0123, 3); w3 += 2;
            vtmp01x4567 = vld2_lane_u16(w7, vtmp01x4567, 3); w7 += 2;

            vst1_u16(packed_weights, vtmp01x0123.val[0]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp01x4567.val[0]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp01x0123.val[1]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp01x4567.val[1]); packed_weights += 4;
            break;
          }
          // KC remainder of 8x3
          case 3:
          {
            uint16x4x3_t vtmp012x0123;
            vtmp012x0123.val[0] = vdup_n_u16(0);
            vtmp012x0123.val[1] = vdup_n_u16(0);
            vtmp012x0123.val[2] = vdup_n_u16(0);
            uint16x4x3_t vtmp012x4567;
            vtmp012x4567.val[0] = vdup_n_u16(0);
            vtmp012x4567.val[1] = vdup_n_u16(0);
            vtmp012x4567.val[2] = vdup_n_u16(0);

            vtmp012x0123 = vld3_lane_u16(w0, vtmp012x0123, 0); w0 += 3;
            vtmp012x4567 = vld3_lane_u16(w4, vtmp012x4567, 0); w4 += 3;
            vtmp012x0123 = vld3_lane_u16(w1, vtmp012x0123, 1); w1 += 3;
            vtmp012x4567 = vld3_lane_u16(w5, vtmp012x4567, 1); w5 += 3;
            vtmp012x0123 = vld3_lane_u16(w2, vtmp012x0123, 2); w2 += 3;
            vtmp012x4567 = vld3_lane_u16(w6, vtmp012x4567, 2); w6 += 3;
            vtmp012x0123 = vld3_lane_u16(w3, vtmp012x0123, 3); w3 += 3;
            vtmp012x4567 = vld3_lane_u16(w7, vtmp012x4567, 3); w7 += 3;

            vst1_u16(packed_weights, vtmp012x0123.val[0]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp012x4567.val[0]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp012x0123.val[1]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp012x4567.val[1]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp012x0123.val[2]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp012x4567.val[2]); packed_weights += 4;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 7);
      if XNN_LIKELY(bias != NULL) {
        size_t nb = n;
        do {
          *packed_weights++  = *bias++;
        } while (--nb != 0);
        packed_weights += (8 - n);
      } else {
        packed_weights += 8;
      }

      // NR remainder has less than 8 rows so last row is not loaded
      const uint16_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint16_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint16_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint16_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint16_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint16_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }

      // KC main loop multiple of 8x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        vtmp0123x0123 = vld4_lane_u16(w0, vtmp0123x0123, 0); w0 += 4;
        vtmp0123x4567 = vld4_lane_u16(w4, vtmp0123x4567, 0); w4 += 4;
        vtmp0123x0123 = vld4_lane_u16(w1, vtmp0123x0123, 1); w1 += 4;
        vtmp0123x4567 = vld4_lane_u16(w5, vtmp0123x4567, 1); w5 += 4;
        vtmp0123x0123 = vld4_lane_u16(w2, vtmp0123x0123, 2); w2 += 4;
        vtmp0123x4567 = vld4_lane_u16(w6, vtmp0123x4567, 2); w6 += 4;
        vtmp0123x0123 = vld4_lane_u16(w3, vtmp0123x0123, 3); w3 += 4;
        vst1_u16(packed_weights, vtmp0123x0123.val[0]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x4567.val[0]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x0123.val[1]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x4567.val[1]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x0123.val[2]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x4567.val[2]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x0123.val[3]); packed_weights += 4;
        vst1_u16(packed_weights, vtmp0123x4567.val[3]); packed_weights += 4;
      }

      // KC remainder of 1..3
      // Same as main loop but ld1, ld2 or ld3
      if XNN_UNLIKELY(k != 0) {
        switch (k) {
          // KC remainder of 8x1
          case 1:
          {
            uint16x4_t vtmp0x0123 = vdup_n_u16(0);
            uint16x4_t vtmp0x4567 = vdup_n_u16(0);

            vtmp0x0123 = vld1_lane_u16(w0, vtmp0x0123, 0); w0 += 1;
            vtmp0x4567 = vld1_lane_u16(w4, vtmp0x4567, 0); w4 += 1;
            vtmp0x0123 = vld1_lane_u16(w1, vtmp0x0123, 1); w1 += 1;
            vtmp0x4567 = vld1_lane_u16(w5, vtmp0x4567, 1); w5 += 1;
            vtmp0x0123 = vld1_lane_u16(w2, vtmp0x0123, 2); w2 += 1;
            vtmp0x4567 = vld1_lane_u16(w6, vtmp0x4567, 2); w6 += 1;
            vtmp0x0123 = vld1_lane_u16(w3, vtmp0x0123, 3); w3 += 1;

            vst1_u16(packed_weights, vtmp0x0123); packed_weights += 4;
            vst1_u16(packed_weights, vtmp0x4567); packed_weights += 4;
            break;
          }
          // KC remainder of 8x2
          case 2:
          {
            uint16x4x2_t vtmp01x0123;
            vtmp01x0123.val[0] = vdup_n_u16(0);
            vtmp01x0123.val[1] = vdup_n_u16(0);
            uint16x4x2_t vtmp01x4567;
            vtmp01x4567.val[0] = vdup_n_u16(0);
            vtmp01x4567.val[1] = vdup_n_u16(0);

            vtmp01x0123 = vld2_lane_u16(w0, vtmp01x0123, 0); w0 += 2;
            vtmp01x4567 = vld2_lane_u16(w4, vtmp01x4567, 0); w4 += 2;
            vtmp01x0123 = vld2_lane_u16(w1, vtmp01x0123, 1); w1 += 2;
            vtmp01x4567 = vld2_lane_u16(w5, vtmp01x4567, 1); w5 += 2;
            vtmp01x0123 = vld2_lane_u16(w2, vtmp01x0123, 2); w2 += 2;
            vtmp01x4567 = vld2_lane_u16(w6, vtmp01x4567, 2); w6 += 2;
            vtmp01x0123 = vld2_lane_u16(w3, vtmp01x0123, 3); w3 += 2;

            vst1_u16(packed_weights, vtmp01x0123.val[0]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp01x4567.val[0]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp01x0123.val[1]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp01x4567.val[1]); packed_weights += 4;
            break;
          }
          // KC remainder of 8x3
          case 3:
          {
            uint16x4x3_t vtmp012x0123;
            vtmp012x0123.val[0] = vdup_n_u16(0);
            vtmp012x0123.val[1] = vdup_n_u16(0);
            vtmp012x0123.val[2] = vdup_n_u16(0);
            uint16x4x3_t vtmp012x4567;
            vtmp012x4567.val[0] = vdup_n_u16(0);
            vtmp012x4567.val[1] = vdup_n_u16(0);
            vtmp012x4567.val[2] = vdup_n_u16(0);

            vtmp012x0123 = vld3_lane_u16(w0, vtmp012x0123, 0); w0 += 3;
            vtmp012x4567 = vld3_lane_u16(w4, vtmp012x4567, 0); w4 += 3;
            vtmp012x0123 = vld3_lane_u16(w1, vtmp012x0123, 1); w1 += 3;
            vtmp012x4567 = vld3_lane_u16(w5, vtmp012x4567, 1); w5 += 3;
            vtmp012x0123 = vld3_lane_u16(w2, vtmp012x0123, 2); w2 += 3;
            vtmp012x4567 = vld3_lane_u16(w6, vtmp012x4567, 2); w6 += 3;
            vtmp012x0123 = vld3_lane_u16(w3, vtmp012x0123, 3); w3 += 3;

            vst1_u16(packed_weights, vtmp012x0123.val[0]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp012x4567.val[0]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp012x0123.val[1]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp012x4567.val[1]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp012x0123.val[2]); packed_weights += 4;
            vst1_u16(packed_weights, vtmp012x4567.val[2]); packed_weights += 4;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }

    weights += nc * kc;
  } while (--g != 0);
}
