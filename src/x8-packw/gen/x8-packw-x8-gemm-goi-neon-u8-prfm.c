// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/prefetch.h"

// Interleave 8-bit elements of two 64-bit vectors into a 128-bit vector.
XNN_INLINE static int8x16_t xnn_zip_s8(int8x8_t va, int8x8_t vb) {
  const int8x8x2_t vz = vzip_s8(va, vb);
  return vcombine_s8(vz.val[0], vz.val[1]);
}


void xnn_x8_packw_gemm_goi_ukernel_x8__neon_u8_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t n_stride,
  const int8_t* weights,
  const uint32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 8);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    const int8_t* wb = weights;
    size_t n = nc;
    // NC main loop multiple of 8
    for (; n >= 8; n -= 8) {
      const int8_t* w0 = wb;
      const int8_t* w1 = w0 + n_stride;
      const int8_t* w2 = w1 + n_stride;
      const int8_t* w3 = w2 + n_stride;
      const int8_t* w4 = w3 + n_stride;
      const int8_t* w5 = w4 + n_stride;
      const int8_t* w6 = w5 + n_stride;
      const int8_t* w7 = w6 + n_stride;

      uint32_t* packed_b = (uint32_t*) out;
      if XNN_LIKELY(b != NULL) {
        const uint32x4_t vb0 = vld1q_u32(b + 0);
        const uint32x4_t vb4 = vld1q_u32(b + 4);
        vst1q_u32(packed_b + 0, vb0);
        vst1q_u32(packed_b + 4, vb4);
        b += 8;
      } else {
        const uint32x4_t vzero = vdupq_n_u32(0);
        vst1q_u32(packed_b + 0, vzero);
        vst1q_u32(packed_b + 4, vzero);
      }
      out += 8 * sizeof(uint32_t);

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

      size_t k = kc;

      // KC loop / remainder of 8x8
      for (; k >= 8; k -= 8) {
        const int8x8_t v0 = vld1_s8(w0);
        w0 += 8;
        const int8x8_t v1 = vld1_s8(w1);
        w1 += 8;
        const int8x8_t v2 = vld1_s8(w2);
        w2 += 8;
        const int8x8_t v3 = vld1_s8(w3);
        w3 += 8;
        const int8x8_t v4 = vld1_s8(w4);
        w4 += 8;
        const int8x8_t v5 = vld1_s8(w5);
        w5 += 8;
        const int8x8_t v6 = vld1_s8(w6);
        w6 += 8;
        const int8x8_t v7 = vld1_s8(w7);
        w7 += 8;
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);

        const int8x16_t t0_0 = xnn_zip_s8(v0, v1);
        const int8x16_t t0_1 = xnn_zip_s8(v2, v3);
        const int8x16_t t0_2 = xnn_zip_s8(v4, v5);
        const int8x16_t t0_3 = xnn_zip_s8(v6, v7);

        const int16x8x2_t u0_0 = vzipq_s16(
            vreinterpretq_s16_s8(t0_0),
            vreinterpretq_s16_s8(t0_1));
        const int16x8x2_t u0_1 = vzipq_s16(
            vreinterpretq_s16_s8(t0_2),
            vreinterpretq_s16_s8(t0_3));

        const int32x4x2_t s0_0 = vzipq_s32(
            vreinterpretq_s32_s16(u0_0.val[0]),
            vreinterpretq_s32_s16(u0_1.val[0]));
        const int32x4x2_t s0_2 = vzipq_s32(
            vreinterpretq_s32_s16(u0_0.val[1]),
            vreinterpretq_s32_s16(u0_1.val[1]));

        vst1q_s8(out + 0, vreinterpretq_s8_s32(s0_0.val[0]));
        vst1q_s8(out + 16, vreinterpretq_s8_s32(s0_0.val[1]));
        vst1q_s8(out + 32, vreinterpretq_s8_s32(s0_2.val[0]));
        vst1q_s8(out + 48, vreinterpretq_s8_s32(s0_2.val[1]));
        out += 64;
      }

      // KC remainder of 1..7
      if XNN_UNLIKELY((k & 7) != 0) {
        if (k & 4) {
          int8x8x4_t vtmp0;
          vtmp0.val[0] = vdup_n_s8(0);
          vtmp0.val[1] = vdup_n_s8(0);
          vtmp0.val[2] = vdup_n_s8(0);
          vtmp0.val[3] = vdup_n_s8(0);
          vtmp0 = vld4_lane_s8(w0, vtmp0, 0);
          w0 += 4;
          vtmp0 = vld4_lane_s8(w1, vtmp0, 1);
          w1 += 4;
          vtmp0 = vld4_lane_s8(w2, vtmp0, 2);
          w2 += 4;
          vtmp0 = vld4_lane_s8(w3, vtmp0, 3);
          w3 += 4;
          vtmp0 = vld4_lane_s8(w4, vtmp0, 4);
          w4 += 4;
          vtmp0 = vld4_lane_s8(w5, vtmp0, 5);
          w5 += 4;
          vtmp0 = vld4_lane_s8(w6, vtmp0, 6);
          w6 += 4;
          vtmp0 = vld4_lane_s8(w7, vtmp0, 7);
          w7 += 4;
          vst1_s8(out + 0, vtmp0.val[0]);
          vst1_s8(out + 8, vtmp0.val[1]);
          vst1_s8(out + 16, vtmp0.val[2]);
          vst1_s8(out + 24, vtmp0.val[3]);
          out += 32;
        }
        if (k & 2) {
          int8x8x2_t vtmp0;
          vtmp0.val[0] = vdup_n_s8(0);
          vtmp0.val[1] = vdup_n_s8(0);
          vtmp0 = vld2_lane_s8(w0, vtmp0, 0);
          w0 += 2;
          vtmp0 = vld2_lane_s8(w1, vtmp0, 1);
          w1 += 2;
          vtmp0 = vld2_lane_s8(w2, vtmp0, 2);
          w2 += 2;
          vtmp0 = vld2_lane_s8(w3, vtmp0, 3);
          w3 += 2;
          vtmp0 = vld2_lane_s8(w4, vtmp0, 4);
          w4 += 2;
          vtmp0 = vld2_lane_s8(w5, vtmp0, 5);
          w5 += 2;
          vtmp0 = vld2_lane_s8(w6, vtmp0, 6);
          w6 += 2;
          vtmp0 = vld2_lane_s8(w7, vtmp0, 7);
          w7 += 2;
          vst1_s8(out + 0, vtmp0.val[0]);
          vst1_s8(out + 8, vtmp0.val[1]);
          out += 16;
        }
        if (k & 1) {
          int8x8_t vtmp0 = vdup_n_s8(0);
          vtmp0 = vld1_lane_s8(w0, vtmp0, 0);
          w0 += 1;
          vtmp0 = vld1_lane_s8(w1, vtmp0, 1);
          w1 += 1;
          vtmp0 = vld1_lane_s8(w2, vtmp0, 2);
          w2 += 1;
          vtmp0 = vld1_lane_s8(w3, vtmp0, 3);
          w3 += 1;
          vtmp0 = vld1_lane_s8(w4, vtmp0, 4);
          w4 += 1;
          vtmp0 = vld1_lane_s8(w5, vtmp0, 5);
          w5 += 1;
          vtmp0 = vld1_lane_s8(w6, vtmp0, 6);
          w6 += 1;
          vtmp0 = vld1_lane_s8(w7, vtmp0, 7);
          w7 += 1;
          vst1_s8(out + 0, vtmp0);
          out += 8;
        }
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      wb += 8 * n_stride;
    }
    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 7);
      const int8_t* w0 = wb;
      const int8_t* w1 = w0 + n_stride;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + n_stride;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int8_t* w3 = w2 + n_stride;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int8_t* w4 = w3 + n_stride;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int8_t* w5 = w4 + n_stride;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int8_t* w6 = w5 + n_stride;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int8_t* w7 = w6 + n_stride;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }

      uint32_t* packed_b = (uint32_t*) out;
      const uint32x4_t vzero = vdupq_n_u32(0);
      vst1q_u32(packed_b + 0, vzero);
      vst1q_u32(packed_b + 4, vzero);
      if XNN_LIKELY(b != NULL) {
        for (size_t nb = 0; nb < n; ++nb) {
          packed_b[nb] = b[nb];
        }
        b += n;
      }
      out += 8 * sizeof(uint32_t);

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

      size_t k = kc;

      // KC loop / remainder of 8x8
      for (; k >= 8; k -= 8) {
        const int8x8_t v0 = vld1_s8(w0);
        w0 += 8;
        const int8x8_t v1 = vld1_s8(w1);
        w1 += 8;
        const int8x8_t v2 = vld1_s8(w2);
        w2 += 8;
        const int8x8_t v3 = vld1_s8(w3);
        w3 += 8;
        const int8x8_t v4 = vld1_s8(w4);
        w4 += 8;
        const int8x8_t v5 = vld1_s8(w5);
        w5 += 8;
        const int8x8_t v6 = vld1_s8(w6);
        w6 += 8;
        const int8x8_t v7 = vld1_s8(w7);
        w7 += 8;
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);

        const int8x16_t t0_0 = xnn_zip_s8(v0, v1);
        const int8x16_t t0_1 = xnn_zip_s8(v2, v3);
        const int8x16_t t0_2 = xnn_zip_s8(v4, v5);
        const int8x16_t t0_3 = xnn_zip_s8(v6, v7);

        const int16x8x2_t u0_0 = vzipq_s16(
            vreinterpretq_s16_s8(t0_0),
            vreinterpretq_s16_s8(t0_1));
        const int16x8x2_t u0_1 = vzipq_s16(
            vreinterpretq_s16_s8(t0_2),
            vreinterpretq_s16_s8(t0_3));

        const int32x4x2_t s0_0 = vzipq_s32(
            vreinterpretq_s32_s16(u0_0.val[0]),
            vreinterpretq_s32_s16(u0_1.val[0]));
        const int32x4x2_t s0_2 = vzipq_s32(
            vreinterpretq_s32_s16(u0_0.val[1]),
            vreinterpretq_s32_s16(u0_1.val[1]));

        vst1q_s8(out + 0, vreinterpretq_s8_s32(s0_0.val[0]));
        vst1q_s8(out + 16, vreinterpretq_s8_s32(s0_0.val[1]));
        vst1q_s8(out + 32, vreinterpretq_s8_s32(s0_2.val[0]));
        vst1q_s8(out + 48, vreinterpretq_s8_s32(s0_2.val[1]));
        out += 64;
      }

      // KC remainder of 1..7
      if XNN_UNLIKELY((k & 7) != 0) {
        if (k & 4) {
          int8x8x4_t vtmp0;
          vtmp0.val[0] = vdup_n_s8(0);
          vtmp0.val[1] = vdup_n_s8(0);
          vtmp0.val[2] = vdup_n_s8(0);
          vtmp0.val[3] = vdup_n_s8(0);
          vtmp0 = vld4_lane_s8(w0, vtmp0, 0);
          w0 += 4;
          vtmp0 = vld4_lane_s8(w1, vtmp0, 1);
          w1 += 4;
          vtmp0 = vld4_lane_s8(w2, vtmp0, 2);
          w2 += 4;
          vtmp0 = vld4_lane_s8(w3, vtmp0, 3);
          w3 += 4;
          vtmp0 = vld4_lane_s8(w4, vtmp0, 4);
          w4 += 4;
          vtmp0 = vld4_lane_s8(w5, vtmp0, 5);
          w5 += 4;
          vtmp0 = vld4_lane_s8(w6, vtmp0, 6);
          w6 += 4;
          vtmp0 = vld4_lane_s8(w7, vtmp0, 7);
          w7 += 4;
          vst1_s8(out + 0, vtmp0.val[0]);
          vst1_s8(out + 8, vtmp0.val[1]);
          vst1_s8(out + 16, vtmp0.val[2]);
          vst1_s8(out + 24, vtmp0.val[3]);
          out += 32;
        }
        if (k & 2) {
          int8x8x2_t vtmp0;
          vtmp0.val[0] = vdup_n_s8(0);
          vtmp0.val[1] = vdup_n_s8(0);
          vtmp0 = vld2_lane_s8(w0, vtmp0, 0);
          w0 += 2;
          vtmp0 = vld2_lane_s8(w1, vtmp0, 1);
          w1 += 2;
          vtmp0 = vld2_lane_s8(w2, vtmp0, 2);
          w2 += 2;
          vtmp0 = vld2_lane_s8(w3, vtmp0, 3);
          w3 += 2;
          vtmp0 = vld2_lane_s8(w4, vtmp0, 4);
          w4 += 2;
          vtmp0 = vld2_lane_s8(w5, vtmp0, 5);
          w5 += 2;
          vtmp0 = vld2_lane_s8(w6, vtmp0, 6);
          w6 += 2;
          vtmp0 = vld2_lane_s8(w7, vtmp0, 7);
          w7 += 2;
          vst1_s8(out + 0, vtmp0.val[0]);
          vst1_s8(out + 8, vtmp0.val[1]);
          out += 16;
        }
        if (k & 1) {
          int8x8_t vtmp0 = vdup_n_s8(0);
          vtmp0 = vld1_lane_s8(w0, vtmp0, 0);
          w0 += 1;
          vtmp0 = vld1_lane_s8(w1, vtmp0, 1);
          w1 += 1;
          vtmp0 = vld1_lane_s8(w2, vtmp0, 2);
          w2 += 1;
          vtmp0 = vld1_lane_s8(w3, vtmp0, 3);
          w3 += 1;
          vtmp0 = vld1_lane_s8(w4, vtmp0, 4);
          w4 += 1;
          vtmp0 = vld1_lane_s8(w5, vtmp0, 5);
          w5 += 1;
          vtmp0 = vld1_lane_s8(w6, vtmp0, 6);
          w6 += 1;
          vtmp0 = vld1_lane_s8(w7, vtmp0, 7);
          w7 += 1;
          vst1_s8(out + 0, vtmp0);
          out += 8;
        }
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights += nc * n_stride;
  } while (--g != 0);
}
