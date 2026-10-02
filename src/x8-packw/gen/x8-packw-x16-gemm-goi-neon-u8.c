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

// Interleave 8-bit elements of two 64-bit vectors into a 128-bit vector.
XNN_INLINE static int8x16_t xnn_zip_s8(int8x8_t va, int8x8_t vb) {
  const int8x8x2_t vz = vzip_s8(va, vb);
  return vcombine_s8(vz.val[0], vz.val[1]);
}

// Transpose 64-bit elements of 2 rows.
// Returns [a0, b0] and [a1, b1].
XNN_INLINE static int8x16x2_t xnn_zip_s64(int8x16_t va, int8x16_t vb) {
  int8x16x2_t vr;
#if XNN_ARCH_ARM64
  vr.val[0] = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(va), vreinterpretq_s64_s8(vb)));
  vr.val[1] = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(va), vreinterpretq_s64_s8(vb)));
#else
  vr.val[0] = vcombine_s8(vget_low_s8(va), vget_low_s8(vb));
  vr.val[1] = vcombine_s8(vget_high_s8(va), vget_high_s8(vb));
#endif
  return vr;
}

void xnn_x8_packw_gemm_goi_ukernel_x16__neon_u8(
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
  assert(nr == 16);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    const int8_t* wb = weights;
    size_t n = nc;
    // NC main loop multiple of 16
    for (; n >= 16; n -= 16) {
      const int8_t* w0 = wb;
      const int8_t* w1 = w0 + n_stride;
      const int8_t* w2 = w1 + n_stride;
      const int8_t* w3 = w2 + n_stride;
      const int8_t* w4 = w3 + n_stride;
      const int8_t* w5 = w4 + n_stride;
      const int8_t* w6 = w5 + n_stride;
      const int8_t* w7 = w6 + n_stride;
      const int8_t* w8 = w7 + n_stride;
      const int8_t* w9 = w8 + n_stride;
      const int8_t* w10 = w9 + n_stride;
      const int8_t* w11 = w10 + n_stride;
      const int8_t* w12 = w11 + n_stride;
      const int8_t* w13 = w12 + n_stride;
      const int8_t* w14 = w13 + n_stride;
      const int8_t* w15 = w14 + n_stride;

      uint32_t* packed_b = (uint32_t*) out;
      if XNN_LIKELY(b != NULL) {
        const uint32x4_t vb0 = vld1q_u32(b + 0);
        const uint32x4_t vb4 = vld1q_u32(b + 4);
        const uint32x4_t vb8 = vld1q_u32(b + 8);
        const uint32x4_t vb12 = vld1q_u32(b + 12);
        vst1q_u32(packed_b + 0, vb0);
        vst1q_u32(packed_b + 4, vb4);
        vst1q_u32(packed_b + 8, vb8);
        vst1q_u32(packed_b + 12, vb12);
        b += 16;
      } else {
        const uint32x4_t vzero = vdupq_n_u32(0);
        vst1q_u32(packed_b + 0, vzero);
        vst1q_u32(packed_b + 4, vzero);
        vst1q_u32(packed_b + 8, vzero);
        vst1q_u32(packed_b + 12, vzero);
      }
      out += 16 * sizeof(uint32_t);


      size_t k = kc;

      // KC loop / remainder of 16x8
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
        const int8x8_t v8 = vld1_s8(w8);
        w8 += 8;
        const int8x8_t v9 = vld1_s8(w9);
        w9 += 8;
        const int8x8_t v10 = vld1_s8(w10);
        w10 += 8;
        const int8x8_t v11 = vld1_s8(w11);
        w11 += 8;
        const int8x8_t v12 = vld1_s8(w12);
        w12 += 8;
        const int8x8_t v13 = vld1_s8(w13);
        w13 += 8;
        const int8x8_t v14 = vld1_s8(w14);
        w14 += 8;
        const int8x8_t v15 = vld1_s8(w15);
        w15 += 8;

        const int8x16_t t8_0 = xnn_zip_s8(v8, v9);
        const int8x16_t t8_1 = xnn_zip_s8(v10, v11);
        const int8x16_t t8_2 = xnn_zip_s8(v12, v13);
        const int8x16_t t8_3 = xnn_zip_s8(v14, v15);

        const int16x8x2_t u8_0 = vzipq_s16(
            vreinterpretq_s16_s8(t8_0),
            vreinterpretq_s16_s8(t8_1));
        const int16x8x2_t u8_1 = vzipq_s16(
            vreinterpretq_s16_s8(t8_2),
            vreinterpretq_s16_s8(t8_3));

        const int32x4x2_t s8_0 = vzipq_s32(
            vreinterpretq_s32_s16(u8_0.val[0]),
            vreinterpretq_s32_s16(u8_1.val[0]));
        const int32x4x2_t s8_2 = vzipq_s32(
            vreinterpretq_s32_s16(u8_0.val[1]),
            vreinterpretq_s32_s16(u8_1.val[1]));

        const int8x16x2_t out0_0 = xnn_zip_s64(
            vreinterpretq_s8_s32(s0_0.val[0]),
            vreinterpretq_s8_s32(s8_0.val[0]));
        vst1q_s8(out + 0, out0_0.val[0]);
        vst1q_s8(out + 16, out0_0.val[1]);
        const int8x16x2_t out0_1 = xnn_zip_s64(
            vreinterpretq_s8_s32(s0_0.val[1]),
            vreinterpretq_s8_s32(s8_0.val[1]));
        vst1q_s8(out + 32, out0_1.val[0]);
        vst1q_s8(out + 48, out0_1.val[1]);
        const int8x16x2_t out0_2 = xnn_zip_s64(
            vreinterpretq_s8_s32(s0_2.val[0]),
            vreinterpretq_s8_s32(s8_2.val[0]));
        vst1q_s8(out + 64, out0_2.val[0]);
        vst1q_s8(out + 80, out0_2.val[1]);
        const int8x16x2_t out0_3 = xnn_zip_s64(
            vreinterpretq_s8_s32(s0_2.val[1]),
            vreinterpretq_s8_s32(s8_2.val[1]));
        vst1q_s8(out + 96, out0_3.val[0]);
        vst1q_s8(out + 112, out0_3.val[1]);
        out += 128;
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
          int8x8x4_t vtmp8;
          vtmp8.val[0] = vdup_n_s8(0);
          vtmp8.val[1] = vdup_n_s8(0);
          vtmp8.val[2] = vdup_n_s8(0);
          vtmp8.val[3] = vdup_n_s8(0);
          vtmp8 = vld4_lane_s8(w8, vtmp8, 0);
          w8 += 4;
          vtmp8 = vld4_lane_s8(w9, vtmp8, 1);
          w9 += 4;
          vtmp8 = vld4_lane_s8(w10, vtmp8, 2);
          w10 += 4;
          vtmp8 = vld4_lane_s8(w11, vtmp8, 3);
          w11 += 4;
          vtmp8 = vld4_lane_s8(w12, vtmp8, 4);
          w12 += 4;
          vtmp8 = vld4_lane_s8(w13, vtmp8, 5);
          w13 += 4;
          vtmp8 = vld4_lane_s8(w14, vtmp8, 6);
          w14 += 4;
          vtmp8 = vld4_lane_s8(w15, vtmp8, 7);
          w15 += 4;
          vst1_s8(out + 0, vtmp0.val[0]);
          vst1_s8(out + 8, vtmp8.val[0]);
          vst1_s8(out + 16, vtmp0.val[1]);
          vst1_s8(out + 24, vtmp8.val[1]);
          vst1_s8(out + 32, vtmp0.val[2]);
          vst1_s8(out + 40, vtmp8.val[2]);
          vst1_s8(out + 48, vtmp0.val[3]);
          vst1_s8(out + 56, vtmp8.val[3]);
          out += 64;
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
          int8x8x2_t vtmp8;
          vtmp8.val[0] = vdup_n_s8(0);
          vtmp8.val[1] = vdup_n_s8(0);
          vtmp8 = vld2_lane_s8(w8, vtmp8, 0);
          w8 += 2;
          vtmp8 = vld2_lane_s8(w9, vtmp8, 1);
          w9 += 2;
          vtmp8 = vld2_lane_s8(w10, vtmp8, 2);
          w10 += 2;
          vtmp8 = vld2_lane_s8(w11, vtmp8, 3);
          w11 += 2;
          vtmp8 = vld2_lane_s8(w12, vtmp8, 4);
          w12 += 2;
          vtmp8 = vld2_lane_s8(w13, vtmp8, 5);
          w13 += 2;
          vtmp8 = vld2_lane_s8(w14, vtmp8, 6);
          w14 += 2;
          vtmp8 = vld2_lane_s8(w15, vtmp8, 7);
          w15 += 2;
          vst1_s8(out + 0, vtmp0.val[0]);
          vst1_s8(out + 8, vtmp8.val[0]);
          vst1_s8(out + 16, vtmp0.val[1]);
          vst1_s8(out + 24, vtmp8.val[1]);
          out += 32;
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
          int8x8_t vtmp8 = vdup_n_s8(0);
          vtmp8 = vld1_lane_s8(w8, vtmp8, 0);
          w8 += 1;
          vtmp8 = vld1_lane_s8(w9, vtmp8, 1);
          w9 += 1;
          vtmp8 = vld1_lane_s8(w10, vtmp8, 2);
          w10 += 1;
          vtmp8 = vld1_lane_s8(w11, vtmp8, 3);
          w11 += 1;
          vtmp8 = vld1_lane_s8(w12, vtmp8, 4);
          w12 += 1;
          vtmp8 = vld1_lane_s8(w13, vtmp8, 5);
          w13 += 1;
          vtmp8 = vld1_lane_s8(w14, vtmp8, 6);
          w14 += 1;
          vtmp8 = vld1_lane_s8(w15, vtmp8, 7);
          w15 += 1;
          vst1_s8(out + 8, vtmp8);
          out += 16;
        }
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      wb += 16 * n_stride;
    }
    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
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
      const int8_t* w8 = w7 + n_stride;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const int8_t* w9 = w8 + n_stride;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const int8_t* w10 = w9 + n_stride;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const int8_t* w11 = w10 + n_stride;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const int8_t* w12 = w11 + n_stride;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const int8_t* w13 = w12 + n_stride;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const int8_t* w14 = w13 + n_stride;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const int8_t* w15 = w14 + n_stride;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }

      uint32_t* packed_b = (uint32_t*) out;
      const uint32x4_t vzero = vdupq_n_u32(0);
      vst1q_u32(packed_b + 0, vzero);
      vst1q_u32(packed_b + 4, vzero);
      vst1q_u32(packed_b + 8, vzero);
      vst1q_u32(packed_b + 12, vzero);
      if XNN_LIKELY(b != NULL) {
        for (size_t nb = 0; nb < n; ++nb) {
          packed_b[nb] = b[nb];
        }
        b += n;
      }
      out += 16 * sizeof(uint32_t);


      size_t k = kc;

      // KC loop / remainder of 16x8
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
        const int8x8_t v8 = vld1_s8(w8);
        w8 += 8;
        const int8x8_t v9 = vld1_s8(w9);
        w9 += 8;
        const int8x8_t v10 = vld1_s8(w10);
        w10 += 8;
        const int8x8_t v11 = vld1_s8(w11);
        w11 += 8;
        const int8x8_t v12 = vld1_s8(w12);
        w12 += 8;
        const int8x8_t v13 = vld1_s8(w13);
        w13 += 8;
        const int8x8_t v14 = vld1_s8(w14);
        w14 += 8;
        const int8x8_t v15 = vld1_s8(w15);
        w15 += 8;

        const int8x16_t t8_0 = xnn_zip_s8(v8, v9);
        const int8x16_t t8_1 = xnn_zip_s8(v10, v11);
        const int8x16_t t8_2 = xnn_zip_s8(v12, v13);
        const int8x16_t t8_3 = xnn_zip_s8(v14, v15);

        const int16x8x2_t u8_0 = vzipq_s16(
            vreinterpretq_s16_s8(t8_0),
            vreinterpretq_s16_s8(t8_1));
        const int16x8x2_t u8_1 = vzipq_s16(
            vreinterpretq_s16_s8(t8_2),
            vreinterpretq_s16_s8(t8_3));

        const int32x4x2_t s8_0 = vzipq_s32(
            vreinterpretq_s32_s16(u8_0.val[0]),
            vreinterpretq_s32_s16(u8_1.val[0]));
        const int32x4x2_t s8_2 = vzipq_s32(
            vreinterpretq_s32_s16(u8_0.val[1]),
            vreinterpretq_s32_s16(u8_1.val[1]));

        const int8x16x2_t out0_0 = xnn_zip_s64(
            vreinterpretq_s8_s32(s0_0.val[0]),
            vreinterpretq_s8_s32(s8_0.val[0]));
        vst1q_s8(out + 0, out0_0.val[0]);
        vst1q_s8(out + 16, out0_0.val[1]);
        const int8x16x2_t out0_1 = xnn_zip_s64(
            vreinterpretq_s8_s32(s0_0.val[1]),
            vreinterpretq_s8_s32(s8_0.val[1]));
        vst1q_s8(out + 32, out0_1.val[0]);
        vst1q_s8(out + 48, out0_1.val[1]);
        const int8x16x2_t out0_2 = xnn_zip_s64(
            vreinterpretq_s8_s32(s0_2.val[0]),
            vreinterpretq_s8_s32(s8_2.val[0]));
        vst1q_s8(out + 64, out0_2.val[0]);
        vst1q_s8(out + 80, out0_2.val[1]);
        const int8x16x2_t out0_3 = xnn_zip_s64(
            vreinterpretq_s8_s32(s0_2.val[1]),
            vreinterpretq_s8_s32(s8_2.val[1]));
        vst1q_s8(out + 96, out0_3.val[0]);
        vst1q_s8(out + 112, out0_3.val[1]);
        out += 128;
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
          int8x8x4_t vtmp8;
          vtmp8.val[0] = vdup_n_s8(0);
          vtmp8.val[1] = vdup_n_s8(0);
          vtmp8.val[2] = vdup_n_s8(0);
          vtmp8.val[3] = vdup_n_s8(0);
          vtmp8 = vld4_lane_s8(w8, vtmp8, 0);
          w8 += 4;
          vtmp8 = vld4_lane_s8(w9, vtmp8, 1);
          w9 += 4;
          vtmp8 = vld4_lane_s8(w10, vtmp8, 2);
          w10 += 4;
          vtmp8 = vld4_lane_s8(w11, vtmp8, 3);
          w11 += 4;
          vtmp8 = vld4_lane_s8(w12, vtmp8, 4);
          w12 += 4;
          vtmp8 = vld4_lane_s8(w13, vtmp8, 5);
          w13 += 4;
          vtmp8 = vld4_lane_s8(w14, vtmp8, 6);
          w14 += 4;
          vtmp8 = vld4_lane_s8(w15, vtmp8, 7);
          w15 += 4;
          vst1_s8(out + 0, vtmp0.val[0]);
          vst1_s8(out + 8, vtmp8.val[0]);
          vst1_s8(out + 16, vtmp0.val[1]);
          vst1_s8(out + 24, vtmp8.val[1]);
          vst1_s8(out + 32, vtmp0.val[2]);
          vst1_s8(out + 40, vtmp8.val[2]);
          vst1_s8(out + 48, vtmp0.val[3]);
          vst1_s8(out + 56, vtmp8.val[3]);
          out += 64;
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
          int8x8x2_t vtmp8;
          vtmp8.val[0] = vdup_n_s8(0);
          vtmp8.val[1] = vdup_n_s8(0);
          vtmp8 = vld2_lane_s8(w8, vtmp8, 0);
          w8 += 2;
          vtmp8 = vld2_lane_s8(w9, vtmp8, 1);
          w9 += 2;
          vtmp8 = vld2_lane_s8(w10, vtmp8, 2);
          w10 += 2;
          vtmp8 = vld2_lane_s8(w11, vtmp8, 3);
          w11 += 2;
          vtmp8 = vld2_lane_s8(w12, vtmp8, 4);
          w12 += 2;
          vtmp8 = vld2_lane_s8(w13, vtmp8, 5);
          w13 += 2;
          vtmp8 = vld2_lane_s8(w14, vtmp8, 6);
          w14 += 2;
          vtmp8 = vld2_lane_s8(w15, vtmp8, 7);
          w15 += 2;
          vst1_s8(out + 0, vtmp0.val[0]);
          vst1_s8(out + 8, vtmp8.val[0]);
          vst1_s8(out + 16, vtmp0.val[1]);
          vst1_s8(out + 24, vtmp8.val[1]);
          out += 32;
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
          int8x8_t vtmp8 = vdup_n_s8(0);
          vtmp8 = vld1_lane_s8(w8, vtmp8, 0);
          w8 += 1;
          vtmp8 = vld1_lane_s8(w9, vtmp8, 1);
          w9 += 1;
          vtmp8 = vld1_lane_s8(w10, vtmp8, 2);
          w10 += 1;
          vtmp8 = vld1_lane_s8(w11, vtmp8, 3);
          w11 += 1;
          vtmp8 = vld1_lane_s8(w12, vtmp8, 4);
          w12 += 1;
          vtmp8 = vld1_lane_s8(w13, vtmp8, 5);
          w13 += 1;
          vtmp8 = vld1_lane_s8(w14, vtmp8, 6);
          w14 += 1;
          vtmp8 = vld1_lane_s8(w15, vtmp8, 7);
          w15 += 1;
          vst1_s8(out + 8, vtmp8);
          out += 16;
        }
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights += nc * n_stride;
  } while (--g != 0);
}
