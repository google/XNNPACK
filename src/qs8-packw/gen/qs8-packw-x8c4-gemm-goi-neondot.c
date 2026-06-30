// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/c4-neon.c.in
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
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/unaligned.h"

XNN_INLINE static int32_t horizontal_add_s32(int32x4_t v) {
#if XNN_ARCH_ARM64
  return vaddvq_s32(v);
#else
  int32x2_t vacc_lo = vadd_s32(vget_low_s32(v), vget_high_s32(v));
  vacc_lo = vpadd_s32(vacc_lo, vacc_lo);
  return vget_lane_s32(vacc_lo, 0);
#endif
}



void xnn_qs8_packw_gemm_goi_ukernel_x8c4__neondot(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* weights,
  const int32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 8);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const int32_t izp = (int32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point): 0);
  const int8x16_t vones = vdupq_n_s8(1);
  const size_t mock_kc = kc;

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;

  do {
    const int8_t* w0 = weights;
    size_t n = nc;
    for (; n >= 8; n -= 8) {
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < 8; ++i) {
          ((int32_t*) out)[i] = b[i];
        }
        b += 8;
      } else {
        for (size_t i = 0; i < 8; ++i) {
          ((int32_t*) out)[i] = 0;
        }
      }
      out = (int8_t*) ((uintptr_t) out + 8 * sizeof(int32_t));

      const int8_t* w1 = w0 + mock_kc;
      const int8_t* w2 = w1 + mock_kc;
      const int8_t* w3 = w2 + mock_kc;
      const int8_t* w4 = w3 + mock_kc;
      const int8_t* w5 = w4 + mock_kc;
      const int8_t* w6 = w5 + mock_kc;
      const int8_t* w7 = w6 + mock_kc;

      int32x4_t vacc0 = vdupq_n_s32(0);
      int32x4_t vacc1 = vdupq_n_s32(0);
      int32x4_t vacc2 = vdupq_n_s32(0);
      int32x4_t vacc3 = vdupq_n_s32(0);
      int32x4_t vacc4 = vdupq_n_s32(0);
      int32x4_t vacc5 = vdupq_n_s32(0);
      int32x4_t vacc6 = vdupq_n_s32(0);
      int32x4_t vacc7 = vdupq_n_s32(0);

      size_t k = mock_kc;
      for (; k >= 16; k -= 16) {
        const int8x16_t v0 = vld1q_s8(w0); w0 += 16;
        vacc0 = vdotq_s32(vacc0, v0, vones);
        const int8x16_t v1 = vld1q_s8(w1); w1 += 16;
        vacc1 = vdotq_s32(vacc1, v1, vones);
        const int8x16_t v2 = vld1q_s8(w2); w2 += 16;
        vacc2 = vdotq_s32(vacc2, v2, vones);
        const int8x16_t v3 = vld1q_s8(w3); w3 += 16;
        vacc3 = vdotq_s32(vacc3, v3, vones);
        const int8x16_t v4 = vld1q_s8(w4); w4 += 16;
        vacc4 = vdotq_s32(vacc4, v4, vones);
        const int8x16_t v5 = vld1q_s8(w5); w5 += 16;
        vacc5 = vdotq_s32(vacc5, v5, vones);
        const int8x16_t v6 = vld1q_s8(w6); w6 += 16;
        vacc6 = vdotq_s32(vacc6, v6, vones);
        const int8x16_t v7 = vld1q_s8(w7); w7 += 16;
        vacc7 = vdotq_s32(vacc7, v7, vones);

        const int32x4_t x0_0 = vreinterpretq_s32_s8(v0);
        const int32x4_t x1_0 = vreinterpretq_s32_s8(v1);
        const int32x4_t x2_0 = vreinterpretq_s32_s8(v2);
        const int32x4_t x3_0 = vreinterpretq_s32_s8(v3);

        const int32x4x2_t t01_0 = vzipq_s32(x0_0, x2_0);
        const int32x4x2_t t23_0 = vzipq_s32(x1_0, x3_0);
        const int32x4_t t0_0 = t01_0.val[0];
        const int32x4_t t1_0 = t01_0.val[1];
        const int32x4_t t2_0 = t23_0.val[0];
        const int32x4_t t3_0 = t23_0.val[1];

        const int32x4x2_t s01_0 = vzipq_s32(t0_0, t2_0);
        const int32x4x2_t s23_0 = vzipq_s32(t1_0, t3_0);
        const int32x4_t s0_0 = s01_0.val[0];
        const int32x4_t s1_0 = s01_0.val[1];
        const int32x4_t s2_0 = s23_0.val[0];
        const int32x4_t s3_0 = s23_0.val[1];
        const int32x4_t x0_4 = vreinterpretq_s32_s8(v4);
        const int32x4_t x1_4 = vreinterpretq_s32_s8(v5);
        const int32x4_t x2_4 = vreinterpretq_s32_s8(v6);
        const int32x4_t x3_4 = vreinterpretq_s32_s8(v7);

        const int32x4x2_t t01_4 = vzipq_s32(x0_4, x2_4);
        const int32x4x2_t t23_4 = vzipq_s32(x1_4, x3_4);
        const int32x4_t t0_4 = t01_4.val[0];
        const int32x4_t t1_4 = t01_4.val[1];
        const int32x4_t t2_4 = t23_4.val[0];
        const int32x4_t t3_4 = t23_4.val[1];

        const int32x4x2_t s01_4 = vzipq_s32(t0_4, t2_4);
        const int32x4x2_t s23_4 = vzipq_s32(t1_4, t3_4);
        const int32x4_t s0_4 = s01_4.val[0];
        const int32x4_t s1_4 = s01_4.val[1];
        const int32x4_t s2_4 = s23_4.val[0];
        const int32x4_t s3_4 = s23_4.val[1];

        vst1q_s32((int32_t*)out + 0, s0_0);
        vst1q_s32((int32_t*)out + 4, s0_4);
        vst1q_s32((int32_t*)out + 8, s1_0);
        vst1q_s32((int32_t*)out + 12, s1_4);
        vst1q_s32((int32_t*)out + 16, s2_0);
        vst1q_s32((int32_t*)out + 20, s2_4);
        vst1q_s32((int32_t*)out + 24, s3_0);
        vst1q_s32((int32_t*)out + 28, s3_4);
        out += 128;
      }

      int32_t rem_ksum0 = 0;
      int32_t rem_ksum1 = 0;
      int32_t rem_ksum2 = 0;
      int32_t rem_ksum3 = 0;
      int32_t rem_ksum4 = 0;
      int32_t rem_ksum5 = 0;
      int32_t rem_ksum6 = 0;
      int32_t rem_ksum7 = 0;

      const size_t rem_step = 4;
      for (; k >= rem_step; k -= rem_step) {
        const uint32_t v0 = unaligned_load_u32(w0); w0 += 4;
        const int8_t* p0 = (const int8_t*)&v0;
        rem_ksum0 += p0[0] + p0[1] + p0[2] + p0[3];
        ((uint32_t*) out)[0] = v0;
        const uint32_t v1 = unaligned_load_u32(w1); w1 += 4;
        const int8_t* p1 = (const int8_t*)&v1;
        rem_ksum1 += p1[0] + p1[1] + p1[2] + p1[3];
        ((uint32_t*) out)[1] = v1;
        const uint32_t v2 = unaligned_load_u32(w2); w2 += 4;
        const int8_t* p2 = (const int8_t*)&v2;
        rem_ksum2 += p2[0] + p2[1] + p2[2] + p2[3];
        ((uint32_t*) out)[2] = v2;
        const uint32_t v3 = unaligned_load_u32(w3); w3 += 4;
        const int8_t* p3 = (const int8_t*)&v3;
        rem_ksum3 += p3[0] + p3[1] + p3[2] + p3[3];
        ((uint32_t*) out)[3] = v3;
        const uint32_t v4 = unaligned_load_u32(w4); w4 += 4;
        const int8_t* p4 = (const int8_t*)&v4;
        rem_ksum4 += p4[0] + p4[1] + p4[2] + p4[3];
        ((uint32_t*) out)[4] = v4;
        const uint32_t v5 = unaligned_load_u32(w5); w5 += 4;
        const int8_t* p5 = (const int8_t*)&v5;
        rem_ksum5 += p5[0] + p5[1] + p5[2] + p5[3];
        ((uint32_t*) out)[5] = v5;
        const uint32_t v6 = unaligned_load_u32(w6); w6 += 4;
        const int8_t* p6 = (const int8_t*)&v6;
        rem_ksum6 += p6[0] + p6[1] + p6[2] + p6[3];
        ((uint32_t*) out)[6] = v6;
        const uint32_t v7 = unaligned_load_u32(w7); w7 += 4;
        const int8_t* p7 = (const int8_t*)&v7;
        rem_ksum7 += p7[0] + p7[1] + p7[2] + p7[3];
        ((uint32_t*) out)[7] = v7;
        out += 8 * rem_step;
      }

      if (k != 0) {
        assert(k >= 1 && k <= 3);
        uint32_t v0 = 0;
        for (size_t i = 0; i < k; ++i) {
          v0 |= ((uint32_t) ((uint8_t) w0[i])) << (i * 8);
        }
        w0 += k;
        const int8_t* p0 = (const int8_t*) &v0;
        rem_ksum0 += p0[0] + p0[1] + p0[2] + p0[3];
        ((uint32_t*) out)[0] = v0;
        assert(k >= 1 && k <= 3);
        uint32_t v1 = 0;
        for (size_t i = 0; i < k; ++i) {
          v1 |= ((uint32_t) ((uint8_t) w1[i])) << (i * 8);
        }
        w1 += k;
        const int8_t* p1 = (const int8_t*) &v1;
        rem_ksum1 += p1[0] + p1[1] + p1[2] + p1[3];
        ((uint32_t*) out)[1] = v1;
        assert(k >= 1 && k <= 3);
        uint32_t v2 = 0;
        for (size_t i = 0; i < k; ++i) {
          v2 |= ((uint32_t) ((uint8_t) w2[i])) << (i * 8);
        }
        w2 += k;
        const int8_t* p2 = (const int8_t*) &v2;
        rem_ksum2 += p2[0] + p2[1] + p2[2] + p2[3];
        ((uint32_t*) out)[2] = v2;
        assert(k >= 1 && k <= 3);
        uint32_t v3 = 0;
        for (size_t i = 0; i < k; ++i) {
          v3 |= ((uint32_t) ((uint8_t) w3[i])) << (i * 8);
        }
        w3 += k;
        const int8_t* p3 = (const int8_t*) &v3;
        rem_ksum3 += p3[0] + p3[1] + p3[2] + p3[3];
        ((uint32_t*) out)[3] = v3;
        assert(k >= 1 && k <= 3);
        uint32_t v4 = 0;
        for (size_t i = 0; i < k; ++i) {
          v4 |= ((uint32_t) ((uint8_t) w4[i])) << (i * 8);
        }
        w4 += k;
        const int8_t* p4 = (const int8_t*) &v4;
        rem_ksum4 += p4[0] + p4[1] + p4[2] + p4[3];
        ((uint32_t*) out)[4] = v4;
        assert(k >= 1 && k <= 3);
        uint32_t v5 = 0;
        for (size_t i = 0; i < k; ++i) {
          v5 |= ((uint32_t) ((uint8_t) w5[i])) << (i * 8);
        }
        w5 += k;
        const int8_t* p5 = (const int8_t*) &v5;
        rem_ksum5 += p5[0] + p5[1] + p5[2] + p5[3];
        ((uint32_t*) out)[5] = v5;
        assert(k >= 1 && k <= 3);
        uint32_t v6 = 0;
        for (size_t i = 0; i < k; ++i) {
          v6 |= ((uint32_t) ((uint8_t) w6[i])) << (i * 8);
        }
        w6 += k;
        const int8_t* p6 = (const int8_t*) &v6;
        rem_ksum6 += p6[0] + p6[1] + p6[2] + p6[3];
        ((uint32_t*) out)[6] = v6;
        assert(k >= 1 && k <= 3);
        uint32_t v7 = 0;
        for (size_t i = 0; i < k; ++i) {
          v7 |= ((uint32_t) ((uint8_t) w7[i])) << (i * 8);
        }
        w7 += k;
        const int8_t* p7 = (const int8_t*) &v7;
        rem_ksum7 += p7[0] + p7[1] + p7[2] + p7[3];
        ((uint32_t*) out)[7] = v7;
        out += 8 * rem_step;
      }

      const int32_t ksum0 = horizontal_add_s32(vacc0) + rem_ksum0;
      packed_b[0] -= ksum0 * izp;
      const int32_t ksum1 = horizontal_add_s32(vacc1) + rem_ksum1;
      packed_b[1] -= ksum1 * izp;
      const int32_t ksum2 = horizontal_add_s32(vacc2) + rem_ksum2;
      packed_b[2] -= ksum2 * izp;
      const int32_t ksum3 = horizontal_add_s32(vacc3) + rem_ksum3;
      packed_b[3] -= ksum3 * izp;
      const int32_t ksum4 = horizontal_add_s32(vacc4) + rem_ksum4;
      packed_b[4] -= ksum4 * izp;
      const int32_t ksum5 = horizontal_add_s32(vacc5) + rem_ksum5;
      packed_b[5] -= ksum5 * izp;
      const int32_t ksum6 = horizontal_add_s32(vacc6) + rem_ksum6;
      packed_b[6] -= ksum6 * izp;
      const int32_t ksum7 = horizontal_add_s32(vacc7) + rem_ksum7;
      packed_b[7] -= ksum7 * izp;

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w7;
    }

    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 7);
      const int8_t* w1 = w0 + mock_kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int8_t* w3 = w2 + mock_kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int8_t* w4 = w3 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int8_t* w5 = w4 + mock_kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int8_t* w6 = w5 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int8_t* w7 = w6 + mock_kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }

      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          ((int32_t*) out)[i] = b[i];
        }
        b += n;
      } else {
        for (size_t i = 0; i < n; ++i) {
          ((int32_t*) out)[i] = 0;
        }
      }
      out = (int8_t*) ((uintptr_t) out + 8 * sizeof(int32_t));

      int32x4_t vacc0 = vdupq_n_s32(0);
      int32x4_t vacc1 = vdupq_n_s32(0);
      int32x4_t vacc2 = vdupq_n_s32(0);
      int32x4_t vacc3 = vdupq_n_s32(0);
      int32x4_t vacc4 = vdupq_n_s32(0);
      int32x4_t vacc5 = vdupq_n_s32(0);
      int32x4_t vacc6 = vdupq_n_s32(0);
      int32x4_t vacc7 = vdupq_n_s32(0);

      size_t k = mock_kc;
      for (; k >= 16; k -= 16) {
        const int8x16_t v0 = vld1q_s8(w0); w0 += 16;
        vacc0 = vdotq_s32(vacc0, v0, vones);
        const int8x16_t v1 = vld1q_s8(w1); w1 += 16;
        vacc1 = vdotq_s32(vacc1, v1, vones);
        const int8x16_t v2 = vld1q_s8(w2); w2 += 16;
        vacc2 = vdotq_s32(vacc2, v2, vones);
        const int8x16_t v3 = vld1q_s8(w3); w3 += 16;
        vacc3 = vdotq_s32(vacc3, v3, vones);
        const int8x16_t v4 = vld1q_s8(w4); w4 += 16;
        vacc4 = vdotq_s32(vacc4, v4, vones);
        const int8x16_t v5 = vld1q_s8(w5); w5 += 16;
        vacc5 = vdotq_s32(vacc5, v5, vones);
        const int8x16_t v6 = vld1q_s8(w6); w6 += 16;
        vacc6 = vdotq_s32(vacc6, v6, vones);
        const int8x16_t v7 = vld1q_s8(w7); w7 += 16;
        vacc7 = vdotq_s32(vacc7, v7, vones);

        const int32x4_t x0_0 = vreinterpretq_s32_s8(v0);
        const int32x4_t x1_0 = vreinterpretq_s32_s8(v1);
        const int32x4_t x2_0 = vreinterpretq_s32_s8(v2);
        const int32x4_t x3_0 = vreinterpretq_s32_s8(v3);

        const int32x4x2_t t01_0 = vzipq_s32(x0_0, x2_0);
        const int32x4x2_t t23_0 = vzipq_s32(x1_0, x3_0);
        const int32x4_t t0_0 = t01_0.val[0];
        const int32x4_t t1_0 = t01_0.val[1];
        const int32x4_t t2_0 = t23_0.val[0];
        const int32x4_t t3_0 = t23_0.val[1];

        const int32x4x2_t s01_0 = vzipq_s32(t0_0, t2_0);
        const int32x4x2_t s23_0 = vzipq_s32(t1_0, t3_0);
        const int32x4_t s0_0 = s01_0.val[0];
        const int32x4_t s1_0 = s01_0.val[1];
        const int32x4_t s2_0 = s23_0.val[0];
        const int32x4_t s3_0 = s23_0.val[1];
        const int32x4_t x0_4 = vreinterpretq_s32_s8(v4);
        const int32x4_t x1_4 = vreinterpretq_s32_s8(v5);
        const int32x4_t x2_4 = vreinterpretq_s32_s8(v6);
        const int32x4_t x3_4 = vreinterpretq_s32_s8(v7);

        const int32x4x2_t t01_4 = vzipq_s32(x0_4, x2_4);
        const int32x4x2_t t23_4 = vzipq_s32(x1_4, x3_4);
        const int32x4_t t0_4 = t01_4.val[0];
        const int32x4_t t1_4 = t01_4.val[1];
        const int32x4_t t2_4 = t23_4.val[0];
        const int32x4_t t3_4 = t23_4.val[1];

        const int32x4x2_t s01_4 = vzipq_s32(t0_4, t2_4);
        const int32x4x2_t s23_4 = vzipq_s32(t1_4, t3_4);
        const int32x4_t s0_4 = s01_4.val[0];
        const int32x4_t s1_4 = s01_4.val[1];
        const int32x4_t s2_4 = s23_4.val[0];
        const int32x4_t s3_4 = s23_4.val[1];

        vst1q_s32((int32_t*)out + 0, s0_0);
        vst1q_s32((int32_t*)out + 4, s0_4);
        vst1q_s32((int32_t*)out + 8, s1_0);
        vst1q_s32((int32_t*)out + 12, s1_4);
        vst1q_s32((int32_t*)out + 16, s2_0);
        vst1q_s32((int32_t*)out + 20, s2_4);
        vst1q_s32((int32_t*)out + 24, s3_0);
        vst1q_s32((int32_t*)out + 28, s3_4);
        out += 128;
      }

      int32_t rem_ksum0 = 0;
      int32_t rem_ksum1 = 0;
      int32_t rem_ksum2 = 0;
      int32_t rem_ksum3 = 0;
      int32_t rem_ksum4 = 0;
      int32_t rem_ksum5 = 0;
      int32_t rem_ksum6 = 0;
      int32_t rem_ksum7 = 0;

      const size_t rem_step = 4;
      for (; k >= rem_step; k -= rem_step) {
        const uint32_t v0 = unaligned_load_u32(w0); w0 += 4;
        const int8_t* p0 = (const int8_t*)&v0;
        rem_ksum0 += p0[0] + p0[1] + p0[2] + p0[3];
        ((uint32_t*) out)[0] = v0;
        const uint32_t v1 = unaligned_load_u32(w1); w1 += 4;
        const int8_t* p1 = (const int8_t*)&v1;
        rem_ksum1 += p1[0] + p1[1] + p1[2] + p1[3];
        ((uint32_t*) out)[1] = v1;
        const uint32_t v2 = unaligned_load_u32(w2); w2 += 4;
        const int8_t* p2 = (const int8_t*)&v2;
        rem_ksum2 += p2[0] + p2[1] + p2[2] + p2[3];
        ((uint32_t*) out)[2] = v2;
        const uint32_t v3 = unaligned_load_u32(w3); w3 += 4;
        const int8_t* p3 = (const int8_t*)&v3;
        rem_ksum3 += p3[0] + p3[1] + p3[2] + p3[3];
        ((uint32_t*) out)[3] = v3;
        const uint32_t v4 = unaligned_load_u32(w4); w4 += 4;
        const int8_t* p4 = (const int8_t*)&v4;
        rem_ksum4 += p4[0] + p4[1] + p4[2] + p4[3];
        ((uint32_t*) out)[4] = v4;
        const uint32_t v5 = unaligned_load_u32(w5); w5 += 4;
        const int8_t* p5 = (const int8_t*)&v5;
        rem_ksum5 += p5[0] + p5[1] + p5[2] + p5[3];
        ((uint32_t*) out)[5] = v5;
        const uint32_t v6 = unaligned_load_u32(w6); w6 += 4;
        const int8_t* p6 = (const int8_t*)&v6;
        rem_ksum6 += p6[0] + p6[1] + p6[2] + p6[3];
        ((uint32_t*) out)[6] = v6;
        const uint32_t v7 = unaligned_load_u32(w7); w7 += 4;
        const int8_t* p7 = (const int8_t*)&v7;
        rem_ksum7 += p7[0] + p7[1] + p7[2] + p7[3];
        ((uint32_t*) out)[7] = v7;
        out += 8 * rem_step;
      }

      if (k != 0) {
        assert(k >= 1 && k <= 3);
        uint32_t v0 = 0;
        for (size_t i = 0; i < k; ++i) {
          v0 |= ((uint32_t) ((uint8_t) w0[i])) << (i * 8);
        }
        w0 += k;
        const int8_t* p0 = (const int8_t*) &v0;
        rem_ksum0 += p0[0] + p0[1] + p0[2] + p0[3];
        ((uint32_t*) out)[0] = v0;
        assert(k >= 1 && k <= 3);
        uint32_t v1 = 0;
        for (size_t i = 0; i < k; ++i) {
          v1 |= ((uint32_t) ((uint8_t) w1[i])) << (i * 8);
        }
        w1 += k;
        const int8_t* p1 = (const int8_t*) &v1;
        rem_ksum1 += p1[0] + p1[1] + p1[2] + p1[3];
        ((uint32_t*) out)[1] = v1;
        assert(k >= 1 && k <= 3);
        uint32_t v2 = 0;
        for (size_t i = 0; i < k; ++i) {
          v2 |= ((uint32_t) ((uint8_t) w2[i])) << (i * 8);
        }
        w2 += k;
        const int8_t* p2 = (const int8_t*) &v2;
        rem_ksum2 += p2[0] + p2[1] + p2[2] + p2[3];
        ((uint32_t*) out)[2] = v2;
        assert(k >= 1 && k <= 3);
        uint32_t v3 = 0;
        for (size_t i = 0; i < k; ++i) {
          v3 |= ((uint32_t) ((uint8_t) w3[i])) << (i * 8);
        }
        w3 += k;
        const int8_t* p3 = (const int8_t*) &v3;
        rem_ksum3 += p3[0] + p3[1] + p3[2] + p3[3];
        ((uint32_t*) out)[3] = v3;
        assert(k >= 1 && k <= 3);
        uint32_t v4 = 0;
        for (size_t i = 0; i < k; ++i) {
          v4 |= ((uint32_t) ((uint8_t) w4[i])) << (i * 8);
        }
        w4 += k;
        const int8_t* p4 = (const int8_t*) &v4;
        rem_ksum4 += p4[0] + p4[1] + p4[2] + p4[3];
        ((uint32_t*) out)[4] = v4;
        assert(k >= 1 && k <= 3);
        uint32_t v5 = 0;
        for (size_t i = 0; i < k; ++i) {
          v5 |= ((uint32_t) ((uint8_t) w5[i])) << (i * 8);
        }
        w5 += k;
        const int8_t* p5 = (const int8_t*) &v5;
        rem_ksum5 += p5[0] + p5[1] + p5[2] + p5[3];
        ((uint32_t*) out)[5] = v5;
        assert(k >= 1 && k <= 3);
        uint32_t v6 = 0;
        for (size_t i = 0; i < k; ++i) {
          v6 |= ((uint32_t) ((uint8_t) w6[i])) << (i * 8);
        }
        w6 += k;
        const int8_t* p6 = (const int8_t*) &v6;
        rem_ksum6 += p6[0] + p6[1] + p6[2] + p6[3];
        ((uint32_t*) out)[6] = v6;
        assert(k >= 1 && k <= 3);
        uint32_t v7 = 0;
        for (size_t i = 0; i < k; ++i) {
          v7 |= ((uint32_t) ((uint8_t) w7[i])) << (i * 8);
        }
        w7 += k;
        const int8_t* p7 = (const int8_t*) &v7;
        rem_ksum7 += p7[0] + p7[1] + p7[2] + p7[3];
        ((uint32_t*) out)[7] = v7;
        out += 8 * rem_step;
      }

      if (0 < n) {
        const int32_t ksum0 = horizontal_add_s32(vacc0) + rem_ksum0;
        packed_b[0] -= ksum0 * izp;
      }
      if (1 < n) {
        const int32_t ksum1 = horizontal_add_s32(vacc1) + rem_ksum1;
        packed_b[1] -= ksum1 * izp;
      }
      if (2 < n) {
        const int32_t ksum2 = horizontal_add_s32(vacc2) + rem_ksum2;
        packed_b[2] -= ksum2 * izp;
      }
      if (3 < n) {
        const int32_t ksum3 = horizontal_add_s32(vacc3) + rem_ksum3;
        packed_b[3] -= ksum3 * izp;
      }
      if (4 < n) {
        const int32_t ksum4 = horizontal_add_s32(vacc4) + rem_ksum4;
        packed_b[4] -= ksum4 * izp;
      }
      if (5 < n) {
        const int32_t ksum5 = horizontal_add_s32(vacc5) + rem_ksum5;
        packed_b[5] -= ksum5 * izp;
      }
      if (6 < n) {
        const int32_t ksum6 = horizontal_add_s32(vacc6) + rem_ksum6;
        packed_b[6] -= ksum6 * izp;
      }
      if (7 < n) {
        const int32_t ksum7 = horizontal_add_s32(vacc7) + rem_ksum7;
        packed_b[7] -= ksum7 * izp;
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights = (const int8_t*)((intptr_t) weights + nc * kc);
  } while (--g != 0);
}
