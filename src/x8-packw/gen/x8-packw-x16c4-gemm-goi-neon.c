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



void xnn_x8_packw_gemm_goi_ukernel_x16c4__neon(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
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
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const size_t mock_kc = kc;

  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    const int8_t* w0 = weights;
    size_t n = nc;
    for (; n >= 16; n -= 16) {
      if (b != NULL) {
        for (size_t i = 0; i < 16; ++i) {
          ((uint32_t*) out)[i] = b[i];
        }
        b += 16;
      } else {
        for (size_t i = 0; i < 16; ++i) {
          ((uint32_t*) out)[i] = 0;
        }
      }
      out = (int8_t*) ((uintptr_t) out + 16 * sizeof(uint32_t));

      const int8_t* w1 = w0 + mock_kc;
      const int8_t* w2 = w1 + mock_kc;
      const int8_t* w3 = w2 + mock_kc;
      const int8_t* w4 = w3 + mock_kc;
      const int8_t* w5 = w4 + mock_kc;
      const int8_t* w6 = w5 + mock_kc;
      const int8_t* w7 = w6 + mock_kc;
      const int8_t* w8 = w7 + mock_kc;
      const int8_t* w9 = w8 + mock_kc;
      const int8_t* w10 = w9 + mock_kc;
      const int8_t* w11 = w10 + mock_kc;
      const int8_t* w12 = w11 + mock_kc;
      const int8_t* w13 = w12 + mock_kc;
      const int8_t* w14 = w13 + mock_kc;
      const int8_t* w15 = w14 + mock_kc;


      size_t k = mock_kc;
      for (; k >= 16; k -= 16) {
        const int8x16_t v0 = vld1q_s8(w0); w0 += 16;
        const int8x16_t v1 = vld1q_s8(w1); w1 += 16;
        const int8x16_t v2 = vld1q_s8(w2); w2 += 16;
        const int8x16_t v3 = vld1q_s8(w3); w3 += 16;
        const int8x16_t v4 = vld1q_s8(w4); w4 += 16;
        const int8x16_t v5 = vld1q_s8(w5); w5 += 16;
        const int8x16_t v6 = vld1q_s8(w6); w6 += 16;
        const int8x16_t v7 = vld1q_s8(w7); w7 += 16;
        const int8x16_t v8 = vld1q_s8(w8); w8 += 16;
        const int8x16_t v9 = vld1q_s8(w9); w9 += 16;
        const int8x16_t v10 = vld1q_s8(w10); w10 += 16;
        const int8x16_t v11 = vld1q_s8(w11); w11 += 16;
        const int8x16_t v12 = vld1q_s8(w12); w12 += 16;
        const int8x16_t v13 = vld1q_s8(w13); w13 += 16;
        const int8x16_t v14 = vld1q_s8(w14); w14 += 16;
        const int8x16_t v15 = vld1q_s8(w15); w15 += 16;

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
        const int32x4_t x0_8 = vreinterpretq_s32_s8(v8);
        const int32x4_t x1_8 = vreinterpretq_s32_s8(v9);
        const int32x4_t x2_8 = vreinterpretq_s32_s8(v10);
        const int32x4_t x3_8 = vreinterpretq_s32_s8(v11);

        const int32x4x2_t t01_8 = vzipq_s32(x0_8, x2_8);
        const int32x4x2_t t23_8 = vzipq_s32(x1_8, x3_8);
        const int32x4_t t0_8 = t01_8.val[0];
        const int32x4_t t1_8 = t01_8.val[1];
        const int32x4_t t2_8 = t23_8.val[0];
        const int32x4_t t3_8 = t23_8.val[1];

        const int32x4x2_t s01_8 = vzipq_s32(t0_8, t2_8);
        const int32x4x2_t s23_8 = vzipq_s32(t1_8, t3_8);
        const int32x4_t s0_8 = s01_8.val[0];
        const int32x4_t s1_8 = s01_8.val[1];
        const int32x4_t s2_8 = s23_8.val[0];
        const int32x4_t s3_8 = s23_8.val[1];
        const int32x4_t x0_12 = vreinterpretq_s32_s8(v12);
        const int32x4_t x1_12 = vreinterpretq_s32_s8(v13);
        const int32x4_t x2_12 = vreinterpretq_s32_s8(v14);
        const int32x4_t x3_12 = vreinterpretq_s32_s8(v15);

        const int32x4x2_t t01_12 = vzipq_s32(x0_12, x2_12);
        const int32x4x2_t t23_12 = vzipq_s32(x1_12, x3_12);
        const int32x4_t t0_12 = t01_12.val[0];
        const int32x4_t t1_12 = t01_12.val[1];
        const int32x4_t t2_12 = t23_12.val[0];
        const int32x4_t t3_12 = t23_12.val[1];

        const int32x4x2_t s01_12 = vzipq_s32(t0_12, t2_12);
        const int32x4x2_t s23_12 = vzipq_s32(t1_12, t3_12);
        const int32x4_t s0_12 = s01_12.val[0];
        const int32x4_t s1_12 = s01_12.val[1];
        const int32x4_t s2_12 = s23_12.val[0];
        const int32x4_t s3_12 = s23_12.val[1];

        vst1q_s32((int32_t*)out + 0, s0_0);
        vst1q_s32((int32_t*)out + 4, s0_4);
        vst1q_s32((int32_t*)out + 8, s0_8);
        vst1q_s32((int32_t*)out + 12, s0_12);
        vst1q_s32((int32_t*)out + 16, s1_0);
        vst1q_s32((int32_t*)out + 20, s1_4);
        vst1q_s32((int32_t*)out + 24, s1_8);
        vst1q_s32((int32_t*)out + 28, s1_12);
        vst1q_s32((int32_t*)out + 32, s2_0);
        vst1q_s32((int32_t*)out + 36, s2_4);
        vst1q_s32((int32_t*)out + 40, s2_8);
        vst1q_s32((int32_t*)out + 44, s2_12);
        vst1q_s32((int32_t*)out + 48, s3_0);
        vst1q_s32((int32_t*)out + 52, s3_4);
        vst1q_s32((int32_t*)out + 56, s3_8);
        vst1q_s32((int32_t*)out + 60, s3_12);
        out += 256;
      }


      const size_t rem_step = 4;
      for (; k >= rem_step; k -= rem_step) {
        const uint32_t v0 = unaligned_load_u32(w0); w0 += 4;
        ((uint32_t*) out)[0] = v0;
        const uint32_t v1 = unaligned_load_u32(w1); w1 += 4;
        ((uint32_t*) out)[1] = v1;
        const uint32_t v2 = unaligned_load_u32(w2); w2 += 4;
        ((uint32_t*) out)[2] = v2;
        const uint32_t v3 = unaligned_load_u32(w3); w3 += 4;
        ((uint32_t*) out)[3] = v3;
        const uint32_t v4 = unaligned_load_u32(w4); w4 += 4;
        ((uint32_t*) out)[4] = v4;
        const uint32_t v5 = unaligned_load_u32(w5); w5 += 4;
        ((uint32_t*) out)[5] = v5;
        const uint32_t v6 = unaligned_load_u32(w6); w6 += 4;
        ((uint32_t*) out)[6] = v6;
        const uint32_t v7 = unaligned_load_u32(w7); w7 += 4;
        ((uint32_t*) out)[7] = v7;
        const uint32_t v8 = unaligned_load_u32(w8); w8 += 4;
        ((uint32_t*) out)[8] = v8;
        const uint32_t v9 = unaligned_load_u32(w9); w9 += 4;
        ((uint32_t*) out)[9] = v9;
        const uint32_t v10 = unaligned_load_u32(w10); w10 += 4;
        ((uint32_t*) out)[10] = v10;
        const uint32_t v11 = unaligned_load_u32(w11); w11 += 4;
        ((uint32_t*) out)[11] = v11;
        const uint32_t v12 = unaligned_load_u32(w12); w12 += 4;
        ((uint32_t*) out)[12] = v12;
        const uint32_t v13 = unaligned_load_u32(w13); w13 += 4;
        ((uint32_t*) out)[13] = v13;
        const uint32_t v14 = unaligned_load_u32(w14); w14 += 4;
        ((uint32_t*) out)[14] = v14;
        const uint32_t v15 = unaligned_load_u32(w15); w15 += 4;
        ((uint32_t*) out)[15] = v15;
        out += 16 * rem_step;
      }

      if (k != 0) {
        assert(k >= 1 && k <= 3);
        uint32_t v0 = 0;
        for (size_t i = 0; i < k; ++i) {
          v0 |= ((uint32_t) ((uint8_t) w0[i])) << (i * 8);
        }
        w0 += k;
        ((uint32_t*) out)[0] = v0;
        assert(k >= 1 && k <= 3);
        uint32_t v1 = 0;
        for (size_t i = 0; i < k; ++i) {
          v1 |= ((uint32_t) ((uint8_t) w1[i])) << (i * 8);
        }
        w1 += k;
        ((uint32_t*) out)[1] = v1;
        assert(k >= 1 && k <= 3);
        uint32_t v2 = 0;
        for (size_t i = 0; i < k; ++i) {
          v2 |= ((uint32_t) ((uint8_t) w2[i])) << (i * 8);
        }
        w2 += k;
        ((uint32_t*) out)[2] = v2;
        assert(k >= 1 && k <= 3);
        uint32_t v3 = 0;
        for (size_t i = 0; i < k; ++i) {
          v3 |= ((uint32_t) ((uint8_t) w3[i])) << (i * 8);
        }
        w3 += k;
        ((uint32_t*) out)[3] = v3;
        assert(k >= 1 && k <= 3);
        uint32_t v4 = 0;
        for (size_t i = 0; i < k; ++i) {
          v4 |= ((uint32_t) ((uint8_t) w4[i])) << (i * 8);
        }
        w4 += k;
        ((uint32_t*) out)[4] = v4;
        assert(k >= 1 && k <= 3);
        uint32_t v5 = 0;
        for (size_t i = 0; i < k; ++i) {
          v5 |= ((uint32_t) ((uint8_t) w5[i])) << (i * 8);
        }
        w5 += k;
        ((uint32_t*) out)[5] = v5;
        assert(k >= 1 && k <= 3);
        uint32_t v6 = 0;
        for (size_t i = 0; i < k; ++i) {
          v6 |= ((uint32_t) ((uint8_t) w6[i])) << (i * 8);
        }
        w6 += k;
        ((uint32_t*) out)[6] = v6;
        assert(k >= 1 && k <= 3);
        uint32_t v7 = 0;
        for (size_t i = 0; i < k; ++i) {
          v7 |= ((uint32_t) ((uint8_t) w7[i])) << (i * 8);
        }
        w7 += k;
        ((uint32_t*) out)[7] = v7;
        assert(k >= 1 && k <= 3);
        uint32_t v8 = 0;
        for (size_t i = 0; i < k; ++i) {
          v8 |= ((uint32_t) ((uint8_t) w8[i])) << (i * 8);
        }
        w8 += k;
        ((uint32_t*) out)[8] = v8;
        assert(k >= 1 && k <= 3);
        uint32_t v9 = 0;
        for (size_t i = 0; i < k; ++i) {
          v9 |= ((uint32_t) ((uint8_t) w9[i])) << (i * 8);
        }
        w9 += k;
        ((uint32_t*) out)[9] = v9;
        assert(k >= 1 && k <= 3);
        uint32_t v10 = 0;
        for (size_t i = 0; i < k; ++i) {
          v10 |= ((uint32_t) ((uint8_t) w10[i])) << (i * 8);
        }
        w10 += k;
        ((uint32_t*) out)[10] = v10;
        assert(k >= 1 && k <= 3);
        uint32_t v11 = 0;
        for (size_t i = 0; i < k; ++i) {
          v11 |= ((uint32_t) ((uint8_t) w11[i])) << (i * 8);
        }
        w11 += k;
        ((uint32_t*) out)[11] = v11;
        assert(k >= 1 && k <= 3);
        uint32_t v12 = 0;
        for (size_t i = 0; i < k; ++i) {
          v12 |= ((uint32_t) ((uint8_t) w12[i])) << (i * 8);
        }
        w12 += k;
        ((uint32_t*) out)[12] = v12;
        assert(k >= 1 && k <= 3);
        uint32_t v13 = 0;
        for (size_t i = 0; i < k; ++i) {
          v13 |= ((uint32_t) ((uint8_t) w13[i])) << (i * 8);
        }
        w13 += k;
        ((uint32_t*) out)[13] = v13;
        assert(k >= 1 && k <= 3);
        uint32_t v14 = 0;
        for (size_t i = 0; i < k; ++i) {
          v14 |= ((uint32_t) ((uint8_t) w14[i])) << (i * 8);
        }
        w14 += k;
        ((uint32_t*) out)[14] = v14;
        assert(k >= 1 && k <= 3);
        uint32_t v15 = 0;
        for (size_t i = 0; i < k; ++i) {
          v15 |= ((uint32_t) ((uint8_t) w15[i])) << (i * 8);
        }
        w15 += k;
        ((uint32_t*) out)[15] = v15;
        out += 16 * rem_step;
      }


      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w15;
    }

    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 15);
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
      const int8_t* w8 = w7 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const int8_t* w9 = w8 + mock_kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const int8_t* w10 = w9 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const int8_t* w11 = w10 + mock_kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const int8_t* w12 = w11 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const int8_t* w13 = w12 + mock_kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const int8_t* w14 = w13 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const int8_t* w15 = w14 + mock_kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }

      if (b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          ((uint32_t*) out)[i] = b[i];
        }
        b += n;
      } else {
        for (size_t i = 0; i < n; ++i) {
          ((uint32_t*) out)[i] = 0;
        }
      }
      out = (int8_t*) ((uintptr_t) out + 16 * sizeof(uint32_t));


      size_t k = mock_kc;
      for (; k >= 16; k -= 16) {
        const int8x16_t v0 = vld1q_s8(w0); w0 += 16;
        const int8x16_t v1 = vld1q_s8(w1); w1 += 16;
        const int8x16_t v2 = vld1q_s8(w2); w2 += 16;
        const int8x16_t v3 = vld1q_s8(w3); w3 += 16;
        const int8x16_t v4 = vld1q_s8(w4); w4 += 16;
        const int8x16_t v5 = vld1q_s8(w5); w5 += 16;
        const int8x16_t v6 = vld1q_s8(w6); w6 += 16;
        const int8x16_t v7 = vld1q_s8(w7); w7 += 16;
        const int8x16_t v8 = vld1q_s8(w8); w8 += 16;
        const int8x16_t v9 = vld1q_s8(w9); w9 += 16;
        const int8x16_t v10 = vld1q_s8(w10); w10 += 16;
        const int8x16_t v11 = vld1q_s8(w11); w11 += 16;
        const int8x16_t v12 = vld1q_s8(w12); w12 += 16;
        const int8x16_t v13 = vld1q_s8(w13); w13 += 16;
        const int8x16_t v14 = vld1q_s8(w14); w14 += 16;
        const int8x16_t v15 = vld1q_s8(w15); w15 += 16;

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
        const int32x4_t x0_8 = vreinterpretq_s32_s8(v8);
        const int32x4_t x1_8 = vreinterpretq_s32_s8(v9);
        const int32x4_t x2_8 = vreinterpretq_s32_s8(v10);
        const int32x4_t x3_8 = vreinterpretq_s32_s8(v11);

        const int32x4x2_t t01_8 = vzipq_s32(x0_8, x2_8);
        const int32x4x2_t t23_8 = vzipq_s32(x1_8, x3_8);
        const int32x4_t t0_8 = t01_8.val[0];
        const int32x4_t t1_8 = t01_8.val[1];
        const int32x4_t t2_8 = t23_8.val[0];
        const int32x4_t t3_8 = t23_8.val[1];

        const int32x4x2_t s01_8 = vzipq_s32(t0_8, t2_8);
        const int32x4x2_t s23_8 = vzipq_s32(t1_8, t3_8);
        const int32x4_t s0_8 = s01_8.val[0];
        const int32x4_t s1_8 = s01_8.val[1];
        const int32x4_t s2_8 = s23_8.val[0];
        const int32x4_t s3_8 = s23_8.val[1];
        const int32x4_t x0_12 = vreinterpretq_s32_s8(v12);
        const int32x4_t x1_12 = vreinterpretq_s32_s8(v13);
        const int32x4_t x2_12 = vreinterpretq_s32_s8(v14);
        const int32x4_t x3_12 = vreinterpretq_s32_s8(v15);

        const int32x4x2_t t01_12 = vzipq_s32(x0_12, x2_12);
        const int32x4x2_t t23_12 = vzipq_s32(x1_12, x3_12);
        const int32x4_t t0_12 = t01_12.val[0];
        const int32x4_t t1_12 = t01_12.val[1];
        const int32x4_t t2_12 = t23_12.val[0];
        const int32x4_t t3_12 = t23_12.val[1];

        const int32x4x2_t s01_12 = vzipq_s32(t0_12, t2_12);
        const int32x4x2_t s23_12 = vzipq_s32(t1_12, t3_12);
        const int32x4_t s0_12 = s01_12.val[0];
        const int32x4_t s1_12 = s01_12.val[1];
        const int32x4_t s2_12 = s23_12.val[0];
        const int32x4_t s3_12 = s23_12.val[1];

        vst1q_s32((int32_t*)out + 0, s0_0);
        vst1q_s32((int32_t*)out + 4, s0_4);
        vst1q_s32((int32_t*)out + 8, s0_8);
        vst1q_s32((int32_t*)out + 12, s0_12);
        vst1q_s32((int32_t*)out + 16, s1_0);
        vst1q_s32((int32_t*)out + 20, s1_4);
        vst1q_s32((int32_t*)out + 24, s1_8);
        vst1q_s32((int32_t*)out + 28, s1_12);
        vst1q_s32((int32_t*)out + 32, s2_0);
        vst1q_s32((int32_t*)out + 36, s2_4);
        vst1q_s32((int32_t*)out + 40, s2_8);
        vst1q_s32((int32_t*)out + 44, s2_12);
        vst1q_s32((int32_t*)out + 48, s3_0);
        vst1q_s32((int32_t*)out + 52, s3_4);
        vst1q_s32((int32_t*)out + 56, s3_8);
        vst1q_s32((int32_t*)out + 60, s3_12);
        out += 256;
      }


      const size_t rem_step = 4;
      for (; k >= rem_step; k -= rem_step) {
        const uint32_t v0 = unaligned_load_u32(w0); w0 += 4;
        ((uint32_t*) out)[0] = v0;
        const uint32_t v1 = unaligned_load_u32(w1); w1 += 4;
        ((uint32_t*) out)[1] = v1;
        const uint32_t v2 = unaligned_load_u32(w2); w2 += 4;
        ((uint32_t*) out)[2] = v2;
        const uint32_t v3 = unaligned_load_u32(w3); w3 += 4;
        ((uint32_t*) out)[3] = v3;
        const uint32_t v4 = unaligned_load_u32(w4); w4 += 4;
        ((uint32_t*) out)[4] = v4;
        const uint32_t v5 = unaligned_load_u32(w5); w5 += 4;
        ((uint32_t*) out)[5] = v5;
        const uint32_t v6 = unaligned_load_u32(w6); w6 += 4;
        ((uint32_t*) out)[6] = v6;
        const uint32_t v7 = unaligned_load_u32(w7); w7 += 4;
        ((uint32_t*) out)[7] = v7;
        const uint32_t v8 = unaligned_load_u32(w8); w8 += 4;
        ((uint32_t*) out)[8] = v8;
        const uint32_t v9 = unaligned_load_u32(w9); w9 += 4;
        ((uint32_t*) out)[9] = v9;
        const uint32_t v10 = unaligned_load_u32(w10); w10 += 4;
        ((uint32_t*) out)[10] = v10;
        const uint32_t v11 = unaligned_load_u32(w11); w11 += 4;
        ((uint32_t*) out)[11] = v11;
        const uint32_t v12 = unaligned_load_u32(w12); w12 += 4;
        ((uint32_t*) out)[12] = v12;
        const uint32_t v13 = unaligned_load_u32(w13); w13 += 4;
        ((uint32_t*) out)[13] = v13;
        const uint32_t v14 = unaligned_load_u32(w14); w14 += 4;
        ((uint32_t*) out)[14] = v14;
        const uint32_t v15 = unaligned_load_u32(w15); w15 += 4;
        ((uint32_t*) out)[15] = v15;
        out += 16 * rem_step;
      }

      if (k != 0) {
        assert(k >= 1 && k <= 3);
        uint32_t v0 = 0;
        for (size_t i = 0; i < k; ++i) {
          v0 |= ((uint32_t) ((uint8_t) w0[i])) << (i * 8);
        }
        w0 += k;
        ((uint32_t*) out)[0] = v0;
        assert(k >= 1 && k <= 3);
        uint32_t v1 = 0;
        for (size_t i = 0; i < k; ++i) {
          v1 |= ((uint32_t) ((uint8_t) w1[i])) << (i * 8);
        }
        w1 += k;
        ((uint32_t*) out)[1] = v1;
        assert(k >= 1 && k <= 3);
        uint32_t v2 = 0;
        for (size_t i = 0; i < k; ++i) {
          v2 |= ((uint32_t) ((uint8_t) w2[i])) << (i * 8);
        }
        w2 += k;
        ((uint32_t*) out)[2] = v2;
        assert(k >= 1 && k <= 3);
        uint32_t v3 = 0;
        for (size_t i = 0; i < k; ++i) {
          v3 |= ((uint32_t) ((uint8_t) w3[i])) << (i * 8);
        }
        w3 += k;
        ((uint32_t*) out)[3] = v3;
        assert(k >= 1 && k <= 3);
        uint32_t v4 = 0;
        for (size_t i = 0; i < k; ++i) {
          v4 |= ((uint32_t) ((uint8_t) w4[i])) << (i * 8);
        }
        w4 += k;
        ((uint32_t*) out)[4] = v4;
        assert(k >= 1 && k <= 3);
        uint32_t v5 = 0;
        for (size_t i = 0; i < k; ++i) {
          v5 |= ((uint32_t) ((uint8_t) w5[i])) << (i * 8);
        }
        w5 += k;
        ((uint32_t*) out)[5] = v5;
        assert(k >= 1 && k <= 3);
        uint32_t v6 = 0;
        for (size_t i = 0; i < k; ++i) {
          v6 |= ((uint32_t) ((uint8_t) w6[i])) << (i * 8);
        }
        w6 += k;
        ((uint32_t*) out)[6] = v6;
        assert(k >= 1 && k <= 3);
        uint32_t v7 = 0;
        for (size_t i = 0; i < k; ++i) {
          v7 |= ((uint32_t) ((uint8_t) w7[i])) << (i * 8);
        }
        w7 += k;
        ((uint32_t*) out)[7] = v7;
        assert(k >= 1 && k <= 3);
        uint32_t v8 = 0;
        for (size_t i = 0; i < k; ++i) {
          v8 |= ((uint32_t) ((uint8_t) w8[i])) << (i * 8);
        }
        w8 += k;
        ((uint32_t*) out)[8] = v8;
        assert(k >= 1 && k <= 3);
        uint32_t v9 = 0;
        for (size_t i = 0; i < k; ++i) {
          v9 |= ((uint32_t) ((uint8_t) w9[i])) << (i * 8);
        }
        w9 += k;
        ((uint32_t*) out)[9] = v9;
        assert(k >= 1 && k <= 3);
        uint32_t v10 = 0;
        for (size_t i = 0; i < k; ++i) {
          v10 |= ((uint32_t) ((uint8_t) w10[i])) << (i * 8);
        }
        w10 += k;
        ((uint32_t*) out)[10] = v10;
        assert(k >= 1 && k <= 3);
        uint32_t v11 = 0;
        for (size_t i = 0; i < k; ++i) {
          v11 |= ((uint32_t) ((uint8_t) w11[i])) << (i * 8);
        }
        w11 += k;
        ((uint32_t*) out)[11] = v11;
        assert(k >= 1 && k <= 3);
        uint32_t v12 = 0;
        for (size_t i = 0; i < k; ++i) {
          v12 |= ((uint32_t) ((uint8_t) w12[i])) << (i * 8);
        }
        w12 += k;
        ((uint32_t*) out)[12] = v12;
        assert(k >= 1 && k <= 3);
        uint32_t v13 = 0;
        for (size_t i = 0; i < k; ++i) {
          v13 |= ((uint32_t) ((uint8_t) w13[i])) << (i * 8);
        }
        w13 += k;
        ((uint32_t*) out)[13] = v13;
        assert(k >= 1 && k <= 3);
        uint32_t v14 = 0;
        for (size_t i = 0; i < k; ++i) {
          v14 |= ((uint32_t) ((uint8_t) w14[i])) << (i * 8);
        }
        w14 += k;
        ((uint32_t*) out)[14] = v14;
        assert(k >= 1 && k <= 3);
        uint32_t v15 = 0;
        for (size_t i = 0; i < k; ++i) {
          v15 |= ((uint32_t) ((uint8_t) w15[i])) << (i * 8);
        }
        w15 += k;
        ((uint32_t*) out)[15] = v15;
        out += 16 * rem_step;
      }


      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights = (const int8_t*)((intptr_t) weights + nc * kc);
  } while (--g != 0);
}
