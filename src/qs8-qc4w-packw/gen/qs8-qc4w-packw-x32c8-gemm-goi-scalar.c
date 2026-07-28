// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/unaligned.h"

// Sign extend 4-bit to 8-bit
inline static int8_t sign_extend_int4(uint8_t v) {
  return (int8_t)(v << 4) >> 4;
}

void xnn_qs8_qc4w_packw_gemm_goi_ukernel_x32c8__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* weights,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 32);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);
  assert(kc % 2 == 0);

  uint8_t* out = (uint8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  // QS4 or QC4UW

  const size_t mock_kc = kc >> 1; // kc in bytes (each byte has 2 weights)
  const uint32_t izp = (uint32_t) params->input_zero_point + 0;
  const uint32_t kernel_zero_point = (uint32_t) params->kernel_zero_point;

  do {
    // NC main loop multiple of 32
    const uint8_t* w0 = weights;
    size_t n = nc;
    for (; n >= 32; n -= 32) {
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < 32; ++i) {
          packed_b[i] = b[i] * 16;
        }
        b += 32;
      } else {
        for (size_t i = 0; i < 32; ++i) {
          packed_b[i] = 0;
        }
      }
      out += 32 * sizeof(int32_t);

      const uint8_t* w1 = w0 + mock_kc;
      const uint8_t* w2 = w1 + mock_kc;
      const uint8_t* w3 = w2 + mock_kc;
      const uint8_t* w4 = w3 + mock_kc;
      const uint8_t* w5 = w4 + mock_kc;
      const uint8_t* w6 = w5 + mock_kc;
      const uint8_t* w7 = w6 + mock_kc;
      const uint8_t* w8 = w7 + mock_kc;
      const uint8_t* w9 = w8 + mock_kc;
      const uint8_t* w10 = w9 + mock_kc;
      const uint8_t* w11 = w10 + mock_kc;
      const uint8_t* w12 = w11 + mock_kc;
      const uint8_t* w13 = w12 + mock_kc;
      const uint8_t* w14 = w13 + mock_kc;
      const uint8_t* w15 = w14 + mock_kc;
      const uint8_t* w16 = w15 + mock_kc;
      const uint8_t* w17 = w16 + mock_kc;
      const uint8_t* w18 = w17 + mock_kc;
      const uint8_t* w19 = w18 + mock_kc;
      const uint8_t* w20 = w19 + mock_kc;
      const uint8_t* w21 = w20 + mock_kc;
      const uint8_t* w22 = w21 + mock_kc;
      const uint8_t* w23 = w22 + mock_kc;
      const uint8_t* w24 = w23 + mock_kc;
      const uint8_t* w25 = w24 + mock_kc;
      const uint8_t* w26 = w25 + mock_kc;
      const uint8_t* w27 = w26 + mock_kc;
      const uint8_t* w28 = w27 + mock_kc;
      const uint8_t* w29 = w28 + mock_kc;
      const uint8_t* w30 = w29 + mock_kc;
      const uint8_t* w31 = w30 + mock_kc;

      int32_t ksum0 = 0;
      int32_t ksum1 = 0;
      int32_t ksum2 = 0;
      int32_t ksum3 = 0;
      int32_t ksum4 = 0;
      int32_t ksum5 = 0;
      int32_t ksum6 = 0;
      int32_t ksum7 = 0;
      int32_t ksum8 = 0;
      int32_t ksum9 = 0;
      int32_t ksum10 = 0;
      int32_t ksum11 = 0;
      int32_t ksum12 = 0;
      int32_t ksum13 = 0;
      int32_t ksum14 = 0;
      int32_t ksum15 = 0;
      int32_t ksum16 = 0;
      int32_t ksum17 = 0;
      int32_t ksum18 = 0;
      int32_t ksum19 = 0;
      int32_t ksum20 = 0;
      int32_t ksum21 = 0;
      int32_t ksum22 = 0;
      int32_t ksum23 = 0;
      int32_t ksum24 = 0;
      int32_t ksum25 = 0;
      int32_t ksum26 = 0;
      int32_t ksum27 = 0;
      int32_t ksum28 = 0;
      int32_t ksum29 = 0;
      int32_t ksum30 = 0;
      int32_t ksum31 = 0;

      // KC main loop multiple of 8 (which is 8)
      size_t k = mock_kc;
      for (; k >= 8; k -= 8) {
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w0[i >> 1];
          const uint8_t byte_hi = w0[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[0 * 8 + i] = kv;
            ksum0 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[0 * 8 + i] = kv;
            ksum0 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w0 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w1[i >> 1];
          const uint8_t byte_hi = w1[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[1 * 8 + i] = kv;
            ksum1 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[1 * 8 + i] = kv;
            ksum1 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w1 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w2[i >> 1];
          const uint8_t byte_hi = w2[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[2 * 8 + i] = kv;
            ksum2 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[2 * 8 + i] = kv;
            ksum2 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w2 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w3[i >> 1];
          const uint8_t byte_hi = w3[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[3 * 8 + i] = kv;
            ksum3 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[3 * 8 + i] = kv;
            ksum3 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w3 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w4[i >> 1];
          const uint8_t byte_hi = w4[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[4 * 8 + i] = kv;
            ksum4 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[4 * 8 + i] = kv;
            ksum4 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w4 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w5[i >> 1];
          const uint8_t byte_hi = w5[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[5 * 8 + i] = kv;
            ksum5 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[5 * 8 + i] = kv;
            ksum5 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w5 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w6[i >> 1];
          const uint8_t byte_hi = w6[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[6 * 8 + i] = kv;
            ksum6 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[6 * 8 + i] = kv;
            ksum6 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w6 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w7[i >> 1];
          const uint8_t byte_hi = w7[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[7 * 8 + i] = kv;
            ksum7 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[7 * 8 + i] = kv;
            ksum7 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w7 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w8[i >> 1];
          const uint8_t byte_hi = w8[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[8 * 8 + i] = kv;
            ksum8 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[8 * 8 + i] = kv;
            ksum8 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w8 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w9[i >> 1];
          const uint8_t byte_hi = w9[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[9 * 8 + i] = kv;
            ksum9 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[9 * 8 + i] = kv;
            ksum9 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w9 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w10[i >> 1];
          const uint8_t byte_hi = w10[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[10 * 8 + i] = kv;
            ksum10 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[10 * 8 + i] = kv;
            ksum10 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w10 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w11[i >> 1];
          const uint8_t byte_hi = w11[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[11 * 8 + i] = kv;
            ksum11 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[11 * 8 + i] = kv;
            ksum11 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w11 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w12[i >> 1];
          const uint8_t byte_hi = w12[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[12 * 8 + i] = kv;
            ksum12 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[12 * 8 + i] = kv;
            ksum12 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w12 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w13[i >> 1];
          const uint8_t byte_hi = w13[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[13 * 8 + i] = kv;
            ksum13 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[13 * 8 + i] = kv;
            ksum13 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w13 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w14[i >> 1];
          const uint8_t byte_hi = w14[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[14 * 8 + i] = kv;
            ksum14 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[14 * 8 + i] = kv;
            ksum14 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w14 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w15[i >> 1];
          const uint8_t byte_hi = w15[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[15 * 8 + i] = kv;
            ksum15 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[15 * 8 + i] = kv;
            ksum15 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w15 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w16[i >> 1];
          const uint8_t byte_hi = w16[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[16 * 8 + i] = kv;
            ksum16 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[16 * 8 + i] = kv;
            ksum16 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w16 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w17[i >> 1];
          const uint8_t byte_hi = w17[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[17 * 8 + i] = kv;
            ksum17 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[17 * 8 + i] = kv;
            ksum17 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w17 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w18[i >> 1];
          const uint8_t byte_hi = w18[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[18 * 8 + i] = kv;
            ksum18 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[18 * 8 + i] = kv;
            ksum18 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w18 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w19[i >> 1];
          const uint8_t byte_hi = w19[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[19 * 8 + i] = kv;
            ksum19 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[19 * 8 + i] = kv;
            ksum19 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w19 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w20[i >> 1];
          const uint8_t byte_hi = w20[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[20 * 8 + i] = kv;
            ksum20 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[20 * 8 + i] = kv;
            ksum20 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w20 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w21[i >> 1];
          const uint8_t byte_hi = w21[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[21 * 8 + i] = kv;
            ksum21 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[21 * 8 + i] = kv;
            ksum21 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w21 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w22[i >> 1];
          const uint8_t byte_hi = w22[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[22 * 8 + i] = kv;
            ksum22 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[22 * 8 + i] = kv;
            ksum22 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w22 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w23[i >> 1];
          const uint8_t byte_hi = w23[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[23 * 8 + i] = kv;
            ksum23 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[23 * 8 + i] = kv;
            ksum23 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w23 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w24[i >> 1];
          const uint8_t byte_hi = w24[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[24 * 8 + i] = kv;
            ksum24 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[24 * 8 + i] = kv;
            ksum24 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w24 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w25[i >> 1];
          const uint8_t byte_hi = w25[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[25 * 8 + i] = kv;
            ksum25 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[25 * 8 + i] = kv;
            ksum25 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w25 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w26[i >> 1];
          const uint8_t byte_hi = w26[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[26 * 8 + i] = kv;
            ksum26 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[26 * 8 + i] = kv;
            ksum26 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w26 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w27[i >> 1];
          const uint8_t byte_hi = w27[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[27 * 8 + i] = kv;
            ksum27 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[27 * 8 + i] = kv;
            ksum27 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w27 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w28[i >> 1];
          const uint8_t byte_hi = w28[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[28 * 8 + i] = kv;
            ksum28 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[28 * 8 + i] = kv;
            ksum28 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w28 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w29[i >> 1];
          const uint8_t byte_hi = w29[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[29 * 8 + i] = kv;
            ksum29 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[29 * 8 + i] = kv;
            ksum29 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w29 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w30[i >> 1];
          const uint8_t byte_hi = w30[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[30 * 8 + i] = kv;
            ksum30 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[30 * 8 + i] = kv;
            ksum30 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w30 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w31[i >> 1];
          const uint8_t byte_hi = w31[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[31 * 8 + i] = kv;
            ksum31 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[31 * 8 + i] = kv;
            ksum31 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w31 += 8;
        out += 32 * 8;
      }

      // KC remainder of 1..7 bytes
      if (k != 0) {
        uint8_t temp_w0[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w0[i] = w0[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w0[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w1[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w1[i] = w1[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w1[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w2[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w2[i] = w2[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w2[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w3[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w3[i] = w3[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w3[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w4[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w4[i] = w4[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w4[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w5[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w5[i] = w5[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w5[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w6[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w6[i] = w6[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w6[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w7[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w7[i] = w7[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w7[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w8[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w8[i] = w8[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w8[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w9[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w9[i] = w9[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w9[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w10[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w10[i] = w10[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w10[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w11[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w11[i] = w11[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w11[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w12[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w12[i] = w12[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w12[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w13[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w13[i] = w13[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w13[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w14[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w14[i] = w14[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w14[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w15[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w15[i] = w15[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w15[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w16[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w16[i] = w16[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w16[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w17[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w17[i] = w17[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w17[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w18[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w18[i] = w18[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w18[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w19[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w19[i] = w19[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w19[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w20[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w20[i] = w20[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w20[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w21[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w21[i] = w21[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w21[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w22[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w22[i] = w22[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w22[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w23[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w23[i] = w23[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w23[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w24[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w24[i] = w24[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w24[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w25[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w25[i] = w25[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w25[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w26[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w26[i] = w26[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w26[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w27[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w27[i] = w27[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w27[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w28[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w28[i] = w28[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w28[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w29[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w29[i] = w29[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w29[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w30[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w30[i] = w30[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w30[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w31[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w31[i] = w31[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w31[i] = kernel_zero_point * 0x11;
        }

        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w0[i >> 1];
          const uint8_t byte_hi = temp_w0[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[0 * 8 + i] = kv;
            ksum0 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[0 * 8 + i] = kv;
            ksum0 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w0 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w1[i >> 1];
          const uint8_t byte_hi = temp_w1[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[1 * 8 + i] = kv;
            ksum1 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[1 * 8 + i] = kv;
            ksum1 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w1 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w2[i >> 1];
          const uint8_t byte_hi = temp_w2[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[2 * 8 + i] = kv;
            ksum2 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[2 * 8 + i] = kv;
            ksum2 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w2 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w3[i >> 1];
          const uint8_t byte_hi = temp_w3[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[3 * 8 + i] = kv;
            ksum3 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[3 * 8 + i] = kv;
            ksum3 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w3 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w4[i >> 1];
          const uint8_t byte_hi = temp_w4[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[4 * 8 + i] = kv;
            ksum4 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[4 * 8 + i] = kv;
            ksum4 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w4 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w5[i >> 1];
          const uint8_t byte_hi = temp_w5[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[5 * 8 + i] = kv;
            ksum5 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[5 * 8 + i] = kv;
            ksum5 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w5 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w6[i >> 1];
          const uint8_t byte_hi = temp_w6[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[6 * 8 + i] = kv;
            ksum6 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[6 * 8 + i] = kv;
            ksum6 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w6 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w7[i >> 1];
          const uint8_t byte_hi = temp_w7[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[7 * 8 + i] = kv;
            ksum7 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[7 * 8 + i] = kv;
            ksum7 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w7 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w8[i >> 1];
          const uint8_t byte_hi = temp_w8[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[8 * 8 + i] = kv;
            ksum8 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[8 * 8 + i] = kv;
            ksum8 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w8 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w9[i >> 1];
          const uint8_t byte_hi = temp_w9[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[9 * 8 + i] = kv;
            ksum9 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[9 * 8 + i] = kv;
            ksum9 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w9 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w10[i >> 1];
          const uint8_t byte_hi = temp_w10[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[10 * 8 + i] = kv;
            ksum10 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[10 * 8 + i] = kv;
            ksum10 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w10 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w11[i >> 1];
          const uint8_t byte_hi = temp_w11[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[11 * 8 + i] = kv;
            ksum11 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[11 * 8 + i] = kv;
            ksum11 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w11 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w12[i >> 1];
          const uint8_t byte_hi = temp_w12[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[12 * 8 + i] = kv;
            ksum12 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[12 * 8 + i] = kv;
            ksum12 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w12 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w13[i >> 1];
          const uint8_t byte_hi = temp_w13[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[13 * 8 + i] = kv;
            ksum13 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[13 * 8 + i] = kv;
            ksum13 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w13 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w14[i >> 1];
          const uint8_t byte_hi = temp_w14[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[14 * 8 + i] = kv;
            ksum14 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[14 * 8 + i] = kv;
            ksum14 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w14 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w15[i >> 1];
          const uint8_t byte_hi = temp_w15[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[15 * 8 + i] = kv;
            ksum15 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[15 * 8 + i] = kv;
            ksum15 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w15 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w16[i >> 1];
          const uint8_t byte_hi = temp_w16[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[16 * 8 + i] = kv;
            ksum16 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[16 * 8 + i] = kv;
            ksum16 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w16 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w17[i >> 1];
          const uint8_t byte_hi = temp_w17[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[17 * 8 + i] = kv;
            ksum17 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[17 * 8 + i] = kv;
            ksum17 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w17 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w18[i >> 1];
          const uint8_t byte_hi = temp_w18[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[18 * 8 + i] = kv;
            ksum18 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[18 * 8 + i] = kv;
            ksum18 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w18 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w19[i >> 1];
          const uint8_t byte_hi = temp_w19[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[19 * 8 + i] = kv;
            ksum19 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[19 * 8 + i] = kv;
            ksum19 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w19 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w20[i >> 1];
          const uint8_t byte_hi = temp_w20[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[20 * 8 + i] = kv;
            ksum20 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[20 * 8 + i] = kv;
            ksum20 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w20 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w21[i >> 1];
          const uint8_t byte_hi = temp_w21[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[21 * 8 + i] = kv;
            ksum21 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[21 * 8 + i] = kv;
            ksum21 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w21 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w22[i >> 1];
          const uint8_t byte_hi = temp_w22[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[22 * 8 + i] = kv;
            ksum22 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[22 * 8 + i] = kv;
            ksum22 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w22 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w23[i >> 1];
          const uint8_t byte_hi = temp_w23[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[23 * 8 + i] = kv;
            ksum23 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[23 * 8 + i] = kv;
            ksum23 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w23 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w24[i >> 1];
          const uint8_t byte_hi = temp_w24[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[24 * 8 + i] = kv;
            ksum24 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[24 * 8 + i] = kv;
            ksum24 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w24 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w25[i >> 1];
          const uint8_t byte_hi = temp_w25[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[25 * 8 + i] = kv;
            ksum25 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[25 * 8 + i] = kv;
            ksum25 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w25 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w26[i >> 1];
          const uint8_t byte_hi = temp_w26[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[26 * 8 + i] = kv;
            ksum26 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[26 * 8 + i] = kv;
            ksum26 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w26 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w27[i >> 1];
          const uint8_t byte_hi = temp_w27[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[27 * 8 + i] = kv;
            ksum27 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[27 * 8 + i] = kv;
            ksum27 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w27 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w28[i >> 1];
          const uint8_t byte_hi = temp_w28[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[28 * 8 + i] = kv;
            ksum28 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[28 * 8 + i] = kv;
            ksum28 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w28 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w29[i >> 1];
          const uint8_t byte_hi = temp_w29[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[29 * 8 + i] = kv;
            ksum29 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[29 * 8 + i] = kv;
            ksum29 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w29 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w30[i >> 1];
          const uint8_t byte_hi = temp_w30[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[30 * 8 + i] = kv;
            ksum30 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[30 * 8 + i] = kv;
            ksum30 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w30 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w31[i >> 1];
          const uint8_t byte_hi = temp_w31[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[31 * 8 + i] = kv;
            ksum31 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[31 * 8 + i] = kv;
            ksum31 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w31 += k;
        out += 32 * 8;
      }

      packed_b[0] -= ksum0 * izp * 16;
      packed_b[1] -= ksum1 * izp * 16;
      packed_b[2] -= ksum2 * izp * 16;
      packed_b[3] -= ksum3 * izp * 16;
      packed_b[4] -= ksum4 * izp * 16;
      packed_b[5] -= ksum5 * izp * 16;
      packed_b[6] -= ksum6 * izp * 16;
      packed_b[7] -= ksum7 * izp * 16;
      packed_b[8] -= ksum8 * izp * 16;
      packed_b[9] -= ksum9 * izp * 16;
      packed_b[10] -= ksum10 * izp * 16;
      packed_b[11] -= ksum11 * izp * 16;
      packed_b[12] -= ksum12 * izp * 16;
      packed_b[13] -= ksum13 * izp * 16;
      packed_b[14] -= ksum14 * izp * 16;
      packed_b[15] -= ksum15 * izp * 16;
      packed_b[16] -= ksum16 * izp * 16;
      packed_b[17] -= ksum17 * izp * 16;
      packed_b[18] -= ksum18 * izp * 16;
      packed_b[19] -= ksum19 * izp * 16;
      packed_b[20] -= ksum20 * izp * 16;
      packed_b[21] -= ksum21 * izp * 16;
      packed_b[22] -= ksum22 * izp * 16;
      packed_b[23] -= ksum23 * izp * 16;
      packed_b[24] -= ksum24 * izp * 16;
      packed_b[25] -= ksum25 * izp * 16;
      packed_b[26] -= ksum26 * izp * 16;
      packed_b[27] -= ksum27 * izp * 16;
      packed_b[28] -= ksum28 * izp * 16;
      packed_b[29] -= ksum29 * izp * 16;
      packed_b[30] -= ksum30 * izp * 16;
      packed_b[31] -= ksum31 * izp * 16;
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w31;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          packed_b[i] = b[i] * 16;
        }
        b += n;
      } else {
        for (size_t i = 0; i < n; ++i) {
          packed_b[i] = 0;
        }
      }
      out += 32 * sizeof(int32_t);

      // Clamp weight pointers
      const uint8_t* w1 = w0 + mock_kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint8_t* w2 = w1 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint8_t* w3 = w2 + mock_kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint8_t* w4 = w3 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint8_t* w5 = w4 + mock_kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint8_t* w6 = w5 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint8_t* w7 = w6 + mock_kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint8_t* w8 = w7 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint8_t* w9 = w8 + mock_kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint8_t* w10 = w9 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint8_t* w11 = w10 + mock_kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint8_t* w12 = w11 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint8_t* w13 = w12 + mock_kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint8_t* w14 = w13 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const uint8_t* w15 = w14 + mock_kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const uint8_t* w16 = w15 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const uint8_t* w17 = w16 + mock_kc;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const uint8_t* w18 = w17 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const uint8_t* w19 = w18 + mock_kc;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const uint8_t* w20 = w19 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const uint8_t* w21 = w20 + mock_kc;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const uint8_t* w22 = w21 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const uint8_t* w23 = w22 + mock_kc;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const uint8_t* w24 = w23 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const uint8_t* w25 = w24 + mock_kc;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const uint8_t* w26 = w25 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const uint8_t* w27 = w26 + mock_kc;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const uint8_t* w28 = w27 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const uint8_t* w29 = w28 + mock_kc;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const uint8_t* w30 = w29 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }

      int32_t ksum0 = 0;
      int32_t ksum1 = 0;
      int32_t ksum2 = 0;
      int32_t ksum3 = 0;
      int32_t ksum4 = 0;
      int32_t ksum5 = 0;
      int32_t ksum6 = 0;
      int32_t ksum7 = 0;
      int32_t ksum8 = 0;
      int32_t ksum9 = 0;
      int32_t ksum10 = 0;
      int32_t ksum11 = 0;
      int32_t ksum12 = 0;
      int32_t ksum13 = 0;
      int32_t ksum14 = 0;
      int32_t ksum15 = 0;
      int32_t ksum16 = 0;
      int32_t ksum17 = 0;
      int32_t ksum18 = 0;
      int32_t ksum19 = 0;
      int32_t ksum20 = 0;
      int32_t ksum21 = 0;
      int32_t ksum22 = 0;
      int32_t ksum23 = 0;
      int32_t ksum24 = 0;
      int32_t ksum25 = 0;
      int32_t ksum26 = 0;
      int32_t ksum27 = 0;
      int32_t ksum28 = 0;
      int32_t ksum29 = 0;
      int32_t ksum30 = 0;

      // KC main loop multiple of 8 (which is 8)
      size_t k = mock_kc;
      for (; k >= 8; k -= 8) {
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w0[i >> 1];
          const uint8_t byte_hi = w0[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[0 * 8 + i] = kv;
            ksum0 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[0 * 8 + i] = kv;
            ksum0 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w0 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w1[i >> 1];
          const uint8_t byte_hi = w1[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[1 * 8 + i] = kv;
            ksum1 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[1 * 8 + i] = kv;
            ksum1 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w1 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w2[i >> 1];
          const uint8_t byte_hi = w2[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[2 * 8 + i] = kv;
            ksum2 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[2 * 8 + i] = kv;
            ksum2 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w2 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w3[i >> 1];
          const uint8_t byte_hi = w3[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[3 * 8 + i] = kv;
            ksum3 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[3 * 8 + i] = kv;
            ksum3 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w3 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w4[i >> 1];
          const uint8_t byte_hi = w4[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[4 * 8 + i] = kv;
            ksum4 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[4 * 8 + i] = kv;
            ksum4 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w4 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w5[i >> 1];
          const uint8_t byte_hi = w5[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[5 * 8 + i] = kv;
            ksum5 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[5 * 8 + i] = kv;
            ksum5 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w5 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w6[i >> 1];
          const uint8_t byte_hi = w6[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[6 * 8 + i] = kv;
            ksum6 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[6 * 8 + i] = kv;
            ksum6 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w6 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w7[i >> 1];
          const uint8_t byte_hi = w7[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[7 * 8 + i] = kv;
            ksum7 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[7 * 8 + i] = kv;
            ksum7 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w7 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w8[i >> 1];
          const uint8_t byte_hi = w8[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[8 * 8 + i] = kv;
            ksum8 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[8 * 8 + i] = kv;
            ksum8 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w8 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w9[i >> 1];
          const uint8_t byte_hi = w9[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[9 * 8 + i] = kv;
            ksum9 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[9 * 8 + i] = kv;
            ksum9 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w9 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w10[i >> 1];
          const uint8_t byte_hi = w10[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[10 * 8 + i] = kv;
            ksum10 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[10 * 8 + i] = kv;
            ksum10 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w10 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w11[i >> 1];
          const uint8_t byte_hi = w11[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[11 * 8 + i] = kv;
            ksum11 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[11 * 8 + i] = kv;
            ksum11 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w11 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w12[i >> 1];
          const uint8_t byte_hi = w12[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[12 * 8 + i] = kv;
            ksum12 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[12 * 8 + i] = kv;
            ksum12 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w12 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w13[i >> 1];
          const uint8_t byte_hi = w13[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[13 * 8 + i] = kv;
            ksum13 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[13 * 8 + i] = kv;
            ksum13 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w13 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w14[i >> 1];
          const uint8_t byte_hi = w14[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[14 * 8 + i] = kv;
            ksum14 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[14 * 8 + i] = kv;
            ksum14 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w14 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w15[i >> 1];
          const uint8_t byte_hi = w15[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[15 * 8 + i] = kv;
            ksum15 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[15 * 8 + i] = kv;
            ksum15 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w15 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w16[i >> 1];
          const uint8_t byte_hi = w16[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[16 * 8 + i] = kv;
            ksum16 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[16 * 8 + i] = kv;
            ksum16 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w16 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w17[i >> 1];
          const uint8_t byte_hi = w17[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[17 * 8 + i] = kv;
            ksum17 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[17 * 8 + i] = kv;
            ksum17 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w17 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w18[i >> 1];
          const uint8_t byte_hi = w18[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[18 * 8 + i] = kv;
            ksum18 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[18 * 8 + i] = kv;
            ksum18 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w18 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w19[i >> 1];
          const uint8_t byte_hi = w19[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[19 * 8 + i] = kv;
            ksum19 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[19 * 8 + i] = kv;
            ksum19 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w19 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w20[i >> 1];
          const uint8_t byte_hi = w20[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[20 * 8 + i] = kv;
            ksum20 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[20 * 8 + i] = kv;
            ksum20 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w20 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w21[i >> 1];
          const uint8_t byte_hi = w21[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[21 * 8 + i] = kv;
            ksum21 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[21 * 8 + i] = kv;
            ksum21 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w21 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w22[i >> 1];
          const uint8_t byte_hi = w22[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[22 * 8 + i] = kv;
            ksum22 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[22 * 8 + i] = kv;
            ksum22 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w22 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w23[i >> 1];
          const uint8_t byte_hi = w23[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[23 * 8 + i] = kv;
            ksum23 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[23 * 8 + i] = kv;
            ksum23 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w23 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w24[i >> 1];
          const uint8_t byte_hi = w24[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[24 * 8 + i] = kv;
            ksum24 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[24 * 8 + i] = kv;
            ksum24 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w24 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w25[i >> 1];
          const uint8_t byte_hi = w25[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[25 * 8 + i] = kv;
            ksum25 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[25 * 8 + i] = kv;
            ksum25 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w25 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w26[i >> 1];
          const uint8_t byte_hi = w26[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[26 * 8 + i] = kv;
            ksum26 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[26 * 8 + i] = kv;
            ksum26 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w26 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w27[i >> 1];
          const uint8_t byte_hi = w27[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[27 * 8 + i] = kv;
            ksum27 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[27 * 8 + i] = kv;
            ksum27 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w27 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w28[i >> 1];
          const uint8_t byte_hi = w28[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[28 * 8 + i] = kv;
            ksum28 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[28 * 8 + i] = kv;
            ksum28 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w28 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w29[i >> 1];
          const uint8_t byte_hi = w29[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[29 * 8 + i] = kv;
            ksum29 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[29 * 8 + i] = kv;
            ksum29 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w29 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = w30[i >> 1];
          const uint8_t byte_hi = w30[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[30 * 8 + i] = kv;
            ksum30 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[30 * 8 + i] = kv;
            ksum30 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w30 += 8;

        // Pad the remaining row(s)
        for (size_t i = 0; i < 8; ++i) {
          out[31 * 8 + i] = 0;
        }
        out += 32 * 8;
      }

      // KC remainder of 1..7 bytes
      if (k != 0) {
        uint8_t temp_w0[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w0[i] = w0[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w0[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w1[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w1[i] = w1[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w1[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w2[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w2[i] = w2[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w2[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w3[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w3[i] = w3[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w3[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w4[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w4[i] = w4[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w4[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w5[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w5[i] = w5[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w5[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w6[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w6[i] = w6[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w6[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w7[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w7[i] = w7[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w7[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w8[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w8[i] = w8[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w8[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w9[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w9[i] = w9[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w9[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w10[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w10[i] = w10[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w10[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w11[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w11[i] = w11[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w11[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w12[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w12[i] = w12[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w12[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w13[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w13[i] = w13[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w13[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w14[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w14[i] = w14[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w14[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w15[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w15[i] = w15[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w15[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w16[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w16[i] = w16[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w16[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w17[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w17[i] = w17[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w17[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w18[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w18[i] = w18[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w18[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w19[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w19[i] = w19[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w19[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w20[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w20[i] = w20[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w20[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w21[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w21[i] = w21[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w21[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w22[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w22[i] = w22[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w22[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w23[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w23[i] = w23[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w23[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w24[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w24[i] = w24[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w24[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w25[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w25[i] = w25[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w25[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w26[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w26[i] = w26[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w26[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w27[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w27[i] = w27[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w27[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w28[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w28[i] = w28[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w28[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w29[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w29[i] = w29[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w29[i] = kernel_zero_point * 0x11;
        }
        uint8_t temp_w30[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w30[i] = w30[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w30[i] = kernel_zero_point * 0x11;
        }

        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w0[i >> 1];
          const uint8_t byte_hi = temp_w0[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[0 * 8 + i] = kv;
            ksum0 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[0 * 8 + i] = kv;
            ksum0 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w0 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w1[i >> 1];
          const uint8_t byte_hi = temp_w1[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[1 * 8 + i] = kv;
            ksum1 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[1 * 8 + i] = kv;
            ksum1 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w1 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w2[i >> 1];
          const uint8_t byte_hi = temp_w2[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[2 * 8 + i] = kv;
            ksum2 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[2 * 8 + i] = kv;
            ksum2 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w2 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w3[i >> 1];
          const uint8_t byte_hi = temp_w3[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[3 * 8 + i] = kv;
            ksum3 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[3 * 8 + i] = kv;
            ksum3 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w3 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w4[i >> 1];
          const uint8_t byte_hi = temp_w4[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[4 * 8 + i] = kv;
            ksum4 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[4 * 8 + i] = kv;
            ksum4 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w4 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w5[i >> 1];
          const uint8_t byte_hi = temp_w5[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[5 * 8 + i] = kv;
            ksum5 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[5 * 8 + i] = kv;
            ksum5 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w5 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w6[i >> 1];
          const uint8_t byte_hi = temp_w6[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[6 * 8 + i] = kv;
            ksum6 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[6 * 8 + i] = kv;
            ksum6 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w6 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w7[i >> 1];
          const uint8_t byte_hi = temp_w7[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[7 * 8 + i] = kv;
            ksum7 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[7 * 8 + i] = kv;
            ksum7 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w7 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w8[i >> 1];
          const uint8_t byte_hi = temp_w8[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[8 * 8 + i] = kv;
            ksum8 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[8 * 8 + i] = kv;
            ksum8 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w8 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w9[i >> 1];
          const uint8_t byte_hi = temp_w9[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[9 * 8 + i] = kv;
            ksum9 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[9 * 8 + i] = kv;
            ksum9 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w9 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w10[i >> 1];
          const uint8_t byte_hi = temp_w10[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[10 * 8 + i] = kv;
            ksum10 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[10 * 8 + i] = kv;
            ksum10 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w10 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w11[i >> 1];
          const uint8_t byte_hi = temp_w11[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[11 * 8 + i] = kv;
            ksum11 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[11 * 8 + i] = kv;
            ksum11 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w11 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w12[i >> 1];
          const uint8_t byte_hi = temp_w12[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[12 * 8 + i] = kv;
            ksum12 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[12 * 8 + i] = kv;
            ksum12 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w12 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w13[i >> 1];
          const uint8_t byte_hi = temp_w13[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[13 * 8 + i] = kv;
            ksum13 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[13 * 8 + i] = kv;
            ksum13 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w13 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w14[i >> 1];
          const uint8_t byte_hi = temp_w14[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[14 * 8 + i] = kv;
            ksum14 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[14 * 8 + i] = kv;
            ksum14 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w14 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w15[i >> 1];
          const uint8_t byte_hi = temp_w15[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[15 * 8 + i] = kv;
            ksum15 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[15 * 8 + i] = kv;
            ksum15 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w15 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w16[i >> 1];
          const uint8_t byte_hi = temp_w16[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[16 * 8 + i] = kv;
            ksum16 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[16 * 8 + i] = kv;
            ksum16 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w16 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w17[i >> 1];
          const uint8_t byte_hi = temp_w17[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[17 * 8 + i] = kv;
            ksum17 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[17 * 8 + i] = kv;
            ksum17 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w17 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w18[i >> 1];
          const uint8_t byte_hi = temp_w18[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[18 * 8 + i] = kv;
            ksum18 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[18 * 8 + i] = kv;
            ksum18 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w18 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w19[i >> 1];
          const uint8_t byte_hi = temp_w19[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[19 * 8 + i] = kv;
            ksum19 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[19 * 8 + i] = kv;
            ksum19 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w19 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w20[i >> 1];
          const uint8_t byte_hi = temp_w20[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[20 * 8 + i] = kv;
            ksum20 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[20 * 8 + i] = kv;
            ksum20 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w20 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w21[i >> 1];
          const uint8_t byte_hi = temp_w21[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[21 * 8 + i] = kv;
            ksum21 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[21 * 8 + i] = kv;
            ksum21 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w21 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w22[i >> 1];
          const uint8_t byte_hi = temp_w22[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[22 * 8 + i] = kv;
            ksum22 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[22 * 8 + i] = kv;
            ksum22 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w22 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w23[i >> 1];
          const uint8_t byte_hi = temp_w23[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[23 * 8 + i] = kv;
            ksum23 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[23 * 8 + i] = kv;
            ksum23 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w23 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w24[i >> 1];
          const uint8_t byte_hi = temp_w24[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[24 * 8 + i] = kv;
            ksum24 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[24 * 8 + i] = kv;
            ksum24 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w24 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w25[i >> 1];
          const uint8_t byte_hi = temp_w25[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[25 * 8 + i] = kv;
            ksum25 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[25 * 8 + i] = kv;
            ksum25 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w25 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w26[i >> 1];
          const uint8_t byte_hi = temp_w26[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[26 * 8 + i] = kv;
            ksum26 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[26 * 8 + i] = kv;
            ksum26 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w26 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w27[i >> 1];
          const uint8_t byte_hi = temp_w27[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[27 * 8 + i] = kv;
            ksum27 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[27 * 8 + i] = kv;
            ksum27 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w27 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w28[i >> 1];
          const uint8_t byte_hi = temp_w28[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[28 * 8 + i] = kv;
            ksum28 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[28 * 8 + i] = kv;
            ksum28 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w28 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w29[i >> 1];
          const uint8_t byte_hi = temp_w29[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[29 * 8 + i] = kv;
            ksum29 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[29 * 8 + i] = kv;
            ksum29 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w29 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t byte_lo = temp_w30[i >> 1];
          const uint8_t byte_hi = temp_w30[(i >> 1) + 4];
          uint8_t val_lo = (i & 1) ? (byte_lo >> 4) : (byte_lo & 0xF);
          uint8_t val_hi = (i & 1) ? (byte_hi >> 4) : (byte_hi & 0xF);

          if (kernel_zero_point == 0) {
            const uint8_t kv = val_lo | (val_hi << 4);
            out[30 * 8 + i] = kv;
            ksum30 += sign_extend_int4(val_lo) + sign_extend_int4(val_hi);
          } else {
            const uint8_t kv = (val_lo | (val_hi << 4)) ^ 0x88;
            out[30 * 8 + i] = kv;
            ksum30 += (int32_t)val_lo + (int32_t)val_hi - 16;
          }
        }
        w30 += k;

        // Pad the remaining row(s)
        for (size_t i = 0; i < 8; ++i) {
          out[31 * 8 + i] = 0;
        }
        out += 32 * 8;
      }

      if (0 < n) {
        packed_b[0] -= ksum0 * izp * 16;
      }
      if (1 < n) {
        packed_b[1] -= ksum1 * izp * 16;
      }
      if (2 < n) {
        packed_b[2] -= ksum2 * izp * 16;
      }
      if (3 < n) {
        packed_b[3] -= ksum3 * izp * 16;
      }
      if (4 < n) {
        packed_b[4] -= ksum4 * izp * 16;
      }
      if (5 < n) {
        packed_b[5] -= ksum5 * izp * 16;
      }
      if (6 < n) {
        packed_b[6] -= ksum6 * izp * 16;
      }
      if (7 < n) {
        packed_b[7] -= ksum7 * izp * 16;
      }
      if (8 < n) {
        packed_b[8] -= ksum8 * izp * 16;
      }
      if (9 < n) {
        packed_b[9] -= ksum9 * izp * 16;
      }
      if (10 < n) {
        packed_b[10] -= ksum10 * izp * 16;
      }
      if (11 < n) {
        packed_b[11] -= ksum11 * izp * 16;
      }
      if (12 < n) {
        packed_b[12] -= ksum12 * izp * 16;
      }
      if (13 < n) {
        packed_b[13] -= ksum13 * izp * 16;
      }
      if (14 < n) {
        packed_b[14] -= ksum14 * izp * 16;
      }
      if (15 < n) {
        packed_b[15] -= ksum15 * izp * 16;
      }
      if (16 < n) {
        packed_b[16] -= ksum16 * izp * 16;
      }
      if (17 < n) {
        packed_b[17] -= ksum17 * izp * 16;
      }
      if (18 < n) {
        packed_b[18] -= ksum18 * izp * 16;
      }
      if (19 < n) {
        packed_b[19] -= ksum19 * izp * 16;
      }
      if (20 < n) {
        packed_b[20] -= ksum20 * izp * 16;
      }
      if (21 < n) {
        packed_b[21] -= ksum21 * izp * 16;
      }
      if (22 < n) {
        packed_b[22] -= ksum22 * izp * 16;
      }
      if (23 < n) {
        packed_b[23] -= ksum23 * izp * 16;
      }
      if (24 < n) {
        packed_b[24] -= ksum24 * izp * 16;
      }
      if (25 < n) {
        packed_b[25] -= ksum25 * izp * 16;
      }
      if (26 < n) {
        packed_b[26] -= ksum26 * izp * 16;
      }
      if (27 < n) {
        packed_b[27] -= ksum27 * izp * 16;
      }
      if (28 < n) {
        packed_b[28] -= ksum28 * izp * 16;
      }
      if (29 < n) {
        packed_b[29] -= ksum29 * izp * 16;
      }
      if (30 < n) {
        packed_b[30] -= ksum30 * izp * 16;
      }
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights = (const uint8_t*)((intptr_t) weights + nc * kc);
  } while (--g != 0);
}
