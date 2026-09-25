// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-qc2w-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/packw.h"


// Sign extend 2-bit to 8-bit
inline static int8_t sign_extend_int2(uint8_t v) {
  return (int8_t)(v << 6) >> 6;
}

void xnn_qs8_qc2w_packw_gemm_goi_ukernel_x16c8__scalar(
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
  const struct xnn_qs8_qc2w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);
  assert(kc % 4 == 0);

  const size_t mock_kc = kc >> 2; // kc in bytes (each byte has 4 2-bit weights)
  const uint32_t izp = (uint32_t) params->input_zero_point + 0;

  uint8_t* out = (uint8_t*) packed_weights;
  const int32_t* b = bias;

  do {
    const uint8_t* w0 = weights;
    size_t n = nc;
    for (; n >= 16; n -= 16) {
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < 16; ++i) {
          packed_b[i] = b[i];
        }
        b += 16;
      } else {
        for (size_t i = 0; i < 16; ++i) {
          packed_b[i] = 0;
        }
      }
      out = (uint8_t*) (packed_b + 16);

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

      // KC main loop multiple of 8 bytes (32 2-bit elements)
      size_t k = mock_kc;
      for (; k >= 8; k -= 8) {
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w0[(i >> 2) + 0 * 2];
          const uint8_t b1 = w0[(i >> 2) + 1 * 2];
          const uint8_t b2 = w0[(i >> 2) + 2 * 2];
          const uint8_t b3 = w0[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[0 * 8 + i] = kv;
          ksum0 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w0 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w1[(i >> 2) + 0 * 2];
          const uint8_t b1 = w1[(i >> 2) + 1 * 2];
          const uint8_t b2 = w1[(i >> 2) + 2 * 2];
          const uint8_t b3 = w1[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[1 * 8 + i] = kv;
          ksum1 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w1 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w2[(i >> 2) + 0 * 2];
          const uint8_t b1 = w2[(i >> 2) + 1 * 2];
          const uint8_t b2 = w2[(i >> 2) + 2 * 2];
          const uint8_t b3 = w2[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[2 * 8 + i] = kv;
          ksum2 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w2 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w3[(i >> 2) + 0 * 2];
          const uint8_t b1 = w3[(i >> 2) + 1 * 2];
          const uint8_t b2 = w3[(i >> 2) + 2 * 2];
          const uint8_t b3 = w3[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[3 * 8 + i] = kv;
          ksum3 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w3 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w4[(i >> 2) + 0 * 2];
          const uint8_t b1 = w4[(i >> 2) + 1 * 2];
          const uint8_t b2 = w4[(i >> 2) + 2 * 2];
          const uint8_t b3 = w4[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[4 * 8 + i] = kv;
          ksum4 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w4 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w5[(i >> 2) + 0 * 2];
          const uint8_t b1 = w5[(i >> 2) + 1 * 2];
          const uint8_t b2 = w5[(i >> 2) + 2 * 2];
          const uint8_t b3 = w5[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[5 * 8 + i] = kv;
          ksum5 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w5 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w6[(i >> 2) + 0 * 2];
          const uint8_t b1 = w6[(i >> 2) + 1 * 2];
          const uint8_t b2 = w6[(i >> 2) + 2 * 2];
          const uint8_t b3 = w6[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[6 * 8 + i] = kv;
          ksum6 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w6 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w7[(i >> 2) + 0 * 2];
          const uint8_t b1 = w7[(i >> 2) + 1 * 2];
          const uint8_t b2 = w7[(i >> 2) + 2 * 2];
          const uint8_t b3 = w7[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[7 * 8 + i] = kv;
          ksum7 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w7 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w8[(i >> 2) + 0 * 2];
          const uint8_t b1 = w8[(i >> 2) + 1 * 2];
          const uint8_t b2 = w8[(i >> 2) + 2 * 2];
          const uint8_t b3 = w8[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[8 * 8 + i] = kv;
          ksum8 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w8 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w9[(i >> 2) + 0 * 2];
          const uint8_t b1 = w9[(i >> 2) + 1 * 2];
          const uint8_t b2 = w9[(i >> 2) + 2 * 2];
          const uint8_t b3 = w9[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[9 * 8 + i] = kv;
          ksum9 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w9 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w10[(i >> 2) + 0 * 2];
          const uint8_t b1 = w10[(i >> 2) + 1 * 2];
          const uint8_t b2 = w10[(i >> 2) + 2 * 2];
          const uint8_t b3 = w10[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[10 * 8 + i] = kv;
          ksum10 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w10 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w11[(i >> 2) + 0 * 2];
          const uint8_t b1 = w11[(i >> 2) + 1 * 2];
          const uint8_t b2 = w11[(i >> 2) + 2 * 2];
          const uint8_t b3 = w11[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[11 * 8 + i] = kv;
          ksum11 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w11 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w12[(i >> 2) + 0 * 2];
          const uint8_t b1 = w12[(i >> 2) + 1 * 2];
          const uint8_t b2 = w12[(i >> 2) + 2 * 2];
          const uint8_t b3 = w12[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[12 * 8 + i] = kv;
          ksum12 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w12 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w13[(i >> 2) + 0 * 2];
          const uint8_t b1 = w13[(i >> 2) + 1 * 2];
          const uint8_t b2 = w13[(i >> 2) + 2 * 2];
          const uint8_t b3 = w13[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[13 * 8 + i] = kv;
          ksum13 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w13 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w14[(i >> 2) + 0 * 2];
          const uint8_t b1 = w14[(i >> 2) + 1 * 2];
          const uint8_t b2 = w14[(i >> 2) + 2 * 2];
          const uint8_t b3 = w14[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[14 * 8 + i] = kv;
          ksum14 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w14 += 8;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = w15[(i >> 2) + 0 * 2];
          const uint8_t b1 = w15[(i >> 2) + 1 * 2];
          const uint8_t b2 = w15[(i >> 2) + 2 * 2];
          const uint8_t b3 = w15[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[15 * 8 + i] = kv;
          ksum15 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w15 += 8;
        out += 128;
      }

      // KC remainder of 1..7 bytes
      if (k != 0) {
        uint8_t temp_w0[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w0[i] = w0[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w0[i] = 0;
        }
        uint8_t temp_w1[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w1[i] = w1[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w1[i] = 0;
        }
        uint8_t temp_w2[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w2[i] = w2[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w2[i] = 0;
        }
        uint8_t temp_w3[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w3[i] = w3[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w3[i] = 0;
        }
        uint8_t temp_w4[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w4[i] = w4[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w4[i] = 0;
        }
        uint8_t temp_w5[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w5[i] = w5[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w5[i] = 0;
        }
        uint8_t temp_w6[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w6[i] = w6[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w6[i] = 0;
        }
        uint8_t temp_w7[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w7[i] = w7[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w7[i] = 0;
        }
        uint8_t temp_w8[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w8[i] = w8[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w8[i] = 0;
        }
        uint8_t temp_w9[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w9[i] = w9[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w9[i] = 0;
        }
        uint8_t temp_w10[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w10[i] = w10[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w10[i] = 0;
        }
        uint8_t temp_w11[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w11[i] = w11[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w11[i] = 0;
        }
        uint8_t temp_w12[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w12[i] = w12[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w12[i] = 0;
        }
        uint8_t temp_w13[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w13[i] = w13[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w13[i] = 0;
        }
        uint8_t temp_w14[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w14[i] = w14[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w14[i] = 0;
        }
        uint8_t temp_w15[8];
        for (size_t i = 0; i < k; ++i) {
          temp_w15[i] = w15[i];
        }
        for (size_t i = k; i < 8; ++i) {
          temp_w15[i] = 0;
        }

        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w0[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w0[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w0[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w0[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[0 * 8 + i] = kv;
          ksum0 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w0 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w1[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w1[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w1[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w1[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[1 * 8 + i] = kv;
          ksum1 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w1 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w2[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w2[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w2[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w2[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[2 * 8 + i] = kv;
          ksum2 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w2 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w3[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w3[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w3[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w3[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[3 * 8 + i] = kv;
          ksum3 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w3 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w4[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w4[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w4[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w4[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[4 * 8 + i] = kv;
          ksum4 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w4 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w5[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w5[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w5[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w5[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[5 * 8 + i] = kv;
          ksum5 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w5 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w6[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w6[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w6[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w6[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[6 * 8 + i] = kv;
          ksum6 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w6 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w7[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w7[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w7[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w7[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[7 * 8 + i] = kv;
          ksum7 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w7 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w8[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w8[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w8[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w8[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[8 * 8 + i] = kv;
          ksum8 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w8 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w9[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w9[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w9[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w9[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[9 * 8 + i] = kv;
          ksum9 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w9 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w10[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w10[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w10[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w10[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[10 * 8 + i] = kv;
          ksum10 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w10 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w11[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w11[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w11[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w11[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[11 * 8 + i] = kv;
          ksum11 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w11 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w12[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w12[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w12[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w12[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[12 * 8 + i] = kv;
          ksum12 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w12 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w13[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w13[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w13[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w13[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[13 * 8 + i] = kv;
          ksum13 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w13 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w14[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w14[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w14[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w14[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[14 * 8 + i] = kv;
          ksum14 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w14 += k;
        for (size_t i = 0; i < 8; ++i) {
          const uint8_t b0 = temp_w15[(i >> 2) + 0 * 2];
          const uint8_t b1 = temp_w15[(i >> 2) + 1 * 2];
          const uint8_t b2 = temp_w15[(i >> 2) + 2 * 2];
          const uint8_t b3 = temp_w15[(i >> 2) + 3 * 2];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[15 * 8 + i] = kv;
          ksum15 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w15 += k;
        out += 128;
      }

      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      packed_b[2] -= ksum2 * izp;
      packed_b[3] -= ksum3 * izp;
      packed_b[4] -= ksum4 * izp;
      packed_b[5] -= ksum5 * izp;
      packed_b[6] -= ksum6 * izp;
      packed_b[7] -= ksum7 * izp;
      packed_b[8] -= ksum8 * izp;
      packed_b[9] -= ksum9 * izp;
      packed_b[10] -= ksum10 * izp;
      packed_b[11] -= ksum11 * izp;
      packed_b[12] -= ksum12 * izp;
      packed_b[13] -= ksum13 * izp;
      packed_b[14] -= ksum14 * izp;
      packed_b[15] -= ksum15 * izp;
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 15);
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          packed_b[i] = b[i];
        }
        for (size_t i = n; i < 16; ++i) {
          packed_b[i] = 0;
        }
        b += n;
      } else {
        for (size_t i = 0; i < 16; ++i) {
          packed_b[i] = 0;
        }
      }
      out = (uint8_t*) (packed_b + 16);

      const uint8_t* w1 = w0 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 1) {
        w1 = w0;
      }
      const uint8_t* w2 = w1 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint8_t* w3 = w2 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 3) {
        w3 = w2;
      }
      const uint8_t* w4 = w3 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint8_t* w5 = w4 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 5) {
        w5 = w4;
      }
      const uint8_t* w6 = w5 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint8_t* w7 = w6 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 7) {
        w7 = w6;
      }
      const uint8_t* w8 = w7 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint8_t* w9 = w8 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 9) {
        w9 = w8;
      }
      const uint8_t* w10 = w9 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint8_t* w11 = w10 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 11) {
        w11 = w10;
      }
      const uint8_t* w12 = w11 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint8_t* w13 = w12 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 13) {
        w13 = w12;
      }
      const uint8_t* w14 = w13 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const uint8_t* w15 = w14 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 15) {
        w15 = w14;
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

      // KC main loop multiple of 8 bytes (32 2-bit elements)
      size_t k = mock_kc;
      for (; k >= 8; k -= 8) {
        if XNN_LIKELY(0 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w0[(i >> 2) + 0 * 2];
            const uint8_t b1 = w0[(i >> 2) + 1 * 2];
            const uint8_t b2 = w0[(i >> 2) + 2 * 2];
            const uint8_t b3 = w0[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[0 * 8 + i] = kv;
            ksum0 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w0 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[0 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(1 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w1[(i >> 2) + 0 * 2];
            const uint8_t b1 = w1[(i >> 2) + 1 * 2];
            const uint8_t b2 = w1[(i >> 2) + 2 * 2];
            const uint8_t b3 = w1[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[1 * 8 + i] = kv;
            ksum1 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w1 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[1 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(2 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w2[(i >> 2) + 0 * 2];
            const uint8_t b1 = w2[(i >> 2) + 1 * 2];
            const uint8_t b2 = w2[(i >> 2) + 2 * 2];
            const uint8_t b3 = w2[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[2 * 8 + i] = kv;
            ksum2 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w2 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[2 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(3 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w3[(i >> 2) + 0 * 2];
            const uint8_t b1 = w3[(i >> 2) + 1 * 2];
            const uint8_t b2 = w3[(i >> 2) + 2 * 2];
            const uint8_t b3 = w3[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[3 * 8 + i] = kv;
            ksum3 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w3 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[3 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(4 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w4[(i >> 2) + 0 * 2];
            const uint8_t b1 = w4[(i >> 2) + 1 * 2];
            const uint8_t b2 = w4[(i >> 2) + 2 * 2];
            const uint8_t b3 = w4[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[4 * 8 + i] = kv;
            ksum4 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w4 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[4 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(5 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w5[(i >> 2) + 0 * 2];
            const uint8_t b1 = w5[(i >> 2) + 1 * 2];
            const uint8_t b2 = w5[(i >> 2) + 2 * 2];
            const uint8_t b3 = w5[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[5 * 8 + i] = kv;
            ksum5 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w5 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[5 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(6 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w6[(i >> 2) + 0 * 2];
            const uint8_t b1 = w6[(i >> 2) + 1 * 2];
            const uint8_t b2 = w6[(i >> 2) + 2 * 2];
            const uint8_t b3 = w6[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[6 * 8 + i] = kv;
            ksum6 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w6 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[6 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(7 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w7[(i >> 2) + 0 * 2];
            const uint8_t b1 = w7[(i >> 2) + 1 * 2];
            const uint8_t b2 = w7[(i >> 2) + 2 * 2];
            const uint8_t b3 = w7[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[7 * 8 + i] = kv;
            ksum7 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w7 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[7 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(8 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w8[(i >> 2) + 0 * 2];
            const uint8_t b1 = w8[(i >> 2) + 1 * 2];
            const uint8_t b2 = w8[(i >> 2) + 2 * 2];
            const uint8_t b3 = w8[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[8 * 8 + i] = kv;
            ksum8 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w8 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[8 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(9 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w9[(i >> 2) + 0 * 2];
            const uint8_t b1 = w9[(i >> 2) + 1 * 2];
            const uint8_t b2 = w9[(i >> 2) + 2 * 2];
            const uint8_t b3 = w9[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[9 * 8 + i] = kv;
            ksum9 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w9 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[9 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(10 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w10[(i >> 2) + 0 * 2];
            const uint8_t b1 = w10[(i >> 2) + 1 * 2];
            const uint8_t b2 = w10[(i >> 2) + 2 * 2];
            const uint8_t b3 = w10[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[10 * 8 + i] = kv;
            ksum10 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w10 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[10 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(11 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w11[(i >> 2) + 0 * 2];
            const uint8_t b1 = w11[(i >> 2) + 1 * 2];
            const uint8_t b2 = w11[(i >> 2) + 2 * 2];
            const uint8_t b3 = w11[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[11 * 8 + i] = kv;
            ksum11 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w11 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[11 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(12 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w12[(i >> 2) + 0 * 2];
            const uint8_t b1 = w12[(i >> 2) + 1 * 2];
            const uint8_t b2 = w12[(i >> 2) + 2 * 2];
            const uint8_t b3 = w12[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[12 * 8 + i] = kv;
            ksum12 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w12 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[12 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(13 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w13[(i >> 2) + 0 * 2];
            const uint8_t b1 = w13[(i >> 2) + 1 * 2];
            const uint8_t b2 = w13[(i >> 2) + 2 * 2];
            const uint8_t b3 = w13[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[13 * 8 + i] = kv;
            ksum13 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w13 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[13 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(14 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w14[(i >> 2) + 0 * 2];
            const uint8_t b1 = w14[(i >> 2) + 1 * 2];
            const uint8_t b2 = w14[(i >> 2) + 2 * 2];
            const uint8_t b3 = w14[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[14 * 8 + i] = kv;
            ksum14 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w14 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[14 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(15 < n) {
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = w15[(i >> 2) + 0 * 2];
            const uint8_t b1 = w15[(i >> 2) + 1 * 2];
            const uint8_t b2 = w15[(i >> 2) + 2 * 2];
            const uint8_t b3 = w15[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[15 * 8 + i] = kv;
            ksum15 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w15 += 8;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[15 * 8 + i] = 0;
          }
        }
        out += 128;
      }

      // KC remainder of 1..7 bytes
      if (k != 0) {
        if XNN_LIKELY(0 < n) {
          uint8_t temp_w0[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w0[i] = w0[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w0[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w0[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w0[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w0[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w0[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[0 * 8 + i] = kv;
            ksum0 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w0 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[0 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(1 < n) {
          uint8_t temp_w1[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w1[i] = w1[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w1[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w1[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w1[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w1[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w1[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[1 * 8 + i] = kv;
            ksum1 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w1 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[1 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(2 < n) {
          uint8_t temp_w2[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w2[i] = w2[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w2[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w2[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w2[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w2[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w2[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[2 * 8 + i] = kv;
            ksum2 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w2 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[2 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(3 < n) {
          uint8_t temp_w3[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w3[i] = w3[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w3[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w3[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w3[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w3[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w3[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[3 * 8 + i] = kv;
            ksum3 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w3 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[3 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(4 < n) {
          uint8_t temp_w4[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w4[i] = w4[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w4[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w4[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w4[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w4[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w4[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[4 * 8 + i] = kv;
            ksum4 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w4 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[4 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(5 < n) {
          uint8_t temp_w5[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w5[i] = w5[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w5[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w5[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w5[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w5[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w5[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[5 * 8 + i] = kv;
            ksum5 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w5 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[5 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(6 < n) {
          uint8_t temp_w6[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w6[i] = w6[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w6[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w6[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w6[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w6[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w6[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[6 * 8 + i] = kv;
            ksum6 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w6 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[6 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(7 < n) {
          uint8_t temp_w7[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w7[i] = w7[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w7[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w7[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w7[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w7[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w7[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[7 * 8 + i] = kv;
            ksum7 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w7 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[7 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(8 < n) {
          uint8_t temp_w8[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w8[i] = w8[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w8[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w8[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w8[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w8[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w8[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[8 * 8 + i] = kv;
            ksum8 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w8 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[8 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(9 < n) {
          uint8_t temp_w9[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w9[i] = w9[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w9[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w9[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w9[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w9[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w9[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[9 * 8 + i] = kv;
            ksum9 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w9 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[9 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(10 < n) {
          uint8_t temp_w10[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w10[i] = w10[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w10[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w10[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w10[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w10[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w10[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[10 * 8 + i] = kv;
            ksum10 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w10 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[10 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(11 < n) {
          uint8_t temp_w11[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w11[i] = w11[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w11[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w11[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w11[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w11[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w11[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[11 * 8 + i] = kv;
            ksum11 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w11 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[11 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(12 < n) {
          uint8_t temp_w12[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w12[i] = w12[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w12[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w12[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w12[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w12[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w12[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[12 * 8 + i] = kv;
            ksum12 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w12 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[12 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(13 < n) {
          uint8_t temp_w13[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w13[i] = w13[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w13[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w13[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w13[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w13[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w13[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[13 * 8 + i] = kv;
            ksum13 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w13 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[13 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(14 < n) {
          uint8_t temp_w14[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w14[i] = w14[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w14[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w14[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w14[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w14[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w14[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[14 * 8 + i] = kv;
            ksum14 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w14 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[14 * 8 + i] = 0;
          }
        }
        if XNN_LIKELY(15 < n) {
          uint8_t temp_w15[8];
          for (size_t i = 0; i < k; ++i) {
            temp_w15[i] = w15[i];
          }
          for (size_t i = k; i < 8; ++i) {
            temp_w15[i] = 0;
          }
          for (size_t i = 0; i < 8; ++i) {
            const uint8_t b0 = temp_w15[(i >> 2) + 0 * 2];
            const uint8_t b1 = temp_w15[(i >> 2) + 1 * 2];
            const uint8_t b2 = temp_w15[(i >> 2) + 2 * 2];
            const uint8_t b3 = temp_w15[(i >> 2) + 3 * 2];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[15 * 8 + i] = kv;
            ksum15 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w15 += k;
        } else {
          for (size_t i = 0; i < 8; ++i) {
            out[15 * 8 + i] = 0;
          }
        }
        out += 128;
      }

      if XNN_LIKELY(0 < n) {
        packed_b[0] -= ksum0 * izp;
      }
      if XNN_LIKELY(1 < n) {
        packed_b[1] -= ksum1 * izp;
      }
      if XNN_LIKELY(2 < n) {
        packed_b[2] -= ksum2 * izp;
      }
      if XNN_LIKELY(3 < n) {
        packed_b[3] -= ksum3 * izp;
      }
      if XNN_LIKELY(4 < n) {
        packed_b[4] -= ksum4 * izp;
      }
      if XNN_LIKELY(5 < n) {
        packed_b[5] -= ksum5 * izp;
      }
      if XNN_LIKELY(6 < n) {
        packed_b[6] -= ksum6 * izp;
      }
      if XNN_LIKELY(7 < n) {
        packed_b[7] -= ksum7 * izp;
      }
      if XNN_LIKELY(8 < n) {
        packed_b[8] -= ksum8 * izp;
      }
      if XNN_LIKELY(9 < n) {
        packed_b[9] -= ksum9 * izp;
      }
      if XNN_LIKELY(10 < n) {
        packed_b[10] -= ksum10 * izp;
      }
      if XNN_LIKELY(11 < n) {
        packed_b[11] -= ksum11 * izp;
      }
      if XNN_LIKELY(12 < n) {
        packed_b[12] -= ksum12 * izp;
      }
      if XNN_LIKELY(13 < n) {
        packed_b[13] -= ksum13 * izp;
      }
      if XNN_LIKELY(14 < n) {
        packed_b[14] -= ksum14 * izp;
      }
      if XNN_LIKELY(15 < n) {
        packed_b[15] -= ksum15 * izp;
      }
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * mock_kc;
  } while (--g != 0);
}
