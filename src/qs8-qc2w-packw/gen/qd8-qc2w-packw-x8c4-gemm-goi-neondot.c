// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/c4-qc2w-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/packw.h"


// Sign extend 2-bit to 8-bit
inline static int8_t sign_extend_int2(uint8_t v) {
  return (int8_t)(v << 6) >> 6;
}

void xnn_qd8_qc2w_packw_gemm_goi_ukernel_x8c4__neondot(
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
  const struct xnn_qd8_qc2w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 8);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);
  assert(kc % 4 == 0);

  const size_t mock_kc = kc >> 2; // kc in bytes (each byte has 4 2-bit weights)
  const uint32_t izp = (uint32_t) params->input_zero_point;
  const float* kernel_zero_points = params->kernel_zero_point;

  uint8_t* out = (uint8_t*) packed_weights;
  const int32_t* b = bias;

  do {
    const uint8_t* w0 = weights;
    size_t n = nc;
    for (; n >= 8; n -= 8) {
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < 8; ++i) {
          packed_b[i] = b[i];
        }
        b += 8;
      } else {
        for (size_t i = 0; i < 8; ++i) {
          packed_b[i] = 0;
        }
      }
      float* packed_zp = (float*) (packed_b + 8);
      if (kernel_zero_points != NULL) {
        for (size_t i = 0; i < 8; ++i) {
          packed_zp[i] = kernel_zero_points[i];
        }
        kernel_zero_points += 8;
      } else {
        for (size_t i = 0; i < 8; ++i) {
          packed_zp[i] = 0.0f;
        }
      }
      out = (uint8_t*) (packed_zp + 8);

      const uint8_t* w1 = w0 + mock_kc;
      const uint8_t* w2 = w1 + mock_kc;
      const uint8_t* w3 = w2 + mock_kc;
      const uint8_t* w4 = w3 + mock_kc;
      const uint8_t* w5 = w4 + mock_kc;
      const uint8_t* w6 = w5 + mock_kc;
      const uint8_t* w7 = w6 + mock_kc;

      int32_t ksum0 = 0;
      int32_t ksum1 = 0;
      int32_t ksum2 = 0;
      int32_t ksum3 = 0;
      int32_t ksum4 = 0;
      int32_t ksum5 = 0;
      int32_t ksum6 = 0;
      int32_t ksum7 = 0;

      // KC main loop multiple of 4 bytes (16 2-bit elements)
      size_t k = mock_kc;
      for (; k >= 4; k -= 4) {
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = w0[(i >> 2) + 0 * 1];
          const uint8_t b1 = w0[(i >> 2) + 1 * 1];
          const uint8_t b2 = w0[(i >> 2) + 2 * 1];
          const uint8_t b3 = w0[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[0 * 4 + i] = kv ^ 0xAA;
          ksum0 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w0 += 4;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = w1[(i >> 2) + 0 * 1];
          const uint8_t b1 = w1[(i >> 2) + 1 * 1];
          const uint8_t b2 = w1[(i >> 2) + 2 * 1];
          const uint8_t b3 = w1[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[1 * 4 + i] = kv ^ 0xAA;
          ksum1 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w1 += 4;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = w2[(i >> 2) + 0 * 1];
          const uint8_t b1 = w2[(i >> 2) + 1 * 1];
          const uint8_t b2 = w2[(i >> 2) + 2 * 1];
          const uint8_t b3 = w2[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[2 * 4 + i] = kv ^ 0xAA;
          ksum2 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w2 += 4;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = w3[(i >> 2) + 0 * 1];
          const uint8_t b1 = w3[(i >> 2) + 1 * 1];
          const uint8_t b2 = w3[(i >> 2) + 2 * 1];
          const uint8_t b3 = w3[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[3 * 4 + i] = kv ^ 0xAA;
          ksum3 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w3 += 4;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = w4[(i >> 2) + 0 * 1];
          const uint8_t b1 = w4[(i >> 2) + 1 * 1];
          const uint8_t b2 = w4[(i >> 2) + 2 * 1];
          const uint8_t b3 = w4[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[4 * 4 + i] = kv ^ 0xAA;
          ksum4 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w4 += 4;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = w5[(i >> 2) + 0 * 1];
          const uint8_t b1 = w5[(i >> 2) + 1 * 1];
          const uint8_t b2 = w5[(i >> 2) + 2 * 1];
          const uint8_t b3 = w5[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[5 * 4 + i] = kv ^ 0xAA;
          ksum5 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w5 += 4;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = w6[(i >> 2) + 0 * 1];
          const uint8_t b1 = w6[(i >> 2) + 1 * 1];
          const uint8_t b2 = w6[(i >> 2) + 2 * 1];
          const uint8_t b3 = w6[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[6 * 4 + i] = kv ^ 0xAA;
          ksum6 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w6 += 4;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = w7[(i >> 2) + 0 * 1];
          const uint8_t b1 = w7[(i >> 2) + 1 * 1];
          const uint8_t b2 = w7[(i >> 2) + 2 * 1];
          const uint8_t b3 = w7[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[7 * 4 + i] = kv ^ 0xAA;
          ksum7 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w7 += 4;
        out += 32;
      }

      // KC remainder of 1..3 bytes
      if (k != 0) {
        uint8_t temp_w0[4];
        for (size_t i = 0; i < k; ++i) {
          temp_w0[i] = w0[i];
        }
        for (size_t i = k; i < 4; ++i) {
          temp_w0[i] = 0;
        }
        uint8_t temp_w1[4];
        for (size_t i = 0; i < k; ++i) {
          temp_w1[i] = w1[i];
        }
        for (size_t i = k; i < 4; ++i) {
          temp_w1[i] = 0;
        }
        uint8_t temp_w2[4];
        for (size_t i = 0; i < k; ++i) {
          temp_w2[i] = w2[i];
        }
        for (size_t i = k; i < 4; ++i) {
          temp_w2[i] = 0;
        }
        uint8_t temp_w3[4];
        for (size_t i = 0; i < k; ++i) {
          temp_w3[i] = w3[i];
        }
        for (size_t i = k; i < 4; ++i) {
          temp_w3[i] = 0;
        }
        uint8_t temp_w4[4];
        for (size_t i = 0; i < k; ++i) {
          temp_w4[i] = w4[i];
        }
        for (size_t i = k; i < 4; ++i) {
          temp_w4[i] = 0;
        }
        uint8_t temp_w5[4];
        for (size_t i = 0; i < k; ++i) {
          temp_w5[i] = w5[i];
        }
        for (size_t i = k; i < 4; ++i) {
          temp_w5[i] = 0;
        }
        uint8_t temp_w6[4];
        for (size_t i = 0; i < k; ++i) {
          temp_w6[i] = w6[i];
        }
        for (size_t i = k; i < 4; ++i) {
          temp_w6[i] = 0;
        }
        uint8_t temp_w7[4];
        for (size_t i = 0; i < k; ++i) {
          temp_w7[i] = w7[i];
        }
        for (size_t i = k; i < 4; ++i) {
          temp_w7[i] = 0;
        }

        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = temp_w0[(i >> 2) + 0 * 1];
          const uint8_t b1 = temp_w0[(i >> 2) + 1 * 1];
          const uint8_t b2 = temp_w0[(i >> 2) + 2 * 1];
          const uint8_t b3 = temp_w0[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[0 * 4 + i] = kv ^ 0xAA;
          ksum0 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w0 += k;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = temp_w1[(i >> 2) + 0 * 1];
          const uint8_t b1 = temp_w1[(i >> 2) + 1 * 1];
          const uint8_t b2 = temp_w1[(i >> 2) + 2 * 1];
          const uint8_t b3 = temp_w1[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[1 * 4 + i] = kv ^ 0xAA;
          ksum1 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w1 += k;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = temp_w2[(i >> 2) + 0 * 1];
          const uint8_t b1 = temp_w2[(i >> 2) + 1 * 1];
          const uint8_t b2 = temp_w2[(i >> 2) + 2 * 1];
          const uint8_t b3 = temp_w2[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[2 * 4 + i] = kv ^ 0xAA;
          ksum2 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w2 += k;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = temp_w3[(i >> 2) + 0 * 1];
          const uint8_t b1 = temp_w3[(i >> 2) + 1 * 1];
          const uint8_t b2 = temp_w3[(i >> 2) + 2 * 1];
          const uint8_t b3 = temp_w3[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[3 * 4 + i] = kv ^ 0xAA;
          ksum3 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w3 += k;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = temp_w4[(i >> 2) + 0 * 1];
          const uint8_t b1 = temp_w4[(i >> 2) + 1 * 1];
          const uint8_t b2 = temp_w4[(i >> 2) + 2 * 1];
          const uint8_t b3 = temp_w4[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[4 * 4 + i] = kv ^ 0xAA;
          ksum4 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w4 += k;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = temp_w5[(i >> 2) + 0 * 1];
          const uint8_t b1 = temp_w5[(i >> 2) + 1 * 1];
          const uint8_t b2 = temp_w5[(i >> 2) + 2 * 1];
          const uint8_t b3 = temp_w5[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[5 * 4 + i] = kv ^ 0xAA;
          ksum5 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w5 += k;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = temp_w6[(i >> 2) + 0 * 1];
          const uint8_t b1 = temp_w6[(i >> 2) + 1 * 1];
          const uint8_t b2 = temp_w6[(i >> 2) + 2 * 1];
          const uint8_t b3 = temp_w6[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[6 * 4 + i] = kv ^ 0xAA;
          ksum6 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w6 += k;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t b0 = temp_w7[(i >> 2) + 0 * 1];
          const uint8_t b1 = temp_w7[(i >> 2) + 1 * 1];
          const uint8_t b2 = temp_w7[(i >> 2) + 2 * 1];
          const uint8_t b3 = temp_w7[(i >> 2) + 3 * 1];
          const uint8_t shift = (i & 3) * 2;
          const uint8_t val0 = (b0 >> shift) & 3;
          const uint8_t val1 = (b1 >> shift) & 3;
          const uint8_t val2 = (b2 >> shift) & 3;
          const uint8_t val3 = (b3 >> shift) & 3;
          const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
          out[7 * 4 + i] = kv ^ 0xAA;
          ksum7 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
        }
        w7 += k;
        out += 32;
      }

      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      packed_b[2] -= ksum2 * izp;
      packed_b[3] -= ksum3 * izp;
      packed_b[4] -= ksum4 * izp;
      packed_b[5] -= ksum5 * izp;
      packed_b[6] -= ksum6 * izp;
      packed_b[7] -= ksum7 * izp;
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 7);
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          packed_b[i] = b[i];
        }
        for (size_t i = n; i < 8; ++i) {
          packed_b[i] = 0;
        }
        b += n;
      } else {
        for (size_t i = 0; i < 8; ++i) {
          packed_b[i] = 0;
        }
      }
      float* packed_zp = (float*) (packed_b + 8);
      if (kernel_zero_points != NULL) {
        for (size_t i = 0; i < n; ++i) {
          packed_zp[i] = kernel_zero_points[i];
        }
        for (size_t i = n; i < 8; ++i) {
          packed_zp[i] = 0.0f;
        }
        kernel_zero_points += n;
      } else {
        for (size_t i = 0; i < 8; ++i) {
          packed_zp[i] = 0.0f;
        }
      }
      out = (uint8_t*) (packed_zp + 8);

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

      int32_t ksum0 = 0;
      int32_t ksum1 = 0;
      int32_t ksum2 = 0;
      int32_t ksum3 = 0;
      int32_t ksum4 = 0;
      int32_t ksum5 = 0;
      int32_t ksum6 = 0;
      int32_t ksum7 = 0;

      // KC main loop multiple of 4 bytes (16 2-bit elements)
      size_t k = mock_kc;
      for (; k >= 4; k -= 4) {
        if XNN_LIKELY(0 < n) {
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = w0[(i >> 2) + 0 * 1];
            const uint8_t b1 = w0[(i >> 2) + 1 * 1];
            const uint8_t b2 = w0[(i >> 2) + 2 * 1];
            const uint8_t b3 = w0[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[0 * 4 + i] = kv ^ 0xAA;
            ksum0 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w0 += 4;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[0 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(1 < n) {
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = w1[(i >> 2) + 0 * 1];
            const uint8_t b1 = w1[(i >> 2) + 1 * 1];
            const uint8_t b2 = w1[(i >> 2) + 2 * 1];
            const uint8_t b3 = w1[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[1 * 4 + i] = kv ^ 0xAA;
            ksum1 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w1 += 4;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[1 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(2 < n) {
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = w2[(i >> 2) + 0 * 1];
            const uint8_t b1 = w2[(i >> 2) + 1 * 1];
            const uint8_t b2 = w2[(i >> 2) + 2 * 1];
            const uint8_t b3 = w2[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[2 * 4 + i] = kv ^ 0xAA;
            ksum2 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w2 += 4;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[2 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(3 < n) {
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = w3[(i >> 2) + 0 * 1];
            const uint8_t b1 = w3[(i >> 2) + 1 * 1];
            const uint8_t b2 = w3[(i >> 2) + 2 * 1];
            const uint8_t b3 = w3[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[3 * 4 + i] = kv ^ 0xAA;
            ksum3 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w3 += 4;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[3 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(4 < n) {
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = w4[(i >> 2) + 0 * 1];
            const uint8_t b1 = w4[(i >> 2) + 1 * 1];
            const uint8_t b2 = w4[(i >> 2) + 2 * 1];
            const uint8_t b3 = w4[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[4 * 4 + i] = kv ^ 0xAA;
            ksum4 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w4 += 4;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[4 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(5 < n) {
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = w5[(i >> 2) + 0 * 1];
            const uint8_t b1 = w5[(i >> 2) + 1 * 1];
            const uint8_t b2 = w5[(i >> 2) + 2 * 1];
            const uint8_t b3 = w5[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[5 * 4 + i] = kv ^ 0xAA;
            ksum5 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w5 += 4;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[5 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(6 < n) {
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = w6[(i >> 2) + 0 * 1];
            const uint8_t b1 = w6[(i >> 2) + 1 * 1];
            const uint8_t b2 = w6[(i >> 2) + 2 * 1];
            const uint8_t b3 = w6[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[6 * 4 + i] = kv ^ 0xAA;
            ksum6 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w6 += 4;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[6 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(7 < n) {
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = w7[(i >> 2) + 0 * 1];
            const uint8_t b1 = w7[(i >> 2) + 1 * 1];
            const uint8_t b2 = w7[(i >> 2) + 2 * 1];
            const uint8_t b3 = w7[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[7 * 4 + i] = kv ^ 0xAA;
            ksum7 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w7 += 4;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[7 * 4 + i] = 0;
          }
        }
        out += 32;
      }

      // KC remainder of 1..3 bytes
      if (k != 0) {
        if XNN_LIKELY(0 < n) {
          uint8_t temp_w0[4];
          for (size_t i = 0; i < k; ++i) {
            temp_w0[i] = w0[i];
          }
          for (size_t i = k; i < 4; ++i) {
            temp_w0[i] = 0;
          }
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = temp_w0[(i >> 2) + 0 * 1];
            const uint8_t b1 = temp_w0[(i >> 2) + 1 * 1];
            const uint8_t b2 = temp_w0[(i >> 2) + 2 * 1];
            const uint8_t b3 = temp_w0[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[0 * 4 + i] = kv ^ 0xAA;
            ksum0 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w0 += k;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[0 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(1 < n) {
          uint8_t temp_w1[4];
          for (size_t i = 0; i < k; ++i) {
            temp_w1[i] = w1[i];
          }
          for (size_t i = k; i < 4; ++i) {
            temp_w1[i] = 0;
          }
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = temp_w1[(i >> 2) + 0 * 1];
            const uint8_t b1 = temp_w1[(i >> 2) + 1 * 1];
            const uint8_t b2 = temp_w1[(i >> 2) + 2 * 1];
            const uint8_t b3 = temp_w1[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[1 * 4 + i] = kv ^ 0xAA;
            ksum1 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w1 += k;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[1 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(2 < n) {
          uint8_t temp_w2[4];
          for (size_t i = 0; i < k; ++i) {
            temp_w2[i] = w2[i];
          }
          for (size_t i = k; i < 4; ++i) {
            temp_w2[i] = 0;
          }
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = temp_w2[(i >> 2) + 0 * 1];
            const uint8_t b1 = temp_w2[(i >> 2) + 1 * 1];
            const uint8_t b2 = temp_w2[(i >> 2) + 2 * 1];
            const uint8_t b3 = temp_w2[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[2 * 4 + i] = kv ^ 0xAA;
            ksum2 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w2 += k;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[2 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(3 < n) {
          uint8_t temp_w3[4];
          for (size_t i = 0; i < k; ++i) {
            temp_w3[i] = w3[i];
          }
          for (size_t i = k; i < 4; ++i) {
            temp_w3[i] = 0;
          }
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = temp_w3[(i >> 2) + 0 * 1];
            const uint8_t b1 = temp_w3[(i >> 2) + 1 * 1];
            const uint8_t b2 = temp_w3[(i >> 2) + 2 * 1];
            const uint8_t b3 = temp_w3[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[3 * 4 + i] = kv ^ 0xAA;
            ksum3 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w3 += k;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[3 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(4 < n) {
          uint8_t temp_w4[4];
          for (size_t i = 0; i < k; ++i) {
            temp_w4[i] = w4[i];
          }
          for (size_t i = k; i < 4; ++i) {
            temp_w4[i] = 0;
          }
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = temp_w4[(i >> 2) + 0 * 1];
            const uint8_t b1 = temp_w4[(i >> 2) + 1 * 1];
            const uint8_t b2 = temp_w4[(i >> 2) + 2 * 1];
            const uint8_t b3 = temp_w4[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[4 * 4 + i] = kv ^ 0xAA;
            ksum4 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w4 += k;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[4 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(5 < n) {
          uint8_t temp_w5[4];
          for (size_t i = 0; i < k; ++i) {
            temp_w5[i] = w5[i];
          }
          for (size_t i = k; i < 4; ++i) {
            temp_w5[i] = 0;
          }
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = temp_w5[(i >> 2) + 0 * 1];
            const uint8_t b1 = temp_w5[(i >> 2) + 1 * 1];
            const uint8_t b2 = temp_w5[(i >> 2) + 2 * 1];
            const uint8_t b3 = temp_w5[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[5 * 4 + i] = kv ^ 0xAA;
            ksum5 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w5 += k;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[5 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(6 < n) {
          uint8_t temp_w6[4];
          for (size_t i = 0; i < k; ++i) {
            temp_w6[i] = w6[i];
          }
          for (size_t i = k; i < 4; ++i) {
            temp_w6[i] = 0;
          }
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = temp_w6[(i >> 2) + 0 * 1];
            const uint8_t b1 = temp_w6[(i >> 2) + 1 * 1];
            const uint8_t b2 = temp_w6[(i >> 2) + 2 * 1];
            const uint8_t b3 = temp_w6[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[6 * 4 + i] = kv ^ 0xAA;
            ksum6 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w6 += k;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[6 * 4 + i] = 0;
          }
        }
        if XNN_LIKELY(7 < n) {
          uint8_t temp_w7[4];
          for (size_t i = 0; i < k; ++i) {
            temp_w7[i] = w7[i];
          }
          for (size_t i = k; i < 4; ++i) {
            temp_w7[i] = 0;
          }
          for (size_t i = 0; i < 4; ++i) {
            const uint8_t b0 = temp_w7[(i >> 2) + 0 * 1];
            const uint8_t b1 = temp_w7[(i >> 2) + 1 * 1];
            const uint8_t b2 = temp_w7[(i >> 2) + 2 * 1];
            const uint8_t b3 = temp_w7[(i >> 2) + 3 * 1];
            const uint8_t shift = (i & 3) * 2;
            const uint8_t val0 = (b0 >> shift) & 3;
            const uint8_t val1 = (b1 >> shift) & 3;
            const uint8_t val2 = (b2 >> shift) & 3;
            const uint8_t val3 = (b3 >> shift) & 3;
            const uint8_t kv = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6);
            out[7 * 4 + i] = kv ^ 0xAA;
            ksum7 += sign_extend_int2(val0) + sign_extend_int2(val1) + sign_extend_int2(val2) + sign_extend_int2(val3);
          }
          w7 += k;
        } else {
          for (size_t i = 0; i < 4; ++i) {
            out[7 * 4 + i] = 0;
          }
        }
        out += 32;
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
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * mock_kc;
  } while (--g != 0);
}
