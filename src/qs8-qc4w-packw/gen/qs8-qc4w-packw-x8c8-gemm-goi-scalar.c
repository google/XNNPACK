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

void xnn_qs8_qc4w_packw_gemm_goi_ukernel_x8c8__scalar(
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
  assert(nr == 8);
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
    // NC main loop multiple of 8
    const uint8_t* w0 = weights;
    size_t n = nc;
    for (; n >= 8; n -= 8) {
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < 8; ++i) {
          packed_b[i] = b[i] * 16;
        }
        b += 8;
      } else {
        for (size_t i = 0; i < 8; ++i) {
          packed_b[i] = 0;
        }
      }
      out += 8 * sizeof(int32_t);

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
        out += 8 * 8;
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
        out += 8 * 8;
      }

      packed_b[0] -= ksum0 * izp * 16;
      packed_b[1] -= ksum1 * izp * 16;
      packed_b[2] -= ksum2 * izp * 16;
      packed_b[3] -= ksum3 * izp * 16;
      packed_b[4] -= ksum4 * izp * 16;
      packed_b[5] -= ksum5 * izp * 16;
      packed_b[6] -= ksum6 * izp * 16;
      packed_b[7] -= ksum7 * izp * 16;
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
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
      out += 8 * sizeof(int32_t);

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

      int32_t ksum0 = 0;
      int32_t ksum1 = 0;
      int32_t ksum2 = 0;
      int32_t ksum3 = 0;
      int32_t ksum4 = 0;
      int32_t ksum5 = 0;
      int32_t ksum6 = 0;

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

        // Pad the remaining row(s)
        for (size_t i = 0; i < 8; ++i) {
          out[7 * 8 + i] = 0;
        }
        out += 8 * 8;
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

        // Pad the remaining row(s)
        for (size_t i = 0; i < 8; ++i) {
          out[7 * 8 + i] = 0;
        }
        out += 8 * 8;
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
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights = (const uint8_t*)((intptr_t) weights + nc * kc);
  } while (--g != 0);
}
