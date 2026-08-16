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

void xnn_qs8_to_qu8_qc2w_packw_gemm_goi_ukernel_x2c4__scalar(
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
  assert(nr == 2);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);
  assert(kc % 4 == 0);

  const size_t mock_kc = kc >> 2; // kc in bytes (each byte has 4 2-bit weights)
  const uint32_t izp = (uint32_t) params->input_zero_point + 128;

  uint8_t* out = (uint8_t*) packed_weights;
  const int32_t* b = bias;

  do {
    const uint8_t* w0 = weights;
    size_t n = nc;
    for (; n >= 2; n -= 2) {
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < 2; ++i) {
          packed_b[i] = b[i];
        }
        b += 2;
      } else {
        for (size_t i = 0; i < 2; ++i) {
          packed_b[i] = 0;
        }
      }
      out = (uint8_t*) (packed_b + 2);

      const uint8_t* w1 = w0 + mock_kc;

      int32_t ksum0 = 0;
      int32_t ksum1 = 0;

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
        out += 8;
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
        out += 8;
      }

      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w1;
    }

    // NC remainder (1..1)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 1);
      int32_t* packed_b = (int32_t*) out;
      if (b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          packed_b[i] = b[i];
        }
        for (size_t i = n; i < 2; ++i) {
          packed_b[i] = 0;
        }
        b += n;
      } else {
        for (size_t i = 0; i < 2; ++i) {
          packed_b[i] = 0;
        }
      }
      out = (uint8_t*) (packed_b + 2);

      const uint8_t* w1 = w0 + mock_kc;
      if XNN_UNPREDICTABLE(n <= 1) {
        w1 = w0;
      }

      int32_t ksum0 = 0;
      int32_t ksum1 = 0;

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
        out += 8;
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
        out += 8;
      }

      if XNN_LIKELY(0 < n) {
        packed_b[0] -= ksum0 * izp;
      }
      if XNN_LIKELY(1 < n) {
        packed_b[1] -= ksum1 * izp;
      }
      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * mock_kc;
  } while (--g != 0);
}
