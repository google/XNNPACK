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

#include "xnnpack/packw.h"

void xnn_qs8_packw_gemm_goi_ukernel_x8c4__scalar(
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

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = params ? (uint32_t) ((const struct xnn_qs8_packw_params*) params)->input_zero_point : 0;

  do {
    // NC main loop multiple of 8
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 8; n -= 8) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        ((int32_t*) out)[0] = b[0];
        ((int32_t*) out)[1] = b[1];
        ((int32_t*) out)[2] = b[2];
        ((int32_t*) out)[3] = b[3];
        ((int32_t*) out)[4] = b[4];
        ((int32_t*) out)[5] = b[5];
        ((int32_t*) out)[6] = b[6];
        ((int32_t*) out)[7] = b[7];
        b += 8;
      } else {
        ((int32_t*) out)[0] = 0;
        ((int32_t*) out)[1] = 0;
        ((int32_t*) out)[2] = 0;
        ((int32_t*) out)[3] = 0;
        ((int32_t*) out)[4] = 0;
        ((int32_t*) out)[5] = 0;
        ((int32_t*) out)[6] = 0;
        ((int32_t*) out)[7] = 0;
      }
      out += 8 * sizeof(int32_t);

      const int8_t* w1 = w0 + kc;
      const int8_t* w2 = w1 + kc;
      const int8_t* w3 = w2 + kc;
      const int8_t* w4 = w3 + kc;
      const int8_t* w5 = w4 + kc;
      const int8_t* w6 = w5 + kc;
      const int8_t* w7 = w6 + kc;
      uint32_t ksum0 = 0;
      uint32_t ksum1 = 0;
      uint32_t ksum2 = 0;
      uint32_t ksum3 = 0;
      uint32_t ksum4 = 0;
      uint32_t ksum5 = 0;
      uint32_t ksum6 = 0;
      uint32_t ksum7 = 0;

      // KC main loop multiple of 8x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const int8_t v0x0 = w0[0];
        const int8_t v0x1 = w0[1];
        const int8_t v0x2 = w0[2];
        const int8_t v0x3 = w0[3];
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v0x1;
        ksum0 += (uint32_t) v0x2;
        ksum0 += (uint32_t) v0x3;
        out[0] = v0x0;
        out[1] = v0x1;
        out[2] = v0x2;
        out[3] = v0x3;
        w0 += 4;
        const int8_t v1x0 = w1[0];
        const int8_t v1x1 = w1[1];
        const int8_t v1x2 = w1[2];
        const int8_t v1x3 = w1[3];
        ksum1 += (uint32_t) v1x0;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v1x2;
        ksum1 += (uint32_t) v1x3;
        out[4] = v1x0;
        out[5] = v1x1;
        out[6] = v1x2;
        out[7] = v1x3;
        w1 += 4;
        const int8_t v2x0 = w2[0];
        const int8_t v2x1 = w2[1];
        const int8_t v2x2 = w2[2];
        const int8_t v2x3 = w2[3];
        ksum2 += (uint32_t) v2x0;
        ksum2 += (uint32_t) v2x1;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v2x3;
        out[8] = v2x0;
        out[9] = v2x1;
        out[10] = v2x2;
        out[11] = v2x3;
        w2 += 4;
        const int8_t v3x0 = w3[0];
        const int8_t v3x1 = w3[1];
        const int8_t v3x2 = w3[2];
        const int8_t v3x3 = w3[3];
        ksum3 += (uint32_t) v3x0;
        ksum3 += (uint32_t) v3x1;
        ksum3 += (uint32_t) v3x2;
        ksum3 += (uint32_t) v3x3;
        out[12] = v3x0;
        out[13] = v3x1;
        out[14] = v3x2;
        out[15] = v3x3;
        w3 += 4;
        const int8_t v4x0 = w4[0];
        const int8_t v4x1 = w4[1];
        const int8_t v4x2 = w4[2];
        const int8_t v4x3 = w4[3];
        ksum4 += (uint32_t) v4x0;
        ksum4 += (uint32_t) v4x1;
        ksum4 += (uint32_t) v4x2;
        ksum4 += (uint32_t) v4x3;
        out[16] = v4x0;
        out[17] = v4x1;
        out[18] = v4x2;
        out[19] = v4x3;
        w4 += 4;
        const int8_t v5x0 = w5[0];
        const int8_t v5x1 = w5[1];
        const int8_t v5x2 = w5[2];
        const int8_t v5x3 = w5[3];
        ksum5 += (uint32_t) v5x0;
        ksum5 += (uint32_t) v5x1;
        ksum5 += (uint32_t) v5x2;
        ksum5 += (uint32_t) v5x3;
        out[20] = v5x0;
        out[21] = v5x1;
        out[22] = v5x2;
        out[23] = v5x3;
        w5 += 4;
        const int8_t v6x0 = w6[0];
        const int8_t v6x1 = w6[1];
        const int8_t v6x2 = w6[2];
        const int8_t v6x3 = w6[3];
        ksum6 += (uint32_t) v6x0;
        ksum6 += (uint32_t) v6x1;
        ksum6 += (uint32_t) v6x2;
        ksum6 += (uint32_t) v6x3;
        out[24] = v6x0;
        out[25] = v6x1;
        out[26] = v6x2;
        out[27] = v6x3;
        w6 += 4;
        const int8_t v7x0 = w7[0];
        const int8_t v7x1 = w7[1];
        const int8_t v7x2 = w7[2];
        const int8_t v7x3 = w7[3];
        ksum7 += (uint32_t) v7x0;
        ksum7 += (uint32_t) v7x1;
        ksum7 += (uint32_t) v7x2;
        ksum7 += (uint32_t) v7x3;
        out[28] = v7x0;
        out[29] = v7x1;
        out[30] = v7x2;
        out[31] = v7x3;
        w7 += 4;
        out += 32;
      }

      // KC remainder 1..KR-1
      if (k != 0) {
        const int8_t v0x0 = 0 < k ? w0[0] : izp;
        const int8_t v0x1 = 1 < k ? w0[1] : izp;
        const int8_t v0x2 = 2 < k ? w0[2] : izp;
        const int8_t v0x3 = 3 < k ? w0[3] : izp;
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v0x1;
        ksum0 += (uint32_t) v0x2;
        ksum0 += (uint32_t) v0x3;
        if (0 < k) {
          out[0] = v0x0;
        }
        if (1 < k) {
          out[1] = v0x1;
        }
        if (2 < k) {
          out[2] = v0x2;
        }
        if (3 < k) {
          out[3] = v0x3;
        }
        w0 += 4;
        const int8_t v1x0 = 0 < k ? w1[0] : izp;
        const int8_t v1x1 = 1 < k ? w1[1] : izp;
        const int8_t v1x2 = 2 < k ? w1[2] : izp;
        const int8_t v1x3 = 3 < k ? w1[3] : izp;
        ksum1 += (uint32_t) v1x0;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v1x2;
        ksum1 += (uint32_t) v1x3;
        if (0 < k) {
          out[4] = v1x0;
        }
        if (1 < k) {
          out[5] = v1x1;
        }
        if (2 < k) {
          out[6] = v1x2;
        }
        if (3 < k) {
          out[7] = v1x3;
        }
        w1 += 4;
        const int8_t v2x0 = 0 < k ? w2[0] : izp;
        const int8_t v2x1 = 1 < k ? w2[1] : izp;
        const int8_t v2x2 = 2 < k ? w2[2] : izp;
        const int8_t v2x3 = 3 < k ? w2[3] : izp;
        ksum2 += (uint32_t) v2x0;
        ksum2 += (uint32_t) v2x1;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v2x3;
        if (0 < k) {
          out[8] = v2x0;
        }
        if (1 < k) {
          out[9] = v2x1;
        }
        if (2 < k) {
          out[10] = v2x2;
        }
        if (3 < k) {
          out[11] = v2x3;
        }
        w2 += 4;
        const int8_t v3x0 = 0 < k ? w3[0] : izp;
        const int8_t v3x1 = 1 < k ? w3[1] : izp;
        const int8_t v3x2 = 2 < k ? w3[2] : izp;
        const int8_t v3x3 = 3 < k ? w3[3] : izp;
        ksum3 += (uint32_t) v3x0;
        ksum3 += (uint32_t) v3x1;
        ksum3 += (uint32_t) v3x2;
        ksum3 += (uint32_t) v3x3;
        if (0 < k) {
          out[12] = v3x0;
        }
        if (1 < k) {
          out[13] = v3x1;
        }
        if (2 < k) {
          out[14] = v3x2;
        }
        if (3 < k) {
          out[15] = v3x3;
        }
        w3 += 4;
        const int8_t v4x0 = 0 < k ? w4[0] : izp;
        const int8_t v4x1 = 1 < k ? w4[1] : izp;
        const int8_t v4x2 = 2 < k ? w4[2] : izp;
        const int8_t v4x3 = 3 < k ? w4[3] : izp;
        ksum4 += (uint32_t) v4x0;
        ksum4 += (uint32_t) v4x1;
        ksum4 += (uint32_t) v4x2;
        ksum4 += (uint32_t) v4x3;
        if (0 < k) {
          out[16] = v4x0;
        }
        if (1 < k) {
          out[17] = v4x1;
        }
        if (2 < k) {
          out[18] = v4x2;
        }
        if (3 < k) {
          out[19] = v4x3;
        }
        w4 += 4;
        const int8_t v5x0 = 0 < k ? w5[0] : izp;
        const int8_t v5x1 = 1 < k ? w5[1] : izp;
        const int8_t v5x2 = 2 < k ? w5[2] : izp;
        const int8_t v5x3 = 3 < k ? w5[3] : izp;
        ksum5 += (uint32_t) v5x0;
        ksum5 += (uint32_t) v5x1;
        ksum5 += (uint32_t) v5x2;
        ksum5 += (uint32_t) v5x3;
        if (0 < k) {
          out[20] = v5x0;
        }
        if (1 < k) {
          out[21] = v5x1;
        }
        if (2 < k) {
          out[22] = v5x2;
        }
        if (3 < k) {
          out[23] = v5x3;
        }
        w5 += 4;
        const int8_t v6x0 = 0 < k ? w6[0] : izp;
        const int8_t v6x1 = 1 < k ? w6[1] : izp;
        const int8_t v6x2 = 2 < k ? w6[2] : izp;
        const int8_t v6x3 = 3 < k ? w6[3] : izp;
        ksum6 += (uint32_t) v6x0;
        ksum6 += (uint32_t) v6x1;
        ksum6 += (uint32_t) v6x2;
        ksum6 += (uint32_t) v6x3;
        if (0 < k) {
          out[24] = v6x0;
        }
        if (1 < k) {
          out[25] = v6x1;
        }
        if (2 < k) {
          out[26] = v6x2;
        }
        if (3 < k) {
          out[27] = v6x3;
        }
        w6 += 4;
        const int8_t v7x0 = 0 < k ? w7[0] : izp;
        const int8_t v7x1 = 1 < k ? w7[1] : izp;
        const int8_t v7x2 = 2 < k ? w7[2] : izp;
        const int8_t v7x3 = 3 < k ? w7[3] : izp;
        ksum7 += (uint32_t) v7x0;
        ksum7 += (uint32_t) v7x1;
        ksum7 += (uint32_t) v7x2;
        ksum7 += (uint32_t) v7x3;
        if (0 < k) {
          out[28] = v7x0;
        }
        if (1 < k) {
          out[29] = v7x1;
        }
        if (2 < k) {
          out[30] = v7x2;
        }
        if (3 < k) {
          out[31] = v7x3;
        }
        w7 += 4;
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
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *((int32_t*) out) = *b++;
          out += sizeof(int32_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *((int32_t*) out) = 0;
          out += sizeof(int32_t);
        } while (--nb != 0);
      }
      out += (8 - n) * sizeof(int32_t);

     // NR remainder has less than 8 rows so last row is not loaded
      const int8_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int8_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int8_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int8_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int8_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }

      uint32_t ksum0 = 0;
      uint32_t ksum1 = 0;
      uint32_t ksum2 = 0;
      uint32_t ksum3 = 0;
      uint32_t ksum4 = 0;
      uint32_t ksum5 = 0;
      uint32_t ksum6 = 0;

      // KC main loop multiple of 8x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const int8_t v0x0 = w0[0];
        const int8_t v0x1 = w0[1];
        const int8_t v0x2 = w0[2];
        const int8_t v0x3 = w0[3];
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v0x1;
        ksum0 += (uint32_t) v0x2;
        ksum0 += (uint32_t) v0x3;
        out[0] = v0x0;
        out[1] = v0x1;
        out[2] = v0x2;
        out[3] = v0x3;
        w0 += 4;
        const int8_t v1x0 = w1[0];
        const int8_t v1x1 = w1[1];
        const int8_t v1x2 = w1[2];
        const int8_t v1x3 = w1[3];
        ksum1 += (uint32_t) v1x0;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v1x2;
        ksum1 += (uint32_t) v1x3;
        out[4] = v1x0;
        out[5] = v1x1;
        out[6] = v1x2;
        out[7] = v1x3;
        w1 += 4;
        const int8_t v2x0 = w2[0];
        const int8_t v2x1 = w2[1];
        const int8_t v2x2 = w2[2];
        const int8_t v2x3 = w2[3];
        ksum2 += (uint32_t) v2x0;
        ksum2 += (uint32_t) v2x1;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v2x3;
        out[8] = v2x0;
        out[9] = v2x1;
        out[10] = v2x2;
        out[11] = v2x3;
        w2 += 4;
        const int8_t v3x0 = w3[0];
        const int8_t v3x1 = w3[1];
        const int8_t v3x2 = w3[2];
        const int8_t v3x3 = w3[3];
        ksum3 += (uint32_t) v3x0;
        ksum3 += (uint32_t) v3x1;
        ksum3 += (uint32_t) v3x2;
        ksum3 += (uint32_t) v3x3;
        out[12] = v3x0;
        out[13] = v3x1;
        out[14] = v3x2;
        out[15] = v3x3;
        w3 += 4;
        const int8_t v4x0 = w4[0];
        const int8_t v4x1 = w4[1];
        const int8_t v4x2 = w4[2];
        const int8_t v4x3 = w4[3];
        ksum4 += (uint32_t) v4x0;
        ksum4 += (uint32_t) v4x1;
        ksum4 += (uint32_t) v4x2;
        ksum4 += (uint32_t) v4x3;
        out[16] = v4x0;
        out[17] = v4x1;
        out[18] = v4x2;
        out[19] = v4x3;
        w4 += 4;
        const int8_t v5x0 = w5[0];
        const int8_t v5x1 = w5[1];
        const int8_t v5x2 = w5[2];
        const int8_t v5x3 = w5[3];
        ksum5 += (uint32_t) v5x0;
        ksum5 += (uint32_t) v5x1;
        ksum5 += (uint32_t) v5x2;
        ksum5 += (uint32_t) v5x3;
        out[20] = v5x0;
        out[21] = v5x1;
        out[22] = v5x2;
        out[23] = v5x3;
        w5 += 4;
        const int8_t v6x0 = w6[0];
        const int8_t v6x1 = w6[1];
        const int8_t v6x2 = w6[2];
        const int8_t v6x3 = w6[3];
        ksum6 += (uint32_t) v6x0;
        ksum6 += (uint32_t) v6x1;
        ksum6 += (uint32_t) v6x2;
        ksum6 += (uint32_t) v6x3;
        out[24] = v6x0;
        out[25] = v6x1;
        out[26] = v6x2;
        out[27] = v6x3;
        w6 += 4;
        out += 32;
      }

      // KC remainder of 1..3
      if (k != 0) {
        const int8_t v0x0 = 0 < k ? w0[0] : izp;
        const int8_t v0x1 = 1 < k ? w0[1] : izp;
        const int8_t v0x2 = 2 < k ? w0[2] : izp;
        const int8_t v0x3 = 3 < k ? w0[3] : izp;
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v0x1;
        ksum0 += (uint32_t) v0x2;
        ksum0 += (uint32_t) v0x3;
        if (0 < k) {
          out[0] = v0x0;
        }
        if (1 < k) {
          out[1] = v0x1;
        }
        if (2 < k) {
          out[2] = v0x2;
        }
        if (3 < k) {
          out[3] = v0x3;
        }
        w0 += 4;
        const int8_t v1x0 = 0 < k ? w1[0] : izp;
        const int8_t v1x1 = 1 < k ? w1[1] : izp;
        const int8_t v1x2 = 2 < k ? w1[2] : izp;
        const int8_t v1x3 = 3 < k ? w1[3] : izp;
        ksum1 += (uint32_t) v1x0;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v1x2;
        ksum1 += (uint32_t) v1x3;
        if (0 < k) {
          out[4] = v1x0;
        }
        if (1 < k) {
          out[5] = v1x1;
        }
        if (2 < k) {
          out[6] = v1x2;
        }
        if (3 < k) {
          out[7] = v1x3;
        }
        w1 += 4;
        const int8_t v2x0 = 0 < k ? w2[0] : izp;
        const int8_t v2x1 = 1 < k ? w2[1] : izp;
        const int8_t v2x2 = 2 < k ? w2[2] : izp;
        const int8_t v2x3 = 3 < k ? w2[3] : izp;
        ksum2 += (uint32_t) v2x0;
        ksum2 += (uint32_t) v2x1;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v2x3;
        if (0 < k) {
          out[8] = v2x0;
        }
        if (1 < k) {
          out[9] = v2x1;
        }
        if (2 < k) {
          out[10] = v2x2;
        }
        if (3 < k) {
          out[11] = v2x3;
        }
        w2 += 4;
        const int8_t v3x0 = 0 < k ? w3[0] : izp;
        const int8_t v3x1 = 1 < k ? w3[1] : izp;
        const int8_t v3x2 = 2 < k ? w3[2] : izp;
        const int8_t v3x3 = 3 < k ? w3[3] : izp;
        ksum3 += (uint32_t) v3x0;
        ksum3 += (uint32_t) v3x1;
        ksum3 += (uint32_t) v3x2;
        ksum3 += (uint32_t) v3x3;
        if (0 < k) {
          out[12] = v3x0;
        }
        if (1 < k) {
          out[13] = v3x1;
        }
        if (2 < k) {
          out[14] = v3x2;
        }
        if (3 < k) {
          out[15] = v3x3;
        }
        w3 += 4;
        const int8_t v4x0 = 0 < k ? w4[0] : izp;
        const int8_t v4x1 = 1 < k ? w4[1] : izp;
        const int8_t v4x2 = 2 < k ? w4[2] : izp;
        const int8_t v4x3 = 3 < k ? w4[3] : izp;
        ksum4 += (uint32_t) v4x0;
        ksum4 += (uint32_t) v4x1;
        ksum4 += (uint32_t) v4x2;
        ksum4 += (uint32_t) v4x3;
        if (0 < k) {
          out[16] = v4x0;
        }
        if (1 < k) {
          out[17] = v4x1;
        }
        if (2 < k) {
          out[18] = v4x2;
        }
        if (3 < k) {
          out[19] = v4x3;
        }
        w4 += 4;
        const int8_t v5x0 = 0 < k ? w5[0] : izp;
        const int8_t v5x1 = 1 < k ? w5[1] : izp;
        const int8_t v5x2 = 2 < k ? w5[2] : izp;
        const int8_t v5x3 = 3 < k ? w5[3] : izp;
        ksum5 += (uint32_t) v5x0;
        ksum5 += (uint32_t) v5x1;
        ksum5 += (uint32_t) v5x2;
        ksum5 += (uint32_t) v5x3;
        if (0 < k) {
          out[20] = v5x0;
        }
        if (1 < k) {
          out[21] = v5x1;
        }
        if (2 < k) {
          out[22] = v5x2;
        }
        if (3 < k) {
          out[23] = v5x3;
        }
        w5 += 4;
        const int8_t v6x0 = 0 < k ? w6[0] : izp;
        const int8_t v6x1 = 1 < k ? w6[1] : izp;
        const int8_t v6x2 = 2 < k ? w6[2] : izp;
        const int8_t v6x3 = 3 < k ? w6[3] : izp;
        ksum6 += (uint32_t) v6x0;
        ksum6 += (uint32_t) v6x1;
        ksum6 += (uint32_t) v6x2;
        ksum6 += (uint32_t) v6x3;
        if (0 < k) {
          out[24] = v6x0;
        }
        if (1 < k) {
          out[25] = v6x1;
        }
        if (2 < k) {
          out[26] = v6x2;
        }
        if (3 < k) {
          out[27] = v6x3;
        }
        w6 += 4;
        out += 32;
      }

      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      packed_b[2] -= ksum2 * izp;
      packed_b[3] -= ksum3 * izp;
      packed_b[4] -= ksum4 * izp;
      packed_b[5] -= ksum5 * izp;
      packed_b[6] -= ksum6 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
