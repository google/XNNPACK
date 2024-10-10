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

void xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x8c8__scalar(
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
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 128): 128);

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

      // KC main loop multiple of 8x8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        const int8_t v0x0 = w0[0];
        const int8_t v0x1 = w0[1];
        const int8_t v0x2 = w0[2];
        const int8_t v0x3 = w0[3];
        const int8_t v0x4 = w0[4];
        const int8_t v0x5 = w0[5];
        const int8_t v0x6 = w0[6];
        const int8_t v0x7 = w0[7];
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v0x1;
        ksum0 += (uint32_t) v0x2;
        ksum0 += (uint32_t) v0x3;
        ksum0 += (uint32_t) v0x4;
        ksum0 += (uint32_t) v0x5;
        ksum0 += (uint32_t) v0x6;
        ksum0 += (uint32_t) v0x7;
        out[0] = v0x0;
        out[1] = v0x1;
        out[2] = v0x2;
        out[3] = v0x3;
        out[4] = v0x4;
        out[5] = v0x5;
        out[6] = v0x6;
        out[7] = v0x7;
        w0 += 8;
        const int8_t v1x0 = w1[0];
        const int8_t v1x1 = w1[1];
        const int8_t v1x2 = w1[2];
        const int8_t v1x3 = w1[3];
        const int8_t v1x4 = w1[4];
        const int8_t v1x5 = w1[5];
        const int8_t v1x6 = w1[6];
        const int8_t v1x7 = w1[7];
        ksum1 += (uint32_t) v1x0;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v1x2;
        ksum1 += (uint32_t) v1x3;
        ksum1 += (uint32_t) v1x4;
        ksum1 += (uint32_t) v1x5;
        ksum1 += (uint32_t) v1x6;
        ksum1 += (uint32_t) v1x7;
        out[8] = v1x0;
        out[9] = v1x1;
        out[10] = v1x2;
        out[11] = v1x3;
        out[12] = v1x4;
        out[13] = v1x5;
        out[14] = v1x6;
        out[15] = v1x7;
        w1 += 8;
        const int8_t v2x0 = w2[0];
        const int8_t v2x1 = w2[1];
        const int8_t v2x2 = w2[2];
        const int8_t v2x3 = w2[3];
        const int8_t v2x4 = w2[4];
        const int8_t v2x5 = w2[5];
        const int8_t v2x6 = w2[6];
        const int8_t v2x7 = w2[7];
        ksum2 += (uint32_t) v2x0;
        ksum2 += (uint32_t) v2x1;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v2x3;
        ksum2 += (uint32_t) v2x4;
        ksum2 += (uint32_t) v2x5;
        ksum2 += (uint32_t) v2x6;
        ksum2 += (uint32_t) v2x7;
        out[16] = v2x0;
        out[17] = v2x1;
        out[18] = v2x2;
        out[19] = v2x3;
        out[20] = v2x4;
        out[21] = v2x5;
        out[22] = v2x6;
        out[23] = v2x7;
        w2 += 8;
        const int8_t v3x0 = w3[0];
        const int8_t v3x1 = w3[1];
        const int8_t v3x2 = w3[2];
        const int8_t v3x3 = w3[3];
        const int8_t v3x4 = w3[4];
        const int8_t v3x5 = w3[5];
        const int8_t v3x6 = w3[6];
        const int8_t v3x7 = w3[7];
        ksum3 += (uint32_t) v3x0;
        ksum3 += (uint32_t) v3x1;
        ksum3 += (uint32_t) v3x2;
        ksum3 += (uint32_t) v3x3;
        ksum3 += (uint32_t) v3x4;
        ksum3 += (uint32_t) v3x5;
        ksum3 += (uint32_t) v3x6;
        ksum3 += (uint32_t) v3x7;
        out[24] = v3x0;
        out[25] = v3x1;
        out[26] = v3x2;
        out[27] = v3x3;
        out[28] = v3x4;
        out[29] = v3x5;
        out[30] = v3x6;
        out[31] = v3x7;
        w3 += 8;
        const int8_t v4x0 = w4[0];
        const int8_t v4x1 = w4[1];
        const int8_t v4x2 = w4[2];
        const int8_t v4x3 = w4[3];
        const int8_t v4x4 = w4[4];
        const int8_t v4x5 = w4[5];
        const int8_t v4x6 = w4[6];
        const int8_t v4x7 = w4[7];
        ksum4 += (uint32_t) v4x0;
        ksum4 += (uint32_t) v4x1;
        ksum4 += (uint32_t) v4x2;
        ksum4 += (uint32_t) v4x3;
        ksum4 += (uint32_t) v4x4;
        ksum4 += (uint32_t) v4x5;
        ksum4 += (uint32_t) v4x6;
        ksum4 += (uint32_t) v4x7;
        out[32] = v4x0;
        out[33] = v4x1;
        out[34] = v4x2;
        out[35] = v4x3;
        out[36] = v4x4;
        out[37] = v4x5;
        out[38] = v4x6;
        out[39] = v4x7;
        w4 += 8;
        const int8_t v5x0 = w5[0];
        const int8_t v5x1 = w5[1];
        const int8_t v5x2 = w5[2];
        const int8_t v5x3 = w5[3];
        const int8_t v5x4 = w5[4];
        const int8_t v5x5 = w5[5];
        const int8_t v5x6 = w5[6];
        const int8_t v5x7 = w5[7];
        ksum5 += (uint32_t) v5x0;
        ksum5 += (uint32_t) v5x1;
        ksum5 += (uint32_t) v5x2;
        ksum5 += (uint32_t) v5x3;
        ksum5 += (uint32_t) v5x4;
        ksum5 += (uint32_t) v5x5;
        ksum5 += (uint32_t) v5x6;
        ksum5 += (uint32_t) v5x7;
        out[40] = v5x0;
        out[41] = v5x1;
        out[42] = v5x2;
        out[43] = v5x3;
        out[44] = v5x4;
        out[45] = v5x5;
        out[46] = v5x6;
        out[47] = v5x7;
        w5 += 8;
        const int8_t v6x0 = w6[0];
        const int8_t v6x1 = w6[1];
        const int8_t v6x2 = w6[2];
        const int8_t v6x3 = w6[3];
        const int8_t v6x4 = w6[4];
        const int8_t v6x5 = w6[5];
        const int8_t v6x6 = w6[6];
        const int8_t v6x7 = w6[7];
        ksum6 += (uint32_t) v6x0;
        ksum6 += (uint32_t) v6x1;
        ksum6 += (uint32_t) v6x2;
        ksum6 += (uint32_t) v6x3;
        ksum6 += (uint32_t) v6x4;
        ksum6 += (uint32_t) v6x5;
        ksum6 += (uint32_t) v6x6;
        ksum6 += (uint32_t) v6x7;
        out[48] = v6x0;
        out[49] = v6x1;
        out[50] = v6x2;
        out[51] = v6x3;
        out[52] = v6x4;
        out[53] = v6x5;
        out[54] = v6x6;
        out[55] = v6x7;
        w6 += 8;
        const int8_t v7x0 = w7[0];
        const int8_t v7x1 = w7[1];
        const int8_t v7x2 = w7[2];
        const int8_t v7x3 = w7[3];
        const int8_t v7x4 = w7[4];
        const int8_t v7x5 = w7[5];
        const int8_t v7x6 = w7[6];
        const int8_t v7x7 = w7[7];
        ksum7 += (uint32_t) v7x0;
        ksum7 += (uint32_t) v7x1;
        ksum7 += (uint32_t) v7x2;
        ksum7 += (uint32_t) v7x3;
        ksum7 += (uint32_t) v7x4;
        ksum7 += (uint32_t) v7x5;
        ksum7 += (uint32_t) v7x6;
        ksum7 += (uint32_t) v7x7;
        out[56] = v7x0;
        out[57] = v7x1;
        out[58] = v7x2;
        out[59] = v7x3;
        out[60] = v7x4;
        out[61] = v7x5;
        out[62] = v7x6;
        out[63] = v7x7;
        w7 += 8;
        out += 64;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);
        const int8_t v0x0 = w0[0];
        ksum0 += (uint32_t) v0x0;
        out[0] = v0x0;
        if (1 < k) {
          const int8_t v0x1 = w0[1];
          ksum0 += (uint32_t) v0x1;
          out[1] = v0x1;
        }
        if (2 < k) {
          const int8_t v0x2 = w0[2];
          ksum0 += (uint32_t) v0x2;
          out[2] = v0x2;
        }
        if (3 < k) {
          const int8_t v0x3 = w0[3];
          ksum0 += (uint32_t) v0x3;
          out[3] = v0x3;
        }
        if (4 < k) {
          const int8_t v0x4 = w0[4];
          ksum0 += (uint32_t) v0x4;
          out[4] = v0x4;
        }
        if (5 < k) {
          const int8_t v0x5 = w0[5];
          ksum0 += (uint32_t) v0x5;
          out[5] = v0x5;
        }
        if (6 < k) {
          const int8_t v0x6 = w0[6];
          ksum0 += (uint32_t) v0x6;
          out[6] = v0x6;
        }
        if (7 < k) {
          const int8_t v0x7 = w0[7];
          ksum0 += (uint32_t) v0x7;
          out[7] = v0x7;
        }
        w0 += k;
        const int8_t v1x0 = w1[0];
        ksum1 += (uint32_t) v1x0;
        out[8] = v1x0;
        if (1 < k) {
          const int8_t v1x1 = w1[1];
          ksum1 += (uint32_t) v1x1;
          out[9] = v1x1;
        }
        if (2 < k) {
          const int8_t v1x2 = w1[2];
          ksum1 += (uint32_t) v1x2;
          out[10] = v1x2;
        }
        if (3 < k) {
          const int8_t v1x3 = w1[3];
          ksum1 += (uint32_t) v1x3;
          out[11] = v1x3;
        }
        if (4 < k) {
          const int8_t v1x4 = w1[4];
          ksum1 += (uint32_t) v1x4;
          out[12] = v1x4;
        }
        if (5 < k) {
          const int8_t v1x5 = w1[5];
          ksum1 += (uint32_t) v1x5;
          out[13] = v1x5;
        }
        if (6 < k) {
          const int8_t v1x6 = w1[6];
          ksum1 += (uint32_t) v1x6;
          out[14] = v1x6;
        }
        if (7 < k) {
          const int8_t v1x7 = w1[7];
          ksum1 += (uint32_t) v1x7;
          out[15] = v1x7;
        }
        w1 += k;
        const int8_t v2x0 = w2[0];
        ksum2 += (uint32_t) v2x0;
        out[16] = v2x0;
        if (1 < k) {
          const int8_t v2x1 = w2[1];
          ksum2 += (uint32_t) v2x1;
          out[17] = v2x1;
        }
        if (2 < k) {
          const int8_t v2x2 = w2[2];
          ksum2 += (uint32_t) v2x2;
          out[18] = v2x2;
        }
        if (3 < k) {
          const int8_t v2x3 = w2[3];
          ksum2 += (uint32_t) v2x3;
          out[19] = v2x3;
        }
        if (4 < k) {
          const int8_t v2x4 = w2[4];
          ksum2 += (uint32_t) v2x4;
          out[20] = v2x4;
        }
        if (5 < k) {
          const int8_t v2x5 = w2[5];
          ksum2 += (uint32_t) v2x5;
          out[21] = v2x5;
        }
        if (6 < k) {
          const int8_t v2x6 = w2[6];
          ksum2 += (uint32_t) v2x6;
          out[22] = v2x6;
        }
        if (7 < k) {
          const int8_t v2x7 = w2[7];
          ksum2 += (uint32_t) v2x7;
          out[23] = v2x7;
        }
        w2 += k;
        const int8_t v3x0 = w3[0];
        ksum3 += (uint32_t) v3x0;
        out[24] = v3x0;
        if (1 < k) {
          const int8_t v3x1 = w3[1];
          ksum3 += (uint32_t) v3x1;
          out[25] = v3x1;
        }
        if (2 < k) {
          const int8_t v3x2 = w3[2];
          ksum3 += (uint32_t) v3x2;
          out[26] = v3x2;
        }
        if (3 < k) {
          const int8_t v3x3 = w3[3];
          ksum3 += (uint32_t) v3x3;
          out[27] = v3x3;
        }
        if (4 < k) {
          const int8_t v3x4 = w3[4];
          ksum3 += (uint32_t) v3x4;
          out[28] = v3x4;
        }
        if (5 < k) {
          const int8_t v3x5 = w3[5];
          ksum3 += (uint32_t) v3x5;
          out[29] = v3x5;
        }
        if (6 < k) {
          const int8_t v3x6 = w3[6];
          ksum3 += (uint32_t) v3x6;
          out[30] = v3x6;
        }
        if (7 < k) {
          const int8_t v3x7 = w3[7];
          ksum3 += (uint32_t) v3x7;
          out[31] = v3x7;
        }
        w3 += k;
        const int8_t v4x0 = w4[0];
        ksum4 += (uint32_t) v4x0;
        out[32] = v4x0;
        if (1 < k) {
          const int8_t v4x1 = w4[1];
          ksum4 += (uint32_t) v4x1;
          out[33] = v4x1;
        }
        if (2 < k) {
          const int8_t v4x2 = w4[2];
          ksum4 += (uint32_t) v4x2;
          out[34] = v4x2;
        }
        if (3 < k) {
          const int8_t v4x3 = w4[3];
          ksum4 += (uint32_t) v4x3;
          out[35] = v4x3;
        }
        if (4 < k) {
          const int8_t v4x4 = w4[4];
          ksum4 += (uint32_t) v4x4;
          out[36] = v4x4;
        }
        if (5 < k) {
          const int8_t v4x5 = w4[5];
          ksum4 += (uint32_t) v4x5;
          out[37] = v4x5;
        }
        if (6 < k) {
          const int8_t v4x6 = w4[6];
          ksum4 += (uint32_t) v4x6;
          out[38] = v4x6;
        }
        if (7 < k) {
          const int8_t v4x7 = w4[7];
          ksum4 += (uint32_t) v4x7;
          out[39] = v4x7;
        }
        w4 += k;
        const int8_t v5x0 = w5[0];
        ksum5 += (uint32_t) v5x0;
        out[40] = v5x0;
        if (1 < k) {
          const int8_t v5x1 = w5[1];
          ksum5 += (uint32_t) v5x1;
          out[41] = v5x1;
        }
        if (2 < k) {
          const int8_t v5x2 = w5[2];
          ksum5 += (uint32_t) v5x2;
          out[42] = v5x2;
        }
        if (3 < k) {
          const int8_t v5x3 = w5[3];
          ksum5 += (uint32_t) v5x3;
          out[43] = v5x3;
        }
        if (4 < k) {
          const int8_t v5x4 = w5[4];
          ksum5 += (uint32_t) v5x4;
          out[44] = v5x4;
        }
        if (5 < k) {
          const int8_t v5x5 = w5[5];
          ksum5 += (uint32_t) v5x5;
          out[45] = v5x5;
        }
        if (6 < k) {
          const int8_t v5x6 = w5[6];
          ksum5 += (uint32_t) v5x6;
          out[46] = v5x6;
        }
        if (7 < k) {
          const int8_t v5x7 = w5[7];
          ksum5 += (uint32_t) v5x7;
          out[47] = v5x7;
        }
        w5 += k;
        const int8_t v6x0 = w6[0];
        ksum6 += (uint32_t) v6x0;
        out[48] = v6x0;
        if (1 < k) {
          const int8_t v6x1 = w6[1];
          ksum6 += (uint32_t) v6x1;
          out[49] = v6x1;
        }
        if (2 < k) {
          const int8_t v6x2 = w6[2];
          ksum6 += (uint32_t) v6x2;
          out[50] = v6x2;
        }
        if (3 < k) {
          const int8_t v6x3 = w6[3];
          ksum6 += (uint32_t) v6x3;
          out[51] = v6x3;
        }
        if (4 < k) {
          const int8_t v6x4 = w6[4];
          ksum6 += (uint32_t) v6x4;
          out[52] = v6x4;
        }
        if (5 < k) {
          const int8_t v6x5 = w6[5];
          ksum6 += (uint32_t) v6x5;
          out[53] = v6x5;
        }
        if (6 < k) {
          const int8_t v6x6 = w6[6];
          ksum6 += (uint32_t) v6x6;
          out[54] = v6x6;
        }
        if (7 < k) {
          const int8_t v6x7 = w6[7];
          ksum6 += (uint32_t) v6x7;
          out[55] = v6x7;
        }
        w6 += k;
        const int8_t v7x0 = w7[0];
        ksum7 += (uint32_t) v7x0;
        out[56] = v7x0;
        if (1 < k) {
          const int8_t v7x1 = w7[1];
          ksum7 += (uint32_t) v7x1;
          out[57] = v7x1;
        }
        if (2 < k) {
          const int8_t v7x2 = w7[2];
          ksum7 += (uint32_t) v7x2;
          out[58] = v7x2;
        }
        if (3 < k) {
          const int8_t v7x3 = w7[3];
          ksum7 += (uint32_t) v7x3;
          out[59] = v7x3;
        }
        if (4 < k) {
          const int8_t v7x4 = w7[4];
          ksum7 += (uint32_t) v7x4;
          out[60] = v7x4;
        }
        if (5 < k) {
          const int8_t v7x5 = w7[5];
          ksum7 += (uint32_t) v7x5;
          out[61] = v7x5;
        }
        if (6 < k) {
          const int8_t v7x6 = w7[6];
          ksum7 += (uint32_t) v7x6;
          out[62] = v7x6;
        }
        if (7 < k) {
          const int8_t v7x7 = w7[7];
          ksum7 += (uint32_t) v7x7;
          out[63] = v7x7;
        }
        w7 += k;
        out += 64;
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

      // KC main loop multiple of 8x8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        const int8_t v0x0 = w0[0];
        const int8_t v0x1 = w0[1];
        const int8_t v0x2 = w0[2];
        const int8_t v0x3 = w0[3];
        const int8_t v0x4 = w0[4];
        const int8_t v0x5 = w0[5];
        const int8_t v0x6 = w0[6];
        const int8_t v0x7 = w0[7];
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v0x1;
        ksum0 += (uint32_t) v0x2;
        ksum0 += (uint32_t) v0x3;
        ksum0 += (uint32_t) v0x4;
        ksum0 += (uint32_t) v0x5;
        ksum0 += (uint32_t) v0x6;
        ksum0 += (uint32_t) v0x7;
        out[0] = v0x0;
        out[1] = v0x1;
        out[2] = v0x2;
        out[3] = v0x3;
        out[4] = v0x4;
        out[5] = v0x5;
        out[6] = v0x6;
        out[7] = v0x7;
        w0 += 8;
        const int8_t v1x0 = w1[0];
        const int8_t v1x1 = w1[1];
        const int8_t v1x2 = w1[2];
        const int8_t v1x3 = w1[3];
        const int8_t v1x4 = w1[4];
        const int8_t v1x5 = w1[5];
        const int8_t v1x6 = w1[6];
        const int8_t v1x7 = w1[7];
        ksum1 += (uint32_t) v1x0;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v1x2;
        ksum1 += (uint32_t) v1x3;
        ksum1 += (uint32_t) v1x4;
        ksum1 += (uint32_t) v1x5;
        ksum1 += (uint32_t) v1x6;
        ksum1 += (uint32_t) v1x7;
        out[8] = v1x0;
        out[9] = v1x1;
        out[10] = v1x2;
        out[11] = v1x3;
        out[12] = v1x4;
        out[13] = v1x5;
        out[14] = v1x6;
        out[15] = v1x7;
        w1 += 8;
        const int8_t v2x0 = w2[0];
        const int8_t v2x1 = w2[1];
        const int8_t v2x2 = w2[2];
        const int8_t v2x3 = w2[3];
        const int8_t v2x4 = w2[4];
        const int8_t v2x5 = w2[5];
        const int8_t v2x6 = w2[6];
        const int8_t v2x7 = w2[7];
        ksum2 += (uint32_t) v2x0;
        ksum2 += (uint32_t) v2x1;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v2x3;
        ksum2 += (uint32_t) v2x4;
        ksum2 += (uint32_t) v2x5;
        ksum2 += (uint32_t) v2x6;
        ksum2 += (uint32_t) v2x7;
        out[16] = v2x0;
        out[17] = v2x1;
        out[18] = v2x2;
        out[19] = v2x3;
        out[20] = v2x4;
        out[21] = v2x5;
        out[22] = v2x6;
        out[23] = v2x7;
        w2 += 8;
        const int8_t v3x0 = w3[0];
        const int8_t v3x1 = w3[1];
        const int8_t v3x2 = w3[2];
        const int8_t v3x3 = w3[3];
        const int8_t v3x4 = w3[4];
        const int8_t v3x5 = w3[5];
        const int8_t v3x6 = w3[6];
        const int8_t v3x7 = w3[7];
        ksum3 += (uint32_t) v3x0;
        ksum3 += (uint32_t) v3x1;
        ksum3 += (uint32_t) v3x2;
        ksum3 += (uint32_t) v3x3;
        ksum3 += (uint32_t) v3x4;
        ksum3 += (uint32_t) v3x5;
        ksum3 += (uint32_t) v3x6;
        ksum3 += (uint32_t) v3x7;
        out[24] = v3x0;
        out[25] = v3x1;
        out[26] = v3x2;
        out[27] = v3x3;
        out[28] = v3x4;
        out[29] = v3x5;
        out[30] = v3x6;
        out[31] = v3x7;
        w3 += 8;
        const int8_t v4x0 = w4[0];
        const int8_t v4x1 = w4[1];
        const int8_t v4x2 = w4[2];
        const int8_t v4x3 = w4[3];
        const int8_t v4x4 = w4[4];
        const int8_t v4x5 = w4[5];
        const int8_t v4x6 = w4[6];
        const int8_t v4x7 = w4[7];
        ksum4 += (uint32_t) v4x0;
        ksum4 += (uint32_t) v4x1;
        ksum4 += (uint32_t) v4x2;
        ksum4 += (uint32_t) v4x3;
        ksum4 += (uint32_t) v4x4;
        ksum4 += (uint32_t) v4x5;
        ksum4 += (uint32_t) v4x6;
        ksum4 += (uint32_t) v4x7;
        out[32] = v4x0;
        out[33] = v4x1;
        out[34] = v4x2;
        out[35] = v4x3;
        out[36] = v4x4;
        out[37] = v4x5;
        out[38] = v4x6;
        out[39] = v4x7;
        w4 += 8;
        const int8_t v5x0 = w5[0];
        const int8_t v5x1 = w5[1];
        const int8_t v5x2 = w5[2];
        const int8_t v5x3 = w5[3];
        const int8_t v5x4 = w5[4];
        const int8_t v5x5 = w5[5];
        const int8_t v5x6 = w5[6];
        const int8_t v5x7 = w5[7];
        ksum5 += (uint32_t) v5x0;
        ksum5 += (uint32_t) v5x1;
        ksum5 += (uint32_t) v5x2;
        ksum5 += (uint32_t) v5x3;
        ksum5 += (uint32_t) v5x4;
        ksum5 += (uint32_t) v5x5;
        ksum5 += (uint32_t) v5x6;
        ksum5 += (uint32_t) v5x7;
        out[40] = v5x0;
        out[41] = v5x1;
        out[42] = v5x2;
        out[43] = v5x3;
        out[44] = v5x4;
        out[45] = v5x5;
        out[46] = v5x6;
        out[47] = v5x7;
        w5 += 8;
        const int8_t v6x0 = w6[0];
        const int8_t v6x1 = w6[1];
        const int8_t v6x2 = w6[2];
        const int8_t v6x3 = w6[3];
        const int8_t v6x4 = w6[4];
        const int8_t v6x5 = w6[5];
        const int8_t v6x6 = w6[6];
        const int8_t v6x7 = w6[7];
        ksum6 += (uint32_t) v6x0;
        ksum6 += (uint32_t) v6x1;
        ksum6 += (uint32_t) v6x2;
        ksum6 += (uint32_t) v6x3;
        ksum6 += (uint32_t) v6x4;
        ksum6 += (uint32_t) v6x5;
        ksum6 += (uint32_t) v6x6;
        ksum6 += (uint32_t) v6x7;
        out[48] = v6x0;
        out[49] = v6x1;
        out[50] = v6x2;
        out[51] = v6x3;
        out[52] = v6x4;
        out[53] = v6x5;
        out[54] = v6x6;
        out[55] = v6x7;
        w6 += 8;
        out += 64;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);
        const int8_t v0x0 = w0[0];
        ksum0 += (uint32_t) v0x0;
        out[0] = v0x0;
        if (1 < k) {
          const int8_t v0x1 = w0[1];
          ksum0 += (uint32_t) v0x1;
          out[1] = v0x1;
        }
        if (2 < k) {
          const int8_t v0x2 = w0[2];
          ksum0 += (uint32_t) v0x2;
          out[2] = v0x2;
        }
        if (3 < k) {
          const int8_t v0x3 = w0[3];
          ksum0 += (uint32_t) v0x3;
          out[3] = v0x3;
        }
        if (4 < k) {
          const int8_t v0x4 = w0[4];
          ksum0 += (uint32_t) v0x4;
          out[4] = v0x4;
        }
        if (5 < k) {
          const int8_t v0x5 = w0[5];
          ksum0 += (uint32_t) v0x5;
          out[5] = v0x5;
        }
        if (6 < k) {
          const int8_t v0x6 = w0[6];
          ksum0 += (uint32_t) v0x6;
          out[6] = v0x6;
        }
        if (7 < k) {
          const int8_t v0x7 = w0[7];
          ksum0 += (uint32_t) v0x7;
          out[7] = v0x7;
        }
        w0 += k;
        const int8_t v1x0 = w1[0];
        ksum1 += (uint32_t) v1x0;
        out[8] = v1x0;
        if (1 < k) {
          const int8_t v1x1 = w1[1];
          ksum1 += (uint32_t) v1x1;
          out[9] = v1x1;
        }
        if (2 < k) {
          const int8_t v1x2 = w1[2];
          ksum1 += (uint32_t) v1x2;
          out[10] = v1x2;
        }
        if (3 < k) {
          const int8_t v1x3 = w1[3];
          ksum1 += (uint32_t) v1x3;
          out[11] = v1x3;
        }
        if (4 < k) {
          const int8_t v1x4 = w1[4];
          ksum1 += (uint32_t) v1x4;
          out[12] = v1x4;
        }
        if (5 < k) {
          const int8_t v1x5 = w1[5];
          ksum1 += (uint32_t) v1x5;
          out[13] = v1x5;
        }
        if (6 < k) {
          const int8_t v1x6 = w1[6];
          ksum1 += (uint32_t) v1x6;
          out[14] = v1x6;
        }
        if (7 < k) {
          const int8_t v1x7 = w1[7];
          ksum1 += (uint32_t) v1x7;
          out[15] = v1x7;
        }
        w1 += k;
        const int8_t v2x0 = w2[0];
        ksum2 += (uint32_t) v2x0;
        out[16] = v2x0;
        if (1 < k) {
          const int8_t v2x1 = w2[1];
          ksum2 += (uint32_t) v2x1;
          out[17] = v2x1;
        }
        if (2 < k) {
          const int8_t v2x2 = w2[2];
          ksum2 += (uint32_t) v2x2;
          out[18] = v2x2;
        }
        if (3 < k) {
          const int8_t v2x3 = w2[3];
          ksum2 += (uint32_t) v2x3;
          out[19] = v2x3;
        }
        if (4 < k) {
          const int8_t v2x4 = w2[4];
          ksum2 += (uint32_t) v2x4;
          out[20] = v2x4;
        }
        if (5 < k) {
          const int8_t v2x5 = w2[5];
          ksum2 += (uint32_t) v2x5;
          out[21] = v2x5;
        }
        if (6 < k) {
          const int8_t v2x6 = w2[6];
          ksum2 += (uint32_t) v2x6;
          out[22] = v2x6;
        }
        if (7 < k) {
          const int8_t v2x7 = w2[7];
          ksum2 += (uint32_t) v2x7;
          out[23] = v2x7;
        }
        w2 += k;
        const int8_t v3x0 = w3[0];
        ksum3 += (uint32_t) v3x0;
        out[24] = v3x0;
        if (1 < k) {
          const int8_t v3x1 = w3[1];
          ksum3 += (uint32_t) v3x1;
          out[25] = v3x1;
        }
        if (2 < k) {
          const int8_t v3x2 = w3[2];
          ksum3 += (uint32_t) v3x2;
          out[26] = v3x2;
        }
        if (3 < k) {
          const int8_t v3x3 = w3[3];
          ksum3 += (uint32_t) v3x3;
          out[27] = v3x3;
        }
        if (4 < k) {
          const int8_t v3x4 = w3[4];
          ksum3 += (uint32_t) v3x4;
          out[28] = v3x4;
        }
        if (5 < k) {
          const int8_t v3x5 = w3[5];
          ksum3 += (uint32_t) v3x5;
          out[29] = v3x5;
        }
        if (6 < k) {
          const int8_t v3x6 = w3[6];
          ksum3 += (uint32_t) v3x6;
          out[30] = v3x6;
        }
        if (7 < k) {
          const int8_t v3x7 = w3[7];
          ksum3 += (uint32_t) v3x7;
          out[31] = v3x7;
        }
        w3 += k;
        const int8_t v4x0 = w4[0];
        ksum4 += (uint32_t) v4x0;
        out[32] = v4x0;
        if (1 < k) {
          const int8_t v4x1 = w4[1];
          ksum4 += (uint32_t) v4x1;
          out[33] = v4x1;
        }
        if (2 < k) {
          const int8_t v4x2 = w4[2];
          ksum4 += (uint32_t) v4x2;
          out[34] = v4x2;
        }
        if (3 < k) {
          const int8_t v4x3 = w4[3];
          ksum4 += (uint32_t) v4x3;
          out[35] = v4x3;
        }
        if (4 < k) {
          const int8_t v4x4 = w4[4];
          ksum4 += (uint32_t) v4x4;
          out[36] = v4x4;
        }
        if (5 < k) {
          const int8_t v4x5 = w4[5];
          ksum4 += (uint32_t) v4x5;
          out[37] = v4x5;
        }
        if (6 < k) {
          const int8_t v4x6 = w4[6];
          ksum4 += (uint32_t) v4x6;
          out[38] = v4x6;
        }
        if (7 < k) {
          const int8_t v4x7 = w4[7];
          ksum4 += (uint32_t) v4x7;
          out[39] = v4x7;
        }
        w4 += k;
        const int8_t v5x0 = w5[0];
        ksum5 += (uint32_t) v5x0;
        out[40] = v5x0;
        if (1 < k) {
          const int8_t v5x1 = w5[1];
          ksum5 += (uint32_t) v5x1;
          out[41] = v5x1;
        }
        if (2 < k) {
          const int8_t v5x2 = w5[2];
          ksum5 += (uint32_t) v5x2;
          out[42] = v5x2;
        }
        if (3 < k) {
          const int8_t v5x3 = w5[3];
          ksum5 += (uint32_t) v5x3;
          out[43] = v5x3;
        }
        if (4 < k) {
          const int8_t v5x4 = w5[4];
          ksum5 += (uint32_t) v5x4;
          out[44] = v5x4;
        }
        if (5 < k) {
          const int8_t v5x5 = w5[5];
          ksum5 += (uint32_t) v5x5;
          out[45] = v5x5;
        }
        if (6 < k) {
          const int8_t v5x6 = w5[6];
          ksum5 += (uint32_t) v5x6;
          out[46] = v5x6;
        }
        if (7 < k) {
          const int8_t v5x7 = w5[7];
          ksum5 += (uint32_t) v5x7;
          out[47] = v5x7;
        }
        w5 += k;
        const int8_t v6x0 = w6[0];
        ksum6 += (uint32_t) v6x0;
        out[48] = v6x0;
        if (1 < k) {
          const int8_t v6x1 = w6[1];
          ksum6 += (uint32_t) v6x1;
          out[49] = v6x1;
        }
        if (2 < k) {
          const int8_t v6x2 = w6[2];
          ksum6 += (uint32_t) v6x2;
          out[50] = v6x2;
        }
        if (3 < k) {
          const int8_t v6x3 = w6[3];
          ksum6 += (uint32_t) v6x3;
          out[51] = v6x3;
        }
        if (4 < k) {
          const int8_t v6x4 = w6[4];
          ksum6 += (uint32_t) v6x4;
          out[52] = v6x4;
        }
        if (5 < k) {
          const int8_t v6x5 = w6[5];
          ksum6 += (uint32_t) v6x5;
          out[53] = v6x5;
        }
        if (6 < k) {
          const int8_t v6x6 = w6[6];
          ksum6 += (uint32_t) v6x6;
          out[54] = v6x6;
        }
        if (7 < k) {
          const int8_t v6x7 = w6[7];
          ksum6 += (uint32_t) v6x7;
          out[55] = v6x7;
        }
        w6 += k;
        out += 64;
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
