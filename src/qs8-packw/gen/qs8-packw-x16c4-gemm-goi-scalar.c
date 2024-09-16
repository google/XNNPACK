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

void xnn_qs8_packw_gemm_goi_ukernel_x16c4__scalar(
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
  assert(nr == 16);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = params ? (uint32_t) ((const struct xnn_qs8_packw_params*) params)->input_zero_point : 0;

  do {
    // NC main loop multiple of 16
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 16; n -= 16) {
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
        ((int32_t*) out)[8] = b[8];
        ((int32_t*) out)[9] = b[9];
        ((int32_t*) out)[10] = b[10];
        ((int32_t*) out)[11] = b[11];
        ((int32_t*) out)[12] = b[12];
        ((int32_t*) out)[13] = b[13];
        ((int32_t*) out)[14] = b[14];
        ((int32_t*) out)[15] = b[15];
        b += 16;
      } else {
        ((int32_t*) out)[0] = 0;
        ((int32_t*) out)[1] = 0;
        ((int32_t*) out)[2] = 0;
        ((int32_t*) out)[3] = 0;
        ((int32_t*) out)[4] = 0;
        ((int32_t*) out)[5] = 0;
        ((int32_t*) out)[6] = 0;
        ((int32_t*) out)[7] = 0;
        ((int32_t*) out)[8] = 0;
        ((int32_t*) out)[9] = 0;
        ((int32_t*) out)[10] = 0;
        ((int32_t*) out)[11] = 0;
        ((int32_t*) out)[12] = 0;
        ((int32_t*) out)[13] = 0;
        ((int32_t*) out)[14] = 0;
        ((int32_t*) out)[15] = 0;
      }
      out += 16 * sizeof(int32_t);

      const int8_t* w1 = w0 + kc;
      const int8_t* w2 = w1 + kc;
      const int8_t* w3 = w2 + kc;
      const int8_t* w4 = w3 + kc;
      const int8_t* w5 = w4 + kc;
      const int8_t* w6 = w5 + kc;
      const int8_t* w7 = w6 + kc;
      const int8_t* w8 = w7 + kc;
      const int8_t* w9 = w8 + kc;
      const int8_t* w10 = w9 + kc;
      const int8_t* w11 = w10 + kc;
      const int8_t* w12 = w11 + kc;
      const int8_t* w13 = w12 + kc;
      const int8_t* w14 = w13 + kc;
      const int8_t* w15 = w14 + kc;
      uint32_t ksum0 = 0;
      uint32_t ksum1 = 0;
      uint32_t ksum2 = 0;
      uint32_t ksum3 = 0;
      uint32_t ksum4 = 0;
      uint32_t ksum5 = 0;
      uint32_t ksum6 = 0;
      uint32_t ksum7 = 0;
      uint32_t ksum8 = 0;
      uint32_t ksum9 = 0;
      uint32_t ksum10 = 0;
      uint32_t ksum11 = 0;
      uint32_t ksum12 = 0;
      uint32_t ksum13 = 0;
      uint32_t ksum14 = 0;
      uint32_t ksum15 = 0;

      // KC main loop multiple of 16x4
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
        const int8_t v8x0 = w8[0];
        const int8_t v8x1 = w8[1];
        const int8_t v8x2 = w8[2];
        const int8_t v8x3 = w8[3];
        ksum8 += (uint32_t) v8x0;
        ksum8 += (uint32_t) v8x1;
        ksum8 += (uint32_t) v8x2;
        ksum8 += (uint32_t) v8x3;
        out[32] = v8x0;
        out[33] = v8x1;
        out[34] = v8x2;
        out[35] = v8x3;
        w8 += 4;
        const int8_t v9x0 = w9[0];
        const int8_t v9x1 = w9[1];
        const int8_t v9x2 = w9[2];
        const int8_t v9x3 = w9[3];
        ksum9 += (uint32_t) v9x0;
        ksum9 += (uint32_t) v9x1;
        ksum9 += (uint32_t) v9x2;
        ksum9 += (uint32_t) v9x3;
        out[36] = v9x0;
        out[37] = v9x1;
        out[38] = v9x2;
        out[39] = v9x3;
        w9 += 4;
        const int8_t v10x0 = w10[0];
        const int8_t v10x1 = w10[1];
        const int8_t v10x2 = w10[2];
        const int8_t v10x3 = w10[3];
        ksum10 += (uint32_t) v10x0;
        ksum10 += (uint32_t) v10x1;
        ksum10 += (uint32_t) v10x2;
        ksum10 += (uint32_t) v10x3;
        out[40] = v10x0;
        out[41] = v10x1;
        out[42] = v10x2;
        out[43] = v10x3;
        w10 += 4;
        const int8_t v11x0 = w11[0];
        const int8_t v11x1 = w11[1];
        const int8_t v11x2 = w11[2];
        const int8_t v11x3 = w11[3];
        ksum11 += (uint32_t) v11x0;
        ksum11 += (uint32_t) v11x1;
        ksum11 += (uint32_t) v11x2;
        ksum11 += (uint32_t) v11x3;
        out[44] = v11x0;
        out[45] = v11x1;
        out[46] = v11x2;
        out[47] = v11x3;
        w11 += 4;
        const int8_t v12x0 = w12[0];
        const int8_t v12x1 = w12[1];
        const int8_t v12x2 = w12[2];
        const int8_t v12x3 = w12[3];
        ksum12 += (uint32_t) v12x0;
        ksum12 += (uint32_t) v12x1;
        ksum12 += (uint32_t) v12x2;
        ksum12 += (uint32_t) v12x3;
        out[48] = v12x0;
        out[49] = v12x1;
        out[50] = v12x2;
        out[51] = v12x3;
        w12 += 4;
        const int8_t v13x0 = w13[0];
        const int8_t v13x1 = w13[1];
        const int8_t v13x2 = w13[2];
        const int8_t v13x3 = w13[3];
        ksum13 += (uint32_t) v13x0;
        ksum13 += (uint32_t) v13x1;
        ksum13 += (uint32_t) v13x2;
        ksum13 += (uint32_t) v13x3;
        out[52] = v13x0;
        out[53] = v13x1;
        out[54] = v13x2;
        out[55] = v13x3;
        w13 += 4;
        const int8_t v14x0 = w14[0];
        const int8_t v14x1 = w14[1];
        const int8_t v14x2 = w14[2];
        const int8_t v14x3 = w14[3];
        ksum14 += (uint32_t) v14x0;
        ksum14 += (uint32_t) v14x1;
        ksum14 += (uint32_t) v14x2;
        ksum14 += (uint32_t) v14x3;
        out[56] = v14x0;
        out[57] = v14x1;
        out[58] = v14x2;
        out[59] = v14x3;
        w14 += 4;
        const int8_t v15x0 = w15[0];
        const int8_t v15x1 = w15[1];
        const int8_t v15x2 = w15[2];
        const int8_t v15x3 = w15[3];
        ksum15 += (uint32_t) v15x0;
        ksum15 += (uint32_t) v15x1;
        ksum15 += (uint32_t) v15x2;
        ksum15 += (uint32_t) v15x3;
        out[60] = v15x0;
        out[61] = v15x1;
        out[62] = v15x2;
        out[63] = v15x3;
        w15 += 4;
        out += 64;
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
        const int8_t v8x0 = 0 < k ? w8[0] : izp;
        const int8_t v8x1 = 1 < k ? w8[1] : izp;
        const int8_t v8x2 = 2 < k ? w8[2] : izp;
        const int8_t v8x3 = 3 < k ? w8[3] : izp;
        ksum8 += (uint32_t) v8x0;
        ksum8 += (uint32_t) v8x1;
        ksum8 += (uint32_t) v8x2;
        ksum8 += (uint32_t) v8x3;
        if (0 < k) {
          out[32] = v8x0;
        }
        if (1 < k) {
          out[33] = v8x1;
        }
        if (2 < k) {
          out[34] = v8x2;
        }
        if (3 < k) {
          out[35] = v8x3;
        }
        w8 += 4;
        const int8_t v9x0 = 0 < k ? w9[0] : izp;
        const int8_t v9x1 = 1 < k ? w9[1] : izp;
        const int8_t v9x2 = 2 < k ? w9[2] : izp;
        const int8_t v9x3 = 3 < k ? w9[3] : izp;
        ksum9 += (uint32_t) v9x0;
        ksum9 += (uint32_t) v9x1;
        ksum9 += (uint32_t) v9x2;
        ksum9 += (uint32_t) v9x3;
        if (0 < k) {
          out[36] = v9x0;
        }
        if (1 < k) {
          out[37] = v9x1;
        }
        if (2 < k) {
          out[38] = v9x2;
        }
        if (3 < k) {
          out[39] = v9x3;
        }
        w9 += 4;
        const int8_t v10x0 = 0 < k ? w10[0] : izp;
        const int8_t v10x1 = 1 < k ? w10[1] : izp;
        const int8_t v10x2 = 2 < k ? w10[2] : izp;
        const int8_t v10x3 = 3 < k ? w10[3] : izp;
        ksum10 += (uint32_t) v10x0;
        ksum10 += (uint32_t) v10x1;
        ksum10 += (uint32_t) v10x2;
        ksum10 += (uint32_t) v10x3;
        if (0 < k) {
          out[40] = v10x0;
        }
        if (1 < k) {
          out[41] = v10x1;
        }
        if (2 < k) {
          out[42] = v10x2;
        }
        if (3 < k) {
          out[43] = v10x3;
        }
        w10 += 4;
        const int8_t v11x0 = 0 < k ? w11[0] : izp;
        const int8_t v11x1 = 1 < k ? w11[1] : izp;
        const int8_t v11x2 = 2 < k ? w11[2] : izp;
        const int8_t v11x3 = 3 < k ? w11[3] : izp;
        ksum11 += (uint32_t) v11x0;
        ksum11 += (uint32_t) v11x1;
        ksum11 += (uint32_t) v11x2;
        ksum11 += (uint32_t) v11x3;
        if (0 < k) {
          out[44] = v11x0;
        }
        if (1 < k) {
          out[45] = v11x1;
        }
        if (2 < k) {
          out[46] = v11x2;
        }
        if (3 < k) {
          out[47] = v11x3;
        }
        w11 += 4;
        const int8_t v12x0 = 0 < k ? w12[0] : izp;
        const int8_t v12x1 = 1 < k ? w12[1] : izp;
        const int8_t v12x2 = 2 < k ? w12[2] : izp;
        const int8_t v12x3 = 3 < k ? w12[3] : izp;
        ksum12 += (uint32_t) v12x0;
        ksum12 += (uint32_t) v12x1;
        ksum12 += (uint32_t) v12x2;
        ksum12 += (uint32_t) v12x3;
        if (0 < k) {
          out[48] = v12x0;
        }
        if (1 < k) {
          out[49] = v12x1;
        }
        if (2 < k) {
          out[50] = v12x2;
        }
        if (3 < k) {
          out[51] = v12x3;
        }
        w12 += 4;
        const int8_t v13x0 = 0 < k ? w13[0] : izp;
        const int8_t v13x1 = 1 < k ? w13[1] : izp;
        const int8_t v13x2 = 2 < k ? w13[2] : izp;
        const int8_t v13x3 = 3 < k ? w13[3] : izp;
        ksum13 += (uint32_t) v13x0;
        ksum13 += (uint32_t) v13x1;
        ksum13 += (uint32_t) v13x2;
        ksum13 += (uint32_t) v13x3;
        if (0 < k) {
          out[52] = v13x0;
        }
        if (1 < k) {
          out[53] = v13x1;
        }
        if (2 < k) {
          out[54] = v13x2;
        }
        if (3 < k) {
          out[55] = v13x3;
        }
        w13 += 4;
        const int8_t v14x0 = 0 < k ? w14[0] : izp;
        const int8_t v14x1 = 1 < k ? w14[1] : izp;
        const int8_t v14x2 = 2 < k ? w14[2] : izp;
        const int8_t v14x3 = 3 < k ? w14[3] : izp;
        ksum14 += (uint32_t) v14x0;
        ksum14 += (uint32_t) v14x1;
        ksum14 += (uint32_t) v14x2;
        ksum14 += (uint32_t) v14x3;
        if (0 < k) {
          out[56] = v14x0;
        }
        if (1 < k) {
          out[57] = v14x1;
        }
        if (2 < k) {
          out[58] = v14x2;
        }
        if (3 < k) {
          out[59] = v14x3;
        }
        w14 += 4;
        const int8_t v15x0 = 0 < k ? w15[0] : izp;
        const int8_t v15x1 = 1 < k ? w15[1] : izp;
        const int8_t v15x2 = 2 < k ? w15[2] : izp;
        const int8_t v15x3 = 3 < k ? w15[3] : izp;
        ksum15 += (uint32_t) v15x0;
        ksum15 += (uint32_t) v15x1;
        ksum15 += (uint32_t) v15x2;
        ksum15 += (uint32_t) v15x3;
        if (0 < k) {
          out[60] = v15x0;
        }
        if (1 < k) {
          out[61] = v15x1;
        }
        if (2 < k) {
          out[62] = v15x2;
        }
        if (3 < k) {
          out[63] = v15x3;
        }
        w15 += 4;
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
      packed_b[8] -= ksum8 * izp;
      packed_b[9] -= ksum9 * izp;
      packed_b[10] -= ksum10 * izp;
      packed_b[11] -= ksum11 * izp;
      packed_b[12] -= ksum12 * izp;
      packed_b[13] -= ksum13 * izp;
      packed_b[14] -= ksum14 * izp;
      packed_b[15] -= ksum15 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
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
      out += (16 - n) * sizeof(int32_t);

     // NR remainder has less than 16 rows so last row is not loaded
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
      const int8_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const int8_t* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const int8_t* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const int8_t* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const int8_t* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const int8_t* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const int8_t* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const int8_t* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }

      uint32_t ksum0 = 0;
      uint32_t ksum1 = 0;
      uint32_t ksum2 = 0;
      uint32_t ksum3 = 0;
      uint32_t ksum4 = 0;
      uint32_t ksum5 = 0;
      uint32_t ksum6 = 0;
      uint32_t ksum7 = 0;
      uint32_t ksum8 = 0;
      uint32_t ksum9 = 0;
      uint32_t ksum10 = 0;
      uint32_t ksum11 = 0;
      uint32_t ksum12 = 0;
      uint32_t ksum13 = 0;
      uint32_t ksum14 = 0;

      // KC main loop multiple of 16x4
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
        const int8_t v8x0 = w8[0];
        const int8_t v8x1 = w8[1];
        const int8_t v8x2 = w8[2];
        const int8_t v8x3 = w8[3];
        ksum8 += (uint32_t) v8x0;
        ksum8 += (uint32_t) v8x1;
        ksum8 += (uint32_t) v8x2;
        ksum8 += (uint32_t) v8x3;
        out[32] = v8x0;
        out[33] = v8x1;
        out[34] = v8x2;
        out[35] = v8x3;
        w8 += 4;
        const int8_t v9x0 = w9[0];
        const int8_t v9x1 = w9[1];
        const int8_t v9x2 = w9[2];
        const int8_t v9x3 = w9[3];
        ksum9 += (uint32_t) v9x0;
        ksum9 += (uint32_t) v9x1;
        ksum9 += (uint32_t) v9x2;
        ksum9 += (uint32_t) v9x3;
        out[36] = v9x0;
        out[37] = v9x1;
        out[38] = v9x2;
        out[39] = v9x3;
        w9 += 4;
        const int8_t v10x0 = w10[0];
        const int8_t v10x1 = w10[1];
        const int8_t v10x2 = w10[2];
        const int8_t v10x3 = w10[3];
        ksum10 += (uint32_t) v10x0;
        ksum10 += (uint32_t) v10x1;
        ksum10 += (uint32_t) v10x2;
        ksum10 += (uint32_t) v10x3;
        out[40] = v10x0;
        out[41] = v10x1;
        out[42] = v10x2;
        out[43] = v10x3;
        w10 += 4;
        const int8_t v11x0 = w11[0];
        const int8_t v11x1 = w11[1];
        const int8_t v11x2 = w11[2];
        const int8_t v11x3 = w11[3];
        ksum11 += (uint32_t) v11x0;
        ksum11 += (uint32_t) v11x1;
        ksum11 += (uint32_t) v11x2;
        ksum11 += (uint32_t) v11x3;
        out[44] = v11x0;
        out[45] = v11x1;
        out[46] = v11x2;
        out[47] = v11x3;
        w11 += 4;
        const int8_t v12x0 = w12[0];
        const int8_t v12x1 = w12[1];
        const int8_t v12x2 = w12[2];
        const int8_t v12x3 = w12[3];
        ksum12 += (uint32_t) v12x0;
        ksum12 += (uint32_t) v12x1;
        ksum12 += (uint32_t) v12x2;
        ksum12 += (uint32_t) v12x3;
        out[48] = v12x0;
        out[49] = v12x1;
        out[50] = v12x2;
        out[51] = v12x3;
        w12 += 4;
        const int8_t v13x0 = w13[0];
        const int8_t v13x1 = w13[1];
        const int8_t v13x2 = w13[2];
        const int8_t v13x3 = w13[3];
        ksum13 += (uint32_t) v13x0;
        ksum13 += (uint32_t) v13x1;
        ksum13 += (uint32_t) v13x2;
        ksum13 += (uint32_t) v13x3;
        out[52] = v13x0;
        out[53] = v13x1;
        out[54] = v13x2;
        out[55] = v13x3;
        w13 += 4;
        const int8_t v14x0 = w14[0];
        const int8_t v14x1 = w14[1];
        const int8_t v14x2 = w14[2];
        const int8_t v14x3 = w14[3];
        ksum14 += (uint32_t) v14x0;
        ksum14 += (uint32_t) v14x1;
        ksum14 += (uint32_t) v14x2;
        ksum14 += (uint32_t) v14x3;
        out[56] = v14x0;
        out[57] = v14x1;
        out[58] = v14x2;
        out[59] = v14x3;
        w14 += 4;
        out += 64;
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
        const int8_t v8x0 = 0 < k ? w8[0] : izp;
        const int8_t v8x1 = 1 < k ? w8[1] : izp;
        const int8_t v8x2 = 2 < k ? w8[2] : izp;
        const int8_t v8x3 = 3 < k ? w8[3] : izp;
        ksum8 += (uint32_t) v8x0;
        ksum8 += (uint32_t) v8x1;
        ksum8 += (uint32_t) v8x2;
        ksum8 += (uint32_t) v8x3;
        if (0 < k) {
          out[32] = v8x0;
        }
        if (1 < k) {
          out[33] = v8x1;
        }
        if (2 < k) {
          out[34] = v8x2;
        }
        if (3 < k) {
          out[35] = v8x3;
        }
        w8 += 4;
        const int8_t v9x0 = 0 < k ? w9[0] : izp;
        const int8_t v9x1 = 1 < k ? w9[1] : izp;
        const int8_t v9x2 = 2 < k ? w9[2] : izp;
        const int8_t v9x3 = 3 < k ? w9[3] : izp;
        ksum9 += (uint32_t) v9x0;
        ksum9 += (uint32_t) v9x1;
        ksum9 += (uint32_t) v9x2;
        ksum9 += (uint32_t) v9x3;
        if (0 < k) {
          out[36] = v9x0;
        }
        if (1 < k) {
          out[37] = v9x1;
        }
        if (2 < k) {
          out[38] = v9x2;
        }
        if (3 < k) {
          out[39] = v9x3;
        }
        w9 += 4;
        const int8_t v10x0 = 0 < k ? w10[0] : izp;
        const int8_t v10x1 = 1 < k ? w10[1] : izp;
        const int8_t v10x2 = 2 < k ? w10[2] : izp;
        const int8_t v10x3 = 3 < k ? w10[3] : izp;
        ksum10 += (uint32_t) v10x0;
        ksum10 += (uint32_t) v10x1;
        ksum10 += (uint32_t) v10x2;
        ksum10 += (uint32_t) v10x3;
        if (0 < k) {
          out[40] = v10x0;
        }
        if (1 < k) {
          out[41] = v10x1;
        }
        if (2 < k) {
          out[42] = v10x2;
        }
        if (3 < k) {
          out[43] = v10x3;
        }
        w10 += 4;
        const int8_t v11x0 = 0 < k ? w11[0] : izp;
        const int8_t v11x1 = 1 < k ? w11[1] : izp;
        const int8_t v11x2 = 2 < k ? w11[2] : izp;
        const int8_t v11x3 = 3 < k ? w11[3] : izp;
        ksum11 += (uint32_t) v11x0;
        ksum11 += (uint32_t) v11x1;
        ksum11 += (uint32_t) v11x2;
        ksum11 += (uint32_t) v11x3;
        if (0 < k) {
          out[44] = v11x0;
        }
        if (1 < k) {
          out[45] = v11x1;
        }
        if (2 < k) {
          out[46] = v11x2;
        }
        if (3 < k) {
          out[47] = v11x3;
        }
        w11 += 4;
        const int8_t v12x0 = 0 < k ? w12[0] : izp;
        const int8_t v12x1 = 1 < k ? w12[1] : izp;
        const int8_t v12x2 = 2 < k ? w12[2] : izp;
        const int8_t v12x3 = 3 < k ? w12[3] : izp;
        ksum12 += (uint32_t) v12x0;
        ksum12 += (uint32_t) v12x1;
        ksum12 += (uint32_t) v12x2;
        ksum12 += (uint32_t) v12x3;
        if (0 < k) {
          out[48] = v12x0;
        }
        if (1 < k) {
          out[49] = v12x1;
        }
        if (2 < k) {
          out[50] = v12x2;
        }
        if (3 < k) {
          out[51] = v12x3;
        }
        w12 += 4;
        const int8_t v13x0 = 0 < k ? w13[0] : izp;
        const int8_t v13x1 = 1 < k ? w13[1] : izp;
        const int8_t v13x2 = 2 < k ? w13[2] : izp;
        const int8_t v13x3 = 3 < k ? w13[3] : izp;
        ksum13 += (uint32_t) v13x0;
        ksum13 += (uint32_t) v13x1;
        ksum13 += (uint32_t) v13x2;
        ksum13 += (uint32_t) v13x3;
        if (0 < k) {
          out[52] = v13x0;
        }
        if (1 < k) {
          out[53] = v13x1;
        }
        if (2 < k) {
          out[54] = v13x2;
        }
        if (3 < k) {
          out[55] = v13x3;
        }
        w13 += 4;
        const int8_t v14x0 = 0 < k ? w14[0] : izp;
        const int8_t v14x1 = 1 < k ? w14[1] : izp;
        const int8_t v14x2 = 2 < k ? w14[2] : izp;
        const int8_t v14x3 = 3 < k ? w14[3] : izp;
        ksum14 += (uint32_t) v14x0;
        ksum14 += (uint32_t) v14x1;
        ksum14 += (uint32_t) v14x2;
        ksum14 += (uint32_t) v14x3;
        if (0 < k) {
          out[56] = v14x0;
        }
        if (1 < k) {
          out[57] = v14x1;
        }
        if (2 < k) {
          out[58] = v14x2;
        }
        if (3 < k) {
          out[59] = v14x3;
        }
        w14 += 4;
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
      packed_b[8] -= ksum8 * izp;
      packed_b[9] -= ksum9 * izp;
      packed_b[10] -= ksum10 * izp;
      packed_b[11] -= ksum11 * izp;
      packed_b[12] -= ksum12 * izp;
      packed_b[13] -= ksum13 * izp;
      packed_b[14] -= ksum14 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
