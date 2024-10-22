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

void xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x16c8__scalar(
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
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 128): 128);

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

      // KC main loop multiple of 16x8
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
        const int8_t v8x0 = w8[0];
        const int8_t v8x1 = w8[1];
        const int8_t v8x2 = w8[2];
        const int8_t v8x3 = w8[3];
        const int8_t v8x4 = w8[4];
        const int8_t v8x5 = w8[5];
        const int8_t v8x6 = w8[6];
        const int8_t v8x7 = w8[7];
        ksum8 += (uint32_t) v8x0;
        ksum8 += (uint32_t) v8x1;
        ksum8 += (uint32_t) v8x2;
        ksum8 += (uint32_t) v8x3;
        ksum8 += (uint32_t) v8x4;
        ksum8 += (uint32_t) v8x5;
        ksum8 += (uint32_t) v8x6;
        ksum8 += (uint32_t) v8x7;
        out[64] = v8x0;
        out[65] = v8x1;
        out[66] = v8x2;
        out[67] = v8x3;
        out[68] = v8x4;
        out[69] = v8x5;
        out[70] = v8x6;
        out[71] = v8x7;
        w8 += 8;
        const int8_t v9x0 = w9[0];
        const int8_t v9x1 = w9[1];
        const int8_t v9x2 = w9[2];
        const int8_t v9x3 = w9[3];
        const int8_t v9x4 = w9[4];
        const int8_t v9x5 = w9[5];
        const int8_t v9x6 = w9[6];
        const int8_t v9x7 = w9[7];
        ksum9 += (uint32_t) v9x0;
        ksum9 += (uint32_t) v9x1;
        ksum9 += (uint32_t) v9x2;
        ksum9 += (uint32_t) v9x3;
        ksum9 += (uint32_t) v9x4;
        ksum9 += (uint32_t) v9x5;
        ksum9 += (uint32_t) v9x6;
        ksum9 += (uint32_t) v9x7;
        out[72] = v9x0;
        out[73] = v9x1;
        out[74] = v9x2;
        out[75] = v9x3;
        out[76] = v9x4;
        out[77] = v9x5;
        out[78] = v9x6;
        out[79] = v9x7;
        w9 += 8;
        const int8_t v10x0 = w10[0];
        const int8_t v10x1 = w10[1];
        const int8_t v10x2 = w10[2];
        const int8_t v10x3 = w10[3];
        const int8_t v10x4 = w10[4];
        const int8_t v10x5 = w10[5];
        const int8_t v10x6 = w10[6];
        const int8_t v10x7 = w10[7];
        ksum10 += (uint32_t) v10x0;
        ksum10 += (uint32_t) v10x1;
        ksum10 += (uint32_t) v10x2;
        ksum10 += (uint32_t) v10x3;
        ksum10 += (uint32_t) v10x4;
        ksum10 += (uint32_t) v10x5;
        ksum10 += (uint32_t) v10x6;
        ksum10 += (uint32_t) v10x7;
        out[80] = v10x0;
        out[81] = v10x1;
        out[82] = v10x2;
        out[83] = v10x3;
        out[84] = v10x4;
        out[85] = v10x5;
        out[86] = v10x6;
        out[87] = v10x7;
        w10 += 8;
        const int8_t v11x0 = w11[0];
        const int8_t v11x1 = w11[1];
        const int8_t v11x2 = w11[2];
        const int8_t v11x3 = w11[3];
        const int8_t v11x4 = w11[4];
        const int8_t v11x5 = w11[5];
        const int8_t v11x6 = w11[6];
        const int8_t v11x7 = w11[7];
        ksum11 += (uint32_t) v11x0;
        ksum11 += (uint32_t) v11x1;
        ksum11 += (uint32_t) v11x2;
        ksum11 += (uint32_t) v11x3;
        ksum11 += (uint32_t) v11x4;
        ksum11 += (uint32_t) v11x5;
        ksum11 += (uint32_t) v11x6;
        ksum11 += (uint32_t) v11x7;
        out[88] = v11x0;
        out[89] = v11x1;
        out[90] = v11x2;
        out[91] = v11x3;
        out[92] = v11x4;
        out[93] = v11x5;
        out[94] = v11x6;
        out[95] = v11x7;
        w11 += 8;
        const int8_t v12x0 = w12[0];
        const int8_t v12x1 = w12[1];
        const int8_t v12x2 = w12[2];
        const int8_t v12x3 = w12[3];
        const int8_t v12x4 = w12[4];
        const int8_t v12x5 = w12[5];
        const int8_t v12x6 = w12[6];
        const int8_t v12x7 = w12[7];
        ksum12 += (uint32_t) v12x0;
        ksum12 += (uint32_t) v12x1;
        ksum12 += (uint32_t) v12x2;
        ksum12 += (uint32_t) v12x3;
        ksum12 += (uint32_t) v12x4;
        ksum12 += (uint32_t) v12x5;
        ksum12 += (uint32_t) v12x6;
        ksum12 += (uint32_t) v12x7;
        out[96] = v12x0;
        out[97] = v12x1;
        out[98] = v12x2;
        out[99] = v12x3;
        out[100] = v12x4;
        out[101] = v12x5;
        out[102] = v12x6;
        out[103] = v12x7;
        w12 += 8;
        const int8_t v13x0 = w13[0];
        const int8_t v13x1 = w13[1];
        const int8_t v13x2 = w13[2];
        const int8_t v13x3 = w13[3];
        const int8_t v13x4 = w13[4];
        const int8_t v13x5 = w13[5];
        const int8_t v13x6 = w13[6];
        const int8_t v13x7 = w13[7];
        ksum13 += (uint32_t) v13x0;
        ksum13 += (uint32_t) v13x1;
        ksum13 += (uint32_t) v13x2;
        ksum13 += (uint32_t) v13x3;
        ksum13 += (uint32_t) v13x4;
        ksum13 += (uint32_t) v13x5;
        ksum13 += (uint32_t) v13x6;
        ksum13 += (uint32_t) v13x7;
        out[104] = v13x0;
        out[105] = v13x1;
        out[106] = v13x2;
        out[107] = v13x3;
        out[108] = v13x4;
        out[109] = v13x5;
        out[110] = v13x6;
        out[111] = v13x7;
        w13 += 8;
        const int8_t v14x0 = w14[0];
        const int8_t v14x1 = w14[1];
        const int8_t v14x2 = w14[2];
        const int8_t v14x3 = w14[3];
        const int8_t v14x4 = w14[4];
        const int8_t v14x5 = w14[5];
        const int8_t v14x6 = w14[6];
        const int8_t v14x7 = w14[7];
        ksum14 += (uint32_t) v14x0;
        ksum14 += (uint32_t) v14x1;
        ksum14 += (uint32_t) v14x2;
        ksum14 += (uint32_t) v14x3;
        ksum14 += (uint32_t) v14x4;
        ksum14 += (uint32_t) v14x5;
        ksum14 += (uint32_t) v14x6;
        ksum14 += (uint32_t) v14x7;
        out[112] = v14x0;
        out[113] = v14x1;
        out[114] = v14x2;
        out[115] = v14x3;
        out[116] = v14x4;
        out[117] = v14x5;
        out[118] = v14x6;
        out[119] = v14x7;
        w14 += 8;
        const int8_t v15x0 = w15[0];
        const int8_t v15x1 = w15[1];
        const int8_t v15x2 = w15[2];
        const int8_t v15x3 = w15[3];
        const int8_t v15x4 = w15[4];
        const int8_t v15x5 = w15[5];
        const int8_t v15x6 = w15[6];
        const int8_t v15x7 = w15[7];
        ksum15 += (uint32_t) v15x0;
        ksum15 += (uint32_t) v15x1;
        ksum15 += (uint32_t) v15x2;
        ksum15 += (uint32_t) v15x3;
        ksum15 += (uint32_t) v15x4;
        ksum15 += (uint32_t) v15x5;
        ksum15 += (uint32_t) v15x6;
        ksum15 += (uint32_t) v15x7;
        out[120] = v15x0;
        out[121] = v15x1;
        out[122] = v15x2;
        out[123] = v15x3;
        out[124] = v15x4;
        out[125] = v15x5;
        out[126] = v15x6;
        out[127] = v15x7;
        w15 += 8;
        out += 128;
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
        const int8_t v8x0 = w8[0];
        ksum8 += (uint32_t) v8x0;
        out[64] = v8x0;
        if (1 < k) {
          const int8_t v8x1 = w8[1];
          ksum8 += (uint32_t) v8x1;
          out[65] = v8x1;
        }
        if (2 < k) {
          const int8_t v8x2 = w8[2];
          ksum8 += (uint32_t) v8x2;
          out[66] = v8x2;
        }
        if (3 < k) {
          const int8_t v8x3 = w8[3];
          ksum8 += (uint32_t) v8x3;
          out[67] = v8x3;
        }
        if (4 < k) {
          const int8_t v8x4 = w8[4];
          ksum8 += (uint32_t) v8x4;
          out[68] = v8x4;
        }
        if (5 < k) {
          const int8_t v8x5 = w8[5];
          ksum8 += (uint32_t) v8x5;
          out[69] = v8x5;
        }
        if (6 < k) {
          const int8_t v8x6 = w8[6];
          ksum8 += (uint32_t) v8x6;
          out[70] = v8x6;
        }
        if (7 < k) {
          const int8_t v8x7 = w8[7];
          ksum8 += (uint32_t) v8x7;
          out[71] = v8x7;
        }
        w8 += k;
        const int8_t v9x0 = w9[0];
        ksum9 += (uint32_t) v9x0;
        out[72] = v9x0;
        if (1 < k) {
          const int8_t v9x1 = w9[1];
          ksum9 += (uint32_t) v9x1;
          out[73] = v9x1;
        }
        if (2 < k) {
          const int8_t v9x2 = w9[2];
          ksum9 += (uint32_t) v9x2;
          out[74] = v9x2;
        }
        if (3 < k) {
          const int8_t v9x3 = w9[3];
          ksum9 += (uint32_t) v9x3;
          out[75] = v9x3;
        }
        if (4 < k) {
          const int8_t v9x4 = w9[4];
          ksum9 += (uint32_t) v9x4;
          out[76] = v9x4;
        }
        if (5 < k) {
          const int8_t v9x5 = w9[5];
          ksum9 += (uint32_t) v9x5;
          out[77] = v9x5;
        }
        if (6 < k) {
          const int8_t v9x6 = w9[6];
          ksum9 += (uint32_t) v9x6;
          out[78] = v9x6;
        }
        if (7 < k) {
          const int8_t v9x7 = w9[7];
          ksum9 += (uint32_t) v9x7;
          out[79] = v9x7;
        }
        w9 += k;
        const int8_t v10x0 = w10[0];
        ksum10 += (uint32_t) v10x0;
        out[80] = v10x0;
        if (1 < k) {
          const int8_t v10x1 = w10[1];
          ksum10 += (uint32_t) v10x1;
          out[81] = v10x1;
        }
        if (2 < k) {
          const int8_t v10x2 = w10[2];
          ksum10 += (uint32_t) v10x2;
          out[82] = v10x2;
        }
        if (3 < k) {
          const int8_t v10x3 = w10[3];
          ksum10 += (uint32_t) v10x3;
          out[83] = v10x3;
        }
        if (4 < k) {
          const int8_t v10x4 = w10[4];
          ksum10 += (uint32_t) v10x4;
          out[84] = v10x4;
        }
        if (5 < k) {
          const int8_t v10x5 = w10[5];
          ksum10 += (uint32_t) v10x5;
          out[85] = v10x5;
        }
        if (6 < k) {
          const int8_t v10x6 = w10[6];
          ksum10 += (uint32_t) v10x6;
          out[86] = v10x6;
        }
        if (7 < k) {
          const int8_t v10x7 = w10[7];
          ksum10 += (uint32_t) v10x7;
          out[87] = v10x7;
        }
        w10 += k;
        const int8_t v11x0 = w11[0];
        ksum11 += (uint32_t) v11x0;
        out[88] = v11x0;
        if (1 < k) {
          const int8_t v11x1 = w11[1];
          ksum11 += (uint32_t) v11x1;
          out[89] = v11x1;
        }
        if (2 < k) {
          const int8_t v11x2 = w11[2];
          ksum11 += (uint32_t) v11x2;
          out[90] = v11x2;
        }
        if (3 < k) {
          const int8_t v11x3 = w11[3];
          ksum11 += (uint32_t) v11x3;
          out[91] = v11x3;
        }
        if (4 < k) {
          const int8_t v11x4 = w11[4];
          ksum11 += (uint32_t) v11x4;
          out[92] = v11x4;
        }
        if (5 < k) {
          const int8_t v11x5 = w11[5];
          ksum11 += (uint32_t) v11x5;
          out[93] = v11x5;
        }
        if (6 < k) {
          const int8_t v11x6 = w11[6];
          ksum11 += (uint32_t) v11x6;
          out[94] = v11x6;
        }
        if (7 < k) {
          const int8_t v11x7 = w11[7];
          ksum11 += (uint32_t) v11x7;
          out[95] = v11x7;
        }
        w11 += k;
        const int8_t v12x0 = w12[0];
        ksum12 += (uint32_t) v12x0;
        out[96] = v12x0;
        if (1 < k) {
          const int8_t v12x1 = w12[1];
          ksum12 += (uint32_t) v12x1;
          out[97] = v12x1;
        }
        if (2 < k) {
          const int8_t v12x2 = w12[2];
          ksum12 += (uint32_t) v12x2;
          out[98] = v12x2;
        }
        if (3 < k) {
          const int8_t v12x3 = w12[3];
          ksum12 += (uint32_t) v12x3;
          out[99] = v12x3;
        }
        if (4 < k) {
          const int8_t v12x4 = w12[4];
          ksum12 += (uint32_t) v12x4;
          out[100] = v12x4;
        }
        if (5 < k) {
          const int8_t v12x5 = w12[5];
          ksum12 += (uint32_t) v12x5;
          out[101] = v12x5;
        }
        if (6 < k) {
          const int8_t v12x6 = w12[6];
          ksum12 += (uint32_t) v12x6;
          out[102] = v12x6;
        }
        if (7 < k) {
          const int8_t v12x7 = w12[7];
          ksum12 += (uint32_t) v12x7;
          out[103] = v12x7;
        }
        w12 += k;
        const int8_t v13x0 = w13[0];
        ksum13 += (uint32_t) v13x0;
        out[104] = v13x0;
        if (1 < k) {
          const int8_t v13x1 = w13[1];
          ksum13 += (uint32_t) v13x1;
          out[105] = v13x1;
        }
        if (2 < k) {
          const int8_t v13x2 = w13[2];
          ksum13 += (uint32_t) v13x2;
          out[106] = v13x2;
        }
        if (3 < k) {
          const int8_t v13x3 = w13[3];
          ksum13 += (uint32_t) v13x3;
          out[107] = v13x3;
        }
        if (4 < k) {
          const int8_t v13x4 = w13[4];
          ksum13 += (uint32_t) v13x4;
          out[108] = v13x4;
        }
        if (5 < k) {
          const int8_t v13x5 = w13[5];
          ksum13 += (uint32_t) v13x5;
          out[109] = v13x5;
        }
        if (6 < k) {
          const int8_t v13x6 = w13[6];
          ksum13 += (uint32_t) v13x6;
          out[110] = v13x6;
        }
        if (7 < k) {
          const int8_t v13x7 = w13[7];
          ksum13 += (uint32_t) v13x7;
          out[111] = v13x7;
        }
        w13 += k;
        const int8_t v14x0 = w14[0];
        ksum14 += (uint32_t) v14x0;
        out[112] = v14x0;
        if (1 < k) {
          const int8_t v14x1 = w14[1];
          ksum14 += (uint32_t) v14x1;
          out[113] = v14x1;
        }
        if (2 < k) {
          const int8_t v14x2 = w14[2];
          ksum14 += (uint32_t) v14x2;
          out[114] = v14x2;
        }
        if (3 < k) {
          const int8_t v14x3 = w14[3];
          ksum14 += (uint32_t) v14x3;
          out[115] = v14x3;
        }
        if (4 < k) {
          const int8_t v14x4 = w14[4];
          ksum14 += (uint32_t) v14x4;
          out[116] = v14x4;
        }
        if (5 < k) {
          const int8_t v14x5 = w14[5];
          ksum14 += (uint32_t) v14x5;
          out[117] = v14x5;
        }
        if (6 < k) {
          const int8_t v14x6 = w14[6];
          ksum14 += (uint32_t) v14x6;
          out[118] = v14x6;
        }
        if (7 < k) {
          const int8_t v14x7 = w14[7];
          ksum14 += (uint32_t) v14x7;
          out[119] = v14x7;
        }
        w14 += k;
        const int8_t v15x0 = w15[0];
        ksum15 += (uint32_t) v15x0;
        out[120] = v15x0;
        if (1 < k) {
          const int8_t v15x1 = w15[1];
          ksum15 += (uint32_t) v15x1;
          out[121] = v15x1;
        }
        if (2 < k) {
          const int8_t v15x2 = w15[2];
          ksum15 += (uint32_t) v15x2;
          out[122] = v15x2;
        }
        if (3 < k) {
          const int8_t v15x3 = w15[3];
          ksum15 += (uint32_t) v15x3;
          out[123] = v15x3;
        }
        if (4 < k) {
          const int8_t v15x4 = w15[4];
          ksum15 += (uint32_t) v15x4;
          out[124] = v15x4;
        }
        if (5 < k) {
          const int8_t v15x5 = w15[5];
          ksum15 += (uint32_t) v15x5;
          out[125] = v15x5;
        }
        if (6 < k) {
          const int8_t v15x6 = w15[6];
          ksum15 += (uint32_t) v15x6;
          out[126] = v15x6;
        }
        if (7 < k) {
          const int8_t v15x7 = w15[7];
          ksum15 += (uint32_t) v15x7;
          out[127] = v15x7;
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

      // KC main loop multiple of 16x8
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
        const int8_t v8x0 = w8[0];
        const int8_t v8x1 = w8[1];
        const int8_t v8x2 = w8[2];
        const int8_t v8x3 = w8[3];
        const int8_t v8x4 = w8[4];
        const int8_t v8x5 = w8[5];
        const int8_t v8x6 = w8[6];
        const int8_t v8x7 = w8[7];
        ksum8 += (uint32_t) v8x0;
        ksum8 += (uint32_t) v8x1;
        ksum8 += (uint32_t) v8x2;
        ksum8 += (uint32_t) v8x3;
        ksum8 += (uint32_t) v8x4;
        ksum8 += (uint32_t) v8x5;
        ksum8 += (uint32_t) v8x6;
        ksum8 += (uint32_t) v8x7;
        out[64] = v8x0;
        out[65] = v8x1;
        out[66] = v8x2;
        out[67] = v8x3;
        out[68] = v8x4;
        out[69] = v8x5;
        out[70] = v8x6;
        out[71] = v8x7;
        w8 += 8;
        const int8_t v9x0 = w9[0];
        const int8_t v9x1 = w9[1];
        const int8_t v9x2 = w9[2];
        const int8_t v9x3 = w9[3];
        const int8_t v9x4 = w9[4];
        const int8_t v9x5 = w9[5];
        const int8_t v9x6 = w9[6];
        const int8_t v9x7 = w9[7];
        ksum9 += (uint32_t) v9x0;
        ksum9 += (uint32_t) v9x1;
        ksum9 += (uint32_t) v9x2;
        ksum9 += (uint32_t) v9x3;
        ksum9 += (uint32_t) v9x4;
        ksum9 += (uint32_t) v9x5;
        ksum9 += (uint32_t) v9x6;
        ksum9 += (uint32_t) v9x7;
        out[72] = v9x0;
        out[73] = v9x1;
        out[74] = v9x2;
        out[75] = v9x3;
        out[76] = v9x4;
        out[77] = v9x5;
        out[78] = v9x6;
        out[79] = v9x7;
        w9 += 8;
        const int8_t v10x0 = w10[0];
        const int8_t v10x1 = w10[1];
        const int8_t v10x2 = w10[2];
        const int8_t v10x3 = w10[3];
        const int8_t v10x4 = w10[4];
        const int8_t v10x5 = w10[5];
        const int8_t v10x6 = w10[6];
        const int8_t v10x7 = w10[7];
        ksum10 += (uint32_t) v10x0;
        ksum10 += (uint32_t) v10x1;
        ksum10 += (uint32_t) v10x2;
        ksum10 += (uint32_t) v10x3;
        ksum10 += (uint32_t) v10x4;
        ksum10 += (uint32_t) v10x5;
        ksum10 += (uint32_t) v10x6;
        ksum10 += (uint32_t) v10x7;
        out[80] = v10x0;
        out[81] = v10x1;
        out[82] = v10x2;
        out[83] = v10x3;
        out[84] = v10x4;
        out[85] = v10x5;
        out[86] = v10x6;
        out[87] = v10x7;
        w10 += 8;
        const int8_t v11x0 = w11[0];
        const int8_t v11x1 = w11[1];
        const int8_t v11x2 = w11[2];
        const int8_t v11x3 = w11[3];
        const int8_t v11x4 = w11[4];
        const int8_t v11x5 = w11[5];
        const int8_t v11x6 = w11[6];
        const int8_t v11x7 = w11[7];
        ksum11 += (uint32_t) v11x0;
        ksum11 += (uint32_t) v11x1;
        ksum11 += (uint32_t) v11x2;
        ksum11 += (uint32_t) v11x3;
        ksum11 += (uint32_t) v11x4;
        ksum11 += (uint32_t) v11x5;
        ksum11 += (uint32_t) v11x6;
        ksum11 += (uint32_t) v11x7;
        out[88] = v11x0;
        out[89] = v11x1;
        out[90] = v11x2;
        out[91] = v11x3;
        out[92] = v11x4;
        out[93] = v11x5;
        out[94] = v11x6;
        out[95] = v11x7;
        w11 += 8;
        const int8_t v12x0 = w12[0];
        const int8_t v12x1 = w12[1];
        const int8_t v12x2 = w12[2];
        const int8_t v12x3 = w12[3];
        const int8_t v12x4 = w12[4];
        const int8_t v12x5 = w12[5];
        const int8_t v12x6 = w12[6];
        const int8_t v12x7 = w12[7];
        ksum12 += (uint32_t) v12x0;
        ksum12 += (uint32_t) v12x1;
        ksum12 += (uint32_t) v12x2;
        ksum12 += (uint32_t) v12x3;
        ksum12 += (uint32_t) v12x4;
        ksum12 += (uint32_t) v12x5;
        ksum12 += (uint32_t) v12x6;
        ksum12 += (uint32_t) v12x7;
        out[96] = v12x0;
        out[97] = v12x1;
        out[98] = v12x2;
        out[99] = v12x3;
        out[100] = v12x4;
        out[101] = v12x5;
        out[102] = v12x6;
        out[103] = v12x7;
        w12 += 8;
        const int8_t v13x0 = w13[0];
        const int8_t v13x1 = w13[1];
        const int8_t v13x2 = w13[2];
        const int8_t v13x3 = w13[3];
        const int8_t v13x4 = w13[4];
        const int8_t v13x5 = w13[5];
        const int8_t v13x6 = w13[6];
        const int8_t v13x7 = w13[7];
        ksum13 += (uint32_t) v13x0;
        ksum13 += (uint32_t) v13x1;
        ksum13 += (uint32_t) v13x2;
        ksum13 += (uint32_t) v13x3;
        ksum13 += (uint32_t) v13x4;
        ksum13 += (uint32_t) v13x5;
        ksum13 += (uint32_t) v13x6;
        ksum13 += (uint32_t) v13x7;
        out[104] = v13x0;
        out[105] = v13x1;
        out[106] = v13x2;
        out[107] = v13x3;
        out[108] = v13x4;
        out[109] = v13x5;
        out[110] = v13x6;
        out[111] = v13x7;
        w13 += 8;
        const int8_t v14x0 = w14[0];
        const int8_t v14x1 = w14[1];
        const int8_t v14x2 = w14[2];
        const int8_t v14x3 = w14[3];
        const int8_t v14x4 = w14[4];
        const int8_t v14x5 = w14[5];
        const int8_t v14x6 = w14[6];
        const int8_t v14x7 = w14[7];
        ksum14 += (uint32_t) v14x0;
        ksum14 += (uint32_t) v14x1;
        ksum14 += (uint32_t) v14x2;
        ksum14 += (uint32_t) v14x3;
        ksum14 += (uint32_t) v14x4;
        ksum14 += (uint32_t) v14x5;
        ksum14 += (uint32_t) v14x6;
        ksum14 += (uint32_t) v14x7;
        out[112] = v14x0;
        out[113] = v14x1;
        out[114] = v14x2;
        out[115] = v14x3;
        out[116] = v14x4;
        out[117] = v14x5;
        out[118] = v14x6;
        out[119] = v14x7;
        w14 += 8;
        out += 128;
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
        const int8_t v8x0 = w8[0];
        ksum8 += (uint32_t) v8x0;
        out[64] = v8x0;
        if (1 < k) {
          const int8_t v8x1 = w8[1];
          ksum8 += (uint32_t) v8x1;
          out[65] = v8x1;
        }
        if (2 < k) {
          const int8_t v8x2 = w8[2];
          ksum8 += (uint32_t) v8x2;
          out[66] = v8x2;
        }
        if (3 < k) {
          const int8_t v8x3 = w8[3];
          ksum8 += (uint32_t) v8x3;
          out[67] = v8x3;
        }
        if (4 < k) {
          const int8_t v8x4 = w8[4];
          ksum8 += (uint32_t) v8x4;
          out[68] = v8x4;
        }
        if (5 < k) {
          const int8_t v8x5 = w8[5];
          ksum8 += (uint32_t) v8x5;
          out[69] = v8x5;
        }
        if (6 < k) {
          const int8_t v8x6 = w8[6];
          ksum8 += (uint32_t) v8x6;
          out[70] = v8x6;
        }
        if (7 < k) {
          const int8_t v8x7 = w8[7];
          ksum8 += (uint32_t) v8x7;
          out[71] = v8x7;
        }
        w8 += k;
        const int8_t v9x0 = w9[0];
        ksum9 += (uint32_t) v9x0;
        out[72] = v9x0;
        if (1 < k) {
          const int8_t v9x1 = w9[1];
          ksum9 += (uint32_t) v9x1;
          out[73] = v9x1;
        }
        if (2 < k) {
          const int8_t v9x2 = w9[2];
          ksum9 += (uint32_t) v9x2;
          out[74] = v9x2;
        }
        if (3 < k) {
          const int8_t v9x3 = w9[3];
          ksum9 += (uint32_t) v9x3;
          out[75] = v9x3;
        }
        if (4 < k) {
          const int8_t v9x4 = w9[4];
          ksum9 += (uint32_t) v9x4;
          out[76] = v9x4;
        }
        if (5 < k) {
          const int8_t v9x5 = w9[5];
          ksum9 += (uint32_t) v9x5;
          out[77] = v9x5;
        }
        if (6 < k) {
          const int8_t v9x6 = w9[6];
          ksum9 += (uint32_t) v9x6;
          out[78] = v9x6;
        }
        if (7 < k) {
          const int8_t v9x7 = w9[7];
          ksum9 += (uint32_t) v9x7;
          out[79] = v9x7;
        }
        w9 += k;
        const int8_t v10x0 = w10[0];
        ksum10 += (uint32_t) v10x0;
        out[80] = v10x0;
        if (1 < k) {
          const int8_t v10x1 = w10[1];
          ksum10 += (uint32_t) v10x1;
          out[81] = v10x1;
        }
        if (2 < k) {
          const int8_t v10x2 = w10[2];
          ksum10 += (uint32_t) v10x2;
          out[82] = v10x2;
        }
        if (3 < k) {
          const int8_t v10x3 = w10[3];
          ksum10 += (uint32_t) v10x3;
          out[83] = v10x3;
        }
        if (4 < k) {
          const int8_t v10x4 = w10[4];
          ksum10 += (uint32_t) v10x4;
          out[84] = v10x4;
        }
        if (5 < k) {
          const int8_t v10x5 = w10[5];
          ksum10 += (uint32_t) v10x5;
          out[85] = v10x5;
        }
        if (6 < k) {
          const int8_t v10x6 = w10[6];
          ksum10 += (uint32_t) v10x6;
          out[86] = v10x6;
        }
        if (7 < k) {
          const int8_t v10x7 = w10[7];
          ksum10 += (uint32_t) v10x7;
          out[87] = v10x7;
        }
        w10 += k;
        const int8_t v11x0 = w11[0];
        ksum11 += (uint32_t) v11x0;
        out[88] = v11x0;
        if (1 < k) {
          const int8_t v11x1 = w11[1];
          ksum11 += (uint32_t) v11x1;
          out[89] = v11x1;
        }
        if (2 < k) {
          const int8_t v11x2 = w11[2];
          ksum11 += (uint32_t) v11x2;
          out[90] = v11x2;
        }
        if (3 < k) {
          const int8_t v11x3 = w11[3];
          ksum11 += (uint32_t) v11x3;
          out[91] = v11x3;
        }
        if (4 < k) {
          const int8_t v11x4 = w11[4];
          ksum11 += (uint32_t) v11x4;
          out[92] = v11x4;
        }
        if (5 < k) {
          const int8_t v11x5 = w11[5];
          ksum11 += (uint32_t) v11x5;
          out[93] = v11x5;
        }
        if (6 < k) {
          const int8_t v11x6 = w11[6];
          ksum11 += (uint32_t) v11x6;
          out[94] = v11x6;
        }
        if (7 < k) {
          const int8_t v11x7 = w11[7];
          ksum11 += (uint32_t) v11x7;
          out[95] = v11x7;
        }
        w11 += k;
        const int8_t v12x0 = w12[0];
        ksum12 += (uint32_t) v12x0;
        out[96] = v12x0;
        if (1 < k) {
          const int8_t v12x1 = w12[1];
          ksum12 += (uint32_t) v12x1;
          out[97] = v12x1;
        }
        if (2 < k) {
          const int8_t v12x2 = w12[2];
          ksum12 += (uint32_t) v12x2;
          out[98] = v12x2;
        }
        if (3 < k) {
          const int8_t v12x3 = w12[3];
          ksum12 += (uint32_t) v12x3;
          out[99] = v12x3;
        }
        if (4 < k) {
          const int8_t v12x4 = w12[4];
          ksum12 += (uint32_t) v12x4;
          out[100] = v12x4;
        }
        if (5 < k) {
          const int8_t v12x5 = w12[5];
          ksum12 += (uint32_t) v12x5;
          out[101] = v12x5;
        }
        if (6 < k) {
          const int8_t v12x6 = w12[6];
          ksum12 += (uint32_t) v12x6;
          out[102] = v12x6;
        }
        if (7 < k) {
          const int8_t v12x7 = w12[7];
          ksum12 += (uint32_t) v12x7;
          out[103] = v12x7;
        }
        w12 += k;
        const int8_t v13x0 = w13[0];
        ksum13 += (uint32_t) v13x0;
        out[104] = v13x0;
        if (1 < k) {
          const int8_t v13x1 = w13[1];
          ksum13 += (uint32_t) v13x1;
          out[105] = v13x1;
        }
        if (2 < k) {
          const int8_t v13x2 = w13[2];
          ksum13 += (uint32_t) v13x2;
          out[106] = v13x2;
        }
        if (3 < k) {
          const int8_t v13x3 = w13[3];
          ksum13 += (uint32_t) v13x3;
          out[107] = v13x3;
        }
        if (4 < k) {
          const int8_t v13x4 = w13[4];
          ksum13 += (uint32_t) v13x4;
          out[108] = v13x4;
        }
        if (5 < k) {
          const int8_t v13x5 = w13[5];
          ksum13 += (uint32_t) v13x5;
          out[109] = v13x5;
        }
        if (6 < k) {
          const int8_t v13x6 = w13[6];
          ksum13 += (uint32_t) v13x6;
          out[110] = v13x6;
        }
        if (7 < k) {
          const int8_t v13x7 = w13[7];
          ksum13 += (uint32_t) v13x7;
          out[111] = v13x7;
        }
        w13 += k;
        const int8_t v14x0 = w14[0];
        ksum14 += (uint32_t) v14x0;
        out[112] = v14x0;
        if (1 < k) {
          const int8_t v14x1 = w14[1];
          ksum14 += (uint32_t) v14x1;
          out[113] = v14x1;
        }
        if (2 < k) {
          const int8_t v14x2 = w14[2];
          ksum14 += (uint32_t) v14x2;
          out[114] = v14x2;
        }
        if (3 < k) {
          const int8_t v14x3 = w14[3];
          ksum14 += (uint32_t) v14x3;
          out[115] = v14x3;
        }
        if (4 < k) {
          const int8_t v14x4 = w14[4];
          ksum14 += (uint32_t) v14x4;
          out[116] = v14x4;
        }
        if (5 < k) {
          const int8_t v14x5 = w14[5];
          ksum14 += (uint32_t) v14x5;
          out[117] = v14x5;
        }
        if (6 < k) {
          const int8_t v14x6 = w14[6];
          ksum14 += (uint32_t) v14x6;
          out[118] = v14x6;
        }
        if (7 < k) {
          const int8_t v14x7 = w14[7];
          ksum14 += (uint32_t) v14x7;
          out[119] = v14x7;
        }
        w14 += k;
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
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
