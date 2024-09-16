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

void xnn_qs8_packw_gemm_goi_ukernel_x64c4__scalar(
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
  assert(nr == 64);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = params ? (uint32_t) ((const struct xnn_qs8_packw_params*) params)->input_zero_point : 0;

  do {
    // NC main loop multiple of 64
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 64; n -= 64) {
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
        ((int32_t*) out)[16] = b[16];
        ((int32_t*) out)[17] = b[17];
        ((int32_t*) out)[18] = b[18];
        ((int32_t*) out)[19] = b[19];
        ((int32_t*) out)[20] = b[20];
        ((int32_t*) out)[21] = b[21];
        ((int32_t*) out)[22] = b[22];
        ((int32_t*) out)[23] = b[23];
        ((int32_t*) out)[24] = b[24];
        ((int32_t*) out)[25] = b[25];
        ((int32_t*) out)[26] = b[26];
        ((int32_t*) out)[27] = b[27];
        ((int32_t*) out)[28] = b[28];
        ((int32_t*) out)[29] = b[29];
        ((int32_t*) out)[30] = b[30];
        ((int32_t*) out)[31] = b[31];
        ((int32_t*) out)[32] = b[32];
        ((int32_t*) out)[33] = b[33];
        ((int32_t*) out)[34] = b[34];
        ((int32_t*) out)[35] = b[35];
        ((int32_t*) out)[36] = b[36];
        ((int32_t*) out)[37] = b[37];
        ((int32_t*) out)[38] = b[38];
        ((int32_t*) out)[39] = b[39];
        ((int32_t*) out)[40] = b[40];
        ((int32_t*) out)[41] = b[41];
        ((int32_t*) out)[42] = b[42];
        ((int32_t*) out)[43] = b[43];
        ((int32_t*) out)[44] = b[44];
        ((int32_t*) out)[45] = b[45];
        ((int32_t*) out)[46] = b[46];
        ((int32_t*) out)[47] = b[47];
        ((int32_t*) out)[48] = b[48];
        ((int32_t*) out)[49] = b[49];
        ((int32_t*) out)[50] = b[50];
        ((int32_t*) out)[51] = b[51];
        ((int32_t*) out)[52] = b[52];
        ((int32_t*) out)[53] = b[53];
        ((int32_t*) out)[54] = b[54];
        ((int32_t*) out)[55] = b[55];
        ((int32_t*) out)[56] = b[56];
        ((int32_t*) out)[57] = b[57];
        ((int32_t*) out)[58] = b[58];
        ((int32_t*) out)[59] = b[59];
        ((int32_t*) out)[60] = b[60];
        ((int32_t*) out)[61] = b[61];
        ((int32_t*) out)[62] = b[62];
        ((int32_t*) out)[63] = b[63];
        b += 64;
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
        ((int32_t*) out)[16] = 0;
        ((int32_t*) out)[17] = 0;
        ((int32_t*) out)[18] = 0;
        ((int32_t*) out)[19] = 0;
        ((int32_t*) out)[20] = 0;
        ((int32_t*) out)[21] = 0;
        ((int32_t*) out)[22] = 0;
        ((int32_t*) out)[23] = 0;
        ((int32_t*) out)[24] = 0;
        ((int32_t*) out)[25] = 0;
        ((int32_t*) out)[26] = 0;
        ((int32_t*) out)[27] = 0;
        ((int32_t*) out)[28] = 0;
        ((int32_t*) out)[29] = 0;
        ((int32_t*) out)[30] = 0;
        ((int32_t*) out)[31] = 0;
        ((int32_t*) out)[32] = 0;
        ((int32_t*) out)[33] = 0;
        ((int32_t*) out)[34] = 0;
        ((int32_t*) out)[35] = 0;
        ((int32_t*) out)[36] = 0;
        ((int32_t*) out)[37] = 0;
        ((int32_t*) out)[38] = 0;
        ((int32_t*) out)[39] = 0;
        ((int32_t*) out)[40] = 0;
        ((int32_t*) out)[41] = 0;
        ((int32_t*) out)[42] = 0;
        ((int32_t*) out)[43] = 0;
        ((int32_t*) out)[44] = 0;
        ((int32_t*) out)[45] = 0;
        ((int32_t*) out)[46] = 0;
        ((int32_t*) out)[47] = 0;
        ((int32_t*) out)[48] = 0;
        ((int32_t*) out)[49] = 0;
        ((int32_t*) out)[50] = 0;
        ((int32_t*) out)[51] = 0;
        ((int32_t*) out)[52] = 0;
        ((int32_t*) out)[53] = 0;
        ((int32_t*) out)[54] = 0;
        ((int32_t*) out)[55] = 0;
        ((int32_t*) out)[56] = 0;
        ((int32_t*) out)[57] = 0;
        ((int32_t*) out)[58] = 0;
        ((int32_t*) out)[59] = 0;
        ((int32_t*) out)[60] = 0;
        ((int32_t*) out)[61] = 0;
        ((int32_t*) out)[62] = 0;
        ((int32_t*) out)[63] = 0;
      }
      out += 64 * sizeof(int32_t);

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
      const int8_t* w16 = w15 + kc;
      const int8_t* w17 = w16 + kc;
      const int8_t* w18 = w17 + kc;
      const int8_t* w19 = w18 + kc;
      const int8_t* w20 = w19 + kc;
      const int8_t* w21 = w20 + kc;
      const int8_t* w22 = w21 + kc;
      const int8_t* w23 = w22 + kc;
      const int8_t* w24 = w23 + kc;
      const int8_t* w25 = w24 + kc;
      const int8_t* w26 = w25 + kc;
      const int8_t* w27 = w26 + kc;
      const int8_t* w28 = w27 + kc;
      const int8_t* w29 = w28 + kc;
      const int8_t* w30 = w29 + kc;
      const int8_t* w31 = w30 + kc;
      const int8_t* w32 = w31 + kc;
      const int8_t* w33 = w32 + kc;
      const int8_t* w34 = w33 + kc;
      const int8_t* w35 = w34 + kc;
      const int8_t* w36 = w35 + kc;
      const int8_t* w37 = w36 + kc;
      const int8_t* w38 = w37 + kc;
      const int8_t* w39 = w38 + kc;
      const int8_t* w40 = w39 + kc;
      const int8_t* w41 = w40 + kc;
      const int8_t* w42 = w41 + kc;
      const int8_t* w43 = w42 + kc;
      const int8_t* w44 = w43 + kc;
      const int8_t* w45 = w44 + kc;
      const int8_t* w46 = w45 + kc;
      const int8_t* w47 = w46 + kc;
      const int8_t* w48 = w47 + kc;
      const int8_t* w49 = w48 + kc;
      const int8_t* w50 = w49 + kc;
      const int8_t* w51 = w50 + kc;
      const int8_t* w52 = w51 + kc;
      const int8_t* w53 = w52 + kc;
      const int8_t* w54 = w53 + kc;
      const int8_t* w55 = w54 + kc;
      const int8_t* w56 = w55 + kc;
      const int8_t* w57 = w56 + kc;
      const int8_t* w58 = w57 + kc;
      const int8_t* w59 = w58 + kc;
      const int8_t* w60 = w59 + kc;
      const int8_t* w61 = w60 + kc;
      const int8_t* w62 = w61 + kc;
      const int8_t* w63 = w62 + kc;
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
      uint32_t ksum16 = 0;
      uint32_t ksum17 = 0;
      uint32_t ksum18 = 0;
      uint32_t ksum19 = 0;
      uint32_t ksum20 = 0;
      uint32_t ksum21 = 0;
      uint32_t ksum22 = 0;
      uint32_t ksum23 = 0;
      uint32_t ksum24 = 0;
      uint32_t ksum25 = 0;
      uint32_t ksum26 = 0;
      uint32_t ksum27 = 0;
      uint32_t ksum28 = 0;
      uint32_t ksum29 = 0;
      uint32_t ksum30 = 0;
      uint32_t ksum31 = 0;
      uint32_t ksum32 = 0;
      uint32_t ksum33 = 0;
      uint32_t ksum34 = 0;
      uint32_t ksum35 = 0;
      uint32_t ksum36 = 0;
      uint32_t ksum37 = 0;
      uint32_t ksum38 = 0;
      uint32_t ksum39 = 0;
      uint32_t ksum40 = 0;
      uint32_t ksum41 = 0;
      uint32_t ksum42 = 0;
      uint32_t ksum43 = 0;
      uint32_t ksum44 = 0;
      uint32_t ksum45 = 0;
      uint32_t ksum46 = 0;
      uint32_t ksum47 = 0;
      uint32_t ksum48 = 0;
      uint32_t ksum49 = 0;
      uint32_t ksum50 = 0;
      uint32_t ksum51 = 0;
      uint32_t ksum52 = 0;
      uint32_t ksum53 = 0;
      uint32_t ksum54 = 0;
      uint32_t ksum55 = 0;
      uint32_t ksum56 = 0;
      uint32_t ksum57 = 0;
      uint32_t ksum58 = 0;
      uint32_t ksum59 = 0;
      uint32_t ksum60 = 0;
      uint32_t ksum61 = 0;
      uint32_t ksum62 = 0;
      uint32_t ksum63 = 0;

      // KC main loop multiple of 64x4
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
        const int8_t v16x0 = w16[0];
        const int8_t v16x1 = w16[1];
        const int8_t v16x2 = w16[2];
        const int8_t v16x3 = w16[3];
        ksum16 += (uint32_t) v16x0;
        ksum16 += (uint32_t) v16x1;
        ksum16 += (uint32_t) v16x2;
        ksum16 += (uint32_t) v16x3;
        out[64] = v16x0;
        out[65] = v16x1;
        out[66] = v16x2;
        out[67] = v16x3;
        w16 += 4;
        const int8_t v17x0 = w17[0];
        const int8_t v17x1 = w17[1];
        const int8_t v17x2 = w17[2];
        const int8_t v17x3 = w17[3];
        ksum17 += (uint32_t) v17x0;
        ksum17 += (uint32_t) v17x1;
        ksum17 += (uint32_t) v17x2;
        ksum17 += (uint32_t) v17x3;
        out[68] = v17x0;
        out[69] = v17x1;
        out[70] = v17x2;
        out[71] = v17x3;
        w17 += 4;
        const int8_t v18x0 = w18[0];
        const int8_t v18x1 = w18[1];
        const int8_t v18x2 = w18[2];
        const int8_t v18x3 = w18[3];
        ksum18 += (uint32_t) v18x0;
        ksum18 += (uint32_t) v18x1;
        ksum18 += (uint32_t) v18x2;
        ksum18 += (uint32_t) v18x3;
        out[72] = v18x0;
        out[73] = v18x1;
        out[74] = v18x2;
        out[75] = v18x3;
        w18 += 4;
        const int8_t v19x0 = w19[0];
        const int8_t v19x1 = w19[1];
        const int8_t v19x2 = w19[2];
        const int8_t v19x3 = w19[3];
        ksum19 += (uint32_t) v19x0;
        ksum19 += (uint32_t) v19x1;
        ksum19 += (uint32_t) v19x2;
        ksum19 += (uint32_t) v19x3;
        out[76] = v19x0;
        out[77] = v19x1;
        out[78] = v19x2;
        out[79] = v19x3;
        w19 += 4;
        const int8_t v20x0 = w20[0];
        const int8_t v20x1 = w20[1];
        const int8_t v20x2 = w20[2];
        const int8_t v20x3 = w20[3];
        ksum20 += (uint32_t) v20x0;
        ksum20 += (uint32_t) v20x1;
        ksum20 += (uint32_t) v20x2;
        ksum20 += (uint32_t) v20x3;
        out[80] = v20x0;
        out[81] = v20x1;
        out[82] = v20x2;
        out[83] = v20x3;
        w20 += 4;
        const int8_t v21x0 = w21[0];
        const int8_t v21x1 = w21[1];
        const int8_t v21x2 = w21[2];
        const int8_t v21x3 = w21[3];
        ksum21 += (uint32_t) v21x0;
        ksum21 += (uint32_t) v21x1;
        ksum21 += (uint32_t) v21x2;
        ksum21 += (uint32_t) v21x3;
        out[84] = v21x0;
        out[85] = v21x1;
        out[86] = v21x2;
        out[87] = v21x3;
        w21 += 4;
        const int8_t v22x0 = w22[0];
        const int8_t v22x1 = w22[1];
        const int8_t v22x2 = w22[2];
        const int8_t v22x3 = w22[3];
        ksum22 += (uint32_t) v22x0;
        ksum22 += (uint32_t) v22x1;
        ksum22 += (uint32_t) v22x2;
        ksum22 += (uint32_t) v22x3;
        out[88] = v22x0;
        out[89] = v22x1;
        out[90] = v22x2;
        out[91] = v22x3;
        w22 += 4;
        const int8_t v23x0 = w23[0];
        const int8_t v23x1 = w23[1];
        const int8_t v23x2 = w23[2];
        const int8_t v23x3 = w23[3];
        ksum23 += (uint32_t) v23x0;
        ksum23 += (uint32_t) v23x1;
        ksum23 += (uint32_t) v23x2;
        ksum23 += (uint32_t) v23x3;
        out[92] = v23x0;
        out[93] = v23x1;
        out[94] = v23x2;
        out[95] = v23x3;
        w23 += 4;
        const int8_t v24x0 = w24[0];
        const int8_t v24x1 = w24[1];
        const int8_t v24x2 = w24[2];
        const int8_t v24x3 = w24[3];
        ksum24 += (uint32_t) v24x0;
        ksum24 += (uint32_t) v24x1;
        ksum24 += (uint32_t) v24x2;
        ksum24 += (uint32_t) v24x3;
        out[96] = v24x0;
        out[97] = v24x1;
        out[98] = v24x2;
        out[99] = v24x3;
        w24 += 4;
        const int8_t v25x0 = w25[0];
        const int8_t v25x1 = w25[1];
        const int8_t v25x2 = w25[2];
        const int8_t v25x3 = w25[3];
        ksum25 += (uint32_t) v25x0;
        ksum25 += (uint32_t) v25x1;
        ksum25 += (uint32_t) v25x2;
        ksum25 += (uint32_t) v25x3;
        out[100] = v25x0;
        out[101] = v25x1;
        out[102] = v25x2;
        out[103] = v25x3;
        w25 += 4;
        const int8_t v26x0 = w26[0];
        const int8_t v26x1 = w26[1];
        const int8_t v26x2 = w26[2];
        const int8_t v26x3 = w26[3];
        ksum26 += (uint32_t) v26x0;
        ksum26 += (uint32_t) v26x1;
        ksum26 += (uint32_t) v26x2;
        ksum26 += (uint32_t) v26x3;
        out[104] = v26x0;
        out[105] = v26x1;
        out[106] = v26x2;
        out[107] = v26x3;
        w26 += 4;
        const int8_t v27x0 = w27[0];
        const int8_t v27x1 = w27[1];
        const int8_t v27x2 = w27[2];
        const int8_t v27x3 = w27[3];
        ksum27 += (uint32_t) v27x0;
        ksum27 += (uint32_t) v27x1;
        ksum27 += (uint32_t) v27x2;
        ksum27 += (uint32_t) v27x3;
        out[108] = v27x0;
        out[109] = v27x1;
        out[110] = v27x2;
        out[111] = v27x3;
        w27 += 4;
        const int8_t v28x0 = w28[0];
        const int8_t v28x1 = w28[1];
        const int8_t v28x2 = w28[2];
        const int8_t v28x3 = w28[3];
        ksum28 += (uint32_t) v28x0;
        ksum28 += (uint32_t) v28x1;
        ksum28 += (uint32_t) v28x2;
        ksum28 += (uint32_t) v28x3;
        out[112] = v28x0;
        out[113] = v28x1;
        out[114] = v28x2;
        out[115] = v28x3;
        w28 += 4;
        const int8_t v29x0 = w29[0];
        const int8_t v29x1 = w29[1];
        const int8_t v29x2 = w29[2];
        const int8_t v29x3 = w29[3];
        ksum29 += (uint32_t) v29x0;
        ksum29 += (uint32_t) v29x1;
        ksum29 += (uint32_t) v29x2;
        ksum29 += (uint32_t) v29x3;
        out[116] = v29x0;
        out[117] = v29x1;
        out[118] = v29x2;
        out[119] = v29x3;
        w29 += 4;
        const int8_t v30x0 = w30[0];
        const int8_t v30x1 = w30[1];
        const int8_t v30x2 = w30[2];
        const int8_t v30x3 = w30[3];
        ksum30 += (uint32_t) v30x0;
        ksum30 += (uint32_t) v30x1;
        ksum30 += (uint32_t) v30x2;
        ksum30 += (uint32_t) v30x3;
        out[120] = v30x0;
        out[121] = v30x1;
        out[122] = v30x2;
        out[123] = v30x3;
        w30 += 4;
        const int8_t v31x0 = w31[0];
        const int8_t v31x1 = w31[1];
        const int8_t v31x2 = w31[2];
        const int8_t v31x3 = w31[3];
        ksum31 += (uint32_t) v31x0;
        ksum31 += (uint32_t) v31x1;
        ksum31 += (uint32_t) v31x2;
        ksum31 += (uint32_t) v31x3;
        out[124] = v31x0;
        out[125] = v31x1;
        out[126] = v31x2;
        out[127] = v31x3;
        w31 += 4;
        const int8_t v32x0 = w32[0];
        const int8_t v32x1 = w32[1];
        const int8_t v32x2 = w32[2];
        const int8_t v32x3 = w32[3];
        ksum32 += (uint32_t) v32x0;
        ksum32 += (uint32_t) v32x1;
        ksum32 += (uint32_t) v32x2;
        ksum32 += (uint32_t) v32x3;
        out[128] = v32x0;
        out[129] = v32x1;
        out[130] = v32x2;
        out[131] = v32x3;
        w32 += 4;
        const int8_t v33x0 = w33[0];
        const int8_t v33x1 = w33[1];
        const int8_t v33x2 = w33[2];
        const int8_t v33x3 = w33[3];
        ksum33 += (uint32_t) v33x0;
        ksum33 += (uint32_t) v33x1;
        ksum33 += (uint32_t) v33x2;
        ksum33 += (uint32_t) v33x3;
        out[132] = v33x0;
        out[133] = v33x1;
        out[134] = v33x2;
        out[135] = v33x3;
        w33 += 4;
        const int8_t v34x0 = w34[0];
        const int8_t v34x1 = w34[1];
        const int8_t v34x2 = w34[2];
        const int8_t v34x3 = w34[3];
        ksum34 += (uint32_t) v34x0;
        ksum34 += (uint32_t) v34x1;
        ksum34 += (uint32_t) v34x2;
        ksum34 += (uint32_t) v34x3;
        out[136] = v34x0;
        out[137] = v34x1;
        out[138] = v34x2;
        out[139] = v34x3;
        w34 += 4;
        const int8_t v35x0 = w35[0];
        const int8_t v35x1 = w35[1];
        const int8_t v35x2 = w35[2];
        const int8_t v35x3 = w35[3];
        ksum35 += (uint32_t) v35x0;
        ksum35 += (uint32_t) v35x1;
        ksum35 += (uint32_t) v35x2;
        ksum35 += (uint32_t) v35x3;
        out[140] = v35x0;
        out[141] = v35x1;
        out[142] = v35x2;
        out[143] = v35x3;
        w35 += 4;
        const int8_t v36x0 = w36[0];
        const int8_t v36x1 = w36[1];
        const int8_t v36x2 = w36[2];
        const int8_t v36x3 = w36[3];
        ksum36 += (uint32_t) v36x0;
        ksum36 += (uint32_t) v36x1;
        ksum36 += (uint32_t) v36x2;
        ksum36 += (uint32_t) v36x3;
        out[144] = v36x0;
        out[145] = v36x1;
        out[146] = v36x2;
        out[147] = v36x3;
        w36 += 4;
        const int8_t v37x0 = w37[0];
        const int8_t v37x1 = w37[1];
        const int8_t v37x2 = w37[2];
        const int8_t v37x3 = w37[3];
        ksum37 += (uint32_t) v37x0;
        ksum37 += (uint32_t) v37x1;
        ksum37 += (uint32_t) v37x2;
        ksum37 += (uint32_t) v37x3;
        out[148] = v37x0;
        out[149] = v37x1;
        out[150] = v37x2;
        out[151] = v37x3;
        w37 += 4;
        const int8_t v38x0 = w38[0];
        const int8_t v38x1 = w38[1];
        const int8_t v38x2 = w38[2];
        const int8_t v38x3 = w38[3];
        ksum38 += (uint32_t) v38x0;
        ksum38 += (uint32_t) v38x1;
        ksum38 += (uint32_t) v38x2;
        ksum38 += (uint32_t) v38x3;
        out[152] = v38x0;
        out[153] = v38x1;
        out[154] = v38x2;
        out[155] = v38x3;
        w38 += 4;
        const int8_t v39x0 = w39[0];
        const int8_t v39x1 = w39[1];
        const int8_t v39x2 = w39[2];
        const int8_t v39x3 = w39[3];
        ksum39 += (uint32_t) v39x0;
        ksum39 += (uint32_t) v39x1;
        ksum39 += (uint32_t) v39x2;
        ksum39 += (uint32_t) v39x3;
        out[156] = v39x0;
        out[157] = v39x1;
        out[158] = v39x2;
        out[159] = v39x3;
        w39 += 4;
        const int8_t v40x0 = w40[0];
        const int8_t v40x1 = w40[1];
        const int8_t v40x2 = w40[2];
        const int8_t v40x3 = w40[3];
        ksum40 += (uint32_t) v40x0;
        ksum40 += (uint32_t) v40x1;
        ksum40 += (uint32_t) v40x2;
        ksum40 += (uint32_t) v40x3;
        out[160] = v40x0;
        out[161] = v40x1;
        out[162] = v40x2;
        out[163] = v40x3;
        w40 += 4;
        const int8_t v41x0 = w41[0];
        const int8_t v41x1 = w41[1];
        const int8_t v41x2 = w41[2];
        const int8_t v41x3 = w41[3];
        ksum41 += (uint32_t) v41x0;
        ksum41 += (uint32_t) v41x1;
        ksum41 += (uint32_t) v41x2;
        ksum41 += (uint32_t) v41x3;
        out[164] = v41x0;
        out[165] = v41x1;
        out[166] = v41x2;
        out[167] = v41x3;
        w41 += 4;
        const int8_t v42x0 = w42[0];
        const int8_t v42x1 = w42[1];
        const int8_t v42x2 = w42[2];
        const int8_t v42x3 = w42[3];
        ksum42 += (uint32_t) v42x0;
        ksum42 += (uint32_t) v42x1;
        ksum42 += (uint32_t) v42x2;
        ksum42 += (uint32_t) v42x3;
        out[168] = v42x0;
        out[169] = v42x1;
        out[170] = v42x2;
        out[171] = v42x3;
        w42 += 4;
        const int8_t v43x0 = w43[0];
        const int8_t v43x1 = w43[1];
        const int8_t v43x2 = w43[2];
        const int8_t v43x3 = w43[3];
        ksum43 += (uint32_t) v43x0;
        ksum43 += (uint32_t) v43x1;
        ksum43 += (uint32_t) v43x2;
        ksum43 += (uint32_t) v43x3;
        out[172] = v43x0;
        out[173] = v43x1;
        out[174] = v43x2;
        out[175] = v43x3;
        w43 += 4;
        const int8_t v44x0 = w44[0];
        const int8_t v44x1 = w44[1];
        const int8_t v44x2 = w44[2];
        const int8_t v44x3 = w44[3];
        ksum44 += (uint32_t) v44x0;
        ksum44 += (uint32_t) v44x1;
        ksum44 += (uint32_t) v44x2;
        ksum44 += (uint32_t) v44x3;
        out[176] = v44x0;
        out[177] = v44x1;
        out[178] = v44x2;
        out[179] = v44x3;
        w44 += 4;
        const int8_t v45x0 = w45[0];
        const int8_t v45x1 = w45[1];
        const int8_t v45x2 = w45[2];
        const int8_t v45x3 = w45[3];
        ksum45 += (uint32_t) v45x0;
        ksum45 += (uint32_t) v45x1;
        ksum45 += (uint32_t) v45x2;
        ksum45 += (uint32_t) v45x3;
        out[180] = v45x0;
        out[181] = v45x1;
        out[182] = v45x2;
        out[183] = v45x3;
        w45 += 4;
        const int8_t v46x0 = w46[0];
        const int8_t v46x1 = w46[1];
        const int8_t v46x2 = w46[2];
        const int8_t v46x3 = w46[3];
        ksum46 += (uint32_t) v46x0;
        ksum46 += (uint32_t) v46x1;
        ksum46 += (uint32_t) v46x2;
        ksum46 += (uint32_t) v46x3;
        out[184] = v46x0;
        out[185] = v46x1;
        out[186] = v46x2;
        out[187] = v46x3;
        w46 += 4;
        const int8_t v47x0 = w47[0];
        const int8_t v47x1 = w47[1];
        const int8_t v47x2 = w47[2];
        const int8_t v47x3 = w47[3];
        ksum47 += (uint32_t) v47x0;
        ksum47 += (uint32_t) v47x1;
        ksum47 += (uint32_t) v47x2;
        ksum47 += (uint32_t) v47x3;
        out[188] = v47x0;
        out[189] = v47x1;
        out[190] = v47x2;
        out[191] = v47x3;
        w47 += 4;
        const int8_t v48x0 = w48[0];
        const int8_t v48x1 = w48[1];
        const int8_t v48x2 = w48[2];
        const int8_t v48x3 = w48[3];
        ksum48 += (uint32_t) v48x0;
        ksum48 += (uint32_t) v48x1;
        ksum48 += (uint32_t) v48x2;
        ksum48 += (uint32_t) v48x3;
        out[192] = v48x0;
        out[193] = v48x1;
        out[194] = v48x2;
        out[195] = v48x3;
        w48 += 4;
        const int8_t v49x0 = w49[0];
        const int8_t v49x1 = w49[1];
        const int8_t v49x2 = w49[2];
        const int8_t v49x3 = w49[3];
        ksum49 += (uint32_t) v49x0;
        ksum49 += (uint32_t) v49x1;
        ksum49 += (uint32_t) v49x2;
        ksum49 += (uint32_t) v49x3;
        out[196] = v49x0;
        out[197] = v49x1;
        out[198] = v49x2;
        out[199] = v49x3;
        w49 += 4;
        const int8_t v50x0 = w50[0];
        const int8_t v50x1 = w50[1];
        const int8_t v50x2 = w50[2];
        const int8_t v50x3 = w50[3];
        ksum50 += (uint32_t) v50x0;
        ksum50 += (uint32_t) v50x1;
        ksum50 += (uint32_t) v50x2;
        ksum50 += (uint32_t) v50x3;
        out[200] = v50x0;
        out[201] = v50x1;
        out[202] = v50x2;
        out[203] = v50x3;
        w50 += 4;
        const int8_t v51x0 = w51[0];
        const int8_t v51x1 = w51[1];
        const int8_t v51x2 = w51[2];
        const int8_t v51x3 = w51[3];
        ksum51 += (uint32_t) v51x0;
        ksum51 += (uint32_t) v51x1;
        ksum51 += (uint32_t) v51x2;
        ksum51 += (uint32_t) v51x3;
        out[204] = v51x0;
        out[205] = v51x1;
        out[206] = v51x2;
        out[207] = v51x3;
        w51 += 4;
        const int8_t v52x0 = w52[0];
        const int8_t v52x1 = w52[1];
        const int8_t v52x2 = w52[2];
        const int8_t v52x3 = w52[3];
        ksum52 += (uint32_t) v52x0;
        ksum52 += (uint32_t) v52x1;
        ksum52 += (uint32_t) v52x2;
        ksum52 += (uint32_t) v52x3;
        out[208] = v52x0;
        out[209] = v52x1;
        out[210] = v52x2;
        out[211] = v52x3;
        w52 += 4;
        const int8_t v53x0 = w53[0];
        const int8_t v53x1 = w53[1];
        const int8_t v53x2 = w53[2];
        const int8_t v53x3 = w53[3];
        ksum53 += (uint32_t) v53x0;
        ksum53 += (uint32_t) v53x1;
        ksum53 += (uint32_t) v53x2;
        ksum53 += (uint32_t) v53x3;
        out[212] = v53x0;
        out[213] = v53x1;
        out[214] = v53x2;
        out[215] = v53x3;
        w53 += 4;
        const int8_t v54x0 = w54[0];
        const int8_t v54x1 = w54[1];
        const int8_t v54x2 = w54[2];
        const int8_t v54x3 = w54[3];
        ksum54 += (uint32_t) v54x0;
        ksum54 += (uint32_t) v54x1;
        ksum54 += (uint32_t) v54x2;
        ksum54 += (uint32_t) v54x3;
        out[216] = v54x0;
        out[217] = v54x1;
        out[218] = v54x2;
        out[219] = v54x3;
        w54 += 4;
        const int8_t v55x0 = w55[0];
        const int8_t v55x1 = w55[1];
        const int8_t v55x2 = w55[2];
        const int8_t v55x3 = w55[3];
        ksum55 += (uint32_t) v55x0;
        ksum55 += (uint32_t) v55x1;
        ksum55 += (uint32_t) v55x2;
        ksum55 += (uint32_t) v55x3;
        out[220] = v55x0;
        out[221] = v55x1;
        out[222] = v55x2;
        out[223] = v55x3;
        w55 += 4;
        const int8_t v56x0 = w56[0];
        const int8_t v56x1 = w56[1];
        const int8_t v56x2 = w56[2];
        const int8_t v56x3 = w56[3];
        ksum56 += (uint32_t) v56x0;
        ksum56 += (uint32_t) v56x1;
        ksum56 += (uint32_t) v56x2;
        ksum56 += (uint32_t) v56x3;
        out[224] = v56x0;
        out[225] = v56x1;
        out[226] = v56x2;
        out[227] = v56x3;
        w56 += 4;
        const int8_t v57x0 = w57[0];
        const int8_t v57x1 = w57[1];
        const int8_t v57x2 = w57[2];
        const int8_t v57x3 = w57[3];
        ksum57 += (uint32_t) v57x0;
        ksum57 += (uint32_t) v57x1;
        ksum57 += (uint32_t) v57x2;
        ksum57 += (uint32_t) v57x3;
        out[228] = v57x0;
        out[229] = v57x1;
        out[230] = v57x2;
        out[231] = v57x3;
        w57 += 4;
        const int8_t v58x0 = w58[0];
        const int8_t v58x1 = w58[1];
        const int8_t v58x2 = w58[2];
        const int8_t v58x3 = w58[3];
        ksum58 += (uint32_t) v58x0;
        ksum58 += (uint32_t) v58x1;
        ksum58 += (uint32_t) v58x2;
        ksum58 += (uint32_t) v58x3;
        out[232] = v58x0;
        out[233] = v58x1;
        out[234] = v58x2;
        out[235] = v58x3;
        w58 += 4;
        const int8_t v59x0 = w59[0];
        const int8_t v59x1 = w59[1];
        const int8_t v59x2 = w59[2];
        const int8_t v59x3 = w59[3];
        ksum59 += (uint32_t) v59x0;
        ksum59 += (uint32_t) v59x1;
        ksum59 += (uint32_t) v59x2;
        ksum59 += (uint32_t) v59x3;
        out[236] = v59x0;
        out[237] = v59x1;
        out[238] = v59x2;
        out[239] = v59x3;
        w59 += 4;
        const int8_t v60x0 = w60[0];
        const int8_t v60x1 = w60[1];
        const int8_t v60x2 = w60[2];
        const int8_t v60x3 = w60[3];
        ksum60 += (uint32_t) v60x0;
        ksum60 += (uint32_t) v60x1;
        ksum60 += (uint32_t) v60x2;
        ksum60 += (uint32_t) v60x3;
        out[240] = v60x0;
        out[241] = v60x1;
        out[242] = v60x2;
        out[243] = v60x3;
        w60 += 4;
        const int8_t v61x0 = w61[0];
        const int8_t v61x1 = w61[1];
        const int8_t v61x2 = w61[2];
        const int8_t v61x3 = w61[3];
        ksum61 += (uint32_t) v61x0;
        ksum61 += (uint32_t) v61x1;
        ksum61 += (uint32_t) v61x2;
        ksum61 += (uint32_t) v61x3;
        out[244] = v61x0;
        out[245] = v61x1;
        out[246] = v61x2;
        out[247] = v61x3;
        w61 += 4;
        const int8_t v62x0 = w62[0];
        const int8_t v62x1 = w62[1];
        const int8_t v62x2 = w62[2];
        const int8_t v62x3 = w62[3];
        ksum62 += (uint32_t) v62x0;
        ksum62 += (uint32_t) v62x1;
        ksum62 += (uint32_t) v62x2;
        ksum62 += (uint32_t) v62x3;
        out[248] = v62x0;
        out[249] = v62x1;
        out[250] = v62x2;
        out[251] = v62x3;
        w62 += 4;
        const int8_t v63x0 = w63[0];
        const int8_t v63x1 = w63[1];
        const int8_t v63x2 = w63[2];
        const int8_t v63x3 = w63[3];
        ksum63 += (uint32_t) v63x0;
        ksum63 += (uint32_t) v63x1;
        ksum63 += (uint32_t) v63x2;
        ksum63 += (uint32_t) v63x3;
        out[252] = v63x0;
        out[253] = v63x1;
        out[254] = v63x2;
        out[255] = v63x3;
        w63 += 4;
        out += 256;
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
        const int8_t v16x0 = 0 < k ? w16[0] : izp;
        const int8_t v16x1 = 1 < k ? w16[1] : izp;
        const int8_t v16x2 = 2 < k ? w16[2] : izp;
        const int8_t v16x3 = 3 < k ? w16[3] : izp;
        ksum16 += (uint32_t) v16x0;
        ksum16 += (uint32_t) v16x1;
        ksum16 += (uint32_t) v16x2;
        ksum16 += (uint32_t) v16x3;
        if (0 < k) {
          out[64] = v16x0;
        }
        if (1 < k) {
          out[65] = v16x1;
        }
        if (2 < k) {
          out[66] = v16x2;
        }
        if (3 < k) {
          out[67] = v16x3;
        }
        w16 += 4;
        const int8_t v17x0 = 0 < k ? w17[0] : izp;
        const int8_t v17x1 = 1 < k ? w17[1] : izp;
        const int8_t v17x2 = 2 < k ? w17[2] : izp;
        const int8_t v17x3 = 3 < k ? w17[3] : izp;
        ksum17 += (uint32_t) v17x0;
        ksum17 += (uint32_t) v17x1;
        ksum17 += (uint32_t) v17x2;
        ksum17 += (uint32_t) v17x3;
        if (0 < k) {
          out[68] = v17x0;
        }
        if (1 < k) {
          out[69] = v17x1;
        }
        if (2 < k) {
          out[70] = v17x2;
        }
        if (3 < k) {
          out[71] = v17x3;
        }
        w17 += 4;
        const int8_t v18x0 = 0 < k ? w18[0] : izp;
        const int8_t v18x1 = 1 < k ? w18[1] : izp;
        const int8_t v18x2 = 2 < k ? w18[2] : izp;
        const int8_t v18x3 = 3 < k ? w18[3] : izp;
        ksum18 += (uint32_t) v18x0;
        ksum18 += (uint32_t) v18x1;
        ksum18 += (uint32_t) v18x2;
        ksum18 += (uint32_t) v18x3;
        if (0 < k) {
          out[72] = v18x0;
        }
        if (1 < k) {
          out[73] = v18x1;
        }
        if (2 < k) {
          out[74] = v18x2;
        }
        if (3 < k) {
          out[75] = v18x3;
        }
        w18 += 4;
        const int8_t v19x0 = 0 < k ? w19[0] : izp;
        const int8_t v19x1 = 1 < k ? w19[1] : izp;
        const int8_t v19x2 = 2 < k ? w19[2] : izp;
        const int8_t v19x3 = 3 < k ? w19[3] : izp;
        ksum19 += (uint32_t) v19x0;
        ksum19 += (uint32_t) v19x1;
        ksum19 += (uint32_t) v19x2;
        ksum19 += (uint32_t) v19x3;
        if (0 < k) {
          out[76] = v19x0;
        }
        if (1 < k) {
          out[77] = v19x1;
        }
        if (2 < k) {
          out[78] = v19x2;
        }
        if (3 < k) {
          out[79] = v19x3;
        }
        w19 += 4;
        const int8_t v20x0 = 0 < k ? w20[0] : izp;
        const int8_t v20x1 = 1 < k ? w20[1] : izp;
        const int8_t v20x2 = 2 < k ? w20[2] : izp;
        const int8_t v20x3 = 3 < k ? w20[3] : izp;
        ksum20 += (uint32_t) v20x0;
        ksum20 += (uint32_t) v20x1;
        ksum20 += (uint32_t) v20x2;
        ksum20 += (uint32_t) v20x3;
        if (0 < k) {
          out[80] = v20x0;
        }
        if (1 < k) {
          out[81] = v20x1;
        }
        if (2 < k) {
          out[82] = v20x2;
        }
        if (3 < k) {
          out[83] = v20x3;
        }
        w20 += 4;
        const int8_t v21x0 = 0 < k ? w21[0] : izp;
        const int8_t v21x1 = 1 < k ? w21[1] : izp;
        const int8_t v21x2 = 2 < k ? w21[2] : izp;
        const int8_t v21x3 = 3 < k ? w21[3] : izp;
        ksum21 += (uint32_t) v21x0;
        ksum21 += (uint32_t) v21x1;
        ksum21 += (uint32_t) v21x2;
        ksum21 += (uint32_t) v21x3;
        if (0 < k) {
          out[84] = v21x0;
        }
        if (1 < k) {
          out[85] = v21x1;
        }
        if (2 < k) {
          out[86] = v21x2;
        }
        if (3 < k) {
          out[87] = v21x3;
        }
        w21 += 4;
        const int8_t v22x0 = 0 < k ? w22[0] : izp;
        const int8_t v22x1 = 1 < k ? w22[1] : izp;
        const int8_t v22x2 = 2 < k ? w22[2] : izp;
        const int8_t v22x3 = 3 < k ? w22[3] : izp;
        ksum22 += (uint32_t) v22x0;
        ksum22 += (uint32_t) v22x1;
        ksum22 += (uint32_t) v22x2;
        ksum22 += (uint32_t) v22x3;
        if (0 < k) {
          out[88] = v22x0;
        }
        if (1 < k) {
          out[89] = v22x1;
        }
        if (2 < k) {
          out[90] = v22x2;
        }
        if (3 < k) {
          out[91] = v22x3;
        }
        w22 += 4;
        const int8_t v23x0 = 0 < k ? w23[0] : izp;
        const int8_t v23x1 = 1 < k ? w23[1] : izp;
        const int8_t v23x2 = 2 < k ? w23[2] : izp;
        const int8_t v23x3 = 3 < k ? w23[3] : izp;
        ksum23 += (uint32_t) v23x0;
        ksum23 += (uint32_t) v23x1;
        ksum23 += (uint32_t) v23x2;
        ksum23 += (uint32_t) v23x3;
        if (0 < k) {
          out[92] = v23x0;
        }
        if (1 < k) {
          out[93] = v23x1;
        }
        if (2 < k) {
          out[94] = v23x2;
        }
        if (3 < k) {
          out[95] = v23x3;
        }
        w23 += 4;
        const int8_t v24x0 = 0 < k ? w24[0] : izp;
        const int8_t v24x1 = 1 < k ? w24[1] : izp;
        const int8_t v24x2 = 2 < k ? w24[2] : izp;
        const int8_t v24x3 = 3 < k ? w24[3] : izp;
        ksum24 += (uint32_t) v24x0;
        ksum24 += (uint32_t) v24x1;
        ksum24 += (uint32_t) v24x2;
        ksum24 += (uint32_t) v24x3;
        if (0 < k) {
          out[96] = v24x0;
        }
        if (1 < k) {
          out[97] = v24x1;
        }
        if (2 < k) {
          out[98] = v24x2;
        }
        if (3 < k) {
          out[99] = v24x3;
        }
        w24 += 4;
        const int8_t v25x0 = 0 < k ? w25[0] : izp;
        const int8_t v25x1 = 1 < k ? w25[1] : izp;
        const int8_t v25x2 = 2 < k ? w25[2] : izp;
        const int8_t v25x3 = 3 < k ? w25[3] : izp;
        ksum25 += (uint32_t) v25x0;
        ksum25 += (uint32_t) v25x1;
        ksum25 += (uint32_t) v25x2;
        ksum25 += (uint32_t) v25x3;
        if (0 < k) {
          out[100] = v25x0;
        }
        if (1 < k) {
          out[101] = v25x1;
        }
        if (2 < k) {
          out[102] = v25x2;
        }
        if (3 < k) {
          out[103] = v25x3;
        }
        w25 += 4;
        const int8_t v26x0 = 0 < k ? w26[0] : izp;
        const int8_t v26x1 = 1 < k ? w26[1] : izp;
        const int8_t v26x2 = 2 < k ? w26[2] : izp;
        const int8_t v26x3 = 3 < k ? w26[3] : izp;
        ksum26 += (uint32_t) v26x0;
        ksum26 += (uint32_t) v26x1;
        ksum26 += (uint32_t) v26x2;
        ksum26 += (uint32_t) v26x3;
        if (0 < k) {
          out[104] = v26x0;
        }
        if (1 < k) {
          out[105] = v26x1;
        }
        if (2 < k) {
          out[106] = v26x2;
        }
        if (3 < k) {
          out[107] = v26x3;
        }
        w26 += 4;
        const int8_t v27x0 = 0 < k ? w27[0] : izp;
        const int8_t v27x1 = 1 < k ? w27[1] : izp;
        const int8_t v27x2 = 2 < k ? w27[2] : izp;
        const int8_t v27x3 = 3 < k ? w27[3] : izp;
        ksum27 += (uint32_t) v27x0;
        ksum27 += (uint32_t) v27x1;
        ksum27 += (uint32_t) v27x2;
        ksum27 += (uint32_t) v27x3;
        if (0 < k) {
          out[108] = v27x0;
        }
        if (1 < k) {
          out[109] = v27x1;
        }
        if (2 < k) {
          out[110] = v27x2;
        }
        if (3 < k) {
          out[111] = v27x3;
        }
        w27 += 4;
        const int8_t v28x0 = 0 < k ? w28[0] : izp;
        const int8_t v28x1 = 1 < k ? w28[1] : izp;
        const int8_t v28x2 = 2 < k ? w28[2] : izp;
        const int8_t v28x3 = 3 < k ? w28[3] : izp;
        ksum28 += (uint32_t) v28x0;
        ksum28 += (uint32_t) v28x1;
        ksum28 += (uint32_t) v28x2;
        ksum28 += (uint32_t) v28x3;
        if (0 < k) {
          out[112] = v28x0;
        }
        if (1 < k) {
          out[113] = v28x1;
        }
        if (2 < k) {
          out[114] = v28x2;
        }
        if (3 < k) {
          out[115] = v28x3;
        }
        w28 += 4;
        const int8_t v29x0 = 0 < k ? w29[0] : izp;
        const int8_t v29x1 = 1 < k ? w29[1] : izp;
        const int8_t v29x2 = 2 < k ? w29[2] : izp;
        const int8_t v29x3 = 3 < k ? w29[3] : izp;
        ksum29 += (uint32_t) v29x0;
        ksum29 += (uint32_t) v29x1;
        ksum29 += (uint32_t) v29x2;
        ksum29 += (uint32_t) v29x3;
        if (0 < k) {
          out[116] = v29x0;
        }
        if (1 < k) {
          out[117] = v29x1;
        }
        if (2 < k) {
          out[118] = v29x2;
        }
        if (3 < k) {
          out[119] = v29x3;
        }
        w29 += 4;
        const int8_t v30x0 = 0 < k ? w30[0] : izp;
        const int8_t v30x1 = 1 < k ? w30[1] : izp;
        const int8_t v30x2 = 2 < k ? w30[2] : izp;
        const int8_t v30x3 = 3 < k ? w30[3] : izp;
        ksum30 += (uint32_t) v30x0;
        ksum30 += (uint32_t) v30x1;
        ksum30 += (uint32_t) v30x2;
        ksum30 += (uint32_t) v30x3;
        if (0 < k) {
          out[120] = v30x0;
        }
        if (1 < k) {
          out[121] = v30x1;
        }
        if (2 < k) {
          out[122] = v30x2;
        }
        if (3 < k) {
          out[123] = v30x3;
        }
        w30 += 4;
        const int8_t v31x0 = 0 < k ? w31[0] : izp;
        const int8_t v31x1 = 1 < k ? w31[1] : izp;
        const int8_t v31x2 = 2 < k ? w31[2] : izp;
        const int8_t v31x3 = 3 < k ? w31[3] : izp;
        ksum31 += (uint32_t) v31x0;
        ksum31 += (uint32_t) v31x1;
        ksum31 += (uint32_t) v31x2;
        ksum31 += (uint32_t) v31x3;
        if (0 < k) {
          out[124] = v31x0;
        }
        if (1 < k) {
          out[125] = v31x1;
        }
        if (2 < k) {
          out[126] = v31x2;
        }
        if (3 < k) {
          out[127] = v31x3;
        }
        w31 += 4;
        const int8_t v32x0 = 0 < k ? w32[0] : izp;
        const int8_t v32x1 = 1 < k ? w32[1] : izp;
        const int8_t v32x2 = 2 < k ? w32[2] : izp;
        const int8_t v32x3 = 3 < k ? w32[3] : izp;
        ksum32 += (uint32_t) v32x0;
        ksum32 += (uint32_t) v32x1;
        ksum32 += (uint32_t) v32x2;
        ksum32 += (uint32_t) v32x3;
        if (0 < k) {
          out[128] = v32x0;
        }
        if (1 < k) {
          out[129] = v32x1;
        }
        if (2 < k) {
          out[130] = v32x2;
        }
        if (3 < k) {
          out[131] = v32x3;
        }
        w32 += 4;
        const int8_t v33x0 = 0 < k ? w33[0] : izp;
        const int8_t v33x1 = 1 < k ? w33[1] : izp;
        const int8_t v33x2 = 2 < k ? w33[2] : izp;
        const int8_t v33x3 = 3 < k ? w33[3] : izp;
        ksum33 += (uint32_t) v33x0;
        ksum33 += (uint32_t) v33x1;
        ksum33 += (uint32_t) v33x2;
        ksum33 += (uint32_t) v33x3;
        if (0 < k) {
          out[132] = v33x0;
        }
        if (1 < k) {
          out[133] = v33x1;
        }
        if (2 < k) {
          out[134] = v33x2;
        }
        if (3 < k) {
          out[135] = v33x3;
        }
        w33 += 4;
        const int8_t v34x0 = 0 < k ? w34[0] : izp;
        const int8_t v34x1 = 1 < k ? w34[1] : izp;
        const int8_t v34x2 = 2 < k ? w34[2] : izp;
        const int8_t v34x3 = 3 < k ? w34[3] : izp;
        ksum34 += (uint32_t) v34x0;
        ksum34 += (uint32_t) v34x1;
        ksum34 += (uint32_t) v34x2;
        ksum34 += (uint32_t) v34x3;
        if (0 < k) {
          out[136] = v34x0;
        }
        if (1 < k) {
          out[137] = v34x1;
        }
        if (2 < k) {
          out[138] = v34x2;
        }
        if (3 < k) {
          out[139] = v34x3;
        }
        w34 += 4;
        const int8_t v35x0 = 0 < k ? w35[0] : izp;
        const int8_t v35x1 = 1 < k ? w35[1] : izp;
        const int8_t v35x2 = 2 < k ? w35[2] : izp;
        const int8_t v35x3 = 3 < k ? w35[3] : izp;
        ksum35 += (uint32_t) v35x0;
        ksum35 += (uint32_t) v35x1;
        ksum35 += (uint32_t) v35x2;
        ksum35 += (uint32_t) v35x3;
        if (0 < k) {
          out[140] = v35x0;
        }
        if (1 < k) {
          out[141] = v35x1;
        }
        if (2 < k) {
          out[142] = v35x2;
        }
        if (3 < k) {
          out[143] = v35x3;
        }
        w35 += 4;
        const int8_t v36x0 = 0 < k ? w36[0] : izp;
        const int8_t v36x1 = 1 < k ? w36[1] : izp;
        const int8_t v36x2 = 2 < k ? w36[2] : izp;
        const int8_t v36x3 = 3 < k ? w36[3] : izp;
        ksum36 += (uint32_t) v36x0;
        ksum36 += (uint32_t) v36x1;
        ksum36 += (uint32_t) v36x2;
        ksum36 += (uint32_t) v36x3;
        if (0 < k) {
          out[144] = v36x0;
        }
        if (1 < k) {
          out[145] = v36x1;
        }
        if (2 < k) {
          out[146] = v36x2;
        }
        if (3 < k) {
          out[147] = v36x3;
        }
        w36 += 4;
        const int8_t v37x0 = 0 < k ? w37[0] : izp;
        const int8_t v37x1 = 1 < k ? w37[1] : izp;
        const int8_t v37x2 = 2 < k ? w37[2] : izp;
        const int8_t v37x3 = 3 < k ? w37[3] : izp;
        ksum37 += (uint32_t) v37x0;
        ksum37 += (uint32_t) v37x1;
        ksum37 += (uint32_t) v37x2;
        ksum37 += (uint32_t) v37x3;
        if (0 < k) {
          out[148] = v37x0;
        }
        if (1 < k) {
          out[149] = v37x1;
        }
        if (2 < k) {
          out[150] = v37x2;
        }
        if (3 < k) {
          out[151] = v37x3;
        }
        w37 += 4;
        const int8_t v38x0 = 0 < k ? w38[0] : izp;
        const int8_t v38x1 = 1 < k ? w38[1] : izp;
        const int8_t v38x2 = 2 < k ? w38[2] : izp;
        const int8_t v38x3 = 3 < k ? w38[3] : izp;
        ksum38 += (uint32_t) v38x0;
        ksum38 += (uint32_t) v38x1;
        ksum38 += (uint32_t) v38x2;
        ksum38 += (uint32_t) v38x3;
        if (0 < k) {
          out[152] = v38x0;
        }
        if (1 < k) {
          out[153] = v38x1;
        }
        if (2 < k) {
          out[154] = v38x2;
        }
        if (3 < k) {
          out[155] = v38x3;
        }
        w38 += 4;
        const int8_t v39x0 = 0 < k ? w39[0] : izp;
        const int8_t v39x1 = 1 < k ? w39[1] : izp;
        const int8_t v39x2 = 2 < k ? w39[2] : izp;
        const int8_t v39x3 = 3 < k ? w39[3] : izp;
        ksum39 += (uint32_t) v39x0;
        ksum39 += (uint32_t) v39x1;
        ksum39 += (uint32_t) v39x2;
        ksum39 += (uint32_t) v39x3;
        if (0 < k) {
          out[156] = v39x0;
        }
        if (1 < k) {
          out[157] = v39x1;
        }
        if (2 < k) {
          out[158] = v39x2;
        }
        if (3 < k) {
          out[159] = v39x3;
        }
        w39 += 4;
        const int8_t v40x0 = 0 < k ? w40[0] : izp;
        const int8_t v40x1 = 1 < k ? w40[1] : izp;
        const int8_t v40x2 = 2 < k ? w40[2] : izp;
        const int8_t v40x3 = 3 < k ? w40[3] : izp;
        ksum40 += (uint32_t) v40x0;
        ksum40 += (uint32_t) v40x1;
        ksum40 += (uint32_t) v40x2;
        ksum40 += (uint32_t) v40x3;
        if (0 < k) {
          out[160] = v40x0;
        }
        if (1 < k) {
          out[161] = v40x1;
        }
        if (2 < k) {
          out[162] = v40x2;
        }
        if (3 < k) {
          out[163] = v40x3;
        }
        w40 += 4;
        const int8_t v41x0 = 0 < k ? w41[0] : izp;
        const int8_t v41x1 = 1 < k ? w41[1] : izp;
        const int8_t v41x2 = 2 < k ? w41[2] : izp;
        const int8_t v41x3 = 3 < k ? w41[3] : izp;
        ksum41 += (uint32_t) v41x0;
        ksum41 += (uint32_t) v41x1;
        ksum41 += (uint32_t) v41x2;
        ksum41 += (uint32_t) v41x3;
        if (0 < k) {
          out[164] = v41x0;
        }
        if (1 < k) {
          out[165] = v41x1;
        }
        if (2 < k) {
          out[166] = v41x2;
        }
        if (3 < k) {
          out[167] = v41x3;
        }
        w41 += 4;
        const int8_t v42x0 = 0 < k ? w42[0] : izp;
        const int8_t v42x1 = 1 < k ? w42[1] : izp;
        const int8_t v42x2 = 2 < k ? w42[2] : izp;
        const int8_t v42x3 = 3 < k ? w42[3] : izp;
        ksum42 += (uint32_t) v42x0;
        ksum42 += (uint32_t) v42x1;
        ksum42 += (uint32_t) v42x2;
        ksum42 += (uint32_t) v42x3;
        if (0 < k) {
          out[168] = v42x0;
        }
        if (1 < k) {
          out[169] = v42x1;
        }
        if (2 < k) {
          out[170] = v42x2;
        }
        if (3 < k) {
          out[171] = v42x3;
        }
        w42 += 4;
        const int8_t v43x0 = 0 < k ? w43[0] : izp;
        const int8_t v43x1 = 1 < k ? w43[1] : izp;
        const int8_t v43x2 = 2 < k ? w43[2] : izp;
        const int8_t v43x3 = 3 < k ? w43[3] : izp;
        ksum43 += (uint32_t) v43x0;
        ksum43 += (uint32_t) v43x1;
        ksum43 += (uint32_t) v43x2;
        ksum43 += (uint32_t) v43x3;
        if (0 < k) {
          out[172] = v43x0;
        }
        if (1 < k) {
          out[173] = v43x1;
        }
        if (2 < k) {
          out[174] = v43x2;
        }
        if (3 < k) {
          out[175] = v43x3;
        }
        w43 += 4;
        const int8_t v44x0 = 0 < k ? w44[0] : izp;
        const int8_t v44x1 = 1 < k ? w44[1] : izp;
        const int8_t v44x2 = 2 < k ? w44[2] : izp;
        const int8_t v44x3 = 3 < k ? w44[3] : izp;
        ksum44 += (uint32_t) v44x0;
        ksum44 += (uint32_t) v44x1;
        ksum44 += (uint32_t) v44x2;
        ksum44 += (uint32_t) v44x3;
        if (0 < k) {
          out[176] = v44x0;
        }
        if (1 < k) {
          out[177] = v44x1;
        }
        if (2 < k) {
          out[178] = v44x2;
        }
        if (3 < k) {
          out[179] = v44x3;
        }
        w44 += 4;
        const int8_t v45x0 = 0 < k ? w45[0] : izp;
        const int8_t v45x1 = 1 < k ? w45[1] : izp;
        const int8_t v45x2 = 2 < k ? w45[2] : izp;
        const int8_t v45x3 = 3 < k ? w45[3] : izp;
        ksum45 += (uint32_t) v45x0;
        ksum45 += (uint32_t) v45x1;
        ksum45 += (uint32_t) v45x2;
        ksum45 += (uint32_t) v45x3;
        if (0 < k) {
          out[180] = v45x0;
        }
        if (1 < k) {
          out[181] = v45x1;
        }
        if (2 < k) {
          out[182] = v45x2;
        }
        if (3 < k) {
          out[183] = v45x3;
        }
        w45 += 4;
        const int8_t v46x0 = 0 < k ? w46[0] : izp;
        const int8_t v46x1 = 1 < k ? w46[1] : izp;
        const int8_t v46x2 = 2 < k ? w46[2] : izp;
        const int8_t v46x3 = 3 < k ? w46[3] : izp;
        ksum46 += (uint32_t) v46x0;
        ksum46 += (uint32_t) v46x1;
        ksum46 += (uint32_t) v46x2;
        ksum46 += (uint32_t) v46x3;
        if (0 < k) {
          out[184] = v46x0;
        }
        if (1 < k) {
          out[185] = v46x1;
        }
        if (2 < k) {
          out[186] = v46x2;
        }
        if (3 < k) {
          out[187] = v46x3;
        }
        w46 += 4;
        const int8_t v47x0 = 0 < k ? w47[0] : izp;
        const int8_t v47x1 = 1 < k ? w47[1] : izp;
        const int8_t v47x2 = 2 < k ? w47[2] : izp;
        const int8_t v47x3 = 3 < k ? w47[3] : izp;
        ksum47 += (uint32_t) v47x0;
        ksum47 += (uint32_t) v47x1;
        ksum47 += (uint32_t) v47x2;
        ksum47 += (uint32_t) v47x3;
        if (0 < k) {
          out[188] = v47x0;
        }
        if (1 < k) {
          out[189] = v47x1;
        }
        if (2 < k) {
          out[190] = v47x2;
        }
        if (3 < k) {
          out[191] = v47x3;
        }
        w47 += 4;
        const int8_t v48x0 = 0 < k ? w48[0] : izp;
        const int8_t v48x1 = 1 < k ? w48[1] : izp;
        const int8_t v48x2 = 2 < k ? w48[2] : izp;
        const int8_t v48x3 = 3 < k ? w48[3] : izp;
        ksum48 += (uint32_t) v48x0;
        ksum48 += (uint32_t) v48x1;
        ksum48 += (uint32_t) v48x2;
        ksum48 += (uint32_t) v48x3;
        if (0 < k) {
          out[192] = v48x0;
        }
        if (1 < k) {
          out[193] = v48x1;
        }
        if (2 < k) {
          out[194] = v48x2;
        }
        if (3 < k) {
          out[195] = v48x3;
        }
        w48 += 4;
        const int8_t v49x0 = 0 < k ? w49[0] : izp;
        const int8_t v49x1 = 1 < k ? w49[1] : izp;
        const int8_t v49x2 = 2 < k ? w49[2] : izp;
        const int8_t v49x3 = 3 < k ? w49[3] : izp;
        ksum49 += (uint32_t) v49x0;
        ksum49 += (uint32_t) v49x1;
        ksum49 += (uint32_t) v49x2;
        ksum49 += (uint32_t) v49x3;
        if (0 < k) {
          out[196] = v49x0;
        }
        if (1 < k) {
          out[197] = v49x1;
        }
        if (2 < k) {
          out[198] = v49x2;
        }
        if (3 < k) {
          out[199] = v49x3;
        }
        w49 += 4;
        const int8_t v50x0 = 0 < k ? w50[0] : izp;
        const int8_t v50x1 = 1 < k ? w50[1] : izp;
        const int8_t v50x2 = 2 < k ? w50[2] : izp;
        const int8_t v50x3 = 3 < k ? w50[3] : izp;
        ksum50 += (uint32_t) v50x0;
        ksum50 += (uint32_t) v50x1;
        ksum50 += (uint32_t) v50x2;
        ksum50 += (uint32_t) v50x3;
        if (0 < k) {
          out[200] = v50x0;
        }
        if (1 < k) {
          out[201] = v50x1;
        }
        if (2 < k) {
          out[202] = v50x2;
        }
        if (3 < k) {
          out[203] = v50x3;
        }
        w50 += 4;
        const int8_t v51x0 = 0 < k ? w51[0] : izp;
        const int8_t v51x1 = 1 < k ? w51[1] : izp;
        const int8_t v51x2 = 2 < k ? w51[2] : izp;
        const int8_t v51x3 = 3 < k ? w51[3] : izp;
        ksum51 += (uint32_t) v51x0;
        ksum51 += (uint32_t) v51x1;
        ksum51 += (uint32_t) v51x2;
        ksum51 += (uint32_t) v51x3;
        if (0 < k) {
          out[204] = v51x0;
        }
        if (1 < k) {
          out[205] = v51x1;
        }
        if (2 < k) {
          out[206] = v51x2;
        }
        if (3 < k) {
          out[207] = v51x3;
        }
        w51 += 4;
        const int8_t v52x0 = 0 < k ? w52[0] : izp;
        const int8_t v52x1 = 1 < k ? w52[1] : izp;
        const int8_t v52x2 = 2 < k ? w52[2] : izp;
        const int8_t v52x3 = 3 < k ? w52[3] : izp;
        ksum52 += (uint32_t) v52x0;
        ksum52 += (uint32_t) v52x1;
        ksum52 += (uint32_t) v52x2;
        ksum52 += (uint32_t) v52x3;
        if (0 < k) {
          out[208] = v52x0;
        }
        if (1 < k) {
          out[209] = v52x1;
        }
        if (2 < k) {
          out[210] = v52x2;
        }
        if (3 < k) {
          out[211] = v52x3;
        }
        w52 += 4;
        const int8_t v53x0 = 0 < k ? w53[0] : izp;
        const int8_t v53x1 = 1 < k ? w53[1] : izp;
        const int8_t v53x2 = 2 < k ? w53[2] : izp;
        const int8_t v53x3 = 3 < k ? w53[3] : izp;
        ksum53 += (uint32_t) v53x0;
        ksum53 += (uint32_t) v53x1;
        ksum53 += (uint32_t) v53x2;
        ksum53 += (uint32_t) v53x3;
        if (0 < k) {
          out[212] = v53x0;
        }
        if (1 < k) {
          out[213] = v53x1;
        }
        if (2 < k) {
          out[214] = v53x2;
        }
        if (3 < k) {
          out[215] = v53x3;
        }
        w53 += 4;
        const int8_t v54x0 = 0 < k ? w54[0] : izp;
        const int8_t v54x1 = 1 < k ? w54[1] : izp;
        const int8_t v54x2 = 2 < k ? w54[2] : izp;
        const int8_t v54x3 = 3 < k ? w54[3] : izp;
        ksum54 += (uint32_t) v54x0;
        ksum54 += (uint32_t) v54x1;
        ksum54 += (uint32_t) v54x2;
        ksum54 += (uint32_t) v54x3;
        if (0 < k) {
          out[216] = v54x0;
        }
        if (1 < k) {
          out[217] = v54x1;
        }
        if (2 < k) {
          out[218] = v54x2;
        }
        if (3 < k) {
          out[219] = v54x3;
        }
        w54 += 4;
        const int8_t v55x0 = 0 < k ? w55[0] : izp;
        const int8_t v55x1 = 1 < k ? w55[1] : izp;
        const int8_t v55x2 = 2 < k ? w55[2] : izp;
        const int8_t v55x3 = 3 < k ? w55[3] : izp;
        ksum55 += (uint32_t) v55x0;
        ksum55 += (uint32_t) v55x1;
        ksum55 += (uint32_t) v55x2;
        ksum55 += (uint32_t) v55x3;
        if (0 < k) {
          out[220] = v55x0;
        }
        if (1 < k) {
          out[221] = v55x1;
        }
        if (2 < k) {
          out[222] = v55x2;
        }
        if (3 < k) {
          out[223] = v55x3;
        }
        w55 += 4;
        const int8_t v56x0 = 0 < k ? w56[0] : izp;
        const int8_t v56x1 = 1 < k ? w56[1] : izp;
        const int8_t v56x2 = 2 < k ? w56[2] : izp;
        const int8_t v56x3 = 3 < k ? w56[3] : izp;
        ksum56 += (uint32_t) v56x0;
        ksum56 += (uint32_t) v56x1;
        ksum56 += (uint32_t) v56x2;
        ksum56 += (uint32_t) v56x3;
        if (0 < k) {
          out[224] = v56x0;
        }
        if (1 < k) {
          out[225] = v56x1;
        }
        if (2 < k) {
          out[226] = v56x2;
        }
        if (3 < k) {
          out[227] = v56x3;
        }
        w56 += 4;
        const int8_t v57x0 = 0 < k ? w57[0] : izp;
        const int8_t v57x1 = 1 < k ? w57[1] : izp;
        const int8_t v57x2 = 2 < k ? w57[2] : izp;
        const int8_t v57x3 = 3 < k ? w57[3] : izp;
        ksum57 += (uint32_t) v57x0;
        ksum57 += (uint32_t) v57x1;
        ksum57 += (uint32_t) v57x2;
        ksum57 += (uint32_t) v57x3;
        if (0 < k) {
          out[228] = v57x0;
        }
        if (1 < k) {
          out[229] = v57x1;
        }
        if (2 < k) {
          out[230] = v57x2;
        }
        if (3 < k) {
          out[231] = v57x3;
        }
        w57 += 4;
        const int8_t v58x0 = 0 < k ? w58[0] : izp;
        const int8_t v58x1 = 1 < k ? w58[1] : izp;
        const int8_t v58x2 = 2 < k ? w58[2] : izp;
        const int8_t v58x3 = 3 < k ? w58[3] : izp;
        ksum58 += (uint32_t) v58x0;
        ksum58 += (uint32_t) v58x1;
        ksum58 += (uint32_t) v58x2;
        ksum58 += (uint32_t) v58x3;
        if (0 < k) {
          out[232] = v58x0;
        }
        if (1 < k) {
          out[233] = v58x1;
        }
        if (2 < k) {
          out[234] = v58x2;
        }
        if (3 < k) {
          out[235] = v58x3;
        }
        w58 += 4;
        const int8_t v59x0 = 0 < k ? w59[0] : izp;
        const int8_t v59x1 = 1 < k ? w59[1] : izp;
        const int8_t v59x2 = 2 < k ? w59[2] : izp;
        const int8_t v59x3 = 3 < k ? w59[3] : izp;
        ksum59 += (uint32_t) v59x0;
        ksum59 += (uint32_t) v59x1;
        ksum59 += (uint32_t) v59x2;
        ksum59 += (uint32_t) v59x3;
        if (0 < k) {
          out[236] = v59x0;
        }
        if (1 < k) {
          out[237] = v59x1;
        }
        if (2 < k) {
          out[238] = v59x2;
        }
        if (3 < k) {
          out[239] = v59x3;
        }
        w59 += 4;
        const int8_t v60x0 = 0 < k ? w60[0] : izp;
        const int8_t v60x1 = 1 < k ? w60[1] : izp;
        const int8_t v60x2 = 2 < k ? w60[2] : izp;
        const int8_t v60x3 = 3 < k ? w60[3] : izp;
        ksum60 += (uint32_t) v60x0;
        ksum60 += (uint32_t) v60x1;
        ksum60 += (uint32_t) v60x2;
        ksum60 += (uint32_t) v60x3;
        if (0 < k) {
          out[240] = v60x0;
        }
        if (1 < k) {
          out[241] = v60x1;
        }
        if (2 < k) {
          out[242] = v60x2;
        }
        if (3 < k) {
          out[243] = v60x3;
        }
        w60 += 4;
        const int8_t v61x0 = 0 < k ? w61[0] : izp;
        const int8_t v61x1 = 1 < k ? w61[1] : izp;
        const int8_t v61x2 = 2 < k ? w61[2] : izp;
        const int8_t v61x3 = 3 < k ? w61[3] : izp;
        ksum61 += (uint32_t) v61x0;
        ksum61 += (uint32_t) v61x1;
        ksum61 += (uint32_t) v61x2;
        ksum61 += (uint32_t) v61x3;
        if (0 < k) {
          out[244] = v61x0;
        }
        if (1 < k) {
          out[245] = v61x1;
        }
        if (2 < k) {
          out[246] = v61x2;
        }
        if (3 < k) {
          out[247] = v61x3;
        }
        w61 += 4;
        const int8_t v62x0 = 0 < k ? w62[0] : izp;
        const int8_t v62x1 = 1 < k ? w62[1] : izp;
        const int8_t v62x2 = 2 < k ? w62[2] : izp;
        const int8_t v62x3 = 3 < k ? w62[3] : izp;
        ksum62 += (uint32_t) v62x0;
        ksum62 += (uint32_t) v62x1;
        ksum62 += (uint32_t) v62x2;
        ksum62 += (uint32_t) v62x3;
        if (0 < k) {
          out[248] = v62x0;
        }
        if (1 < k) {
          out[249] = v62x1;
        }
        if (2 < k) {
          out[250] = v62x2;
        }
        if (3 < k) {
          out[251] = v62x3;
        }
        w62 += 4;
        const int8_t v63x0 = 0 < k ? w63[0] : izp;
        const int8_t v63x1 = 1 < k ? w63[1] : izp;
        const int8_t v63x2 = 2 < k ? w63[2] : izp;
        const int8_t v63x3 = 3 < k ? w63[3] : izp;
        ksum63 += (uint32_t) v63x0;
        ksum63 += (uint32_t) v63x1;
        ksum63 += (uint32_t) v63x2;
        ksum63 += (uint32_t) v63x3;
        if (0 < k) {
          out[252] = v63x0;
        }
        if (1 < k) {
          out[253] = v63x1;
        }
        if (2 < k) {
          out[254] = v63x2;
        }
        if (3 < k) {
          out[255] = v63x3;
        }
        w63 += 4;
        out += 256;
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
      packed_b[16] -= ksum16 * izp;
      packed_b[17] -= ksum17 * izp;
      packed_b[18] -= ksum18 * izp;
      packed_b[19] -= ksum19 * izp;
      packed_b[20] -= ksum20 * izp;
      packed_b[21] -= ksum21 * izp;
      packed_b[22] -= ksum22 * izp;
      packed_b[23] -= ksum23 * izp;
      packed_b[24] -= ksum24 * izp;
      packed_b[25] -= ksum25 * izp;
      packed_b[26] -= ksum26 * izp;
      packed_b[27] -= ksum27 * izp;
      packed_b[28] -= ksum28 * izp;
      packed_b[29] -= ksum29 * izp;
      packed_b[30] -= ksum30 * izp;
      packed_b[31] -= ksum31 * izp;
      packed_b[32] -= ksum32 * izp;
      packed_b[33] -= ksum33 * izp;
      packed_b[34] -= ksum34 * izp;
      packed_b[35] -= ksum35 * izp;
      packed_b[36] -= ksum36 * izp;
      packed_b[37] -= ksum37 * izp;
      packed_b[38] -= ksum38 * izp;
      packed_b[39] -= ksum39 * izp;
      packed_b[40] -= ksum40 * izp;
      packed_b[41] -= ksum41 * izp;
      packed_b[42] -= ksum42 * izp;
      packed_b[43] -= ksum43 * izp;
      packed_b[44] -= ksum44 * izp;
      packed_b[45] -= ksum45 * izp;
      packed_b[46] -= ksum46 * izp;
      packed_b[47] -= ksum47 * izp;
      packed_b[48] -= ksum48 * izp;
      packed_b[49] -= ksum49 * izp;
      packed_b[50] -= ksum50 * izp;
      packed_b[51] -= ksum51 * izp;
      packed_b[52] -= ksum52 * izp;
      packed_b[53] -= ksum53 * izp;
      packed_b[54] -= ksum54 * izp;
      packed_b[55] -= ksum55 * izp;
      packed_b[56] -= ksum56 * izp;
      packed_b[57] -= ksum57 * izp;
      packed_b[58] -= ksum58 * izp;
      packed_b[59] -= ksum59 * izp;
      packed_b[60] -= ksum60 * izp;
      packed_b[61] -= ksum61 * izp;
      packed_b[62] -= ksum62 * izp;
      packed_b[63] -= ksum63 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w63;
    }

    // NC remainder (1..63)
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
      out += (64 - n) * sizeof(int32_t);

     // NR remainder has less than 64 rows so last row is not loaded
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
      const int8_t* w15 = w14 + kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const int8_t* w16 = w15 + kc;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const int8_t* w17 = w16 + kc;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const int8_t* w18 = w17 + kc;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const int8_t* w19 = w18 + kc;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const int8_t* w20 = w19 + kc;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const int8_t* w21 = w20 + kc;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const int8_t* w22 = w21 + kc;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const int8_t* w23 = w22 + kc;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const int8_t* w24 = w23 + kc;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const int8_t* w25 = w24 + kc;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const int8_t* w26 = w25 + kc;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const int8_t* w27 = w26 + kc;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const int8_t* w28 = w27 + kc;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const int8_t* w29 = w28 + kc;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const int8_t* w30 = w29 + kc;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }
      const int8_t* w31 = w30 + kc;
      if XNN_UNPREDICTABLE(n < 32) {
        w31 = w30;
      }
      const int8_t* w32 = w31 + kc;
      if XNN_UNPREDICTABLE(n <= 32) {
        w32 = w31;
      }
      const int8_t* w33 = w32 + kc;
      if XNN_UNPREDICTABLE(n < 34) {
        w33 = w32;
      }
      const int8_t* w34 = w33 + kc;
      if XNN_UNPREDICTABLE(n <= 34) {
        w34 = w33;
      }
      const int8_t* w35 = w34 + kc;
      if XNN_UNPREDICTABLE(n < 36) {
        w35 = w34;
      }
      const int8_t* w36 = w35 + kc;
      if XNN_UNPREDICTABLE(n <= 36) {
        w36 = w35;
      }
      const int8_t* w37 = w36 + kc;
      if XNN_UNPREDICTABLE(n < 38) {
        w37 = w36;
      }
      const int8_t* w38 = w37 + kc;
      if XNN_UNPREDICTABLE(n <= 38) {
        w38 = w37;
      }
      const int8_t* w39 = w38 + kc;
      if XNN_UNPREDICTABLE(n < 40) {
        w39 = w38;
      }
      const int8_t* w40 = w39 + kc;
      if XNN_UNPREDICTABLE(n <= 40) {
        w40 = w39;
      }
      const int8_t* w41 = w40 + kc;
      if XNN_UNPREDICTABLE(n < 42) {
        w41 = w40;
      }
      const int8_t* w42 = w41 + kc;
      if XNN_UNPREDICTABLE(n <= 42) {
        w42 = w41;
      }
      const int8_t* w43 = w42 + kc;
      if XNN_UNPREDICTABLE(n < 44) {
        w43 = w42;
      }
      const int8_t* w44 = w43 + kc;
      if XNN_UNPREDICTABLE(n <= 44) {
        w44 = w43;
      }
      const int8_t* w45 = w44 + kc;
      if XNN_UNPREDICTABLE(n < 46) {
        w45 = w44;
      }
      const int8_t* w46 = w45 + kc;
      if XNN_UNPREDICTABLE(n <= 46) {
        w46 = w45;
      }
      const int8_t* w47 = w46 + kc;
      if XNN_UNPREDICTABLE(n < 48) {
        w47 = w46;
      }
      const int8_t* w48 = w47 + kc;
      if XNN_UNPREDICTABLE(n <= 48) {
        w48 = w47;
      }
      const int8_t* w49 = w48 + kc;
      if XNN_UNPREDICTABLE(n < 50) {
        w49 = w48;
      }
      const int8_t* w50 = w49 + kc;
      if XNN_UNPREDICTABLE(n <= 50) {
        w50 = w49;
      }
      const int8_t* w51 = w50 + kc;
      if XNN_UNPREDICTABLE(n < 52) {
        w51 = w50;
      }
      const int8_t* w52 = w51 + kc;
      if XNN_UNPREDICTABLE(n <= 52) {
        w52 = w51;
      }
      const int8_t* w53 = w52 + kc;
      if XNN_UNPREDICTABLE(n < 54) {
        w53 = w52;
      }
      const int8_t* w54 = w53 + kc;
      if XNN_UNPREDICTABLE(n <= 54) {
        w54 = w53;
      }
      const int8_t* w55 = w54 + kc;
      if XNN_UNPREDICTABLE(n < 56) {
        w55 = w54;
      }
      const int8_t* w56 = w55 + kc;
      if XNN_UNPREDICTABLE(n <= 56) {
        w56 = w55;
      }
      const int8_t* w57 = w56 + kc;
      if XNN_UNPREDICTABLE(n < 58) {
        w57 = w56;
      }
      const int8_t* w58 = w57 + kc;
      if XNN_UNPREDICTABLE(n <= 58) {
        w58 = w57;
      }
      const int8_t* w59 = w58 + kc;
      if XNN_UNPREDICTABLE(n < 60) {
        w59 = w58;
      }
      const int8_t* w60 = w59 + kc;
      if XNN_UNPREDICTABLE(n <= 60) {
        w60 = w59;
      }
      const int8_t* w61 = w60 + kc;
      if XNN_UNPREDICTABLE(n < 62) {
        w61 = w60;
      }
      const int8_t* w62 = w61 + kc;
      if XNN_UNPREDICTABLE(n <= 62) {
        w62 = w61;
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
      uint32_t ksum15 = 0;
      uint32_t ksum16 = 0;
      uint32_t ksum17 = 0;
      uint32_t ksum18 = 0;
      uint32_t ksum19 = 0;
      uint32_t ksum20 = 0;
      uint32_t ksum21 = 0;
      uint32_t ksum22 = 0;
      uint32_t ksum23 = 0;
      uint32_t ksum24 = 0;
      uint32_t ksum25 = 0;
      uint32_t ksum26 = 0;
      uint32_t ksum27 = 0;
      uint32_t ksum28 = 0;
      uint32_t ksum29 = 0;
      uint32_t ksum30 = 0;
      uint32_t ksum31 = 0;
      uint32_t ksum32 = 0;
      uint32_t ksum33 = 0;
      uint32_t ksum34 = 0;
      uint32_t ksum35 = 0;
      uint32_t ksum36 = 0;
      uint32_t ksum37 = 0;
      uint32_t ksum38 = 0;
      uint32_t ksum39 = 0;
      uint32_t ksum40 = 0;
      uint32_t ksum41 = 0;
      uint32_t ksum42 = 0;
      uint32_t ksum43 = 0;
      uint32_t ksum44 = 0;
      uint32_t ksum45 = 0;
      uint32_t ksum46 = 0;
      uint32_t ksum47 = 0;
      uint32_t ksum48 = 0;
      uint32_t ksum49 = 0;
      uint32_t ksum50 = 0;
      uint32_t ksum51 = 0;
      uint32_t ksum52 = 0;
      uint32_t ksum53 = 0;
      uint32_t ksum54 = 0;
      uint32_t ksum55 = 0;
      uint32_t ksum56 = 0;
      uint32_t ksum57 = 0;
      uint32_t ksum58 = 0;
      uint32_t ksum59 = 0;
      uint32_t ksum60 = 0;
      uint32_t ksum61 = 0;
      uint32_t ksum62 = 0;

      // KC main loop multiple of 64x4
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
        const int8_t v16x0 = w16[0];
        const int8_t v16x1 = w16[1];
        const int8_t v16x2 = w16[2];
        const int8_t v16x3 = w16[3];
        ksum16 += (uint32_t) v16x0;
        ksum16 += (uint32_t) v16x1;
        ksum16 += (uint32_t) v16x2;
        ksum16 += (uint32_t) v16x3;
        out[64] = v16x0;
        out[65] = v16x1;
        out[66] = v16x2;
        out[67] = v16x3;
        w16 += 4;
        const int8_t v17x0 = w17[0];
        const int8_t v17x1 = w17[1];
        const int8_t v17x2 = w17[2];
        const int8_t v17x3 = w17[3];
        ksum17 += (uint32_t) v17x0;
        ksum17 += (uint32_t) v17x1;
        ksum17 += (uint32_t) v17x2;
        ksum17 += (uint32_t) v17x3;
        out[68] = v17x0;
        out[69] = v17x1;
        out[70] = v17x2;
        out[71] = v17x3;
        w17 += 4;
        const int8_t v18x0 = w18[0];
        const int8_t v18x1 = w18[1];
        const int8_t v18x2 = w18[2];
        const int8_t v18x3 = w18[3];
        ksum18 += (uint32_t) v18x0;
        ksum18 += (uint32_t) v18x1;
        ksum18 += (uint32_t) v18x2;
        ksum18 += (uint32_t) v18x3;
        out[72] = v18x0;
        out[73] = v18x1;
        out[74] = v18x2;
        out[75] = v18x3;
        w18 += 4;
        const int8_t v19x0 = w19[0];
        const int8_t v19x1 = w19[1];
        const int8_t v19x2 = w19[2];
        const int8_t v19x3 = w19[3];
        ksum19 += (uint32_t) v19x0;
        ksum19 += (uint32_t) v19x1;
        ksum19 += (uint32_t) v19x2;
        ksum19 += (uint32_t) v19x3;
        out[76] = v19x0;
        out[77] = v19x1;
        out[78] = v19x2;
        out[79] = v19x3;
        w19 += 4;
        const int8_t v20x0 = w20[0];
        const int8_t v20x1 = w20[1];
        const int8_t v20x2 = w20[2];
        const int8_t v20x3 = w20[3];
        ksum20 += (uint32_t) v20x0;
        ksum20 += (uint32_t) v20x1;
        ksum20 += (uint32_t) v20x2;
        ksum20 += (uint32_t) v20x3;
        out[80] = v20x0;
        out[81] = v20x1;
        out[82] = v20x2;
        out[83] = v20x3;
        w20 += 4;
        const int8_t v21x0 = w21[0];
        const int8_t v21x1 = w21[1];
        const int8_t v21x2 = w21[2];
        const int8_t v21x3 = w21[3];
        ksum21 += (uint32_t) v21x0;
        ksum21 += (uint32_t) v21x1;
        ksum21 += (uint32_t) v21x2;
        ksum21 += (uint32_t) v21x3;
        out[84] = v21x0;
        out[85] = v21x1;
        out[86] = v21x2;
        out[87] = v21x3;
        w21 += 4;
        const int8_t v22x0 = w22[0];
        const int8_t v22x1 = w22[1];
        const int8_t v22x2 = w22[2];
        const int8_t v22x3 = w22[3];
        ksum22 += (uint32_t) v22x0;
        ksum22 += (uint32_t) v22x1;
        ksum22 += (uint32_t) v22x2;
        ksum22 += (uint32_t) v22x3;
        out[88] = v22x0;
        out[89] = v22x1;
        out[90] = v22x2;
        out[91] = v22x3;
        w22 += 4;
        const int8_t v23x0 = w23[0];
        const int8_t v23x1 = w23[1];
        const int8_t v23x2 = w23[2];
        const int8_t v23x3 = w23[3];
        ksum23 += (uint32_t) v23x0;
        ksum23 += (uint32_t) v23x1;
        ksum23 += (uint32_t) v23x2;
        ksum23 += (uint32_t) v23x3;
        out[92] = v23x0;
        out[93] = v23x1;
        out[94] = v23x2;
        out[95] = v23x3;
        w23 += 4;
        const int8_t v24x0 = w24[0];
        const int8_t v24x1 = w24[1];
        const int8_t v24x2 = w24[2];
        const int8_t v24x3 = w24[3];
        ksum24 += (uint32_t) v24x0;
        ksum24 += (uint32_t) v24x1;
        ksum24 += (uint32_t) v24x2;
        ksum24 += (uint32_t) v24x3;
        out[96] = v24x0;
        out[97] = v24x1;
        out[98] = v24x2;
        out[99] = v24x3;
        w24 += 4;
        const int8_t v25x0 = w25[0];
        const int8_t v25x1 = w25[1];
        const int8_t v25x2 = w25[2];
        const int8_t v25x3 = w25[3];
        ksum25 += (uint32_t) v25x0;
        ksum25 += (uint32_t) v25x1;
        ksum25 += (uint32_t) v25x2;
        ksum25 += (uint32_t) v25x3;
        out[100] = v25x0;
        out[101] = v25x1;
        out[102] = v25x2;
        out[103] = v25x3;
        w25 += 4;
        const int8_t v26x0 = w26[0];
        const int8_t v26x1 = w26[1];
        const int8_t v26x2 = w26[2];
        const int8_t v26x3 = w26[3];
        ksum26 += (uint32_t) v26x0;
        ksum26 += (uint32_t) v26x1;
        ksum26 += (uint32_t) v26x2;
        ksum26 += (uint32_t) v26x3;
        out[104] = v26x0;
        out[105] = v26x1;
        out[106] = v26x2;
        out[107] = v26x3;
        w26 += 4;
        const int8_t v27x0 = w27[0];
        const int8_t v27x1 = w27[1];
        const int8_t v27x2 = w27[2];
        const int8_t v27x3 = w27[3];
        ksum27 += (uint32_t) v27x0;
        ksum27 += (uint32_t) v27x1;
        ksum27 += (uint32_t) v27x2;
        ksum27 += (uint32_t) v27x3;
        out[108] = v27x0;
        out[109] = v27x1;
        out[110] = v27x2;
        out[111] = v27x3;
        w27 += 4;
        const int8_t v28x0 = w28[0];
        const int8_t v28x1 = w28[1];
        const int8_t v28x2 = w28[2];
        const int8_t v28x3 = w28[3];
        ksum28 += (uint32_t) v28x0;
        ksum28 += (uint32_t) v28x1;
        ksum28 += (uint32_t) v28x2;
        ksum28 += (uint32_t) v28x3;
        out[112] = v28x0;
        out[113] = v28x1;
        out[114] = v28x2;
        out[115] = v28x3;
        w28 += 4;
        const int8_t v29x0 = w29[0];
        const int8_t v29x1 = w29[1];
        const int8_t v29x2 = w29[2];
        const int8_t v29x3 = w29[3];
        ksum29 += (uint32_t) v29x0;
        ksum29 += (uint32_t) v29x1;
        ksum29 += (uint32_t) v29x2;
        ksum29 += (uint32_t) v29x3;
        out[116] = v29x0;
        out[117] = v29x1;
        out[118] = v29x2;
        out[119] = v29x3;
        w29 += 4;
        const int8_t v30x0 = w30[0];
        const int8_t v30x1 = w30[1];
        const int8_t v30x2 = w30[2];
        const int8_t v30x3 = w30[3];
        ksum30 += (uint32_t) v30x0;
        ksum30 += (uint32_t) v30x1;
        ksum30 += (uint32_t) v30x2;
        ksum30 += (uint32_t) v30x3;
        out[120] = v30x0;
        out[121] = v30x1;
        out[122] = v30x2;
        out[123] = v30x3;
        w30 += 4;
        const int8_t v31x0 = w31[0];
        const int8_t v31x1 = w31[1];
        const int8_t v31x2 = w31[2];
        const int8_t v31x3 = w31[3];
        ksum31 += (uint32_t) v31x0;
        ksum31 += (uint32_t) v31x1;
        ksum31 += (uint32_t) v31x2;
        ksum31 += (uint32_t) v31x3;
        out[124] = v31x0;
        out[125] = v31x1;
        out[126] = v31x2;
        out[127] = v31x3;
        w31 += 4;
        const int8_t v32x0 = w32[0];
        const int8_t v32x1 = w32[1];
        const int8_t v32x2 = w32[2];
        const int8_t v32x3 = w32[3];
        ksum32 += (uint32_t) v32x0;
        ksum32 += (uint32_t) v32x1;
        ksum32 += (uint32_t) v32x2;
        ksum32 += (uint32_t) v32x3;
        out[128] = v32x0;
        out[129] = v32x1;
        out[130] = v32x2;
        out[131] = v32x3;
        w32 += 4;
        const int8_t v33x0 = w33[0];
        const int8_t v33x1 = w33[1];
        const int8_t v33x2 = w33[2];
        const int8_t v33x3 = w33[3];
        ksum33 += (uint32_t) v33x0;
        ksum33 += (uint32_t) v33x1;
        ksum33 += (uint32_t) v33x2;
        ksum33 += (uint32_t) v33x3;
        out[132] = v33x0;
        out[133] = v33x1;
        out[134] = v33x2;
        out[135] = v33x3;
        w33 += 4;
        const int8_t v34x0 = w34[0];
        const int8_t v34x1 = w34[1];
        const int8_t v34x2 = w34[2];
        const int8_t v34x3 = w34[3];
        ksum34 += (uint32_t) v34x0;
        ksum34 += (uint32_t) v34x1;
        ksum34 += (uint32_t) v34x2;
        ksum34 += (uint32_t) v34x3;
        out[136] = v34x0;
        out[137] = v34x1;
        out[138] = v34x2;
        out[139] = v34x3;
        w34 += 4;
        const int8_t v35x0 = w35[0];
        const int8_t v35x1 = w35[1];
        const int8_t v35x2 = w35[2];
        const int8_t v35x3 = w35[3];
        ksum35 += (uint32_t) v35x0;
        ksum35 += (uint32_t) v35x1;
        ksum35 += (uint32_t) v35x2;
        ksum35 += (uint32_t) v35x3;
        out[140] = v35x0;
        out[141] = v35x1;
        out[142] = v35x2;
        out[143] = v35x3;
        w35 += 4;
        const int8_t v36x0 = w36[0];
        const int8_t v36x1 = w36[1];
        const int8_t v36x2 = w36[2];
        const int8_t v36x3 = w36[3];
        ksum36 += (uint32_t) v36x0;
        ksum36 += (uint32_t) v36x1;
        ksum36 += (uint32_t) v36x2;
        ksum36 += (uint32_t) v36x3;
        out[144] = v36x0;
        out[145] = v36x1;
        out[146] = v36x2;
        out[147] = v36x3;
        w36 += 4;
        const int8_t v37x0 = w37[0];
        const int8_t v37x1 = w37[1];
        const int8_t v37x2 = w37[2];
        const int8_t v37x3 = w37[3];
        ksum37 += (uint32_t) v37x0;
        ksum37 += (uint32_t) v37x1;
        ksum37 += (uint32_t) v37x2;
        ksum37 += (uint32_t) v37x3;
        out[148] = v37x0;
        out[149] = v37x1;
        out[150] = v37x2;
        out[151] = v37x3;
        w37 += 4;
        const int8_t v38x0 = w38[0];
        const int8_t v38x1 = w38[1];
        const int8_t v38x2 = w38[2];
        const int8_t v38x3 = w38[3];
        ksum38 += (uint32_t) v38x0;
        ksum38 += (uint32_t) v38x1;
        ksum38 += (uint32_t) v38x2;
        ksum38 += (uint32_t) v38x3;
        out[152] = v38x0;
        out[153] = v38x1;
        out[154] = v38x2;
        out[155] = v38x3;
        w38 += 4;
        const int8_t v39x0 = w39[0];
        const int8_t v39x1 = w39[1];
        const int8_t v39x2 = w39[2];
        const int8_t v39x3 = w39[3];
        ksum39 += (uint32_t) v39x0;
        ksum39 += (uint32_t) v39x1;
        ksum39 += (uint32_t) v39x2;
        ksum39 += (uint32_t) v39x3;
        out[156] = v39x0;
        out[157] = v39x1;
        out[158] = v39x2;
        out[159] = v39x3;
        w39 += 4;
        const int8_t v40x0 = w40[0];
        const int8_t v40x1 = w40[1];
        const int8_t v40x2 = w40[2];
        const int8_t v40x3 = w40[3];
        ksum40 += (uint32_t) v40x0;
        ksum40 += (uint32_t) v40x1;
        ksum40 += (uint32_t) v40x2;
        ksum40 += (uint32_t) v40x3;
        out[160] = v40x0;
        out[161] = v40x1;
        out[162] = v40x2;
        out[163] = v40x3;
        w40 += 4;
        const int8_t v41x0 = w41[0];
        const int8_t v41x1 = w41[1];
        const int8_t v41x2 = w41[2];
        const int8_t v41x3 = w41[3];
        ksum41 += (uint32_t) v41x0;
        ksum41 += (uint32_t) v41x1;
        ksum41 += (uint32_t) v41x2;
        ksum41 += (uint32_t) v41x3;
        out[164] = v41x0;
        out[165] = v41x1;
        out[166] = v41x2;
        out[167] = v41x3;
        w41 += 4;
        const int8_t v42x0 = w42[0];
        const int8_t v42x1 = w42[1];
        const int8_t v42x2 = w42[2];
        const int8_t v42x3 = w42[3];
        ksum42 += (uint32_t) v42x0;
        ksum42 += (uint32_t) v42x1;
        ksum42 += (uint32_t) v42x2;
        ksum42 += (uint32_t) v42x3;
        out[168] = v42x0;
        out[169] = v42x1;
        out[170] = v42x2;
        out[171] = v42x3;
        w42 += 4;
        const int8_t v43x0 = w43[0];
        const int8_t v43x1 = w43[1];
        const int8_t v43x2 = w43[2];
        const int8_t v43x3 = w43[3];
        ksum43 += (uint32_t) v43x0;
        ksum43 += (uint32_t) v43x1;
        ksum43 += (uint32_t) v43x2;
        ksum43 += (uint32_t) v43x3;
        out[172] = v43x0;
        out[173] = v43x1;
        out[174] = v43x2;
        out[175] = v43x3;
        w43 += 4;
        const int8_t v44x0 = w44[0];
        const int8_t v44x1 = w44[1];
        const int8_t v44x2 = w44[2];
        const int8_t v44x3 = w44[3];
        ksum44 += (uint32_t) v44x0;
        ksum44 += (uint32_t) v44x1;
        ksum44 += (uint32_t) v44x2;
        ksum44 += (uint32_t) v44x3;
        out[176] = v44x0;
        out[177] = v44x1;
        out[178] = v44x2;
        out[179] = v44x3;
        w44 += 4;
        const int8_t v45x0 = w45[0];
        const int8_t v45x1 = w45[1];
        const int8_t v45x2 = w45[2];
        const int8_t v45x3 = w45[3];
        ksum45 += (uint32_t) v45x0;
        ksum45 += (uint32_t) v45x1;
        ksum45 += (uint32_t) v45x2;
        ksum45 += (uint32_t) v45x3;
        out[180] = v45x0;
        out[181] = v45x1;
        out[182] = v45x2;
        out[183] = v45x3;
        w45 += 4;
        const int8_t v46x0 = w46[0];
        const int8_t v46x1 = w46[1];
        const int8_t v46x2 = w46[2];
        const int8_t v46x3 = w46[3];
        ksum46 += (uint32_t) v46x0;
        ksum46 += (uint32_t) v46x1;
        ksum46 += (uint32_t) v46x2;
        ksum46 += (uint32_t) v46x3;
        out[184] = v46x0;
        out[185] = v46x1;
        out[186] = v46x2;
        out[187] = v46x3;
        w46 += 4;
        const int8_t v47x0 = w47[0];
        const int8_t v47x1 = w47[1];
        const int8_t v47x2 = w47[2];
        const int8_t v47x3 = w47[3];
        ksum47 += (uint32_t) v47x0;
        ksum47 += (uint32_t) v47x1;
        ksum47 += (uint32_t) v47x2;
        ksum47 += (uint32_t) v47x3;
        out[188] = v47x0;
        out[189] = v47x1;
        out[190] = v47x2;
        out[191] = v47x3;
        w47 += 4;
        const int8_t v48x0 = w48[0];
        const int8_t v48x1 = w48[1];
        const int8_t v48x2 = w48[2];
        const int8_t v48x3 = w48[3];
        ksum48 += (uint32_t) v48x0;
        ksum48 += (uint32_t) v48x1;
        ksum48 += (uint32_t) v48x2;
        ksum48 += (uint32_t) v48x3;
        out[192] = v48x0;
        out[193] = v48x1;
        out[194] = v48x2;
        out[195] = v48x3;
        w48 += 4;
        const int8_t v49x0 = w49[0];
        const int8_t v49x1 = w49[1];
        const int8_t v49x2 = w49[2];
        const int8_t v49x3 = w49[3];
        ksum49 += (uint32_t) v49x0;
        ksum49 += (uint32_t) v49x1;
        ksum49 += (uint32_t) v49x2;
        ksum49 += (uint32_t) v49x3;
        out[196] = v49x0;
        out[197] = v49x1;
        out[198] = v49x2;
        out[199] = v49x3;
        w49 += 4;
        const int8_t v50x0 = w50[0];
        const int8_t v50x1 = w50[1];
        const int8_t v50x2 = w50[2];
        const int8_t v50x3 = w50[3];
        ksum50 += (uint32_t) v50x0;
        ksum50 += (uint32_t) v50x1;
        ksum50 += (uint32_t) v50x2;
        ksum50 += (uint32_t) v50x3;
        out[200] = v50x0;
        out[201] = v50x1;
        out[202] = v50x2;
        out[203] = v50x3;
        w50 += 4;
        const int8_t v51x0 = w51[0];
        const int8_t v51x1 = w51[1];
        const int8_t v51x2 = w51[2];
        const int8_t v51x3 = w51[3];
        ksum51 += (uint32_t) v51x0;
        ksum51 += (uint32_t) v51x1;
        ksum51 += (uint32_t) v51x2;
        ksum51 += (uint32_t) v51x3;
        out[204] = v51x0;
        out[205] = v51x1;
        out[206] = v51x2;
        out[207] = v51x3;
        w51 += 4;
        const int8_t v52x0 = w52[0];
        const int8_t v52x1 = w52[1];
        const int8_t v52x2 = w52[2];
        const int8_t v52x3 = w52[3];
        ksum52 += (uint32_t) v52x0;
        ksum52 += (uint32_t) v52x1;
        ksum52 += (uint32_t) v52x2;
        ksum52 += (uint32_t) v52x3;
        out[208] = v52x0;
        out[209] = v52x1;
        out[210] = v52x2;
        out[211] = v52x3;
        w52 += 4;
        const int8_t v53x0 = w53[0];
        const int8_t v53x1 = w53[1];
        const int8_t v53x2 = w53[2];
        const int8_t v53x3 = w53[3];
        ksum53 += (uint32_t) v53x0;
        ksum53 += (uint32_t) v53x1;
        ksum53 += (uint32_t) v53x2;
        ksum53 += (uint32_t) v53x3;
        out[212] = v53x0;
        out[213] = v53x1;
        out[214] = v53x2;
        out[215] = v53x3;
        w53 += 4;
        const int8_t v54x0 = w54[0];
        const int8_t v54x1 = w54[1];
        const int8_t v54x2 = w54[2];
        const int8_t v54x3 = w54[3];
        ksum54 += (uint32_t) v54x0;
        ksum54 += (uint32_t) v54x1;
        ksum54 += (uint32_t) v54x2;
        ksum54 += (uint32_t) v54x3;
        out[216] = v54x0;
        out[217] = v54x1;
        out[218] = v54x2;
        out[219] = v54x3;
        w54 += 4;
        const int8_t v55x0 = w55[0];
        const int8_t v55x1 = w55[1];
        const int8_t v55x2 = w55[2];
        const int8_t v55x3 = w55[3];
        ksum55 += (uint32_t) v55x0;
        ksum55 += (uint32_t) v55x1;
        ksum55 += (uint32_t) v55x2;
        ksum55 += (uint32_t) v55x3;
        out[220] = v55x0;
        out[221] = v55x1;
        out[222] = v55x2;
        out[223] = v55x3;
        w55 += 4;
        const int8_t v56x0 = w56[0];
        const int8_t v56x1 = w56[1];
        const int8_t v56x2 = w56[2];
        const int8_t v56x3 = w56[3];
        ksum56 += (uint32_t) v56x0;
        ksum56 += (uint32_t) v56x1;
        ksum56 += (uint32_t) v56x2;
        ksum56 += (uint32_t) v56x3;
        out[224] = v56x0;
        out[225] = v56x1;
        out[226] = v56x2;
        out[227] = v56x3;
        w56 += 4;
        const int8_t v57x0 = w57[0];
        const int8_t v57x1 = w57[1];
        const int8_t v57x2 = w57[2];
        const int8_t v57x3 = w57[3];
        ksum57 += (uint32_t) v57x0;
        ksum57 += (uint32_t) v57x1;
        ksum57 += (uint32_t) v57x2;
        ksum57 += (uint32_t) v57x3;
        out[228] = v57x0;
        out[229] = v57x1;
        out[230] = v57x2;
        out[231] = v57x3;
        w57 += 4;
        const int8_t v58x0 = w58[0];
        const int8_t v58x1 = w58[1];
        const int8_t v58x2 = w58[2];
        const int8_t v58x3 = w58[3];
        ksum58 += (uint32_t) v58x0;
        ksum58 += (uint32_t) v58x1;
        ksum58 += (uint32_t) v58x2;
        ksum58 += (uint32_t) v58x3;
        out[232] = v58x0;
        out[233] = v58x1;
        out[234] = v58x2;
        out[235] = v58x3;
        w58 += 4;
        const int8_t v59x0 = w59[0];
        const int8_t v59x1 = w59[1];
        const int8_t v59x2 = w59[2];
        const int8_t v59x3 = w59[3];
        ksum59 += (uint32_t) v59x0;
        ksum59 += (uint32_t) v59x1;
        ksum59 += (uint32_t) v59x2;
        ksum59 += (uint32_t) v59x3;
        out[236] = v59x0;
        out[237] = v59x1;
        out[238] = v59x2;
        out[239] = v59x3;
        w59 += 4;
        const int8_t v60x0 = w60[0];
        const int8_t v60x1 = w60[1];
        const int8_t v60x2 = w60[2];
        const int8_t v60x3 = w60[3];
        ksum60 += (uint32_t) v60x0;
        ksum60 += (uint32_t) v60x1;
        ksum60 += (uint32_t) v60x2;
        ksum60 += (uint32_t) v60x3;
        out[240] = v60x0;
        out[241] = v60x1;
        out[242] = v60x2;
        out[243] = v60x3;
        w60 += 4;
        const int8_t v61x0 = w61[0];
        const int8_t v61x1 = w61[1];
        const int8_t v61x2 = w61[2];
        const int8_t v61x3 = w61[3];
        ksum61 += (uint32_t) v61x0;
        ksum61 += (uint32_t) v61x1;
        ksum61 += (uint32_t) v61x2;
        ksum61 += (uint32_t) v61x3;
        out[244] = v61x0;
        out[245] = v61x1;
        out[246] = v61x2;
        out[247] = v61x3;
        w61 += 4;
        const int8_t v62x0 = w62[0];
        const int8_t v62x1 = w62[1];
        const int8_t v62x2 = w62[2];
        const int8_t v62x3 = w62[3];
        ksum62 += (uint32_t) v62x0;
        ksum62 += (uint32_t) v62x1;
        ksum62 += (uint32_t) v62x2;
        ksum62 += (uint32_t) v62x3;
        out[248] = v62x0;
        out[249] = v62x1;
        out[250] = v62x2;
        out[251] = v62x3;
        w62 += 4;
        out += 256;
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
        const int8_t v16x0 = 0 < k ? w16[0] : izp;
        const int8_t v16x1 = 1 < k ? w16[1] : izp;
        const int8_t v16x2 = 2 < k ? w16[2] : izp;
        const int8_t v16x3 = 3 < k ? w16[3] : izp;
        ksum16 += (uint32_t) v16x0;
        ksum16 += (uint32_t) v16x1;
        ksum16 += (uint32_t) v16x2;
        ksum16 += (uint32_t) v16x3;
        if (0 < k) {
          out[64] = v16x0;
        }
        if (1 < k) {
          out[65] = v16x1;
        }
        if (2 < k) {
          out[66] = v16x2;
        }
        if (3 < k) {
          out[67] = v16x3;
        }
        w16 += 4;
        const int8_t v17x0 = 0 < k ? w17[0] : izp;
        const int8_t v17x1 = 1 < k ? w17[1] : izp;
        const int8_t v17x2 = 2 < k ? w17[2] : izp;
        const int8_t v17x3 = 3 < k ? w17[3] : izp;
        ksum17 += (uint32_t) v17x0;
        ksum17 += (uint32_t) v17x1;
        ksum17 += (uint32_t) v17x2;
        ksum17 += (uint32_t) v17x3;
        if (0 < k) {
          out[68] = v17x0;
        }
        if (1 < k) {
          out[69] = v17x1;
        }
        if (2 < k) {
          out[70] = v17x2;
        }
        if (3 < k) {
          out[71] = v17x3;
        }
        w17 += 4;
        const int8_t v18x0 = 0 < k ? w18[0] : izp;
        const int8_t v18x1 = 1 < k ? w18[1] : izp;
        const int8_t v18x2 = 2 < k ? w18[2] : izp;
        const int8_t v18x3 = 3 < k ? w18[3] : izp;
        ksum18 += (uint32_t) v18x0;
        ksum18 += (uint32_t) v18x1;
        ksum18 += (uint32_t) v18x2;
        ksum18 += (uint32_t) v18x3;
        if (0 < k) {
          out[72] = v18x0;
        }
        if (1 < k) {
          out[73] = v18x1;
        }
        if (2 < k) {
          out[74] = v18x2;
        }
        if (3 < k) {
          out[75] = v18x3;
        }
        w18 += 4;
        const int8_t v19x0 = 0 < k ? w19[0] : izp;
        const int8_t v19x1 = 1 < k ? w19[1] : izp;
        const int8_t v19x2 = 2 < k ? w19[2] : izp;
        const int8_t v19x3 = 3 < k ? w19[3] : izp;
        ksum19 += (uint32_t) v19x0;
        ksum19 += (uint32_t) v19x1;
        ksum19 += (uint32_t) v19x2;
        ksum19 += (uint32_t) v19x3;
        if (0 < k) {
          out[76] = v19x0;
        }
        if (1 < k) {
          out[77] = v19x1;
        }
        if (2 < k) {
          out[78] = v19x2;
        }
        if (3 < k) {
          out[79] = v19x3;
        }
        w19 += 4;
        const int8_t v20x0 = 0 < k ? w20[0] : izp;
        const int8_t v20x1 = 1 < k ? w20[1] : izp;
        const int8_t v20x2 = 2 < k ? w20[2] : izp;
        const int8_t v20x3 = 3 < k ? w20[3] : izp;
        ksum20 += (uint32_t) v20x0;
        ksum20 += (uint32_t) v20x1;
        ksum20 += (uint32_t) v20x2;
        ksum20 += (uint32_t) v20x3;
        if (0 < k) {
          out[80] = v20x0;
        }
        if (1 < k) {
          out[81] = v20x1;
        }
        if (2 < k) {
          out[82] = v20x2;
        }
        if (3 < k) {
          out[83] = v20x3;
        }
        w20 += 4;
        const int8_t v21x0 = 0 < k ? w21[0] : izp;
        const int8_t v21x1 = 1 < k ? w21[1] : izp;
        const int8_t v21x2 = 2 < k ? w21[2] : izp;
        const int8_t v21x3 = 3 < k ? w21[3] : izp;
        ksum21 += (uint32_t) v21x0;
        ksum21 += (uint32_t) v21x1;
        ksum21 += (uint32_t) v21x2;
        ksum21 += (uint32_t) v21x3;
        if (0 < k) {
          out[84] = v21x0;
        }
        if (1 < k) {
          out[85] = v21x1;
        }
        if (2 < k) {
          out[86] = v21x2;
        }
        if (3 < k) {
          out[87] = v21x3;
        }
        w21 += 4;
        const int8_t v22x0 = 0 < k ? w22[0] : izp;
        const int8_t v22x1 = 1 < k ? w22[1] : izp;
        const int8_t v22x2 = 2 < k ? w22[2] : izp;
        const int8_t v22x3 = 3 < k ? w22[3] : izp;
        ksum22 += (uint32_t) v22x0;
        ksum22 += (uint32_t) v22x1;
        ksum22 += (uint32_t) v22x2;
        ksum22 += (uint32_t) v22x3;
        if (0 < k) {
          out[88] = v22x0;
        }
        if (1 < k) {
          out[89] = v22x1;
        }
        if (2 < k) {
          out[90] = v22x2;
        }
        if (3 < k) {
          out[91] = v22x3;
        }
        w22 += 4;
        const int8_t v23x0 = 0 < k ? w23[0] : izp;
        const int8_t v23x1 = 1 < k ? w23[1] : izp;
        const int8_t v23x2 = 2 < k ? w23[2] : izp;
        const int8_t v23x3 = 3 < k ? w23[3] : izp;
        ksum23 += (uint32_t) v23x0;
        ksum23 += (uint32_t) v23x1;
        ksum23 += (uint32_t) v23x2;
        ksum23 += (uint32_t) v23x3;
        if (0 < k) {
          out[92] = v23x0;
        }
        if (1 < k) {
          out[93] = v23x1;
        }
        if (2 < k) {
          out[94] = v23x2;
        }
        if (3 < k) {
          out[95] = v23x3;
        }
        w23 += 4;
        const int8_t v24x0 = 0 < k ? w24[0] : izp;
        const int8_t v24x1 = 1 < k ? w24[1] : izp;
        const int8_t v24x2 = 2 < k ? w24[2] : izp;
        const int8_t v24x3 = 3 < k ? w24[3] : izp;
        ksum24 += (uint32_t) v24x0;
        ksum24 += (uint32_t) v24x1;
        ksum24 += (uint32_t) v24x2;
        ksum24 += (uint32_t) v24x3;
        if (0 < k) {
          out[96] = v24x0;
        }
        if (1 < k) {
          out[97] = v24x1;
        }
        if (2 < k) {
          out[98] = v24x2;
        }
        if (3 < k) {
          out[99] = v24x3;
        }
        w24 += 4;
        const int8_t v25x0 = 0 < k ? w25[0] : izp;
        const int8_t v25x1 = 1 < k ? w25[1] : izp;
        const int8_t v25x2 = 2 < k ? w25[2] : izp;
        const int8_t v25x3 = 3 < k ? w25[3] : izp;
        ksum25 += (uint32_t) v25x0;
        ksum25 += (uint32_t) v25x1;
        ksum25 += (uint32_t) v25x2;
        ksum25 += (uint32_t) v25x3;
        if (0 < k) {
          out[100] = v25x0;
        }
        if (1 < k) {
          out[101] = v25x1;
        }
        if (2 < k) {
          out[102] = v25x2;
        }
        if (3 < k) {
          out[103] = v25x3;
        }
        w25 += 4;
        const int8_t v26x0 = 0 < k ? w26[0] : izp;
        const int8_t v26x1 = 1 < k ? w26[1] : izp;
        const int8_t v26x2 = 2 < k ? w26[2] : izp;
        const int8_t v26x3 = 3 < k ? w26[3] : izp;
        ksum26 += (uint32_t) v26x0;
        ksum26 += (uint32_t) v26x1;
        ksum26 += (uint32_t) v26x2;
        ksum26 += (uint32_t) v26x3;
        if (0 < k) {
          out[104] = v26x0;
        }
        if (1 < k) {
          out[105] = v26x1;
        }
        if (2 < k) {
          out[106] = v26x2;
        }
        if (3 < k) {
          out[107] = v26x3;
        }
        w26 += 4;
        const int8_t v27x0 = 0 < k ? w27[0] : izp;
        const int8_t v27x1 = 1 < k ? w27[1] : izp;
        const int8_t v27x2 = 2 < k ? w27[2] : izp;
        const int8_t v27x3 = 3 < k ? w27[3] : izp;
        ksum27 += (uint32_t) v27x0;
        ksum27 += (uint32_t) v27x1;
        ksum27 += (uint32_t) v27x2;
        ksum27 += (uint32_t) v27x3;
        if (0 < k) {
          out[108] = v27x0;
        }
        if (1 < k) {
          out[109] = v27x1;
        }
        if (2 < k) {
          out[110] = v27x2;
        }
        if (3 < k) {
          out[111] = v27x3;
        }
        w27 += 4;
        const int8_t v28x0 = 0 < k ? w28[0] : izp;
        const int8_t v28x1 = 1 < k ? w28[1] : izp;
        const int8_t v28x2 = 2 < k ? w28[2] : izp;
        const int8_t v28x3 = 3 < k ? w28[3] : izp;
        ksum28 += (uint32_t) v28x0;
        ksum28 += (uint32_t) v28x1;
        ksum28 += (uint32_t) v28x2;
        ksum28 += (uint32_t) v28x3;
        if (0 < k) {
          out[112] = v28x0;
        }
        if (1 < k) {
          out[113] = v28x1;
        }
        if (2 < k) {
          out[114] = v28x2;
        }
        if (3 < k) {
          out[115] = v28x3;
        }
        w28 += 4;
        const int8_t v29x0 = 0 < k ? w29[0] : izp;
        const int8_t v29x1 = 1 < k ? w29[1] : izp;
        const int8_t v29x2 = 2 < k ? w29[2] : izp;
        const int8_t v29x3 = 3 < k ? w29[3] : izp;
        ksum29 += (uint32_t) v29x0;
        ksum29 += (uint32_t) v29x1;
        ksum29 += (uint32_t) v29x2;
        ksum29 += (uint32_t) v29x3;
        if (0 < k) {
          out[116] = v29x0;
        }
        if (1 < k) {
          out[117] = v29x1;
        }
        if (2 < k) {
          out[118] = v29x2;
        }
        if (3 < k) {
          out[119] = v29x3;
        }
        w29 += 4;
        const int8_t v30x0 = 0 < k ? w30[0] : izp;
        const int8_t v30x1 = 1 < k ? w30[1] : izp;
        const int8_t v30x2 = 2 < k ? w30[2] : izp;
        const int8_t v30x3 = 3 < k ? w30[3] : izp;
        ksum30 += (uint32_t) v30x0;
        ksum30 += (uint32_t) v30x1;
        ksum30 += (uint32_t) v30x2;
        ksum30 += (uint32_t) v30x3;
        if (0 < k) {
          out[120] = v30x0;
        }
        if (1 < k) {
          out[121] = v30x1;
        }
        if (2 < k) {
          out[122] = v30x2;
        }
        if (3 < k) {
          out[123] = v30x3;
        }
        w30 += 4;
        const int8_t v31x0 = 0 < k ? w31[0] : izp;
        const int8_t v31x1 = 1 < k ? w31[1] : izp;
        const int8_t v31x2 = 2 < k ? w31[2] : izp;
        const int8_t v31x3 = 3 < k ? w31[3] : izp;
        ksum31 += (uint32_t) v31x0;
        ksum31 += (uint32_t) v31x1;
        ksum31 += (uint32_t) v31x2;
        ksum31 += (uint32_t) v31x3;
        if (0 < k) {
          out[124] = v31x0;
        }
        if (1 < k) {
          out[125] = v31x1;
        }
        if (2 < k) {
          out[126] = v31x2;
        }
        if (3 < k) {
          out[127] = v31x3;
        }
        w31 += 4;
        const int8_t v32x0 = 0 < k ? w32[0] : izp;
        const int8_t v32x1 = 1 < k ? w32[1] : izp;
        const int8_t v32x2 = 2 < k ? w32[2] : izp;
        const int8_t v32x3 = 3 < k ? w32[3] : izp;
        ksum32 += (uint32_t) v32x0;
        ksum32 += (uint32_t) v32x1;
        ksum32 += (uint32_t) v32x2;
        ksum32 += (uint32_t) v32x3;
        if (0 < k) {
          out[128] = v32x0;
        }
        if (1 < k) {
          out[129] = v32x1;
        }
        if (2 < k) {
          out[130] = v32x2;
        }
        if (3 < k) {
          out[131] = v32x3;
        }
        w32 += 4;
        const int8_t v33x0 = 0 < k ? w33[0] : izp;
        const int8_t v33x1 = 1 < k ? w33[1] : izp;
        const int8_t v33x2 = 2 < k ? w33[2] : izp;
        const int8_t v33x3 = 3 < k ? w33[3] : izp;
        ksum33 += (uint32_t) v33x0;
        ksum33 += (uint32_t) v33x1;
        ksum33 += (uint32_t) v33x2;
        ksum33 += (uint32_t) v33x3;
        if (0 < k) {
          out[132] = v33x0;
        }
        if (1 < k) {
          out[133] = v33x1;
        }
        if (2 < k) {
          out[134] = v33x2;
        }
        if (3 < k) {
          out[135] = v33x3;
        }
        w33 += 4;
        const int8_t v34x0 = 0 < k ? w34[0] : izp;
        const int8_t v34x1 = 1 < k ? w34[1] : izp;
        const int8_t v34x2 = 2 < k ? w34[2] : izp;
        const int8_t v34x3 = 3 < k ? w34[3] : izp;
        ksum34 += (uint32_t) v34x0;
        ksum34 += (uint32_t) v34x1;
        ksum34 += (uint32_t) v34x2;
        ksum34 += (uint32_t) v34x3;
        if (0 < k) {
          out[136] = v34x0;
        }
        if (1 < k) {
          out[137] = v34x1;
        }
        if (2 < k) {
          out[138] = v34x2;
        }
        if (3 < k) {
          out[139] = v34x3;
        }
        w34 += 4;
        const int8_t v35x0 = 0 < k ? w35[0] : izp;
        const int8_t v35x1 = 1 < k ? w35[1] : izp;
        const int8_t v35x2 = 2 < k ? w35[2] : izp;
        const int8_t v35x3 = 3 < k ? w35[3] : izp;
        ksum35 += (uint32_t) v35x0;
        ksum35 += (uint32_t) v35x1;
        ksum35 += (uint32_t) v35x2;
        ksum35 += (uint32_t) v35x3;
        if (0 < k) {
          out[140] = v35x0;
        }
        if (1 < k) {
          out[141] = v35x1;
        }
        if (2 < k) {
          out[142] = v35x2;
        }
        if (3 < k) {
          out[143] = v35x3;
        }
        w35 += 4;
        const int8_t v36x0 = 0 < k ? w36[0] : izp;
        const int8_t v36x1 = 1 < k ? w36[1] : izp;
        const int8_t v36x2 = 2 < k ? w36[2] : izp;
        const int8_t v36x3 = 3 < k ? w36[3] : izp;
        ksum36 += (uint32_t) v36x0;
        ksum36 += (uint32_t) v36x1;
        ksum36 += (uint32_t) v36x2;
        ksum36 += (uint32_t) v36x3;
        if (0 < k) {
          out[144] = v36x0;
        }
        if (1 < k) {
          out[145] = v36x1;
        }
        if (2 < k) {
          out[146] = v36x2;
        }
        if (3 < k) {
          out[147] = v36x3;
        }
        w36 += 4;
        const int8_t v37x0 = 0 < k ? w37[0] : izp;
        const int8_t v37x1 = 1 < k ? w37[1] : izp;
        const int8_t v37x2 = 2 < k ? w37[2] : izp;
        const int8_t v37x3 = 3 < k ? w37[3] : izp;
        ksum37 += (uint32_t) v37x0;
        ksum37 += (uint32_t) v37x1;
        ksum37 += (uint32_t) v37x2;
        ksum37 += (uint32_t) v37x3;
        if (0 < k) {
          out[148] = v37x0;
        }
        if (1 < k) {
          out[149] = v37x1;
        }
        if (2 < k) {
          out[150] = v37x2;
        }
        if (3 < k) {
          out[151] = v37x3;
        }
        w37 += 4;
        const int8_t v38x0 = 0 < k ? w38[0] : izp;
        const int8_t v38x1 = 1 < k ? w38[1] : izp;
        const int8_t v38x2 = 2 < k ? w38[2] : izp;
        const int8_t v38x3 = 3 < k ? w38[3] : izp;
        ksum38 += (uint32_t) v38x0;
        ksum38 += (uint32_t) v38x1;
        ksum38 += (uint32_t) v38x2;
        ksum38 += (uint32_t) v38x3;
        if (0 < k) {
          out[152] = v38x0;
        }
        if (1 < k) {
          out[153] = v38x1;
        }
        if (2 < k) {
          out[154] = v38x2;
        }
        if (3 < k) {
          out[155] = v38x3;
        }
        w38 += 4;
        const int8_t v39x0 = 0 < k ? w39[0] : izp;
        const int8_t v39x1 = 1 < k ? w39[1] : izp;
        const int8_t v39x2 = 2 < k ? w39[2] : izp;
        const int8_t v39x3 = 3 < k ? w39[3] : izp;
        ksum39 += (uint32_t) v39x0;
        ksum39 += (uint32_t) v39x1;
        ksum39 += (uint32_t) v39x2;
        ksum39 += (uint32_t) v39x3;
        if (0 < k) {
          out[156] = v39x0;
        }
        if (1 < k) {
          out[157] = v39x1;
        }
        if (2 < k) {
          out[158] = v39x2;
        }
        if (3 < k) {
          out[159] = v39x3;
        }
        w39 += 4;
        const int8_t v40x0 = 0 < k ? w40[0] : izp;
        const int8_t v40x1 = 1 < k ? w40[1] : izp;
        const int8_t v40x2 = 2 < k ? w40[2] : izp;
        const int8_t v40x3 = 3 < k ? w40[3] : izp;
        ksum40 += (uint32_t) v40x0;
        ksum40 += (uint32_t) v40x1;
        ksum40 += (uint32_t) v40x2;
        ksum40 += (uint32_t) v40x3;
        if (0 < k) {
          out[160] = v40x0;
        }
        if (1 < k) {
          out[161] = v40x1;
        }
        if (2 < k) {
          out[162] = v40x2;
        }
        if (3 < k) {
          out[163] = v40x3;
        }
        w40 += 4;
        const int8_t v41x0 = 0 < k ? w41[0] : izp;
        const int8_t v41x1 = 1 < k ? w41[1] : izp;
        const int8_t v41x2 = 2 < k ? w41[2] : izp;
        const int8_t v41x3 = 3 < k ? w41[3] : izp;
        ksum41 += (uint32_t) v41x0;
        ksum41 += (uint32_t) v41x1;
        ksum41 += (uint32_t) v41x2;
        ksum41 += (uint32_t) v41x3;
        if (0 < k) {
          out[164] = v41x0;
        }
        if (1 < k) {
          out[165] = v41x1;
        }
        if (2 < k) {
          out[166] = v41x2;
        }
        if (3 < k) {
          out[167] = v41x3;
        }
        w41 += 4;
        const int8_t v42x0 = 0 < k ? w42[0] : izp;
        const int8_t v42x1 = 1 < k ? w42[1] : izp;
        const int8_t v42x2 = 2 < k ? w42[2] : izp;
        const int8_t v42x3 = 3 < k ? w42[3] : izp;
        ksum42 += (uint32_t) v42x0;
        ksum42 += (uint32_t) v42x1;
        ksum42 += (uint32_t) v42x2;
        ksum42 += (uint32_t) v42x3;
        if (0 < k) {
          out[168] = v42x0;
        }
        if (1 < k) {
          out[169] = v42x1;
        }
        if (2 < k) {
          out[170] = v42x2;
        }
        if (3 < k) {
          out[171] = v42x3;
        }
        w42 += 4;
        const int8_t v43x0 = 0 < k ? w43[0] : izp;
        const int8_t v43x1 = 1 < k ? w43[1] : izp;
        const int8_t v43x2 = 2 < k ? w43[2] : izp;
        const int8_t v43x3 = 3 < k ? w43[3] : izp;
        ksum43 += (uint32_t) v43x0;
        ksum43 += (uint32_t) v43x1;
        ksum43 += (uint32_t) v43x2;
        ksum43 += (uint32_t) v43x3;
        if (0 < k) {
          out[172] = v43x0;
        }
        if (1 < k) {
          out[173] = v43x1;
        }
        if (2 < k) {
          out[174] = v43x2;
        }
        if (3 < k) {
          out[175] = v43x3;
        }
        w43 += 4;
        const int8_t v44x0 = 0 < k ? w44[0] : izp;
        const int8_t v44x1 = 1 < k ? w44[1] : izp;
        const int8_t v44x2 = 2 < k ? w44[2] : izp;
        const int8_t v44x3 = 3 < k ? w44[3] : izp;
        ksum44 += (uint32_t) v44x0;
        ksum44 += (uint32_t) v44x1;
        ksum44 += (uint32_t) v44x2;
        ksum44 += (uint32_t) v44x3;
        if (0 < k) {
          out[176] = v44x0;
        }
        if (1 < k) {
          out[177] = v44x1;
        }
        if (2 < k) {
          out[178] = v44x2;
        }
        if (3 < k) {
          out[179] = v44x3;
        }
        w44 += 4;
        const int8_t v45x0 = 0 < k ? w45[0] : izp;
        const int8_t v45x1 = 1 < k ? w45[1] : izp;
        const int8_t v45x2 = 2 < k ? w45[2] : izp;
        const int8_t v45x3 = 3 < k ? w45[3] : izp;
        ksum45 += (uint32_t) v45x0;
        ksum45 += (uint32_t) v45x1;
        ksum45 += (uint32_t) v45x2;
        ksum45 += (uint32_t) v45x3;
        if (0 < k) {
          out[180] = v45x0;
        }
        if (1 < k) {
          out[181] = v45x1;
        }
        if (2 < k) {
          out[182] = v45x2;
        }
        if (3 < k) {
          out[183] = v45x3;
        }
        w45 += 4;
        const int8_t v46x0 = 0 < k ? w46[0] : izp;
        const int8_t v46x1 = 1 < k ? w46[1] : izp;
        const int8_t v46x2 = 2 < k ? w46[2] : izp;
        const int8_t v46x3 = 3 < k ? w46[3] : izp;
        ksum46 += (uint32_t) v46x0;
        ksum46 += (uint32_t) v46x1;
        ksum46 += (uint32_t) v46x2;
        ksum46 += (uint32_t) v46x3;
        if (0 < k) {
          out[184] = v46x0;
        }
        if (1 < k) {
          out[185] = v46x1;
        }
        if (2 < k) {
          out[186] = v46x2;
        }
        if (3 < k) {
          out[187] = v46x3;
        }
        w46 += 4;
        const int8_t v47x0 = 0 < k ? w47[0] : izp;
        const int8_t v47x1 = 1 < k ? w47[1] : izp;
        const int8_t v47x2 = 2 < k ? w47[2] : izp;
        const int8_t v47x3 = 3 < k ? w47[3] : izp;
        ksum47 += (uint32_t) v47x0;
        ksum47 += (uint32_t) v47x1;
        ksum47 += (uint32_t) v47x2;
        ksum47 += (uint32_t) v47x3;
        if (0 < k) {
          out[188] = v47x0;
        }
        if (1 < k) {
          out[189] = v47x1;
        }
        if (2 < k) {
          out[190] = v47x2;
        }
        if (3 < k) {
          out[191] = v47x3;
        }
        w47 += 4;
        const int8_t v48x0 = 0 < k ? w48[0] : izp;
        const int8_t v48x1 = 1 < k ? w48[1] : izp;
        const int8_t v48x2 = 2 < k ? w48[2] : izp;
        const int8_t v48x3 = 3 < k ? w48[3] : izp;
        ksum48 += (uint32_t) v48x0;
        ksum48 += (uint32_t) v48x1;
        ksum48 += (uint32_t) v48x2;
        ksum48 += (uint32_t) v48x3;
        if (0 < k) {
          out[192] = v48x0;
        }
        if (1 < k) {
          out[193] = v48x1;
        }
        if (2 < k) {
          out[194] = v48x2;
        }
        if (3 < k) {
          out[195] = v48x3;
        }
        w48 += 4;
        const int8_t v49x0 = 0 < k ? w49[0] : izp;
        const int8_t v49x1 = 1 < k ? w49[1] : izp;
        const int8_t v49x2 = 2 < k ? w49[2] : izp;
        const int8_t v49x3 = 3 < k ? w49[3] : izp;
        ksum49 += (uint32_t) v49x0;
        ksum49 += (uint32_t) v49x1;
        ksum49 += (uint32_t) v49x2;
        ksum49 += (uint32_t) v49x3;
        if (0 < k) {
          out[196] = v49x0;
        }
        if (1 < k) {
          out[197] = v49x1;
        }
        if (2 < k) {
          out[198] = v49x2;
        }
        if (3 < k) {
          out[199] = v49x3;
        }
        w49 += 4;
        const int8_t v50x0 = 0 < k ? w50[0] : izp;
        const int8_t v50x1 = 1 < k ? w50[1] : izp;
        const int8_t v50x2 = 2 < k ? w50[2] : izp;
        const int8_t v50x3 = 3 < k ? w50[3] : izp;
        ksum50 += (uint32_t) v50x0;
        ksum50 += (uint32_t) v50x1;
        ksum50 += (uint32_t) v50x2;
        ksum50 += (uint32_t) v50x3;
        if (0 < k) {
          out[200] = v50x0;
        }
        if (1 < k) {
          out[201] = v50x1;
        }
        if (2 < k) {
          out[202] = v50x2;
        }
        if (3 < k) {
          out[203] = v50x3;
        }
        w50 += 4;
        const int8_t v51x0 = 0 < k ? w51[0] : izp;
        const int8_t v51x1 = 1 < k ? w51[1] : izp;
        const int8_t v51x2 = 2 < k ? w51[2] : izp;
        const int8_t v51x3 = 3 < k ? w51[3] : izp;
        ksum51 += (uint32_t) v51x0;
        ksum51 += (uint32_t) v51x1;
        ksum51 += (uint32_t) v51x2;
        ksum51 += (uint32_t) v51x3;
        if (0 < k) {
          out[204] = v51x0;
        }
        if (1 < k) {
          out[205] = v51x1;
        }
        if (2 < k) {
          out[206] = v51x2;
        }
        if (3 < k) {
          out[207] = v51x3;
        }
        w51 += 4;
        const int8_t v52x0 = 0 < k ? w52[0] : izp;
        const int8_t v52x1 = 1 < k ? w52[1] : izp;
        const int8_t v52x2 = 2 < k ? w52[2] : izp;
        const int8_t v52x3 = 3 < k ? w52[3] : izp;
        ksum52 += (uint32_t) v52x0;
        ksum52 += (uint32_t) v52x1;
        ksum52 += (uint32_t) v52x2;
        ksum52 += (uint32_t) v52x3;
        if (0 < k) {
          out[208] = v52x0;
        }
        if (1 < k) {
          out[209] = v52x1;
        }
        if (2 < k) {
          out[210] = v52x2;
        }
        if (3 < k) {
          out[211] = v52x3;
        }
        w52 += 4;
        const int8_t v53x0 = 0 < k ? w53[0] : izp;
        const int8_t v53x1 = 1 < k ? w53[1] : izp;
        const int8_t v53x2 = 2 < k ? w53[2] : izp;
        const int8_t v53x3 = 3 < k ? w53[3] : izp;
        ksum53 += (uint32_t) v53x0;
        ksum53 += (uint32_t) v53x1;
        ksum53 += (uint32_t) v53x2;
        ksum53 += (uint32_t) v53x3;
        if (0 < k) {
          out[212] = v53x0;
        }
        if (1 < k) {
          out[213] = v53x1;
        }
        if (2 < k) {
          out[214] = v53x2;
        }
        if (3 < k) {
          out[215] = v53x3;
        }
        w53 += 4;
        const int8_t v54x0 = 0 < k ? w54[0] : izp;
        const int8_t v54x1 = 1 < k ? w54[1] : izp;
        const int8_t v54x2 = 2 < k ? w54[2] : izp;
        const int8_t v54x3 = 3 < k ? w54[3] : izp;
        ksum54 += (uint32_t) v54x0;
        ksum54 += (uint32_t) v54x1;
        ksum54 += (uint32_t) v54x2;
        ksum54 += (uint32_t) v54x3;
        if (0 < k) {
          out[216] = v54x0;
        }
        if (1 < k) {
          out[217] = v54x1;
        }
        if (2 < k) {
          out[218] = v54x2;
        }
        if (3 < k) {
          out[219] = v54x3;
        }
        w54 += 4;
        const int8_t v55x0 = 0 < k ? w55[0] : izp;
        const int8_t v55x1 = 1 < k ? w55[1] : izp;
        const int8_t v55x2 = 2 < k ? w55[2] : izp;
        const int8_t v55x3 = 3 < k ? w55[3] : izp;
        ksum55 += (uint32_t) v55x0;
        ksum55 += (uint32_t) v55x1;
        ksum55 += (uint32_t) v55x2;
        ksum55 += (uint32_t) v55x3;
        if (0 < k) {
          out[220] = v55x0;
        }
        if (1 < k) {
          out[221] = v55x1;
        }
        if (2 < k) {
          out[222] = v55x2;
        }
        if (3 < k) {
          out[223] = v55x3;
        }
        w55 += 4;
        const int8_t v56x0 = 0 < k ? w56[0] : izp;
        const int8_t v56x1 = 1 < k ? w56[1] : izp;
        const int8_t v56x2 = 2 < k ? w56[2] : izp;
        const int8_t v56x3 = 3 < k ? w56[3] : izp;
        ksum56 += (uint32_t) v56x0;
        ksum56 += (uint32_t) v56x1;
        ksum56 += (uint32_t) v56x2;
        ksum56 += (uint32_t) v56x3;
        if (0 < k) {
          out[224] = v56x0;
        }
        if (1 < k) {
          out[225] = v56x1;
        }
        if (2 < k) {
          out[226] = v56x2;
        }
        if (3 < k) {
          out[227] = v56x3;
        }
        w56 += 4;
        const int8_t v57x0 = 0 < k ? w57[0] : izp;
        const int8_t v57x1 = 1 < k ? w57[1] : izp;
        const int8_t v57x2 = 2 < k ? w57[2] : izp;
        const int8_t v57x3 = 3 < k ? w57[3] : izp;
        ksum57 += (uint32_t) v57x0;
        ksum57 += (uint32_t) v57x1;
        ksum57 += (uint32_t) v57x2;
        ksum57 += (uint32_t) v57x3;
        if (0 < k) {
          out[228] = v57x0;
        }
        if (1 < k) {
          out[229] = v57x1;
        }
        if (2 < k) {
          out[230] = v57x2;
        }
        if (3 < k) {
          out[231] = v57x3;
        }
        w57 += 4;
        const int8_t v58x0 = 0 < k ? w58[0] : izp;
        const int8_t v58x1 = 1 < k ? w58[1] : izp;
        const int8_t v58x2 = 2 < k ? w58[2] : izp;
        const int8_t v58x3 = 3 < k ? w58[3] : izp;
        ksum58 += (uint32_t) v58x0;
        ksum58 += (uint32_t) v58x1;
        ksum58 += (uint32_t) v58x2;
        ksum58 += (uint32_t) v58x3;
        if (0 < k) {
          out[232] = v58x0;
        }
        if (1 < k) {
          out[233] = v58x1;
        }
        if (2 < k) {
          out[234] = v58x2;
        }
        if (3 < k) {
          out[235] = v58x3;
        }
        w58 += 4;
        const int8_t v59x0 = 0 < k ? w59[0] : izp;
        const int8_t v59x1 = 1 < k ? w59[1] : izp;
        const int8_t v59x2 = 2 < k ? w59[2] : izp;
        const int8_t v59x3 = 3 < k ? w59[3] : izp;
        ksum59 += (uint32_t) v59x0;
        ksum59 += (uint32_t) v59x1;
        ksum59 += (uint32_t) v59x2;
        ksum59 += (uint32_t) v59x3;
        if (0 < k) {
          out[236] = v59x0;
        }
        if (1 < k) {
          out[237] = v59x1;
        }
        if (2 < k) {
          out[238] = v59x2;
        }
        if (3 < k) {
          out[239] = v59x3;
        }
        w59 += 4;
        const int8_t v60x0 = 0 < k ? w60[0] : izp;
        const int8_t v60x1 = 1 < k ? w60[1] : izp;
        const int8_t v60x2 = 2 < k ? w60[2] : izp;
        const int8_t v60x3 = 3 < k ? w60[3] : izp;
        ksum60 += (uint32_t) v60x0;
        ksum60 += (uint32_t) v60x1;
        ksum60 += (uint32_t) v60x2;
        ksum60 += (uint32_t) v60x3;
        if (0 < k) {
          out[240] = v60x0;
        }
        if (1 < k) {
          out[241] = v60x1;
        }
        if (2 < k) {
          out[242] = v60x2;
        }
        if (3 < k) {
          out[243] = v60x3;
        }
        w60 += 4;
        const int8_t v61x0 = 0 < k ? w61[0] : izp;
        const int8_t v61x1 = 1 < k ? w61[1] : izp;
        const int8_t v61x2 = 2 < k ? w61[2] : izp;
        const int8_t v61x3 = 3 < k ? w61[3] : izp;
        ksum61 += (uint32_t) v61x0;
        ksum61 += (uint32_t) v61x1;
        ksum61 += (uint32_t) v61x2;
        ksum61 += (uint32_t) v61x3;
        if (0 < k) {
          out[244] = v61x0;
        }
        if (1 < k) {
          out[245] = v61x1;
        }
        if (2 < k) {
          out[246] = v61x2;
        }
        if (3 < k) {
          out[247] = v61x3;
        }
        w61 += 4;
        const int8_t v62x0 = 0 < k ? w62[0] : izp;
        const int8_t v62x1 = 1 < k ? w62[1] : izp;
        const int8_t v62x2 = 2 < k ? w62[2] : izp;
        const int8_t v62x3 = 3 < k ? w62[3] : izp;
        ksum62 += (uint32_t) v62x0;
        ksum62 += (uint32_t) v62x1;
        ksum62 += (uint32_t) v62x2;
        ksum62 += (uint32_t) v62x3;
        if (0 < k) {
          out[248] = v62x0;
        }
        if (1 < k) {
          out[249] = v62x1;
        }
        if (2 < k) {
          out[250] = v62x2;
        }
        if (3 < k) {
          out[251] = v62x3;
        }
        w62 += 4;
        out += 256;
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
      packed_b[16] -= ksum16 * izp;
      packed_b[17] -= ksum17 * izp;
      packed_b[18] -= ksum18 * izp;
      packed_b[19] -= ksum19 * izp;
      packed_b[20] -= ksum20 * izp;
      packed_b[21] -= ksum21 * izp;
      packed_b[22] -= ksum22 * izp;
      packed_b[23] -= ksum23 * izp;
      packed_b[24] -= ksum24 * izp;
      packed_b[25] -= ksum25 * izp;
      packed_b[26] -= ksum26 * izp;
      packed_b[27] -= ksum27 * izp;
      packed_b[28] -= ksum28 * izp;
      packed_b[29] -= ksum29 * izp;
      packed_b[30] -= ksum30 * izp;
      packed_b[31] -= ksum31 * izp;
      packed_b[32] -= ksum32 * izp;
      packed_b[33] -= ksum33 * izp;
      packed_b[34] -= ksum34 * izp;
      packed_b[35] -= ksum35 * izp;
      packed_b[36] -= ksum36 * izp;
      packed_b[37] -= ksum37 * izp;
      packed_b[38] -= ksum38 * izp;
      packed_b[39] -= ksum39 * izp;
      packed_b[40] -= ksum40 * izp;
      packed_b[41] -= ksum41 * izp;
      packed_b[42] -= ksum42 * izp;
      packed_b[43] -= ksum43 * izp;
      packed_b[44] -= ksum44 * izp;
      packed_b[45] -= ksum45 * izp;
      packed_b[46] -= ksum46 * izp;
      packed_b[47] -= ksum47 * izp;
      packed_b[48] -= ksum48 * izp;
      packed_b[49] -= ksum49 * izp;
      packed_b[50] -= ksum50 * izp;
      packed_b[51] -= ksum51 * izp;
      packed_b[52] -= ksum52 * izp;
      packed_b[53] -= ksum53 * izp;
      packed_b[54] -= ksum54 * izp;
      packed_b[55] -= ksum55 * izp;
      packed_b[56] -= ksum56 * izp;
      packed_b[57] -= ksum57 * izp;
      packed_b[58] -= ksum58 * izp;
      packed_b[59] -= ksum59 * izp;
      packed_b[60] -= ksum60 * izp;
      packed_b[61] -= ksum61 * izp;
      packed_b[62] -= ksum62 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
