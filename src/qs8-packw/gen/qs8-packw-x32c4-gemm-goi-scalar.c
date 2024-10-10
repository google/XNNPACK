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

void xnn_qs8_packw_gemm_goi_ukernel_x32c4__scalar(
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
  assert(nr == 32);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 0): 0);

  do {
    // NC main loop multiple of 32
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 32; n -= 32) {
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
        b += 32;
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
      }
      out += 32 * sizeof(int32_t);

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

      // KC main loop multiple of 32x4
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
        out += 128;
      }

      // KC remainder of 1..3
      if (k != 0) {
        assert(k >= 1 && k <= 3);
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
        w0 += k;
        const int8_t v1x0 = w1[0];
        ksum1 += (uint32_t) v1x0;
        out[4] = v1x0;
        if (1 < k) {
          const int8_t v1x1 = w1[1];
          ksum1 += (uint32_t) v1x1;
          out[5] = v1x1;
        }
        if (2 < k) {
          const int8_t v1x2 = w1[2];
          ksum1 += (uint32_t) v1x2;
          out[6] = v1x2;
        }
        if (3 < k) {
          const int8_t v1x3 = w1[3];
          ksum1 += (uint32_t) v1x3;
          out[7] = v1x3;
        }
        w1 += k;
        const int8_t v2x0 = w2[0];
        ksum2 += (uint32_t) v2x0;
        out[8] = v2x0;
        if (1 < k) {
          const int8_t v2x1 = w2[1];
          ksum2 += (uint32_t) v2x1;
          out[9] = v2x1;
        }
        if (2 < k) {
          const int8_t v2x2 = w2[2];
          ksum2 += (uint32_t) v2x2;
          out[10] = v2x2;
        }
        if (3 < k) {
          const int8_t v2x3 = w2[3];
          ksum2 += (uint32_t) v2x3;
          out[11] = v2x3;
        }
        w2 += k;
        const int8_t v3x0 = w3[0];
        ksum3 += (uint32_t) v3x0;
        out[12] = v3x0;
        if (1 < k) {
          const int8_t v3x1 = w3[1];
          ksum3 += (uint32_t) v3x1;
          out[13] = v3x1;
        }
        if (2 < k) {
          const int8_t v3x2 = w3[2];
          ksum3 += (uint32_t) v3x2;
          out[14] = v3x2;
        }
        if (3 < k) {
          const int8_t v3x3 = w3[3];
          ksum3 += (uint32_t) v3x3;
          out[15] = v3x3;
        }
        w3 += k;
        const int8_t v4x0 = w4[0];
        ksum4 += (uint32_t) v4x0;
        out[16] = v4x0;
        if (1 < k) {
          const int8_t v4x1 = w4[1];
          ksum4 += (uint32_t) v4x1;
          out[17] = v4x1;
        }
        if (2 < k) {
          const int8_t v4x2 = w4[2];
          ksum4 += (uint32_t) v4x2;
          out[18] = v4x2;
        }
        if (3 < k) {
          const int8_t v4x3 = w4[3];
          ksum4 += (uint32_t) v4x3;
          out[19] = v4x3;
        }
        w4 += k;
        const int8_t v5x0 = w5[0];
        ksum5 += (uint32_t) v5x0;
        out[20] = v5x0;
        if (1 < k) {
          const int8_t v5x1 = w5[1];
          ksum5 += (uint32_t) v5x1;
          out[21] = v5x1;
        }
        if (2 < k) {
          const int8_t v5x2 = w5[2];
          ksum5 += (uint32_t) v5x2;
          out[22] = v5x2;
        }
        if (3 < k) {
          const int8_t v5x3 = w5[3];
          ksum5 += (uint32_t) v5x3;
          out[23] = v5x3;
        }
        w5 += k;
        const int8_t v6x0 = w6[0];
        ksum6 += (uint32_t) v6x0;
        out[24] = v6x0;
        if (1 < k) {
          const int8_t v6x1 = w6[1];
          ksum6 += (uint32_t) v6x1;
          out[25] = v6x1;
        }
        if (2 < k) {
          const int8_t v6x2 = w6[2];
          ksum6 += (uint32_t) v6x2;
          out[26] = v6x2;
        }
        if (3 < k) {
          const int8_t v6x3 = w6[3];
          ksum6 += (uint32_t) v6x3;
          out[27] = v6x3;
        }
        w6 += k;
        const int8_t v7x0 = w7[0];
        ksum7 += (uint32_t) v7x0;
        out[28] = v7x0;
        if (1 < k) {
          const int8_t v7x1 = w7[1];
          ksum7 += (uint32_t) v7x1;
          out[29] = v7x1;
        }
        if (2 < k) {
          const int8_t v7x2 = w7[2];
          ksum7 += (uint32_t) v7x2;
          out[30] = v7x2;
        }
        if (3 < k) {
          const int8_t v7x3 = w7[3];
          ksum7 += (uint32_t) v7x3;
          out[31] = v7x3;
        }
        w7 += k;
        const int8_t v8x0 = w8[0];
        ksum8 += (uint32_t) v8x0;
        out[32] = v8x0;
        if (1 < k) {
          const int8_t v8x1 = w8[1];
          ksum8 += (uint32_t) v8x1;
          out[33] = v8x1;
        }
        if (2 < k) {
          const int8_t v8x2 = w8[2];
          ksum8 += (uint32_t) v8x2;
          out[34] = v8x2;
        }
        if (3 < k) {
          const int8_t v8x3 = w8[3];
          ksum8 += (uint32_t) v8x3;
          out[35] = v8x3;
        }
        w8 += k;
        const int8_t v9x0 = w9[0];
        ksum9 += (uint32_t) v9x0;
        out[36] = v9x0;
        if (1 < k) {
          const int8_t v9x1 = w9[1];
          ksum9 += (uint32_t) v9x1;
          out[37] = v9x1;
        }
        if (2 < k) {
          const int8_t v9x2 = w9[2];
          ksum9 += (uint32_t) v9x2;
          out[38] = v9x2;
        }
        if (3 < k) {
          const int8_t v9x3 = w9[3];
          ksum9 += (uint32_t) v9x3;
          out[39] = v9x3;
        }
        w9 += k;
        const int8_t v10x0 = w10[0];
        ksum10 += (uint32_t) v10x0;
        out[40] = v10x0;
        if (1 < k) {
          const int8_t v10x1 = w10[1];
          ksum10 += (uint32_t) v10x1;
          out[41] = v10x1;
        }
        if (2 < k) {
          const int8_t v10x2 = w10[2];
          ksum10 += (uint32_t) v10x2;
          out[42] = v10x2;
        }
        if (3 < k) {
          const int8_t v10x3 = w10[3];
          ksum10 += (uint32_t) v10x3;
          out[43] = v10x3;
        }
        w10 += k;
        const int8_t v11x0 = w11[0];
        ksum11 += (uint32_t) v11x0;
        out[44] = v11x0;
        if (1 < k) {
          const int8_t v11x1 = w11[1];
          ksum11 += (uint32_t) v11x1;
          out[45] = v11x1;
        }
        if (2 < k) {
          const int8_t v11x2 = w11[2];
          ksum11 += (uint32_t) v11x2;
          out[46] = v11x2;
        }
        if (3 < k) {
          const int8_t v11x3 = w11[3];
          ksum11 += (uint32_t) v11x3;
          out[47] = v11x3;
        }
        w11 += k;
        const int8_t v12x0 = w12[0];
        ksum12 += (uint32_t) v12x0;
        out[48] = v12x0;
        if (1 < k) {
          const int8_t v12x1 = w12[1];
          ksum12 += (uint32_t) v12x1;
          out[49] = v12x1;
        }
        if (2 < k) {
          const int8_t v12x2 = w12[2];
          ksum12 += (uint32_t) v12x2;
          out[50] = v12x2;
        }
        if (3 < k) {
          const int8_t v12x3 = w12[3];
          ksum12 += (uint32_t) v12x3;
          out[51] = v12x3;
        }
        w12 += k;
        const int8_t v13x0 = w13[0];
        ksum13 += (uint32_t) v13x0;
        out[52] = v13x0;
        if (1 < k) {
          const int8_t v13x1 = w13[1];
          ksum13 += (uint32_t) v13x1;
          out[53] = v13x1;
        }
        if (2 < k) {
          const int8_t v13x2 = w13[2];
          ksum13 += (uint32_t) v13x2;
          out[54] = v13x2;
        }
        if (3 < k) {
          const int8_t v13x3 = w13[3];
          ksum13 += (uint32_t) v13x3;
          out[55] = v13x3;
        }
        w13 += k;
        const int8_t v14x0 = w14[0];
        ksum14 += (uint32_t) v14x0;
        out[56] = v14x0;
        if (1 < k) {
          const int8_t v14x1 = w14[1];
          ksum14 += (uint32_t) v14x1;
          out[57] = v14x1;
        }
        if (2 < k) {
          const int8_t v14x2 = w14[2];
          ksum14 += (uint32_t) v14x2;
          out[58] = v14x2;
        }
        if (3 < k) {
          const int8_t v14x3 = w14[3];
          ksum14 += (uint32_t) v14x3;
          out[59] = v14x3;
        }
        w14 += k;
        const int8_t v15x0 = w15[0];
        ksum15 += (uint32_t) v15x0;
        out[60] = v15x0;
        if (1 < k) {
          const int8_t v15x1 = w15[1];
          ksum15 += (uint32_t) v15x1;
          out[61] = v15x1;
        }
        if (2 < k) {
          const int8_t v15x2 = w15[2];
          ksum15 += (uint32_t) v15x2;
          out[62] = v15x2;
        }
        if (3 < k) {
          const int8_t v15x3 = w15[3];
          ksum15 += (uint32_t) v15x3;
          out[63] = v15x3;
        }
        w15 += k;
        const int8_t v16x0 = w16[0];
        ksum16 += (uint32_t) v16x0;
        out[64] = v16x0;
        if (1 < k) {
          const int8_t v16x1 = w16[1];
          ksum16 += (uint32_t) v16x1;
          out[65] = v16x1;
        }
        if (2 < k) {
          const int8_t v16x2 = w16[2];
          ksum16 += (uint32_t) v16x2;
          out[66] = v16x2;
        }
        if (3 < k) {
          const int8_t v16x3 = w16[3];
          ksum16 += (uint32_t) v16x3;
          out[67] = v16x3;
        }
        w16 += k;
        const int8_t v17x0 = w17[0];
        ksum17 += (uint32_t) v17x0;
        out[68] = v17x0;
        if (1 < k) {
          const int8_t v17x1 = w17[1];
          ksum17 += (uint32_t) v17x1;
          out[69] = v17x1;
        }
        if (2 < k) {
          const int8_t v17x2 = w17[2];
          ksum17 += (uint32_t) v17x2;
          out[70] = v17x2;
        }
        if (3 < k) {
          const int8_t v17x3 = w17[3];
          ksum17 += (uint32_t) v17x3;
          out[71] = v17x3;
        }
        w17 += k;
        const int8_t v18x0 = w18[0];
        ksum18 += (uint32_t) v18x0;
        out[72] = v18x0;
        if (1 < k) {
          const int8_t v18x1 = w18[1];
          ksum18 += (uint32_t) v18x1;
          out[73] = v18x1;
        }
        if (2 < k) {
          const int8_t v18x2 = w18[2];
          ksum18 += (uint32_t) v18x2;
          out[74] = v18x2;
        }
        if (3 < k) {
          const int8_t v18x3 = w18[3];
          ksum18 += (uint32_t) v18x3;
          out[75] = v18x3;
        }
        w18 += k;
        const int8_t v19x0 = w19[0];
        ksum19 += (uint32_t) v19x0;
        out[76] = v19x0;
        if (1 < k) {
          const int8_t v19x1 = w19[1];
          ksum19 += (uint32_t) v19x1;
          out[77] = v19x1;
        }
        if (2 < k) {
          const int8_t v19x2 = w19[2];
          ksum19 += (uint32_t) v19x2;
          out[78] = v19x2;
        }
        if (3 < k) {
          const int8_t v19x3 = w19[3];
          ksum19 += (uint32_t) v19x3;
          out[79] = v19x3;
        }
        w19 += k;
        const int8_t v20x0 = w20[0];
        ksum20 += (uint32_t) v20x0;
        out[80] = v20x0;
        if (1 < k) {
          const int8_t v20x1 = w20[1];
          ksum20 += (uint32_t) v20x1;
          out[81] = v20x1;
        }
        if (2 < k) {
          const int8_t v20x2 = w20[2];
          ksum20 += (uint32_t) v20x2;
          out[82] = v20x2;
        }
        if (3 < k) {
          const int8_t v20x3 = w20[3];
          ksum20 += (uint32_t) v20x3;
          out[83] = v20x3;
        }
        w20 += k;
        const int8_t v21x0 = w21[0];
        ksum21 += (uint32_t) v21x0;
        out[84] = v21x0;
        if (1 < k) {
          const int8_t v21x1 = w21[1];
          ksum21 += (uint32_t) v21x1;
          out[85] = v21x1;
        }
        if (2 < k) {
          const int8_t v21x2 = w21[2];
          ksum21 += (uint32_t) v21x2;
          out[86] = v21x2;
        }
        if (3 < k) {
          const int8_t v21x3 = w21[3];
          ksum21 += (uint32_t) v21x3;
          out[87] = v21x3;
        }
        w21 += k;
        const int8_t v22x0 = w22[0];
        ksum22 += (uint32_t) v22x0;
        out[88] = v22x0;
        if (1 < k) {
          const int8_t v22x1 = w22[1];
          ksum22 += (uint32_t) v22x1;
          out[89] = v22x1;
        }
        if (2 < k) {
          const int8_t v22x2 = w22[2];
          ksum22 += (uint32_t) v22x2;
          out[90] = v22x2;
        }
        if (3 < k) {
          const int8_t v22x3 = w22[3];
          ksum22 += (uint32_t) v22x3;
          out[91] = v22x3;
        }
        w22 += k;
        const int8_t v23x0 = w23[0];
        ksum23 += (uint32_t) v23x0;
        out[92] = v23x0;
        if (1 < k) {
          const int8_t v23x1 = w23[1];
          ksum23 += (uint32_t) v23x1;
          out[93] = v23x1;
        }
        if (2 < k) {
          const int8_t v23x2 = w23[2];
          ksum23 += (uint32_t) v23x2;
          out[94] = v23x2;
        }
        if (3 < k) {
          const int8_t v23x3 = w23[3];
          ksum23 += (uint32_t) v23x3;
          out[95] = v23x3;
        }
        w23 += k;
        const int8_t v24x0 = w24[0];
        ksum24 += (uint32_t) v24x0;
        out[96] = v24x0;
        if (1 < k) {
          const int8_t v24x1 = w24[1];
          ksum24 += (uint32_t) v24x1;
          out[97] = v24x1;
        }
        if (2 < k) {
          const int8_t v24x2 = w24[2];
          ksum24 += (uint32_t) v24x2;
          out[98] = v24x2;
        }
        if (3 < k) {
          const int8_t v24x3 = w24[3];
          ksum24 += (uint32_t) v24x3;
          out[99] = v24x3;
        }
        w24 += k;
        const int8_t v25x0 = w25[0];
        ksum25 += (uint32_t) v25x0;
        out[100] = v25x0;
        if (1 < k) {
          const int8_t v25x1 = w25[1];
          ksum25 += (uint32_t) v25x1;
          out[101] = v25x1;
        }
        if (2 < k) {
          const int8_t v25x2 = w25[2];
          ksum25 += (uint32_t) v25x2;
          out[102] = v25x2;
        }
        if (3 < k) {
          const int8_t v25x3 = w25[3];
          ksum25 += (uint32_t) v25x3;
          out[103] = v25x3;
        }
        w25 += k;
        const int8_t v26x0 = w26[0];
        ksum26 += (uint32_t) v26x0;
        out[104] = v26x0;
        if (1 < k) {
          const int8_t v26x1 = w26[1];
          ksum26 += (uint32_t) v26x1;
          out[105] = v26x1;
        }
        if (2 < k) {
          const int8_t v26x2 = w26[2];
          ksum26 += (uint32_t) v26x2;
          out[106] = v26x2;
        }
        if (3 < k) {
          const int8_t v26x3 = w26[3];
          ksum26 += (uint32_t) v26x3;
          out[107] = v26x3;
        }
        w26 += k;
        const int8_t v27x0 = w27[0];
        ksum27 += (uint32_t) v27x0;
        out[108] = v27x0;
        if (1 < k) {
          const int8_t v27x1 = w27[1];
          ksum27 += (uint32_t) v27x1;
          out[109] = v27x1;
        }
        if (2 < k) {
          const int8_t v27x2 = w27[2];
          ksum27 += (uint32_t) v27x2;
          out[110] = v27x2;
        }
        if (3 < k) {
          const int8_t v27x3 = w27[3];
          ksum27 += (uint32_t) v27x3;
          out[111] = v27x3;
        }
        w27 += k;
        const int8_t v28x0 = w28[0];
        ksum28 += (uint32_t) v28x0;
        out[112] = v28x0;
        if (1 < k) {
          const int8_t v28x1 = w28[1];
          ksum28 += (uint32_t) v28x1;
          out[113] = v28x1;
        }
        if (2 < k) {
          const int8_t v28x2 = w28[2];
          ksum28 += (uint32_t) v28x2;
          out[114] = v28x2;
        }
        if (3 < k) {
          const int8_t v28x3 = w28[3];
          ksum28 += (uint32_t) v28x3;
          out[115] = v28x3;
        }
        w28 += k;
        const int8_t v29x0 = w29[0];
        ksum29 += (uint32_t) v29x0;
        out[116] = v29x0;
        if (1 < k) {
          const int8_t v29x1 = w29[1];
          ksum29 += (uint32_t) v29x1;
          out[117] = v29x1;
        }
        if (2 < k) {
          const int8_t v29x2 = w29[2];
          ksum29 += (uint32_t) v29x2;
          out[118] = v29x2;
        }
        if (3 < k) {
          const int8_t v29x3 = w29[3];
          ksum29 += (uint32_t) v29x3;
          out[119] = v29x3;
        }
        w29 += k;
        const int8_t v30x0 = w30[0];
        ksum30 += (uint32_t) v30x0;
        out[120] = v30x0;
        if (1 < k) {
          const int8_t v30x1 = w30[1];
          ksum30 += (uint32_t) v30x1;
          out[121] = v30x1;
        }
        if (2 < k) {
          const int8_t v30x2 = w30[2];
          ksum30 += (uint32_t) v30x2;
          out[122] = v30x2;
        }
        if (3 < k) {
          const int8_t v30x3 = w30[3];
          ksum30 += (uint32_t) v30x3;
          out[123] = v30x3;
        }
        w30 += k;
        const int8_t v31x0 = w31[0];
        ksum31 += (uint32_t) v31x0;
        out[124] = v31x0;
        if (1 < k) {
          const int8_t v31x1 = w31[1];
          ksum31 += (uint32_t) v31x1;
          out[125] = v31x1;
        }
        if (2 < k) {
          const int8_t v31x2 = w31[2];
          ksum31 += (uint32_t) v31x2;
          out[126] = v31x2;
        }
        if (3 < k) {
          const int8_t v31x3 = w31[3];
          ksum31 += (uint32_t) v31x3;
          out[127] = v31x3;
        }
        w31 += k;
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
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w31;
    }

    // NC remainder (1..31)
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
      out += (32 - n) * sizeof(int32_t);

     // NR remainder has less than 32 rows so last row is not loaded
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

      // KC main loop multiple of 32x4
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
        out += 128;
      }

      // KC remainder of 1..3
      if (k != 0) {
        assert(k >= 1 && k <= 3);
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
        w0 += k;
        const int8_t v1x0 = w1[0];
        ksum1 += (uint32_t) v1x0;
        out[4] = v1x0;
        if (1 < k) {
          const int8_t v1x1 = w1[1];
          ksum1 += (uint32_t) v1x1;
          out[5] = v1x1;
        }
        if (2 < k) {
          const int8_t v1x2 = w1[2];
          ksum1 += (uint32_t) v1x2;
          out[6] = v1x2;
        }
        if (3 < k) {
          const int8_t v1x3 = w1[3];
          ksum1 += (uint32_t) v1x3;
          out[7] = v1x3;
        }
        w1 += k;
        const int8_t v2x0 = w2[0];
        ksum2 += (uint32_t) v2x0;
        out[8] = v2x0;
        if (1 < k) {
          const int8_t v2x1 = w2[1];
          ksum2 += (uint32_t) v2x1;
          out[9] = v2x1;
        }
        if (2 < k) {
          const int8_t v2x2 = w2[2];
          ksum2 += (uint32_t) v2x2;
          out[10] = v2x2;
        }
        if (3 < k) {
          const int8_t v2x3 = w2[3];
          ksum2 += (uint32_t) v2x3;
          out[11] = v2x3;
        }
        w2 += k;
        const int8_t v3x0 = w3[0];
        ksum3 += (uint32_t) v3x0;
        out[12] = v3x0;
        if (1 < k) {
          const int8_t v3x1 = w3[1];
          ksum3 += (uint32_t) v3x1;
          out[13] = v3x1;
        }
        if (2 < k) {
          const int8_t v3x2 = w3[2];
          ksum3 += (uint32_t) v3x2;
          out[14] = v3x2;
        }
        if (3 < k) {
          const int8_t v3x3 = w3[3];
          ksum3 += (uint32_t) v3x3;
          out[15] = v3x3;
        }
        w3 += k;
        const int8_t v4x0 = w4[0];
        ksum4 += (uint32_t) v4x0;
        out[16] = v4x0;
        if (1 < k) {
          const int8_t v4x1 = w4[1];
          ksum4 += (uint32_t) v4x1;
          out[17] = v4x1;
        }
        if (2 < k) {
          const int8_t v4x2 = w4[2];
          ksum4 += (uint32_t) v4x2;
          out[18] = v4x2;
        }
        if (3 < k) {
          const int8_t v4x3 = w4[3];
          ksum4 += (uint32_t) v4x3;
          out[19] = v4x3;
        }
        w4 += k;
        const int8_t v5x0 = w5[0];
        ksum5 += (uint32_t) v5x0;
        out[20] = v5x0;
        if (1 < k) {
          const int8_t v5x1 = w5[1];
          ksum5 += (uint32_t) v5x1;
          out[21] = v5x1;
        }
        if (2 < k) {
          const int8_t v5x2 = w5[2];
          ksum5 += (uint32_t) v5x2;
          out[22] = v5x2;
        }
        if (3 < k) {
          const int8_t v5x3 = w5[3];
          ksum5 += (uint32_t) v5x3;
          out[23] = v5x3;
        }
        w5 += k;
        const int8_t v6x0 = w6[0];
        ksum6 += (uint32_t) v6x0;
        out[24] = v6x0;
        if (1 < k) {
          const int8_t v6x1 = w6[1];
          ksum6 += (uint32_t) v6x1;
          out[25] = v6x1;
        }
        if (2 < k) {
          const int8_t v6x2 = w6[2];
          ksum6 += (uint32_t) v6x2;
          out[26] = v6x2;
        }
        if (3 < k) {
          const int8_t v6x3 = w6[3];
          ksum6 += (uint32_t) v6x3;
          out[27] = v6x3;
        }
        w6 += k;
        const int8_t v7x0 = w7[0];
        ksum7 += (uint32_t) v7x0;
        out[28] = v7x0;
        if (1 < k) {
          const int8_t v7x1 = w7[1];
          ksum7 += (uint32_t) v7x1;
          out[29] = v7x1;
        }
        if (2 < k) {
          const int8_t v7x2 = w7[2];
          ksum7 += (uint32_t) v7x2;
          out[30] = v7x2;
        }
        if (3 < k) {
          const int8_t v7x3 = w7[3];
          ksum7 += (uint32_t) v7x3;
          out[31] = v7x3;
        }
        w7 += k;
        const int8_t v8x0 = w8[0];
        ksum8 += (uint32_t) v8x0;
        out[32] = v8x0;
        if (1 < k) {
          const int8_t v8x1 = w8[1];
          ksum8 += (uint32_t) v8x1;
          out[33] = v8x1;
        }
        if (2 < k) {
          const int8_t v8x2 = w8[2];
          ksum8 += (uint32_t) v8x2;
          out[34] = v8x2;
        }
        if (3 < k) {
          const int8_t v8x3 = w8[3];
          ksum8 += (uint32_t) v8x3;
          out[35] = v8x3;
        }
        w8 += k;
        const int8_t v9x0 = w9[0];
        ksum9 += (uint32_t) v9x0;
        out[36] = v9x0;
        if (1 < k) {
          const int8_t v9x1 = w9[1];
          ksum9 += (uint32_t) v9x1;
          out[37] = v9x1;
        }
        if (2 < k) {
          const int8_t v9x2 = w9[2];
          ksum9 += (uint32_t) v9x2;
          out[38] = v9x2;
        }
        if (3 < k) {
          const int8_t v9x3 = w9[3];
          ksum9 += (uint32_t) v9x3;
          out[39] = v9x3;
        }
        w9 += k;
        const int8_t v10x0 = w10[0];
        ksum10 += (uint32_t) v10x0;
        out[40] = v10x0;
        if (1 < k) {
          const int8_t v10x1 = w10[1];
          ksum10 += (uint32_t) v10x1;
          out[41] = v10x1;
        }
        if (2 < k) {
          const int8_t v10x2 = w10[2];
          ksum10 += (uint32_t) v10x2;
          out[42] = v10x2;
        }
        if (3 < k) {
          const int8_t v10x3 = w10[3];
          ksum10 += (uint32_t) v10x3;
          out[43] = v10x3;
        }
        w10 += k;
        const int8_t v11x0 = w11[0];
        ksum11 += (uint32_t) v11x0;
        out[44] = v11x0;
        if (1 < k) {
          const int8_t v11x1 = w11[1];
          ksum11 += (uint32_t) v11x1;
          out[45] = v11x1;
        }
        if (2 < k) {
          const int8_t v11x2 = w11[2];
          ksum11 += (uint32_t) v11x2;
          out[46] = v11x2;
        }
        if (3 < k) {
          const int8_t v11x3 = w11[3];
          ksum11 += (uint32_t) v11x3;
          out[47] = v11x3;
        }
        w11 += k;
        const int8_t v12x0 = w12[0];
        ksum12 += (uint32_t) v12x0;
        out[48] = v12x0;
        if (1 < k) {
          const int8_t v12x1 = w12[1];
          ksum12 += (uint32_t) v12x1;
          out[49] = v12x1;
        }
        if (2 < k) {
          const int8_t v12x2 = w12[2];
          ksum12 += (uint32_t) v12x2;
          out[50] = v12x2;
        }
        if (3 < k) {
          const int8_t v12x3 = w12[3];
          ksum12 += (uint32_t) v12x3;
          out[51] = v12x3;
        }
        w12 += k;
        const int8_t v13x0 = w13[0];
        ksum13 += (uint32_t) v13x0;
        out[52] = v13x0;
        if (1 < k) {
          const int8_t v13x1 = w13[1];
          ksum13 += (uint32_t) v13x1;
          out[53] = v13x1;
        }
        if (2 < k) {
          const int8_t v13x2 = w13[2];
          ksum13 += (uint32_t) v13x2;
          out[54] = v13x2;
        }
        if (3 < k) {
          const int8_t v13x3 = w13[3];
          ksum13 += (uint32_t) v13x3;
          out[55] = v13x3;
        }
        w13 += k;
        const int8_t v14x0 = w14[0];
        ksum14 += (uint32_t) v14x0;
        out[56] = v14x0;
        if (1 < k) {
          const int8_t v14x1 = w14[1];
          ksum14 += (uint32_t) v14x1;
          out[57] = v14x1;
        }
        if (2 < k) {
          const int8_t v14x2 = w14[2];
          ksum14 += (uint32_t) v14x2;
          out[58] = v14x2;
        }
        if (3 < k) {
          const int8_t v14x3 = w14[3];
          ksum14 += (uint32_t) v14x3;
          out[59] = v14x3;
        }
        w14 += k;
        const int8_t v15x0 = w15[0];
        ksum15 += (uint32_t) v15x0;
        out[60] = v15x0;
        if (1 < k) {
          const int8_t v15x1 = w15[1];
          ksum15 += (uint32_t) v15x1;
          out[61] = v15x1;
        }
        if (2 < k) {
          const int8_t v15x2 = w15[2];
          ksum15 += (uint32_t) v15x2;
          out[62] = v15x2;
        }
        if (3 < k) {
          const int8_t v15x3 = w15[3];
          ksum15 += (uint32_t) v15x3;
          out[63] = v15x3;
        }
        w15 += k;
        const int8_t v16x0 = w16[0];
        ksum16 += (uint32_t) v16x0;
        out[64] = v16x0;
        if (1 < k) {
          const int8_t v16x1 = w16[1];
          ksum16 += (uint32_t) v16x1;
          out[65] = v16x1;
        }
        if (2 < k) {
          const int8_t v16x2 = w16[2];
          ksum16 += (uint32_t) v16x2;
          out[66] = v16x2;
        }
        if (3 < k) {
          const int8_t v16x3 = w16[3];
          ksum16 += (uint32_t) v16x3;
          out[67] = v16x3;
        }
        w16 += k;
        const int8_t v17x0 = w17[0];
        ksum17 += (uint32_t) v17x0;
        out[68] = v17x0;
        if (1 < k) {
          const int8_t v17x1 = w17[1];
          ksum17 += (uint32_t) v17x1;
          out[69] = v17x1;
        }
        if (2 < k) {
          const int8_t v17x2 = w17[2];
          ksum17 += (uint32_t) v17x2;
          out[70] = v17x2;
        }
        if (3 < k) {
          const int8_t v17x3 = w17[3];
          ksum17 += (uint32_t) v17x3;
          out[71] = v17x3;
        }
        w17 += k;
        const int8_t v18x0 = w18[0];
        ksum18 += (uint32_t) v18x0;
        out[72] = v18x0;
        if (1 < k) {
          const int8_t v18x1 = w18[1];
          ksum18 += (uint32_t) v18x1;
          out[73] = v18x1;
        }
        if (2 < k) {
          const int8_t v18x2 = w18[2];
          ksum18 += (uint32_t) v18x2;
          out[74] = v18x2;
        }
        if (3 < k) {
          const int8_t v18x3 = w18[3];
          ksum18 += (uint32_t) v18x3;
          out[75] = v18x3;
        }
        w18 += k;
        const int8_t v19x0 = w19[0];
        ksum19 += (uint32_t) v19x0;
        out[76] = v19x0;
        if (1 < k) {
          const int8_t v19x1 = w19[1];
          ksum19 += (uint32_t) v19x1;
          out[77] = v19x1;
        }
        if (2 < k) {
          const int8_t v19x2 = w19[2];
          ksum19 += (uint32_t) v19x2;
          out[78] = v19x2;
        }
        if (3 < k) {
          const int8_t v19x3 = w19[3];
          ksum19 += (uint32_t) v19x3;
          out[79] = v19x3;
        }
        w19 += k;
        const int8_t v20x0 = w20[0];
        ksum20 += (uint32_t) v20x0;
        out[80] = v20x0;
        if (1 < k) {
          const int8_t v20x1 = w20[1];
          ksum20 += (uint32_t) v20x1;
          out[81] = v20x1;
        }
        if (2 < k) {
          const int8_t v20x2 = w20[2];
          ksum20 += (uint32_t) v20x2;
          out[82] = v20x2;
        }
        if (3 < k) {
          const int8_t v20x3 = w20[3];
          ksum20 += (uint32_t) v20x3;
          out[83] = v20x3;
        }
        w20 += k;
        const int8_t v21x0 = w21[0];
        ksum21 += (uint32_t) v21x0;
        out[84] = v21x0;
        if (1 < k) {
          const int8_t v21x1 = w21[1];
          ksum21 += (uint32_t) v21x1;
          out[85] = v21x1;
        }
        if (2 < k) {
          const int8_t v21x2 = w21[2];
          ksum21 += (uint32_t) v21x2;
          out[86] = v21x2;
        }
        if (3 < k) {
          const int8_t v21x3 = w21[3];
          ksum21 += (uint32_t) v21x3;
          out[87] = v21x3;
        }
        w21 += k;
        const int8_t v22x0 = w22[0];
        ksum22 += (uint32_t) v22x0;
        out[88] = v22x0;
        if (1 < k) {
          const int8_t v22x1 = w22[1];
          ksum22 += (uint32_t) v22x1;
          out[89] = v22x1;
        }
        if (2 < k) {
          const int8_t v22x2 = w22[2];
          ksum22 += (uint32_t) v22x2;
          out[90] = v22x2;
        }
        if (3 < k) {
          const int8_t v22x3 = w22[3];
          ksum22 += (uint32_t) v22x3;
          out[91] = v22x3;
        }
        w22 += k;
        const int8_t v23x0 = w23[0];
        ksum23 += (uint32_t) v23x0;
        out[92] = v23x0;
        if (1 < k) {
          const int8_t v23x1 = w23[1];
          ksum23 += (uint32_t) v23x1;
          out[93] = v23x1;
        }
        if (2 < k) {
          const int8_t v23x2 = w23[2];
          ksum23 += (uint32_t) v23x2;
          out[94] = v23x2;
        }
        if (3 < k) {
          const int8_t v23x3 = w23[3];
          ksum23 += (uint32_t) v23x3;
          out[95] = v23x3;
        }
        w23 += k;
        const int8_t v24x0 = w24[0];
        ksum24 += (uint32_t) v24x0;
        out[96] = v24x0;
        if (1 < k) {
          const int8_t v24x1 = w24[1];
          ksum24 += (uint32_t) v24x1;
          out[97] = v24x1;
        }
        if (2 < k) {
          const int8_t v24x2 = w24[2];
          ksum24 += (uint32_t) v24x2;
          out[98] = v24x2;
        }
        if (3 < k) {
          const int8_t v24x3 = w24[3];
          ksum24 += (uint32_t) v24x3;
          out[99] = v24x3;
        }
        w24 += k;
        const int8_t v25x0 = w25[0];
        ksum25 += (uint32_t) v25x0;
        out[100] = v25x0;
        if (1 < k) {
          const int8_t v25x1 = w25[1];
          ksum25 += (uint32_t) v25x1;
          out[101] = v25x1;
        }
        if (2 < k) {
          const int8_t v25x2 = w25[2];
          ksum25 += (uint32_t) v25x2;
          out[102] = v25x2;
        }
        if (3 < k) {
          const int8_t v25x3 = w25[3];
          ksum25 += (uint32_t) v25x3;
          out[103] = v25x3;
        }
        w25 += k;
        const int8_t v26x0 = w26[0];
        ksum26 += (uint32_t) v26x0;
        out[104] = v26x0;
        if (1 < k) {
          const int8_t v26x1 = w26[1];
          ksum26 += (uint32_t) v26x1;
          out[105] = v26x1;
        }
        if (2 < k) {
          const int8_t v26x2 = w26[2];
          ksum26 += (uint32_t) v26x2;
          out[106] = v26x2;
        }
        if (3 < k) {
          const int8_t v26x3 = w26[3];
          ksum26 += (uint32_t) v26x3;
          out[107] = v26x3;
        }
        w26 += k;
        const int8_t v27x0 = w27[0];
        ksum27 += (uint32_t) v27x0;
        out[108] = v27x0;
        if (1 < k) {
          const int8_t v27x1 = w27[1];
          ksum27 += (uint32_t) v27x1;
          out[109] = v27x1;
        }
        if (2 < k) {
          const int8_t v27x2 = w27[2];
          ksum27 += (uint32_t) v27x2;
          out[110] = v27x2;
        }
        if (3 < k) {
          const int8_t v27x3 = w27[3];
          ksum27 += (uint32_t) v27x3;
          out[111] = v27x3;
        }
        w27 += k;
        const int8_t v28x0 = w28[0];
        ksum28 += (uint32_t) v28x0;
        out[112] = v28x0;
        if (1 < k) {
          const int8_t v28x1 = w28[1];
          ksum28 += (uint32_t) v28x1;
          out[113] = v28x1;
        }
        if (2 < k) {
          const int8_t v28x2 = w28[2];
          ksum28 += (uint32_t) v28x2;
          out[114] = v28x2;
        }
        if (3 < k) {
          const int8_t v28x3 = w28[3];
          ksum28 += (uint32_t) v28x3;
          out[115] = v28x3;
        }
        w28 += k;
        const int8_t v29x0 = w29[0];
        ksum29 += (uint32_t) v29x0;
        out[116] = v29x0;
        if (1 < k) {
          const int8_t v29x1 = w29[1];
          ksum29 += (uint32_t) v29x1;
          out[117] = v29x1;
        }
        if (2 < k) {
          const int8_t v29x2 = w29[2];
          ksum29 += (uint32_t) v29x2;
          out[118] = v29x2;
        }
        if (3 < k) {
          const int8_t v29x3 = w29[3];
          ksum29 += (uint32_t) v29x3;
          out[119] = v29x3;
        }
        w29 += k;
        const int8_t v30x0 = w30[0];
        ksum30 += (uint32_t) v30x0;
        out[120] = v30x0;
        if (1 < k) {
          const int8_t v30x1 = w30[1];
          ksum30 += (uint32_t) v30x1;
          out[121] = v30x1;
        }
        if (2 < k) {
          const int8_t v30x2 = w30[2];
          ksum30 += (uint32_t) v30x2;
          out[122] = v30x2;
        }
        if (3 < k) {
          const int8_t v30x3 = w30[3];
          ksum30 += (uint32_t) v30x3;
          out[123] = v30x3;
        }
        w30 += k;
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
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
