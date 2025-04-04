// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-packw/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/packw.h"



void xnn_x32_packw_gemm_goi_ukernel_x32__scalar_int_u2(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 32);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint32_t* out = (uint32_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    // NC main loop multiple of 32
    const uint32_t* w0 = (const uint32_t*) weights;
    size_t n = nc;
    for (;n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        out[0] = b[0];
        out[1] = b[1];
        out[2] = b[2];
        out[3] = b[3];
        out[4] = b[4];
        out[5] = b[5];
        out[6] = b[6];
        out[7] = b[7];
        out[8] = b[8];
        out[9] = b[9];
        out[10] = b[10];
        out[11] = b[11];
        out[12] = b[12];
        out[13] = b[13];
        out[14] = b[14];
        out[15] = b[15];
        out[16] = b[16];
        out[17] = b[17];
        out[18] = b[18];
        out[19] = b[19];
        out[20] = b[20];
        out[21] = b[21];
        out[22] = b[22];
        out[23] = b[23];
        out[24] = b[24];
        out[25] = b[25];
        out[26] = b[26];
        out[27] = b[27];
        out[28] = b[28];
        out[29] = b[29];
        out[30] = b[30];
        out[31] = b[31];
        b += 32;
      } else {
        out[0] = 0;
        out[1] = 0;
        out[2] = 0;
        out[3] = 0;
        out[4] = 0;
        out[5] = 0;
        out[6] = 0;
        out[7] = 0;
        out[8] = 0;
        out[9] = 0;
        out[10] = 0;
        out[11] = 0;
        out[12] = 0;
        out[13] = 0;
        out[14] = 0;
        out[15] = 0;
        out[16] = 0;
        out[17] = 0;
        out[18] = 0;
        out[19] = 0;
        out[20] = 0;
        out[21] = 0;
        out[22] = 0;
        out[23] = 0;
        out[24] = 0;
        out[25] = 0;
        out[26] = 0;
        out[27] = 0;
        out[28] = 0;
        out[29] = 0;
        out[30] = 0;
        out[31] = 0;
      }
      out += 32;

      const uint32_t* w1 = w0 + kc;
      const uint32_t* w2 = w1 + kc;
      const uint32_t* w3 = w2 + kc;
      const uint32_t* w4 = w3 + kc;
      const uint32_t* w5 = w4 + kc;
      const uint32_t* w6 = w5 + kc;
      const uint32_t* w7 = w6 + kc;
      const uint32_t* w8 = w7 + kc;
      const uint32_t* w9 = w8 + kc;
      const uint32_t* w10 = w9 + kc;
      const uint32_t* w11 = w10 + kc;
      const uint32_t* w12 = w11 + kc;
      const uint32_t* w13 = w12 + kc;
      const uint32_t* w14 = w13 + kc;
      const uint32_t* w15 = w14 + kc;
      const uint32_t* w16 = w15 + kc;
      const uint32_t* w17 = w16 + kc;
      const uint32_t* w18 = w17 + kc;
      const uint32_t* w19 = w18 + kc;
      const uint32_t* w20 = w19 + kc;
      const uint32_t* w21 = w20 + kc;
      const uint32_t* w22 = w21 + kc;
      const uint32_t* w23 = w22 + kc;
      const uint32_t* w24 = w23 + kc;
      const uint32_t* w25 = w24 + kc;
      const uint32_t* w26 = w25 + kc;
      const uint32_t* w27 = w26 + kc;
      const uint32_t* w28 = w27 + kc;
      const uint32_t* w29 = w28 + kc;
      const uint32_t* w30 = w29 + kc;
      const uint32_t* w31 = w30 + kc;

      // KC main loop multiple of 32x2
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const uint32_t v00 = w0[0];
        const uint32_t v01 = w0[1];
        w0 += 2;
        const uint32_t v10 = w1[0];
        const uint32_t v11 = w1[1];
        w1 += 2;
        const uint32_t v20 = w2[0];
        const uint32_t v21 = w2[1];
        w2 += 2;
        const uint32_t v30 = w3[0];
        const uint32_t v31 = w3[1];
        w3 += 2;
        const uint32_t v40 = w4[0];
        const uint32_t v41 = w4[1];
        w4 += 2;
        const uint32_t v50 = w5[0];
        const uint32_t v51 = w5[1];
        w5 += 2;
        const uint32_t v60 = w6[0];
        const uint32_t v61 = w6[1];
        w6 += 2;
        const uint32_t v70 = w7[0];
        const uint32_t v71 = w7[1];
        w7 += 2;
        const uint32_t v80 = w8[0];
        const uint32_t v81 = w8[1];
        w8 += 2;
        const uint32_t v90 = w9[0];
        const uint32_t v91 = w9[1];
        w9 += 2;
        const uint32_t v100 = w10[0];
        const uint32_t v101 = w10[1];
        w10 += 2;
        const uint32_t v110 = w11[0];
        const uint32_t v111 = w11[1];
        w11 += 2;
        const uint32_t v120 = w12[0];
        const uint32_t v121 = w12[1];
        w12 += 2;
        const uint32_t v130 = w13[0];
        const uint32_t v131 = w13[1];
        w13 += 2;
        const uint32_t v140 = w14[0];
        const uint32_t v141 = w14[1];
        w14 += 2;
        const uint32_t v150 = w15[0];
        const uint32_t v151 = w15[1];
        w15 += 2;
        const uint32_t v160 = w16[0];
        const uint32_t v161 = w16[1];
        w16 += 2;
        const uint32_t v170 = w17[0];
        const uint32_t v171 = w17[1];
        w17 += 2;
        const uint32_t v180 = w18[0];
        const uint32_t v181 = w18[1];
        w18 += 2;
        const uint32_t v190 = w19[0];
        const uint32_t v191 = w19[1];
        w19 += 2;
        const uint32_t v200 = w20[0];
        const uint32_t v201 = w20[1];
        w20 += 2;
        const uint32_t v210 = w21[0];
        const uint32_t v211 = w21[1];
        w21 += 2;
        const uint32_t v220 = w22[0];
        const uint32_t v221 = w22[1];
        w22 += 2;
        const uint32_t v230 = w23[0];
        const uint32_t v231 = w23[1];
        w23 += 2;
        const uint32_t v240 = w24[0];
        const uint32_t v241 = w24[1];
        w24 += 2;
        const uint32_t v250 = w25[0];
        const uint32_t v251 = w25[1];
        w25 += 2;
        const uint32_t v260 = w26[0];
        const uint32_t v261 = w26[1];
        w26 += 2;
        const uint32_t v270 = w27[0];
        const uint32_t v271 = w27[1];
        w27 += 2;
        const uint32_t v280 = w28[0];
        const uint32_t v281 = w28[1];
        w28 += 2;
        const uint32_t v290 = w29[0];
        const uint32_t v291 = w29[1];
        w29 += 2;
        const uint32_t v300 = w30[0];
        const uint32_t v301 = w30[1];
        w30 += 2;
        const uint32_t v310 = w31[0];
        const uint32_t v311 = w31[1];
        w31 += 2;
        out[0] = v00;
        out[1] = v10;
        out[2] = v20;
        out[3] = v30;
        out[4] = v40;
        out[5] = v50;
        out[6] = v60;
        out[7] = v70;
        out[8] = v80;
        out[9] = v90;
        out[10] = v100;
        out[11] = v110;
        out[12] = v120;
        out[13] = v130;
        out[14] = v140;
        out[15] = v150;
        out[16] = v160;
        out[17] = v170;
        out[18] = v180;
        out[19] = v190;
        out[20] = v200;
        out[21] = v210;
        out[22] = v220;
        out[23] = v230;
        out[24] = v240;
        out[25] = v250;
        out[26] = v260;
        out[27] = v270;
        out[28] = v280;
        out[29] = v290;
        out[30] = v300;
        out[31] = v310;
        out[32] = v01;
        out[33] = v11;
        out[34] = v21;
        out[35] = v31;
        out[36] = v41;
        out[37] = v51;
        out[38] = v61;
        out[39] = v71;
        out[40] = v81;
        out[41] = v91;
        out[42] = v101;
        out[43] = v111;
        out[44] = v121;
        out[45] = v131;
        out[46] = v141;
        out[47] = v151;
        out[48] = v161;
        out[49] = v171;
        out[50] = v181;
        out[51] = v191;
        out[52] = v201;
        out[53] = v211;
        out[54] = v221;
        out[55] = v231;
        out[56] = v241;
        out[57] = v251;
        out[58] = v261;
        out[59] = v271;
        out[60] = v281;
        out[61] = v291;
        out[62] = v301;
        out[63] = v311;
        out += 64;
      }

      // KC remainder
      for (; k != 0; --k) {
        const uint32_t v0 = *w0++;
        out[0] = v0;
        const uint32_t v1 = *w1++;
        out[1] = v1;
        const uint32_t v2 = *w2++;
        out[2] = v2;
        const uint32_t v3 = *w3++;
        out[3] = v3;
        const uint32_t v4 = *w4++;
        out[4] = v4;
        const uint32_t v5 = *w5++;
        out[5] = v5;
        const uint32_t v6 = *w6++;
        out[6] = v6;
        const uint32_t v7 = *w7++;
        out[7] = v7;
        const uint32_t v8 = *w8++;
        out[8] = v8;
        const uint32_t v9 = *w9++;
        out[9] = v9;
        const uint32_t v10 = *w10++;
        out[10] = v10;
        const uint32_t v11 = *w11++;
        out[11] = v11;
        const uint32_t v12 = *w12++;
        out[12] = v12;
        const uint32_t v13 = *w13++;
        out[13] = v13;
        const uint32_t v14 = *w14++;
        out[14] = v14;
        const uint32_t v15 = *w15++;
        out[15] = v15;
        const uint32_t v16 = *w16++;
        out[16] = v16;
        const uint32_t v17 = *w17++;
        out[17] = v17;
        const uint32_t v18 = *w18++;
        out[18] = v18;
        const uint32_t v19 = *w19++;
        out[19] = v19;
        const uint32_t v20 = *w20++;
        out[20] = v20;
        const uint32_t v21 = *w21++;
        out[21] = v21;
        const uint32_t v22 = *w22++;
        out[22] = v22;
        const uint32_t v23 = *w23++;
        out[23] = v23;
        const uint32_t v24 = *w24++;
        out[24] = v24;
        const uint32_t v25 = *w25++;
        out[25] = v25;
        const uint32_t v26 = *w26++;
        out[26] = v26;
        const uint32_t v27 = *w27++;
        out[27] = v27;
        const uint32_t v28 = *w28++;
        out[28] = v28;
        const uint32_t v29 = *w29++;
        out[29] = v29;
        const uint32_t v30 = *w30++;
        out[30] = v30;
        const uint32_t v31 = *w31++;
        out[31] = v31;
        out += 32;
      }
      out = (uint32_t*) ((uintptr_t) out + extra_bytes);
      w0 = w31;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *out++ = *b++;
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *out++ = 0;
        } while (--nb != 0);
      }
      out += (32 - n);

      // NR remainder has less than 32 rows so last row is not loaded
      const uint32_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint32_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint32_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint32_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint32_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint32_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint32_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint32_t* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint32_t* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint32_t* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint32_t* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint32_t* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint32_t* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint32_t* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const uint32_t* w15 = w14 + kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const uint32_t* w16 = w15 + kc;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const uint32_t* w17 = w16 + kc;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const uint32_t* w18 = w17 + kc;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const uint32_t* w19 = w18 + kc;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const uint32_t* w20 = w19 + kc;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const uint32_t* w21 = w20 + kc;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const uint32_t* w22 = w21 + kc;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const uint32_t* w23 = w22 + kc;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const uint32_t* w24 = w23 + kc;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const uint32_t* w25 = w24 + kc;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const uint32_t* w26 = w25 + kc;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const uint32_t* w27 = w26 + kc;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const uint32_t* w28 = w27 + kc;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const uint32_t* w29 = w28 + kc;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const uint32_t* w30 = w29 + kc;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }

      // KC main loop multiple of 32x2
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const uint32_t v00 = w0[0];
        const uint32_t v01 = w0[1];
        w0 += 2;
        const uint32_t v10 = w1[0];
        const uint32_t v11 = w1[1];
        w1 += 2;
        const uint32_t v20 = w2[0];
        const uint32_t v21 = w2[1];
        w2 += 2;
        const uint32_t v30 = w3[0];
        const uint32_t v31 = w3[1];
        w3 += 2;
        const uint32_t v40 = w4[0];
        const uint32_t v41 = w4[1];
        w4 += 2;
        const uint32_t v50 = w5[0];
        const uint32_t v51 = w5[1];
        w5 += 2;
        const uint32_t v60 = w6[0];
        const uint32_t v61 = w6[1];
        w6 += 2;
        const uint32_t v70 = w7[0];
        const uint32_t v71 = w7[1];
        w7 += 2;
        const uint32_t v80 = w8[0];
        const uint32_t v81 = w8[1];
        w8 += 2;
        const uint32_t v90 = w9[0];
        const uint32_t v91 = w9[1];
        w9 += 2;
        const uint32_t v100 = w10[0];
        const uint32_t v101 = w10[1];
        w10 += 2;
        const uint32_t v110 = w11[0];
        const uint32_t v111 = w11[1];
        w11 += 2;
        const uint32_t v120 = w12[0];
        const uint32_t v121 = w12[1];
        w12 += 2;
        const uint32_t v130 = w13[0];
        const uint32_t v131 = w13[1];
        w13 += 2;
        const uint32_t v140 = w14[0];
        const uint32_t v141 = w14[1];
        w14 += 2;
        const uint32_t v150 = w15[0];
        const uint32_t v151 = w15[1];
        w15 += 2;
        const uint32_t v160 = w16[0];
        const uint32_t v161 = w16[1];
        w16 += 2;
        const uint32_t v170 = w17[0];
        const uint32_t v171 = w17[1];
        w17 += 2;
        const uint32_t v180 = w18[0];
        const uint32_t v181 = w18[1];
        w18 += 2;
        const uint32_t v190 = w19[0];
        const uint32_t v191 = w19[1];
        w19 += 2;
        const uint32_t v200 = w20[0];
        const uint32_t v201 = w20[1];
        w20 += 2;
        const uint32_t v210 = w21[0];
        const uint32_t v211 = w21[1];
        w21 += 2;
        const uint32_t v220 = w22[0];
        const uint32_t v221 = w22[1];
        w22 += 2;
        const uint32_t v230 = w23[0];
        const uint32_t v231 = w23[1];
        w23 += 2;
        const uint32_t v240 = w24[0];
        const uint32_t v241 = w24[1];
        w24 += 2;
        const uint32_t v250 = w25[0];
        const uint32_t v251 = w25[1];
        w25 += 2;
        const uint32_t v260 = w26[0];
        const uint32_t v261 = w26[1];
        w26 += 2;
        const uint32_t v270 = w27[0];
        const uint32_t v271 = w27[1];
        w27 += 2;
        const uint32_t v280 = w28[0];
        const uint32_t v281 = w28[1];
        w28 += 2;
        const uint32_t v290 = w29[0];
        const uint32_t v291 = w29[1];
        w29 += 2;
        const uint32_t v300 = w30[0];
        const uint32_t v301 = w30[1];
        w30 += 2;
        out[0] = v00;
        out[1] = v10;
        out[2] = v20;
        out[3] = v30;
        out[4] = v40;
        out[5] = v50;
        out[6] = v60;
        out[7] = v70;
        out[8] = v80;
        out[9] = v90;
        out[10] = v100;
        out[11] = v110;
        out[12] = v120;
        out[13] = v130;
        out[14] = v140;
        out[15] = v150;
        out[16] = v160;
        out[17] = v170;
        out[18] = v180;
        out[19] = v190;
        out[20] = v200;
        out[21] = v210;
        out[22] = v220;
        out[23] = v230;
        out[24] = v240;
        out[25] = v250;
        out[26] = v260;
        out[27] = v270;
        out[28] = v280;
        out[29] = v290;
        out[30] = v300;
        out[32] = v01;
        out[33] = v11;
        out[34] = v21;
        out[35] = v31;
        out[36] = v41;
        out[37] = v51;
        out[38] = v61;
        out[39] = v71;
        out[40] = v81;
        out[41] = v91;
        out[42] = v101;
        out[43] = v111;
        out[44] = v121;
        out[45] = v131;
        out[46] = v141;
        out[47] = v151;
        out[48] = v161;
        out[49] = v171;
        out[50] = v181;
        out[51] = v191;
        out[52] = v201;
        out[53] = v211;
        out[54] = v221;
        out[55] = v231;
        out[56] = v241;
        out[57] = v251;
        out[58] = v261;
        out[59] = v271;
        out[60] = v281;
        out[61] = v291;
        out[62] = v301;
        out += 64;
      }

      // KC remainder of 1..1
      for (; k != 0; --k) {
        const uint32_t v0 = *w0++;
        out[0] = v0;
        const uint32_t v1 = *w1++;
        out[1] = v1;
        const uint32_t v2 = *w2++;
        out[2] = v2;
        const uint32_t v3 = *w3++;
        out[3] = v3;
        const uint32_t v4 = *w4++;
        out[4] = v4;
        const uint32_t v5 = *w5++;
        out[5] = v5;
        const uint32_t v6 = *w6++;
        out[6] = v6;
        const uint32_t v7 = *w7++;
        out[7] = v7;
        const uint32_t v8 = *w8++;
        out[8] = v8;
        const uint32_t v9 = *w9++;
        out[9] = v9;
        const uint32_t v10 = *w10++;
        out[10] = v10;
        const uint32_t v11 = *w11++;
        out[11] = v11;
        const uint32_t v12 = *w12++;
        out[12] = v12;
        const uint32_t v13 = *w13++;
        out[13] = v13;
        const uint32_t v14 = *w14++;
        out[14] = v14;
        const uint32_t v15 = *w15++;
        out[15] = v15;
        const uint32_t v16 = *w16++;
        out[16] = v16;
        const uint32_t v17 = *w17++;
        out[17] = v17;
        const uint32_t v18 = *w18++;
        out[18] = v18;
        const uint32_t v19 = *w19++;
        out[19] = v19;
        const uint32_t v20 = *w20++;
        out[20] = v20;
        const uint32_t v21 = *w21++;
        out[21] = v21;
        const uint32_t v22 = *w22++;
        out[22] = v22;
        const uint32_t v23 = *w23++;
        out[23] = v23;
        const uint32_t v24 = *w24++;
        out[24] = v24;
        const uint32_t v25 = *w25++;
        out[25] = v25;
        const uint32_t v26 = *w26++;
        out[26] = v26;
        const uint32_t v27 = *w27++;
        out[27] = v27;
        const uint32_t v28 = *w28++;
        out[28] = v28;
        const uint32_t v29 = *w29++;
        out[29] = v29;
        const uint32_t v30 = *w30++;
        out[30] = v30;
        out += 32;
      }
      out = (uint32_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
