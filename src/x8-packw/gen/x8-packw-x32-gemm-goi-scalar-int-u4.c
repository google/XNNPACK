// Auto-generated file. Do not edit!
//   Template: src/x8-packw/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/math.h"
#include "xnnpack/packw.h"
#include "xnnpack/unaligned.h"

void xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* weights,
  const uint32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 32);   // This kernel is for NR=32
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    // NC main loop multiple of 32
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        unaligned_store_s32(out + 0 * sizeof(int32_t), b[0]);
        unaligned_store_s32(out + 1 * sizeof(int32_t), b[1]);
        unaligned_store_s32(out + 2 * sizeof(int32_t), b[2]);
        unaligned_store_s32(out + 3 * sizeof(int32_t), b[3]);
        unaligned_store_s32(out + 4 * sizeof(int32_t), b[4]);
        unaligned_store_s32(out + 5 * sizeof(int32_t), b[5]);
        unaligned_store_s32(out + 6 * sizeof(int32_t), b[6]);
        unaligned_store_s32(out + 7 * sizeof(int32_t), b[7]);
        unaligned_store_s32(out + 8 * sizeof(int32_t), b[8]);
        unaligned_store_s32(out + 9 * sizeof(int32_t), b[9]);
        unaligned_store_s32(out + 10 * sizeof(int32_t), b[10]);
        unaligned_store_s32(out + 11 * sizeof(int32_t), b[11]);
        unaligned_store_s32(out + 12 * sizeof(int32_t), b[12]);
        unaligned_store_s32(out + 13 * sizeof(int32_t), b[13]);
        unaligned_store_s32(out + 14 * sizeof(int32_t), b[14]);
        unaligned_store_s32(out + 15 * sizeof(int32_t), b[15]);
        unaligned_store_s32(out + 16 * sizeof(int32_t), b[16]);
        unaligned_store_s32(out + 17 * sizeof(int32_t), b[17]);
        unaligned_store_s32(out + 18 * sizeof(int32_t), b[18]);
        unaligned_store_s32(out + 19 * sizeof(int32_t), b[19]);
        unaligned_store_s32(out + 20 * sizeof(int32_t), b[20]);
        unaligned_store_s32(out + 21 * sizeof(int32_t), b[21]);
        unaligned_store_s32(out + 22 * sizeof(int32_t), b[22]);
        unaligned_store_s32(out + 23 * sizeof(int32_t), b[23]);
        unaligned_store_s32(out + 24 * sizeof(int32_t), b[24]);
        unaligned_store_s32(out + 25 * sizeof(int32_t), b[25]);
        unaligned_store_s32(out + 26 * sizeof(int32_t), b[26]);
        unaligned_store_s32(out + 27 * sizeof(int32_t), b[27]);
        unaligned_store_s32(out + 28 * sizeof(int32_t), b[28]);
        unaligned_store_s32(out + 29 * sizeof(int32_t), b[29]);
        unaligned_store_s32(out + 30 * sizeof(int32_t), b[30]);
        unaligned_store_s32(out + 31 * sizeof(int32_t), b[31]);
        b += 32;
      } else {
        unaligned_store_s32(out + 0 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 1 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 2 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 3 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 4 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 5 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 6 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 7 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 8 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 9 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 10 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 11 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 12 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 13 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 14 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 15 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 16 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 17 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 18 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 19 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 20 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 21 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 22 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 23 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 24 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 25 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 26 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 27 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 28 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 29 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 30 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 31 * sizeof(int32_t), 0);
      }
      out += 32 * sizeof(uint32_t);

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

      // KC main loop multiple of 32x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const int8_t v00 = w0[0];
        const int8_t v01 = w0[1];
        const int8_t v02 = w0[2];
        const int8_t v03 = w0[3];
        w0 += 4;
        const int8_t v10 = w1[0];
        const int8_t v11 = w1[1];
        const int8_t v12 = w1[2];
        const int8_t v13 = w1[3];
        w1 += 4;
        const int8_t v20 = w2[0];
        const int8_t v21 = w2[1];
        const int8_t v22 = w2[2];
        const int8_t v23 = w2[3];
        w2 += 4;
        const int8_t v30 = w3[0];
        const int8_t v31 = w3[1];
        const int8_t v32 = w3[2];
        const int8_t v33 = w3[3];
        w3 += 4;
        const int8_t v40 = w4[0];
        const int8_t v41 = w4[1];
        const int8_t v42 = w4[2];
        const int8_t v43 = w4[3];
        w4 += 4;
        const int8_t v50 = w5[0];
        const int8_t v51 = w5[1];
        const int8_t v52 = w5[2];
        const int8_t v53 = w5[3];
        w5 += 4;
        const int8_t v60 = w6[0];
        const int8_t v61 = w6[1];
        const int8_t v62 = w6[2];
        const int8_t v63 = w6[3];
        w6 += 4;
        const int8_t v70 = w7[0];
        const int8_t v71 = w7[1];
        const int8_t v72 = w7[2];
        const int8_t v73 = w7[3];
        w7 += 4;
        const int8_t v80 = w8[0];
        const int8_t v81 = w8[1];
        const int8_t v82 = w8[2];
        const int8_t v83 = w8[3];
        w8 += 4;
        const int8_t v90 = w9[0];
        const int8_t v91 = w9[1];
        const int8_t v92 = w9[2];
        const int8_t v93 = w9[3];
        w9 += 4;
        const int8_t v100 = w10[0];
        const int8_t v101 = w10[1];
        const int8_t v102 = w10[2];
        const int8_t v103 = w10[3];
        w10 += 4;
        const int8_t v110 = w11[0];
        const int8_t v111 = w11[1];
        const int8_t v112 = w11[2];
        const int8_t v113 = w11[3];
        w11 += 4;
        const int8_t v120 = w12[0];
        const int8_t v121 = w12[1];
        const int8_t v122 = w12[2];
        const int8_t v123 = w12[3];
        w12 += 4;
        const int8_t v130 = w13[0];
        const int8_t v131 = w13[1];
        const int8_t v132 = w13[2];
        const int8_t v133 = w13[3];
        w13 += 4;
        const int8_t v140 = w14[0];
        const int8_t v141 = w14[1];
        const int8_t v142 = w14[2];
        const int8_t v143 = w14[3];
        w14 += 4;
        const int8_t v150 = w15[0];
        const int8_t v151 = w15[1];
        const int8_t v152 = w15[2];
        const int8_t v153 = w15[3];
        w15 += 4;
        const int8_t v160 = w16[0];
        const int8_t v161 = w16[1];
        const int8_t v162 = w16[2];
        const int8_t v163 = w16[3];
        w16 += 4;
        const int8_t v170 = w17[0];
        const int8_t v171 = w17[1];
        const int8_t v172 = w17[2];
        const int8_t v173 = w17[3];
        w17 += 4;
        const int8_t v180 = w18[0];
        const int8_t v181 = w18[1];
        const int8_t v182 = w18[2];
        const int8_t v183 = w18[3];
        w18 += 4;
        const int8_t v190 = w19[0];
        const int8_t v191 = w19[1];
        const int8_t v192 = w19[2];
        const int8_t v193 = w19[3];
        w19 += 4;
        const int8_t v200 = w20[0];
        const int8_t v201 = w20[1];
        const int8_t v202 = w20[2];
        const int8_t v203 = w20[3];
        w20 += 4;
        const int8_t v210 = w21[0];
        const int8_t v211 = w21[1];
        const int8_t v212 = w21[2];
        const int8_t v213 = w21[3];
        w21 += 4;
        const int8_t v220 = w22[0];
        const int8_t v221 = w22[1];
        const int8_t v222 = w22[2];
        const int8_t v223 = w22[3];
        w22 += 4;
        const int8_t v230 = w23[0];
        const int8_t v231 = w23[1];
        const int8_t v232 = w23[2];
        const int8_t v233 = w23[3];
        w23 += 4;
        const int8_t v240 = w24[0];
        const int8_t v241 = w24[1];
        const int8_t v242 = w24[2];
        const int8_t v243 = w24[3];
        w24 += 4;
        const int8_t v250 = w25[0];
        const int8_t v251 = w25[1];
        const int8_t v252 = w25[2];
        const int8_t v253 = w25[3];
        w25 += 4;
        const int8_t v260 = w26[0];
        const int8_t v261 = w26[1];
        const int8_t v262 = w26[2];
        const int8_t v263 = w26[3];
        w26 += 4;
        const int8_t v270 = w27[0];
        const int8_t v271 = w27[1];
        const int8_t v272 = w27[2];
        const int8_t v273 = w27[3];
        w27 += 4;
        const int8_t v280 = w28[0];
        const int8_t v281 = w28[1];
        const int8_t v282 = w28[2];
        const int8_t v283 = w28[3];
        w28 += 4;
        const int8_t v290 = w29[0];
        const int8_t v291 = w29[1];
        const int8_t v292 = w29[2];
        const int8_t v293 = w29[3];
        w29 += 4;
        const int8_t v300 = w30[0];
        const int8_t v301 = w30[1];
        const int8_t v302 = w30[2];
        const int8_t v303 = w30[3];
        w30 += 4;
        const int8_t v310 = w31[0];
        const int8_t v311 = w31[1];
        const int8_t v312 = w31[2];
        const int8_t v313 = w31[3];
        w31 += 4;
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
        out[64] = v02;
        out[65] = v12;
        out[66] = v22;
        out[67] = v32;
        out[68] = v42;
        out[69] = v52;
        out[70] = v62;
        out[71] = v72;
        out[72] = v82;
        out[73] = v92;
        out[74] = v102;
        out[75] = v112;
        out[76] = v122;
        out[77] = v132;
        out[78] = v142;
        out[79] = v152;
        out[80] = v162;
        out[81] = v172;
        out[82] = v182;
        out[83] = v192;
        out[84] = v202;
        out[85] = v212;
        out[86] = v222;
        out[87] = v232;
        out[88] = v242;
        out[89] = v252;
        out[90] = v262;
        out[91] = v272;
        out[92] = v282;
        out[93] = v292;
        out[94] = v302;
        out[95] = v312;
        out[96] = v03;
        out[97] = v13;
        out[98] = v23;
        out[99] = v33;
        out[100] = v43;
        out[101] = v53;
        out[102] = v63;
        out[103] = v73;
        out[104] = v83;
        out[105] = v93;
        out[106] = v103;
        out[107] = v113;
        out[108] = v123;
        out[109] = v133;
        out[110] = v143;
        out[111] = v153;
        out[112] = v163;
        out[113] = v173;
        out[114] = v183;
        out[115] = v193;
        out[116] = v203;
        out[117] = v213;
        out[118] = v223;
        out[119] = v233;
        out[120] = v243;
        out[121] = v253;
        out[122] = v263;
        out[123] = v273;
        out[124] = v283;
        out[125] = v293;
        out[126] = v303;
        out[127] = v313;
        out += 128;
      }

      // KC remainder
      for (; k != 0; --k) {
        const int8_t v0 = *w0++;
        out[0] = v0;
        const int8_t v1 = *w1++;
        out[1] = v1;
        const int8_t v2 = *w2++;
        out[2] = v2;
        const int8_t v3 = *w3++;
        out[3] = v3;
        const int8_t v4 = *w4++;
        out[4] = v4;
        const int8_t v5 = *w5++;
        out[5] = v5;
        const int8_t v6 = *w6++;
        out[6] = v6;
        const int8_t v7 = *w7++;
        out[7] = v7;
        const int8_t v8 = *w8++;
        out[8] = v8;
        const int8_t v9 = *w9++;
        out[9] = v9;
        const int8_t v10 = *w10++;
        out[10] = v10;
        const int8_t v11 = *w11++;
        out[11] = v11;
        const int8_t v12 = *w12++;
        out[12] = v12;
        const int8_t v13 = *w13++;
        out[13] = v13;
        const int8_t v14 = *w14++;
        out[14] = v14;
        const int8_t v15 = *w15++;
        out[15] = v15;
        const int8_t v16 = *w16++;
        out[16] = v16;
        const int8_t v17 = *w17++;
        out[17] = v17;
        const int8_t v18 = *w18++;
        out[18] = v18;
        const int8_t v19 = *w19++;
        out[19] = v19;
        const int8_t v20 = *w20++;
        out[20] = v20;
        const int8_t v21 = *w21++;
        out[21] = v21;
        const int8_t v22 = *w22++;
        out[22] = v22;
        const int8_t v23 = *w23++;
        out[23] = v23;
        const int8_t v24 = *w24++;
        out[24] = v24;
        const int8_t v25 = *w25++;
        out[25] = v25;
        const int8_t v26 = *w26++;
        out[26] = v26;
        const int8_t v27 = *w27++;
        out[27] = v27;
        const int8_t v28 = *w28++;
        out[28] = v28;
        const int8_t v29 = *w29++;
        out[29] = v29;
        const int8_t v30 = *w30++;
        out[30] = v30;
        const int8_t v31 = *w31++;
        out[31] = v31;
        out += 32;
      }
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w31;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          unaligned_store_s32(out, *b++);
          out += sizeof(uint32_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          unaligned_store_s32(out, 0);
          out += sizeof(uint32_t);
        } while (--nb != 0);
      }
      out += (32 - n) * sizeof(uint32_t);

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

      // KC main loop multiple of 32x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const int8_t v00 = w0[0];
        const int8_t v01 = w0[1];
        const int8_t v02 = w0[2];
        const int8_t v03 = w0[3];
        w0 += 4;
        const int8_t v10 = w1[0];
        const int8_t v11 = w1[1];
        const int8_t v12 = w1[2];
        const int8_t v13 = w1[3];
        w1 += 4;
        const int8_t v20 = w2[0];
        const int8_t v21 = w2[1];
        const int8_t v22 = w2[2];
        const int8_t v23 = w2[3];
        w2 += 4;
        const int8_t v30 = w3[0];
        const int8_t v31 = w3[1];
        const int8_t v32 = w3[2];
        const int8_t v33 = w3[3];
        w3 += 4;
        const int8_t v40 = w4[0];
        const int8_t v41 = w4[1];
        const int8_t v42 = w4[2];
        const int8_t v43 = w4[3];
        w4 += 4;
        const int8_t v50 = w5[0];
        const int8_t v51 = w5[1];
        const int8_t v52 = w5[2];
        const int8_t v53 = w5[3];
        w5 += 4;
        const int8_t v60 = w6[0];
        const int8_t v61 = w6[1];
        const int8_t v62 = w6[2];
        const int8_t v63 = w6[3];
        w6 += 4;
        const int8_t v70 = w7[0];
        const int8_t v71 = w7[1];
        const int8_t v72 = w7[2];
        const int8_t v73 = w7[3];
        w7 += 4;
        const int8_t v80 = w8[0];
        const int8_t v81 = w8[1];
        const int8_t v82 = w8[2];
        const int8_t v83 = w8[3];
        w8 += 4;
        const int8_t v90 = w9[0];
        const int8_t v91 = w9[1];
        const int8_t v92 = w9[2];
        const int8_t v93 = w9[3];
        w9 += 4;
        const int8_t v100 = w10[0];
        const int8_t v101 = w10[1];
        const int8_t v102 = w10[2];
        const int8_t v103 = w10[3];
        w10 += 4;
        const int8_t v110 = w11[0];
        const int8_t v111 = w11[1];
        const int8_t v112 = w11[2];
        const int8_t v113 = w11[3];
        w11 += 4;
        const int8_t v120 = w12[0];
        const int8_t v121 = w12[1];
        const int8_t v122 = w12[2];
        const int8_t v123 = w12[3];
        w12 += 4;
        const int8_t v130 = w13[0];
        const int8_t v131 = w13[1];
        const int8_t v132 = w13[2];
        const int8_t v133 = w13[3];
        w13 += 4;
        const int8_t v140 = w14[0];
        const int8_t v141 = w14[1];
        const int8_t v142 = w14[2];
        const int8_t v143 = w14[3];
        w14 += 4;
        const int8_t v150 = w15[0];
        const int8_t v151 = w15[1];
        const int8_t v152 = w15[2];
        const int8_t v153 = w15[3];
        w15 += 4;
        const int8_t v160 = w16[0];
        const int8_t v161 = w16[1];
        const int8_t v162 = w16[2];
        const int8_t v163 = w16[3];
        w16 += 4;
        const int8_t v170 = w17[0];
        const int8_t v171 = w17[1];
        const int8_t v172 = w17[2];
        const int8_t v173 = w17[3];
        w17 += 4;
        const int8_t v180 = w18[0];
        const int8_t v181 = w18[1];
        const int8_t v182 = w18[2];
        const int8_t v183 = w18[3];
        w18 += 4;
        const int8_t v190 = w19[0];
        const int8_t v191 = w19[1];
        const int8_t v192 = w19[2];
        const int8_t v193 = w19[3];
        w19 += 4;
        const int8_t v200 = w20[0];
        const int8_t v201 = w20[1];
        const int8_t v202 = w20[2];
        const int8_t v203 = w20[3];
        w20 += 4;
        const int8_t v210 = w21[0];
        const int8_t v211 = w21[1];
        const int8_t v212 = w21[2];
        const int8_t v213 = w21[3];
        w21 += 4;
        const int8_t v220 = w22[0];
        const int8_t v221 = w22[1];
        const int8_t v222 = w22[2];
        const int8_t v223 = w22[3];
        w22 += 4;
        const int8_t v230 = w23[0];
        const int8_t v231 = w23[1];
        const int8_t v232 = w23[2];
        const int8_t v233 = w23[3];
        w23 += 4;
        const int8_t v240 = w24[0];
        const int8_t v241 = w24[1];
        const int8_t v242 = w24[2];
        const int8_t v243 = w24[3];
        w24 += 4;
        const int8_t v250 = w25[0];
        const int8_t v251 = w25[1];
        const int8_t v252 = w25[2];
        const int8_t v253 = w25[3];
        w25 += 4;
        const int8_t v260 = w26[0];
        const int8_t v261 = w26[1];
        const int8_t v262 = w26[2];
        const int8_t v263 = w26[3];
        w26 += 4;
        const int8_t v270 = w27[0];
        const int8_t v271 = w27[1];
        const int8_t v272 = w27[2];
        const int8_t v273 = w27[3];
        w27 += 4;
        const int8_t v280 = w28[0];
        const int8_t v281 = w28[1];
        const int8_t v282 = w28[2];
        const int8_t v283 = w28[3];
        w28 += 4;
        const int8_t v290 = w29[0];
        const int8_t v291 = w29[1];
        const int8_t v292 = w29[2];
        const int8_t v293 = w29[3];
        w29 += 4;
        const int8_t v300 = w30[0];
        const int8_t v301 = w30[1];
        const int8_t v302 = w30[2];
        const int8_t v303 = w30[3];
        w30 += 4;
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
        out[64] = v02;
        out[65] = v12;
        out[66] = v22;
        out[67] = v32;
        out[68] = v42;
        out[69] = v52;
        out[70] = v62;
        out[71] = v72;
        out[72] = v82;
        out[73] = v92;
        out[74] = v102;
        out[75] = v112;
        out[76] = v122;
        out[77] = v132;
        out[78] = v142;
        out[79] = v152;
        out[80] = v162;
        out[81] = v172;
        out[82] = v182;
        out[83] = v192;
        out[84] = v202;
        out[85] = v212;
        out[86] = v222;
        out[87] = v232;
        out[88] = v242;
        out[89] = v252;
        out[90] = v262;
        out[91] = v272;
        out[92] = v282;
        out[93] = v292;
        out[94] = v302;
        out[96] = v03;
        out[97] = v13;
        out[98] = v23;
        out[99] = v33;
        out[100] = v43;
        out[101] = v53;
        out[102] = v63;
        out[103] = v73;
        out[104] = v83;
        out[105] = v93;
        out[106] = v103;
        out[107] = v113;
        out[108] = v123;
        out[109] = v133;
        out[110] = v143;
        out[111] = v153;
        out[112] = v163;
        out[113] = v173;
        out[114] = v183;
        out[115] = v193;
        out[116] = v203;
        out[117] = v213;
        out[118] = v223;
        out[119] = v233;
        out[120] = v243;
        out[121] = v253;
        out[122] = v263;
        out[123] = v273;
        out[124] = v283;
        out[125] = v293;
        out[126] = v303;
        out += 128;
      }

      // KC remainder of 1..3
      for (; k != 0; --k) {
        const int8_t v0 = *w0++;
        out[0] = v0;
        const int8_t v1 = *w1++;
        out[1] = v1;
        const int8_t v2 = *w2++;
        out[2] = v2;
        const int8_t v3 = *w3++;
        out[3] = v3;
        const int8_t v4 = *w4++;
        out[4] = v4;
        const int8_t v5 = *w5++;
        out[5] = v5;
        const int8_t v6 = *w6++;
        out[6] = v6;
        const int8_t v7 = *w7++;
        out[7] = v7;
        const int8_t v8 = *w8++;
        out[8] = v8;
        const int8_t v9 = *w9++;
        out[9] = v9;
        const int8_t v10 = *w10++;
        out[10] = v10;
        const int8_t v11 = *w11++;
        out[11] = v11;
        const int8_t v12 = *w12++;
        out[12] = v12;
        const int8_t v13 = *w13++;
        out[13] = v13;
        const int8_t v14 = *w14++;
        out[14] = v14;
        const int8_t v15 = *w15++;
        out[15] = v15;
        const int8_t v16 = *w16++;
        out[16] = v16;
        const int8_t v17 = *w17++;
        out[17] = v17;
        const int8_t v18 = *w18++;
        out[18] = v18;
        const int8_t v19 = *w19++;
        out[19] = v19;
        const int8_t v20 = *w20++;
        out[20] = v20;
        const int8_t v21 = *w21++;
        out[21] = v21;
        const int8_t v22 = *w22++;
        out[22] = v22;
        const int8_t v23 = *w23++;
        out[23] = v23;
        const int8_t v24 = *w24++;
        out[24] = v24;
        const int8_t v25 = *w25++;
        out[25] = v25;
        const int8_t v26 = *w26++;
        out[26] = v26;
        const int8_t v27 = *w27++;
        out[27] = v27;
        const int8_t v28 = *w28++;
        out[28] = v28;
        const int8_t v29 = *w29++;
        out[29] = v29;
        const int8_t v30 = *w30++;
        out[30] = v30;
        out += 32;
      }
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
