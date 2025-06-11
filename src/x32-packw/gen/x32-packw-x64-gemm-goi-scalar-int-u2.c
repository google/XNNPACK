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



void xnn_x32_packw_gemm_goi_ukernel_x64__scalar_int_u2(
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
  assert(nr == 64);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint32_t* out = (uint32_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    // NC main loop multiple of 64
    const uint32_t* w0 = (const uint32_t*) weights;
    size_t n = nc;
    for (;n >= 64; n -= 64) {
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
        out[32] = b[32];
        out[33] = b[33];
        out[34] = b[34];
        out[35] = b[35];
        out[36] = b[36];
        out[37] = b[37];
        out[38] = b[38];
        out[39] = b[39];
        out[40] = b[40];
        out[41] = b[41];
        out[42] = b[42];
        out[43] = b[43];
        out[44] = b[44];
        out[45] = b[45];
        out[46] = b[46];
        out[47] = b[47];
        out[48] = b[48];
        out[49] = b[49];
        out[50] = b[50];
        out[51] = b[51];
        out[52] = b[52];
        out[53] = b[53];
        out[54] = b[54];
        out[55] = b[55];
        out[56] = b[56];
        out[57] = b[57];
        out[58] = b[58];
        out[59] = b[59];
        out[60] = b[60];
        out[61] = b[61];
        out[62] = b[62];
        out[63] = b[63];
        b += 64;
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
        out[32] = 0;
        out[33] = 0;
        out[34] = 0;
        out[35] = 0;
        out[36] = 0;
        out[37] = 0;
        out[38] = 0;
        out[39] = 0;
        out[40] = 0;
        out[41] = 0;
        out[42] = 0;
        out[43] = 0;
        out[44] = 0;
        out[45] = 0;
        out[46] = 0;
        out[47] = 0;
        out[48] = 0;
        out[49] = 0;
        out[50] = 0;
        out[51] = 0;
        out[52] = 0;
        out[53] = 0;
        out[54] = 0;
        out[55] = 0;
        out[56] = 0;
        out[57] = 0;
        out[58] = 0;
        out[59] = 0;
        out[60] = 0;
        out[61] = 0;
        out[62] = 0;
        out[63] = 0;
      }
      out += 64;

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
      const uint32_t* w32 = w31 + kc;
      const uint32_t* w33 = w32 + kc;
      const uint32_t* w34 = w33 + kc;
      const uint32_t* w35 = w34 + kc;
      const uint32_t* w36 = w35 + kc;
      const uint32_t* w37 = w36 + kc;
      const uint32_t* w38 = w37 + kc;
      const uint32_t* w39 = w38 + kc;
      const uint32_t* w40 = w39 + kc;
      const uint32_t* w41 = w40 + kc;
      const uint32_t* w42 = w41 + kc;
      const uint32_t* w43 = w42 + kc;
      const uint32_t* w44 = w43 + kc;
      const uint32_t* w45 = w44 + kc;
      const uint32_t* w46 = w45 + kc;
      const uint32_t* w47 = w46 + kc;
      const uint32_t* w48 = w47 + kc;
      const uint32_t* w49 = w48 + kc;
      const uint32_t* w50 = w49 + kc;
      const uint32_t* w51 = w50 + kc;
      const uint32_t* w52 = w51 + kc;
      const uint32_t* w53 = w52 + kc;
      const uint32_t* w54 = w53 + kc;
      const uint32_t* w55 = w54 + kc;
      const uint32_t* w56 = w55 + kc;
      const uint32_t* w57 = w56 + kc;
      const uint32_t* w58 = w57 + kc;
      const uint32_t* w59 = w58 + kc;
      const uint32_t* w60 = w59 + kc;
      const uint32_t* w61 = w60 + kc;
      const uint32_t* w62 = w61 + kc;
      const uint32_t* w63 = w62 + kc;

      // KC main loop multiple of 64x2
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
        const uint32_t v320 = w32[0];
        const uint32_t v321 = w32[1];
        w32 += 2;
        const uint32_t v330 = w33[0];
        const uint32_t v331 = w33[1];
        w33 += 2;
        const uint32_t v340 = w34[0];
        const uint32_t v341 = w34[1];
        w34 += 2;
        const uint32_t v350 = w35[0];
        const uint32_t v351 = w35[1];
        w35 += 2;
        const uint32_t v360 = w36[0];
        const uint32_t v361 = w36[1];
        w36 += 2;
        const uint32_t v370 = w37[0];
        const uint32_t v371 = w37[1];
        w37 += 2;
        const uint32_t v380 = w38[0];
        const uint32_t v381 = w38[1];
        w38 += 2;
        const uint32_t v390 = w39[0];
        const uint32_t v391 = w39[1];
        w39 += 2;
        const uint32_t v400 = w40[0];
        const uint32_t v401 = w40[1];
        w40 += 2;
        const uint32_t v410 = w41[0];
        const uint32_t v411 = w41[1];
        w41 += 2;
        const uint32_t v420 = w42[0];
        const uint32_t v421 = w42[1];
        w42 += 2;
        const uint32_t v430 = w43[0];
        const uint32_t v431 = w43[1];
        w43 += 2;
        const uint32_t v440 = w44[0];
        const uint32_t v441 = w44[1];
        w44 += 2;
        const uint32_t v450 = w45[0];
        const uint32_t v451 = w45[1];
        w45 += 2;
        const uint32_t v460 = w46[0];
        const uint32_t v461 = w46[1];
        w46 += 2;
        const uint32_t v470 = w47[0];
        const uint32_t v471 = w47[1];
        w47 += 2;
        const uint32_t v480 = w48[0];
        const uint32_t v481 = w48[1];
        w48 += 2;
        const uint32_t v490 = w49[0];
        const uint32_t v491 = w49[1];
        w49 += 2;
        const uint32_t v500 = w50[0];
        const uint32_t v501 = w50[1];
        w50 += 2;
        const uint32_t v510 = w51[0];
        const uint32_t v511 = w51[1];
        w51 += 2;
        const uint32_t v520 = w52[0];
        const uint32_t v521 = w52[1];
        w52 += 2;
        const uint32_t v530 = w53[0];
        const uint32_t v531 = w53[1];
        w53 += 2;
        const uint32_t v540 = w54[0];
        const uint32_t v541 = w54[1];
        w54 += 2;
        const uint32_t v550 = w55[0];
        const uint32_t v551 = w55[1];
        w55 += 2;
        const uint32_t v560 = w56[0];
        const uint32_t v561 = w56[1];
        w56 += 2;
        const uint32_t v570 = w57[0];
        const uint32_t v571 = w57[1];
        w57 += 2;
        const uint32_t v580 = w58[0];
        const uint32_t v581 = w58[1];
        w58 += 2;
        const uint32_t v590 = w59[0];
        const uint32_t v591 = w59[1];
        w59 += 2;
        const uint32_t v600 = w60[0];
        const uint32_t v601 = w60[1];
        w60 += 2;
        const uint32_t v610 = w61[0];
        const uint32_t v611 = w61[1];
        w61 += 2;
        const uint32_t v620 = w62[0];
        const uint32_t v621 = w62[1];
        w62 += 2;
        const uint32_t v630 = w63[0];
        const uint32_t v631 = w63[1];
        w63 += 2;
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
        out[32] = v320;
        out[33] = v330;
        out[34] = v340;
        out[35] = v350;
        out[36] = v360;
        out[37] = v370;
        out[38] = v380;
        out[39] = v390;
        out[40] = v400;
        out[41] = v410;
        out[42] = v420;
        out[43] = v430;
        out[44] = v440;
        out[45] = v450;
        out[46] = v460;
        out[47] = v470;
        out[48] = v480;
        out[49] = v490;
        out[50] = v500;
        out[51] = v510;
        out[52] = v520;
        out[53] = v530;
        out[54] = v540;
        out[55] = v550;
        out[56] = v560;
        out[57] = v570;
        out[58] = v580;
        out[59] = v590;
        out[60] = v600;
        out[61] = v610;
        out[62] = v620;
        out[63] = v630;
        out[64] = v01;
        out[65] = v11;
        out[66] = v21;
        out[67] = v31;
        out[68] = v41;
        out[69] = v51;
        out[70] = v61;
        out[71] = v71;
        out[72] = v81;
        out[73] = v91;
        out[74] = v101;
        out[75] = v111;
        out[76] = v121;
        out[77] = v131;
        out[78] = v141;
        out[79] = v151;
        out[80] = v161;
        out[81] = v171;
        out[82] = v181;
        out[83] = v191;
        out[84] = v201;
        out[85] = v211;
        out[86] = v221;
        out[87] = v231;
        out[88] = v241;
        out[89] = v251;
        out[90] = v261;
        out[91] = v271;
        out[92] = v281;
        out[93] = v291;
        out[94] = v301;
        out[95] = v311;
        out[96] = v321;
        out[97] = v331;
        out[98] = v341;
        out[99] = v351;
        out[100] = v361;
        out[101] = v371;
        out[102] = v381;
        out[103] = v391;
        out[104] = v401;
        out[105] = v411;
        out[106] = v421;
        out[107] = v431;
        out[108] = v441;
        out[109] = v451;
        out[110] = v461;
        out[111] = v471;
        out[112] = v481;
        out[113] = v491;
        out[114] = v501;
        out[115] = v511;
        out[116] = v521;
        out[117] = v531;
        out[118] = v541;
        out[119] = v551;
        out[120] = v561;
        out[121] = v571;
        out[122] = v581;
        out[123] = v591;
        out[124] = v601;
        out[125] = v611;
        out[126] = v621;
        out[127] = v631;
        out += 128;
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
        const uint32_t v32 = *w32++;
        out[32] = v32;
        const uint32_t v33 = *w33++;
        out[33] = v33;
        const uint32_t v34 = *w34++;
        out[34] = v34;
        const uint32_t v35 = *w35++;
        out[35] = v35;
        const uint32_t v36 = *w36++;
        out[36] = v36;
        const uint32_t v37 = *w37++;
        out[37] = v37;
        const uint32_t v38 = *w38++;
        out[38] = v38;
        const uint32_t v39 = *w39++;
        out[39] = v39;
        const uint32_t v40 = *w40++;
        out[40] = v40;
        const uint32_t v41 = *w41++;
        out[41] = v41;
        const uint32_t v42 = *w42++;
        out[42] = v42;
        const uint32_t v43 = *w43++;
        out[43] = v43;
        const uint32_t v44 = *w44++;
        out[44] = v44;
        const uint32_t v45 = *w45++;
        out[45] = v45;
        const uint32_t v46 = *w46++;
        out[46] = v46;
        const uint32_t v47 = *w47++;
        out[47] = v47;
        const uint32_t v48 = *w48++;
        out[48] = v48;
        const uint32_t v49 = *w49++;
        out[49] = v49;
        const uint32_t v50 = *w50++;
        out[50] = v50;
        const uint32_t v51 = *w51++;
        out[51] = v51;
        const uint32_t v52 = *w52++;
        out[52] = v52;
        const uint32_t v53 = *w53++;
        out[53] = v53;
        const uint32_t v54 = *w54++;
        out[54] = v54;
        const uint32_t v55 = *w55++;
        out[55] = v55;
        const uint32_t v56 = *w56++;
        out[56] = v56;
        const uint32_t v57 = *w57++;
        out[57] = v57;
        const uint32_t v58 = *w58++;
        out[58] = v58;
        const uint32_t v59 = *w59++;
        out[59] = v59;
        const uint32_t v60 = *w60++;
        out[60] = v60;
        const uint32_t v61 = *w61++;
        out[61] = v61;
        const uint32_t v62 = *w62++;
        out[62] = v62;
        const uint32_t v63 = *w63++;
        out[63] = v63;
        out += 64;
      }
      out = (uint32_t*) ((uintptr_t) out + extra_bytes);
      w0 = w63;
    }

    // NC remainder (1..63)
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
      out += (64 - n);

      // NR remainder has less than 64 rows so last row is not loaded
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
      const uint32_t* w31 = w30 + kc;
      if XNN_UNPREDICTABLE(n < 32) {
        w31 = w30;
      }
      const uint32_t* w32 = w31 + kc;
      if XNN_UNPREDICTABLE(n <= 32) {
        w32 = w31;
      }
      const uint32_t* w33 = w32 + kc;
      if XNN_UNPREDICTABLE(n < 34) {
        w33 = w32;
      }
      const uint32_t* w34 = w33 + kc;
      if XNN_UNPREDICTABLE(n <= 34) {
        w34 = w33;
      }
      const uint32_t* w35 = w34 + kc;
      if XNN_UNPREDICTABLE(n < 36) {
        w35 = w34;
      }
      const uint32_t* w36 = w35 + kc;
      if XNN_UNPREDICTABLE(n <= 36) {
        w36 = w35;
      }
      const uint32_t* w37 = w36 + kc;
      if XNN_UNPREDICTABLE(n < 38) {
        w37 = w36;
      }
      const uint32_t* w38 = w37 + kc;
      if XNN_UNPREDICTABLE(n <= 38) {
        w38 = w37;
      }
      const uint32_t* w39 = w38 + kc;
      if XNN_UNPREDICTABLE(n < 40) {
        w39 = w38;
      }
      const uint32_t* w40 = w39 + kc;
      if XNN_UNPREDICTABLE(n <= 40) {
        w40 = w39;
      }
      const uint32_t* w41 = w40 + kc;
      if XNN_UNPREDICTABLE(n < 42) {
        w41 = w40;
      }
      const uint32_t* w42 = w41 + kc;
      if XNN_UNPREDICTABLE(n <= 42) {
        w42 = w41;
      }
      const uint32_t* w43 = w42 + kc;
      if XNN_UNPREDICTABLE(n < 44) {
        w43 = w42;
      }
      const uint32_t* w44 = w43 + kc;
      if XNN_UNPREDICTABLE(n <= 44) {
        w44 = w43;
      }
      const uint32_t* w45 = w44 + kc;
      if XNN_UNPREDICTABLE(n < 46) {
        w45 = w44;
      }
      const uint32_t* w46 = w45 + kc;
      if XNN_UNPREDICTABLE(n <= 46) {
        w46 = w45;
      }
      const uint32_t* w47 = w46 + kc;
      if XNN_UNPREDICTABLE(n < 48) {
        w47 = w46;
      }
      const uint32_t* w48 = w47 + kc;
      if XNN_UNPREDICTABLE(n <= 48) {
        w48 = w47;
      }
      const uint32_t* w49 = w48 + kc;
      if XNN_UNPREDICTABLE(n < 50) {
        w49 = w48;
      }
      const uint32_t* w50 = w49 + kc;
      if XNN_UNPREDICTABLE(n <= 50) {
        w50 = w49;
      }
      const uint32_t* w51 = w50 + kc;
      if XNN_UNPREDICTABLE(n < 52) {
        w51 = w50;
      }
      const uint32_t* w52 = w51 + kc;
      if XNN_UNPREDICTABLE(n <= 52) {
        w52 = w51;
      }
      const uint32_t* w53 = w52 + kc;
      if XNN_UNPREDICTABLE(n < 54) {
        w53 = w52;
      }
      const uint32_t* w54 = w53 + kc;
      if XNN_UNPREDICTABLE(n <= 54) {
        w54 = w53;
      }
      const uint32_t* w55 = w54 + kc;
      if XNN_UNPREDICTABLE(n < 56) {
        w55 = w54;
      }
      const uint32_t* w56 = w55 + kc;
      if XNN_UNPREDICTABLE(n <= 56) {
        w56 = w55;
      }
      const uint32_t* w57 = w56 + kc;
      if XNN_UNPREDICTABLE(n < 58) {
        w57 = w56;
      }
      const uint32_t* w58 = w57 + kc;
      if XNN_UNPREDICTABLE(n <= 58) {
        w58 = w57;
      }
      const uint32_t* w59 = w58 + kc;
      if XNN_UNPREDICTABLE(n < 60) {
        w59 = w58;
      }
      const uint32_t* w60 = w59 + kc;
      if XNN_UNPREDICTABLE(n <= 60) {
        w60 = w59;
      }
      const uint32_t* w61 = w60 + kc;
      if XNN_UNPREDICTABLE(n < 62) {
        w61 = w60;
      }
      const uint32_t* w62 = w61 + kc;
      if XNN_UNPREDICTABLE(n <= 62) {
        w62 = w61;
      }

      // KC main loop multiple of 64x2
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
        const uint32_t v320 = w32[0];
        const uint32_t v321 = w32[1];
        w32 += 2;
        const uint32_t v330 = w33[0];
        const uint32_t v331 = w33[1];
        w33 += 2;
        const uint32_t v340 = w34[0];
        const uint32_t v341 = w34[1];
        w34 += 2;
        const uint32_t v350 = w35[0];
        const uint32_t v351 = w35[1];
        w35 += 2;
        const uint32_t v360 = w36[0];
        const uint32_t v361 = w36[1];
        w36 += 2;
        const uint32_t v370 = w37[0];
        const uint32_t v371 = w37[1];
        w37 += 2;
        const uint32_t v380 = w38[0];
        const uint32_t v381 = w38[1];
        w38 += 2;
        const uint32_t v390 = w39[0];
        const uint32_t v391 = w39[1];
        w39 += 2;
        const uint32_t v400 = w40[0];
        const uint32_t v401 = w40[1];
        w40 += 2;
        const uint32_t v410 = w41[0];
        const uint32_t v411 = w41[1];
        w41 += 2;
        const uint32_t v420 = w42[0];
        const uint32_t v421 = w42[1];
        w42 += 2;
        const uint32_t v430 = w43[0];
        const uint32_t v431 = w43[1];
        w43 += 2;
        const uint32_t v440 = w44[0];
        const uint32_t v441 = w44[1];
        w44 += 2;
        const uint32_t v450 = w45[0];
        const uint32_t v451 = w45[1];
        w45 += 2;
        const uint32_t v460 = w46[0];
        const uint32_t v461 = w46[1];
        w46 += 2;
        const uint32_t v470 = w47[0];
        const uint32_t v471 = w47[1];
        w47 += 2;
        const uint32_t v480 = w48[0];
        const uint32_t v481 = w48[1];
        w48 += 2;
        const uint32_t v490 = w49[0];
        const uint32_t v491 = w49[1];
        w49 += 2;
        const uint32_t v500 = w50[0];
        const uint32_t v501 = w50[1];
        w50 += 2;
        const uint32_t v510 = w51[0];
        const uint32_t v511 = w51[1];
        w51 += 2;
        const uint32_t v520 = w52[0];
        const uint32_t v521 = w52[1];
        w52 += 2;
        const uint32_t v530 = w53[0];
        const uint32_t v531 = w53[1];
        w53 += 2;
        const uint32_t v540 = w54[0];
        const uint32_t v541 = w54[1];
        w54 += 2;
        const uint32_t v550 = w55[0];
        const uint32_t v551 = w55[1];
        w55 += 2;
        const uint32_t v560 = w56[0];
        const uint32_t v561 = w56[1];
        w56 += 2;
        const uint32_t v570 = w57[0];
        const uint32_t v571 = w57[1];
        w57 += 2;
        const uint32_t v580 = w58[0];
        const uint32_t v581 = w58[1];
        w58 += 2;
        const uint32_t v590 = w59[0];
        const uint32_t v591 = w59[1];
        w59 += 2;
        const uint32_t v600 = w60[0];
        const uint32_t v601 = w60[1];
        w60 += 2;
        const uint32_t v610 = w61[0];
        const uint32_t v611 = w61[1];
        w61 += 2;
        const uint32_t v620 = w62[0];
        const uint32_t v621 = w62[1];
        w62 += 2;
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
        out[32] = v320;
        out[33] = v330;
        out[34] = v340;
        out[35] = v350;
        out[36] = v360;
        out[37] = v370;
        out[38] = v380;
        out[39] = v390;
        out[40] = v400;
        out[41] = v410;
        out[42] = v420;
        out[43] = v430;
        out[44] = v440;
        out[45] = v450;
        out[46] = v460;
        out[47] = v470;
        out[48] = v480;
        out[49] = v490;
        out[50] = v500;
        out[51] = v510;
        out[52] = v520;
        out[53] = v530;
        out[54] = v540;
        out[55] = v550;
        out[56] = v560;
        out[57] = v570;
        out[58] = v580;
        out[59] = v590;
        out[60] = v600;
        out[61] = v610;
        out[62] = v620;
        out[64] = v01;
        out[65] = v11;
        out[66] = v21;
        out[67] = v31;
        out[68] = v41;
        out[69] = v51;
        out[70] = v61;
        out[71] = v71;
        out[72] = v81;
        out[73] = v91;
        out[74] = v101;
        out[75] = v111;
        out[76] = v121;
        out[77] = v131;
        out[78] = v141;
        out[79] = v151;
        out[80] = v161;
        out[81] = v171;
        out[82] = v181;
        out[83] = v191;
        out[84] = v201;
        out[85] = v211;
        out[86] = v221;
        out[87] = v231;
        out[88] = v241;
        out[89] = v251;
        out[90] = v261;
        out[91] = v271;
        out[92] = v281;
        out[93] = v291;
        out[94] = v301;
        out[95] = v311;
        out[96] = v321;
        out[97] = v331;
        out[98] = v341;
        out[99] = v351;
        out[100] = v361;
        out[101] = v371;
        out[102] = v381;
        out[103] = v391;
        out[104] = v401;
        out[105] = v411;
        out[106] = v421;
        out[107] = v431;
        out[108] = v441;
        out[109] = v451;
        out[110] = v461;
        out[111] = v471;
        out[112] = v481;
        out[113] = v491;
        out[114] = v501;
        out[115] = v511;
        out[116] = v521;
        out[117] = v531;
        out[118] = v541;
        out[119] = v551;
        out[120] = v561;
        out[121] = v571;
        out[122] = v581;
        out[123] = v591;
        out[124] = v601;
        out[125] = v611;
        out[126] = v621;
        out += 128;
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
        const uint32_t v31 = *w31++;
        out[31] = v31;
        const uint32_t v32 = *w32++;
        out[32] = v32;
        const uint32_t v33 = *w33++;
        out[33] = v33;
        const uint32_t v34 = *w34++;
        out[34] = v34;
        const uint32_t v35 = *w35++;
        out[35] = v35;
        const uint32_t v36 = *w36++;
        out[36] = v36;
        const uint32_t v37 = *w37++;
        out[37] = v37;
        const uint32_t v38 = *w38++;
        out[38] = v38;
        const uint32_t v39 = *w39++;
        out[39] = v39;
        const uint32_t v40 = *w40++;
        out[40] = v40;
        const uint32_t v41 = *w41++;
        out[41] = v41;
        const uint32_t v42 = *w42++;
        out[42] = v42;
        const uint32_t v43 = *w43++;
        out[43] = v43;
        const uint32_t v44 = *w44++;
        out[44] = v44;
        const uint32_t v45 = *w45++;
        out[45] = v45;
        const uint32_t v46 = *w46++;
        out[46] = v46;
        const uint32_t v47 = *w47++;
        out[47] = v47;
        const uint32_t v48 = *w48++;
        out[48] = v48;
        const uint32_t v49 = *w49++;
        out[49] = v49;
        const uint32_t v50 = *w50++;
        out[50] = v50;
        const uint32_t v51 = *w51++;
        out[51] = v51;
        const uint32_t v52 = *w52++;
        out[52] = v52;
        const uint32_t v53 = *w53++;
        out[53] = v53;
        const uint32_t v54 = *w54++;
        out[54] = v54;
        const uint32_t v55 = *w55++;
        out[55] = v55;
        const uint32_t v56 = *w56++;
        out[56] = v56;
        const uint32_t v57 = *w57++;
        out[57] = v57;
        const uint32_t v58 = *w58++;
        out[58] = v58;
        const uint32_t v59 = *w59++;
        out[59] = v59;
        const uint32_t v60 = *w60++;
        out[60] = v60;
        const uint32_t v61 = *w61++;
        out[61] = v61;
        const uint32_t v62 = *w62++;
        out[62] = v62;
        out += 64;
      }
      out = (uint32_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
