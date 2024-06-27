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

#include "xnnpack/math.h"
#include "xnnpack/packw.h"



void xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_u4(
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
  assert(nr == 16);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint32_t* out = (uint32_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    // NC main loop multiple of 16
    const uint32_t* w0 = (const uint32_t*) weights;
    size_t n = nc;
    for (;n >= 16; n -= 16) {
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
        b += 16;
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
      }
      out += 16;

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

      // KC main loop multiple of 16x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const uint32_t v00 = w0[0];
        const uint32_t v01 = w0[1];
        const uint32_t v02 = w0[2];
        const uint32_t v03 = w0[3];
        w0 += 4;
        const uint32_t v10 = w1[0];
        const uint32_t v11 = w1[1];
        const uint32_t v12 = w1[2];
        const uint32_t v13 = w1[3];
        w1 += 4;
        const uint32_t v20 = w2[0];
        const uint32_t v21 = w2[1];
        const uint32_t v22 = w2[2];
        const uint32_t v23 = w2[3];
        w2 += 4;
        const uint32_t v30 = w3[0];
        const uint32_t v31 = w3[1];
        const uint32_t v32 = w3[2];
        const uint32_t v33 = w3[3];
        w3 += 4;
        const uint32_t v40 = w4[0];
        const uint32_t v41 = w4[1];
        const uint32_t v42 = w4[2];
        const uint32_t v43 = w4[3];
        w4 += 4;
        const uint32_t v50 = w5[0];
        const uint32_t v51 = w5[1];
        const uint32_t v52 = w5[2];
        const uint32_t v53 = w5[3];
        w5 += 4;
        const uint32_t v60 = w6[0];
        const uint32_t v61 = w6[1];
        const uint32_t v62 = w6[2];
        const uint32_t v63 = w6[3];
        w6 += 4;
        const uint32_t v70 = w7[0];
        const uint32_t v71 = w7[1];
        const uint32_t v72 = w7[2];
        const uint32_t v73 = w7[3];
        w7 += 4;
        const uint32_t v80 = w8[0];
        const uint32_t v81 = w8[1];
        const uint32_t v82 = w8[2];
        const uint32_t v83 = w8[3];
        w8 += 4;
        const uint32_t v90 = w9[0];
        const uint32_t v91 = w9[1];
        const uint32_t v92 = w9[2];
        const uint32_t v93 = w9[3];
        w9 += 4;
        const uint32_t v100 = w10[0];
        const uint32_t v101 = w10[1];
        const uint32_t v102 = w10[2];
        const uint32_t v103 = w10[3];
        w10 += 4;
        const uint32_t v110 = w11[0];
        const uint32_t v111 = w11[1];
        const uint32_t v112 = w11[2];
        const uint32_t v113 = w11[3];
        w11 += 4;
        const uint32_t v120 = w12[0];
        const uint32_t v121 = w12[1];
        const uint32_t v122 = w12[2];
        const uint32_t v123 = w12[3];
        w12 += 4;
        const uint32_t v130 = w13[0];
        const uint32_t v131 = w13[1];
        const uint32_t v132 = w13[2];
        const uint32_t v133 = w13[3];
        w13 += 4;
        const uint32_t v140 = w14[0];
        const uint32_t v141 = w14[1];
        const uint32_t v142 = w14[2];
        const uint32_t v143 = w14[3];
        w14 += 4;
        const uint32_t v150 = w15[0];
        const uint32_t v151 = w15[1];
        const uint32_t v152 = w15[2];
        const uint32_t v153 = w15[3];
        w15 += 4;
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
        out[16] = v01;
        out[17] = v11;
        out[18] = v21;
        out[19] = v31;
        out[20] = v41;
        out[21] = v51;
        out[22] = v61;
        out[23] = v71;
        out[24] = v81;
        out[25] = v91;
        out[26] = v101;
        out[27] = v111;
        out[28] = v121;
        out[29] = v131;
        out[30] = v141;
        out[31] = v151;
        out[32] = v02;
        out[33] = v12;
        out[34] = v22;
        out[35] = v32;
        out[36] = v42;
        out[37] = v52;
        out[38] = v62;
        out[39] = v72;
        out[40] = v82;
        out[41] = v92;
        out[42] = v102;
        out[43] = v112;
        out[44] = v122;
        out[45] = v132;
        out[46] = v142;
        out[47] = v152;
        out[48] = v03;
        out[49] = v13;
        out[50] = v23;
        out[51] = v33;
        out[52] = v43;
        out[53] = v53;
        out[54] = v63;
        out[55] = v73;
        out[56] = v83;
        out[57] = v93;
        out[58] = v103;
        out[59] = v113;
        out[60] = v123;
        out[61] = v133;
        out[62] = v143;
        out[63] = v153;
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
        out += 16;
      }
      out = (uint32_t*) ((uintptr_t) out + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
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
      out += (16 - n);

      // NR remainder has less than 16 rows so last row is not loaded
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

      // KC main loop multiple of 16x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const uint32_t v00 = w0[0];
        const uint32_t v01 = w0[1];
        const uint32_t v02 = w0[2];
        const uint32_t v03 = w0[3];
        w0 += 4;
        const uint32_t v10 = w1[0];
        const uint32_t v11 = w1[1];
        const uint32_t v12 = w1[2];
        const uint32_t v13 = w1[3];
        w1 += 4;
        const uint32_t v20 = w2[0];
        const uint32_t v21 = w2[1];
        const uint32_t v22 = w2[2];
        const uint32_t v23 = w2[3];
        w2 += 4;
        const uint32_t v30 = w3[0];
        const uint32_t v31 = w3[1];
        const uint32_t v32 = w3[2];
        const uint32_t v33 = w3[3];
        w3 += 4;
        const uint32_t v40 = w4[0];
        const uint32_t v41 = w4[1];
        const uint32_t v42 = w4[2];
        const uint32_t v43 = w4[3];
        w4 += 4;
        const uint32_t v50 = w5[0];
        const uint32_t v51 = w5[1];
        const uint32_t v52 = w5[2];
        const uint32_t v53 = w5[3];
        w5 += 4;
        const uint32_t v60 = w6[0];
        const uint32_t v61 = w6[1];
        const uint32_t v62 = w6[2];
        const uint32_t v63 = w6[3];
        w6 += 4;
        const uint32_t v70 = w7[0];
        const uint32_t v71 = w7[1];
        const uint32_t v72 = w7[2];
        const uint32_t v73 = w7[3];
        w7 += 4;
        const uint32_t v80 = w8[0];
        const uint32_t v81 = w8[1];
        const uint32_t v82 = w8[2];
        const uint32_t v83 = w8[3];
        w8 += 4;
        const uint32_t v90 = w9[0];
        const uint32_t v91 = w9[1];
        const uint32_t v92 = w9[2];
        const uint32_t v93 = w9[3];
        w9 += 4;
        const uint32_t v100 = w10[0];
        const uint32_t v101 = w10[1];
        const uint32_t v102 = w10[2];
        const uint32_t v103 = w10[3];
        w10 += 4;
        const uint32_t v110 = w11[0];
        const uint32_t v111 = w11[1];
        const uint32_t v112 = w11[2];
        const uint32_t v113 = w11[3];
        w11 += 4;
        const uint32_t v120 = w12[0];
        const uint32_t v121 = w12[1];
        const uint32_t v122 = w12[2];
        const uint32_t v123 = w12[3];
        w12 += 4;
        const uint32_t v130 = w13[0];
        const uint32_t v131 = w13[1];
        const uint32_t v132 = w13[2];
        const uint32_t v133 = w13[3];
        w13 += 4;
        const uint32_t v140 = w14[0];
        const uint32_t v141 = w14[1];
        const uint32_t v142 = w14[2];
        const uint32_t v143 = w14[3];
        w14 += 4;
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
        out[16] = v01;
        out[17] = v11;
        out[18] = v21;
        out[19] = v31;
        out[20] = v41;
        out[21] = v51;
        out[22] = v61;
        out[23] = v71;
        out[24] = v81;
        out[25] = v91;
        out[26] = v101;
        out[27] = v111;
        out[28] = v121;
        out[29] = v131;
        out[30] = v141;
        out[32] = v02;
        out[33] = v12;
        out[34] = v22;
        out[35] = v32;
        out[36] = v42;
        out[37] = v52;
        out[38] = v62;
        out[39] = v72;
        out[40] = v82;
        out[41] = v92;
        out[42] = v102;
        out[43] = v112;
        out[44] = v122;
        out[45] = v132;
        out[46] = v142;
        out[48] = v03;
        out[49] = v13;
        out[50] = v23;
        out[51] = v33;
        out[52] = v43;
        out[53] = v53;
        out[54] = v63;
        out[55] = v73;
        out[56] = v83;
        out[57] = v93;
        out[58] = v103;
        out[59] = v113;
        out[60] = v123;
        out[61] = v133;
        out[62] = v143;
        out += 64;
      }

      // KC remainder of 1..3
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
        out += 16;
      }
      out = (uint32_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
