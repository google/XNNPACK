// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x16-x32-packw/kr-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/packw.h"


void xnn_x16_x32_packw_gemm_goi_ukernel_x32c2__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 32);
  assert(kr == 2);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint16_t* out = (uint16_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    // NC main loop multiple of 32
    const uint16_t* w0 = (const uint16_t*) weights;
    size_t n = nc;
    for (;n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        memcpy(out, b, 32 * sizeof(uint32_t));
        b += 32;
      } else {
        memset(out, 0, 32 * sizeof(uint32_t));
      }
      out += 32 * sizeof(uint32_t) / sizeof(uint16_t);

      const uint16_t* w1 = w0 + kc;
      const uint16_t* w2 = w1 + kc;
      const uint16_t* w3 = w2 + kc;
      const uint16_t* w4 = w3 + kc;
      const uint16_t* w5 = w4 + kc;
      const uint16_t* w6 = w5 + kc;
      const uint16_t* w7 = w6 + kc;
      const uint16_t* w8 = w7 + kc;
      const uint16_t* w9 = w8 + kc;
      const uint16_t* w10 = w9 + kc;
      const uint16_t* w11 = w10 + kc;
      const uint16_t* w12 = w11 + kc;
      const uint16_t* w13 = w12 + kc;
      const uint16_t* w14 = w13 + kc;
      const uint16_t* w15 = w14 + kc;
      const uint16_t* w16 = w15 + kc;
      const uint16_t* w17 = w16 + kc;
      const uint16_t* w18 = w17 + kc;
      const uint16_t* w19 = w18 + kc;
      const uint16_t* w20 = w19 + kc;
      const uint16_t* w21 = w20 + kc;
      const uint16_t* w22 = w21 + kc;
      const uint16_t* w23 = w22 + kc;
      const uint16_t* w24 = w23 + kc;
      const uint16_t* w25 = w24 + kc;
      const uint16_t* w26 = w25 + kc;
      const uint16_t* w27 = w26 + kc;
      const uint16_t* w28 = w27 + kc;
      const uint16_t* w29 = w28 + kc;
      const uint16_t* w30 = w29 + kc;
      const uint16_t* w31 = w30 + kc;

      // KC main loop multiple of 32x2
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const uint16_t v0x0 = w0[0];
        const uint16_t v0x1 = w0[1];
        out[0] = v0x0;
        out[1] = v0x1;
        w0 += 2;
        const uint16_t v1x0 = w1[0];
        const uint16_t v1x1 = w1[1];
        out[2] = v1x0;
        out[3] = v1x1;
        w1 += 2;
        const uint16_t v2x0 = w2[0];
        const uint16_t v2x1 = w2[1];
        out[4] = v2x0;
        out[5] = v2x1;
        w2 += 2;
        const uint16_t v3x0 = w3[0];
        const uint16_t v3x1 = w3[1];
        out[6] = v3x0;
        out[7] = v3x1;
        w3 += 2;
        const uint16_t v4x0 = w4[0];
        const uint16_t v4x1 = w4[1];
        out[8] = v4x0;
        out[9] = v4x1;
        w4 += 2;
        const uint16_t v5x0 = w5[0];
        const uint16_t v5x1 = w5[1];
        out[10] = v5x0;
        out[11] = v5x1;
        w5 += 2;
        const uint16_t v6x0 = w6[0];
        const uint16_t v6x1 = w6[1];
        out[12] = v6x0;
        out[13] = v6x1;
        w6 += 2;
        const uint16_t v7x0 = w7[0];
        const uint16_t v7x1 = w7[1];
        out[14] = v7x0;
        out[15] = v7x1;
        w7 += 2;
        const uint16_t v8x0 = w8[0];
        const uint16_t v8x1 = w8[1];
        out[16] = v8x0;
        out[17] = v8x1;
        w8 += 2;
        const uint16_t v9x0 = w9[0];
        const uint16_t v9x1 = w9[1];
        out[18] = v9x0;
        out[19] = v9x1;
        w9 += 2;
        const uint16_t v10x0 = w10[0];
        const uint16_t v10x1 = w10[1];
        out[20] = v10x0;
        out[21] = v10x1;
        w10 += 2;
        const uint16_t v11x0 = w11[0];
        const uint16_t v11x1 = w11[1];
        out[22] = v11x0;
        out[23] = v11x1;
        w11 += 2;
        const uint16_t v12x0 = w12[0];
        const uint16_t v12x1 = w12[1];
        out[24] = v12x0;
        out[25] = v12x1;
        w12 += 2;
        const uint16_t v13x0 = w13[0];
        const uint16_t v13x1 = w13[1];
        out[26] = v13x0;
        out[27] = v13x1;
        w13 += 2;
        const uint16_t v14x0 = w14[0];
        const uint16_t v14x1 = w14[1];
        out[28] = v14x0;
        out[29] = v14x1;
        w14 += 2;
        const uint16_t v15x0 = w15[0];
        const uint16_t v15x1 = w15[1];
        out[30] = v15x0;
        out[31] = v15x1;
        w15 += 2;
        const uint16_t v16x0 = w16[0];
        const uint16_t v16x1 = w16[1];
        out[32] = v16x0;
        out[33] = v16x1;
        w16 += 2;
        const uint16_t v17x0 = w17[0];
        const uint16_t v17x1 = w17[1];
        out[34] = v17x0;
        out[35] = v17x1;
        w17 += 2;
        const uint16_t v18x0 = w18[0];
        const uint16_t v18x1 = w18[1];
        out[36] = v18x0;
        out[37] = v18x1;
        w18 += 2;
        const uint16_t v19x0 = w19[0];
        const uint16_t v19x1 = w19[1];
        out[38] = v19x0;
        out[39] = v19x1;
        w19 += 2;
        const uint16_t v20x0 = w20[0];
        const uint16_t v20x1 = w20[1];
        out[40] = v20x0;
        out[41] = v20x1;
        w20 += 2;
        const uint16_t v21x0 = w21[0];
        const uint16_t v21x1 = w21[1];
        out[42] = v21x0;
        out[43] = v21x1;
        w21 += 2;
        const uint16_t v22x0 = w22[0];
        const uint16_t v22x1 = w22[1];
        out[44] = v22x0;
        out[45] = v22x1;
        w22 += 2;
        const uint16_t v23x0 = w23[0];
        const uint16_t v23x1 = w23[1];
        out[46] = v23x0;
        out[47] = v23x1;
        w23 += 2;
        const uint16_t v24x0 = w24[0];
        const uint16_t v24x1 = w24[1];
        out[48] = v24x0;
        out[49] = v24x1;
        w24 += 2;
        const uint16_t v25x0 = w25[0];
        const uint16_t v25x1 = w25[1];
        out[50] = v25x0;
        out[51] = v25x1;
        w25 += 2;
        const uint16_t v26x0 = w26[0];
        const uint16_t v26x1 = w26[1];
        out[52] = v26x0;
        out[53] = v26x1;
        w26 += 2;
        const uint16_t v27x0 = w27[0];
        const uint16_t v27x1 = w27[1];
        out[54] = v27x0;
        out[55] = v27x1;
        w27 += 2;
        const uint16_t v28x0 = w28[0];
        const uint16_t v28x1 = w28[1];
        out[56] = v28x0;
        out[57] = v28x1;
        w28 += 2;
        const uint16_t v29x0 = w29[0];
        const uint16_t v29x1 = w29[1];
        out[58] = v29x0;
        out[59] = v29x1;
        w29 += 2;
        const uint16_t v30x0 = w30[0];
        const uint16_t v30x1 = w30[1];
        out[60] = v30x0;
        out[61] = v30x1;
        w30 += 2;
        const uint16_t v31x0 = w31[0];
        const uint16_t v31x1 = w31[1];
        out[62] = v31x0;
        out[63] = v31x1;
        w31 += 2;
        out += 64;
      }

      // KC remainder of 1..1
      if (k != 0) {
        assert(k >= 1 && k <= 1);
        const uint16_t v0x0 = w0[0];
        const uint16_t v0x1 = 1 < k ? w0[1] : 0;
        out[0] = v0x0;
        out[1] = v0x1;
        w0 += k;
        const uint16_t v1x0 = w1[0];
        const uint16_t v1x1 = 1 < k ? w1[1] : 0;
        out[2] = v1x0;
        out[3] = v1x1;
        w1 += k;
        const uint16_t v2x0 = w2[0];
        const uint16_t v2x1 = 1 < k ? w2[1] : 0;
        out[4] = v2x0;
        out[5] = v2x1;
        w2 += k;
        const uint16_t v3x0 = w3[0];
        const uint16_t v3x1 = 1 < k ? w3[1] : 0;
        out[6] = v3x0;
        out[7] = v3x1;
        w3 += k;
        const uint16_t v4x0 = w4[0];
        const uint16_t v4x1 = 1 < k ? w4[1] : 0;
        out[8] = v4x0;
        out[9] = v4x1;
        w4 += k;
        const uint16_t v5x0 = w5[0];
        const uint16_t v5x1 = 1 < k ? w5[1] : 0;
        out[10] = v5x0;
        out[11] = v5x1;
        w5 += k;
        const uint16_t v6x0 = w6[0];
        const uint16_t v6x1 = 1 < k ? w6[1] : 0;
        out[12] = v6x0;
        out[13] = v6x1;
        w6 += k;
        const uint16_t v7x0 = w7[0];
        const uint16_t v7x1 = 1 < k ? w7[1] : 0;
        out[14] = v7x0;
        out[15] = v7x1;
        w7 += k;
        const uint16_t v8x0 = w8[0];
        const uint16_t v8x1 = 1 < k ? w8[1] : 0;
        out[16] = v8x0;
        out[17] = v8x1;
        w8 += k;
        const uint16_t v9x0 = w9[0];
        const uint16_t v9x1 = 1 < k ? w9[1] : 0;
        out[18] = v9x0;
        out[19] = v9x1;
        w9 += k;
        const uint16_t v10x0 = w10[0];
        const uint16_t v10x1 = 1 < k ? w10[1] : 0;
        out[20] = v10x0;
        out[21] = v10x1;
        w10 += k;
        const uint16_t v11x0 = w11[0];
        const uint16_t v11x1 = 1 < k ? w11[1] : 0;
        out[22] = v11x0;
        out[23] = v11x1;
        w11 += k;
        const uint16_t v12x0 = w12[0];
        const uint16_t v12x1 = 1 < k ? w12[1] : 0;
        out[24] = v12x0;
        out[25] = v12x1;
        w12 += k;
        const uint16_t v13x0 = w13[0];
        const uint16_t v13x1 = 1 < k ? w13[1] : 0;
        out[26] = v13x0;
        out[27] = v13x1;
        w13 += k;
        const uint16_t v14x0 = w14[0];
        const uint16_t v14x1 = 1 < k ? w14[1] : 0;
        out[28] = v14x0;
        out[29] = v14x1;
        w14 += k;
        const uint16_t v15x0 = w15[0];
        const uint16_t v15x1 = 1 < k ? w15[1] : 0;
        out[30] = v15x0;
        out[31] = v15x1;
        w15 += k;
        const uint16_t v16x0 = w16[0];
        const uint16_t v16x1 = 1 < k ? w16[1] : 0;
        out[32] = v16x0;
        out[33] = v16x1;
        w16 += k;
        const uint16_t v17x0 = w17[0];
        const uint16_t v17x1 = 1 < k ? w17[1] : 0;
        out[34] = v17x0;
        out[35] = v17x1;
        w17 += k;
        const uint16_t v18x0 = w18[0];
        const uint16_t v18x1 = 1 < k ? w18[1] : 0;
        out[36] = v18x0;
        out[37] = v18x1;
        w18 += k;
        const uint16_t v19x0 = w19[0];
        const uint16_t v19x1 = 1 < k ? w19[1] : 0;
        out[38] = v19x0;
        out[39] = v19x1;
        w19 += k;
        const uint16_t v20x0 = w20[0];
        const uint16_t v20x1 = 1 < k ? w20[1] : 0;
        out[40] = v20x0;
        out[41] = v20x1;
        w20 += k;
        const uint16_t v21x0 = w21[0];
        const uint16_t v21x1 = 1 < k ? w21[1] : 0;
        out[42] = v21x0;
        out[43] = v21x1;
        w21 += k;
        const uint16_t v22x0 = w22[0];
        const uint16_t v22x1 = 1 < k ? w22[1] : 0;
        out[44] = v22x0;
        out[45] = v22x1;
        w22 += k;
        const uint16_t v23x0 = w23[0];
        const uint16_t v23x1 = 1 < k ? w23[1] : 0;
        out[46] = v23x0;
        out[47] = v23x1;
        w23 += k;
        const uint16_t v24x0 = w24[0];
        const uint16_t v24x1 = 1 < k ? w24[1] : 0;
        out[48] = v24x0;
        out[49] = v24x1;
        w24 += k;
        const uint16_t v25x0 = w25[0];
        const uint16_t v25x1 = 1 < k ? w25[1] : 0;
        out[50] = v25x0;
        out[51] = v25x1;
        w25 += k;
        const uint16_t v26x0 = w26[0];
        const uint16_t v26x1 = 1 < k ? w26[1] : 0;
        out[52] = v26x0;
        out[53] = v26x1;
        w26 += k;
        const uint16_t v27x0 = w27[0];
        const uint16_t v27x1 = 1 < k ? w27[1] : 0;
        out[54] = v27x0;
        out[55] = v27x1;
        w27 += k;
        const uint16_t v28x0 = w28[0];
        const uint16_t v28x1 = 1 < k ? w28[1] : 0;
        out[56] = v28x0;
        out[57] = v28x1;
        w28 += k;
        const uint16_t v29x0 = w29[0];
        const uint16_t v29x1 = 1 < k ? w29[1] : 0;
        out[58] = v29x0;
        out[59] = v29x1;
        w29 += k;
        const uint16_t v30x0 = w30[0];
        const uint16_t v30x1 = 1 < k ? w30[1] : 0;
        out[60] = v30x0;
        out[61] = v30x1;
        w30 += k;
        const uint16_t v31x0 = w31[0];
        const uint16_t v31x1 = 1 < k ? w31[1] : 0;
        out[62] = v31x0;
        out[63] = v31x1;
        w31 += k;
        out += 64;
      }

      out = (uint16_t*) ((uintptr_t) out + extra_bytes);
      w0 = w31;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
            *((uint32_t*) out) = *b++;
            out += sizeof(uint32_t) / sizeof(uint16_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
            *((uint32_t*) out) = 0;
            out += sizeof(uint32_t) / sizeof(uint16_t);
        } while (--nb != 0);
      }
      out += (32 - n) * sizeof(uint32_t) / sizeof(uint16_t);

     // NR remainder has less than 32 rows so last row is not loaded
      const uint16_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint16_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint16_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint16_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint16_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint16_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint16_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint16_t* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint16_t* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint16_t* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint16_t* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint16_t* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint16_t* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint16_t* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const uint16_t* w15 = w14 + kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const uint16_t* w16 = w15 + kc;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const uint16_t* w17 = w16 + kc;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const uint16_t* w18 = w17 + kc;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const uint16_t* w19 = w18 + kc;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const uint16_t* w20 = w19 + kc;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const uint16_t* w21 = w20 + kc;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const uint16_t* w22 = w21 + kc;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const uint16_t* w23 = w22 + kc;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const uint16_t* w24 = w23 + kc;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const uint16_t* w25 = w24 + kc;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const uint16_t* w26 = w25 + kc;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const uint16_t* w27 = w26 + kc;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const uint16_t* w28 = w27 + kc;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const uint16_t* w29 = w28 + kc;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const uint16_t* w30 = w29 + kc;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }

      // KC main loop multiple of 32x2
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const uint16_t v0x0 = w0[0];
        const uint16_t v0x1 = w0[1];
        out[0] = v0x0;
        out[1] = v0x1;
        w0 += 2;
        const uint16_t v1x0 = w1[0];
        const uint16_t v1x1 = w1[1];
        out[2] = v1x0;
        out[3] = v1x1;
        w1 += 2;
        const uint16_t v2x0 = w2[0];
        const uint16_t v2x1 = w2[1];
        out[4] = v2x0;
        out[5] = v2x1;
        w2 += 2;
        const uint16_t v3x0 = w3[0];
        const uint16_t v3x1 = w3[1];
        out[6] = v3x0;
        out[7] = v3x1;
        w3 += 2;
        const uint16_t v4x0 = w4[0];
        const uint16_t v4x1 = w4[1];
        out[8] = v4x0;
        out[9] = v4x1;
        w4 += 2;
        const uint16_t v5x0 = w5[0];
        const uint16_t v5x1 = w5[1];
        out[10] = v5x0;
        out[11] = v5x1;
        w5 += 2;
        const uint16_t v6x0 = w6[0];
        const uint16_t v6x1 = w6[1];
        out[12] = v6x0;
        out[13] = v6x1;
        w6 += 2;
        const uint16_t v7x0 = w7[0];
        const uint16_t v7x1 = w7[1];
        out[14] = v7x0;
        out[15] = v7x1;
        w7 += 2;
        const uint16_t v8x0 = w8[0];
        const uint16_t v8x1 = w8[1];
        out[16] = v8x0;
        out[17] = v8x1;
        w8 += 2;
        const uint16_t v9x0 = w9[0];
        const uint16_t v9x1 = w9[1];
        out[18] = v9x0;
        out[19] = v9x1;
        w9 += 2;
        const uint16_t v10x0 = w10[0];
        const uint16_t v10x1 = w10[1];
        out[20] = v10x0;
        out[21] = v10x1;
        w10 += 2;
        const uint16_t v11x0 = w11[0];
        const uint16_t v11x1 = w11[1];
        out[22] = v11x0;
        out[23] = v11x1;
        w11 += 2;
        const uint16_t v12x0 = w12[0];
        const uint16_t v12x1 = w12[1];
        out[24] = v12x0;
        out[25] = v12x1;
        w12 += 2;
        const uint16_t v13x0 = w13[0];
        const uint16_t v13x1 = w13[1];
        out[26] = v13x0;
        out[27] = v13x1;
        w13 += 2;
        const uint16_t v14x0 = w14[0];
        const uint16_t v14x1 = w14[1];
        out[28] = v14x0;
        out[29] = v14x1;
        w14 += 2;
        const uint16_t v15x0 = w15[0];
        const uint16_t v15x1 = w15[1];
        out[30] = v15x0;
        out[31] = v15x1;
        w15 += 2;
        const uint16_t v16x0 = w16[0];
        const uint16_t v16x1 = w16[1];
        out[32] = v16x0;
        out[33] = v16x1;
        w16 += 2;
        const uint16_t v17x0 = w17[0];
        const uint16_t v17x1 = w17[1];
        out[34] = v17x0;
        out[35] = v17x1;
        w17 += 2;
        const uint16_t v18x0 = w18[0];
        const uint16_t v18x1 = w18[1];
        out[36] = v18x0;
        out[37] = v18x1;
        w18 += 2;
        const uint16_t v19x0 = w19[0];
        const uint16_t v19x1 = w19[1];
        out[38] = v19x0;
        out[39] = v19x1;
        w19 += 2;
        const uint16_t v20x0 = w20[0];
        const uint16_t v20x1 = w20[1];
        out[40] = v20x0;
        out[41] = v20x1;
        w20 += 2;
        const uint16_t v21x0 = w21[0];
        const uint16_t v21x1 = w21[1];
        out[42] = v21x0;
        out[43] = v21x1;
        w21 += 2;
        const uint16_t v22x0 = w22[0];
        const uint16_t v22x1 = w22[1];
        out[44] = v22x0;
        out[45] = v22x1;
        w22 += 2;
        const uint16_t v23x0 = w23[0];
        const uint16_t v23x1 = w23[1];
        out[46] = v23x0;
        out[47] = v23x1;
        w23 += 2;
        const uint16_t v24x0 = w24[0];
        const uint16_t v24x1 = w24[1];
        out[48] = v24x0;
        out[49] = v24x1;
        w24 += 2;
        const uint16_t v25x0 = w25[0];
        const uint16_t v25x1 = w25[1];
        out[50] = v25x0;
        out[51] = v25x1;
        w25 += 2;
        const uint16_t v26x0 = w26[0];
        const uint16_t v26x1 = w26[1];
        out[52] = v26x0;
        out[53] = v26x1;
        w26 += 2;
        const uint16_t v27x0 = w27[0];
        const uint16_t v27x1 = w27[1];
        out[54] = v27x0;
        out[55] = v27x1;
        w27 += 2;
        const uint16_t v28x0 = w28[0];
        const uint16_t v28x1 = w28[1];
        out[56] = v28x0;
        out[57] = v28x1;
        w28 += 2;
        const uint16_t v29x0 = w29[0];
        const uint16_t v29x1 = w29[1];
        out[58] = v29x0;
        out[59] = v29x1;
        w29 += 2;
        const uint16_t v30x0 = w30[0];
        const uint16_t v30x1 = w30[1];
        out[60] = v30x0;
        out[61] = v30x1;
        w30 += 2;
        out += 64;
      }

      // KC remainder of 1..1
      if (k != 0) {
        assert(k >= 1 && k <= 1);
        const uint16_t v0x0 = w0[0];
        const uint16_t v0x1 = 1 < k ? w0[1] : 0;
        out[0] = v0x0;
        out[1] = v0x1;
        w0 += k;
        const uint16_t v1x0 = w1[0];
        const uint16_t v1x1 = 1 < k ? w1[1] : 0;
        out[2] = v1x0;
        out[3] = v1x1;
        w1 += k;
        const uint16_t v2x0 = w2[0];
        const uint16_t v2x1 = 1 < k ? w2[1] : 0;
        out[4] = v2x0;
        out[5] = v2x1;
        w2 += k;
        const uint16_t v3x0 = w3[0];
        const uint16_t v3x1 = 1 < k ? w3[1] : 0;
        out[6] = v3x0;
        out[7] = v3x1;
        w3 += k;
        const uint16_t v4x0 = w4[0];
        const uint16_t v4x1 = 1 < k ? w4[1] : 0;
        out[8] = v4x0;
        out[9] = v4x1;
        w4 += k;
        const uint16_t v5x0 = w5[0];
        const uint16_t v5x1 = 1 < k ? w5[1] : 0;
        out[10] = v5x0;
        out[11] = v5x1;
        w5 += k;
        const uint16_t v6x0 = w6[0];
        const uint16_t v6x1 = 1 < k ? w6[1] : 0;
        out[12] = v6x0;
        out[13] = v6x1;
        w6 += k;
        const uint16_t v7x0 = w7[0];
        const uint16_t v7x1 = 1 < k ? w7[1] : 0;
        out[14] = v7x0;
        out[15] = v7x1;
        w7 += k;
        const uint16_t v8x0 = w8[0];
        const uint16_t v8x1 = 1 < k ? w8[1] : 0;
        out[16] = v8x0;
        out[17] = v8x1;
        w8 += k;
        const uint16_t v9x0 = w9[0];
        const uint16_t v9x1 = 1 < k ? w9[1] : 0;
        out[18] = v9x0;
        out[19] = v9x1;
        w9 += k;
        const uint16_t v10x0 = w10[0];
        const uint16_t v10x1 = 1 < k ? w10[1] : 0;
        out[20] = v10x0;
        out[21] = v10x1;
        w10 += k;
        const uint16_t v11x0 = w11[0];
        const uint16_t v11x1 = 1 < k ? w11[1] : 0;
        out[22] = v11x0;
        out[23] = v11x1;
        w11 += k;
        const uint16_t v12x0 = w12[0];
        const uint16_t v12x1 = 1 < k ? w12[1] : 0;
        out[24] = v12x0;
        out[25] = v12x1;
        w12 += k;
        const uint16_t v13x0 = w13[0];
        const uint16_t v13x1 = 1 < k ? w13[1] : 0;
        out[26] = v13x0;
        out[27] = v13x1;
        w13 += k;
        const uint16_t v14x0 = w14[0];
        const uint16_t v14x1 = 1 < k ? w14[1] : 0;
        out[28] = v14x0;
        out[29] = v14x1;
        w14 += k;
        const uint16_t v15x0 = w15[0];
        const uint16_t v15x1 = 1 < k ? w15[1] : 0;
        out[30] = v15x0;
        out[31] = v15x1;
        w15 += k;
        const uint16_t v16x0 = w16[0];
        const uint16_t v16x1 = 1 < k ? w16[1] : 0;
        out[32] = v16x0;
        out[33] = v16x1;
        w16 += k;
        const uint16_t v17x0 = w17[0];
        const uint16_t v17x1 = 1 < k ? w17[1] : 0;
        out[34] = v17x0;
        out[35] = v17x1;
        w17 += k;
        const uint16_t v18x0 = w18[0];
        const uint16_t v18x1 = 1 < k ? w18[1] : 0;
        out[36] = v18x0;
        out[37] = v18x1;
        w18 += k;
        const uint16_t v19x0 = w19[0];
        const uint16_t v19x1 = 1 < k ? w19[1] : 0;
        out[38] = v19x0;
        out[39] = v19x1;
        w19 += k;
        const uint16_t v20x0 = w20[0];
        const uint16_t v20x1 = 1 < k ? w20[1] : 0;
        out[40] = v20x0;
        out[41] = v20x1;
        w20 += k;
        const uint16_t v21x0 = w21[0];
        const uint16_t v21x1 = 1 < k ? w21[1] : 0;
        out[42] = v21x0;
        out[43] = v21x1;
        w21 += k;
        const uint16_t v22x0 = w22[0];
        const uint16_t v22x1 = 1 < k ? w22[1] : 0;
        out[44] = v22x0;
        out[45] = v22x1;
        w22 += k;
        const uint16_t v23x0 = w23[0];
        const uint16_t v23x1 = 1 < k ? w23[1] : 0;
        out[46] = v23x0;
        out[47] = v23x1;
        w23 += k;
        const uint16_t v24x0 = w24[0];
        const uint16_t v24x1 = 1 < k ? w24[1] : 0;
        out[48] = v24x0;
        out[49] = v24x1;
        w24 += k;
        const uint16_t v25x0 = w25[0];
        const uint16_t v25x1 = 1 < k ? w25[1] : 0;
        out[50] = v25x0;
        out[51] = v25x1;
        w25 += k;
        const uint16_t v26x0 = w26[0];
        const uint16_t v26x1 = 1 < k ? w26[1] : 0;
        out[52] = v26x0;
        out[53] = v26x1;
        w26 += k;
        const uint16_t v27x0 = w27[0];
        const uint16_t v27x1 = 1 < k ? w27[1] : 0;
        out[54] = v27x0;
        out[55] = v27x1;
        w27 += k;
        const uint16_t v28x0 = w28[0];
        const uint16_t v28x1 = 1 < k ? w28[1] : 0;
        out[56] = v28x0;
        out[57] = v28x1;
        w28 += k;
        const uint16_t v29x0 = w29[0];
        const uint16_t v29x1 = 1 < k ? w29[1] : 0;
        out[58] = v29x0;
        out[59] = v29x1;
        w29 += k;
        const uint16_t v30x0 = w30[0];
        const uint16_t v30x1 = 1 < k ? w30[1] : 0;
        out[60] = v30x0;
        out[61] = v30x1;
        w30 += k;
        out += 64;
      }

      out = (uint16_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
