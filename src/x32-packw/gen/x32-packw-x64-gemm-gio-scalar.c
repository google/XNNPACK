// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-packw/gio-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/packw.h"


void xnn_x32_packw_gemm_gio_ukernel_x64__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
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
  assert(nr == 64);   // This kernel is for NR=64
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const uint32_t* b = bias;
  uint32_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 64
    const uint32_t* w = weights;
    size_t n = nc;

    for (; n >= 64; n -= 64) {
      if XNN_LIKELY(b != NULL) {
        const uint32_t v0 = b[0];
        const uint32_t v1 = b[1];
        const uint32_t v2 = b[2];
        const uint32_t v3 = b[3];
        const uint32_t v4 = b[4];
        const uint32_t v5 = b[5];
        const uint32_t v6 = b[6];
        const uint32_t v7 = b[7];
        const uint32_t v8 = b[8];
        const uint32_t v9 = b[9];
        const uint32_t v10 = b[10];
        const uint32_t v11 = b[11];
        const uint32_t v12 = b[12];
        const uint32_t v13 = b[13];
        const uint32_t v14 = b[14];
        const uint32_t v15 = b[15];
        const uint32_t v16 = b[16];
        const uint32_t v17 = b[17];
        const uint32_t v18 = b[18];
        const uint32_t v19 = b[19];
        const uint32_t v20 = b[20];
        const uint32_t v21 = b[21];
        const uint32_t v22 = b[22];
        const uint32_t v23 = b[23];
        const uint32_t v24 = b[24];
        const uint32_t v25 = b[25];
        const uint32_t v26 = b[26];
        const uint32_t v27 = b[27];
        const uint32_t v28 = b[28];
        const uint32_t v29 = b[29];
        const uint32_t v30 = b[30];
        const uint32_t v31 = b[31];
        const uint32_t v32 = b[32];
        const uint32_t v33 = b[33];
        const uint32_t v34 = b[34];
        const uint32_t v35 = b[35];
        const uint32_t v36 = b[36];
        const uint32_t v37 = b[37];
        const uint32_t v38 = b[38];
        const uint32_t v39 = b[39];
        const uint32_t v40 = b[40];
        const uint32_t v41 = b[41];
        const uint32_t v42 = b[42];
        const uint32_t v43 = b[43];
        const uint32_t v44 = b[44];
        const uint32_t v45 = b[45];
        const uint32_t v46 = b[46];
        const uint32_t v47 = b[47];
        const uint32_t v48 = b[48];
        const uint32_t v49 = b[49];
        const uint32_t v50 = b[50];
        const uint32_t v51 = b[51];
        const uint32_t v52 = b[52];
        const uint32_t v53 = b[53];
        const uint32_t v54 = b[54];
        const uint32_t v55 = b[55];
        const uint32_t v56 = b[56];
        const uint32_t v57 = b[57];
        const uint32_t v58 = b[58];
        const uint32_t v59 = b[59];
        const uint32_t v60 = b[60];
        const uint32_t v61 = b[61];
        const uint32_t v62 = b[62];
        const uint32_t v63 = b[63];
        packed_w[0] = v0;
        packed_w[1] = v1;
        packed_w[2] = v2;
        packed_w[3] = v3;
        packed_w[4] = v4;
        packed_w[5] = v5;
        packed_w[6] = v6;
        packed_w[7] = v7;
        packed_w[8] = v8;
        packed_w[9] = v9;
        packed_w[10] = v10;
        packed_w[11] = v11;
        packed_w[12] = v12;
        packed_w[13] = v13;
        packed_w[14] = v14;
        packed_w[15] = v15;
        packed_w[16] = v16;
        packed_w[17] = v17;
        packed_w[18] = v18;
        packed_w[19] = v19;
        packed_w[20] = v20;
        packed_w[21] = v21;
        packed_w[22] = v22;
        packed_w[23] = v23;
        packed_w[24] = v24;
        packed_w[25] = v25;
        packed_w[26] = v26;
        packed_w[27] = v27;
        packed_w[28] = v28;
        packed_w[29] = v29;
        packed_w[30] = v30;
        packed_w[31] = v31;
        packed_w[32] = v32;
        packed_w[33] = v33;
        packed_w[34] = v34;
        packed_w[35] = v35;
        packed_w[36] = v36;
        packed_w[37] = v37;
        packed_w[38] = v38;
        packed_w[39] = v39;
        packed_w[40] = v40;
        packed_w[41] = v41;
        packed_w[42] = v42;
        packed_w[43] = v43;
        packed_w[44] = v44;
        packed_w[45] = v45;
        packed_w[46] = v46;
        packed_w[47] = v47;
        packed_w[48] = v48;
        packed_w[49] = v49;
        packed_w[50] = v50;
        packed_w[51] = v51;
        packed_w[52] = v52;
        packed_w[53] = v53;
        packed_w[54] = v54;
        packed_w[55] = v55;
        packed_w[56] = v56;
        packed_w[57] = v57;
        packed_w[58] = v58;
        packed_w[59] = v59;
        packed_w[60] = v60;
        packed_w[61] = v61;
        packed_w[62] = v62;
        packed_w[63] = v63;
        b += 64;
      } else {
        packed_w[0] = 0;
        packed_w[1] = 0;
        packed_w[2] = 0;
        packed_w[3] = 0;
        packed_w[4] = 0;
        packed_w[5] = 0;
        packed_w[6] = 0;
        packed_w[7] = 0;
        packed_w[8] = 0;
        packed_w[9] = 0;
        packed_w[10] = 0;
        packed_w[11] = 0;
        packed_w[12] = 0;
        packed_w[13] = 0;
        packed_w[14] = 0;
        packed_w[15] = 0;
        packed_w[16] = 0;
        packed_w[17] = 0;
        packed_w[18] = 0;
        packed_w[19] = 0;
        packed_w[20] = 0;
        packed_w[21] = 0;
        packed_w[22] = 0;
        packed_w[23] = 0;
        packed_w[24] = 0;
        packed_w[25] = 0;
        packed_w[26] = 0;
        packed_w[27] = 0;
        packed_w[28] = 0;
        packed_w[29] = 0;
        packed_w[30] = 0;
        packed_w[31] = 0;
        packed_w[32] = 0;
        packed_w[33] = 0;
        packed_w[34] = 0;
        packed_w[35] = 0;
        packed_w[36] = 0;
        packed_w[37] = 0;
        packed_w[38] = 0;
        packed_w[39] = 0;
        packed_w[40] = 0;
        packed_w[41] = 0;
        packed_w[42] = 0;
        packed_w[43] = 0;
        packed_w[44] = 0;
        packed_w[45] = 0;
        packed_w[46] = 0;
        packed_w[47] = 0;
        packed_w[48] = 0;
        packed_w[49] = 0;
        packed_w[50] = 0;
        packed_w[51] = 0;
        packed_w[52] = 0;
        packed_w[53] = 0;
        packed_w[54] = 0;
        packed_w[55] = 0;
        packed_w[56] = 0;
        packed_w[57] = 0;
        packed_w[58] = 0;
        packed_w[59] = 0;
        packed_w[60] = 0;
        packed_w[61] = 0;
        packed_w[62] = 0;
        packed_w[63] = 0;
      }
      packed_w += 64;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const uint32_t v0 = w[0];
        const uint32_t v1 = w[1];
        const uint32_t v2 = w[2];
        const uint32_t v3 = w[3];
        const uint32_t v4 = w[4];
        const uint32_t v5 = w[5];
        const uint32_t v6 = w[6];
        const uint32_t v7 = w[7];
        const uint32_t v8 = w[8];
        const uint32_t v9 = w[9];
        const uint32_t v10 = w[10];
        const uint32_t v11 = w[11];
        const uint32_t v12 = w[12];
        const uint32_t v13 = w[13];
        const uint32_t v14 = w[14];
        const uint32_t v15 = w[15];
        const uint32_t v16 = w[16];
        const uint32_t v17 = w[17];
        const uint32_t v18 = w[18];
        const uint32_t v19 = w[19];
        const uint32_t v20 = w[20];
        const uint32_t v21 = w[21];
        const uint32_t v22 = w[22];
        const uint32_t v23 = w[23];
        const uint32_t v24 = w[24];
        const uint32_t v25 = w[25];
        const uint32_t v26 = w[26];
        const uint32_t v27 = w[27];
        const uint32_t v28 = w[28];
        const uint32_t v29 = w[29];
        const uint32_t v30 = w[30];
        const uint32_t v31 = w[31];
        const uint32_t v32 = w[32];
        const uint32_t v33 = w[33];
        const uint32_t v34 = w[34];
        const uint32_t v35 = w[35];
        const uint32_t v36 = w[36];
        const uint32_t v37 = w[37];
        const uint32_t v38 = w[38];
        const uint32_t v39 = w[39];
        const uint32_t v40 = w[40];
        const uint32_t v41 = w[41];
        const uint32_t v42 = w[42];
        const uint32_t v43 = w[43];
        const uint32_t v44 = w[44];
        const uint32_t v45 = w[45];
        const uint32_t v46 = w[46];
        const uint32_t v47 = w[47];
        const uint32_t v48 = w[48];
        const uint32_t v49 = w[49];
        const uint32_t v50 = w[50];
        const uint32_t v51 = w[51];
        const uint32_t v52 = w[52];
        const uint32_t v53 = w[53];
        const uint32_t v54 = w[54];
        const uint32_t v55 = w[55];
        const uint32_t v56 = w[56];
        const uint32_t v57 = w[57];
        const uint32_t v58 = w[58];
        const uint32_t v59 = w[59];
        const uint32_t v60 = w[60];
        const uint32_t v61 = w[61];
        const uint32_t v62 = w[62];
        const uint32_t v63 = w[63];
        packed_w[0] = v0;
        packed_w[1] = v1;
        packed_w[2] = v2;
        packed_w[3] = v3;
        packed_w[4] = v4;
        packed_w[5] = v5;
        packed_w[6] = v6;
        packed_w[7] = v7;
        packed_w[8] = v8;
        packed_w[9] = v9;
        packed_w[10] = v10;
        packed_w[11] = v11;
        packed_w[12] = v12;
        packed_w[13] = v13;
        packed_w[14] = v14;
        packed_w[15] = v15;
        packed_w[16] = v16;
        packed_w[17] = v17;
        packed_w[18] = v18;
        packed_w[19] = v19;
        packed_w[20] = v20;
        packed_w[21] = v21;
        packed_w[22] = v22;
        packed_w[23] = v23;
        packed_w[24] = v24;
        packed_w[25] = v25;
        packed_w[26] = v26;
        packed_w[27] = v27;
        packed_w[28] = v28;
        packed_w[29] = v29;
        packed_w[30] = v30;
        packed_w[31] = v31;
        packed_w[32] = v32;
        packed_w[33] = v33;
        packed_w[34] = v34;
        packed_w[35] = v35;
        packed_w[36] = v36;
        packed_w[37] = v37;
        packed_w[38] = v38;
        packed_w[39] = v39;
        packed_w[40] = v40;
        packed_w[41] = v41;
        packed_w[42] = v42;
        packed_w[43] = v43;
        packed_w[44] = v44;
        packed_w[45] = v45;
        packed_w[46] = v46;
        packed_w[47] = v47;
        packed_w[48] = v48;
        packed_w[49] = v49;
        packed_w[50] = v50;
        packed_w[51] = v51;
        packed_w[52] = v52;
        packed_w[53] = v53;
        packed_w[54] = v54;
        packed_w[55] = v55;
        packed_w[56] = v56;
        packed_w[57] = v57;
        packed_w[58] = v58;
        packed_w[59] = v59;
        packed_w[60] = v60;
        packed_w[61] = v61;
        packed_w[62] = v62;
        packed_w[63] = v63;
        w += k_stride;
        packed_w += 64;
      }
      w = w - kc * k_stride + 64;  // Advance to next column of 64 uint32_t
    }

    // NC remainder (1..63)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 63);

      if XNN_LIKELY(b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          packed_w[i] = b[i];
        }
        b += n;
      } else {
        packed_w[0] = 0;
        packed_w[1] = 0;
        packed_w[2] = 0;
        packed_w[3] = 0;
        packed_w[4] = 0;
        packed_w[5] = 0;
        packed_w[6] = 0;
        packed_w[7] = 0;
        packed_w[8] = 0;
        packed_w[9] = 0;
        packed_w[10] = 0;
        packed_w[11] = 0;
        packed_w[12] = 0;
        packed_w[13] = 0;
        packed_w[14] = 0;
        packed_w[15] = 0;
        packed_w[16] = 0;
        packed_w[17] = 0;
        packed_w[18] = 0;
        packed_w[19] = 0;
        packed_w[20] = 0;
        packed_w[21] = 0;
        packed_w[22] = 0;
        packed_w[23] = 0;
        packed_w[24] = 0;
        packed_w[25] = 0;
        packed_w[26] = 0;
        packed_w[27] = 0;
        packed_w[28] = 0;
        packed_w[29] = 0;
        packed_w[30] = 0;
        packed_w[31] = 0;
        packed_w[32] = 0;
        packed_w[33] = 0;
        packed_w[34] = 0;
        packed_w[35] = 0;
        packed_w[36] = 0;
        packed_w[37] = 0;
        packed_w[38] = 0;
        packed_w[39] = 0;
        packed_w[40] = 0;
        packed_w[41] = 0;
        packed_w[42] = 0;
        packed_w[43] = 0;
        packed_w[44] = 0;
        packed_w[45] = 0;
        packed_w[46] = 0;
        packed_w[47] = 0;
        packed_w[48] = 0;
        packed_w[49] = 0;
        packed_w[50] = 0;
        packed_w[51] = 0;
        packed_w[52] = 0;
        packed_w[53] = 0;
        packed_w[54] = 0;
        packed_w[55] = 0;
        packed_w[56] = 0;
        packed_w[57] = 0;
        packed_w[58] = 0;
        packed_w[59] = 0;
        packed_w[60] = 0;
        packed_w[61] = 0;
        packed_w[62] = 0;
        packed_w[63] = 0;
      }
      packed_w += 64;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        for (size_t i = 0; i < n; ++i) {
          packed_w[i] = w[i];
        }
        w += k_stride;
        packed_w += 64;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
