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

void xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4(
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
  assert(nr == 8);   // This kernel is for NR=8
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    // NC main loop multiple of 8
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 8; n -= 8) {
      if XNN_LIKELY(b != NULL) {
        unaligned_store_s32(out + 0 * sizeof(int32_t), b[0]);
        unaligned_store_s32(out + 1 * sizeof(int32_t), b[1]);
        unaligned_store_s32(out + 2 * sizeof(int32_t), b[2]);
        unaligned_store_s32(out + 3 * sizeof(int32_t), b[3]);
        unaligned_store_s32(out + 4 * sizeof(int32_t), b[4]);
        unaligned_store_s32(out + 5 * sizeof(int32_t), b[5]);
        unaligned_store_s32(out + 6 * sizeof(int32_t), b[6]);
        unaligned_store_s32(out + 7 * sizeof(int32_t), b[7]);
        b += 8;
      } else {
        unaligned_store_s32(out + 0 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 1 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 2 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 3 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 4 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 5 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 6 * sizeof(int32_t), 0);
        unaligned_store_s32(out + 7 * sizeof(int32_t), 0);
      }
      out += 8 * sizeof(uint32_t);

      const int8_t* w1 = w0 + kc;
      const int8_t* w2 = w1 + kc;
      const int8_t* w3 = w2 + kc;
      const int8_t* w4 = w3 + kc;
      const int8_t* w5 = w4 + kc;
      const int8_t* w6 = w5 + kc;
      const int8_t* w7 = w6 + kc;

      // KC main loop multiple of 8x4
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
        out[0] = v00;
        out[1] = v10;
        out[2] = v20;
        out[3] = v30;
        out[4] = v40;
        out[5] = v50;
        out[6] = v60;
        out[7] = v70;
        out[8] = v01;
        out[9] = v11;
        out[10] = v21;
        out[11] = v31;
        out[12] = v41;
        out[13] = v51;
        out[14] = v61;
        out[15] = v71;
        out[16] = v02;
        out[17] = v12;
        out[18] = v22;
        out[19] = v32;
        out[20] = v42;
        out[21] = v52;
        out[22] = v62;
        out[23] = v72;
        out[24] = v03;
        out[25] = v13;
        out[26] = v23;
        out[27] = v33;
        out[28] = v43;
        out[29] = v53;
        out[30] = v63;
        out[31] = v73;
        out += 32;
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
        out += 8;
      }
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
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
      out += (8 - n) * sizeof(uint32_t);

      // NR remainder has less than 8 rows so last row is not loaded
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

      // KC main loop multiple of 8x4
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
        out[0] = v00;
        out[1] = v10;
        out[2] = v20;
        out[3] = v30;
        out[4] = v40;
        out[5] = v50;
        out[6] = v60;
        out[8] = v01;
        out[9] = v11;
        out[10] = v21;
        out[11] = v31;
        out[12] = v41;
        out[13] = v51;
        out[14] = v61;
        out[16] = v02;
        out[17] = v12;
        out[18] = v22;
        out[19] = v32;
        out[20] = v42;
        out[21] = v52;
        out[22] = v62;
        out[24] = v03;
        out[25] = v13;
        out[26] = v23;
        out[27] = v33;
        out[28] = v43;
        out[29] = v53;
        out[30] = v63;
        out += 32;
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
        out += 8;
      }
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
