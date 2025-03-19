// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-gio-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/packw.h"

void xnn_qs8_packw_gemm_gio_ukernel_x8c4__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
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
  assert(nr == 8);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 0): 0);

  do {
    // NC main loop multiple of 8
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 8; n -= 8) {
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
        b += 8;
      } else {
        ((int32_t*) out)[0] = 0;
        ((int32_t*) out)[1] = 0;
        ((int32_t*) out)[2] = 0;
        ((int32_t*) out)[3] = 0;
        ((int32_t*) out)[4] = 0;
        ((int32_t*) out)[5] = 0;
        ((int32_t*) out)[6] = 0;
        ((int32_t*) out)[7] = 0;
      }
      out += 8 * sizeof(int32_t);

      const int8_t* w1 = w0 + k_stride;
      const int8_t* w2 = w1 + k_stride;
      const int8_t* w3 = w2 + k_stride;
      uint32_t ksum0 = 0;
      uint32_t ksum1 = 0;
      uint32_t ksum2 = 0;
      uint32_t ksum3 = 0;
      uint32_t ksum4 = 0;
      uint32_t ksum5 = 0;
      uint32_t ksum6 = 0;
      uint32_t ksum7 = 0;

      // KC main loop multiple of 8x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const int8_t v0x0 = w0[0];
        const int8_t v1x0 = w1[0];
        const int8_t v2x0 = w2[0];
        const int8_t v3x0 = w3[0];
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v1x0;
        ksum0 += (uint32_t) v2x0;
        ksum0 += (uint32_t) v3x0;
        out[0] = v0x0;
        out[1] = v1x0;
        out[2] = v2x0;
        out[3] = v3x0;
        const int8_t v0x1 = w0[1];
        const int8_t v1x1 = w1[1];
        const int8_t v2x1 = w2[1];
        const int8_t v3x1 = w3[1];
        ksum1 += (uint32_t) v0x1;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v2x1;
        ksum1 += (uint32_t) v3x1;
        out[4] = v0x1;
        out[5] = v1x1;
        out[6] = v2x1;
        out[7] = v3x1;
        const int8_t v0x2 = w0[2];
        const int8_t v1x2 = w1[2];
        const int8_t v2x2 = w2[2];
        const int8_t v3x2 = w3[2];
        ksum2 += (uint32_t) v0x2;
        ksum2 += (uint32_t) v1x2;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v3x2;
        out[8] = v0x2;
        out[9] = v1x2;
        out[10] = v2x2;
        out[11] = v3x2;
        const int8_t v0x3 = w0[3];
        const int8_t v1x3 = w1[3];
        const int8_t v2x3 = w2[3];
        const int8_t v3x3 = w3[3];
        ksum3 += (uint32_t) v0x3;
        ksum3 += (uint32_t) v1x3;
        ksum3 += (uint32_t) v2x3;
        ksum3 += (uint32_t) v3x3;
        out[12] = v0x3;
        out[13] = v1x3;
        out[14] = v2x3;
        out[15] = v3x3;
        const int8_t v0x4 = w0[4];
        const int8_t v1x4 = w1[4];
        const int8_t v2x4 = w2[4];
        const int8_t v3x4 = w3[4];
        ksum4 += (uint32_t) v0x4;
        ksum4 += (uint32_t) v1x4;
        ksum4 += (uint32_t) v2x4;
        ksum4 += (uint32_t) v3x4;
        out[16] = v0x4;
        out[17] = v1x4;
        out[18] = v2x4;
        out[19] = v3x4;
        const int8_t v0x5 = w0[5];
        const int8_t v1x5 = w1[5];
        const int8_t v2x5 = w2[5];
        const int8_t v3x5 = w3[5];
        ksum5 += (uint32_t) v0x5;
        ksum5 += (uint32_t) v1x5;
        ksum5 += (uint32_t) v2x5;
        ksum5 += (uint32_t) v3x5;
        out[20] = v0x5;
        out[21] = v1x5;
        out[22] = v2x5;
        out[23] = v3x5;
        const int8_t v0x6 = w0[6];
        const int8_t v1x6 = w1[6];
        const int8_t v2x6 = w2[6];
        const int8_t v3x6 = w3[6];
        ksum6 += (uint32_t) v0x6;
        ksum6 += (uint32_t) v1x6;
        ksum6 += (uint32_t) v2x6;
        ksum6 += (uint32_t) v3x6;
        out[24] = v0x6;
        out[25] = v1x6;
        out[26] = v2x6;
        out[27] = v3x6;
        const int8_t v0x7 = w0[7];
        const int8_t v1x7 = w1[7];
        const int8_t v2x7 = w2[7];
        const int8_t v3x7 = w3[7];
        ksum7 += (uint32_t) v0x7;
        ksum7 += (uint32_t) v1x7;
        ksum7 += (uint32_t) v2x7;
        ksum7 += (uint32_t) v3x7;
        out[28] = v0x7;
        out[29] = v1x7;
        out[30] = v2x7;
        out[31] = v3x7;
        w0 += 4 * k_stride;
        w1 += 4 * k_stride;
        w2 += 4 * k_stride;
        w3 += 4 * k_stride;
        out += 32;
      }

      // KC remainder of 1..3
      if (k != 0) {
        assert(k >= 1 && k <= 3);
        const int8_t v0x0 = w0[0];
        const int8_t v1x0 = 1 < k ? w1[0] : 0;
        const int8_t v2x0 = 2 < k ? w2[0] : 0;
        const int8_t v3x0 = 3 < k ? w3[0] : 0;
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v1x0;
        ksum0 += (uint32_t) v2x0;
        ksum0 += (uint32_t) v3x0;
        out[0] = v0x0;
        out[1] = v1x0;
        out[2] = v2x0;
        out[3] = v3x0;
        const int8_t v0x1 = w0[1];
        const int8_t v1x1 = 1 < k ? w1[1] : 0;
        const int8_t v2x1 = 2 < k ? w2[1] : 0;
        const int8_t v3x1 = 3 < k ? w3[1] : 0;
        ksum1 += (uint32_t) v0x1;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v2x1;
        ksum1 += (uint32_t) v3x1;
        out[4] = v0x1;
        out[5] = v1x1;
        out[6] = v2x1;
        out[7] = v3x1;
        const int8_t v0x2 = w0[2];
        const int8_t v1x2 = 1 < k ? w1[2] : 0;
        const int8_t v2x2 = 2 < k ? w2[2] : 0;
        const int8_t v3x2 = 3 < k ? w3[2] : 0;
        ksum2 += (uint32_t) v0x2;
        ksum2 += (uint32_t) v1x2;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v3x2;
        out[8] = v0x2;
        out[9] = v1x2;
        out[10] = v2x2;
        out[11] = v3x2;
        const int8_t v0x3 = w0[3];
        const int8_t v1x3 = 1 < k ? w1[3] : 0;
        const int8_t v2x3 = 2 < k ? w2[3] : 0;
        const int8_t v3x3 = 3 < k ? w3[3] : 0;
        ksum3 += (uint32_t) v0x3;
        ksum3 += (uint32_t) v1x3;
        ksum3 += (uint32_t) v2x3;
        ksum3 += (uint32_t) v3x3;
        out[12] = v0x3;
        out[13] = v1x3;
        out[14] = v2x3;
        out[15] = v3x3;
        const int8_t v0x4 = w0[4];
        const int8_t v1x4 = 1 < k ? w1[4] : 0;
        const int8_t v2x4 = 2 < k ? w2[4] : 0;
        const int8_t v3x4 = 3 < k ? w3[4] : 0;
        ksum4 += (uint32_t) v0x4;
        ksum4 += (uint32_t) v1x4;
        ksum4 += (uint32_t) v2x4;
        ksum4 += (uint32_t) v3x4;
        out[16] = v0x4;
        out[17] = v1x4;
        out[18] = v2x4;
        out[19] = v3x4;
        const int8_t v0x5 = w0[5];
        const int8_t v1x5 = 1 < k ? w1[5] : 0;
        const int8_t v2x5 = 2 < k ? w2[5] : 0;
        const int8_t v3x5 = 3 < k ? w3[5] : 0;
        ksum5 += (uint32_t) v0x5;
        ksum5 += (uint32_t) v1x5;
        ksum5 += (uint32_t) v2x5;
        ksum5 += (uint32_t) v3x5;
        out[20] = v0x5;
        out[21] = v1x5;
        out[22] = v2x5;
        out[23] = v3x5;
        const int8_t v0x6 = w0[6];
        const int8_t v1x6 = 1 < k ? w1[6] : 0;
        const int8_t v2x6 = 2 < k ? w2[6] : 0;
        const int8_t v3x6 = 3 < k ? w3[6] : 0;
        ksum6 += (uint32_t) v0x6;
        ksum6 += (uint32_t) v1x6;
        ksum6 += (uint32_t) v2x6;
        ksum6 += (uint32_t) v3x6;
        out[24] = v0x6;
        out[25] = v1x6;
        out[26] = v2x6;
        out[27] = v3x6;
        const int8_t v0x7 = w0[7];
        const int8_t v1x7 = 1 < k ? w1[7] : 0;
        const int8_t v2x7 = 2 < k ? w2[7] : 0;
        const int8_t v3x7 = 3 < k ? w3[7] : 0;
        ksum7 += (uint32_t) v0x7;
        ksum7 += (uint32_t) v1x7;
        ksum7 += (uint32_t) v2x7;
        ksum7 += (uint32_t) v3x7;
        out[28] = v0x7;
        out[29] = v1x7;
        out[30] = v2x7;
        out[31] = v3x7;
        w0 += k * k_stride;
        w1 += k * k_stride;
        w2 += k * k_stride;
        w3 += k * k_stride;
        out += 32;
      }

      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      packed_b[2] -= ksum2 * izp;
      packed_b[3] -= ksum3 * izp;
      packed_b[4] -= ksum4 * izp;
      packed_b[5] -= ksum5 * izp;
      packed_b[6] -= ksum6 * izp;
      packed_b[7] -= ksum7 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w0 - kc * k_stride + 8;
    }

    // NC remainder (1..7)
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
      out += (8 - n) * sizeof(int32_t);

     // NR remainder has less than 8 rows so last row is not loaded
      const int8_t* w1 = w0 + k_stride;
      const int8_t* w2 = w1 + k_stride;
      const int8_t* w3 = w2 + k_stride;

      uint32_t ksum0 = 0;
      uint32_t ksum1 = 0;
      uint32_t ksum2 = 0;
      uint32_t ksum3 = 0;
      uint32_t ksum4 = 0;
      uint32_t ksum5 = 0;
      uint32_t ksum6 = 0;

      // KC main loop multiple of 8x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const int8_t v0x0 = w0[0];
        const int8_t v1x0 = w1[0];
        const int8_t v2x0 = w2[0];
        const int8_t v3x0 = w3[0];
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v1x0;
        ksum0 += (uint32_t) v2x0;
        ksum0 += (uint32_t) v3x0;
        out[0] = v0x0;
        out[1] = v1x0;
        out[2] = v2x0;
        out[3] = v3x0;
        const int8_t v0x1 = 1 < n ? w0[1] : 0;
        const int8_t v1x1 = 1 < n ? w1[1] : 0;
        const int8_t v2x1 = 1 < n ? w2[1] : 0;
        const int8_t v3x1 = 1 < n ? w3[1] : 0;
        ksum1 += (uint32_t) v0x1;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v2x1;
        ksum1 += (uint32_t) v3x1;
        out[4] = v0x1;
        out[5] = v1x1;
        out[6] = v2x1;
        out[7] = v3x1;
        const int8_t v0x2 = 2 < n ? w0[2] : 0;
        const int8_t v1x2 = 2 < n ? w1[2] : 0;
        const int8_t v2x2 = 2 < n ? w2[2] : 0;
        const int8_t v3x2 = 2 < n ? w3[2] : 0;
        ksum2 += (uint32_t) v0x2;
        ksum2 += (uint32_t) v1x2;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v3x2;
        out[8] = v0x2;
        out[9] = v1x2;
        out[10] = v2x2;
        out[11] = v3x2;
        const int8_t v0x3 = 3 < n ? w0[3] : 0;
        const int8_t v1x3 = 3 < n ? w1[3] : 0;
        const int8_t v2x3 = 3 < n ? w2[3] : 0;
        const int8_t v3x3 = 3 < n ? w3[3] : 0;
        ksum3 += (uint32_t) v0x3;
        ksum3 += (uint32_t) v1x3;
        ksum3 += (uint32_t) v2x3;
        ksum3 += (uint32_t) v3x3;
        out[12] = v0x3;
        out[13] = v1x3;
        out[14] = v2x3;
        out[15] = v3x3;
        const int8_t v0x4 = 4 < n ? w0[4] : 0;
        const int8_t v1x4 = 4 < n ? w1[4] : 0;
        const int8_t v2x4 = 4 < n ? w2[4] : 0;
        const int8_t v3x4 = 4 < n ? w3[4] : 0;
        ksum4 += (uint32_t) v0x4;
        ksum4 += (uint32_t) v1x4;
        ksum4 += (uint32_t) v2x4;
        ksum4 += (uint32_t) v3x4;
        out[16] = v0x4;
        out[17] = v1x4;
        out[18] = v2x4;
        out[19] = v3x4;
        const int8_t v0x5 = 5 < n ? w0[5] : 0;
        const int8_t v1x5 = 5 < n ? w1[5] : 0;
        const int8_t v2x5 = 5 < n ? w2[5] : 0;
        const int8_t v3x5 = 5 < n ? w3[5] : 0;
        ksum5 += (uint32_t) v0x5;
        ksum5 += (uint32_t) v1x5;
        ksum5 += (uint32_t) v2x5;
        ksum5 += (uint32_t) v3x5;
        out[20] = v0x5;
        out[21] = v1x5;
        out[22] = v2x5;
        out[23] = v3x5;
        const int8_t v0x6 = 6 < n ? w0[6] : 0;
        const int8_t v1x6 = 6 < n ? w1[6] : 0;
        const int8_t v2x6 = 6 < n ? w2[6] : 0;
        const int8_t v3x6 = 6 < n ? w3[6] : 0;
        ksum6 += (uint32_t) v0x6;
        ksum6 += (uint32_t) v1x6;
        ksum6 += (uint32_t) v2x6;
        ksum6 += (uint32_t) v3x6;
        out[24] = v0x6;
        out[25] = v1x6;
        out[26] = v2x6;
        out[27] = v3x6;
        w0 += 4 * k_stride;
        w1 += 4 * k_stride;
        w2 += 4 * k_stride;
        w3 += 4 * k_stride;
        out += 32;
      }

      // KC remainder of 1..3
      if (k != 0) {
        assert(k >= 1 && k <= 3);
        if (0 < n) {
          const int8_t v0x0 = w0[0];
          const int8_t v1x0 = 1 < k ? w1[0] : 0;
          const int8_t v2x0 = 2 < k ? w2[0] : 0;
          const int8_t v3x0 = 3 < k ? w3[0] : 0;
          ksum0 += (uint32_t) v0x0;
          ksum0 += (uint32_t) v1x0;
          ksum0 += (uint32_t) v2x0;
          ksum0 += (uint32_t) v3x0;
          out[0] = v0x0;
          out[1] = v1x0;
          out[2] = v2x0;
          out[3] = v3x0;
        }
        if (1 < n) {
          const int8_t v0x1 = w0[1];
          const int8_t v1x1 = 1 < k ? w1[1] : 0;
          const int8_t v2x1 = 2 < k ? w2[1] : 0;
          const int8_t v3x1 = 3 < k ? w3[1] : 0;
          ksum1 += (uint32_t) v0x1;
          ksum1 += (uint32_t) v1x1;
          ksum1 += (uint32_t) v2x1;
          ksum1 += (uint32_t) v3x1;
          out[4] = v0x1;
          out[5] = v1x1;
          out[6] = v2x1;
          out[7] = v3x1;
        }
        if (2 < n) {
          const int8_t v0x2 = w0[2];
          const int8_t v1x2 = 1 < k ? w1[2] : 0;
          const int8_t v2x2 = 2 < k ? w2[2] : 0;
          const int8_t v3x2 = 3 < k ? w3[2] : 0;
          ksum2 += (uint32_t) v0x2;
          ksum2 += (uint32_t) v1x2;
          ksum2 += (uint32_t) v2x2;
          ksum2 += (uint32_t) v3x2;
          out[8] = v0x2;
          out[9] = v1x2;
          out[10] = v2x2;
          out[11] = v3x2;
        }
        if (3 < n) {
          const int8_t v0x3 = w0[3];
          const int8_t v1x3 = 1 < k ? w1[3] : 0;
          const int8_t v2x3 = 2 < k ? w2[3] : 0;
          const int8_t v3x3 = 3 < k ? w3[3] : 0;
          ksum3 += (uint32_t) v0x3;
          ksum3 += (uint32_t) v1x3;
          ksum3 += (uint32_t) v2x3;
          ksum3 += (uint32_t) v3x3;
          out[12] = v0x3;
          out[13] = v1x3;
          out[14] = v2x3;
          out[15] = v3x3;
        }
        if (4 < n) {
          const int8_t v0x4 = w0[4];
          const int8_t v1x4 = 1 < k ? w1[4] : 0;
          const int8_t v2x4 = 2 < k ? w2[4] : 0;
          const int8_t v3x4 = 3 < k ? w3[4] : 0;
          ksum4 += (uint32_t) v0x4;
          ksum4 += (uint32_t) v1x4;
          ksum4 += (uint32_t) v2x4;
          ksum4 += (uint32_t) v3x4;
          out[16] = v0x4;
          out[17] = v1x4;
          out[18] = v2x4;
          out[19] = v3x4;
        }
        if (5 < n) {
          const int8_t v0x5 = w0[5];
          const int8_t v1x5 = 1 < k ? w1[5] : 0;
          const int8_t v2x5 = 2 < k ? w2[5] : 0;
          const int8_t v3x5 = 3 < k ? w3[5] : 0;
          ksum5 += (uint32_t) v0x5;
          ksum5 += (uint32_t) v1x5;
          ksum5 += (uint32_t) v2x5;
          ksum5 += (uint32_t) v3x5;
          out[20] = v0x5;
          out[21] = v1x5;
          out[22] = v2x5;
          out[23] = v3x5;
        }
        if (6 < n) {
          const int8_t v0x6 = w0[6];
          const int8_t v1x6 = 1 < k ? w1[6] : 0;
          const int8_t v2x6 = 2 < k ? w2[6] : 0;
          const int8_t v3x6 = 3 < k ? w3[6] : 0;
          ksum6 += (uint32_t) v0x6;
          ksum6 += (uint32_t) v1x6;
          ksum6 += (uint32_t) v2x6;
          ksum6 += (uint32_t) v3x6;
          out[24] = v0x6;
          out[25] = v1x6;
          out[26] = v2x6;
          out[27] = v3x6;
        }
        w0 += k * k_stride;
        w1 += k * k_stride;
        w2 += k * k_stride;
        w3 += k * k_stride;
        out += 32;
      }

      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      packed_b[2] -= ksum2 * izp;
      packed_b[3] -= ksum3 * izp;
      packed_b[4] -= ksum4 * izp;
      packed_b[5] -= ksum5 * izp;
      packed_b[6] -= ksum6 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
