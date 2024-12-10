// Auto-generated file. Do not edit!
//   Template: src/qb4-packw/kr-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "xnnpack/packw.h"

void xnn_qb4_packw_gemm_goi_ukernel_x16c4__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t bl,
  const uint8_t* weights,
  const int32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes_bl,
  size_t extra_bytes_n,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(extra_bytes_bl == nr * sizeof(uint16_t));
  assert(extra_bytes_n == nr * sizeof(float));
  assert(params != NULL);
  assert(kc % bl == 0);
  size_t num_blocks = kc / bl;

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = (uint32_t) (((const struct xnn_qs8_qc4w_packing_params*) params)->input_zero_point + 0);

  do {
    // NC main loop multiple of 16
    const uint8_t* w0 = (const uint8_t*) weights;
    const uint16_t* s0 = (const uint16_t*) scale;
    size_t n = nc;
    for (;n >= 16; n -= 16) {
        float* packed_k_scaled_sum = (float*) out;
        ((float*) out)[0] = 0;
        ((float*) out)[1] = 0;
        ((float*) out)[2] = 0;
        ((float*) out)[3] = 0;
        ((float*) out)[4] = 0;
        ((float*) out)[5] = 0;
        ((float*) out)[6] = 0;
        ((float*) out)[7] = 0;
        ((float*) out)[8] = 0;
        ((float*) out)[9] = 0;
        ((float*) out)[10] = 0;
        ((float*) out)[11] = 0;
        ((float*) out)[12] = 0;
        ((float*) out)[13] = 0;
        ((float*) out)[14] = 0;
        ((float*) out)[15] = 0;
        out += 16 * sizeof(float);

        // KC/2 bytes is KC Nibbles
        const uint8_t* w1 = w0 + (kc >> 1);
        const uint8_t* w2 = w1 + (kc >> 1);
        const uint8_t* w3 = w2 + (kc >> 1);
        const uint8_t* w4 = w3 + (kc >> 1);
        const uint8_t* w5 = w4 + (kc >> 1);
        const uint8_t* w6 = w5 + (kc >> 1);
        const uint8_t* w7 = w6 + (kc >> 1);
        const uint8_t* w8 = w7 + (kc >> 1);
        const uint8_t* w9 = w8 + (kc >> 1);
        const uint8_t* w10 = w9 + (kc >> 1);
        const uint8_t* w11 = w10 + (kc >> 1);
        const uint8_t* w12 = w11 + (kc >> 1);
        const uint8_t* w13 = w12 + (kc >> 1);
        const uint8_t* w14 = w13 + (kc >> 1);
        const uint8_t* w15 = w14 + (kc >> 1);

        // scales
        const uint16_t* s1 = s0 + num_blocks;
        const uint16_t* s2 = s1 + num_blocks;
        const uint16_t* s3 = s2 + num_blocks;
        const uint16_t* s4 = s3 + num_blocks;
        const uint16_t* s5 = s4 + num_blocks;
        const uint16_t* s6 = s5 + num_blocks;
        const uint16_t* s7 = s6 + num_blocks;
        const uint16_t* s8 = s7 + num_blocks;
        const uint16_t* s9 = s8 + num_blocks;
        const uint16_t* s10 = s9 + num_blocks;
        const uint16_t* s11 = s10 + num_blocks;
        const uint16_t* s12 = s11 + num_blocks;
        const uint16_t* s13 = s12 + num_blocks;
        const uint16_t* s14 = s13 + num_blocks;
        const uint16_t* s15 = s14 + num_blocks;


        size_t kb = kc;
        // Process k by blocks (bl)
        for (; kb >= bl; kb-=bl) {
            // Initialize KSum as subtracting bl zero points (8)
            int32_t ksum0 = 0;
            int32_t ksum1 = 0;
            int32_t ksum2 = 0;
            int32_t ksum3 = 0;
            int32_t ksum4 = 0;
            int32_t ksum5 = 0;
            int32_t ksum6 = 0;
            int32_t ksum7 = 0;
            int32_t ksum8 = 0;
            int32_t ksum9 = 0;
            int32_t ksum10 = 0;
            int32_t ksum11 = 0;
            int32_t ksum12 = 0;
            int32_t ksum13 = 0;
            int32_t ksum14 = 0;
            int32_t ksum15 = 0;
            size_t k = bl;
            for(; k >= 8; k-=8) {
                const uint8_t v0x0 = w0[0] & 0xF;
                const uint8_t v0x1 = w0[0] >> 4;
                const uint8_t v0x2 = w0[1] & 0xF;
                const uint8_t v0x3 = w0[1] >> 4;
                const uint8_t v0x4 = w0[2] & 0xF;
                const uint8_t v0x5 = w0[2] >> 4;
                const uint8_t v0x6 = w0[3] & 0xF;
                const uint8_t v0x7 = w0[3] >> 4;
                w0 += 4;

                ksum0 += (uint32_t) (v0x0);
                ksum0 += (uint32_t) (v0x1);
                ksum0 += (uint32_t) (v0x2);
                ksum0 += (uint32_t) (v0x3);
                ksum0 += (uint32_t) (v0x4);
                ksum0 += (uint32_t) (v0x5);
                ksum0 += (uint32_t) (v0x6);
                ksum0 += (uint32_t) (v0x7);
                // Subtract 8 zero points (8)
                ksum0 -= 64;

                out[0] = (v0x0 | (v0x4 << 4)) ^ 0x88;
                out[1] = (v0x1 | (v0x5 << 4)) ^ 0x88;
                out[2] = (v0x2 | (v0x6 << 4)) ^ 0x88;
                out[3] = (v0x3 | (v0x7 << 4)) ^ 0x88;
                const uint8_t v1x0 = w1[0] & 0xF;
                const uint8_t v1x1 = w1[0] >> 4;
                const uint8_t v1x2 = w1[1] & 0xF;
                const uint8_t v1x3 = w1[1] >> 4;
                const uint8_t v1x4 = w1[2] & 0xF;
                const uint8_t v1x5 = w1[2] >> 4;
                const uint8_t v1x6 = w1[3] & 0xF;
                const uint8_t v1x7 = w1[3] >> 4;
                w1 += 4;

                ksum1 += (uint32_t) (v1x0);
                ksum1 += (uint32_t) (v1x1);
                ksum1 += (uint32_t) (v1x2);
                ksum1 += (uint32_t) (v1x3);
                ksum1 += (uint32_t) (v1x4);
                ksum1 += (uint32_t) (v1x5);
                ksum1 += (uint32_t) (v1x6);
                ksum1 += (uint32_t) (v1x7);
                // Subtract 8 zero points (8)
                ksum1 -= 64;

                out[4] = (v1x0 | (v1x4 << 4)) ^ 0x88;
                out[5] = (v1x1 | (v1x5 << 4)) ^ 0x88;
                out[6] = (v1x2 | (v1x6 << 4)) ^ 0x88;
                out[7] = (v1x3 | (v1x7 << 4)) ^ 0x88;
                const uint8_t v2x0 = w2[0] & 0xF;
                const uint8_t v2x1 = w2[0] >> 4;
                const uint8_t v2x2 = w2[1] & 0xF;
                const uint8_t v2x3 = w2[1] >> 4;
                const uint8_t v2x4 = w2[2] & 0xF;
                const uint8_t v2x5 = w2[2] >> 4;
                const uint8_t v2x6 = w2[3] & 0xF;
                const uint8_t v2x7 = w2[3] >> 4;
                w2 += 4;

                ksum2 += (uint32_t) (v2x0);
                ksum2 += (uint32_t) (v2x1);
                ksum2 += (uint32_t) (v2x2);
                ksum2 += (uint32_t) (v2x3);
                ksum2 += (uint32_t) (v2x4);
                ksum2 += (uint32_t) (v2x5);
                ksum2 += (uint32_t) (v2x6);
                ksum2 += (uint32_t) (v2x7);
                // Subtract 8 zero points (8)
                ksum2 -= 64;

                out[8] = (v2x0 | (v2x4 << 4)) ^ 0x88;
                out[9] = (v2x1 | (v2x5 << 4)) ^ 0x88;
                out[10] = (v2x2 | (v2x6 << 4)) ^ 0x88;
                out[11] = (v2x3 | (v2x7 << 4)) ^ 0x88;
                const uint8_t v3x0 = w3[0] & 0xF;
                const uint8_t v3x1 = w3[0] >> 4;
                const uint8_t v3x2 = w3[1] & 0xF;
                const uint8_t v3x3 = w3[1] >> 4;
                const uint8_t v3x4 = w3[2] & 0xF;
                const uint8_t v3x5 = w3[2] >> 4;
                const uint8_t v3x6 = w3[3] & 0xF;
                const uint8_t v3x7 = w3[3] >> 4;
                w3 += 4;

                ksum3 += (uint32_t) (v3x0);
                ksum3 += (uint32_t) (v3x1);
                ksum3 += (uint32_t) (v3x2);
                ksum3 += (uint32_t) (v3x3);
                ksum3 += (uint32_t) (v3x4);
                ksum3 += (uint32_t) (v3x5);
                ksum3 += (uint32_t) (v3x6);
                ksum3 += (uint32_t) (v3x7);
                // Subtract 8 zero points (8)
                ksum3 -= 64;

                out[12] = (v3x0 | (v3x4 << 4)) ^ 0x88;
                out[13] = (v3x1 | (v3x5 << 4)) ^ 0x88;
                out[14] = (v3x2 | (v3x6 << 4)) ^ 0x88;
                out[15] = (v3x3 | (v3x7 << 4)) ^ 0x88;
                const uint8_t v4x0 = w4[0] & 0xF;
                const uint8_t v4x1 = w4[0] >> 4;
                const uint8_t v4x2 = w4[1] & 0xF;
                const uint8_t v4x3 = w4[1] >> 4;
                const uint8_t v4x4 = w4[2] & 0xF;
                const uint8_t v4x5 = w4[2] >> 4;
                const uint8_t v4x6 = w4[3] & 0xF;
                const uint8_t v4x7 = w4[3] >> 4;
                w4 += 4;

                ksum4 += (uint32_t) (v4x0);
                ksum4 += (uint32_t) (v4x1);
                ksum4 += (uint32_t) (v4x2);
                ksum4 += (uint32_t) (v4x3);
                ksum4 += (uint32_t) (v4x4);
                ksum4 += (uint32_t) (v4x5);
                ksum4 += (uint32_t) (v4x6);
                ksum4 += (uint32_t) (v4x7);
                // Subtract 8 zero points (8)
                ksum4 -= 64;

                out[16] = (v4x0 | (v4x4 << 4)) ^ 0x88;
                out[17] = (v4x1 | (v4x5 << 4)) ^ 0x88;
                out[18] = (v4x2 | (v4x6 << 4)) ^ 0x88;
                out[19] = (v4x3 | (v4x7 << 4)) ^ 0x88;
                const uint8_t v5x0 = w5[0] & 0xF;
                const uint8_t v5x1 = w5[0] >> 4;
                const uint8_t v5x2 = w5[1] & 0xF;
                const uint8_t v5x3 = w5[1] >> 4;
                const uint8_t v5x4 = w5[2] & 0xF;
                const uint8_t v5x5 = w5[2] >> 4;
                const uint8_t v5x6 = w5[3] & 0xF;
                const uint8_t v5x7 = w5[3] >> 4;
                w5 += 4;

                ksum5 += (uint32_t) (v5x0);
                ksum5 += (uint32_t) (v5x1);
                ksum5 += (uint32_t) (v5x2);
                ksum5 += (uint32_t) (v5x3);
                ksum5 += (uint32_t) (v5x4);
                ksum5 += (uint32_t) (v5x5);
                ksum5 += (uint32_t) (v5x6);
                ksum5 += (uint32_t) (v5x7);
                // Subtract 8 zero points (8)
                ksum5 -= 64;

                out[20] = (v5x0 | (v5x4 << 4)) ^ 0x88;
                out[21] = (v5x1 | (v5x5 << 4)) ^ 0x88;
                out[22] = (v5x2 | (v5x6 << 4)) ^ 0x88;
                out[23] = (v5x3 | (v5x7 << 4)) ^ 0x88;
                const uint8_t v6x0 = w6[0] & 0xF;
                const uint8_t v6x1 = w6[0] >> 4;
                const uint8_t v6x2 = w6[1] & 0xF;
                const uint8_t v6x3 = w6[1] >> 4;
                const uint8_t v6x4 = w6[2] & 0xF;
                const uint8_t v6x5 = w6[2] >> 4;
                const uint8_t v6x6 = w6[3] & 0xF;
                const uint8_t v6x7 = w6[3] >> 4;
                w6 += 4;

                ksum6 += (uint32_t) (v6x0);
                ksum6 += (uint32_t) (v6x1);
                ksum6 += (uint32_t) (v6x2);
                ksum6 += (uint32_t) (v6x3);
                ksum6 += (uint32_t) (v6x4);
                ksum6 += (uint32_t) (v6x5);
                ksum6 += (uint32_t) (v6x6);
                ksum6 += (uint32_t) (v6x7);
                // Subtract 8 zero points (8)
                ksum6 -= 64;

                out[24] = (v6x0 | (v6x4 << 4)) ^ 0x88;
                out[25] = (v6x1 | (v6x5 << 4)) ^ 0x88;
                out[26] = (v6x2 | (v6x6 << 4)) ^ 0x88;
                out[27] = (v6x3 | (v6x7 << 4)) ^ 0x88;
                const uint8_t v7x0 = w7[0] & 0xF;
                const uint8_t v7x1 = w7[0] >> 4;
                const uint8_t v7x2 = w7[1] & 0xF;
                const uint8_t v7x3 = w7[1] >> 4;
                const uint8_t v7x4 = w7[2] & 0xF;
                const uint8_t v7x5 = w7[2] >> 4;
                const uint8_t v7x6 = w7[3] & 0xF;
                const uint8_t v7x7 = w7[3] >> 4;
                w7 += 4;

                ksum7 += (uint32_t) (v7x0);
                ksum7 += (uint32_t) (v7x1);
                ksum7 += (uint32_t) (v7x2);
                ksum7 += (uint32_t) (v7x3);
                ksum7 += (uint32_t) (v7x4);
                ksum7 += (uint32_t) (v7x5);
                ksum7 += (uint32_t) (v7x6);
                ksum7 += (uint32_t) (v7x7);
                // Subtract 8 zero points (8)
                ksum7 -= 64;

                out[28] = (v7x0 | (v7x4 << 4)) ^ 0x88;
                out[29] = (v7x1 | (v7x5 << 4)) ^ 0x88;
                out[30] = (v7x2 | (v7x6 << 4)) ^ 0x88;
                out[31] = (v7x3 | (v7x7 << 4)) ^ 0x88;
                const uint8_t v8x0 = w8[0] & 0xF;
                const uint8_t v8x1 = w8[0] >> 4;
                const uint8_t v8x2 = w8[1] & 0xF;
                const uint8_t v8x3 = w8[1] >> 4;
                const uint8_t v8x4 = w8[2] & 0xF;
                const uint8_t v8x5 = w8[2] >> 4;
                const uint8_t v8x6 = w8[3] & 0xF;
                const uint8_t v8x7 = w8[3] >> 4;
                w8 += 4;

                ksum8 += (uint32_t) (v8x0);
                ksum8 += (uint32_t) (v8x1);
                ksum8 += (uint32_t) (v8x2);
                ksum8 += (uint32_t) (v8x3);
                ksum8 += (uint32_t) (v8x4);
                ksum8 += (uint32_t) (v8x5);
                ksum8 += (uint32_t) (v8x6);
                ksum8 += (uint32_t) (v8x7);
                // Subtract 8 zero points (8)
                ksum8 -= 64;

                out[32] = (v8x0 | (v8x4 << 4)) ^ 0x88;
                out[33] = (v8x1 | (v8x5 << 4)) ^ 0x88;
                out[34] = (v8x2 | (v8x6 << 4)) ^ 0x88;
                out[35] = (v8x3 | (v8x7 << 4)) ^ 0x88;
                const uint8_t v9x0 = w9[0] & 0xF;
                const uint8_t v9x1 = w9[0] >> 4;
                const uint8_t v9x2 = w9[1] & 0xF;
                const uint8_t v9x3 = w9[1] >> 4;
                const uint8_t v9x4 = w9[2] & 0xF;
                const uint8_t v9x5 = w9[2] >> 4;
                const uint8_t v9x6 = w9[3] & 0xF;
                const uint8_t v9x7 = w9[3] >> 4;
                w9 += 4;

                ksum9 += (uint32_t) (v9x0);
                ksum9 += (uint32_t) (v9x1);
                ksum9 += (uint32_t) (v9x2);
                ksum9 += (uint32_t) (v9x3);
                ksum9 += (uint32_t) (v9x4);
                ksum9 += (uint32_t) (v9x5);
                ksum9 += (uint32_t) (v9x6);
                ksum9 += (uint32_t) (v9x7);
                // Subtract 8 zero points (8)
                ksum9 -= 64;

                out[36] = (v9x0 | (v9x4 << 4)) ^ 0x88;
                out[37] = (v9x1 | (v9x5 << 4)) ^ 0x88;
                out[38] = (v9x2 | (v9x6 << 4)) ^ 0x88;
                out[39] = (v9x3 | (v9x7 << 4)) ^ 0x88;
                const uint8_t v10x0 = w10[0] & 0xF;
                const uint8_t v10x1 = w10[0] >> 4;
                const uint8_t v10x2 = w10[1] & 0xF;
                const uint8_t v10x3 = w10[1] >> 4;
                const uint8_t v10x4 = w10[2] & 0xF;
                const uint8_t v10x5 = w10[2] >> 4;
                const uint8_t v10x6 = w10[3] & 0xF;
                const uint8_t v10x7 = w10[3] >> 4;
                w10 += 4;

                ksum10 += (uint32_t) (v10x0);
                ksum10 += (uint32_t) (v10x1);
                ksum10 += (uint32_t) (v10x2);
                ksum10 += (uint32_t) (v10x3);
                ksum10 += (uint32_t) (v10x4);
                ksum10 += (uint32_t) (v10x5);
                ksum10 += (uint32_t) (v10x6);
                ksum10 += (uint32_t) (v10x7);
                // Subtract 8 zero points (8)
                ksum10 -= 64;

                out[40] = (v10x0 | (v10x4 << 4)) ^ 0x88;
                out[41] = (v10x1 | (v10x5 << 4)) ^ 0x88;
                out[42] = (v10x2 | (v10x6 << 4)) ^ 0x88;
                out[43] = (v10x3 | (v10x7 << 4)) ^ 0x88;
                const uint8_t v11x0 = w11[0] & 0xF;
                const uint8_t v11x1 = w11[0] >> 4;
                const uint8_t v11x2 = w11[1] & 0xF;
                const uint8_t v11x3 = w11[1] >> 4;
                const uint8_t v11x4 = w11[2] & 0xF;
                const uint8_t v11x5 = w11[2] >> 4;
                const uint8_t v11x6 = w11[3] & 0xF;
                const uint8_t v11x7 = w11[3] >> 4;
                w11 += 4;

                ksum11 += (uint32_t) (v11x0);
                ksum11 += (uint32_t) (v11x1);
                ksum11 += (uint32_t) (v11x2);
                ksum11 += (uint32_t) (v11x3);
                ksum11 += (uint32_t) (v11x4);
                ksum11 += (uint32_t) (v11x5);
                ksum11 += (uint32_t) (v11x6);
                ksum11 += (uint32_t) (v11x7);
                // Subtract 8 zero points (8)
                ksum11 -= 64;

                out[44] = (v11x0 | (v11x4 << 4)) ^ 0x88;
                out[45] = (v11x1 | (v11x5 << 4)) ^ 0x88;
                out[46] = (v11x2 | (v11x6 << 4)) ^ 0x88;
                out[47] = (v11x3 | (v11x7 << 4)) ^ 0x88;
                const uint8_t v12x0 = w12[0] & 0xF;
                const uint8_t v12x1 = w12[0] >> 4;
                const uint8_t v12x2 = w12[1] & 0xF;
                const uint8_t v12x3 = w12[1] >> 4;
                const uint8_t v12x4 = w12[2] & 0xF;
                const uint8_t v12x5 = w12[2] >> 4;
                const uint8_t v12x6 = w12[3] & 0xF;
                const uint8_t v12x7 = w12[3] >> 4;
                w12 += 4;

                ksum12 += (uint32_t) (v12x0);
                ksum12 += (uint32_t) (v12x1);
                ksum12 += (uint32_t) (v12x2);
                ksum12 += (uint32_t) (v12x3);
                ksum12 += (uint32_t) (v12x4);
                ksum12 += (uint32_t) (v12x5);
                ksum12 += (uint32_t) (v12x6);
                ksum12 += (uint32_t) (v12x7);
                // Subtract 8 zero points (8)
                ksum12 -= 64;

                out[48] = (v12x0 | (v12x4 << 4)) ^ 0x88;
                out[49] = (v12x1 | (v12x5 << 4)) ^ 0x88;
                out[50] = (v12x2 | (v12x6 << 4)) ^ 0x88;
                out[51] = (v12x3 | (v12x7 << 4)) ^ 0x88;
                const uint8_t v13x0 = w13[0] & 0xF;
                const uint8_t v13x1 = w13[0] >> 4;
                const uint8_t v13x2 = w13[1] & 0xF;
                const uint8_t v13x3 = w13[1] >> 4;
                const uint8_t v13x4 = w13[2] & 0xF;
                const uint8_t v13x5 = w13[2] >> 4;
                const uint8_t v13x6 = w13[3] & 0xF;
                const uint8_t v13x7 = w13[3] >> 4;
                w13 += 4;

                ksum13 += (uint32_t) (v13x0);
                ksum13 += (uint32_t) (v13x1);
                ksum13 += (uint32_t) (v13x2);
                ksum13 += (uint32_t) (v13x3);
                ksum13 += (uint32_t) (v13x4);
                ksum13 += (uint32_t) (v13x5);
                ksum13 += (uint32_t) (v13x6);
                ksum13 += (uint32_t) (v13x7);
                // Subtract 8 zero points (8)
                ksum13 -= 64;

                out[52] = (v13x0 | (v13x4 << 4)) ^ 0x88;
                out[53] = (v13x1 | (v13x5 << 4)) ^ 0x88;
                out[54] = (v13x2 | (v13x6 << 4)) ^ 0x88;
                out[55] = (v13x3 | (v13x7 << 4)) ^ 0x88;
                const uint8_t v14x0 = w14[0] & 0xF;
                const uint8_t v14x1 = w14[0] >> 4;
                const uint8_t v14x2 = w14[1] & 0xF;
                const uint8_t v14x3 = w14[1] >> 4;
                const uint8_t v14x4 = w14[2] & 0xF;
                const uint8_t v14x5 = w14[2] >> 4;
                const uint8_t v14x6 = w14[3] & 0xF;
                const uint8_t v14x7 = w14[3] >> 4;
                w14 += 4;

                ksum14 += (uint32_t) (v14x0);
                ksum14 += (uint32_t) (v14x1);
                ksum14 += (uint32_t) (v14x2);
                ksum14 += (uint32_t) (v14x3);
                ksum14 += (uint32_t) (v14x4);
                ksum14 += (uint32_t) (v14x5);
                ksum14 += (uint32_t) (v14x6);
                ksum14 += (uint32_t) (v14x7);
                // Subtract 8 zero points (8)
                ksum14 -= 64;

                out[56] = (v14x0 | (v14x4 << 4)) ^ 0x88;
                out[57] = (v14x1 | (v14x5 << 4)) ^ 0x88;
                out[58] = (v14x2 | (v14x6 << 4)) ^ 0x88;
                out[59] = (v14x3 | (v14x7 << 4)) ^ 0x88;
                const uint8_t v15x0 = w15[0] & 0xF;
                const uint8_t v15x1 = w15[0] >> 4;
                const uint8_t v15x2 = w15[1] & 0xF;
                const uint8_t v15x3 = w15[1] >> 4;
                const uint8_t v15x4 = w15[2] & 0xF;
                const uint8_t v15x5 = w15[2] >> 4;
                const uint8_t v15x6 = w15[3] & 0xF;
                const uint8_t v15x7 = w15[3] >> 4;
                w15 += 4;

                ksum15 += (uint32_t) (v15x0);
                ksum15 += (uint32_t) (v15x1);
                ksum15 += (uint32_t) (v15x2);
                ksum15 += (uint32_t) (v15x3);
                ksum15 += (uint32_t) (v15x4);
                ksum15 += (uint32_t) (v15x5);
                ksum15 += (uint32_t) (v15x6);
                ksum15 += (uint32_t) (v15x7);
                // Subtract 8 zero points (8)
                ksum15 -= 64;

                out[60] = (v15x0 | (v15x4 << 4)) ^ 0x88;
                out[61] = (v15x1 | (v15x5 << 4)) ^ 0x88;
                out[62] = (v15x2 | (v15x6 << 4)) ^ 0x88;
                out[63] = (v15x3 | (v15x7 << 4)) ^ 0x88;

                out += 64;
            }
            float scale0 = math_cvt_fp32_bf16(s0[0]);
            float scale1 = math_cvt_fp32_bf16(s1[0]);
            float scale2 = math_cvt_fp32_bf16(s2[0]);
            float scale3 = math_cvt_fp32_bf16(s3[0]);
            float scale4 = math_cvt_fp32_bf16(s4[0]);
            float scale5 = math_cvt_fp32_bf16(s5[0]);
            float scale6 = math_cvt_fp32_bf16(s6[0]);
            float scale7 = math_cvt_fp32_bf16(s7[0]);
            float scale8 = math_cvt_fp32_bf16(s8[0]);
            float scale9 = math_cvt_fp32_bf16(s9[0]);
            float scale10 = math_cvt_fp32_bf16(s10[0]);
            float scale11 = math_cvt_fp32_bf16(s11[0]);
            float scale12 = math_cvt_fp32_bf16(s12[0]);
            float scale13 = math_cvt_fp32_bf16(s13[0]);
            float scale14 = math_cvt_fp32_bf16(s14[0]);
            float scale15 = math_cvt_fp32_bf16(s15[0]);
            s0 += 1;
            s1 += 1;
            s2 += 1;
            s3 += 1;
            s4 += 1;
            s5 += 1;
            s6 += 1;
            s7 += 1;
            s8 += 1;
            s9 += 1;
            s10 += 1;
            s11 += 1;
            s12 += 1;
            s13 += 1;
            s14 += 1;
            s15 += 1;


            packed_k_scaled_sum[0] -= (float)ksum0 * izp * scale0;
            packed_k_scaled_sum[1] -= (float)ksum1 * izp * scale1;
            packed_k_scaled_sum[2] -= (float)ksum2 * izp * scale2;
            packed_k_scaled_sum[3] -= (float)ksum3 * izp * scale3;
            packed_k_scaled_sum[4] -= (float)ksum4 * izp * scale4;
            packed_k_scaled_sum[5] -= (float)ksum5 * izp * scale5;
            packed_k_scaled_sum[6] -= (float)ksum6 * izp * scale6;
            packed_k_scaled_sum[7] -= (float)ksum7 * izp * scale7;
            packed_k_scaled_sum[8] -= (float)ksum8 * izp * scale8;
            packed_k_scaled_sum[9] -= (float)ksum9 * izp * scale9;
            packed_k_scaled_sum[10] -= (float)ksum10 * izp * scale10;
            packed_k_scaled_sum[11] -= (float)ksum11 * izp * scale11;
            packed_k_scaled_sum[12] -= (float)ksum12 * izp * scale12;
            packed_k_scaled_sum[13] -= (float)ksum13 * izp * scale13;
            packed_k_scaled_sum[14] -= (float)ksum14 * izp * scale14;
            packed_k_scaled_sum[15] -= (float)ksum15 * izp * scale15;

            ((uint16_t*) out)[0] = math_cvt_bf16_fp32(scale0 / 16.0f);
            ((uint16_t*) out)[1] = math_cvt_bf16_fp32(scale1 / 16.0f);
            ((uint16_t*) out)[2] = math_cvt_bf16_fp32(scale2 / 16.0f);
            ((uint16_t*) out)[3] = math_cvt_bf16_fp32(scale3 / 16.0f);
            ((uint16_t*) out)[4] = math_cvt_bf16_fp32(scale4 / 16.0f);
            ((uint16_t*) out)[5] = math_cvt_bf16_fp32(scale5 / 16.0f);
            ((uint16_t*) out)[6] = math_cvt_bf16_fp32(scale6 / 16.0f);
            ((uint16_t*) out)[7] = math_cvt_bf16_fp32(scale7 / 16.0f);
            ((uint16_t*) out)[8] = math_cvt_bf16_fp32(scale8 / 16.0f);
            ((uint16_t*) out)[9] = math_cvt_bf16_fp32(scale9 / 16.0f);
            ((uint16_t*) out)[10] = math_cvt_bf16_fp32(scale10 / 16.0f);
            ((uint16_t*) out)[11] = math_cvt_bf16_fp32(scale11 / 16.0f);
            ((uint16_t*) out)[12] = math_cvt_bf16_fp32(scale12 / 16.0f);
            ((uint16_t*) out)[13] = math_cvt_bf16_fp32(scale13 / 16.0f);
            ((uint16_t*) out)[14] = math_cvt_bf16_fp32(scale14 / 16.0f);
            ((uint16_t*) out)[15] = math_cvt_bf16_fp32(scale15 / 16.0f);

            out += 16 * sizeof(uint16_t);
        }


        if XNN_LIKELY(b != NULL){
            ((uint32_t*) out)[0] = b[0];
            ((uint32_t*) out)[1] = b[1];
            ((uint32_t*) out)[2] = b[2];
            ((uint32_t*) out)[3] = b[3];
            ((uint32_t*) out)[4] = b[4];
            ((uint32_t*) out)[5] = b[5];
            ((uint32_t*) out)[6] = b[6];
            ((uint32_t*) out)[7] = b[7];
            ((uint32_t*) out)[8] = b[8];
            ((uint32_t*) out)[9] = b[9];
            ((uint32_t*) out)[10] = b[10];
            ((uint32_t*) out)[11] = b[11];
            ((uint32_t*) out)[12] = b[12];
            ((uint32_t*) out)[13] = b[13];
            ((uint32_t*) out)[14] = b[14];
            ((uint32_t*) out)[15] = b[15];
            b += 16;
        } else {
            ((uint32_t*) out)[0] = 0;
            ((uint32_t*) out)[1] = 0;
            ((uint32_t*) out)[2] = 0;
            ((uint32_t*) out)[3] = 0;
            ((uint32_t*) out)[4] = 0;
            ((uint32_t*) out)[5] = 0;
            ((uint32_t*) out)[6] = 0;
            ((uint32_t*) out)[7] = 0;
            ((uint32_t*) out)[8] = 0;
            ((uint32_t*) out)[9] = 0;
            ((uint32_t*) out)[10] = 0;
            ((uint32_t*) out)[11] = 0;
            ((uint32_t*) out)[12] = 0;
            ((uint32_t*) out)[13] = 0;
            ((uint32_t*) out)[14] = 0;
            ((uint32_t*) out)[15] = 0;
        }
        out += 16 * sizeof(uint32_t);
        w0 = w15;
        s0 = s15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
        float* packed_k_scaled_sum = (float*) out;
        ((float*) out)[0] = 0;
        ((float*) out)[1] = 0;
        ((float*) out)[2] = 0;
        ((float*) out)[3] = 0;
        ((float*) out)[4] = 0;
        ((float*) out)[5] = 0;
        ((float*) out)[6] = 0;
        ((float*) out)[7] = 0;
        ((float*) out)[8] = 0;
        ((float*) out)[9] = 0;
        ((float*) out)[10] = 0;
        ((float*) out)[11] = 0;
        ((float*) out)[12] = 0;
        ((float*) out)[13] = 0;
        ((float*) out)[14] = 0;
        ((float*) out)[15] = 0;
        out += 16 * sizeof(float);
        // NR remainder has less than 16
        const uint8_t* w1 = w0 + (kc >> 1);
        const uint16_t* s1 = s0 + num_blocks;
        if XNN_UNPREDICTABLE(n < 2) {
            w1 = w0;
            s1 = s0;
        }
        const uint8_t* w2 = w1 + (kc >> 1);
        const uint16_t* s2 = s1 + num_blocks;
        if XNN_UNPREDICTABLE(n <= 2) {
            w2 = w1;
            s2 = s1;
        }
        const uint8_t* w3 = w2 + (kc >> 1);
        const uint16_t* s3 = s2 + num_blocks;
        if XNN_UNPREDICTABLE(n < 4) {
            w3 = w2;
            s3 = s2;
        }
        const uint8_t* w4 = w3 + (kc >> 1);
        const uint16_t* s4 = s3 + num_blocks;
        if XNN_UNPREDICTABLE(n <= 4) {
            w4 = w3;
            s4 = s3;
        }
        const uint8_t* w5 = w4 + (kc >> 1);
        const uint16_t* s5 = s4 + num_blocks;
        if XNN_UNPREDICTABLE(n < 6) {
            w5 = w4;
            s5 = s4;
        }
        const uint8_t* w6 = w5 + (kc >> 1);
        const uint16_t* s6 = s5 + num_blocks;
        if XNN_UNPREDICTABLE(n <= 6) {
            w6 = w5;
            s6 = s5;
        }
        const uint8_t* w7 = w6 + (kc >> 1);
        const uint16_t* s7 = s6 + num_blocks;
        if XNN_UNPREDICTABLE(n < 8) {
            w7 = w6;
            s7 = s6;
        }
        const uint8_t* w8 = w7 + (kc >> 1);
        const uint16_t* s8 = s7 + num_blocks;
        if XNN_UNPREDICTABLE(n <= 8) {
            w8 = w7;
            s8 = s7;
        }
        const uint8_t* w9 = w8 + (kc >> 1);
        const uint16_t* s9 = s8 + num_blocks;
        if XNN_UNPREDICTABLE(n < 10) {
            w9 = w8;
            s9 = s8;
        }
        const uint8_t* w10 = w9 + (kc >> 1);
        const uint16_t* s10 = s9 + num_blocks;
        if XNN_UNPREDICTABLE(n <= 10) {
            w10 = w9;
            s10 = s9;
        }
        const uint8_t* w11 = w10 + (kc >> 1);
        const uint16_t* s11 = s10 + num_blocks;
        if XNN_UNPREDICTABLE(n < 12) {
            w11 = w10;
            s11 = s10;
        }
        const uint8_t* w12 = w11 + (kc >> 1);
        const uint16_t* s12 = s11 + num_blocks;
        if XNN_UNPREDICTABLE(n <= 12) {
            w12 = w11;
            s12 = s11;
        }
        const uint8_t* w13 = w12 + (kc >> 1);
        const uint16_t* s13 = s12 + num_blocks;
        if XNN_UNPREDICTABLE(n < 14) {
            w13 = w12;
            s13 = s12;
        }
        const uint8_t* w14 = w13 + (kc >> 1);
        const uint16_t* s14 = s13 + num_blocks;
        if XNN_UNPREDICTABLE(n <= 14) {
            w14 = w13;
            s14 = s13;
        }

        size_t kb = kc;
        // Process k by blocks (bl)
        for (; kb >= bl; kb-=bl) {
            // Initialize KSum as subtracting bl zero points (8)
            int32_t ksum0 = 0;
            int32_t ksum1 = 0;
            int32_t ksum2 = 0;
            int32_t ksum3 = 0;
            int32_t ksum4 = 0;
            int32_t ksum5 = 0;
            int32_t ksum6 = 0;
            int32_t ksum7 = 0;
            int32_t ksum8 = 0;
            int32_t ksum9 = 0;
            int32_t ksum10 = 0;
            int32_t ksum11 = 0;
            int32_t ksum12 = 0;
            int32_t ksum13 = 0;
            int32_t ksum14 = 0;
            size_t k = bl;
            for(; k >= 8; k-=8) {
                const uint8_t v0x0 = w0[0] & 0xF;
                const uint8_t v0x1 = w0[0] >> 4;
                const uint8_t v0x2 = w0[1] & 0xF;
                const uint8_t v0x3 = w0[1] >> 4;
                const uint8_t v0x4 = w0[2] & 0xF;
                const uint8_t v0x5 = w0[2] >> 4;
                const uint8_t v0x6 = w0[3] & 0xF;
                const uint8_t v0x7 = w0[3] >> 4;
                w0 += 4;

                ksum0 += (uint32_t) (v0x0);
                ksum0 += (uint32_t) (v0x1);
                ksum0 += (uint32_t) (v0x2);
                ksum0 += (uint32_t) (v0x3);
                ksum0 += (uint32_t) (v0x4);
                ksum0 += (uint32_t) (v0x5);
                ksum0 += (uint32_t) (v0x6);
                ksum0 += (uint32_t) (v0x7);
                // Subtract 8 zero points (8)
                ksum0 -= 64;

                out[0] = (v0x0 | (v0x4 << 4)) ^ 0x88;
                out[1] = (v0x1 | (v0x5 << 4)) ^ 0x88;
                out[2] = (v0x2 | (v0x6 << 4)) ^ 0x88;
                out[3] = (v0x3 | (v0x7 << 4)) ^ 0x88;
                const uint8_t v1x0 = w1[0] & 0xF;
                const uint8_t v1x1 = w1[0] >> 4;
                const uint8_t v1x2 = w1[1] & 0xF;
                const uint8_t v1x3 = w1[1] >> 4;
                const uint8_t v1x4 = w1[2] & 0xF;
                const uint8_t v1x5 = w1[2] >> 4;
                const uint8_t v1x6 = w1[3] & 0xF;
                const uint8_t v1x7 = w1[3] >> 4;
                w1 += 4;

                ksum1 += (uint32_t) (v1x0);
                ksum1 += (uint32_t) (v1x1);
                ksum1 += (uint32_t) (v1x2);
                ksum1 += (uint32_t) (v1x3);
                ksum1 += (uint32_t) (v1x4);
                ksum1 += (uint32_t) (v1x5);
                ksum1 += (uint32_t) (v1x6);
                ksum1 += (uint32_t) (v1x7);
                // Subtract 8 zero points (8)
                ksum1 -= 64;

                out[4] = (v1x0 | (v1x4 << 4)) ^ 0x88;
                out[5] = (v1x1 | (v1x5 << 4)) ^ 0x88;
                out[6] = (v1x2 | (v1x6 << 4)) ^ 0x88;
                out[7] = (v1x3 | (v1x7 << 4)) ^ 0x88;
                const uint8_t v2x0 = w2[0] & 0xF;
                const uint8_t v2x1 = w2[0] >> 4;
                const uint8_t v2x2 = w2[1] & 0xF;
                const uint8_t v2x3 = w2[1] >> 4;
                const uint8_t v2x4 = w2[2] & 0xF;
                const uint8_t v2x5 = w2[2] >> 4;
                const uint8_t v2x6 = w2[3] & 0xF;
                const uint8_t v2x7 = w2[3] >> 4;
                w2 += 4;

                ksum2 += (uint32_t) (v2x0);
                ksum2 += (uint32_t) (v2x1);
                ksum2 += (uint32_t) (v2x2);
                ksum2 += (uint32_t) (v2x3);
                ksum2 += (uint32_t) (v2x4);
                ksum2 += (uint32_t) (v2x5);
                ksum2 += (uint32_t) (v2x6);
                ksum2 += (uint32_t) (v2x7);
                // Subtract 8 zero points (8)
                ksum2 -= 64;

                out[8] = (v2x0 | (v2x4 << 4)) ^ 0x88;
                out[9] = (v2x1 | (v2x5 << 4)) ^ 0x88;
                out[10] = (v2x2 | (v2x6 << 4)) ^ 0x88;
                out[11] = (v2x3 | (v2x7 << 4)) ^ 0x88;
                const uint8_t v3x0 = w3[0] & 0xF;
                const uint8_t v3x1 = w3[0] >> 4;
                const uint8_t v3x2 = w3[1] & 0xF;
                const uint8_t v3x3 = w3[1] >> 4;
                const uint8_t v3x4 = w3[2] & 0xF;
                const uint8_t v3x5 = w3[2] >> 4;
                const uint8_t v3x6 = w3[3] & 0xF;
                const uint8_t v3x7 = w3[3] >> 4;
                w3 += 4;

                ksum3 += (uint32_t) (v3x0);
                ksum3 += (uint32_t) (v3x1);
                ksum3 += (uint32_t) (v3x2);
                ksum3 += (uint32_t) (v3x3);
                ksum3 += (uint32_t) (v3x4);
                ksum3 += (uint32_t) (v3x5);
                ksum3 += (uint32_t) (v3x6);
                ksum3 += (uint32_t) (v3x7);
                // Subtract 8 zero points (8)
                ksum3 -= 64;

                out[12] = (v3x0 | (v3x4 << 4)) ^ 0x88;
                out[13] = (v3x1 | (v3x5 << 4)) ^ 0x88;
                out[14] = (v3x2 | (v3x6 << 4)) ^ 0x88;
                out[15] = (v3x3 | (v3x7 << 4)) ^ 0x88;
                const uint8_t v4x0 = w4[0] & 0xF;
                const uint8_t v4x1 = w4[0] >> 4;
                const uint8_t v4x2 = w4[1] & 0xF;
                const uint8_t v4x3 = w4[1] >> 4;
                const uint8_t v4x4 = w4[2] & 0xF;
                const uint8_t v4x5 = w4[2] >> 4;
                const uint8_t v4x6 = w4[3] & 0xF;
                const uint8_t v4x7 = w4[3] >> 4;
                w4 += 4;

                ksum4 += (uint32_t) (v4x0);
                ksum4 += (uint32_t) (v4x1);
                ksum4 += (uint32_t) (v4x2);
                ksum4 += (uint32_t) (v4x3);
                ksum4 += (uint32_t) (v4x4);
                ksum4 += (uint32_t) (v4x5);
                ksum4 += (uint32_t) (v4x6);
                ksum4 += (uint32_t) (v4x7);
                // Subtract 8 zero points (8)
                ksum4 -= 64;

                out[16] = (v4x0 | (v4x4 << 4)) ^ 0x88;
                out[17] = (v4x1 | (v4x5 << 4)) ^ 0x88;
                out[18] = (v4x2 | (v4x6 << 4)) ^ 0x88;
                out[19] = (v4x3 | (v4x7 << 4)) ^ 0x88;
                const uint8_t v5x0 = w5[0] & 0xF;
                const uint8_t v5x1 = w5[0] >> 4;
                const uint8_t v5x2 = w5[1] & 0xF;
                const uint8_t v5x3 = w5[1] >> 4;
                const uint8_t v5x4 = w5[2] & 0xF;
                const uint8_t v5x5 = w5[2] >> 4;
                const uint8_t v5x6 = w5[3] & 0xF;
                const uint8_t v5x7 = w5[3] >> 4;
                w5 += 4;

                ksum5 += (uint32_t) (v5x0);
                ksum5 += (uint32_t) (v5x1);
                ksum5 += (uint32_t) (v5x2);
                ksum5 += (uint32_t) (v5x3);
                ksum5 += (uint32_t) (v5x4);
                ksum5 += (uint32_t) (v5x5);
                ksum5 += (uint32_t) (v5x6);
                ksum5 += (uint32_t) (v5x7);
                // Subtract 8 zero points (8)
                ksum5 -= 64;

                out[20] = (v5x0 | (v5x4 << 4)) ^ 0x88;
                out[21] = (v5x1 | (v5x5 << 4)) ^ 0x88;
                out[22] = (v5x2 | (v5x6 << 4)) ^ 0x88;
                out[23] = (v5x3 | (v5x7 << 4)) ^ 0x88;
                const uint8_t v6x0 = w6[0] & 0xF;
                const uint8_t v6x1 = w6[0] >> 4;
                const uint8_t v6x2 = w6[1] & 0xF;
                const uint8_t v6x3 = w6[1] >> 4;
                const uint8_t v6x4 = w6[2] & 0xF;
                const uint8_t v6x5 = w6[2] >> 4;
                const uint8_t v6x6 = w6[3] & 0xF;
                const uint8_t v6x7 = w6[3] >> 4;
                w6 += 4;

                ksum6 += (uint32_t) (v6x0);
                ksum6 += (uint32_t) (v6x1);
                ksum6 += (uint32_t) (v6x2);
                ksum6 += (uint32_t) (v6x3);
                ksum6 += (uint32_t) (v6x4);
                ksum6 += (uint32_t) (v6x5);
                ksum6 += (uint32_t) (v6x6);
                ksum6 += (uint32_t) (v6x7);
                // Subtract 8 zero points (8)
                ksum6 -= 64;

                out[24] = (v6x0 | (v6x4 << 4)) ^ 0x88;
                out[25] = (v6x1 | (v6x5 << 4)) ^ 0x88;
                out[26] = (v6x2 | (v6x6 << 4)) ^ 0x88;
                out[27] = (v6x3 | (v6x7 << 4)) ^ 0x88;
                const uint8_t v7x0 = w7[0] & 0xF;
                const uint8_t v7x1 = w7[0] >> 4;
                const uint8_t v7x2 = w7[1] & 0xF;
                const uint8_t v7x3 = w7[1] >> 4;
                const uint8_t v7x4 = w7[2] & 0xF;
                const uint8_t v7x5 = w7[2] >> 4;
                const uint8_t v7x6 = w7[3] & 0xF;
                const uint8_t v7x7 = w7[3] >> 4;
                w7 += 4;

                ksum7 += (uint32_t) (v7x0);
                ksum7 += (uint32_t) (v7x1);
                ksum7 += (uint32_t) (v7x2);
                ksum7 += (uint32_t) (v7x3);
                ksum7 += (uint32_t) (v7x4);
                ksum7 += (uint32_t) (v7x5);
                ksum7 += (uint32_t) (v7x6);
                ksum7 += (uint32_t) (v7x7);
                // Subtract 8 zero points (8)
                ksum7 -= 64;

                out[28] = (v7x0 | (v7x4 << 4)) ^ 0x88;
                out[29] = (v7x1 | (v7x5 << 4)) ^ 0x88;
                out[30] = (v7x2 | (v7x6 << 4)) ^ 0x88;
                out[31] = (v7x3 | (v7x7 << 4)) ^ 0x88;
                const uint8_t v8x0 = w8[0] & 0xF;
                const uint8_t v8x1 = w8[0] >> 4;
                const uint8_t v8x2 = w8[1] & 0xF;
                const uint8_t v8x3 = w8[1] >> 4;
                const uint8_t v8x4 = w8[2] & 0xF;
                const uint8_t v8x5 = w8[2] >> 4;
                const uint8_t v8x6 = w8[3] & 0xF;
                const uint8_t v8x7 = w8[3] >> 4;
                w8 += 4;

                ksum8 += (uint32_t) (v8x0);
                ksum8 += (uint32_t) (v8x1);
                ksum8 += (uint32_t) (v8x2);
                ksum8 += (uint32_t) (v8x3);
                ksum8 += (uint32_t) (v8x4);
                ksum8 += (uint32_t) (v8x5);
                ksum8 += (uint32_t) (v8x6);
                ksum8 += (uint32_t) (v8x7);
                // Subtract 8 zero points (8)
                ksum8 -= 64;

                out[32] = (v8x0 | (v8x4 << 4)) ^ 0x88;
                out[33] = (v8x1 | (v8x5 << 4)) ^ 0x88;
                out[34] = (v8x2 | (v8x6 << 4)) ^ 0x88;
                out[35] = (v8x3 | (v8x7 << 4)) ^ 0x88;
                const uint8_t v9x0 = w9[0] & 0xF;
                const uint8_t v9x1 = w9[0] >> 4;
                const uint8_t v9x2 = w9[1] & 0xF;
                const uint8_t v9x3 = w9[1] >> 4;
                const uint8_t v9x4 = w9[2] & 0xF;
                const uint8_t v9x5 = w9[2] >> 4;
                const uint8_t v9x6 = w9[3] & 0xF;
                const uint8_t v9x7 = w9[3] >> 4;
                w9 += 4;

                ksum9 += (uint32_t) (v9x0);
                ksum9 += (uint32_t) (v9x1);
                ksum9 += (uint32_t) (v9x2);
                ksum9 += (uint32_t) (v9x3);
                ksum9 += (uint32_t) (v9x4);
                ksum9 += (uint32_t) (v9x5);
                ksum9 += (uint32_t) (v9x6);
                ksum9 += (uint32_t) (v9x7);
                // Subtract 8 zero points (8)
                ksum9 -= 64;

                out[36] = (v9x0 | (v9x4 << 4)) ^ 0x88;
                out[37] = (v9x1 | (v9x5 << 4)) ^ 0x88;
                out[38] = (v9x2 | (v9x6 << 4)) ^ 0x88;
                out[39] = (v9x3 | (v9x7 << 4)) ^ 0x88;
                const uint8_t v10x0 = w10[0] & 0xF;
                const uint8_t v10x1 = w10[0] >> 4;
                const uint8_t v10x2 = w10[1] & 0xF;
                const uint8_t v10x3 = w10[1] >> 4;
                const uint8_t v10x4 = w10[2] & 0xF;
                const uint8_t v10x5 = w10[2] >> 4;
                const uint8_t v10x6 = w10[3] & 0xF;
                const uint8_t v10x7 = w10[3] >> 4;
                w10 += 4;

                ksum10 += (uint32_t) (v10x0);
                ksum10 += (uint32_t) (v10x1);
                ksum10 += (uint32_t) (v10x2);
                ksum10 += (uint32_t) (v10x3);
                ksum10 += (uint32_t) (v10x4);
                ksum10 += (uint32_t) (v10x5);
                ksum10 += (uint32_t) (v10x6);
                ksum10 += (uint32_t) (v10x7);
                // Subtract 8 zero points (8)
                ksum10 -= 64;

                out[40] = (v10x0 | (v10x4 << 4)) ^ 0x88;
                out[41] = (v10x1 | (v10x5 << 4)) ^ 0x88;
                out[42] = (v10x2 | (v10x6 << 4)) ^ 0x88;
                out[43] = (v10x3 | (v10x7 << 4)) ^ 0x88;
                const uint8_t v11x0 = w11[0] & 0xF;
                const uint8_t v11x1 = w11[0] >> 4;
                const uint8_t v11x2 = w11[1] & 0xF;
                const uint8_t v11x3 = w11[1] >> 4;
                const uint8_t v11x4 = w11[2] & 0xF;
                const uint8_t v11x5 = w11[2] >> 4;
                const uint8_t v11x6 = w11[3] & 0xF;
                const uint8_t v11x7 = w11[3] >> 4;
                w11 += 4;

                ksum11 += (uint32_t) (v11x0);
                ksum11 += (uint32_t) (v11x1);
                ksum11 += (uint32_t) (v11x2);
                ksum11 += (uint32_t) (v11x3);
                ksum11 += (uint32_t) (v11x4);
                ksum11 += (uint32_t) (v11x5);
                ksum11 += (uint32_t) (v11x6);
                ksum11 += (uint32_t) (v11x7);
                // Subtract 8 zero points (8)
                ksum11 -= 64;

                out[44] = (v11x0 | (v11x4 << 4)) ^ 0x88;
                out[45] = (v11x1 | (v11x5 << 4)) ^ 0x88;
                out[46] = (v11x2 | (v11x6 << 4)) ^ 0x88;
                out[47] = (v11x3 | (v11x7 << 4)) ^ 0x88;
                const uint8_t v12x0 = w12[0] & 0xF;
                const uint8_t v12x1 = w12[0] >> 4;
                const uint8_t v12x2 = w12[1] & 0xF;
                const uint8_t v12x3 = w12[1] >> 4;
                const uint8_t v12x4 = w12[2] & 0xF;
                const uint8_t v12x5 = w12[2] >> 4;
                const uint8_t v12x6 = w12[3] & 0xF;
                const uint8_t v12x7 = w12[3] >> 4;
                w12 += 4;

                ksum12 += (uint32_t) (v12x0);
                ksum12 += (uint32_t) (v12x1);
                ksum12 += (uint32_t) (v12x2);
                ksum12 += (uint32_t) (v12x3);
                ksum12 += (uint32_t) (v12x4);
                ksum12 += (uint32_t) (v12x5);
                ksum12 += (uint32_t) (v12x6);
                ksum12 += (uint32_t) (v12x7);
                // Subtract 8 zero points (8)
                ksum12 -= 64;

                out[48] = (v12x0 | (v12x4 << 4)) ^ 0x88;
                out[49] = (v12x1 | (v12x5 << 4)) ^ 0x88;
                out[50] = (v12x2 | (v12x6 << 4)) ^ 0x88;
                out[51] = (v12x3 | (v12x7 << 4)) ^ 0x88;
                const uint8_t v13x0 = w13[0] & 0xF;
                const uint8_t v13x1 = w13[0] >> 4;
                const uint8_t v13x2 = w13[1] & 0xF;
                const uint8_t v13x3 = w13[1] >> 4;
                const uint8_t v13x4 = w13[2] & 0xF;
                const uint8_t v13x5 = w13[2] >> 4;
                const uint8_t v13x6 = w13[3] & 0xF;
                const uint8_t v13x7 = w13[3] >> 4;
                w13 += 4;

                ksum13 += (uint32_t) (v13x0);
                ksum13 += (uint32_t) (v13x1);
                ksum13 += (uint32_t) (v13x2);
                ksum13 += (uint32_t) (v13x3);
                ksum13 += (uint32_t) (v13x4);
                ksum13 += (uint32_t) (v13x5);
                ksum13 += (uint32_t) (v13x6);
                ksum13 += (uint32_t) (v13x7);
                // Subtract 8 zero points (8)
                ksum13 -= 64;

                out[52] = (v13x0 | (v13x4 << 4)) ^ 0x88;
                out[53] = (v13x1 | (v13x5 << 4)) ^ 0x88;
                out[54] = (v13x2 | (v13x6 << 4)) ^ 0x88;
                out[55] = (v13x3 | (v13x7 << 4)) ^ 0x88;
                const uint8_t v14x0 = w14[0] & 0xF;
                const uint8_t v14x1 = w14[0] >> 4;
                const uint8_t v14x2 = w14[1] & 0xF;
                const uint8_t v14x3 = w14[1] >> 4;
                const uint8_t v14x4 = w14[2] & 0xF;
                const uint8_t v14x5 = w14[2] >> 4;
                const uint8_t v14x6 = w14[3] & 0xF;
                const uint8_t v14x7 = w14[3] >> 4;
                w14 += 4;

                ksum14 += (uint32_t) (v14x0);
                ksum14 += (uint32_t) (v14x1);
                ksum14 += (uint32_t) (v14x2);
                ksum14 += (uint32_t) (v14x3);
                ksum14 += (uint32_t) (v14x4);
                ksum14 += (uint32_t) (v14x5);
                ksum14 += (uint32_t) (v14x6);
                ksum14 += (uint32_t) (v14x7);
                // Subtract 8 zero points (8)
                ksum14 -= 64;

                out[56] = (v14x0 | (v14x4 << 4)) ^ 0x88;
                out[57] = (v14x1 | (v14x5 << 4)) ^ 0x88;
                out[58] = (v14x2 | (v14x6 << 4)) ^ 0x88;
                out[59] = (v14x3 | (v14x7 << 4)) ^ 0x88;

                out += 64;
            }
            float scale0 = math_cvt_fp32_bf16(s0[0]);
            float scale1 = math_cvt_fp32_bf16(s1[0]);
            float scale2 = math_cvt_fp32_bf16(s2[0]);
            float scale3 = math_cvt_fp32_bf16(s3[0]);
            float scale4 = math_cvt_fp32_bf16(s4[0]);
            float scale5 = math_cvt_fp32_bf16(s5[0]);
            float scale6 = math_cvt_fp32_bf16(s6[0]);
            float scale7 = math_cvt_fp32_bf16(s7[0]);
            float scale8 = math_cvt_fp32_bf16(s8[0]);
            float scale9 = math_cvt_fp32_bf16(s9[0]);
            float scale10 = math_cvt_fp32_bf16(s10[0]);
            float scale11 = math_cvt_fp32_bf16(s11[0]);
            float scale12 = math_cvt_fp32_bf16(s12[0]);
            float scale13 = math_cvt_fp32_bf16(s13[0]);
            float scale14 = math_cvt_fp32_bf16(s14[0]);
            s0 += 1;
            s1 += 1;
            s2 += 1;
            s3 += 1;
            s4 += 1;
            s5 += 1;
            s6 += 1;
            s7 += 1;
            s8 += 1;
            s9 += 1;
            s10 += 1;
            s11 += 1;
            s12 += 1;
            s13 += 1;
            s14 += 1;


            packed_k_scaled_sum[0] -= (float)ksum0 * izp * scale0;
            packed_k_scaled_sum[1] -= (float)ksum1 * izp * scale1;
            packed_k_scaled_sum[2] -= (float)ksum2 * izp * scale2;
            packed_k_scaled_sum[3] -= (float)ksum3 * izp * scale3;
            packed_k_scaled_sum[4] -= (float)ksum4 * izp * scale4;
            packed_k_scaled_sum[5] -= (float)ksum5 * izp * scale5;
            packed_k_scaled_sum[6] -= (float)ksum6 * izp * scale6;
            packed_k_scaled_sum[7] -= (float)ksum7 * izp * scale7;
            packed_k_scaled_sum[8] -= (float)ksum8 * izp * scale8;
            packed_k_scaled_sum[9] -= (float)ksum9 * izp * scale9;
            packed_k_scaled_sum[10] -= (float)ksum10 * izp * scale10;
            packed_k_scaled_sum[11] -= (float)ksum11 * izp * scale11;
            packed_k_scaled_sum[12] -= (float)ksum12 * izp * scale12;
            packed_k_scaled_sum[13] -= (float)ksum13 * izp * scale13;
            packed_k_scaled_sum[14] -= (float)ksum14 * izp * scale14;

            ((uint16_t*) out)[0] = math_cvt_bf16_fp32(scale0 / 16.0f);
            ((uint16_t*) out)[1] = math_cvt_bf16_fp32(scale1 / 16.0f);
            ((uint16_t*) out)[2] = math_cvt_bf16_fp32(scale2 / 16.0f);
            ((uint16_t*) out)[3] = math_cvt_bf16_fp32(scale3 / 16.0f);
            ((uint16_t*) out)[4] = math_cvt_bf16_fp32(scale4 / 16.0f);
            ((uint16_t*) out)[5] = math_cvt_bf16_fp32(scale5 / 16.0f);
            ((uint16_t*) out)[6] = math_cvt_bf16_fp32(scale6 / 16.0f);
            ((uint16_t*) out)[7] = math_cvt_bf16_fp32(scale7 / 16.0f);
            ((uint16_t*) out)[8] = math_cvt_bf16_fp32(scale8 / 16.0f);
            ((uint16_t*) out)[9] = math_cvt_bf16_fp32(scale9 / 16.0f);
            ((uint16_t*) out)[10] = math_cvt_bf16_fp32(scale10 / 16.0f);
            ((uint16_t*) out)[11] = math_cvt_bf16_fp32(scale11 / 16.0f);
            ((uint16_t*) out)[12] = math_cvt_bf16_fp32(scale12 / 16.0f);
            ((uint16_t*) out)[13] = math_cvt_bf16_fp32(scale13 / 16.0f);
            ((uint16_t*) out)[14] = math_cvt_bf16_fp32(scale14 / 16.0f);

            out += 16 * sizeof(uint16_t);
        }


        if XNN_LIKELY(b != NULL){
            size_t nb = n;
            do {
                *((uint32_t*) out) = *b++;
                out += sizeof(uint32_t);
            } while(--nb != 0);
        } else {
            size_t nb = n;
            do {
                *((uint32_t*) out) = 0;
                out += sizeof(uint32_t);
            } while(--nb != 0);
        }
        out += 16 * sizeof(uint32_t);
    }
  } while (--g != 0);
}
