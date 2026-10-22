// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-packw/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/simd/s32-hvx.h"


void xnn_x32_packw_gemm_goi_ukernel_x64__hvx_u32(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t n_stride,
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

  const xnn_simd_s32_t vzero = xnn_set1_s32(0);
  int32_t* out = (int32_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;

  do {
    // NC main loop multiple of 64
    const int32_t* w0 = (const int32_t*) weights;
    size_t n = nc;
    for (; n >= 64; n -= 64) {
      if XNN_LIKELY(b != NULL) {
        const xnn_simd_s32_t vb0 = xnn_loadu_s32(b + 0);
        const xnn_simd_s32_t vb1 = xnn_loadu_s32(b + 32);
        xnn_store_s32(out + 0, vb0);
        xnn_store_s32(out + 32, vb1);
        b += 64;
      } else {
        xnn_store_s32(out + 0, vzero);
        xnn_store_s32(out + 32, vzero);
      }
      out += 64;

      const int32_t* w1 = w0 + n_stride;
      const int32_t* w2 = w1 + n_stride;
      const int32_t* w3 = w2 + n_stride;
      const int32_t* w4 = w3 + n_stride;
      const int32_t* w5 = w4 + n_stride;
      const int32_t* w6 = w5 + n_stride;
      const int32_t* w7 = w6 + n_stride;
      const int32_t* w8 = w7 + n_stride;
      const int32_t* w9 = w8 + n_stride;
      const int32_t* w10 = w9 + n_stride;
      const int32_t* w11 = w10 + n_stride;
      const int32_t* w12 = w11 + n_stride;
      const int32_t* w13 = w12 + n_stride;
      const int32_t* w14 = w13 + n_stride;
      const int32_t* w15 = w14 + n_stride;
      const int32_t* w16 = w15 + n_stride;
      const int32_t* w17 = w16 + n_stride;
      const int32_t* w18 = w17 + n_stride;
      const int32_t* w19 = w18 + n_stride;
      const int32_t* w20 = w19 + n_stride;
      const int32_t* w21 = w20 + n_stride;
      const int32_t* w22 = w21 + n_stride;
      const int32_t* w23 = w22 + n_stride;
      const int32_t* w24 = w23 + n_stride;
      const int32_t* w25 = w24 + n_stride;
      const int32_t* w26 = w25 + n_stride;
      const int32_t* w27 = w26 + n_stride;
      const int32_t* w28 = w27 + n_stride;
      const int32_t* w29 = w28 + n_stride;
      const int32_t* w30 = w29 + n_stride;
      const int32_t* w31 = w30 + n_stride;
      const int32_t* w32 = w31 + n_stride;
      const int32_t* w33 = w32 + n_stride;
      const int32_t* w34 = w33 + n_stride;
      const int32_t* w35 = w34 + n_stride;
      const int32_t* w36 = w35 + n_stride;
      const int32_t* w37 = w36 + n_stride;
      const int32_t* w38 = w37 + n_stride;
      const int32_t* w39 = w38 + n_stride;
      const int32_t* w40 = w39 + n_stride;
      const int32_t* w41 = w40 + n_stride;
      const int32_t* w42 = w41 + n_stride;
      const int32_t* w43 = w42 + n_stride;
      const int32_t* w44 = w43 + n_stride;
      const int32_t* w45 = w44 + n_stride;
      const int32_t* w46 = w45 + n_stride;
      const int32_t* w47 = w46 + n_stride;
      const int32_t* w48 = w47 + n_stride;
      const int32_t* w49 = w48 + n_stride;
      const int32_t* w50 = w49 + n_stride;
      const int32_t* w51 = w50 + n_stride;
      const int32_t* w52 = w51 + n_stride;
      const int32_t* w53 = w52 + n_stride;
      const int32_t* w54 = w53 + n_stride;
      const int32_t* w55 = w54 + n_stride;
      const int32_t* w56 = w55 + n_stride;
      const int32_t* w57 = w56 + n_stride;
      const int32_t* w58 = w57 + n_stride;
      const int32_t* w59 = w58 + n_stride;
      const int32_t* w60 = w59 + n_stride;
      const int32_t* w61 = w60 + n_stride;
      const int32_t* w62 = w61 + n_stride;
      const int32_t* w63 = w62 + n_stride;

      // KC main loop multiple of 64x32 using 32x32 HVX butterfly transpose
      size_t k = kc;
      for (; k >= 32; k -= 32) {
        {
          const HVX_Vector v5_0 = xnn_loadu_s32(w0);
          w0 += 32;
          const HVX_Vector v5_1 = xnn_loadu_s32(w1);
          w1 += 32;
          const HVX_Vector v5_2 = xnn_loadu_s32(w2);
          w2 += 32;
          const HVX_Vector v5_3 = xnn_loadu_s32(w3);
          w3 += 32;
          const HVX_Vector v5_4 = xnn_loadu_s32(w4);
          w4 += 32;
          const HVX_Vector v5_5 = xnn_loadu_s32(w5);
          w5 += 32;
          const HVX_Vector v5_6 = xnn_loadu_s32(w6);
          w6 += 32;
          const HVX_Vector v5_7 = xnn_loadu_s32(w7);
          w7 += 32;
          const HVX_Vector v5_8 = xnn_loadu_s32(w8);
          w8 += 32;
          const HVX_Vector v5_9 = xnn_loadu_s32(w9);
          w9 += 32;
          const HVX_Vector v5_10 = xnn_loadu_s32(w10);
          w10 += 32;
          const HVX_Vector v5_11 = xnn_loadu_s32(w11);
          w11 += 32;
          const HVX_Vector v5_12 = xnn_loadu_s32(w12);
          w12 += 32;
          const HVX_Vector v5_13 = xnn_loadu_s32(w13);
          w13 += 32;
          const HVX_Vector v5_14 = xnn_loadu_s32(w14);
          w14 += 32;
          const HVX_Vector v5_15 = xnn_loadu_s32(w15);
          w15 += 32;
          const HVX_Vector v5_16 = xnn_loadu_s32(w16);
          w16 += 32;
          const HVX_Vector v5_17 = xnn_loadu_s32(w17);
          w17 += 32;
          const HVX_Vector v5_18 = xnn_loadu_s32(w18);
          w18 += 32;
          const HVX_Vector v5_19 = xnn_loadu_s32(w19);
          w19 += 32;
          const HVX_Vector v5_20 = xnn_loadu_s32(w20);
          w20 += 32;
          const HVX_Vector v5_21 = xnn_loadu_s32(w21);
          w21 += 32;
          const HVX_Vector v5_22 = xnn_loadu_s32(w22);
          w22 += 32;
          const HVX_Vector v5_23 = xnn_loadu_s32(w23);
          w23 += 32;
          const HVX_Vector v5_24 = xnn_loadu_s32(w24);
          w24 += 32;
          const HVX_Vector v5_25 = xnn_loadu_s32(w25);
          w25 += 32;
          const HVX_Vector v5_26 = xnn_loadu_s32(w26);
          w26 += 32;
          const HVX_Vector v5_27 = xnn_loadu_s32(w27);
          w27 += 32;
          const HVX_Vector v5_28 = xnn_loadu_s32(w28);
          w28 += 32;
          const HVX_Vector v5_29 = xnn_loadu_s32(w29);
          w29 += 32;
          const HVX_Vector v5_30 = xnn_loadu_s32(w30);
          w30 += 32;
          const HVX_Vector v5_31 = xnn_loadu_s32(w31);
          w31 += 32;

          const HVX_VectorPair v4_0 = Q6_W_vshuff_VVR(v5_1, v5_0, -4);
          const HVX_VectorPair v4_1 = Q6_W_vshuff_VVR(v5_3, v5_2, -4);
          const HVX_VectorPair v4_2 = Q6_W_vshuff_VVR(v5_5, v5_4, -4);
          const HVX_VectorPair v4_3 = Q6_W_vshuff_VVR(v5_7, v5_6, -4);
          const HVX_VectorPair v4_4 = Q6_W_vshuff_VVR(v5_9, v5_8, -4);
          const HVX_VectorPair v4_5 = Q6_W_vshuff_VVR(v5_11, v5_10, -4);
          const HVX_VectorPair v4_6 = Q6_W_vshuff_VVR(v5_13, v5_12, -4);
          const HVX_VectorPair v4_7 = Q6_W_vshuff_VVR(v5_15, v5_14, -4);
          const HVX_VectorPair v4_8 = Q6_W_vshuff_VVR(v5_17, v5_16, -4);
          const HVX_VectorPair v4_9 = Q6_W_vshuff_VVR(v5_19, v5_18, -4);
          const HVX_VectorPair v4_10 = Q6_W_vshuff_VVR(v5_21, v5_20, -4);
          const HVX_VectorPair v4_11 = Q6_W_vshuff_VVR(v5_23, v5_22, -4);
          const HVX_VectorPair v4_12 = Q6_W_vshuff_VVR(v5_25, v5_24, -4);
          const HVX_VectorPair v4_13 = Q6_W_vshuff_VVR(v5_27, v5_26, -4);
          const HVX_VectorPair v4_14 = Q6_W_vshuff_VVR(v5_29, v5_28, -4);
          const HVX_VectorPair v4_15 = Q6_W_vshuff_VVR(v5_31, v5_30, -4);

          const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), -8);
          const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), -8);
          const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), -8);
          const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), -8);
          const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), -8);
          const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), -8);
          const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), -8);
          const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), -8);
          const HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), -8);
          const HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), -8);
          const HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), -8);
          const HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), -8);
          const HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), -8);
          const HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), -8);
          const HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), -8);
          const HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), -8);

          const HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), -16);
          const HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), -16);
          const HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), -16);
          const HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), -16);
          const HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), -16);
          const HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), -16);
          const HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), -16);
          const HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), -16);
          const HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), -16);
          const HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), -16);
          const HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), -16);
          const HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), -16);
          const HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), -16);
          const HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), -16);
          const HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), -16);
          const HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), -16);

          const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), -32);
          const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), -32);
          const HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), -32);
          const HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), -32);
          const HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), -32);
          const HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), -32);
          const HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), -32);
          const HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), -32);
          const HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), -32);
          const HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), -32);
          const HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), -32);
          const HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), -32);
          const HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), -32);
          const HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), -32);
          const HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), -32);
          const HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), -32);

          const HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), -64);
          const HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), -64);
          const HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), -64);
          const HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), -64);
          const HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), -64);
          const HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), -64);
          const HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), -64);
          const HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), -64);
          const HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), -64);
          const HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), -64);
          const HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), -64);
          const HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), -64);
          const HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), -64);
          const HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), -64);
          const HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), -64);
          const HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), -64);

          xnn_store_s32(out + 0, Q6_V_lo_W(v0_0));
          xnn_store_s32(out + 64, Q6_V_hi_W(v0_0));
          xnn_store_s32(out + 128, Q6_V_lo_W(v0_1));
          xnn_store_s32(out + 192, Q6_V_hi_W(v0_1));
          xnn_store_s32(out + 256, Q6_V_lo_W(v0_2));
          xnn_store_s32(out + 320, Q6_V_hi_W(v0_2));
          xnn_store_s32(out + 384, Q6_V_lo_W(v0_3));
          xnn_store_s32(out + 448, Q6_V_hi_W(v0_3));
          xnn_store_s32(out + 512, Q6_V_lo_W(v0_4));
          xnn_store_s32(out + 576, Q6_V_hi_W(v0_4));
          xnn_store_s32(out + 640, Q6_V_lo_W(v0_5));
          xnn_store_s32(out + 704, Q6_V_hi_W(v0_5));
          xnn_store_s32(out + 768, Q6_V_lo_W(v0_6));
          xnn_store_s32(out + 832, Q6_V_hi_W(v0_6));
          xnn_store_s32(out + 896, Q6_V_lo_W(v0_7));
          xnn_store_s32(out + 960, Q6_V_hi_W(v0_7));
          xnn_store_s32(out + 1024, Q6_V_lo_W(v0_8));
          xnn_store_s32(out + 1088, Q6_V_hi_W(v0_8));
          xnn_store_s32(out + 1152, Q6_V_lo_W(v0_9));
          xnn_store_s32(out + 1216, Q6_V_hi_W(v0_9));
          xnn_store_s32(out + 1280, Q6_V_lo_W(v0_10));
          xnn_store_s32(out + 1344, Q6_V_hi_W(v0_10));
          xnn_store_s32(out + 1408, Q6_V_lo_W(v0_11));
          xnn_store_s32(out + 1472, Q6_V_hi_W(v0_11));
          xnn_store_s32(out + 1536, Q6_V_lo_W(v0_12));
          xnn_store_s32(out + 1600, Q6_V_hi_W(v0_12));
          xnn_store_s32(out + 1664, Q6_V_lo_W(v0_13));
          xnn_store_s32(out + 1728, Q6_V_hi_W(v0_13));
          xnn_store_s32(out + 1792, Q6_V_lo_W(v0_14));
          xnn_store_s32(out + 1856, Q6_V_hi_W(v0_14));
          xnn_store_s32(out + 1920, Q6_V_lo_W(v0_15));
          xnn_store_s32(out + 1984, Q6_V_hi_W(v0_15));
        }
        {
          const HVX_Vector v5_0 = xnn_loadu_s32(w32);
          w32 += 32;
          const HVX_Vector v5_1 = xnn_loadu_s32(w33);
          w33 += 32;
          const HVX_Vector v5_2 = xnn_loadu_s32(w34);
          w34 += 32;
          const HVX_Vector v5_3 = xnn_loadu_s32(w35);
          w35 += 32;
          const HVX_Vector v5_4 = xnn_loadu_s32(w36);
          w36 += 32;
          const HVX_Vector v5_5 = xnn_loadu_s32(w37);
          w37 += 32;
          const HVX_Vector v5_6 = xnn_loadu_s32(w38);
          w38 += 32;
          const HVX_Vector v5_7 = xnn_loadu_s32(w39);
          w39 += 32;
          const HVX_Vector v5_8 = xnn_loadu_s32(w40);
          w40 += 32;
          const HVX_Vector v5_9 = xnn_loadu_s32(w41);
          w41 += 32;
          const HVX_Vector v5_10 = xnn_loadu_s32(w42);
          w42 += 32;
          const HVX_Vector v5_11 = xnn_loadu_s32(w43);
          w43 += 32;
          const HVX_Vector v5_12 = xnn_loadu_s32(w44);
          w44 += 32;
          const HVX_Vector v5_13 = xnn_loadu_s32(w45);
          w45 += 32;
          const HVX_Vector v5_14 = xnn_loadu_s32(w46);
          w46 += 32;
          const HVX_Vector v5_15 = xnn_loadu_s32(w47);
          w47 += 32;
          const HVX_Vector v5_16 = xnn_loadu_s32(w48);
          w48 += 32;
          const HVX_Vector v5_17 = xnn_loadu_s32(w49);
          w49 += 32;
          const HVX_Vector v5_18 = xnn_loadu_s32(w50);
          w50 += 32;
          const HVX_Vector v5_19 = xnn_loadu_s32(w51);
          w51 += 32;
          const HVX_Vector v5_20 = xnn_loadu_s32(w52);
          w52 += 32;
          const HVX_Vector v5_21 = xnn_loadu_s32(w53);
          w53 += 32;
          const HVX_Vector v5_22 = xnn_loadu_s32(w54);
          w54 += 32;
          const HVX_Vector v5_23 = xnn_loadu_s32(w55);
          w55 += 32;
          const HVX_Vector v5_24 = xnn_loadu_s32(w56);
          w56 += 32;
          const HVX_Vector v5_25 = xnn_loadu_s32(w57);
          w57 += 32;
          const HVX_Vector v5_26 = xnn_loadu_s32(w58);
          w58 += 32;
          const HVX_Vector v5_27 = xnn_loadu_s32(w59);
          w59 += 32;
          const HVX_Vector v5_28 = xnn_loadu_s32(w60);
          w60 += 32;
          const HVX_Vector v5_29 = xnn_loadu_s32(w61);
          w61 += 32;
          const HVX_Vector v5_30 = xnn_loadu_s32(w62);
          w62 += 32;
          const HVX_Vector v5_31 = xnn_loadu_s32(w63);
          w63 += 32;

          const HVX_VectorPair v4_0 = Q6_W_vshuff_VVR(v5_1, v5_0, -4);
          const HVX_VectorPair v4_1 = Q6_W_vshuff_VVR(v5_3, v5_2, -4);
          const HVX_VectorPair v4_2 = Q6_W_vshuff_VVR(v5_5, v5_4, -4);
          const HVX_VectorPair v4_3 = Q6_W_vshuff_VVR(v5_7, v5_6, -4);
          const HVX_VectorPair v4_4 = Q6_W_vshuff_VVR(v5_9, v5_8, -4);
          const HVX_VectorPair v4_5 = Q6_W_vshuff_VVR(v5_11, v5_10, -4);
          const HVX_VectorPair v4_6 = Q6_W_vshuff_VVR(v5_13, v5_12, -4);
          const HVX_VectorPair v4_7 = Q6_W_vshuff_VVR(v5_15, v5_14, -4);
          const HVX_VectorPair v4_8 = Q6_W_vshuff_VVR(v5_17, v5_16, -4);
          const HVX_VectorPair v4_9 = Q6_W_vshuff_VVR(v5_19, v5_18, -4);
          const HVX_VectorPair v4_10 = Q6_W_vshuff_VVR(v5_21, v5_20, -4);
          const HVX_VectorPair v4_11 = Q6_W_vshuff_VVR(v5_23, v5_22, -4);
          const HVX_VectorPair v4_12 = Q6_W_vshuff_VVR(v5_25, v5_24, -4);
          const HVX_VectorPair v4_13 = Q6_W_vshuff_VVR(v5_27, v5_26, -4);
          const HVX_VectorPair v4_14 = Q6_W_vshuff_VVR(v5_29, v5_28, -4);
          const HVX_VectorPair v4_15 = Q6_W_vshuff_VVR(v5_31, v5_30, -4);

          const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), -8);
          const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), -8);
          const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), -8);
          const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), -8);
          const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), -8);
          const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), -8);
          const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), -8);
          const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), -8);
          const HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), -8);
          const HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), -8);
          const HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), -8);
          const HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), -8);
          const HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), -8);
          const HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), -8);
          const HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), -8);
          const HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), -8);

          const HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), -16);
          const HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), -16);
          const HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), -16);
          const HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), -16);
          const HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), -16);
          const HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), -16);
          const HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), -16);
          const HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), -16);
          const HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), -16);
          const HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), -16);
          const HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), -16);
          const HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), -16);
          const HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), -16);
          const HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), -16);
          const HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), -16);
          const HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), -16);

          const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), -32);
          const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), -32);
          const HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), -32);
          const HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), -32);
          const HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), -32);
          const HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), -32);
          const HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), -32);
          const HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), -32);
          const HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), -32);
          const HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), -32);
          const HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), -32);
          const HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), -32);
          const HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), -32);
          const HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), -32);
          const HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), -32);
          const HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), -32);

          const HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), -64);
          const HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), -64);
          const HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), -64);
          const HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), -64);
          const HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), -64);
          const HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), -64);
          const HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), -64);
          const HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), -64);
          const HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), -64);
          const HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), -64);
          const HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), -64);
          const HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), -64);
          const HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), -64);
          const HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), -64);
          const HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), -64);
          const HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), -64);

          xnn_store_s32(out + 32, Q6_V_lo_W(v0_0));
          xnn_store_s32(out + 96, Q6_V_hi_W(v0_0));
          xnn_store_s32(out + 160, Q6_V_lo_W(v0_1));
          xnn_store_s32(out + 224, Q6_V_hi_W(v0_1));
          xnn_store_s32(out + 288, Q6_V_lo_W(v0_2));
          xnn_store_s32(out + 352, Q6_V_hi_W(v0_2));
          xnn_store_s32(out + 416, Q6_V_lo_W(v0_3));
          xnn_store_s32(out + 480, Q6_V_hi_W(v0_3));
          xnn_store_s32(out + 544, Q6_V_lo_W(v0_4));
          xnn_store_s32(out + 608, Q6_V_hi_W(v0_4));
          xnn_store_s32(out + 672, Q6_V_lo_W(v0_5));
          xnn_store_s32(out + 736, Q6_V_hi_W(v0_5));
          xnn_store_s32(out + 800, Q6_V_lo_W(v0_6));
          xnn_store_s32(out + 864, Q6_V_hi_W(v0_6));
          xnn_store_s32(out + 928, Q6_V_lo_W(v0_7));
          xnn_store_s32(out + 992, Q6_V_hi_W(v0_7));
          xnn_store_s32(out + 1056, Q6_V_lo_W(v0_8));
          xnn_store_s32(out + 1120, Q6_V_hi_W(v0_8));
          xnn_store_s32(out + 1184, Q6_V_lo_W(v0_9));
          xnn_store_s32(out + 1248, Q6_V_hi_W(v0_9));
          xnn_store_s32(out + 1312, Q6_V_lo_W(v0_10));
          xnn_store_s32(out + 1376, Q6_V_hi_W(v0_10));
          xnn_store_s32(out + 1440, Q6_V_lo_W(v0_11));
          xnn_store_s32(out + 1504, Q6_V_hi_W(v0_11));
          xnn_store_s32(out + 1568, Q6_V_lo_W(v0_12));
          xnn_store_s32(out + 1632, Q6_V_hi_W(v0_12));
          xnn_store_s32(out + 1696, Q6_V_lo_W(v0_13));
          xnn_store_s32(out + 1760, Q6_V_hi_W(v0_13));
          xnn_store_s32(out + 1824, Q6_V_lo_W(v0_14));
          xnn_store_s32(out + 1888, Q6_V_hi_W(v0_14));
          xnn_store_s32(out + 1952, Q6_V_lo_W(v0_15));
          xnn_store_s32(out + 2016, Q6_V_hi_W(v0_15));
        }
        out += 2048;
      }

      // KC remainder (1..31)
      for (; k >= 2; k -= 2) {
        const int32_t v0_0 = w0[0];
        const int32_t v0_1 = w0[1];
        w0 += 2;
        const int32_t v1_0 = w1[0];
        const int32_t v1_1 = w1[1];
        w1 += 2;
        const int32_t v2_0 = w2[0];
        const int32_t v2_1 = w2[1];
        w2 += 2;
        const int32_t v3_0 = w3[0];
        const int32_t v3_1 = w3[1];
        w3 += 2;
        const int32_t v4_0 = w4[0];
        const int32_t v4_1 = w4[1];
        w4 += 2;
        const int32_t v5_0 = w5[0];
        const int32_t v5_1 = w5[1];
        w5 += 2;
        const int32_t v6_0 = w6[0];
        const int32_t v6_1 = w6[1];
        w6 += 2;
        const int32_t v7_0 = w7[0];
        const int32_t v7_1 = w7[1];
        w7 += 2;
        const int32_t v8_0 = w8[0];
        const int32_t v8_1 = w8[1];
        w8 += 2;
        const int32_t v9_0 = w9[0];
        const int32_t v9_1 = w9[1];
        w9 += 2;
        const int32_t v10_0 = w10[0];
        const int32_t v10_1 = w10[1];
        w10 += 2;
        const int32_t v11_0 = w11[0];
        const int32_t v11_1 = w11[1];
        w11 += 2;
        const int32_t v12_0 = w12[0];
        const int32_t v12_1 = w12[1];
        w12 += 2;
        const int32_t v13_0 = w13[0];
        const int32_t v13_1 = w13[1];
        w13 += 2;
        const int32_t v14_0 = w14[0];
        const int32_t v14_1 = w14[1];
        w14 += 2;
        const int32_t v15_0 = w15[0];
        const int32_t v15_1 = w15[1];
        w15 += 2;
        const int32_t v16_0 = w16[0];
        const int32_t v16_1 = w16[1];
        w16 += 2;
        const int32_t v17_0 = w17[0];
        const int32_t v17_1 = w17[1];
        w17 += 2;
        const int32_t v18_0 = w18[0];
        const int32_t v18_1 = w18[1];
        w18 += 2;
        const int32_t v19_0 = w19[0];
        const int32_t v19_1 = w19[1];
        w19 += 2;
        const int32_t v20_0 = w20[0];
        const int32_t v20_1 = w20[1];
        w20 += 2;
        const int32_t v21_0 = w21[0];
        const int32_t v21_1 = w21[1];
        w21 += 2;
        const int32_t v22_0 = w22[0];
        const int32_t v22_1 = w22[1];
        w22 += 2;
        const int32_t v23_0 = w23[0];
        const int32_t v23_1 = w23[1];
        w23 += 2;
        const int32_t v24_0 = w24[0];
        const int32_t v24_1 = w24[1];
        w24 += 2;
        const int32_t v25_0 = w25[0];
        const int32_t v25_1 = w25[1];
        w25 += 2;
        const int32_t v26_0 = w26[0];
        const int32_t v26_1 = w26[1];
        w26 += 2;
        const int32_t v27_0 = w27[0];
        const int32_t v27_1 = w27[1];
        w27 += 2;
        const int32_t v28_0 = w28[0];
        const int32_t v28_1 = w28[1];
        w28 += 2;
        const int32_t v29_0 = w29[0];
        const int32_t v29_1 = w29[1];
        w29 += 2;
        const int32_t v30_0 = w30[0];
        const int32_t v30_1 = w30[1];
        w30 += 2;
        const int32_t v31_0 = w31[0];
        const int32_t v31_1 = w31[1];
        w31 += 2;
        const int32_t v32_0 = w32[0];
        const int32_t v32_1 = w32[1];
        w32 += 2;
        const int32_t v33_0 = w33[0];
        const int32_t v33_1 = w33[1];
        w33 += 2;
        const int32_t v34_0 = w34[0];
        const int32_t v34_1 = w34[1];
        w34 += 2;
        const int32_t v35_0 = w35[0];
        const int32_t v35_1 = w35[1];
        w35 += 2;
        const int32_t v36_0 = w36[0];
        const int32_t v36_1 = w36[1];
        w36 += 2;
        const int32_t v37_0 = w37[0];
        const int32_t v37_1 = w37[1];
        w37 += 2;
        const int32_t v38_0 = w38[0];
        const int32_t v38_1 = w38[1];
        w38 += 2;
        const int32_t v39_0 = w39[0];
        const int32_t v39_1 = w39[1];
        w39 += 2;
        const int32_t v40_0 = w40[0];
        const int32_t v40_1 = w40[1];
        w40 += 2;
        const int32_t v41_0 = w41[0];
        const int32_t v41_1 = w41[1];
        w41 += 2;
        const int32_t v42_0 = w42[0];
        const int32_t v42_1 = w42[1];
        w42 += 2;
        const int32_t v43_0 = w43[0];
        const int32_t v43_1 = w43[1];
        w43 += 2;
        const int32_t v44_0 = w44[0];
        const int32_t v44_1 = w44[1];
        w44 += 2;
        const int32_t v45_0 = w45[0];
        const int32_t v45_1 = w45[1];
        w45 += 2;
        const int32_t v46_0 = w46[0];
        const int32_t v46_1 = w46[1];
        w46 += 2;
        const int32_t v47_0 = w47[0];
        const int32_t v47_1 = w47[1];
        w47 += 2;
        const int32_t v48_0 = w48[0];
        const int32_t v48_1 = w48[1];
        w48 += 2;
        const int32_t v49_0 = w49[0];
        const int32_t v49_1 = w49[1];
        w49 += 2;
        const int32_t v50_0 = w50[0];
        const int32_t v50_1 = w50[1];
        w50 += 2;
        const int32_t v51_0 = w51[0];
        const int32_t v51_1 = w51[1];
        w51 += 2;
        const int32_t v52_0 = w52[0];
        const int32_t v52_1 = w52[1];
        w52 += 2;
        const int32_t v53_0 = w53[0];
        const int32_t v53_1 = w53[1];
        w53 += 2;
        const int32_t v54_0 = w54[0];
        const int32_t v54_1 = w54[1];
        w54 += 2;
        const int32_t v55_0 = w55[0];
        const int32_t v55_1 = w55[1];
        w55 += 2;
        const int32_t v56_0 = w56[0];
        const int32_t v56_1 = w56[1];
        w56 += 2;
        const int32_t v57_0 = w57[0];
        const int32_t v57_1 = w57[1];
        w57 += 2;
        const int32_t v58_0 = w58[0];
        const int32_t v58_1 = w58[1];
        w58 += 2;
        const int32_t v59_0 = w59[0];
        const int32_t v59_1 = w59[1];
        w59 += 2;
        const int32_t v60_0 = w60[0];
        const int32_t v60_1 = w60[1];
        w60 += 2;
        const int32_t v61_0 = w61[0];
        const int32_t v61_1 = w61[1];
        w61 += 2;
        const int32_t v62_0 = w62[0];
        const int32_t v62_1 = w62[1];
        w62 += 2;
        const int32_t v63_0 = w63[0];
        const int32_t v63_1 = w63[1];
        w63 += 2;
        out[0] = v0_0;
        out[64] = v0_1;
        out[1] = v1_0;
        out[65] = v1_1;
        out[2] = v2_0;
        out[66] = v2_1;
        out[3] = v3_0;
        out[67] = v3_1;
        out[4] = v4_0;
        out[68] = v4_1;
        out[5] = v5_0;
        out[69] = v5_1;
        out[6] = v6_0;
        out[70] = v6_1;
        out[7] = v7_0;
        out[71] = v7_1;
        out[8] = v8_0;
        out[72] = v8_1;
        out[9] = v9_0;
        out[73] = v9_1;
        out[10] = v10_0;
        out[74] = v10_1;
        out[11] = v11_0;
        out[75] = v11_1;
        out[12] = v12_0;
        out[76] = v12_1;
        out[13] = v13_0;
        out[77] = v13_1;
        out[14] = v14_0;
        out[78] = v14_1;
        out[15] = v15_0;
        out[79] = v15_1;
        out[16] = v16_0;
        out[80] = v16_1;
        out[17] = v17_0;
        out[81] = v17_1;
        out[18] = v18_0;
        out[82] = v18_1;
        out[19] = v19_0;
        out[83] = v19_1;
        out[20] = v20_0;
        out[84] = v20_1;
        out[21] = v21_0;
        out[85] = v21_1;
        out[22] = v22_0;
        out[86] = v22_1;
        out[23] = v23_0;
        out[87] = v23_1;
        out[24] = v24_0;
        out[88] = v24_1;
        out[25] = v25_0;
        out[89] = v25_1;
        out[26] = v26_0;
        out[90] = v26_1;
        out[27] = v27_0;
        out[91] = v27_1;
        out[28] = v28_0;
        out[92] = v28_1;
        out[29] = v29_0;
        out[93] = v29_1;
        out[30] = v30_0;
        out[94] = v30_1;
        out[31] = v31_0;
        out[95] = v31_1;
        out[32] = v32_0;
        out[96] = v32_1;
        out[33] = v33_0;
        out[97] = v33_1;
        out[34] = v34_0;
        out[98] = v34_1;
        out[35] = v35_0;
        out[99] = v35_1;
        out[36] = v36_0;
        out[100] = v36_1;
        out[37] = v37_0;
        out[101] = v37_1;
        out[38] = v38_0;
        out[102] = v38_1;
        out[39] = v39_0;
        out[103] = v39_1;
        out[40] = v40_0;
        out[104] = v40_1;
        out[41] = v41_0;
        out[105] = v41_1;
        out[42] = v42_0;
        out[106] = v42_1;
        out[43] = v43_0;
        out[107] = v43_1;
        out[44] = v44_0;
        out[108] = v44_1;
        out[45] = v45_0;
        out[109] = v45_1;
        out[46] = v46_0;
        out[110] = v46_1;
        out[47] = v47_0;
        out[111] = v47_1;
        out[48] = v48_0;
        out[112] = v48_1;
        out[49] = v49_0;
        out[113] = v49_1;
        out[50] = v50_0;
        out[114] = v50_1;
        out[51] = v51_0;
        out[115] = v51_1;
        out[52] = v52_0;
        out[116] = v52_1;
        out[53] = v53_0;
        out[117] = v53_1;
        out[54] = v54_0;
        out[118] = v54_1;
        out[55] = v55_0;
        out[119] = v55_1;
        out[56] = v56_0;
        out[120] = v56_1;
        out[57] = v57_0;
        out[121] = v57_1;
        out[58] = v58_0;
        out[122] = v58_1;
        out[59] = v59_0;
        out[123] = v59_1;
        out[60] = v60_0;
        out[124] = v60_1;
        out[61] = v61_0;
        out[125] = v61_1;
        out[62] = v62_0;
        out[126] = v62_1;
        out[63] = v63_0;
        out[127] = v63_1;
        out += 128;
      }
      for (; k != 0; --k) {
        const int32_t v0 = *w0++;
        out[0] = v0;
        const int32_t v1 = *w1++;
        out[1] = v1;
        const int32_t v2 = *w2++;
        out[2] = v2;
        const int32_t v3 = *w3++;
        out[3] = v3;
        const int32_t v4 = *w4++;
        out[4] = v4;
        const int32_t v5 = *w5++;
        out[5] = v5;
        const int32_t v6 = *w6++;
        out[6] = v6;
        const int32_t v7 = *w7++;
        out[7] = v7;
        const int32_t v8 = *w8++;
        out[8] = v8;
        const int32_t v9 = *w9++;
        out[9] = v9;
        const int32_t v10 = *w10++;
        out[10] = v10;
        const int32_t v11 = *w11++;
        out[11] = v11;
        const int32_t v12 = *w12++;
        out[12] = v12;
        const int32_t v13 = *w13++;
        out[13] = v13;
        const int32_t v14 = *w14++;
        out[14] = v14;
        const int32_t v15 = *w15++;
        out[15] = v15;
        const int32_t v16 = *w16++;
        out[16] = v16;
        const int32_t v17 = *w17++;
        out[17] = v17;
        const int32_t v18 = *w18++;
        out[18] = v18;
        const int32_t v19 = *w19++;
        out[19] = v19;
        const int32_t v20 = *w20++;
        out[20] = v20;
        const int32_t v21 = *w21++;
        out[21] = v21;
        const int32_t v22 = *w22++;
        out[22] = v22;
        const int32_t v23 = *w23++;
        out[23] = v23;
        const int32_t v24 = *w24++;
        out[24] = v24;
        const int32_t v25 = *w25++;
        out[25] = v25;
        const int32_t v26 = *w26++;
        out[26] = v26;
        const int32_t v27 = *w27++;
        out[27] = v27;
        const int32_t v28 = *w28++;
        out[28] = v28;
        const int32_t v29 = *w29++;
        out[29] = v29;
        const int32_t v30 = *w30++;
        out[30] = v30;
        const int32_t v31 = *w31++;
        out[31] = v31;
        const int32_t v32 = *w32++;
        out[32] = v32;
        const int32_t v33 = *w33++;
        out[33] = v33;
        const int32_t v34 = *w34++;
        out[34] = v34;
        const int32_t v35 = *w35++;
        out[35] = v35;
        const int32_t v36 = *w36++;
        out[36] = v36;
        const int32_t v37 = *w37++;
        out[37] = v37;
        const int32_t v38 = *w38++;
        out[38] = v38;
        const int32_t v39 = *w39++;
        out[39] = v39;
        const int32_t v40 = *w40++;
        out[40] = v40;
        const int32_t v41 = *w41++;
        out[41] = v41;
        const int32_t v42 = *w42++;
        out[42] = v42;
        const int32_t v43 = *w43++;
        out[43] = v43;
        const int32_t v44 = *w44++;
        out[44] = v44;
        const int32_t v45 = *w45++;
        out[45] = v45;
        const int32_t v46 = *w46++;
        out[46] = v46;
        const int32_t v47 = *w47++;
        out[47] = v47;
        const int32_t v48 = *w48++;
        out[48] = v48;
        const int32_t v49 = *w49++;
        out[49] = v49;
        const int32_t v50 = *w50++;
        out[50] = v50;
        const int32_t v51 = *w51++;
        out[51] = v51;
        const int32_t v52 = *w52++;
        out[52] = v52;
        const int32_t v53 = *w53++;
        out[53] = v53;
        const int32_t v54 = *w54++;
        out[54] = v54;
        const int32_t v55 = *w55++;
        out[55] = v55;
        const int32_t v56 = *w56++;
        out[56] = v56;
        const int32_t v57 = *w57++;
        out[57] = v57;
        const int32_t v58 = *w58++;
        out[58] = v58;
        const int32_t v59 = *w59++;
        out[59] = v59;
        const int32_t v60 = *w60++;
        out[60] = v60;
        const int32_t v61 = *w61++;
        out[61] = v61;
        const int32_t v62 = *w62++;
        out[62] = v62;
        const int32_t v63 = *w63++;
        out[63] = v63;
        out += 64;
      }
      out = (int32_t*) ((uintptr_t) out + extra_bytes);
      w0 = w63 + n_stride - kc;
    }

    // NC remainder (1..63)
    if XNN_UNLIKELY(n != 0) {
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *out++ = *b++;
        } while (--nb != 0);
        out += (64 - n);
      } else {
        xnn_store_s32(out + 0, vzero);
        xnn_store_s32(out + 32, vzero);
        out += 64;
      }

      // NR remainder has less than 64 rows so last row is not loaded
      const int32_t* w1 = w0 + n_stride;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int32_t* w2 = w1 + n_stride;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int32_t* w3 = w2 + n_stride;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int32_t* w4 = w3 + n_stride;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int32_t* w5 = w4 + n_stride;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int32_t* w6 = w5 + n_stride;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int32_t* w7 = w6 + n_stride;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const int32_t* w8 = w7 + n_stride;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const int32_t* w9 = w8 + n_stride;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const int32_t* w10 = w9 + n_stride;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const int32_t* w11 = w10 + n_stride;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const int32_t* w12 = w11 + n_stride;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const int32_t* w13 = w12 + n_stride;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const int32_t* w14 = w13 + n_stride;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const int32_t* w15 = w14 + n_stride;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const int32_t* w16 = w15 + n_stride;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const int32_t* w17 = w16 + n_stride;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const int32_t* w18 = w17 + n_stride;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const int32_t* w19 = w18 + n_stride;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const int32_t* w20 = w19 + n_stride;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const int32_t* w21 = w20 + n_stride;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const int32_t* w22 = w21 + n_stride;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const int32_t* w23 = w22 + n_stride;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const int32_t* w24 = w23 + n_stride;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const int32_t* w25 = w24 + n_stride;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const int32_t* w26 = w25 + n_stride;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const int32_t* w27 = w26 + n_stride;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const int32_t* w28 = w27 + n_stride;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const int32_t* w29 = w28 + n_stride;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const int32_t* w30 = w29 + n_stride;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }
      const int32_t* w31 = w30 + n_stride;
      if XNN_UNPREDICTABLE(n < 32) {
        w31 = w30;
      }
      const int32_t* w32 = w31 + n_stride;
      if XNN_UNPREDICTABLE(n <= 32) {
        w32 = w31;
      }
      const int32_t* w33 = w32 + n_stride;
      if XNN_UNPREDICTABLE(n < 34) {
        w33 = w32;
      }
      const int32_t* w34 = w33 + n_stride;
      if XNN_UNPREDICTABLE(n <= 34) {
        w34 = w33;
      }
      const int32_t* w35 = w34 + n_stride;
      if XNN_UNPREDICTABLE(n < 36) {
        w35 = w34;
      }
      const int32_t* w36 = w35 + n_stride;
      if XNN_UNPREDICTABLE(n <= 36) {
        w36 = w35;
      }
      const int32_t* w37 = w36 + n_stride;
      if XNN_UNPREDICTABLE(n < 38) {
        w37 = w36;
      }
      const int32_t* w38 = w37 + n_stride;
      if XNN_UNPREDICTABLE(n <= 38) {
        w38 = w37;
      }
      const int32_t* w39 = w38 + n_stride;
      if XNN_UNPREDICTABLE(n < 40) {
        w39 = w38;
      }
      const int32_t* w40 = w39 + n_stride;
      if XNN_UNPREDICTABLE(n <= 40) {
        w40 = w39;
      }
      const int32_t* w41 = w40 + n_stride;
      if XNN_UNPREDICTABLE(n < 42) {
        w41 = w40;
      }
      const int32_t* w42 = w41 + n_stride;
      if XNN_UNPREDICTABLE(n <= 42) {
        w42 = w41;
      }
      const int32_t* w43 = w42 + n_stride;
      if XNN_UNPREDICTABLE(n < 44) {
        w43 = w42;
      }
      const int32_t* w44 = w43 + n_stride;
      if XNN_UNPREDICTABLE(n <= 44) {
        w44 = w43;
      }
      const int32_t* w45 = w44 + n_stride;
      if XNN_UNPREDICTABLE(n < 46) {
        w45 = w44;
      }
      const int32_t* w46 = w45 + n_stride;
      if XNN_UNPREDICTABLE(n <= 46) {
        w46 = w45;
      }
      const int32_t* w47 = w46 + n_stride;
      if XNN_UNPREDICTABLE(n < 48) {
        w47 = w46;
      }
      const int32_t* w48 = w47 + n_stride;
      if XNN_UNPREDICTABLE(n <= 48) {
        w48 = w47;
      }
      const int32_t* w49 = w48 + n_stride;
      if XNN_UNPREDICTABLE(n < 50) {
        w49 = w48;
      }
      const int32_t* w50 = w49 + n_stride;
      if XNN_UNPREDICTABLE(n <= 50) {
        w50 = w49;
      }
      const int32_t* w51 = w50 + n_stride;
      if XNN_UNPREDICTABLE(n < 52) {
        w51 = w50;
      }
      const int32_t* w52 = w51 + n_stride;
      if XNN_UNPREDICTABLE(n <= 52) {
        w52 = w51;
      }
      const int32_t* w53 = w52 + n_stride;
      if XNN_UNPREDICTABLE(n < 54) {
        w53 = w52;
      }
      const int32_t* w54 = w53 + n_stride;
      if XNN_UNPREDICTABLE(n <= 54) {
        w54 = w53;
      }
      const int32_t* w55 = w54 + n_stride;
      if XNN_UNPREDICTABLE(n < 56) {
        w55 = w54;
      }
      const int32_t* w56 = w55 + n_stride;
      if XNN_UNPREDICTABLE(n <= 56) {
        w56 = w55;
      }
      const int32_t* w57 = w56 + n_stride;
      if XNN_UNPREDICTABLE(n < 58) {
        w57 = w56;
      }
      const int32_t* w58 = w57 + n_stride;
      if XNN_UNPREDICTABLE(n <= 58) {
        w58 = w57;
      }
      const int32_t* w59 = w58 + n_stride;
      if XNN_UNPREDICTABLE(n < 60) {
        w59 = w58;
      }
      const int32_t* w60 = w59 + n_stride;
      if XNN_UNPREDICTABLE(n <= 60) {
        w60 = w59;
      }
      const int32_t* w61 = w60 + n_stride;
      if XNN_UNPREDICTABLE(n < 62) {
        w61 = w60;
      }
      const int32_t* w62 = w61 + n_stride;
      if XNN_UNPREDICTABLE(n <= 62) {
        w62 = w61;
      }

      // KC main loop multiple of 64x32 using 32x32 HVX butterfly transpose
      size_t k = kc;
      for (; k >= 32; k -= 32) {
        {
          const HVX_Vector v5_0 = xnn_loadu_s32(w0);
          w0 += 32;
          const HVX_Vector v5_1 = xnn_loadu_s32(w1);
          w1 += 32;
          const HVX_Vector v5_2 = xnn_loadu_s32(w2);
          w2 += 32;
          const HVX_Vector v5_3 = xnn_loadu_s32(w3);
          w3 += 32;
          const HVX_Vector v5_4 = xnn_loadu_s32(w4);
          w4 += 32;
          const HVX_Vector v5_5 = xnn_loadu_s32(w5);
          w5 += 32;
          const HVX_Vector v5_6 = xnn_loadu_s32(w6);
          w6 += 32;
          const HVX_Vector v5_7 = xnn_loadu_s32(w7);
          w7 += 32;
          const HVX_Vector v5_8 = xnn_loadu_s32(w8);
          w8 += 32;
          const HVX_Vector v5_9 = xnn_loadu_s32(w9);
          w9 += 32;
          const HVX_Vector v5_10 = xnn_loadu_s32(w10);
          w10 += 32;
          const HVX_Vector v5_11 = xnn_loadu_s32(w11);
          w11 += 32;
          const HVX_Vector v5_12 = xnn_loadu_s32(w12);
          w12 += 32;
          const HVX_Vector v5_13 = xnn_loadu_s32(w13);
          w13 += 32;
          const HVX_Vector v5_14 = xnn_loadu_s32(w14);
          w14 += 32;
          const HVX_Vector v5_15 = xnn_loadu_s32(w15);
          w15 += 32;
          const HVX_Vector v5_16 = xnn_loadu_s32(w16);
          w16 += 32;
          const HVX_Vector v5_17 = xnn_loadu_s32(w17);
          w17 += 32;
          const HVX_Vector v5_18 = xnn_loadu_s32(w18);
          w18 += 32;
          const HVX_Vector v5_19 = xnn_loadu_s32(w19);
          w19 += 32;
          const HVX_Vector v5_20 = xnn_loadu_s32(w20);
          w20 += 32;
          const HVX_Vector v5_21 = xnn_loadu_s32(w21);
          w21 += 32;
          const HVX_Vector v5_22 = xnn_loadu_s32(w22);
          w22 += 32;
          const HVX_Vector v5_23 = xnn_loadu_s32(w23);
          w23 += 32;
          const HVX_Vector v5_24 = xnn_loadu_s32(w24);
          w24 += 32;
          const HVX_Vector v5_25 = xnn_loadu_s32(w25);
          w25 += 32;
          const HVX_Vector v5_26 = xnn_loadu_s32(w26);
          w26 += 32;
          const HVX_Vector v5_27 = xnn_loadu_s32(w27);
          w27 += 32;
          const HVX_Vector v5_28 = xnn_loadu_s32(w28);
          w28 += 32;
          const HVX_Vector v5_29 = xnn_loadu_s32(w29);
          w29 += 32;
          const HVX_Vector v5_30 = xnn_loadu_s32(w30);
          w30 += 32;
          const HVX_Vector v5_31 = xnn_loadu_s32(w31);
          w31 += 32;

          const HVX_VectorPair v4_0 = Q6_W_vshuff_VVR(v5_1, v5_0, -4);
          const HVX_VectorPair v4_1 = Q6_W_vshuff_VVR(v5_3, v5_2, -4);
          const HVX_VectorPair v4_2 = Q6_W_vshuff_VVR(v5_5, v5_4, -4);
          const HVX_VectorPair v4_3 = Q6_W_vshuff_VVR(v5_7, v5_6, -4);
          const HVX_VectorPair v4_4 = Q6_W_vshuff_VVR(v5_9, v5_8, -4);
          const HVX_VectorPair v4_5 = Q6_W_vshuff_VVR(v5_11, v5_10, -4);
          const HVX_VectorPair v4_6 = Q6_W_vshuff_VVR(v5_13, v5_12, -4);
          const HVX_VectorPair v4_7 = Q6_W_vshuff_VVR(v5_15, v5_14, -4);
          const HVX_VectorPair v4_8 = Q6_W_vshuff_VVR(v5_17, v5_16, -4);
          const HVX_VectorPair v4_9 = Q6_W_vshuff_VVR(v5_19, v5_18, -4);
          const HVX_VectorPair v4_10 = Q6_W_vshuff_VVR(v5_21, v5_20, -4);
          const HVX_VectorPair v4_11 = Q6_W_vshuff_VVR(v5_23, v5_22, -4);
          const HVX_VectorPair v4_12 = Q6_W_vshuff_VVR(v5_25, v5_24, -4);
          const HVX_VectorPair v4_13 = Q6_W_vshuff_VVR(v5_27, v5_26, -4);
          const HVX_VectorPair v4_14 = Q6_W_vshuff_VVR(v5_29, v5_28, -4);
          const HVX_VectorPair v4_15 = Q6_W_vshuff_VVR(v5_31, v5_30, -4);

          const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), -8);
          const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), -8);
          const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), -8);
          const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), -8);
          const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), -8);
          const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), -8);
          const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), -8);
          const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), -8);
          const HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), -8);
          const HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), -8);
          const HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), -8);
          const HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), -8);
          const HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), -8);
          const HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), -8);
          const HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), -8);
          const HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), -8);

          const HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), -16);
          const HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), -16);
          const HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), -16);
          const HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), -16);
          const HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), -16);
          const HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), -16);
          const HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), -16);
          const HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), -16);
          const HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), -16);
          const HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), -16);
          const HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), -16);
          const HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), -16);
          const HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), -16);
          const HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), -16);
          const HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), -16);
          const HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), -16);

          const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), -32);
          const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), -32);
          const HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), -32);
          const HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), -32);
          const HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), -32);
          const HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), -32);
          const HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), -32);
          const HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), -32);
          const HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), -32);
          const HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), -32);
          const HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), -32);
          const HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), -32);
          const HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), -32);
          const HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), -32);
          const HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), -32);
          const HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), -32);

          const HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), -64);
          const HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), -64);
          const HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), -64);
          const HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), -64);
          const HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), -64);
          const HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), -64);
          const HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), -64);
          const HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), -64);
          const HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), -64);
          const HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), -64);
          const HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), -64);
          const HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), -64);
          const HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), -64);
          const HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), -64);
          const HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), -64);
          const HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), -64);

          xnn_store_s32(out + 0, Q6_V_lo_W(v0_0));
          xnn_store_s32(out + 64, Q6_V_hi_W(v0_0));
          xnn_store_s32(out + 128, Q6_V_lo_W(v0_1));
          xnn_store_s32(out + 192, Q6_V_hi_W(v0_1));
          xnn_store_s32(out + 256, Q6_V_lo_W(v0_2));
          xnn_store_s32(out + 320, Q6_V_hi_W(v0_2));
          xnn_store_s32(out + 384, Q6_V_lo_W(v0_3));
          xnn_store_s32(out + 448, Q6_V_hi_W(v0_3));
          xnn_store_s32(out + 512, Q6_V_lo_W(v0_4));
          xnn_store_s32(out + 576, Q6_V_hi_W(v0_4));
          xnn_store_s32(out + 640, Q6_V_lo_W(v0_5));
          xnn_store_s32(out + 704, Q6_V_hi_W(v0_5));
          xnn_store_s32(out + 768, Q6_V_lo_W(v0_6));
          xnn_store_s32(out + 832, Q6_V_hi_W(v0_6));
          xnn_store_s32(out + 896, Q6_V_lo_W(v0_7));
          xnn_store_s32(out + 960, Q6_V_hi_W(v0_7));
          xnn_store_s32(out + 1024, Q6_V_lo_W(v0_8));
          xnn_store_s32(out + 1088, Q6_V_hi_W(v0_8));
          xnn_store_s32(out + 1152, Q6_V_lo_W(v0_9));
          xnn_store_s32(out + 1216, Q6_V_hi_W(v0_9));
          xnn_store_s32(out + 1280, Q6_V_lo_W(v0_10));
          xnn_store_s32(out + 1344, Q6_V_hi_W(v0_10));
          xnn_store_s32(out + 1408, Q6_V_lo_W(v0_11));
          xnn_store_s32(out + 1472, Q6_V_hi_W(v0_11));
          xnn_store_s32(out + 1536, Q6_V_lo_W(v0_12));
          xnn_store_s32(out + 1600, Q6_V_hi_W(v0_12));
          xnn_store_s32(out + 1664, Q6_V_lo_W(v0_13));
          xnn_store_s32(out + 1728, Q6_V_hi_W(v0_13));
          xnn_store_s32(out + 1792, Q6_V_lo_W(v0_14));
          xnn_store_s32(out + 1856, Q6_V_hi_W(v0_14));
          xnn_store_s32(out + 1920, Q6_V_lo_W(v0_15));
          xnn_store_s32(out + 1984, Q6_V_hi_W(v0_15));
        }
        {
          const HVX_Vector v5_0 = xnn_loadu_s32(w32);
          w32 += 32;
          const HVX_Vector v5_1 = xnn_loadu_s32(w33);
          w33 += 32;
          const HVX_Vector v5_2 = xnn_loadu_s32(w34);
          w34 += 32;
          const HVX_Vector v5_3 = xnn_loadu_s32(w35);
          w35 += 32;
          const HVX_Vector v5_4 = xnn_loadu_s32(w36);
          w36 += 32;
          const HVX_Vector v5_5 = xnn_loadu_s32(w37);
          w37 += 32;
          const HVX_Vector v5_6 = xnn_loadu_s32(w38);
          w38 += 32;
          const HVX_Vector v5_7 = xnn_loadu_s32(w39);
          w39 += 32;
          const HVX_Vector v5_8 = xnn_loadu_s32(w40);
          w40 += 32;
          const HVX_Vector v5_9 = xnn_loadu_s32(w41);
          w41 += 32;
          const HVX_Vector v5_10 = xnn_loadu_s32(w42);
          w42 += 32;
          const HVX_Vector v5_11 = xnn_loadu_s32(w43);
          w43 += 32;
          const HVX_Vector v5_12 = xnn_loadu_s32(w44);
          w44 += 32;
          const HVX_Vector v5_13 = xnn_loadu_s32(w45);
          w45 += 32;
          const HVX_Vector v5_14 = xnn_loadu_s32(w46);
          w46 += 32;
          const HVX_Vector v5_15 = xnn_loadu_s32(w47);
          w47 += 32;
          const HVX_Vector v5_16 = xnn_loadu_s32(w48);
          w48 += 32;
          const HVX_Vector v5_17 = xnn_loadu_s32(w49);
          w49 += 32;
          const HVX_Vector v5_18 = xnn_loadu_s32(w50);
          w50 += 32;
          const HVX_Vector v5_19 = xnn_loadu_s32(w51);
          w51 += 32;
          const HVX_Vector v5_20 = xnn_loadu_s32(w52);
          w52 += 32;
          const HVX_Vector v5_21 = xnn_loadu_s32(w53);
          w53 += 32;
          const HVX_Vector v5_22 = xnn_loadu_s32(w54);
          w54 += 32;
          const HVX_Vector v5_23 = xnn_loadu_s32(w55);
          w55 += 32;
          const HVX_Vector v5_24 = xnn_loadu_s32(w56);
          w56 += 32;
          const HVX_Vector v5_25 = xnn_loadu_s32(w57);
          w57 += 32;
          const HVX_Vector v5_26 = xnn_loadu_s32(w58);
          w58 += 32;
          const HVX_Vector v5_27 = xnn_loadu_s32(w59);
          w59 += 32;
          const HVX_Vector v5_28 = xnn_loadu_s32(w60);
          w60 += 32;
          const HVX_Vector v5_29 = xnn_loadu_s32(w61);
          w61 += 32;
          const HVX_Vector v5_30 = xnn_loadu_s32(w62);
          w62 += 32;
          const HVX_Vector v5_31 = vzero;

          const HVX_VectorPair v4_0 = Q6_W_vshuff_VVR(v5_1, v5_0, -4);
          const HVX_VectorPair v4_1 = Q6_W_vshuff_VVR(v5_3, v5_2, -4);
          const HVX_VectorPair v4_2 = Q6_W_vshuff_VVR(v5_5, v5_4, -4);
          const HVX_VectorPair v4_3 = Q6_W_vshuff_VVR(v5_7, v5_6, -4);
          const HVX_VectorPair v4_4 = Q6_W_vshuff_VVR(v5_9, v5_8, -4);
          const HVX_VectorPair v4_5 = Q6_W_vshuff_VVR(v5_11, v5_10, -4);
          const HVX_VectorPair v4_6 = Q6_W_vshuff_VVR(v5_13, v5_12, -4);
          const HVX_VectorPair v4_7 = Q6_W_vshuff_VVR(v5_15, v5_14, -4);
          const HVX_VectorPair v4_8 = Q6_W_vshuff_VVR(v5_17, v5_16, -4);
          const HVX_VectorPair v4_9 = Q6_W_vshuff_VVR(v5_19, v5_18, -4);
          const HVX_VectorPair v4_10 = Q6_W_vshuff_VVR(v5_21, v5_20, -4);
          const HVX_VectorPair v4_11 = Q6_W_vshuff_VVR(v5_23, v5_22, -4);
          const HVX_VectorPair v4_12 = Q6_W_vshuff_VVR(v5_25, v5_24, -4);
          const HVX_VectorPair v4_13 = Q6_W_vshuff_VVR(v5_27, v5_26, -4);
          const HVX_VectorPair v4_14 = Q6_W_vshuff_VVR(v5_29, v5_28, -4);
          const HVX_VectorPair v4_15 = Q6_W_vshuff_VVR(v5_31, v5_30, -4);

          const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), -8);
          const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), -8);
          const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), -8);
          const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), -8);
          const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), -8);
          const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), -8);
          const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), -8);
          const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), -8);
          const HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), -8);
          const HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), -8);
          const HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), -8);
          const HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), -8);
          const HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), -8);
          const HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), -8);
          const HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), -8);
          const HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), -8);

          const HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), -16);
          const HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), -16);
          const HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), -16);
          const HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), -16);
          const HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), -16);
          const HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), -16);
          const HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), -16);
          const HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), -16);
          const HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), -16);
          const HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), -16);
          const HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), -16);
          const HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), -16);
          const HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), -16);
          const HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), -16);
          const HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), -16);
          const HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), -16);

          const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), -32);
          const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), -32);
          const HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), -32);
          const HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), -32);
          const HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), -32);
          const HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), -32);
          const HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), -32);
          const HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), -32);
          const HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), -32);
          const HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), -32);
          const HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), -32);
          const HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), -32);
          const HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), -32);
          const HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), -32);
          const HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), -32);
          const HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), -32);

          const HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), -64);
          const HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), -64);
          const HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), -64);
          const HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), -64);
          const HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), -64);
          const HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), -64);
          const HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), -64);
          const HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), -64);
          const HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), -64);
          const HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), -64);
          const HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), -64);
          const HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), -64);
          const HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), -64);
          const HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), -64);
          const HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), -64);
          const HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), -64);

          xnn_store_s32(out + 32, Q6_V_lo_W(v0_0));
          xnn_store_s32(out + 96, Q6_V_hi_W(v0_0));
          xnn_store_s32(out + 160, Q6_V_lo_W(v0_1));
          xnn_store_s32(out + 224, Q6_V_hi_W(v0_1));
          xnn_store_s32(out + 288, Q6_V_lo_W(v0_2));
          xnn_store_s32(out + 352, Q6_V_hi_W(v0_2));
          xnn_store_s32(out + 416, Q6_V_lo_W(v0_3));
          xnn_store_s32(out + 480, Q6_V_hi_W(v0_3));
          xnn_store_s32(out + 544, Q6_V_lo_W(v0_4));
          xnn_store_s32(out + 608, Q6_V_hi_W(v0_4));
          xnn_store_s32(out + 672, Q6_V_lo_W(v0_5));
          xnn_store_s32(out + 736, Q6_V_hi_W(v0_5));
          xnn_store_s32(out + 800, Q6_V_lo_W(v0_6));
          xnn_store_s32(out + 864, Q6_V_hi_W(v0_6));
          xnn_store_s32(out + 928, Q6_V_lo_W(v0_7));
          xnn_store_s32(out + 992, Q6_V_hi_W(v0_7));
          xnn_store_s32(out + 1056, Q6_V_lo_W(v0_8));
          xnn_store_s32(out + 1120, Q6_V_hi_W(v0_8));
          xnn_store_s32(out + 1184, Q6_V_lo_W(v0_9));
          xnn_store_s32(out + 1248, Q6_V_hi_W(v0_9));
          xnn_store_s32(out + 1312, Q6_V_lo_W(v0_10));
          xnn_store_s32(out + 1376, Q6_V_hi_W(v0_10));
          xnn_store_s32(out + 1440, Q6_V_lo_W(v0_11));
          xnn_store_s32(out + 1504, Q6_V_hi_W(v0_11));
          xnn_store_s32(out + 1568, Q6_V_lo_W(v0_12));
          xnn_store_s32(out + 1632, Q6_V_hi_W(v0_12));
          xnn_store_s32(out + 1696, Q6_V_lo_W(v0_13));
          xnn_store_s32(out + 1760, Q6_V_hi_W(v0_13));
          xnn_store_s32(out + 1824, Q6_V_lo_W(v0_14));
          xnn_store_s32(out + 1888, Q6_V_hi_W(v0_14));
          xnn_store_s32(out + 1952, Q6_V_lo_W(v0_15));
          xnn_store_s32(out + 2016, Q6_V_hi_W(v0_15));
        }
        out += 2048;
      }

      // KC remainder (1..31)
      for (; k >= 2; k -= 2) {
        const int32_t v0_0 = w0[0];
        const int32_t v0_1 = w0[1];
        w0 += 2;
        const int32_t v1_0 = w1[0];
        const int32_t v1_1 = w1[1];
        w1 += 2;
        const int32_t v2_0 = w2[0];
        const int32_t v2_1 = w2[1];
        w2 += 2;
        const int32_t v3_0 = w3[0];
        const int32_t v3_1 = w3[1];
        w3 += 2;
        const int32_t v4_0 = w4[0];
        const int32_t v4_1 = w4[1];
        w4 += 2;
        const int32_t v5_0 = w5[0];
        const int32_t v5_1 = w5[1];
        w5 += 2;
        const int32_t v6_0 = w6[0];
        const int32_t v6_1 = w6[1];
        w6 += 2;
        const int32_t v7_0 = w7[0];
        const int32_t v7_1 = w7[1];
        w7 += 2;
        const int32_t v8_0 = w8[0];
        const int32_t v8_1 = w8[1];
        w8 += 2;
        const int32_t v9_0 = w9[0];
        const int32_t v9_1 = w9[1];
        w9 += 2;
        const int32_t v10_0 = w10[0];
        const int32_t v10_1 = w10[1];
        w10 += 2;
        const int32_t v11_0 = w11[0];
        const int32_t v11_1 = w11[1];
        w11 += 2;
        const int32_t v12_0 = w12[0];
        const int32_t v12_1 = w12[1];
        w12 += 2;
        const int32_t v13_0 = w13[0];
        const int32_t v13_1 = w13[1];
        w13 += 2;
        const int32_t v14_0 = w14[0];
        const int32_t v14_1 = w14[1];
        w14 += 2;
        const int32_t v15_0 = w15[0];
        const int32_t v15_1 = w15[1];
        w15 += 2;
        const int32_t v16_0 = w16[0];
        const int32_t v16_1 = w16[1];
        w16 += 2;
        const int32_t v17_0 = w17[0];
        const int32_t v17_1 = w17[1];
        w17 += 2;
        const int32_t v18_0 = w18[0];
        const int32_t v18_1 = w18[1];
        w18 += 2;
        const int32_t v19_0 = w19[0];
        const int32_t v19_1 = w19[1];
        w19 += 2;
        const int32_t v20_0 = w20[0];
        const int32_t v20_1 = w20[1];
        w20 += 2;
        const int32_t v21_0 = w21[0];
        const int32_t v21_1 = w21[1];
        w21 += 2;
        const int32_t v22_0 = w22[0];
        const int32_t v22_1 = w22[1];
        w22 += 2;
        const int32_t v23_0 = w23[0];
        const int32_t v23_1 = w23[1];
        w23 += 2;
        const int32_t v24_0 = w24[0];
        const int32_t v24_1 = w24[1];
        w24 += 2;
        const int32_t v25_0 = w25[0];
        const int32_t v25_1 = w25[1];
        w25 += 2;
        const int32_t v26_0 = w26[0];
        const int32_t v26_1 = w26[1];
        w26 += 2;
        const int32_t v27_0 = w27[0];
        const int32_t v27_1 = w27[1];
        w27 += 2;
        const int32_t v28_0 = w28[0];
        const int32_t v28_1 = w28[1];
        w28 += 2;
        const int32_t v29_0 = w29[0];
        const int32_t v29_1 = w29[1];
        w29 += 2;
        const int32_t v30_0 = w30[0];
        const int32_t v30_1 = w30[1];
        w30 += 2;
        const int32_t v31_0 = w31[0];
        const int32_t v31_1 = w31[1];
        w31 += 2;
        const int32_t v32_0 = w32[0];
        const int32_t v32_1 = w32[1];
        w32 += 2;
        const int32_t v33_0 = w33[0];
        const int32_t v33_1 = w33[1];
        w33 += 2;
        const int32_t v34_0 = w34[0];
        const int32_t v34_1 = w34[1];
        w34 += 2;
        const int32_t v35_0 = w35[0];
        const int32_t v35_1 = w35[1];
        w35 += 2;
        const int32_t v36_0 = w36[0];
        const int32_t v36_1 = w36[1];
        w36 += 2;
        const int32_t v37_0 = w37[0];
        const int32_t v37_1 = w37[1];
        w37 += 2;
        const int32_t v38_0 = w38[0];
        const int32_t v38_1 = w38[1];
        w38 += 2;
        const int32_t v39_0 = w39[0];
        const int32_t v39_1 = w39[1];
        w39 += 2;
        const int32_t v40_0 = w40[0];
        const int32_t v40_1 = w40[1];
        w40 += 2;
        const int32_t v41_0 = w41[0];
        const int32_t v41_1 = w41[1];
        w41 += 2;
        const int32_t v42_0 = w42[0];
        const int32_t v42_1 = w42[1];
        w42 += 2;
        const int32_t v43_0 = w43[0];
        const int32_t v43_1 = w43[1];
        w43 += 2;
        const int32_t v44_0 = w44[0];
        const int32_t v44_1 = w44[1];
        w44 += 2;
        const int32_t v45_0 = w45[0];
        const int32_t v45_1 = w45[1];
        w45 += 2;
        const int32_t v46_0 = w46[0];
        const int32_t v46_1 = w46[1];
        w46 += 2;
        const int32_t v47_0 = w47[0];
        const int32_t v47_1 = w47[1];
        w47 += 2;
        const int32_t v48_0 = w48[0];
        const int32_t v48_1 = w48[1];
        w48 += 2;
        const int32_t v49_0 = w49[0];
        const int32_t v49_1 = w49[1];
        w49 += 2;
        const int32_t v50_0 = w50[0];
        const int32_t v50_1 = w50[1];
        w50 += 2;
        const int32_t v51_0 = w51[0];
        const int32_t v51_1 = w51[1];
        w51 += 2;
        const int32_t v52_0 = w52[0];
        const int32_t v52_1 = w52[1];
        w52 += 2;
        const int32_t v53_0 = w53[0];
        const int32_t v53_1 = w53[1];
        w53 += 2;
        const int32_t v54_0 = w54[0];
        const int32_t v54_1 = w54[1];
        w54 += 2;
        const int32_t v55_0 = w55[0];
        const int32_t v55_1 = w55[1];
        w55 += 2;
        const int32_t v56_0 = w56[0];
        const int32_t v56_1 = w56[1];
        w56 += 2;
        const int32_t v57_0 = w57[0];
        const int32_t v57_1 = w57[1];
        w57 += 2;
        const int32_t v58_0 = w58[0];
        const int32_t v58_1 = w58[1];
        w58 += 2;
        const int32_t v59_0 = w59[0];
        const int32_t v59_1 = w59[1];
        w59 += 2;
        const int32_t v60_0 = w60[0];
        const int32_t v60_1 = w60[1];
        w60 += 2;
        const int32_t v61_0 = w61[0];
        const int32_t v61_1 = w61[1];
        w61 += 2;
        const int32_t v62_0 = w62[0];
        const int32_t v62_1 = w62[1];
        w62 += 2;
        out[0] = v0_0;
        out[64] = v0_1;
        out[1] = v1_0;
        out[65] = v1_1;
        out[2] = v2_0;
        out[66] = v2_1;
        out[3] = v3_0;
        out[67] = v3_1;
        out[4] = v4_0;
        out[68] = v4_1;
        out[5] = v5_0;
        out[69] = v5_1;
        out[6] = v6_0;
        out[70] = v6_1;
        out[7] = v7_0;
        out[71] = v7_1;
        out[8] = v8_0;
        out[72] = v8_1;
        out[9] = v9_0;
        out[73] = v9_1;
        out[10] = v10_0;
        out[74] = v10_1;
        out[11] = v11_0;
        out[75] = v11_1;
        out[12] = v12_0;
        out[76] = v12_1;
        out[13] = v13_0;
        out[77] = v13_1;
        out[14] = v14_0;
        out[78] = v14_1;
        out[15] = v15_0;
        out[79] = v15_1;
        out[16] = v16_0;
        out[80] = v16_1;
        out[17] = v17_0;
        out[81] = v17_1;
        out[18] = v18_0;
        out[82] = v18_1;
        out[19] = v19_0;
        out[83] = v19_1;
        out[20] = v20_0;
        out[84] = v20_1;
        out[21] = v21_0;
        out[85] = v21_1;
        out[22] = v22_0;
        out[86] = v22_1;
        out[23] = v23_0;
        out[87] = v23_1;
        out[24] = v24_0;
        out[88] = v24_1;
        out[25] = v25_0;
        out[89] = v25_1;
        out[26] = v26_0;
        out[90] = v26_1;
        out[27] = v27_0;
        out[91] = v27_1;
        out[28] = v28_0;
        out[92] = v28_1;
        out[29] = v29_0;
        out[93] = v29_1;
        out[30] = v30_0;
        out[94] = v30_1;
        out[31] = v31_0;
        out[95] = v31_1;
        out[32] = v32_0;
        out[96] = v32_1;
        out[33] = v33_0;
        out[97] = v33_1;
        out[34] = v34_0;
        out[98] = v34_1;
        out[35] = v35_0;
        out[99] = v35_1;
        out[36] = v36_0;
        out[100] = v36_1;
        out[37] = v37_0;
        out[101] = v37_1;
        out[38] = v38_0;
        out[102] = v38_1;
        out[39] = v39_0;
        out[103] = v39_1;
        out[40] = v40_0;
        out[104] = v40_1;
        out[41] = v41_0;
        out[105] = v41_1;
        out[42] = v42_0;
        out[106] = v42_1;
        out[43] = v43_0;
        out[107] = v43_1;
        out[44] = v44_0;
        out[108] = v44_1;
        out[45] = v45_0;
        out[109] = v45_1;
        out[46] = v46_0;
        out[110] = v46_1;
        out[47] = v47_0;
        out[111] = v47_1;
        out[48] = v48_0;
        out[112] = v48_1;
        out[49] = v49_0;
        out[113] = v49_1;
        out[50] = v50_0;
        out[114] = v50_1;
        out[51] = v51_0;
        out[115] = v51_1;
        out[52] = v52_0;
        out[116] = v52_1;
        out[53] = v53_0;
        out[117] = v53_1;
        out[54] = v54_0;
        out[118] = v54_1;
        out[55] = v55_0;
        out[119] = v55_1;
        out[56] = v56_0;
        out[120] = v56_1;
        out[57] = v57_0;
        out[121] = v57_1;
        out[58] = v58_0;
        out[122] = v58_1;
        out[59] = v59_0;
        out[123] = v59_1;
        out[60] = v60_0;
        out[124] = v60_1;
        out[61] = v61_0;
        out[125] = v61_1;
        out[62] = v62_0;
        out[126] = v62_1;
        out += 128;
      }
      for (; k != 0; --k) {
        const int32_t v0 = *w0++;
        out[0] = v0;
        const int32_t v1 = *w1++;
        out[1] = v1;
        const int32_t v2 = *w2++;
        out[2] = v2;
        const int32_t v3 = *w3++;
        out[3] = v3;
        const int32_t v4 = *w4++;
        out[4] = v4;
        const int32_t v5 = *w5++;
        out[5] = v5;
        const int32_t v6 = *w6++;
        out[6] = v6;
        const int32_t v7 = *w7++;
        out[7] = v7;
        const int32_t v8 = *w8++;
        out[8] = v8;
        const int32_t v9 = *w9++;
        out[9] = v9;
        const int32_t v10 = *w10++;
        out[10] = v10;
        const int32_t v11 = *w11++;
        out[11] = v11;
        const int32_t v12 = *w12++;
        out[12] = v12;
        const int32_t v13 = *w13++;
        out[13] = v13;
        const int32_t v14 = *w14++;
        out[14] = v14;
        const int32_t v15 = *w15++;
        out[15] = v15;
        const int32_t v16 = *w16++;
        out[16] = v16;
        const int32_t v17 = *w17++;
        out[17] = v17;
        const int32_t v18 = *w18++;
        out[18] = v18;
        const int32_t v19 = *w19++;
        out[19] = v19;
        const int32_t v20 = *w20++;
        out[20] = v20;
        const int32_t v21 = *w21++;
        out[21] = v21;
        const int32_t v22 = *w22++;
        out[22] = v22;
        const int32_t v23 = *w23++;
        out[23] = v23;
        const int32_t v24 = *w24++;
        out[24] = v24;
        const int32_t v25 = *w25++;
        out[25] = v25;
        const int32_t v26 = *w26++;
        out[26] = v26;
        const int32_t v27 = *w27++;
        out[27] = v27;
        const int32_t v28 = *w28++;
        out[28] = v28;
        const int32_t v29 = *w29++;
        out[29] = v29;
        const int32_t v30 = *w30++;
        out[30] = v30;
        const int32_t v31 = *w31++;
        out[31] = v31;
        const int32_t v32 = *w32++;
        out[32] = v32;
        const int32_t v33 = *w33++;
        out[33] = v33;
        const int32_t v34 = *w34++;
        out[34] = v34;
        const int32_t v35 = *w35++;
        out[35] = v35;
        const int32_t v36 = *w36++;
        out[36] = v36;
        const int32_t v37 = *w37++;
        out[37] = v37;
        const int32_t v38 = *w38++;
        out[38] = v38;
        const int32_t v39 = *w39++;
        out[39] = v39;
        const int32_t v40 = *w40++;
        out[40] = v40;
        const int32_t v41 = *w41++;
        out[41] = v41;
        const int32_t v42 = *w42++;
        out[42] = v42;
        const int32_t v43 = *w43++;
        out[43] = v43;
        const int32_t v44 = *w44++;
        out[44] = v44;
        const int32_t v45 = *w45++;
        out[45] = v45;
        const int32_t v46 = *w46++;
        out[46] = v46;
        const int32_t v47 = *w47++;
        out[47] = v47;
        const int32_t v48 = *w48++;
        out[48] = v48;
        const int32_t v49 = *w49++;
        out[49] = v49;
        const int32_t v50 = *w50++;
        out[50] = v50;
        const int32_t v51 = *w51++;
        out[51] = v51;
        const int32_t v52 = *w52++;
        out[52] = v52;
        const int32_t v53 = *w53++;
        out[53] = v53;
        const int32_t v54 = *w54++;
        out[54] = v54;
        const int32_t v55 = *w55++;
        out[55] = v55;
        const int32_t v56 = *w56++;
        out[56] = v56;
        const int32_t v57 = *w57++;
        out[57] = v57;
        const int32_t v58 = *w58++;
        out[58] = v58;
        const int32_t v59 = *w59++;
        out[59] = v59;
        const int32_t v60 = *w60++;
        out[60] = v60;
        const int32_t v61 = *w61++;
        out[61] = v61;
        const int32_t v62 = *w62++;
        out[62] = v62;
        out += 64;
      }
      out = (int32_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * n_stride;
  } while (--g != 0);
}
