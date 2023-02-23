// Auto-generated file. Do not edit!
//   Template: src/x32-packw/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/packw.h>

#include <arm_neon.h>


void xnn_x32_packw_gemm_goi_ukernel_x12__neon(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 12);   // This kernel is for NR=12
  assert(kr == 1);
  assert(sr == 1);
  assert(nr >= sr);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint32x4x4_t v00;
  v00.val[0] = vdupq_n_u32(0);
  v00.val[1] = vdupq_n_u32(0);
  v00.val[2] = vdupq_n_u32(0);
  v00.val[3] = vdupq_n_u32(0);
  uint32x4x4_t v40;
  v40.val[0] = vdupq_n_u32(0);
  v40.val[1] = vdupq_n_u32(0);
  v40.val[2] = vdupq_n_u32(0);
  v40.val[3] = vdupq_n_u32(0);
  uint32x4x4_t v80;
  v80.val[0] = vdupq_n_u32(0);
  v80.val[1] = vdupq_n_u32(0);
  v80.val[2] = vdupq_n_u32(0);
  v80.val[3] = vdupq_n_u32(0);

  do {
    // NC main loop multiple of 12
    const uint32_t* w = weights;
    size_t n = nc;
    if XNN_LIKELY(n >= 12) {
      do {
        if XNN_LIKELY(bias != NULL) {
          uint32x4_t vb0 = vld1q_u32(bias + 0);
          uint32x4_t vb4 = vld1q_u32(bias + 4);
          uint32x4_t vb8 = vld1q_u32(bias + 8);
          vst1q_u32(packed_weights + 0, vb0);
          vst1q_u32(packed_weights + 4, vb4);
          vst1q_u32(packed_weights + 8, vb8);
          bias += 12;
        }
        packed_weights += 12;

        const uint32_t* w0 = w;
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

        // KC main loop multiple of 12x4
        size_t k = kc;
        for (; k >= 4; k -= 4) {
          v00 = vld4q_lane_u32(w0 + 0, v00, 0);
          w0 += 4;
          v00 = vld4q_lane_u32(w1 + 0, v00, 1);
          w1 += 4;
          v00 = vld4q_lane_u32(w2 + 0, v00, 2);
          w2 += 4;
          v00 = vld4q_lane_u32(w3 + 0, v00, 3);
          w3 += 4;
          v40 = vld4q_lane_u32(w4 + 0, v40, 0);
          w4 += 4;
          v40 = vld4q_lane_u32(w5 + 0, v40, 1);
          w5 += 4;
          v40 = vld4q_lane_u32(w6 + 0, v40, 2);
          w6 += 4;
          v40 = vld4q_lane_u32(w7 + 0, v40, 3);
          w7 += 4;
          v80 = vld4q_lane_u32(w8 + 0, v80, 0);
          w8 += 4;
          v80 = vld4q_lane_u32(w9 + 0, v80, 1);
          w9 += 4;
          v80 = vld4q_lane_u32(w10 + 0, v80, 2);
          w10 += 4;
          v80 = vld4q_lane_u32(w11 + 0, v80, 3);
          w11 += 4;
          vst1q_u32(packed_weights + 0, v00.val[0]);
          vst1q_u32(packed_weights + 4, v40.val[0]);
          vst1q_u32(packed_weights + 8, v80.val[0]);
          vst1q_u32(packed_weights + 12, v00.val[1]);
          vst1q_u32(packed_weights + 16, v40.val[1]);
          vst1q_u32(packed_weights + 20, v80.val[1]);
          vst1q_u32(packed_weights + 24, v00.val[2]);
          vst1q_u32(packed_weights + 28, v40.val[2]);
          vst1q_u32(packed_weights + 32, v80.val[2]);
          vst1q_u32(packed_weights + 36, v00.val[3]);
          vst1q_u32(packed_weights + 40, v40.val[3]);
          vst1q_u32(packed_weights + 44, v80.val[3]);
          packed_weights += 48;
        }

        // KC remainder
        for (; k >= 1; --k) {
          v00.val[0] = vld1q_lane_u32(w0 + 0, v00.val[0], 0);
          w0 += 1;
          v00.val[0] = vld1q_lane_u32(w1 + 0, v00.val[0], 1);
          w1 += 1;
          v00.val[0] = vld1q_lane_u32(w2 + 0, v00.val[0], 2);
          w2 += 1;
          v00.val[0] = vld1q_lane_u32(w3 + 0, v00.val[0], 3);
          w3 += 1;
          v40.val[0] = vld1q_lane_u32(w4 + 0, v40.val[0], 0);
          w4 += 1;
          v40.val[0] = vld1q_lane_u32(w5 + 0, v40.val[0], 1);
          w5 += 1;
          v40.val[0] = vld1q_lane_u32(w6 + 0, v40.val[0], 2);
          w6 += 1;
          v40.val[0] = vld1q_lane_u32(w7 + 0, v40.val[0], 3);
          w7 += 1;
          v80.val[0] = vld1q_lane_u32(w8 + 0, v80.val[0], 0);
          w8 += 1;
          v80.val[0] = vld1q_lane_u32(w9 + 0, v80.val[0], 1);
          w9 += 1;
          v80.val[0] = vld1q_lane_u32(w10 + 0, v80.val[0], 2);
          w10 += 1;
          v80.val[0] = vld1q_lane_u32(w11 + 0, v80.val[0], 3);
          w11 += 1;
          vst1q_u32(packed_weights + 0, v00.val[0]);
          vst1q_u32(packed_weights + 4, v40.val[0]);
          vst1q_u32(packed_weights + 8, v80.val[0]);
          packed_weights += 12;
        }
        packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
        w = w11;
        n -= 12;
      } while (n >= 12);
    }

    if XNN_UNLIKELY(n != 0) {
      // NC remainder (1..11)
      if XNN_LIKELY(bias != NULL) {
        size_t nb = n;
        do {
          *packed_weights++  = *bias++;
        } while (--nb != 0);
        packed_weights += (12 - n);
      } else {
        packed_weights += 12;
      }

      size_t k = kc;
      do {
        const uint32_t* wn = w;
        size_t nw = n;
        do {
          *packed_weights++ = wn[0];
          wn += kc;
        } while (--nw != 0);
        ++w;
        packed_weights += (12 - n);
      } while (--k);
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
