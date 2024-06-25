// Auto-generated file. Do not edit!
//   Template: src/x32-packw/NR2-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include "xnnpack/packw.h"


void xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2(
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
  assert(nr == 2);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint32x2x2_t v00;
  v00.val[0] = vdup_n_u32(0);
  v00.val[1] = vdup_n_u32(0);

  do {
    // NC main loop multiple of 2
    const uint32_t* w0 = weights;
    size_t n = nc;

    for (; n >= 2; n -= 2) {
      if XNN_LIKELY(bias != NULL) {
        uint32x2_t vb0 = vld1_u32(bias); bias += 2;
        vst1_u32(packed_weights, vb0); packed_weights += 2;
      } else {
        const uint32x2_t vzero = vmov_n_u32(0);
        vst1_u32(packed_weights, vzero); packed_weights += 2;
      }

      const uint32_t* w1 = w0 + kc;

      // KC main loop multiple of 2
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        v00 = vld2_lane_u32(w0, v00, 0); w0 += 2;
        v00 = vld2_lane_u32(w1, v00, 1); w1 += 2;
        vst1_u32(packed_weights + 0, v00.val[0]);
        vst1_u32(packed_weights + 2, v00.val[1]);
        packed_weights += 4;
      }

      // KC remainder
      for (; k != 0; --k) {
        v00.val[0] = vld1_lane_u32(w0, v00.val[0], 0);  w0 += 1;
        v00.val[0] = vld1_lane_u32(w1, v00.val[0], 1);  w1 += 1;
        vst1_u32(packed_weights + 0, v00.val[0]);
        packed_weights += 2;
      }
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
      w0 = w1;
    }

    if XNN_UNLIKELY(n != 0) {
      // NC remainder of 1
      if XNN_LIKELY(bias != NULL) {
        *packed_weights = *bias++;
      } else {
        const uint32x2_t vzero = vmov_n_u32(0);
        vst1_u32(packed_weights + 0, vzero);
      }
      packed_weights += 2;
      size_t k = kc;
      do {
        *packed_weights = *w0++;
        packed_weights += 2;
      } while (--k);
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
