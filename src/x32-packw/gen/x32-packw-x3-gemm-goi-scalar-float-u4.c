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



void xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_u4(
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
  assert(nr == 3);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  float* out = (float*) packed_weights;
  const float* b = (const float*) bias;

  do {
    // NC main loop multiple of 3
    const float* w0 = (const float*) weights;
    size_t n = nc;
    for (;n >= 3; n -= 3) {
      if XNN_LIKELY(b != NULL) {
        out[0] = b[0];
        out[1] = b[1];
        out[2] = b[2];
        b += 3;
      } else {
        out[0] = 0;
        out[1] = 0;
        out[2] = 0;
      }
      out += 3;

      const float* w1 = w0 + kc;
      const float* w2 = w1 + kc;

      // KC main loop multiple of 3x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const float v00 = w0[0];
        const float v01 = w0[1];
        const float v02 = w0[2];
        const float v03 = w0[3];
        w0 += 4;
        const float v10 = w1[0];
        const float v11 = w1[1];
        const float v12 = w1[2];
        const float v13 = w1[3];
        w1 += 4;
        const float v20 = w2[0];
        const float v21 = w2[1];
        const float v22 = w2[2];
        const float v23 = w2[3];
        w2 += 4;
        out[0] = v00;
        out[1] = v10;
        out[2] = v20;
        out[3] = v01;
        out[4] = v11;
        out[5] = v21;
        out[6] = v02;
        out[7] = v12;
        out[8] = v22;
        out[9] = v03;
        out[10] = v13;
        out[11] = v23;
        out += 12;
      }

      // KC remainder
      for (; k != 0; --k) {
        const float v0 = *w0++;
        out[0] = v0;
        const float v1 = *w1++;
        out[1] = v1;
        const float v2 = *w2++;
        out[2] = v2;
        out += 3;
      }
      out = (float*) ((uintptr_t) out + extra_bytes);
      w0 = w2;
    }

    // NC remainder (1..2)
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
      out += (3 - n);

      // NR remainder has less than 3 rows so last row is not loaded
      const float* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }

      // KC main loop multiple of 3x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const float v00 = w0[0];
        const float v01 = w0[1];
        const float v02 = w0[2];
        const float v03 = w0[3];
        w0 += 4;
        const float v10 = w1[0];
        const float v11 = w1[1];
        const float v12 = w1[2];
        const float v13 = w1[3];
        w1 += 4;
        out[0] = v00;
        out[1] = v10;
        out[3] = v01;
        out[4] = v11;
        out[6] = v02;
        out[7] = v12;
        out[9] = v03;
        out[10] = v13;
        out += 12;
      }

      // KC remainder of 1..3
      for (; k != 0; --k) {
        const float v0 = *w0++;
        out[0] = v0;
        const float v1 = *w1++;
        out[1] = v1;
        out += 3;
      }
      out = (float*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
