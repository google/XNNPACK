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

#include <xnnpack/math.h>
#include <xnnpack/packw.h>



void xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_x2(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* weights,
  const int32_t* bias,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 2);   // This kernel is for NR=2
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = (uint32_t) ((const struct xnn_qs8_packw_params*) params)->input_zero_point;

  do {
    // NC main loop multiple of 2
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 2; n -= 2) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        ((int32_t*) out)[0] = b[0];
        ((int32_t*) out)[1] = b[1];
        b += 2;
      } else {
        ((int32_t*) out)[0] = 0;
        ((int32_t*) out)[1] = 0;
      }
      out += 2 * sizeof(int32_t);

      const int8_t* w1 = w0 + kc;
      uint32_t ksum0 = 0;
      uint32_t ksum1 = 0;

      // KC main loop multiple of 2x2
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const int8_t v00 = w0[0];
        ksum0 += (uint32_t) v00;
        const int8_t v01 = w0[1];
        ksum0 += (uint32_t) v01;
        w0 += 2;
        const int8_t v10 = w1[0];
        ksum1 += (uint32_t) v10;
        const int8_t v11 = w1[1];
        ksum1 += (uint32_t) v11;
        w1 += 2;
        out[0] = v00;
        out[1] = v10;
        out[2] = v01;
        out[3] = v11;
        out += 4;
      }

      // KC remainder
      for (; k != 0; --k) {
        const int8_t v0 = *w0++;
        ksum0 += (uint32_t) v0;
        out[0] = v0;
        const int8_t v1 = *w1++;
        ksum1 += (uint32_t) v1;
        out[1] = v1;
        out += 2;
      }
      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w1;
    }

    // NC remainder (1..1)
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
      out += (2 - n) * sizeof(int32_t);

      uint32_t ksum0 = 0;

      // KC main loop multiple of 2x2
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const int8_t v00 = w0[0];
        ksum0 += (uint32_t) v00;
        const int8_t v01 = w0[1];
        ksum0 += (uint32_t) v01;
        w0 += 2;
        out[0] = v00;
        out[2] = v01;
        out += 4;
      }

      // KC remainder of 1..1
      for (; k != 0; --k) {
        const int8_t v0 = *w0++;
        ksum0 += (uint32_t) v0;
        out[0] = v0;
        out += 2;
      }
      packed_b[0] -= ksum0 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
