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



void xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_x4(
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
  assert(nr == 4);   // This kernel is for NR=4
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = (uint32_t) ((const struct xnn_qs8_packw_params*) params)->input_zero_point;

  do {
    // NC main loop multiple of 4
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 4; n -= 4) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        ((int32_t*) out)[0] = b[0];
        ((int32_t*) out)[1] = b[1];
        ((int32_t*) out)[2] = b[2];
        ((int32_t*) out)[3] = b[3];
        b += 4;
      } else {
        ((int32_t*) out)[0] = 0;
        ((int32_t*) out)[1] = 0;
        ((int32_t*) out)[2] = 0;
        ((int32_t*) out)[3] = 0;
      }
      out += 4 * sizeof(int32_t);

      const int8_t* w1 = w0 + kc;
      const int8_t* w2 = w1 + kc;
      const int8_t* w3 = w2 + kc;
      uint32_t ksum0 = 0;
      uint32_t ksum1 = 0;
      uint32_t ksum2 = 0;
      uint32_t ksum3 = 0;

      // KC main loop multiple of 4x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const int8_t v00 = w0[0];
        ksum0 += (uint32_t) v00;
        const int8_t v01 = w0[1];
        ksum0 += (uint32_t) v01;
        const int8_t v02 = w0[2];
        ksum0 += (uint32_t) v02;
        const int8_t v03 = w0[3];
        ksum0 += (uint32_t) v03;
        w0 += 4;
        const int8_t v10 = w1[0];
        ksum1 += (uint32_t) v10;
        const int8_t v11 = w1[1];
        ksum1 += (uint32_t) v11;
        const int8_t v12 = w1[2];
        ksum1 += (uint32_t) v12;
        const int8_t v13 = w1[3];
        ksum1 += (uint32_t) v13;
        w1 += 4;
        const int8_t v20 = w2[0];
        ksum2 += (uint32_t) v20;
        const int8_t v21 = w2[1];
        ksum2 += (uint32_t) v21;
        const int8_t v22 = w2[2];
        ksum2 += (uint32_t) v22;
        const int8_t v23 = w2[3];
        ksum2 += (uint32_t) v23;
        w2 += 4;
        const int8_t v30 = w3[0];
        ksum3 += (uint32_t) v30;
        const int8_t v31 = w3[1];
        ksum3 += (uint32_t) v31;
        const int8_t v32 = w3[2];
        ksum3 += (uint32_t) v32;
        const int8_t v33 = w3[3];
        ksum3 += (uint32_t) v33;
        w3 += 4;
        out[0] = v00;
        out[1] = v10;
        out[2] = v20;
        out[3] = v30;
        out[4] = v01;
        out[5] = v11;
        out[6] = v21;
        out[7] = v31;
        out[8] = v02;
        out[9] = v12;
        out[10] = v22;
        out[11] = v32;
        out[12] = v03;
        out[13] = v13;
        out[14] = v23;
        out[15] = v33;
        out += 16;
      }

      // KC remainder
      for (; k != 0; --k) {
        const int8_t v0 = *w0++;
        ksum0 += (uint32_t) v0;
        out[0] = v0;
        const int8_t v1 = *w1++;
        ksum1 += (uint32_t) v1;
        out[1] = v1;
        const int8_t v2 = *w2++;
        ksum2 += (uint32_t) v2;
        out[2] = v2;
        const int8_t v3 = *w3++;
        ksum3 += (uint32_t) v3;
        out[3] = v3;
        out += 4;
      }
      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      packed_b[2] -= ksum2 * izp;
      packed_b[3] -= ksum3 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w3;
    }

    // NC remainder (1..3)
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
      out += (4 - n) * sizeof(int32_t);

      // NR remainder has less than 4 rows so last row is not loaded
      const int8_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      uint32_t ksum0 = 0;
      uint32_t ksum1 = 0;
      uint32_t ksum2 = 0;

      // KC main loop multiple of 4x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const int8_t v00 = w0[0];
        ksum0 += (uint32_t) v00;
        const int8_t v01 = w0[1];
        ksum0 += (uint32_t) v01;
        const int8_t v02 = w0[2];
        ksum0 += (uint32_t) v02;
        const int8_t v03 = w0[3];
        ksum0 += (uint32_t) v03;
        w0 += 4;
        const int8_t v10 = w1[0];
        ksum1 += (uint32_t) v10;
        const int8_t v11 = w1[1];
        ksum1 += (uint32_t) v11;
        const int8_t v12 = w1[2];
        ksum1 += (uint32_t) v12;
        const int8_t v13 = w1[3];
        ksum1 += (uint32_t) v13;
        w1 += 4;
        const int8_t v20 = w2[0];
        ksum2 += (uint32_t) v20;
        const int8_t v21 = w2[1];
        ksum2 += (uint32_t) v21;
        const int8_t v22 = w2[2];
        ksum2 += (uint32_t) v22;
        const int8_t v23 = w2[3];
        ksum2 += (uint32_t) v23;
        w2 += 4;
        out[0] = v00;
        out[1] = v10;
        out[2] = v20;
        out[4] = v01;
        out[5] = v11;
        out[6] = v21;
        out[8] = v02;
        out[9] = v12;
        out[10] = v22;
        out[12] = v03;
        out[13] = v13;
        out[14] = v23;
        out += 16;
      }

      // KC remainder of 1..3
      for (; k != 0; --k) {
        const int8_t v0 = *w0++;
        ksum0 += (uint32_t) v0;
        out[0] = v0;
        const int8_t v1 = *w1++;
        ksum1 += (uint32_t) v1;
        out[1] = v1;
        const int8_t v2 = *w2++;
        ksum2 += (uint32_t) v2;
        out[2] = v2;
        out += 4;
      }
      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      packed_b[2] -= ksum2 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
