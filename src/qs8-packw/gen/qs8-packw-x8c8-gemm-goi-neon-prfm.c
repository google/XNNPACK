// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/c8-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/prefetch.h"


// Load 'n' bytes from 'address' into a uint64_t, padding the remaining bytes
// with 'pad_value'.
XNN_INLINE static uint64_t safe_load_u64(const void* address, size_t n, uint8_t pad_value) {
  uint64_t value = (uint64_t) pad_value * 0x0101010101010101ULL;
  assert(n <= sizeof(uint64_t));
  const uint8_t* bytes = (const uint8_t*) address;
  for (size_t i = 0; i < n; ++i) {
    ((uint8_t*) &value)[i] = bytes[i];
  }
  return value;
}

// Pairwise add of 2 vectors of 32 bit sums laid out as [a, a, b, b].
// Returns [a0 + a1, b0 + b1, a2 + a3, b2 + b3] of both inputs.
XNN_INLINE static int32x4_t xnn_padd_s32(int32x4_t va, int32x4_t vb) {
#if XNN_ARCH_ARM64
  return vpaddq_s32(va, vb);
#else
  return vcombine_s32(vpadd_s32(vget_low_s32(va), vget_high_s32(va)),
                      vpadd_s32(vget_low_s32(vb), vget_high_s32(vb)));
#endif
}

// Transpose 64 bit elements of 2 rows.
// Returns [a0, b0] and [a1, b1]
XNN_INLINE static int8x16x2_t xnn_zip_s64(int8x16_t va, int8x16_t vb) {
  int8x16x2_t vr;
#if XNN_ARCH_ARM64
  vr.val[0] = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(va), vreinterpretq_s64_s8(vb)));
  vr.val[1] = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(va), vreinterpretq_s64_s8(vb)));
#else
  vr.val[0] = vcombine_s8(vget_low_s8(va), vget_low_s8(vb));
  vr.val[1] = vcombine_s8(vget_high_s8(va), vget_high_s8(vb));
#endif
  return vr;
}

// Accumulate the sum of int8 weights of 2 rows.  Each input vector holds 8
// bytes of row a followed by 8 bytes of row b.  Accumulator is [a, a, b, b].
XNN_INLINE static int32x4_t xnn_ksum2_s8(int32x4_t vacc, int8x16_t v0, int8x16_t v1) {
  return vpadalq_s16(vacc, vpadalq_s8(vpaddlq_s8(v0), v1));
}

XNN_INLINE static int32x4_t xnn_ksum_s8(int32x4_t vacc, int8x16_t v) {
  return vpadalq_s16(vacc, vpaddlq_s8(v));
}

void xnn_qs8_packw_gemm_goi_ukernel_x8c8__neon_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t n_stride,
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
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const size_t w_stride = n_stride;
  const int32x4_t vzeropoint = vdupq_n_s32((int32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 0) : 0));

  uint8_t* out = (uint8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;

  do {
    const int8_t* wb = (const int8_t*) weights;
    size_t n = nc;
    // NC main loop multiple of 8
    for (; n >= 8; n -= 8) {
      const int8_t* w0 = wb;
      const int8_t* w1 = w0 + w_stride;
      const int8_t* w2 = w1 + w_stride;
      const int8_t* w3 = w2 + w_stride;
      const int8_t* w4 = w3 + w_stride;
      const int8_t* w5 = w4 + w_stride;
      const int8_t* w6 = w5 + w_stride;
      const int8_t* w7 = w6 + w_stride;

      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        const int32x4_t vb0 = vld1q_s32(b + 0);
        const int32x4_t vb4 = vld1q_s32(b + 4);
        vst1q_s32(packed_b + 0, vb0);
        vst1q_s32(packed_b + 4, vb4);
        b += 8;
      } else {
        vst1q_s32(packed_b + 0, vdupq_n_s32(0));
        vst1q_s32(packed_b + 4, vdupq_n_s32(0));
      }
      out += 8 * sizeof(int32_t);

      xnn_prefetch_to_l1((const int8_t*) w0 + 0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) w0 + 128);
      xnn_prefetch_to_l1((const int8_t*) w0 + 192);
      xnn_prefetch_to_l1((const int8_t*) w1 + 0);
      xnn_prefetch_to_l1((const int8_t*) w1 + 64);
      xnn_prefetch_to_l1((const int8_t*) w1 + 128);
      xnn_prefetch_to_l1((const int8_t*) w1 + 192);
      xnn_prefetch_to_l1((const int8_t*) w2 + 0);
      xnn_prefetch_to_l1((const int8_t*) w2 + 64);
      xnn_prefetch_to_l1((const int8_t*) w2 + 128);
      xnn_prefetch_to_l1((const int8_t*) w2 + 192);
      xnn_prefetch_to_l1((const int8_t*) w3 + 0);
      xnn_prefetch_to_l1((const int8_t*) w3 + 64);
      xnn_prefetch_to_l1((const int8_t*) w3 + 128);
      xnn_prefetch_to_l1((const int8_t*) w3 + 192);
      xnn_prefetch_to_l1((const int8_t*) w4 + 0);
      xnn_prefetch_to_l1((const int8_t*) w4 + 64);
      xnn_prefetch_to_l1((const int8_t*) w4 + 128);
      xnn_prefetch_to_l1((const int8_t*) w4 + 192);
      xnn_prefetch_to_l1((const int8_t*) w5 + 0);
      xnn_prefetch_to_l1((const int8_t*) w5 + 64);
      xnn_prefetch_to_l1((const int8_t*) w5 + 128);
      xnn_prefetch_to_l1((const int8_t*) w5 + 192);
      xnn_prefetch_to_l1((const int8_t*) w6 + 0);
      xnn_prefetch_to_l1((const int8_t*) w6 + 64);
      xnn_prefetch_to_l1((const int8_t*) w6 + 128);
      xnn_prefetch_to_l1((const int8_t*) w6 + 192);
      xnn_prefetch_to_l1((const int8_t*) w7 + 0);
      xnn_prefetch_to_l1((const int8_t*) w7 + 64);
      xnn_prefetch_to_l1((const int8_t*) w7 + 128);
      xnn_prefetch_to_l1((const int8_t*) w7 + 192);

      size_t k = kc;
      // ksum of rows [a, a, b, b]
      int32x4_t vacc0 = vdupq_n_s32(0);
      int32x4_t vacc2 = vdupq_n_s32(0);
      int32x4_t vacc4 = vdupq_n_s32(0);
      int32x4_t vacc6 = vdupq_n_s32(0);

      // KC main loop multiple of 8x16
      for (; k >= 16; k -= 16) {
        const int8x16_t v0 = vld1q_s8(w0);
        const int8x16_t v1 = vld1q_s8(w1);
        const int8x16x2_t vp0 = xnn_zip_s64(v0, v1);
        vacc0 = xnn_ksum2_s8(vacc0, vp0.val[0], vp0.val[1]);
        vst1q_s8((int8_t*) out + 0, vp0.val[0]);
        vst1q_s8((int8_t*) out + 64, vp0.val[1]);
        const int8x16_t v2 = vld1q_s8(w2);
        const int8x16_t v3 = vld1q_s8(w3);
        const int8x16x2_t vp2 = xnn_zip_s64(v2, v3);
        vacc2 = xnn_ksum2_s8(vacc2, vp2.val[0], vp2.val[1]);
        vst1q_s8((int8_t*) out + 16, vp2.val[0]);
        vst1q_s8((int8_t*) out + 80, vp2.val[1]);
        const int8x16_t v4 = vld1q_s8(w4);
        const int8x16_t v5 = vld1q_s8(w5);
        const int8x16x2_t vp4 = xnn_zip_s64(v4, v5);
        vacc4 = xnn_ksum2_s8(vacc4, vp4.val[0], vp4.val[1]);
        vst1q_s8((int8_t*) out + 32, vp4.val[0]);
        vst1q_s8((int8_t*) out + 96, vp4.val[1]);
        const int8x16_t v6 = vld1q_s8(w6);
        const int8x16_t v7 = vld1q_s8(w7);
        const int8x16x2_t vp6 = xnn_zip_s64(v6, v7);
        vacc6 = xnn_ksum2_s8(vacc6, vp6.val[0], vp6.val[1]);
        vst1q_s8((int8_t*) out + 48, vp6.val[0]);
        vst1q_s8((int8_t*) out + 112, vp6.val[1]);
        xnn_prefetch_to_l1((const int8_t*) w0 + 256);
        xnn_prefetch_to_l1((const int8_t*) w1 + 256);
        xnn_prefetch_to_l1((const int8_t*) w2 + 256);
        xnn_prefetch_to_l1((const int8_t*) w3 + 256);
        xnn_prefetch_to_l1((const int8_t*) w4 + 256);
        xnn_prefetch_to_l1((const int8_t*) w5 + 256);
        xnn_prefetch_to_l1((const int8_t*) w6 + 256);
        xnn_prefetch_to_l1((const int8_t*) w7 + 256);

        w0 += 16;
        w1 += 16;
        w2 += 16;
        w3 += 16;
        w4 += 16;
        w5 += 16;
        w6 += 16;
        w7 += 16;
        out += 128;
      }

      // KC remainder of 8
      if (k >= 8) {
        const int8x16_t vp0 = vcombine_s8(vld1_s8(w0), vld1_s8(w1));
        vacc0 = xnn_ksum_s8(vacc0, vp0);
        vst1q_s8((int8_t*) out + 0, vp0);
        const int8x16_t vp2 = vcombine_s8(vld1_s8(w2), vld1_s8(w3));
        vacc2 = xnn_ksum_s8(vacc2, vp2);
        vst1q_s8((int8_t*) out + 16, vp2);
        const int8x16_t vp4 = vcombine_s8(vld1_s8(w4), vld1_s8(w5));
        vacc4 = xnn_ksum_s8(vacc4, vp4);
        vst1q_s8((int8_t*) out + 32, vp4);
        const int8x16_t vp6 = vcombine_s8(vld1_s8(w6), vld1_s8(w7));
        vacc6 = xnn_ksum_s8(vacc6, vp6);
        vst1q_s8((int8_t*) out + 48, vp6);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        out += 64;
        k -= 8;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);
        const int8x16_t vp0 = vcombine_s8(vcreate_s8(safe_load_u64(w0, k, 0)), vcreate_s8(safe_load_u64(w1, k, 0)));
        vacc0 = xnn_ksum_s8(vacc0, vp0);
        vst1q_s8((int8_t*) out + 0, vp0);
        const int8x16_t vp2 = vcombine_s8(vcreate_s8(safe_load_u64(w2, k, 0)), vcreate_s8(safe_load_u64(w3, k, 0)));
        vacc2 = xnn_ksum_s8(vacc2, vp2);
        vst1q_s8((int8_t*) out + 16, vp2);
        const int8x16_t vp4 = vcombine_s8(vcreate_s8(safe_load_u64(w4, k, 0)), vcreate_s8(safe_load_u64(w5, k, 0)));
        vacc4 = xnn_ksum_s8(vacc4, vp4);
        vst1q_s8((int8_t*) out + 32, vp4);
        const int8x16_t vp6 = vcombine_s8(vcreate_s8(safe_load_u64(w6, k, 0)), vcreate_s8(safe_load_u64(w7, k, 0)));
        vacc6 = xnn_ksum_s8(vacc6, vp6);
        vst1q_s8((int8_t*) out + 48, vp6);

        out += 64;
      }

      // Subtract ksum * input_zero_point from bias
      const int32x4_t vksum0 = xnn_padd_s32(vacc0, vacc2);
      const int32x4_t vksum4 = xnn_padd_s32(vacc4, vacc6);
      vst1q_s32(packed_b + 0, vmlsq_s32(vld1q_s32(packed_b + 0), vksum0, vzeropoint));
      vst1q_s32(packed_b + 4, vmlsq_s32(vld1q_s32(packed_b + 4), vksum4, vzeropoint));

      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
      wb += 8 * w_stride;
    }
    // NC remainder (1..7)
    // Same as main loop except bias is copied and w pointers are clamped
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 7);
      const int8_t* w0 = wb;
      const int8_t* w1 = w0 + w_stride;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + w_stride;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int8_t* w3 = w2 + w_stride;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int8_t* w4 = w3 + w_stride;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int8_t* w5 = w4 + w_stride;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int8_t* w6 = w5 + w_stride;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int8_t* w7 = w6 + w_stride;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }

      int32_t* packed_b = (int32_t*) out;
      vst1q_s32(packed_b + 0, vdupq_n_s32(0));
      vst1q_s32(packed_b + 4, vdupq_n_s32(0));
      if XNN_LIKELY(b != NULL) {
        for (size_t nb = 0; nb < n; ++nb) {
          packed_b[nb] = b[nb];
        }
        b += n;
      }
      out += 8 * sizeof(int32_t);

      xnn_prefetch_to_l1((const int8_t*) w0 + 0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) w0 + 128);
      xnn_prefetch_to_l1((const int8_t*) w0 + 192);
      xnn_prefetch_to_l1((const int8_t*) w1 + 0);
      xnn_prefetch_to_l1((const int8_t*) w1 + 64);
      xnn_prefetch_to_l1((const int8_t*) w1 + 128);
      xnn_prefetch_to_l1((const int8_t*) w1 + 192);
      xnn_prefetch_to_l1((const int8_t*) w2 + 0);
      xnn_prefetch_to_l1((const int8_t*) w2 + 64);
      xnn_prefetch_to_l1((const int8_t*) w2 + 128);
      xnn_prefetch_to_l1((const int8_t*) w2 + 192);
      xnn_prefetch_to_l1((const int8_t*) w3 + 0);
      xnn_prefetch_to_l1((const int8_t*) w3 + 64);
      xnn_prefetch_to_l1((const int8_t*) w3 + 128);
      xnn_prefetch_to_l1((const int8_t*) w3 + 192);
      xnn_prefetch_to_l1((const int8_t*) w4 + 0);
      xnn_prefetch_to_l1((const int8_t*) w4 + 64);
      xnn_prefetch_to_l1((const int8_t*) w4 + 128);
      xnn_prefetch_to_l1((const int8_t*) w4 + 192);
      xnn_prefetch_to_l1((const int8_t*) w5 + 0);
      xnn_prefetch_to_l1((const int8_t*) w5 + 64);
      xnn_prefetch_to_l1((const int8_t*) w5 + 128);
      xnn_prefetch_to_l1((const int8_t*) w5 + 192);
      xnn_prefetch_to_l1((const int8_t*) w6 + 0);
      xnn_prefetch_to_l1((const int8_t*) w6 + 64);
      xnn_prefetch_to_l1((const int8_t*) w6 + 128);
      xnn_prefetch_to_l1((const int8_t*) w6 + 192);
      xnn_prefetch_to_l1((const int8_t*) w7 + 0);
      xnn_prefetch_to_l1((const int8_t*) w7 + 64);
      xnn_prefetch_to_l1((const int8_t*) w7 + 128);
      xnn_prefetch_to_l1((const int8_t*) w7 + 192);

      size_t k = kc;
      // ksum of rows [a, a, b, b]
      int32x4_t vacc0 = vdupq_n_s32(0);
      int32x4_t vacc2 = vdupq_n_s32(0);
      int32x4_t vacc4 = vdupq_n_s32(0);
      int32x4_t vacc6 = vdupq_n_s32(0);

      // KC main loop multiple of 8x16
      for (; k >= 16; k -= 16) {
        const int8x16_t v0 = vld1q_s8(w0);
        const int8x16_t v1 = vld1q_s8(w1);
        const int8x16x2_t vp0 = xnn_zip_s64(v0, v1);
        vacc0 = xnn_ksum2_s8(vacc0, vp0.val[0], vp0.val[1]);
        vst1q_s8((int8_t*) out + 0, vp0.val[0]);
        vst1q_s8((int8_t*) out + 64, vp0.val[1]);
        const int8x16_t v2 = vld1q_s8(w2);
        const int8x16_t v3 = vld1q_s8(w3);
        const int8x16x2_t vp2 = xnn_zip_s64(v2, v3);
        vacc2 = xnn_ksum2_s8(vacc2, vp2.val[0], vp2.val[1]);
        vst1q_s8((int8_t*) out + 16, vp2.val[0]);
        vst1q_s8((int8_t*) out + 80, vp2.val[1]);
        const int8x16_t v4 = vld1q_s8(w4);
        const int8x16_t v5 = vld1q_s8(w5);
        const int8x16x2_t vp4 = xnn_zip_s64(v4, v5);
        vacc4 = xnn_ksum2_s8(vacc4, vp4.val[0], vp4.val[1]);
        vst1q_s8((int8_t*) out + 32, vp4.val[0]);
        vst1q_s8((int8_t*) out + 96, vp4.val[1]);
        const int8x16_t v6 = vld1q_s8(w6);
        const int8x16_t v7 = vld1q_s8(w7);
        const int8x16x2_t vp6 = xnn_zip_s64(v6, v7);
        vacc6 = xnn_ksum2_s8(vacc6, vp6.val[0], vp6.val[1]);
        vst1q_s8((int8_t*) out + 48, vp6.val[0]);
        vst1q_s8((int8_t*) out + 112, vp6.val[1]);
        xnn_prefetch_to_l1((const int8_t*) w0 + 256);
        xnn_prefetch_to_l1((const int8_t*) w1 + 256);
        xnn_prefetch_to_l1((const int8_t*) w2 + 256);
        xnn_prefetch_to_l1((const int8_t*) w3 + 256);
        xnn_prefetch_to_l1((const int8_t*) w4 + 256);
        xnn_prefetch_to_l1((const int8_t*) w5 + 256);
        xnn_prefetch_to_l1((const int8_t*) w6 + 256);
        xnn_prefetch_to_l1((const int8_t*) w7 + 256);

        w0 += 16;
        w1 += 16;
        w2 += 16;
        w3 += 16;
        w4 += 16;
        w5 += 16;
        w6 += 16;
        w7 += 16;
        out += 128;
      }

      // KC remainder of 8
      if (k >= 8) {
        const int8x16_t vp0 = vcombine_s8(vld1_s8(w0), vld1_s8(w1));
        vacc0 = xnn_ksum_s8(vacc0, vp0);
        vst1q_s8((int8_t*) out + 0, vp0);
        const int8x16_t vp2 = vcombine_s8(vld1_s8(w2), vld1_s8(w3));
        vacc2 = xnn_ksum_s8(vacc2, vp2);
        vst1q_s8((int8_t*) out + 16, vp2);
        const int8x16_t vp4 = vcombine_s8(vld1_s8(w4), vld1_s8(w5));
        vacc4 = xnn_ksum_s8(vacc4, vp4);
        vst1q_s8((int8_t*) out + 32, vp4);
        const int8x16_t vp6 = vcombine_s8(vld1_s8(w6), vld1_s8(w7));
        vacc6 = xnn_ksum_s8(vacc6, vp6);
        vst1q_s8((int8_t*) out + 48, vp6);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        out += 64;
        k -= 8;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);
        const int8x16_t vp0 = vcombine_s8(vcreate_s8(safe_load_u64(w0, k, 0)), vcreate_s8(safe_load_u64(w1, k, 0)));
        vacc0 = xnn_ksum_s8(vacc0, vp0);
        vst1q_s8((int8_t*) out + 0, vp0);
        const int8x16_t vp2 = vcombine_s8(vcreate_s8(safe_load_u64(w2, k, 0)), vcreate_s8(safe_load_u64(w3, k, 0)));
        vacc2 = xnn_ksum_s8(vacc2, vp2);
        vst1q_s8((int8_t*) out + 16, vp2);
        const int8x16_t vp4 = vcombine_s8(vcreate_s8(safe_load_u64(w4, k, 0)), vcreate_s8(safe_load_u64(w5, k, 0)));
        vacc4 = xnn_ksum_s8(vacc4, vp4);
        vst1q_s8((int8_t*) out + 32, vp4);
        const int8x16_t vp6 = vcombine_s8(vcreate_s8(safe_load_u64(w6, k, 0)), vcreate_s8(safe_load_u64(w7, k, 0)));
        vacc6 = xnn_ksum_s8(vacc6, vp6);
        vst1q_s8((int8_t*) out + 48, vp6);

        out += 64;
      }

      // Subtract ksum * input_zero_point from bias
      const int32x4_t vksum0 = xnn_padd_s32(vacc0, vacc2);
      const int32x4_t vksum4 = xnn_padd_s32(vacc4, vacc6);
      vst1q_s32(packed_b + 0, vmlsq_s32(vld1q_s32(packed_b + 0), vksum0, vzeropoint));
      vst1q_s32(packed_b + 4, vmlsq_s32(vld1q_s32(packed_b + 4), vksum4, vzeropoint));

      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights = (const int8_t*) ((intptr_t) weights + nc * w_stride);
  } while (--g != 0);
}
