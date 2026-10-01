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
// If 'odd' is set, one more byte is loaded, but only its low nibble is
// used and its high nibble is replaced by the high nibble of 'pad_value'.
// This handles an odd number of 4 bit weights (KC) per row.
XNN_INLINE static uint64_t safe_load_u64(const void* address, size_t n, size_t odd, uint8_t pad_value) {
  uint64_t value = (uint64_t) pad_value * 0x0101010101010101ULL;
  assert(n + odd <= sizeof(uint64_t));
  const uint8_t* bytes = (const uint8_t*) address;
  for (size_t i = 0; i < n; ++i) {
    ((uint8_t*) &value)[i] = bytes[i];
  }
  if (odd) {
    ((uint8_t*) &value)[n] = (uint8_t) ((bytes[n] & 0x0F) | (pad_value & 0xF0));
  }
  return value;
}

// Convert KR blocks of packed nibbles to planar nibbles.
//
// Each 8 byte KR block of the source holds 16 int4 weights k0..k15 with
// byte i = k[2i] | k[2i+1] << 4.  The planar layout has byte j = k[j] |
// k[j+8] << 4.  With L = bytes 0..3 (k0..k7) and H = bytes 4..7 (k8..k15):
//   planar[2i]   = (L[i] & 0x0F) | (H[i] << 4)    = vsli(L, H, 4)
//   planar[2i+1] = (L[i] >> 4)   | (H[i] & 0xF0)  = vsri(H, L, 4)
// followed by a byte interleave of the 2 results.
//
// vl holds L and vh holds H of 4 KR blocks, one block per 32 bit lane.
// Returns the 4 planar KR blocks in lane order: val[0] = blocks 0 and 1,
// val[1] = blocks 2 and 3.
//
// TODO: Move xnn_packed2planar to a shared header (for reuse by 2 bit
// packing) and add a unit test for it.
XNN_INLINE static uint8x16x2_t xnn_packed2planar(uint8x16_t vl, uint8x16_t vh) {
  return vzipq_u8(vsliq_n_u8(vl, vh, 4), vsriq_n_u8(vh, vl, 4));
}

// Add the 2 signed nibbles of each byte of v to the 8 bit sums.
XNN_INLINE static int8x16_t xnn_nibble_acc_s4(int8x16_t vsum, uint8x16_t v) {
  vsum = vsraq_n_s8(vsum, vreinterpretq_s8_u8(v), 4);  // high nibble
  return vsraq_n_s8(vsum, vshlq_n_s8(vreinterpretq_s8_u8(v), 4), 4);  // low nibble
}

void xnn_qs8_qc4w_packw_gemm_goi_ukernel_x16c8__neon_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t n_stride,
  const uint8_t* weights,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  assert(params != NULL);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);
  // KR=8 4 bit with 2 planes is 8 bytes.  Measure in bytes.
  // TODO: kr-avxvnni QS4 asserts kc is even and drops the last nibble when
  // kc is odd, while this kernel packs it.  Fix AVX to handle odd kc, or
  // make this kernel match AVX.
  const size_t kc_odd = kc & 1;  // Last byte of each row has 1 weight.
  kc >>= 1;
  const size_t w_stride = n_stride >> 1;
  // XOR converts uint4 to int4 when kernel_zero_point is 8.
  const uint8_t kzp = (uint8_t) (params->kernel_zero_point * 0x11);
  const uint8x16_t vkzp = vdupq_n_u8(kzp);
  // Bias and ksum are scaled by 16 for the planar int4 GEMM.
  const int32x4_t vzeropoint = vdupq_n_s32(((int32_t) params->input_zero_point + 0) * 16);
  // The KC remainder transposes 4 rows at a time by loading 32 bit lanes.
  // Every lane is loaded before use.
  uint32x4x2_t vtb0;
  uint32x4x2_t vtb4;
  uint32x4x2_t vtb8;
  uint32x4x2_t vtb12;

  uint8_t* out = (uint8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;

  do {
    const uint8_t* wb = (const uint8_t*) weights;
    size_t n = nc;
    // NC main loop multiple of 16
    for (; n >= 16; n -= 16) {
      const uint8_t* w0 = wb;
      const uint8_t* w1 = w0 + w_stride;
      const uint8_t* w2 = w1 + w_stride;
      const uint8_t* w3 = w2 + w_stride;
      const uint8_t* w4 = w3 + w_stride;
      const uint8_t* w5 = w4 + w_stride;
      const uint8_t* w6 = w5 + w_stride;
      const uint8_t* w7 = w6 + w_stride;
      const uint8_t* w8 = w7 + w_stride;
      const uint8_t* w9 = w8 + w_stride;
      const uint8_t* w10 = w9 + w_stride;
      const uint8_t* w11 = w10 + w_stride;
      const uint8_t* w12 = w11 + w_stride;
      const uint8_t* w13 = w12 + w_stride;
      const uint8_t* w14 = w13 + w_stride;
      const uint8_t* w15 = w14 + w_stride;

      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        const int32x4_t vb0 = vshlq_n_s32(vld1q_s32(b + 0), 4);
        const int32x4_t vb4 = vshlq_n_s32(vld1q_s32(b + 4), 4);
        const int32x4_t vb8 = vshlq_n_s32(vld1q_s32(b + 8), 4);
        const int32x4_t vb12 = vshlq_n_s32(vld1q_s32(b + 12), 4);
        vst1q_s32(packed_b + 0, vb0);
        vst1q_s32(packed_b + 4, vb4);
        vst1q_s32(packed_b + 8, vb8);
        vst1q_s32(packed_b + 12, vb12);
        b += 16;
      } else {
        vst1q_s32(packed_b + 0, vdupq_n_s32(0));
        vst1q_s32(packed_b + 4, vdupq_n_s32(0));
        vst1q_s32(packed_b + 8, vdupq_n_s32(0));
        vst1q_s32(packed_b + 12, vdupq_n_s32(0));
      }
      out += 16 * sizeof(int32_t);

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
      xnn_prefetch_to_l1((const int8_t*) w8 + 0);
      xnn_prefetch_to_l1((const int8_t*) w8 + 64);
      xnn_prefetch_to_l1((const int8_t*) w8 + 128);
      xnn_prefetch_to_l1((const int8_t*) w8 + 192);
      xnn_prefetch_to_l1((const int8_t*) w9 + 0);
      xnn_prefetch_to_l1((const int8_t*) w9 + 64);
      xnn_prefetch_to_l1((const int8_t*) w9 + 128);
      xnn_prefetch_to_l1((const int8_t*) w9 + 192);
      xnn_prefetch_to_l1((const int8_t*) w10 + 0);
      xnn_prefetch_to_l1((const int8_t*) w10 + 64);
      xnn_prefetch_to_l1((const int8_t*) w10 + 128);
      xnn_prefetch_to_l1((const int8_t*) w10 + 192);
      xnn_prefetch_to_l1((const int8_t*) w11 + 0);
      xnn_prefetch_to_l1((const int8_t*) w11 + 64);
      xnn_prefetch_to_l1((const int8_t*) w11 + 128);
      xnn_prefetch_to_l1((const int8_t*) w11 + 192);
      xnn_prefetch_to_l1((const int8_t*) w12 + 0);
      xnn_prefetch_to_l1((const int8_t*) w12 + 64);
      xnn_prefetch_to_l1((const int8_t*) w12 + 128);
      xnn_prefetch_to_l1((const int8_t*) w12 + 192);
      xnn_prefetch_to_l1((const int8_t*) w13 + 0);
      xnn_prefetch_to_l1((const int8_t*) w13 + 64);
      xnn_prefetch_to_l1((const int8_t*) w13 + 128);
      xnn_prefetch_to_l1((const int8_t*) w13 + 192);
      xnn_prefetch_to_l1((const int8_t*) w14 + 0);
      xnn_prefetch_to_l1((const int8_t*) w14 + 64);
      xnn_prefetch_to_l1((const int8_t*) w14 + 128);
      xnn_prefetch_to_l1((const int8_t*) w14 + 192);
      xnn_prefetch_to_l1((const int8_t*) w15 + 0);
      xnn_prefetch_to_l1((const int8_t*) w15 + 64);
      xnn_prefetch_to_l1((const int8_t*) w15 + 128);
      xnn_prefetch_to_l1((const int8_t*) w15 + 192);

      size_t k = kc;
      // ksum of rows [a, b, a, b] from the main loop
      int32x4_t vaccp0 = vdupq_n_s32(0);
      int32x4_t vaccp2 = vdupq_n_s32(0);
      int32x4_t vaccp4 = vdupq_n_s32(0);
      int32x4_t vaccp6 = vdupq_n_s32(0);
      int32x4_t vaccp8 = vdupq_n_s32(0);
      int32x4_t vaccp10 = vdupq_n_s32(0);
      int32x4_t vaccp12 = vdupq_n_s32(0);
      int32x4_t vaccp14 = vdupq_n_s32(0);
      // ksum of rows [a, b, c, d] from the remainders
      int32x4_t vacc0 = vdupq_n_s32(0);
      int32x4_t vacc4 = vdupq_n_s32(0);
      int32x4_t vacc8 = vdupq_n_s32(0);
      int32x4_t vacc12 = vdupq_n_s32(0);

      // KC main loop multiple of 16x32
      for (; k >= 32; k -= 32) {
        const uint8x16_t va0_0 = veorq_u8(vld1q_u8(w0 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb0_0 = veorq_u8(vld1q_u8(w1 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt0_0 = vtrnq_u32(vreinterpretq_u32_u8(va0_0), vreinterpretq_u32_u8(vb0_0));
        const uint8x16_t vl0_0 = vreinterpretq_u8_u32(vt0_0.val[0]);
        const uint8x16_t vh0_0 = vreinterpretq_u8_u32(vt0_0.val[1]);
        const uint8x16x2_t vp0_0 = xnn_packed2planar(vl0_0, vh0_0);
        vst1q_u8(out + 0, vp0_0.val[0]);
        vst1q_u8(out + 128, vp0_0.val[1]);
        int8x16_t vsum0 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl0_0);
        vsum0 = xnn_nibble_acc_s4(vsum0, vh0_0);
        const uint8x16_t va2_0 = veorq_u8(vld1q_u8(w2 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb2_0 = veorq_u8(vld1q_u8(w3 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt2_0 = vtrnq_u32(vreinterpretq_u32_u8(va2_0), vreinterpretq_u32_u8(vb2_0));
        const uint8x16_t vl2_0 = vreinterpretq_u8_u32(vt2_0.val[0]);
        const uint8x16_t vh2_0 = vreinterpretq_u8_u32(vt2_0.val[1]);
        const uint8x16x2_t vp2_0 = xnn_packed2planar(vl2_0, vh2_0);
        vst1q_u8(out + 16, vp2_0.val[0]);
        vst1q_u8(out + 144, vp2_0.val[1]);
        int8x16_t vsum2 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl2_0);
        vsum2 = xnn_nibble_acc_s4(vsum2, vh2_0);
        const uint8x16_t va4_0 = veorq_u8(vld1q_u8(w4 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb4_0 = veorq_u8(vld1q_u8(w5 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt4_0 = vtrnq_u32(vreinterpretq_u32_u8(va4_0), vreinterpretq_u32_u8(vb4_0));
        const uint8x16_t vl4_0 = vreinterpretq_u8_u32(vt4_0.val[0]);
        const uint8x16_t vh4_0 = vreinterpretq_u8_u32(vt4_0.val[1]);
        const uint8x16x2_t vp4_0 = xnn_packed2planar(vl4_0, vh4_0);
        vst1q_u8(out + 32, vp4_0.val[0]);
        vst1q_u8(out + 160, vp4_0.val[1]);
        int8x16_t vsum4 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl4_0);
        vsum4 = xnn_nibble_acc_s4(vsum4, vh4_0);
        const uint8x16_t va6_0 = veorq_u8(vld1q_u8(w6 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb6_0 = veorq_u8(vld1q_u8(w7 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt6_0 = vtrnq_u32(vreinterpretq_u32_u8(va6_0), vreinterpretq_u32_u8(vb6_0));
        const uint8x16_t vl6_0 = vreinterpretq_u8_u32(vt6_0.val[0]);
        const uint8x16_t vh6_0 = vreinterpretq_u8_u32(vt6_0.val[1]);
        const uint8x16x2_t vp6_0 = xnn_packed2planar(vl6_0, vh6_0);
        vst1q_u8(out + 48, vp6_0.val[0]);
        vst1q_u8(out + 176, vp6_0.val[1]);
        int8x16_t vsum6 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl6_0);
        vsum6 = xnn_nibble_acc_s4(vsum6, vh6_0);
        const uint8x16_t va8_0 = veorq_u8(vld1q_u8(w8 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb8_0 = veorq_u8(vld1q_u8(w9 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt8_0 = vtrnq_u32(vreinterpretq_u32_u8(va8_0), vreinterpretq_u32_u8(vb8_0));
        const uint8x16_t vl8_0 = vreinterpretq_u8_u32(vt8_0.val[0]);
        const uint8x16_t vh8_0 = vreinterpretq_u8_u32(vt8_0.val[1]);
        const uint8x16x2_t vp8_0 = xnn_packed2planar(vl8_0, vh8_0);
        vst1q_u8(out + 64, vp8_0.val[0]);
        vst1q_u8(out + 192, vp8_0.val[1]);
        int8x16_t vsum8 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl8_0);
        vsum8 = xnn_nibble_acc_s4(vsum8, vh8_0);
        const uint8x16_t va10_0 = veorq_u8(vld1q_u8(w10 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb10_0 = veorq_u8(vld1q_u8(w11 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt10_0 = vtrnq_u32(vreinterpretq_u32_u8(va10_0), vreinterpretq_u32_u8(vb10_0));
        const uint8x16_t vl10_0 = vreinterpretq_u8_u32(vt10_0.val[0]);
        const uint8x16_t vh10_0 = vreinterpretq_u8_u32(vt10_0.val[1]);
        const uint8x16x2_t vp10_0 = xnn_packed2planar(vl10_0, vh10_0);
        vst1q_u8(out + 80, vp10_0.val[0]);
        vst1q_u8(out + 208, vp10_0.val[1]);
        int8x16_t vsum10 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl10_0);
        vsum10 = xnn_nibble_acc_s4(vsum10, vh10_0);
        const uint8x16_t va12_0 = veorq_u8(vld1q_u8(w12 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb12_0 = veorq_u8(vld1q_u8(w13 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt12_0 = vtrnq_u32(vreinterpretq_u32_u8(va12_0), vreinterpretq_u32_u8(vb12_0));
        const uint8x16_t vl12_0 = vreinterpretq_u8_u32(vt12_0.val[0]);
        const uint8x16_t vh12_0 = vreinterpretq_u8_u32(vt12_0.val[1]);
        const uint8x16x2_t vp12_0 = xnn_packed2planar(vl12_0, vh12_0);
        vst1q_u8(out + 96, vp12_0.val[0]);
        vst1q_u8(out + 224, vp12_0.val[1]);
        int8x16_t vsum12 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl12_0);
        vsum12 = xnn_nibble_acc_s4(vsum12, vh12_0);
        const uint8x16_t va14_0 = veorq_u8(vld1q_u8(w14 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb14_0 = veorq_u8(vld1q_u8(w15 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt14_0 = vtrnq_u32(vreinterpretq_u32_u8(va14_0), vreinterpretq_u32_u8(vb14_0));
        const uint8x16_t vl14_0 = vreinterpretq_u8_u32(vt14_0.val[0]);
        const uint8x16_t vh14_0 = vreinterpretq_u8_u32(vt14_0.val[1]);
        const uint8x16x2_t vp14_0 = xnn_packed2planar(vl14_0, vh14_0);
        vst1q_u8(out + 112, vp14_0.val[0]);
        vst1q_u8(out + 240, vp14_0.val[1]);
        int8x16_t vsum14 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl14_0);
        vsum14 = xnn_nibble_acc_s4(vsum14, vh14_0);
        const uint8x16_t va0_2 = veorq_u8(vld1q_u8(w0 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb0_2 = veorq_u8(vld1q_u8(w1 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt0_2 = vtrnq_u32(vreinterpretq_u32_u8(va0_2), vreinterpretq_u32_u8(vb0_2));
        const uint8x16_t vl0_2 = vreinterpretq_u8_u32(vt0_2.val[0]);
        const uint8x16_t vh0_2 = vreinterpretq_u8_u32(vt0_2.val[1]);
        const uint8x16x2_t vp0_2 = xnn_packed2planar(vl0_2, vh0_2);
        vst1q_u8(out + 256, vp0_2.val[0]);
        vst1q_u8(out + 384, vp0_2.val[1]);
        vsum0 = xnn_nibble_acc_s4(vsum0, vl0_2);
        vsum0 = xnn_nibble_acc_s4(vsum0, vh0_2);
        const uint8x16_t va2_2 = veorq_u8(vld1q_u8(w2 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb2_2 = veorq_u8(vld1q_u8(w3 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt2_2 = vtrnq_u32(vreinterpretq_u32_u8(va2_2), vreinterpretq_u32_u8(vb2_2));
        const uint8x16_t vl2_2 = vreinterpretq_u8_u32(vt2_2.val[0]);
        const uint8x16_t vh2_2 = vreinterpretq_u8_u32(vt2_2.val[1]);
        const uint8x16x2_t vp2_2 = xnn_packed2planar(vl2_2, vh2_2);
        vst1q_u8(out + 272, vp2_2.val[0]);
        vst1q_u8(out + 400, vp2_2.val[1]);
        vsum2 = xnn_nibble_acc_s4(vsum2, vl2_2);
        vsum2 = xnn_nibble_acc_s4(vsum2, vh2_2);
        const uint8x16_t va4_2 = veorq_u8(vld1q_u8(w4 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb4_2 = veorq_u8(vld1q_u8(w5 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt4_2 = vtrnq_u32(vreinterpretq_u32_u8(va4_2), vreinterpretq_u32_u8(vb4_2));
        const uint8x16_t vl4_2 = vreinterpretq_u8_u32(vt4_2.val[0]);
        const uint8x16_t vh4_2 = vreinterpretq_u8_u32(vt4_2.val[1]);
        const uint8x16x2_t vp4_2 = xnn_packed2planar(vl4_2, vh4_2);
        vst1q_u8(out + 288, vp4_2.val[0]);
        vst1q_u8(out + 416, vp4_2.val[1]);
        vsum4 = xnn_nibble_acc_s4(vsum4, vl4_2);
        vsum4 = xnn_nibble_acc_s4(vsum4, vh4_2);
        const uint8x16_t va6_2 = veorq_u8(vld1q_u8(w6 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb6_2 = veorq_u8(vld1q_u8(w7 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt6_2 = vtrnq_u32(vreinterpretq_u32_u8(va6_2), vreinterpretq_u32_u8(vb6_2));
        const uint8x16_t vl6_2 = vreinterpretq_u8_u32(vt6_2.val[0]);
        const uint8x16_t vh6_2 = vreinterpretq_u8_u32(vt6_2.val[1]);
        const uint8x16x2_t vp6_2 = xnn_packed2planar(vl6_2, vh6_2);
        vst1q_u8(out + 304, vp6_2.val[0]);
        vst1q_u8(out + 432, vp6_2.val[1]);
        vsum6 = xnn_nibble_acc_s4(vsum6, vl6_2);
        vsum6 = xnn_nibble_acc_s4(vsum6, vh6_2);
        const uint8x16_t va8_2 = veorq_u8(vld1q_u8(w8 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb8_2 = veorq_u8(vld1q_u8(w9 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt8_2 = vtrnq_u32(vreinterpretq_u32_u8(va8_2), vreinterpretq_u32_u8(vb8_2));
        const uint8x16_t vl8_2 = vreinterpretq_u8_u32(vt8_2.val[0]);
        const uint8x16_t vh8_2 = vreinterpretq_u8_u32(vt8_2.val[1]);
        const uint8x16x2_t vp8_2 = xnn_packed2planar(vl8_2, vh8_2);
        vst1q_u8(out + 320, vp8_2.val[0]);
        vst1q_u8(out + 448, vp8_2.val[1]);
        vsum8 = xnn_nibble_acc_s4(vsum8, vl8_2);
        vsum8 = xnn_nibble_acc_s4(vsum8, vh8_2);
        const uint8x16_t va10_2 = veorq_u8(vld1q_u8(w10 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb10_2 = veorq_u8(vld1q_u8(w11 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt10_2 = vtrnq_u32(vreinterpretq_u32_u8(va10_2), vreinterpretq_u32_u8(vb10_2));
        const uint8x16_t vl10_2 = vreinterpretq_u8_u32(vt10_2.val[0]);
        const uint8x16_t vh10_2 = vreinterpretq_u8_u32(vt10_2.val[1]);
        const uint8x16x2_t vp10_2 = xnn_packed2planar(vl10_2, vh10_2);
        vst1q_u8(out + 336, vp10_2.val[0]);
        vst1q_u8(out + 464, vp10_2.val[1]);
        vsum10 = xnn_nibble_acc_s4(vsum10, vl10_2);
        vsum10 = xnn_nibble_acc_s4(vsum10, vh10_2);
        const uint8x16_t va12_2 = veorq_u8(vld1q_u8(w12 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb12_2 = veorq_u8(vld1q_u8(w13 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt12_2 = vtrnq_u32(vreinterpretq_u32_u8(va12_2), vreinterpretq_u32_u8(vb12_2));
        const uint8x16_t vl12_2 = vreinterpretq_u8_u32(vt12_2.val[0]);
        const uint8x16_t vh12_2 = vreinterpretq_u8_u32(vt12_2.val[1]);
        const uint8x16x2_t vp12_2 = xnn_packed2planar(vl12_2, vh12_2);
        vst1q_u8(out + 352, vp12_2.val[0]);
        vst1q_u8(out + 480, vp12_2.val[1]);
        vsum12 = xnn_nibble_acc_s4(vsum12, vl12_2);
        vsum12 = xnn_nibble_acc_s4(vsum12, vh12_2);
        const uint8x16_t va14_2 = veorq_u8(vld1q_u8(w14 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb14_2 = veorq_u8(vld1q_u8(w15 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt14_2 = vtrnq_u32(vreinterpretq_u32_u8(va14_2), vreinterpretq_u32_u8(vb14_2));
        const uint8x16_t vl14_2 = vreinterpretq_u8_u32(vt14_2.val[0]);
        const uint8x16_t vh14_2 = vreinterpretq_u8_u32(vt14_2.val[1]);
        const uint8x16x2_t vp14_2 = xnn_packed2planar(vl14_2, vh14_2);
        vst1q_u8(out + 368, vp14_2.val[0]);
        vst1q_u8(out + 496, vp14_2.val[1]);
        vsum14 = xnn_nibble_acc_s4(vsum14, vl14_2);
        vsum14 = xnn_nibble_acc_s4(vsum14, vh14_2);
        // 8 bit sums of 8 nibbles are in [-64, 56].
        vaccp0 = vpadalq_s16(vaccp0, vpaddlq_s8(vsum0));
        vaccp2 = vpadalq_s16(vaccp2, vpaddlq_s8(vsum2));
        vaccp4 = vpadalq_s16(vaccp4, vpaddlq_s8(vsum4));
        vaccp6 = vpadalq_s16(vaccp6, vpaddlq_s8(vsum6));
        vaccp8 = vpadalq_s16(vaccp8, vpaddlq_s8(vsum8));
        vaccp10 = vpadalq_s16(vaccp10, vpaddlq_s8(vsum10));
        vaccp12 = vpadalq_s16(vaccp12, vpaddlq_s8(vsum12));
        vaccp14 = vpadalq_s16(vaccp14, vpaddlq_s8(vsum14));
        xnn_prefetch_to_l1((const int8_t*) w0 + 256);
        xnn_prefetch_to_l1((const int8_t*) w1 + 256);
        xnn_prefetch_to_l1((const int8_t*) w2 + 256);
        xnn_prefetch_to_l1((const int8_t*) w3 + 256);
        xnn_prefetch_to_l1((const int8_t*) w4 + 256);
        xnn_prefetch_to_l1((const int8_t*) w5 + 256);
        xnn_prefetch_to_l1((const int8_t*) w6 + 256);
        xnn_prefetch_to_l1((const int8_t*) w7 + 256);
        xnn_prefetch_to_l1((const int8_t*) w8 + 256);
        xnn_prefetch_to_l1((const int8_t*) w9 + 256);
        xnn_prefetch_to_l1((const int8_t*) w10 + 256);
        xnn_prefetch_to_l1((const int8_t*) w11 + 256);
        xnn_prefetch_to_l1((const int8_t*) w12 + 256);
        xnn_prefetch_to_l1((const int8_t*) w13 + 256);
        xnn_prefetch_to_l1((const int8_t*) w14 + 256);
        xnn_prefetch_to_l1((const int8_t*) w15 + 256);

        w0 += 32;
        w1 += 32;
        w2 += 32;
        w3 += 32;
        w4 += 32;
        w5 += 32;
        w6 += 32;
        w7 += 32;
        w8 += 32;
        w9 += 32;
        w10 += 32;
        w11 += 32;
        w12 += 32;
        w13 += 32;
        w14 += 32;
        w15 += 32;
        out += 512;
      }

      // KC remainder of multiples of 8
      for (; k >= 8; k -= 8) {
        // vtb.val[0] = [a.L, b.L, c.L, d.L], vtb.val[1] = [a.H, b.H, c.H, d.H]
        vtb0 = vld2q_lane_u32((const uint32_t*) w0, vtb0, 0);
        vtb0 = vld2q_lane_u32((const uint32_t*) w1, vtb0, 1);
        vtb0 = vld2q_lane_u32((const uint32_t*) w2, vtb0, 2);
        vtb0 = vld2q_lane_u32((const uint32_t*) w3, vtb0, 3);
        vtb4 = vld2q_lane_u32((const uint32_t*) w4, vtb4, 0);
        vtb4 = vld2q_lane_u32((const uint32_t*) w5, vtb4, 1);
        vtb4 = vld2q_lane_u32((const uint32_t*) w6, vtb4, 2);
        vtb4 = vld2q_lane_u32((const uint32_t*) w7, vtb4, 3);
        vtb8 = vld2q_lane_u32((const uint32_t*) w8, vtb8, 0);
        vtb8 = vld2q_lane_u32((const uint32_t*) w9, vtb8, 1);
        vtb8 = vld2q_lane_u32((const uint32_t*) w10, vtb8, 2);
        vtb8 = vld2q_lane_u32((const uint32_t*) w11, vtb8, 3);
        vtb12 = vld2q_lane_u32((const uint32_t*) w12, vtb12, 0);
        vtb12 = vld2q_lane_u32((const uint32_t*) w13, vtb12, 1);
        vtb12 = vld2q_lane_u32((const uint32_t*) w14, vtb12, 2);
        vtb12 = vld2q_lane_u32((const uint32_t*) w15, vtb12, 3);
        const uint8x16_t vl0 = veorq_u8(vreinterpretq_u8_u32(vtb0.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh0 = veorq_u8(vreinterpretq_u8_u32(vtb0.val[1]), vkzp);
        const uint8x16x2_t vp0 = xnn_packed2planar(vl0, vh0);
        vst1q_u8(out + 0, vp0.val[0]);
        vst1q_u8(out + 16, vp0.val[1]);
        vacc0 = vpadalq_s16(vacc0, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl0), vh0)));
        const uint8x16_t vl4 = veorq_u8(vreinterpretq_u8_u32(vtb4.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh4 = veorq_u8(vreinterpretq_u8_u32(vtb4.val[1]), vkzp);
        const uint8x16x2_t vp4 = xnn_packed2planar(vl4, vh4);
        vst1q_u8(out + 32, vp4.val[0]);
        vst1q_u8(out + 48, vp4.val[1]);
        vacc4 = vpadalq_s16(vacc4, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl4), vh4)));
        const uint8x16_t vl8 = veorq_u8(vreinterpretq_u8_u32(vtb8.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh8 = veorq_u8(vreinterpretq_u8_u32(vtb8.val[1]), vkzp);
        const uint8x16x2_t vp8 = xnn_packed2planar(vl8, vh8);
        vst1q_u8(out + 64, vp8.val[0]);
        vst1q_u8(out + 80, vp8.val[1]);
        vacc8 = vpadalq_s16(vacc8, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl8), vh8)));
        const uint8x16_t vl12 = veorq_u8(vreinterpretq_u8_u32(vtb12.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh12 = veorq_u8(vreinterpretq_u8_u32(vtb12.val[1]), vkzp);
        const uint8x16x2_t vp12 = xnn_packed2planar(vl12, vh12);
        vst1q_u8(out + 96, vp12.val[0]);
        vst1q_u8(out + 112, vp12.val[1]);
        vacc12 = vpadalq_s16(vacc12, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl12), vh12)));

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        w8 += 8;
        w9 += 8;
        w10 += 8;
        w11 += 8;
        w12 += 8;
        w13 += 8;
        w14 += 8;
        w15 += 8;
        out += 128;
      }

      // KC remainder of 1..7 bytes
      if (k != 0 || kc_odd != 0) {
        assert(k + kc_odd >= 1 && k + kc_odd <= 8);
        // vu.val[0] = [a.L, b.L, c.L, d.L], vu.val[1] = [a.H, b.H, c.H, d.H]
        const uint32x4x2_t vu0 = vuzpq_u32(
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w0, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w1, k, kc_odd, kzp)))),
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w2, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w3, k, kc_odd, kzp)))));
        const uint8x16_t vl0 = veorq_u8(vreinterpretq_u8_u32(vu0.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh0 = veorq_u8(vreinterpretq_u8_u32(vu0.val[1]), vkzp);
        const uint8x16x2_t vp0 = xnn_packed2planar(vl0, vh0);
        vst1q_u8(out + 0, vp0.val[0]);
        vst1q_u8(out + 16, vp0.val[1]);
        vacc0 = vpadalq_s16(vacc0, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl0), vh0)));
        // vu.val[0] = [a.L, b.L, c.L, d.L], vu.val[1] = [a.H, b.H, c.H, d.H]
        const uint32x4x2_t vu4 = vuzpq_u32(
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w4, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w5, k, kc_odd, kzp)))),
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w6, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w7, k, kc_odd, kzp)))));
        const uint8x16_t vl4 = veorq_u8(vreinterpretq_u8_u32(vu4.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh4 = veorq_u8(vreinterpretq_u8_u32(vu4.val[1]), vkzp);
        const uint8x16x2_t vp4 = xnn_packed2planar(vl4, vh4);
        vst1q_u8(out + 32, vp4.val[0]);
        vst1q_u8(out + 48, vp4.val[1]);
        vacc4 = vpadalq_s16(vacc4, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl4), vh4)));
        // vu.val[0] = [a.L, b.L, c.L, d.L], vu.val[1] = [a.H, b.H, c.H, d.H]
        const uint32x4x2_t vu8 = vuzpq_u32(
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w8, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w9, k, kc_odd, kzp)))),
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w10, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w11, k, kc_odd, kzp)))));
        const uint8x16_t vl8 = veorq_u8(vreinterpretq_u8_u32(vu8.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh8 = veorq_u8(vreinterpretq_u8_u32(vu8.val[1]), vkzp);
        const uint8x16x2_t vp8 = xnn_packed2planar(vl8, vh8);
        vst1q_u8(out + 64, vp8.val[0]);
        vst1q_u8(out + 80, vp8.val[1]);
        vacc8 = vpadalq_s16(vacc8, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl8), vh8)));
        // vu.val[0] = [a.L, b.L, c.L, d.L], vu.val[1] = [a.H, b.H, c.H, d.H]
        const uint32x4x2_t vu12 = vuzpq_u32(
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w12, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w13, k, kc_odd, kzp)))),
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w14, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w15, k, kc_odd, kzp)))));
        const uint8x16_t vl12 = veorq_u8(vreinterpretq_u8_u32(vu12.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh12 = veorq_u8(vreinterpretq_u8_u32(vu12.val[1]), vkzp);
        const uint8x16x2_t vp12 = xnn_packed2planar(vl12, vh12);
        vst1q_u8(out + 96, vp12.val[0]);
        vst1q_u8(out + 112, vp12.val[1]);
        vacc12 = vpadalq_s16(vacc12, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl12), vh12)));

        out += 128;
      }

      // Subtract ksum * input_zero_point from bias
      vacc0 = vaddq_s32(vacc0, vcombine_s32(
          vadd_s32(vget_low_s32(vaccp0), vget_high_s32(vaccp0)),
          vadd_s32(vget_low_s32(vaccp2), vget_high_s32(vaccp2))));
      vacc4 = vaddq_s32(vacc4, vcombine_s32(
          vadd_s32(vget_low_s32(vaccp4), vget_high_s32(vaccp4)),
          vadd_s32(vget_low_s32(vaccp6), vget_high_s32(vaccp6))));
      vacc8 = vaddq_s32(vacc8, vcombine_s32(
          vadd_s32(vget_low_s32(vaccp8), vget_high_s32(vaccp8)),
          vadd_s32(vget_low_s32(vaccp10), vget_high_s32(vaccp10))));
      vacc12 = vaddq_s32(vacc12, vcombine_s32(
          vadd_s32(vget_low_s32(vaccp12), vget_high_s32(vaccp12)),
          vadd_s32(vget_low_s32(vaccp14), vget_high_s32(vaccp14))));
      vst1q_s32(packed_b + 0, vmlsq_s32(vld1q_s32(packed_b + 0), vacc0, vzeropoint));
      vst1q_s32(packed_b + 4, vmlsq_s32(vld1q_s32(packed_b + 4), vacc4, vzeropoint));
      vst1q_s32(packed_b + 8, vmlsq_s32(vld1q_s32(packed_b + 8), vacc8, vzeropoint));
      vst1q_s32(packed_b + 12, vmlsq_s32(vld1q_s32(packed_b + 12), vacc12, vzeropoint));

      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
      wb += 16 * w_stride;
    }
    // NC remainder (1..15)
    // Same as main loop except bias is copied and w pointers are clamped
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 15);
      const uint8_t* w0 = wb;
      const uint8_t* w1 = w0 + w_stride;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint8_t* w2 = w1 + w_stride;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint8_t* w3 = w2 + w_stride;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint8_t* w4 = w3 + w_stride;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint8_t* w5 = w4 + w_stride;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint8_t* w6 = w5 + w_stride;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint8_t* w7 = w6 + w_stride;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint8_t* w8 = w7 + w_stride;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint8_t* w9 = w8 + w_stride;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint8_t* w10 = w9 + w_stride;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint8_t* w11 = w10 + w_stride;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint8_t* w12 = w11 + w_stride;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint8_t* w13 = w12 + w_stride;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint8_t* w14 = w13 + w_stride;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const uint8_t* w15 = w14 + w_stride;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }

      int32_t* packed_b = (int32_t*) out;
      vst1q_s32(packed_b + 0, vdupq_n_s32(0));
      vst1q_s32(packed_b + 4, vdupq_n_s32(0));
      vst1q_s32(packed_b + 8, vdupq_n_s32(0));
      vst1q_s32(packed_b + 12, vdupq_n_s32(0));
      if XNN_LIKELY(b != NULL) {
        for (size_t nb = 0; nb < n; ++nb) {
          packed_b[nb] = (int32_t) ((uint32_t) b[nb] << 4);
        }
        b += n;
      }
      out += 16 * sizeof(int32_t);

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
      xnn_prefetch_to_l1((const int8_t*) w8 + 0);
      xnn_prefetch_to_l1((const int8_t*) w8 + 64);
      xnn_prefetch_to_l1((const int8_t*) w8 + 128);
      xnn_prefetch_to_l1((const int8_t*) w8 + 192);
      xnn_prefetch_to_l1((const int8_t*) w9 + 0);
      xnn_prefetch_to_l1((const int8_t*) w9 + 64);
      xnn_prefetch_to_l1((const int8_t*) w9 + 128);
      xnn_prefetch_to_l1((const int8_t*) w9 + 192);
      xnn_prefetch_to_l1((const int8_t*) w10 + 0);
      xnn_prefetch_to_l1((const int8_t*) w10 + 64);
      xnn_prefetch_to_l1((const int8_t*) w10 + 128);
      xnn_prefetch_to_l1((const int8_t*) w10 + 192);
      xnn_prefetch_to_l1((const int8_t*) w11 + 0);
      xnn_prefetch_to_l1((const int8_t*) w11 + 64);
      xnn_prefetch_to_l1((const int8_t*) w11 + 128);
      xnn_prefetch_to_l1((const int8_t*) w11 + 192);
      xnn_prefetch_to_l1((const int8_t*) w12 + 0);
      xnn_prefetch_to_l1((const int8_t*) w12 + 64);
      xnn_prefetch_to_l1((const int8_t*) w12 + 128);
      xnn_prefetch_to_l1((const int8_t*) w12 + 192);
      xnn_prefetch_to_l1((const int8_t*) w13 + 0);
      xnn_prefetch_to_l1((const int8_t*) w13 + 64);
      xnn_prefetch_to_l1((const int8_t*) w13 + 128);
      xnn_prefetch_to_l1((const int8_t*) w13 + 192);
      xnn_prefetch_to_l1((const int8_t*) w14 + 0);
      xnn_prefetch_to_l1((const int8_t*) w14 + 64);
      xnn_prefetch_to_l1((const int8_t*) w14 + 128);
      xnn_prefetch_to_l1((const int8_t*) w14 + 192);
      xnn_prefetch_to_l1((const int8_t*) w15 + 0);
      xnn_prefetch_to_l1((const int8_t*) w15 + 64);
      xnn_prefetch_to_l1((const int8_t*) w15 + 128);
      xnn_prefetch_to_l1((const int8_t*) w15 + 192);

      size_t k = kc;
      // ksum of rows [a, b, a, b] from the main loop
      int32x4_t vaccp0 = vdupq_n_s32(0);
      int32x4_t vaccp2 = vdupq_n_s32(0);
      int32x4_t vaccp4 = vdupq_n_s32(0);
      int32x4_t vaccp6 = vdupq_n_s32(0);
      int32x4_t vaccp8 = vdupq_n_s32(0);
      int32x4_t vaccp10 = vdupq_n_s32(0);
      int32x4_t vaccp12 = vdupq_n_s32(0);
      int32x4_t vaccp14 = vdupq_n_s32(0);
      // ksum of rows [a, b, c, d] from the remainders
      int32x4_t vacc0 = vdupq_n_s32(0);
      int32x4_t vacc4 = vdupq_n_s32(0);
      int32x4_t vacc8 = vdupq_n_s32(0);
      int32x4_t vacc12 = vdupq_n_s32(0);

      // KC main loop multiple of 16x32
      for (; k >= 32; k -= 32) {
        const uint8x16_t va0_0 = veorq_u8(vld1q_u8(w0 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb0_0 = veorq_u8(vld1q_u8(w1 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt0_0 = vtrnq_u32(vreinterpretq_u32_u8(va0_0), vreinterpretq_u32_u8(vb0_0));
        const uint8x16_t vl0_0 = vreinterpretq_u8_u32(vt0_0.val[0]);
        const uint8x16_t vh0_0 = vreinterpretq_u8_u32(vt0_0.val[1]);
        const uint8x16x2_t vp0_0 = xnn_packed2planar(vl0_0, vh0_0);
        vst1q_u8(out + 0, vp0_0.val[0]);
        vst1q_u8(out + 128, vp0_0.val[1]);
        int8x16_t vsum0 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl0_0);
        vsum0 = xnn_nibble_acc_s4(vsum0, vh0_0);
        const uint8x16_t va2_0 = veorq_u8(vld1q_u8(w2 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb2_0 = veorq_u8(vld1q_u8(w3 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt2_0 = vtrnq_u32(vreinterpretq_u32_u8(va2_0), vreinterpretq_u32_u8(vb2_0));
        const uint8x16_t vl2_0 = vreinterpretq_u8_u32(vt2_0.val[0]);
        const uint8x16_t vh2_0 = vreinterpretq_u8_u32(vt2_0.val[1]);
        const uint8x16x2_t vp2_0 = xnn_packed2planar(vl2_0, vh2_0);
        vst1q_u8(out + 16, vp2_0.val[0]);
        vst1q_u8(out + 144, vp2_0.val[1]);
        int8x16_t vsum2 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl2_0);
        vsum2 = xnn_nibble_acc_s4(vsum2, vh2_0);
        const uint8x16_t va4_0 = veorq_u8(vld1q_u8(w4 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb4_0 = veorq_u8(vld1q_u8(w5 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt4_0 = vtrnq_u32(vreinterpretq_u32_u8(va4_0), vreinterpretq_u32_u8(vb4_0));
        const uint8x16_t vl4_0 = vreinterpretq_u8_u32(vt4_0.val[0]);
        const uint8x16_t vh4_0 = vreinterpretq_u8_u32(vt4_0.val[1]);
        const uint8x16x2_t vp4_0 = xnn_packed2planar(vl4_0, vh4_0);
        vst1q_u8(out + 32, vp4_0.val[0]);
        vst1q_u8(out + 160, vp4_0.val[1]);
        int8x16_t vsum4 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl4_0);
        vsum4 = xnn_nibble_acc_s4(vsum4, vh4_0);
        const uint8x16_t va6_0 = veorq_u8(vld1q_u8(w6 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb6_0 = veorq_u8(vld1q_u8(w7 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt6_0 = vtrnq_u32(vreinterpretq_u32_u8(va6_0), vreinterpretq_u32_u8(vb6_0));
        const uint8x16_t vl6_0 = vreinterpretq_u8_u32(vt6_0.val[0]);
        const uint8x16_t vh6_0 = vreinterpretq_u8_u32(vt6_0.val[1]);
        const uint8x16x2_t vp6_0 = xnn_packed2planar(vl6_0, vh6_0);
        vst1q_u8(out + 48, vp6_0.val[0]);
        vst1q_u8(out + 176, vp6_0.val[1]);
        int8x16_t vsum6 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl6_0);
        vsum6 = xnn_nibble_acc_s4(vsum6, vh6_0);
        const uint8x16_t va8_0 = veorq_u8(vld1q_u8(w8 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb8_0 = veorq_u8(vld1q_u8(w9 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt8_0 = vtrnq_u32(vreinterpretq_u32_u8(va8_0), vreinterpretq_u32_u8(vb8_0));
        const uint8x16_t vl8_0 = vreinterpretq_u8_u32(vt8_0.val[0]);
        const uint8x16_t vh8_0 = vreinterpretq_u8_u32(vt8_0.val[1]);
        const uint8x16x2_t vp8_0 = xnn_packed2planar(vl8_0, vh8_0);
        vst1q_u8(out + 64, vp8_0.val[0]);
        vst1q_u8(out + 192, vp8_0.val[1]);
        int8x16_t vsum8 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl8_0);
        vsum8 = xnn_nibble_acc_s4(vsum8, vh8_0);
        const uint8x16_t va10_0 = veorq_u8(vld1q_u8(w10 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb10_0 = veorq_u8(vld1q_u8(w11 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt10_0 = vtrnq_u32(vreinterpretq_u32_u8(va10_0), vreinterpretq_u32_u8(vb10_0));
        const uint8x16_t vl10_0 = vreinterpretq_u8_u32(vt10_0.val[0]);
        const uint8x16_t vh10_0 = vreinterpretq_u8_u32(vt10_0.val[1]);
        const uint8x16x2_t vp10_0 = xnn_packed2planar(vl10_0, vh10_0);
        vst1q_u8(out + 80, vp10_0.val[0]);
        vst1q_u8(out + 208, vp10_0.val[1]);
        int8x16_t vsum10 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl10_0);
        vsum10 = xnn_nibble_acc_s4(vsum10, vh10_0);
        const uint8x16_t va12_0 = veorq_u8(vld1q_u8(w12 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb12_0 = veorq_u8(vld1q_u8(w13 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt12_0 = vtrnq_u32(vreinterpretq_u32_u8(va12_0), vreinterpretq_u32_u8(vb12_0));
        const uint8x16_t vl12_0 = vreinterpretq_u8_u32(vt12_0.val[0]);
        const uint8x16_t vh12_0 = vreinterpretq_u8_u32(vt12_0.val[1]);
        const uint8x16x2_t vp12_0 = xnn_packed2planar(vl12_0, vh12_0);
        vst1q_u8(out + 96, vp12_0.val[0]);
        vst1q_u8(out + 224, vp12_0.val[1]);
        int8x16_t vsum12 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl12_0);
        vsum12 = xnn_nibble_acc_s4(vsum12, vh12_0);
        const uint8x16_t va14_0 = veorq_u8(vld1q_u8(w14 + 0), vkzp);  // uint4 -> int4
        const uint8x16_t vb14_0 = veorq_u8(vld1q_u8(w15 + 0), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt14_0 = vtrnq_u32(vreinterpretq_u32_u8(va14_0), vreinterpretq_u32_u8(vb14_0));
        const uint8x16_t vl14_0 = vreinterpretq_u8_u32(vt14_0.val[0]);
        const uint8x16_t vh14_0 = vreinterpretq_u8_u32(vt14_0.val[1]);
        const uint8x16x2_t vp14_0 = xnn_packed2planar(vl14_0, vh14_0);
        vst1q_u8(out + 112, vp14_0.val[0]);
        vst1q_u8(out + 240, vp14_0.val[1]);
        int8x16_t vsum14 = xnn_nibble_acc_s4(vdupq_n_s8(0), vl14_0);
        vsum14 = xnn_nibble_acc_s4(vsum14, vh14_0);
        const uint8x16_t va0_2 = veorq_u8(vld1q_u8(w0 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb0_2 = veorq_u8(vld1q_u8(w1 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt0_2 = vtrnq_u32(vreinterpretq_u32_u8(va0_2), vreinterpretq_u32_u8(vb0_2));
        const uint8x16_t vl0_2 = vreinterpretq_u8_u32(vt0_2.val[0]);
        const uint8x16_t vh0_2 = vreinterpretq_u8_u32(vt0_2.val[1]);
        const uint8x16x2_t vp0_2 = xnn_packed2planar(vl0_2, vh0_2);
        vst1q_u8(out + 256, vp0_2.val[0]);
        vst1q_u8(out + 384, vp0_2.val[1]);
        vsum0 = xnn_nibble_acc_s4(vsum0, vl0_2);
        vsum0 = xnn_nibble_acc_s4(vsum0, vh0_2);
        const uint8x16_t va2_2 = veorq_u8(vld1q_u8(w2 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb2_2 = veorq_u8(vld1q_u8(w3 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt2_2 = vtrnq_u32(vreinterpretq_u32_u8(va2_2), vreinterpretq_u32_u8(vb2_2));
        const uint8x16_t vl2_2 = vreinterpretq_u8_u32(vt2_2.val[0]);
        const uint8x16_t vh2_2 = vreinterpretq_u8_u32(vt2_2.val[1]);
        const uint8x16x2_t vp2_2 = xnn_packed2planar(vl2_2, vh2_2);
        vst1q_u8(out + 272, vp2_2.val[0]);
        vst1q_u8(out + 400, vp2_2.val[1]);
        vsum2 = xnn_nibble_acc_s4(vsum2, vl2_2);
        vsum2 = xnn_nibble_acc_s4(vsum2, vh2_2);
        const uint8x16_t va4_2 = veorq_u8(vld1q_u8(w4 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb4_2 = veorq_u8(vld1q_u8(w5 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt4_2 = vtrnq_u32(vreinterpretq_u32_u8(va4_2), vreinterpretq_u32_u8(vb4_2));
        const uint8x16_t vl4_2 = vreinterpretq_u8_u32(vt4_2.val[0]);
        const uint8x16_t vh4_2 = vreinterpretq_u8_u32(vt4_2.val[1]);
        const uint8x16x2_t vp4_2 = xnn_packed2planar(vl4_2, vh4_2);
        vst1q_u8(out + 288, vp4_2.val[0]);
        vst1q_u8(out + 416, vp4_2.val[1]);
        vsum4 = xnn_nibble_acc_s4(vsum4, vl4_2);
        vsum4 = xnn_nibble_acc_s4(vsum4, vh4_2);
        const uint8x16_t va6_2 = veorq_u8(vld1q_u8(w6 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb6_2 = veorq_u8(vld1q_u8(w7 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt6_2 = vtrnq_u32(vreinterpretq_u32_u8(va6_2), vreinterpretq_u32_u8(vb6_2));
        const uint8x16_t vl6_2 = vreinterpretq_u8_u32(vt6_2.val[0]);
        const uint8x16_t vh6_2 = vreinterpretq_u8_u32(vt6_2.val[1]);
        const uint8x16x2_t vp6_2 = xnn_packed2planar(vl6_2, vh6_2);
        vst1q_u8(out + 304, vp6_2.val[0]);
        vst1q_u8(out + 432, vp6_2.val[1]);
        vsum6 = xnn_nibble_acc_s4(vsum6, vl6_2);
        vsum6 = xnn_nibble_acc_s4(vsum6, vh6_2);
        const uint8x16_t va8_2 = veorq_u8(vld1q_u8(w8 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb8_2 = veorq_u8(vld1q_u8(w9 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt8_2 = vtrnq_u32(vreinterpretq_u32_u8(va8_2), vreinterpretq_u32_u8(vb8_2));
        const uint8x16_t vl8_2 = vreinterpretq_u8_u32(vt8_2.val[0]);
        const uint8x16_t vh8_2 = vreinterpretq_u8_u32(vt8_2.val[1]);
        const uint8x16x2_t vp8_2 = xnn_packed2planar(vl8_2, vh8_2);
        vst1q_u8(out + 320, vp8_2.val[0]);
        vst1q_u8(out + 448, vp8_2.val[1]);
        vsum8 = xnn_nibble_acc_s4(vsum8, vl8_2);
        vsum8 = xnn_nibble_acc_s4(vsum8, vh8_2);
        const uint8x16_t va10_2 = veorq_u8(vld1q_u8(w10 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb10_2 = veorq_u8(vld1q_u8(w11 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt10_2 = vtrnq_u32(vreinterpretq_u32_u8(va10_2), vreinterpretq_u32_u8(vb10_2));
        const uint8x16_t vl10_2 = vreinterpretq_u8_u32(vt10_2.val[0]);
        const uint8x16_t vh10_2 = vreinterpretq_u8_u32(vt10_2.val[1]);
        const uint8x16x2_t vp10_2 = xnn_packed2planar(vl10_2, vh10_2);
        vst1q_u8(out + 336, vp10_2.val[0]);
        vst1q_u8(out + 464, vp10_2.val[1]);
        vsum10 = xnn_nibble_acc_s4(vsum10, vl10_2);
        vsum10 = xnn_nibble_acc_s4(vsum10, vh10_2);
        const uint8x16_t va12_2 = veorq_u8(vld1q_u8(w12 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb12_2 = veorq_u8(vld1q_u8(w13 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt12_2 = vtrnq_u32(vreinterpretq_u32_u8(va12_2), vreinterpretq_u32_u8(vb12_2));
        const uint8x16_t vl12_2 = vreinterpretq_u8_u32(vt12_2.val[0]);
        const uint8x16_t vh12_2 = vreinterpretq_u8_u32(vt12_2.val[1]);
        const uint8x16x2_t vp12_2 = xnn_packed2planar(vl12_2, vh12_2);
        vst1q_u8(out + 352, vp12_2.val[0]);
        vst1q_u8(out + 480, vp12_2.val[1]);
        vsum12 = xnn_nibble_acc_s4(vsum12, vl12_2);
        vsum12 = xnn_nibble_acc_s4(vsum12, vh12_2);
        const uint8x16_t va14_2 = veorq_u8(vld1q_u8(w14 + 16), vkzp);  // uint4 -> int4
        const uint8x16_t vb14_2 = veorq_u8(vld1q_u8(w15 + 16), vkzp);
        // vt.val[0] = [a.L0, b.L0, a.L1, b.L1], vt.val[1] = [a.H0, b.H0, a.H1, b.H1]
        const uint32x4x2_t vt14_2 = vtrnq_u32(vreinterpretq_u32_u8(va14_2), vreinterpretq_u32_u8(vb14_2));
        const uint8x16_t vl14_2 = vreinterpretq_u8_u32(vt14_2.val[0]);
        const uint8x16_t vh14_2 = vreinterpretq_u8_u32(vt14_2.val[1]);
        const uint8x16x2_t vp14_2 = xnn_packed2planar(vl14_2, vh14_2);
        vst1q_u8(out + 368, vp14_2.val[0]);
        vst1q_u8(out + 496, vp14_2.val[1]);
        vsum14 = xnn_nibble_acc_s4(vsum14, vl14_2);
        vsum14 = xnn_nibble_acc_s4(vsum14, vh14_2);
        // 8 bit sums of 8 nibbles are in [-64, 56].
        vaccp0 = vpadalq_s16(vaccp0, vpaddlq_s8(vsum0));
        vaccp2 = vpadalq_s16(vaccp2, vpaddlq_s8(vsum2));
        vaccp4 = vpadalq_s16(vaccp4, vpaddlq_s8(vsum4));
        vaccp6 = vpadalq_s16(vaccp6, vpaddlq_s8(vsum6));
        vaccp8 = vpadalq_s16(vaccp8, vpaddlq_s8(vsum8));
        vaccp10 = vpadalq_s16(vaccp10, vpaddlq_s8(vsum10));
        vaccp12 = vpadalq_s16(vaccp12, vpaddlq_s8(vsum12));
        vaccp14 = vpadalq_s16(vaccp14, vpaddlq_s8(vsum14));
        xnn_prefetch_to_l1((const int8_t*) w0 + 256);
        xnn_prefetch_to_l1((const int8_t*) w1 + 256);
        xnn_prefetch_to_l1((const int8_t*) w2 + 256);
        xnn_prefetch_to_l1((const int8_t*) w3 + 256);
        xnn_prefetch_to_l1((const int8_t*) w4 + 256);
        xnn_prefetch_to_l1((const int8_t*) w5 + 256);
        xnn_prefetch_to_l1((const int8_t*) w6 + 256);
        xnn_prefetch_to_l1((const int8_t*) w7 + 256);
        xnn_prefetch_to_l1((const int8_t*) w8 + 256);
        xnn_prefetch_to_l1((const int8_t*) w9 + 256);
        xnn_prefetch_to_l1((const int8_t*) w10 + 256);
        xnn_prefetch_to_l1((const int8_t*) w11 + 256);
        xnn_prefetch_to_l1((const int8_t*) w12 + 256);
        xnn_prefetch_to_l1((const int8_t*) w13 + 256);
        xnn_prefetch_to_l1((const int8_t*) w14 + 256);
        xnn_prefetch_to_l1((const int8_t*) w15 + 256);

        w0 += 32;
        w1 += 32;
        w2 += 32;
        w3 += 32;
        w4 += 32;
        w5 += 32;
        w6 += 32;
        w7 += 32;
        w8 += 32;
        w9 += 32;
        w10 += 32;
        w11 += 32;
        w12 += 32;
        w13 += 32;
        w14 += 32;
        w15 += 32;
        out += 512;
      }

      // KC remainder of multiples of 8
      for (; k >= 8; k -= 8) {
        // vtb.val[0] = [a.L, b.L, c.L, d.L], vtb.val[1] = [a.H, b.H, c.H, d.H]
        vtb0 = vld2q_lane_u32((const uint32_t*) w0, vtb0, 0);
        vtb0 = vld2q_lane_u32((const uint32_t*) w1, vtb0, 1);
        vtb0 = vld2q_lane_u32((const uint32_t*) w2, vtb0, 2);
        vtb0 = vld2q_lane_u32((const uint32_t*) w3, vtb0, 3);
        vtb4 = vld2q_lane_u32((const uint32_t*) w4, vtb4, 0);
        vtb4 = vld2q_lane_u32((const uint32_t*) w5, vtb4, 1);
        vtb4 = vld2q_lane_u32((const uint32_t*) w6, vtb4, 2);
        vtb4 = vld2q_lane_u32((const uint32_t*) w7, vtb4, 3);
        vtb8 = vld2q_lane_u32((const uint32_t*) w8, vtb8, 0);
        vtb8 = vld2q_lane_u32((const uint32_t*) w9, vtb8, 1);
        vtb8 = vld2q_lane_u32((const uint32_t*) w10, vtb8, 2);
        vtb8 = vld2q_lane_u32((const uint32_t*) w11, vtb8, 3);
        vtb12 = vld2q_lane_u32((const uint32_t*) w12, vtb12, 0);
        vtb12 = vld2q_lane_u32((const uint32_t*) w13, vtb12, 1);
        vtb12 = vld2q_lane_u32((const uint32_t*) w14, vtb12, 2);
        vtb12 = vld2q_lane_u32((const uint32_t*) w15, vtb12, 3);
        const uint8x16_t vl0 = veorq_u8(vreinterpretq_u8_u32(vtb0.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh0 = veorq_u8(vreinterpretq_u8_u32(vtb0.val[1]), vkzp);
        const uint8x16x2_t vp0 = xnn_packed2planar(vl0, vh0);
        vst1q_u8(out + 0, vp0.val[0]);
        vst1q_u8(out + 16, vp0.val[1]);
        vacc0 = vpadalq_s16(vacc0, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl0), vh0)));
        const uint8x16_t vl4 = veorq_u8(vreinterpretq_u8_u32(vtb4.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh4 = veorq_u8(vreinterpretq_u8_u32(vtb4.val[1]), vkzp);
        const uint8x16x2_t vp4 = xnn_packed2planar(vl4, vh4);
        vst1q_u8(out + 32, vp4.val[0]);
        vst1q_u8(out + 48, vp4.val[1]);
        vacc4 = vpadalq_s16(vacc4, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl4), vh4)));
        const uint8x16_t vl8 = veorq_u8(vreinterpretq_u8_u32(vtb8.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh8 = veorq_u8(vreinterpretq_u8_u32(vtb8.val[1]), vkzp);
        const uint8x16x2_t vp8 = xnn_packed2planar(vl8, vh8);
        vst1q_u8(out + 64, vp8.val[0]);
        vst1q_u8(out + 80, vp8.val[1]);
        vacc8 = vpadalq_s16(vacc8, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl8), vh8)));
        const uint8x16_t vl12 = veorq_u8(vreinterpretq_u8_u32(vtb12.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh12 = veorq_u8(vreinterpretq_u8_u32(vtb12.val[1]), vkzp);
        const uint8x16x2_t vp12 = xnn_packed2planar(vl12, vh12);
        vst1q_u8(out + 96, vp12.val[0]);
        vst1q_u8(out + 112, vp12.val[1]);
        vacc12 = vpadalq_s16(vacc12, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl12), vh12)));

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        w8 += 8;
        w9 += 8;
        w10 += 8;
        w11 += 8;
        w12 += 8;
        w13 += 8;
        w14 += 8;
        w15 += 8;
        out += 128;
      }

      // KC remainder of 1..7 bytes
      if (k != 0 || kc_odd != 0) {
        assert(k + kc_odd >= 1 && k + kc_odd <= 8);
        // vu.val[0] = [a.L, b.L, c.L, d.L], vu.val[1] = [a.H, b.H, c.H, d.H]
        const uint32x4x2_t vu0 = vuzpq_u32(
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w0, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w1, k, kc_odd, kzp)))),
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w2, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w3, k, kc_odd, kzp)))));
        const uint8x16_t vl0 = veorq_u8(vreinterpretq_u8_u32(vu0.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh0 = veorq_u8(vreinterpretq_u8_u32(vu0.val[1]), vkzp);
        const uint8x16x2_t vp0 = xnn_packed2planar(vl0, vh0);
        vst1q_u8(out + 0, vp0.val[0]);
        vst1q_u8(out + 16, vp0.val[1]);
        vacc0 = vpadalq_s16(vacc0, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl0), vh0)));
        // vu.val[0] = [a.L, b.L, c.L, d.L], vu.val[1] = [a.H, b.H, c.H, d.H]
        const uint32x4x2_t vu4 = vuzpq_u32(
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w4, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w5, k, kc_odd, kzp)))),
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w6, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w7, k, kc_odd, kzp)))));
        const uint8x16_t vl4 = veorq_u8(vreinterpretq_u8_u32(vu4.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh4 = veorq_u8(vreinterpretq_u8_u32(vu4.val[1]), vkzp);
        const uint8x16x2_t vp4 = xnn_packed2planar(vl4, vh4);
        vst1q_u8(out + 32, vp4.val[0]);
        vst1q_u8(out + 48, vp4.val[1]);
        vacc4 = vpadalq_s16(vacc4, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl4), vh4)));
        // vu.val[0] = [a.L, b.L, c.L, d.L], vu.val[1] = [a.H, b.H, c.H, d.H]
        const uint32x4x2_t vu8 = vuzpq_u32(
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w8, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w9, k, kc_odd, kzp)))),
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w10, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w11, k, kc_odd, kzp)))));
        const uint8x16_t vl8 = veorq_u8(vreinterpretq_u8_u32(vu8.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh8 = veorq_u8(vreinterpretq_u8_u32(vu8.val[1]), vkzp);
        const uint8x16x2_t vp8 = xnn_packed2planar(vl8, vh8);
        vst1q_u8(out + 64, vp8.val[0]);
        vst1q_u8(out + 80, vp8.val[1]);
        vacc8 = vpadalq_s16(vacc8, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl8), vh8)));
        // vu.val[0] = [a.L, b.L, c.L, d.L], vu.val[1] = [a.H, b.H, c.H, d.H]
        const uint32x4x2_t vu12 = vuzpq_u32(
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w12, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w13, k, kc_odd, kzp)))),
            vreinterpretq_u32_u64(vcombine_u64(vcreate_u64(safe_load_u64(w14, k, kc_odd, kzp)), vcreate_u64(safe_load_u64(w15, k, kc_odd, kzp)))));
        const uint8x16_t vl12 = veorq_u8(vreinterpretq_u8_u32(vu12.val[0]), vkzp);  // uint4 -> int4
        const uint8x16_t vh12 = veorq_u8(vreinterpretq_u8_u32(vu12.val[1]), vkzp);
        const uint8x16x2_t vp12 = xnn_packed2planar(vl12, vh12);
        vst1q_u8(out + 96, vp12.val[0]);
        vst1q_u8(out + 112, vp12.val[1]);
        vacc12 = vpadalq_s16(vacc12, vpaddlq_s8(xnn_nibble_acc_s4(xnn_nibble_acc_s4(vdupq_n_s8(0), vl12), vh12)));

        out += 128;
      }

      // Subtract ksum * input_zero_point from bias
      vacc0 = vaddq_s32(vacc0, vcombine_s32(
          vadd_s32(vget_low_s32(vaccp0), vget_high_s32(vaccp0)),
          vadd_s32(vget_low_s32(vaccp2), vget_high_s32(vaccp2))));
      vacc4 = vaddq_s32(vacc4, vcombine_s32(
          vadd_s32(vget_low_s32(vaccp4), vget_high_s32(vaccp4)),
          vadd_s32(vget_low_s32(vaccp6), vget_high_s32(vaccp6))));
      vacc8 = vaddq_s32(vacc8, vcombine_s32(
          vadd_s32(vget_low_s32(vaccp8), vget_high_s32(vaccp8)),
          vadd_s32(vget_low_s32(vaccp10), vget_high_s32(vaccp10))));
      vacc12 = vaddq_s32(vacc12, vcombine_s32(
          vadd_s32(vget_low_s32(vaccp12), vget_high_s32(vaccp12)),
          vadd_s32(vget_low_s32(vaccp14), vget_high_s32(vaccp14))));
      vst1q_s32(packed_b + 0, vmlsq_s32(vld1q_s32(packed_b + 0), vacc0, vzeropoint));
      vst1q_s32(packed_b + 4, vmlsq_s32(vld1q_s32(packed_b + 4), vacc4, vzeropoint));
      vst1q_s32(packed_b + 8, vmlsq_s32(vld1q_s32(packed_b + 8), vacc8, vzeropoint));
      vst1q_s32(packed_b + 12, vmlsq_s32(vld1q_s32(packed_b + 12), vacc12, vzeropoint));

      out = (uint8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights = (const uint8_t*) ((intptr_t) weights + nc * w_stride);
  } while (--g != 0);
}
