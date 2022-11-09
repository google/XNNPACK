// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math.h>
#include <xnnpack/requantization-stubs.h>


void xnn_qs8_requantize_rndnu__neon_qdmulh(
    size_t n,
    const int32_t* input,
    float scale,
    int8_t zero_point,
    int8_t qmin,
    int8_t qmax,
    int8_t* output)
{
  assert(n % 16 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const uint32_t scale_bits = float_as_uint32(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (float_as_uint32(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  /* Split shift into pre_shift + post_shift, post_shift in [1, 31] range */
  const int32_t post_shift = math_max_s32(shift, 1);
  const int32_t pre_shift = shift - post_shift;

  const int32x4_t vmultiplier = vdupq_n_s32(multiplier);
  const int16x8_t vzero_point = vdupq_n_s16((int16_t) zero_point);
  const int32x4_t vpre_shift = vdupq_n_s32(-pre_shift);
  const int32x4_t vpost_shift = vdupq_n_s32(-post_shift);
  const int8x16_t vqmin = vdupq_n_s8(qmin);
  const int8x16_t vqmax = vdupq_n_s8(qmax);
  for (; n != 0; n -= 16) {
    const int32x4_t x = vld1q_s32(input);
    const int32x4_t y = vld1q_s32(input + 4);
    const int32x4_t z = vld1q_s32(input + 8);
    const int32x4_t w = vld1q_s32(input + 12);
    input += 16;

    const int32x4_t x_preshifted = vshlq_s32(x, vpre_shift);
    const int32x4_t y_preshifted = vshlq_s32(y, vpre_shift);
    const int32x4_t z_preshifted = vshlq_s32(z, vpre_shift);
    const int32x4_t w_preshifted = vshlq_s32(w, vpre_shift);

    const int32x4_t x_product = vqdmulhq_s32(x_preshifted, vmultiplier);
    const int32x4_t y_product = vqdmulhq_s32(y_preshifted, vmultiplier);
    const int32x4_t z_product = vqdmulhq_s32(z_preshifted, vmultiplier);
    const int32x4_t w_product = vqdmulhq_s32(w_preshifted, vmultiplier);

    const int32x4_t x_scaled = vrshlq_s32(x_product, vpost_shift);
    const int32x4_t y_scaled = vrshlq_s32(y_product, vpost_shift);
    const int32x4_t z_scaled = vrshlq_s32(z_product, vpost_shift);
    const int32x4_t w_scaled = vrshlq_s32(w_product, vpost_shift);

#ifdef __aarch64__
    const int16x8_t xy_packed = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(x_scaled), y_scaled), vzero_point);
    const int16x8_t zw_packed = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(z_scaled), w_scaled), vzero_point);
    const int8x16_t xyzw_packed = vqmovn_high_s16(vqmovn_s16(xy_packed), zw_packed);
#else
    const int16x8_t xy_packed = vqaddq_s16(vcombine_s16(vqmovn_s32(x_scaled), vqmovn_s32(y_scaled)), vzero_point);
    const int16x8_t zw_packed = vqaddq_s16(vcombine_s16(vqmovn_s32(z_scaled), vqmovn_s32(w_scaled)), vzero_point);
    const int8x16_t xyzw_packed = vcombine_s8(vqmovn_s16(xy_packed), vqmovn_s16(zw_packed));
#endif

    const int8x16_t xyzw_clamped = vmaxq_s8(vminq_s8(xyzw_packed, vqmax), vqmin);

    // AArch32 version:
    //   4x VSHL.S32 Qd, Qm, Qn
    //   4x VQDMULH.S32 Qd, Qm, Qn
    //   4x VRSHL.S32 Qd, Qm, Qn
    //   4x VQMOVN.S32 Dd, Qm
    //   2x VQADD.S16 Qd, Qm, Qn
    //   2x VQMOVUN.S16 Dd, Qm
    //   1x VMAX.U8 Qd, Qm, Qn
    //   1x VMIN.U8 Qd, Qm, Qn
    // ---------------------
    // 22 instructions total
    //
    // AArch64 version:
    //   4x SSHL Vd.4S, Vn.4S, Vm.4S
    //   4x SQDMULH Vd.4S, Vn.4S, Vm.4S
    //   4x SRSHL 4d.4S, Vn.4S, Vm.4S
    //   2x SQXTN Vd.4H, Vn.4S
    //   2x SQXTN2 Vd.8H, Vn.4S
    //   2x SQADD Vd.8H, Vn.8H, Vm.8H
    //   1x SQXTN Vd.8B, Vn.8H
    //   1x SQXTN2 Vd.16B, Vn.8H
    //   1x SMIN Vd.16B, Vn.16B, Vm.16B
    //   1x SMAX Vd.16B, Vn.16B, Vm.16B
    // ---------------------
    // 22 instructions total

    vst1q_s8(output, xyzw_clamped);
    output += 16;
  }
}
