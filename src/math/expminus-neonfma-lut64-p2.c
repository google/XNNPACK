// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


// Table of exp2(k / 64) values, k = 0..63
static const float exp2_k_over_64_table[64] = {
  0x1.000000p+0f, 0x1.02C9A4p+0f, 0x1.059B0Ep+0f, 0x1.087452p+0f,
  0x1.0B5586p+0f, 0x1.0E3EC4p+0f, 0x1.11301Ep+0f, 0x1.1429AAp+0f,
  0x1.172B84p+0f, 0x1.1A35BEp+0f, 0x1.1D4874p+0f, 0x1.2063B8p+0f,
  0x1.2387A6p+0f, 0x1.26B456p+0f, 0x1.29E9E0p+0f, 0x1.2D285Ap+0f,
  0x1.306FE0p+0f, 0x1.33C08Cp+0f, 0x1.371A74p+0f, 0x1.3A7DB4p+0f,
  0x1.3DEA64p+0f, 0x1.4160A2p+0f, 0x1.44E086p+0f, 0x1.486A2Cp+0f,
  0x1.4BFDAEp+0f, 0x1.4F9B28p+0f, 0x1.5342B6p+0f, 0x1.56F474p+0f,
  0x1.5AB07Ep+0f, 0x1.5E76F2p+0f, 0x1.6247ECp+0f, 0x1.662388p+0f,
  0x1.6A09E6p+0f, 0x1.6DFB24p+0f, 0x1.71F75Ep+0f, 0x1.75FEB6p+0f,
  0x1.7A1148p+0f, 0x1.7E2F34p+0f, 0x1.82589Ap+0f, 0x1.868D9Ap+0f,
  0x1.8ACE54p+0f, 0x1.8F1AEAp+0f, 0x1.93737Cp+0f, 0x1.97D82Ap+0f,
  0x1.9C4918p+0f, 0x1.A0C668p+0f, 0x1.A5503Cp+0f, 0x1.A9E6B6p+0f,
  0x1.AE89FAp+0f, 0x1.B33A2Cp+0f, 0x1.B7F770p+0f, 0x1.BCC1EAp+0f,
  0x1.C199BEp+0f, 0x1.C67F12p+0f, 0x1.CB720Ep+0f, 0x1.D072D4p+0f,
  0x1.D5818Ep+0f, 0x1.DA9E60p+0f, 0x1.DFC974p+0f, 0x1.E502EEp+0f,
  0x1.EA4AFAp+0f, 0x1.EFA1BEp+0f, 0x1.F50766p+0f, 0x1.FA7C18p+0f,
};

void xnn_math_f32_expminus__neonfma_lut64_p2(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p23f);
  // The smallest x for which expf(x) is normalized.
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(-0x1.5D589Ep6f);
  const float32x4_t vlog2e_x64  = vmovq_n_f32(0x1.715476p6f);
  const float32x4_t vminus_ln2_o64_hi = vmovq_n_f32(-0x1.62e43p-7f);
  const float32x4_t vminus_ln2_o64_lo = vmovq_n_f32(0x1.05c61p-35f);

  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFF0Ap-2f);

  const int32x4_t vindex_mask = vmovq_n_s32(INT32_C(0x3F));

  for (; n != 0; n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // Compute reduced argument n := round(x * 64 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x * 64 / log(2)| <= 2**22, i.e.
    // |x| <= 0x1.62E43p+15 = 45426.09375), but that is acceptable, because inputs outside of [-87.336540, 0.0]
    // result in denormalized or underflown expf(x). We fixup the result for such inputs at the very end of the
    // algorithm.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e_x64);

    // Create a floating-point number s (scale) such that s := 2**(n / 64) for such inputs that expf(x) is normalized,
    // i.e. -87.33642 <= x <= 0.0. As n has 6 fractional bits, we split s == 2**(n / 64) = 2**e * 2**(n / 64 - e), where
    // e := int(n / 64). We create s in two steps:
    // 1. Fetch 2**(n / 64 - e) = 2**(n % 64) from exp2_k_over_64_table using the 6 low bits of n, as integer. Note that the
    //    fetched values are in the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of e to its floating-point exponent. The result is always a normalized
    //    number, because for -87.33642 <= x <= 0.0 (inputs for which expf(x) is normalized) we have -126 <= e <= 0,
    //    and thus the adjusted exponent is not lower than -126.
    //
    // Extract e from bits 6:14 of n and shift it into bits 23:31 (position of floating-point exponent).
    const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17);

    // Use bits 0:6 bits of n, as integer, as an index for table lookup of l := 2**(n % 64).
    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx01 = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx23 = vgetq_lane_u64(vidx, 1);
    float32x2_t vl01 = vld1_dup_f32(&exp2_k_over_64_table[(uint32_t) vidx01]);
    float32x2_t vl23 = vld1_dup_f32(&exp2_k_over_64_table[(uint32_t) vidx23]);
    vl01 = vld1_lane_f32(&exp2_k_over_64_table[(uint32_t) (vidx01 >> 32)], vl01, 1);
    vl23 = vld1_lane_f32(&exp2_k_over_64_table[(uint32_t) (vidx23 >> 32)], vl23, 1);
    const float32x4_t vl = vcombine_f32(vl01, vl23);
    // Adjust exponent of the value l fetched from the exp2_k_over_64_table to get the final s value.
    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));

    // Subtract the large number back to get final n := round(x * 64 / log(2)) as a floating-point number.
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2) / 64.
    // Use Cody-Waite range reduction method (note the two constants representing log(2) / 64) to improve accuracy.
    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2_o64_hi);
    vt = vfmaq_f32(vt, vn, vminus_ln2_o64_lo);

    // Compute degree-2 polynomial approxiatmion for exp(t) on [-log(2)/128, log(2)/128].
    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vfmaq_f32(vt, vt, vp);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (1 + t * c2))
    //     = s * (1 + t + t * (t * c2))
    //     = s + s * (t + t * (t * c2))
    //     = s + s * p
    float32x4_t vf = vfmaq_f32(vs, vs, vp);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vdenorm_cutoff)));
    vst1q_f32(output, vf); output += 4;
  }
}
