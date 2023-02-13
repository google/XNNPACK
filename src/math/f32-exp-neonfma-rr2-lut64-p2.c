// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 64) values, k = 0..63
extern XNN_INTERNAL const float xnn_table_exp2_k_over_64[64];

void xnn_math_f32_exp__neonfma_rr2_lut64_p2(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p23f);
  // The smallest x for which expf(x) is non-zero.
  const float32x4_t vzero_cutoff = vmovq_n_f32(-0x1.9FE368p6f);
  // The largest x for which expf(x) is finite.
  const float32x4_t vinf_cutoff = vmovq_n_f32(0x1.62E42Ep6f);
  const float32x4_t vlog2e_x64  = vmovq_n_f32(0x1.715476p6f);
  const float32x4_t vminus_ln2_o64_hi = vmovq_n_f32(-0x1.62e43p-7f);
  const float32x4_t vminus_ln2_o64_lo = vmovq_n_f32(0x1.05c61p-35f);
  const float32x4_t vplus_inf = vmovq_n_f32(INFINITY);

  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFF0Ap-2f);

  const int32x4_t vmin_exponent = vmovq_n_s32(INT32_C(0xC1000000));
  const int32x4_t vmax_exponent = vmovq_n_s32(INT32_C(0x3F800000));
  const int32x4_t vdefault_exponent = vmax_exponent;
  const int32x4_t vindex_mask = vmovq_n_s32(INT32_C(0x3F));

  for (; n != 0; n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // Compute reduced argument n := round(x * 64 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but that's ok, because
    // inputs outside of [-103.97207, 88.72283] underflow or overflow expf(x) anyway. We fixup the result for such
    // inputs at the very end of the algorithm.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e_x64);

    // Create two floating-point numbers, sn (scale, normal) and so (scale, overflow) such that sn * so == 2**n
    // for inputs which don't cause overflow, i.e. -103.97207 <= x <= 88.72283, and -150 <= n <= 128 accordingly.
    // We need to use two numbers rather than one because a normalized single-precision exponent must be in [-127, 126]
    // range, which is insufficient to cover [-150, 128] range of n.
    // - When n is within [-127, 126], sn == 2**n and so == 1.0.
    // - When n < -127, sn == 2**(-127) and so == 2**(n + 127).
    // - When n > 126, sn == 2**126 and so == 2**(n - 126).
    // While we explicitly compute sn, the so is fused into the value l fetched from a table by adjusting its exponential.
    int32x4_t veo = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17);
    int32x4_t ven = vmaxq_s32(veo, vmin_exponent);
    ven = vminq_s32(ven, vmax_exponent);
    veo = vsubq_s32(veo, ven);
    const float32x4_t vsn = vreinterpretq_f32_s32(vaddq_s32(ven, vdefault_exponent));

    // Use the low 6 bits of n (as integer) for table lookup.
    const uint64x2_t vidx = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask), 2));
    const uint64_t vidx01 = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx23 = vgetq_lane_u64(vidx, 1);
    float32x2_t vl01 = vld1_dup_f32((const float*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidx01));
    float32x2_t vl23 = vld1_dup_f32((const float*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidx23));
    vl01 = vld1_lane_f32((const float*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidx01 >> 32)), vl01, 1);
    vl23 = vld1_lane_f32((const float*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidx23 >> 32)), vl23, 1);
    float32x4_t vl = vcombine_f32(vl01, vl23);
    // Fuse so into the value l fetched from a table by adjusting its exponential.
    vl = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), veo));

    // Subtract the large number back to get final n := round(x * 64 / log(2)).
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2) / 64.
    // Use Cody-Waite range reduction method (note two constants to represent log(2) / 64) to improve accuracy.
    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2_o64_hi);
    vt = vfmaq_f32(vt, vn, vminus_ln2_o64_lo);

    // Compute degree-2 polynomial approximation for exp(t) on [-log(2)/128, log(2)/128].
    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vfmaq_f32(vt, vt, vp);

    // Reconstruct the final f value:
    //   f = sn * (so * l) * (1 + t * (1 + t * c2))
    //     = sn * (so * l) * (1 + t + t * (t * c2))
    //     = sn * ((so * l) + (so * l) * (t + t * (t * c2)))
    float32x4_t vf = vfmaq_f32(vl, vl, vp);
    vf = vmulq_f32(vf, vsn);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vzero_cutoff)));
    // For inputs above inf cutoff, replace output with +inf.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vbslq_f32(vcgtq_f32(vx, vinf_cutoff), vplus_inf, vf);
    vst1q_f32(output, vf); output += 4;
  }
}
