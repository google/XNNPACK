// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/neon-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/raddstoreexpminusmax.h>


void xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x8(
    size_t elements,
    const float* input,
    float* output,
    float* sum,
    float max)
{
  assert(elements % sizeof(float) == 0);

  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.8000FEp23f);
  // The smallest x for which expf(x) is normalized.
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(-0x1.5D589Ep6f);
  const float32x4_t vlog2e = vmovq_n_f32(0x1.715476p+0f);
  const float32x4_t vminus_ln2_hi = vmovq_n_f32(-0x1.62E43p-1f);
  const float32x4_t vminus_ln2_lo = vmovq_n_f32(0x1.05C61p-29f);

  const float32x4_t vc1 = vmovq_n_f32(0x1.FFFFF6p-1f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFDC6p-2f);
  const float32x4_t vc3 = vmovq_n_f32(0x1.555A80p-3f);
  const float32x4_t vc4 = vmovq_n_f32(0x1.573A1Ap-5f);
  const float32x4_t vc5 = vmovq_n_f32(0x1.0F9F9Cp-7f);

  const float32x4_t vi_max = vdupq_n_f32(max);

  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  for (; elements >= 8 * sizeof(float); elements -= 8 * sizeof(float)) {
    // Load 8 (2x4) inputs at a time.
    const float32x4_t vi0123 = vld1q_f32(input); input += 4;
    const float32x4_t vi4567 = vld1q_f32(input); input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const float32x4_t vx0123 = vsubq_f32(vi0123, vi_max);
    const float32x4_t vx4567 = vsubq_f32(vi4567, vi_max);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs outside of [-87.336540, 0.0] underflow expf(x) anyway. We fixup the result for such inputs at the very end
    // of the algorithm.
    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vx0123, vlog2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vx4567, vlog2e);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= n <= 0 accordingly.
    const float32x4_t vs0123 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 23));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 23));

    // Subtract the large number back to get final n := round(x / log(2)).
    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float32x4_t vt0123 = vfmaq_f32(vx0123, vn0123, vminus_ln2_hi);
    float32x4_t vt4567 = vfmaq_f32(vx4567, vn4567, vminus_ln2_hi);

    vt0123 = vfmaq_f32(vt0123, vn0123, vminus_ln2_lo);
    vt4567 = vfmaq_f32(vt4567, vn4567, vminus_ln2_lo);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    float32x4_t vp0123 = vfmaq_f32(vc4, vc5, vt0123);
    float32x4_t vp4567 = vfmaq_f32(vc4, vc5, vt4567);

    vp0123 = vfmaq_f32(vc3, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc3, vp4567, vt4567);

    vp0123 = vfmaq_f32(vc2, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc2, vp4567, vt4567);

    vp0123 = vfmaq_f32(vc1, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc1, vp4567, vt4567);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0123 = vmulq_f32(vt0123, vs0123);
    vt4567 = vmulq_f32(vt4567, vs4567);

    float32x4_t vf0123 = vfmaq_f32(vs0123, vp0123, vt0123);
    float32x4_t vf4567 = vfmaq_f32(vs4567, vp4567, vt4567);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0123 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf0123), vcltq_f32(vx0123, vdenorm_cutoff)));
    vf4567 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf4567), vcltq_f32(vx4567, vdenorm_cutoff)));

    // Store 8 (2x4) outputs at a time.
    vst1q_f32(output, vf0123); output += 4;
    vst1q_f32(output, vf4567); output += 4;

    // Accumulate computed exponents.
    vacc0 = vaddq_f32(vacc0, vf0123);
    vacc0 = vaddq_f32(vacc0, vf4567);
  }

  float32x4_t vacc = vacc0;
  for (; elements >= 4 * sizeof(float); elements -= 4 * sizeof(float)) {
    // Load 4 inputs at a time.
    const float32x4_t vi = vld1q_f32(input); input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const float32x4_t vx = vsubq_f32(vi, vi_max);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs outside of [-87.336540, 0.0] underflow expf(x) anyway. We fixup the result for such inputs at the very end
    // of the algorithm.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= n <= 0 accordingly.
    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2_hi);
    vt = vfmaq_f32(vt, vn, vminus_ln2_lo);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    float32x4_t vp = vfmaq_f32(vc4, vc5, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmaq_f32(vc1, vp, vt);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = vmulq_f32(vt, vs);
    float32x4_t vf = vfmaq_f32(vs, vp, vt);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vdenorm_cutoff)));

    // Store 4 outputs at a time.
    vst1q_f32(output, vf); output += 4;

    // Accumulate computed exponents.
    vacc = vaddq_f32(vacc, vf);
  }
#if XNN_ARCH_ARM64
  float vacc_lo = vaddvq_f32(vacc);
#else
  float32x2_t vacc_lo = vadd_f32(vget_high_f32(vacc), vget_low_f32(vacc));
#endif
  if (elements != 0) {
    assert(elements >= 1 * sizeof(float));
    assert(elements <= 3 * sizeof(float));
    // Load 4 inputs at a time.
    const float32x4_t vi = vld1q_f32(input); input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const float32x4_t vx = vsubq_f32(vi, vi_max);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs outside of [-87.336540, 0.0] underflow expf(x) anyway. We fixup the result for such inputs at the very end
    // of the algorithm.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= n <= 0 accordingly.
    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2_hi);
    vt = vfmaq_f32(vt, vn, vminus_ln2_lo);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    float32x4_t vp = vfmaq_f32(vc4, vc5, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmaq_f32(vc1, vp, vt);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = vmulq_f32(vt, vs);
    float32x4_t vf = vfmaq_f32(vs, vp, vt);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vdenorm_cutoff)));

    float32x2_t vf_lo = vget_low_f32(vf);
    if (elements & (2 * sizeof(float))) {
      // Store 2 outputs at a time.
      vst1_f32(output, vf_lo); output += 2;

      // Accumulate 2 computed exponents.
      #if XNN_ARCH_ARM64
        vacc_lo += vaddv_f32(vf_lo);
      #else
        vacc_lo = vadd_f32(vacc_lo, vf_lo);
      #endif

      vf_lo = vget_high_f32(vf);
    }
    if (elements & (1 * sizeof(float))) {
      // Store 1 output at a time.
      vst1_lane_f32(output, vf_lo, 0);

      // Accumulate 1 computed exponent.
      #if XNN_ARCH_ARM64
        vacc_lo += vget_lane_f32(vf_lo, 0);
      #else
        vacc_lo = vadd_f32(vacc_lo, vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vf_lo), 32)));
      #endif
    }
  }
  // Reduce 4 elements in the SIMD register
#if XNN_ARCH_ARM64
  *sum = vacc_lo;
#else
  vst1_lane_f32(sum, vpadd_f32(vacc_lo, vacc_lo), 0);
#endif
}
