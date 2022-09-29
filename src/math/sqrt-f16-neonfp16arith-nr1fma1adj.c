// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(__fp16)) == 0);

  // Smallest positive normalized number.
  const float16x8_t vsubnormal_threshold = vmovq_n_f16(0x1.0p-14f);
  // Positive infininity in bit representation.
  const int16x8_t vpositive_infinity = vmovq_n_s16(INT16_C(0x7C00));
  // 0.5f constant used in Newton-Raphson iterations.
  const float16x8_t vhalf = vmovq_n_f16(0.5f);
  // Mask for the top 4 exponent bits of a IEEE FP16 number.
  const uint16x8_t vexp4_mask = vmovq_n_u16(UINT16_C(0x7800));
  // 2 * as_bits(1.0h) - as_bits(0.5h)
  const uint16x8_t vscale_mask = vmovq_n_u16(UINT16_C(16384));
  // Mask for the sign bit of a IEEE FP16 number.
  const uint16x8_t vsign_mask = vmovq_n_u16(UINT16_C(0x8000));

  const __fp16* i = (const __fp16*) input;
  __fp16* o = (__fp16*) output;
  for (; n != 0; n -= 8 * sizeof(__fp16)) {
    const float16x8_t vi = vld1q_f16(i); i += 8;

    // Mask for subnormal inputs, both positive and negative. Results for such inputs must be replaced with +-0.
    const uint16x8_t vsubnormal_mask = vcaltq_f16(vi, vsubnormal_threshold);
    // Mask for positive infininty and NaN inputs. Results for such inputs are replaced with the input itself.
    const uint16x8_t vspecial_mask = vcgeq_s16(vreinterpretq_s16_f16(vi), vpositive_infinity);

    // Replace the top four bits of exponent with 0b0111 to avoid underflow in computations.
    const float16x8_t vx = vbslq_f16(vexp4_mask, vhalf, vi);
    // Extract the high 4 bits of inputs's exponent.
    const uint16x8_t vexp4i = vandq_u16(vreinterpretq_u16_f16(vi), vexp4_mask);
    // Create floating-point scale to apply to the final result to restore the correct exponent.
    const float16x8_t vpostscale = vreinterpretq_f16_u16(vhaddq_u16(vexp4i, vscale_mask));

    // Initial approximation
    const float16x8_t vrsqrtx = vrsqrteq_f16(vx);
    float16x8_t vsqrtx = vmulq_f16(vrsqrtx, vx);
    const float16x8_t vhalfrsqrtx = vmulq_f16(vrsqrtx, vhalf);

    // Netwon-Raphson iteration:
    //   residual   <- 0.5 - sqrtx * halfrsqrtx
    //   sqrtx      <- sqrtx + sqrtx * residual
    const float16x8_t vresidual = vfmsq_f16(vhalf, vsqrtx, vhalfrsqrtx);
    vsqrtx = vfmaq_f16(vsqrtx, vresidual, vsqrtx);

    // Final adjustment:
    //   adjustment <- x - sqrtx * sqrtx
    //   sqrtx      <- sqrtx + halfrsqrtx * adjustment
    const float16x8_t vadjustment = vfmsq_f16(vx, vsqrtx, vsqrtx);
    vsqrtx = vfmaq_f16(vsqrtx, vhalfrsqrtx, vadjustment);

    // Apply exponent adjustment. Use multiplication to propagate NaNs for negative inputs and flush results to zero for subnormal inputs.
    float16x8_t vy = vmulq_f16(vsqrtx, vpostscale);

    // Replace results for positive infinity and NaN inputs with the input itself.
    vy = vbslq_f16(vspecial_mask, vi, vy);

    // Replace results for subnormal inputs with signed zero.
    const float16x8_t vsigni = vreinterpretq_f16_u16(vandq_u16(vreinterpretq_u16_f16(vi), vsign_mask));
    vy = vbslq_f16(vsubnormal_mask, vsigni, vy);

    vst1q_f16(o, vy); o += 8;
  }
}
