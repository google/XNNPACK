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
  assert(n % (8 * sizeof(uint16_t)) == 0);

  // Positive infininity in bit representation.
  const uint16x8_t vpositive_infinity = vmovq_n_u16(UINT16_C(0x7C00));
  // 0.5f constant used in Newton-Raphson iterations.
  const float16x8_t vhalf = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3800)));  // 0.5h
  // Mask for the top 4 exponent bits of a IEEE FP16 number.
  const uint16x8_t vexp4_mask = vmovq_n_u16(UINT16_C(0x7800));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;


    // Mask for positive infininty, NaN, and negative inputs.
    // Results for such inputs are replaced with the special values, typically with NaN.
    uint16x8_t vspecial_mask = vcgeq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);
    uint16x8_t vspecial_value = vmovq_n_u16(UINT16_C(0x7E00));

    // Mask for signed zero inputs, both positive and negative. Results for such inputs must be replaced with the input itself.
    const uint16x8_t vzero_mask = vceqq_f16(vi, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    vspecial_mask = vorrq_u16(vspecial_mask, vzero_mask);

    // Mask for positive infininty inputs. Results for such inputs are replaced with the input itself.
    const uint16x8_t vinfinity_mask = vceqq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);
    const uint16x8_t vinput_mask = vorrq_u16(vinfinity_mask, vzero_mask);
    vspecial_value = vbslq_u16(vinput_mask, vreinterpretq_u16_f16(vi), vspecial_value);

    // Replace the top four bits of exponent with 0b0111 to avoid underflow in computations.
    const float16x8_t vx = vbslq_f16(vexp4_mask, vhalf, vi);
    // Extract the high 4 bits of inputs's exponent.
    const int16x8_t vexp4i = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_f16(vi), vexp4_mask));
    // Create floating-point scale to apply to the final result to restore the correct exponent.
    const int16x8_t vpostscale = vhsubq_s16(vexp4i, vreinterpretq_s16_f16(vhalf));

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

    // Apply exponent adjustment. Use multiplication to propagate NaNs for negative inputs.
    float16x8_t vy = vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(vsqrtx), vpostscale));

    // Replace results for positive infinity and NaN inputs with the input itself.
    vy = vbslq_f16(vspecial_mask, vreinterpretq_f16_u16(vspecial_value), vy);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }
}
