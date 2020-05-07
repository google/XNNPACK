// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundne__neon(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Addition of this number to a floating-point number x cause rounding of the result to an integer. Then this magic
  // number is subtracted back from the result to get original x rounded to integer. This trick works only for
  // 0 <= x < 2**24, but all numbers in 2**23 <= x < 2**24 range are integers, so we can further restrict it to
  // 0 <= x < 2**23. Then the upper bound of the validity interval is conveniently the same as the magic number.
  const float32x4_t vmagic_number = vmovq_n_f32(0x1.000000p+23f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // The rounding trick works only for x >= 0, so we compute absolute value of x, round it, and restore the sign in
    // the end. This method works for round-to-nearest-even because it is an odd function.
    const float32x4_t vabsx = vabsq_f32(vx);
    // Compute bitmask for selection between the value rounded with addition-subtraction trick and the abs(x) value.
    // We use the result of the addition-subtraction trick only on its validity interval, i.e. 0 <= abs(x) < 2**23.
    // Note: we do vcageq_f32(vmagic_number, vx) instead of vcgeq_f32(vmagic_number, vabsx) to reduce dependency chain.
    const uint32x4_t vrndmask = vcageq_f32(vmagic_number, vx);

    // Addition-subtraction trick with the magic number to cause rounding to integer for abs(x).
    // Note: the result is valid only for 0 <= abs(x) < 2**23.
    const float32x4_t vrndabsx = vsubq_f32(vaddq_f32(vabsx, vmagic_number), vmagic_number);
    // Compute bitmask for the sign of x.
    // The bitmask is 0x00000000 when x is positive (including +0) and 0x80000000 when x is negative (including -0).
    const uint32x4_t vsignx = veorq_u32(vreinterpretq_u32_f32(vabsx), vreinterpretq_u32_f32(vx));

    // Combine abs(x) rounded via addition-subtraction trick and the input x value.
    // For abs(x) < 2**23, the result is abs(x) rounded via addition-subtraction trick with the sign of x.
    // For abs(x) >= 2**23 and NaN inputs, the result is x itself.
    const float32x4_t vy = vbslq_f32(vbicq_u32(vrndmask, vsignx), vrndabsx, vx);

    vst1q_f32(output, vy); output += 4;
  }
}
