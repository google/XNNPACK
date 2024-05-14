// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/scalar-rational-9-6.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>

void xnn_f32_vtanh_ukernel__scalar_rational_9_6_u1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  const float max_x = 7.623543739319f;
  const float min_x = -7.623543739319f;
  
  // The monomial coefficients of the numerator polynomial (odd).
  const float alpha_1 = -9.022999554873e-03f;
  const float alpha_3 = -1.146968104877e-03f;
  const float alpha_5 = -2.432360815874e-05f;
  const float alpha_7 = -6.458659385089e-08f;
  const float alpha_9 = 5.535878699892e-11f;

  // The monomial coefficients of the denominator polynomial (even).
  const float beta_0 = -9.023001417518e-03f;
  const float beta_2 = -4.154618829489e-03f;
  const float beta_4 = -2.061512641376e-04f;
  const float beta_6 = -1.774490101525e-06f;
  
  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    float x = *input;
    input++;

    // Clamp the inputs to the interpolation range. Note that we don't use
    //`fminf` or `fmaxf` since they let `NaN`s through.
    x = max_x < x ? max_x : x;
    x = x < min_x ? min_x : x;

    // Since the polynomials are odd/even, we need x^2.
    const float x2 = x * x;

    // Evaluate the numerator polynomial p.
    float p = x2 * alpha_9 + alpha_7;
    p = x2 * p + alpha_5;
    p = x2 * p + alpha_3;
    p = x2 * p + alpha_1;
    p = x * p;

    // Evaluate the denominator polynomial q.
    float q = x2 * beta_6 + beta_4;
    q = x2 * q + beta_2;
    q = x2 * q + beta_0;

    // Divide the numerator by the denominator.
    const float y =  p / q;

    *output = y;
    output++;
  }
}
