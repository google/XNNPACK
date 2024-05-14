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

void xnn_f32_vtanh_ukernel__scalar_rational_9_6_u8(
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
  
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float x_0 = *input++;
    float x_1 = *input++;
    float x_2 = *input++;
    float x_3 = *input++;
    float x_4 = *input++;
    float x_5 = *input++;
    float x_6 = *input++;
    float x_7 = *input++;

    // Clamp the inputs to the interpolation range. Note that we don't use
    //`fminf` or `fmaxf` since they let `NaN`s through.
    x_0 = max_x < x_0 ? max_x : x_0;
    x_1 = max_x < x_1 ? max_x : x_1;
    x_2 = max_x < x_2 ? max_x : x_2;
    x_3 = max_x < x_3 ? max_x : x_3;
    x_4 = max_x < x_4 ? max_x : x_4;
    x_5 = max_x < x_5 ? max_x : x_5;
    x_6 = max_x < x_6 ? max_x : x_6;
    x_7 = max_x < x_7 ? max_x : x_7;
    x_0 = x_0 < min_x ? min_x : x_0;
    x_1 = x_1 < min_x ? min_x : x_1;
    x_2 = x_2 < min_x ? min_x : x_2;
    x_3 = x_3 < min_x ? min_x : x_3;
    x_4 = x_4 < min_x ? min_x : x_4;
    x_5 = x_5 < min_x ? min_x : x_5;
    x_6 = x_6 < min_x ? min_x : x_6;
    x_7 = x_7 < min_x ? min_x : x_7;

    // Since the polynomials are odd/even, we need x^2.
    const float x2_0 = x_0 * x_0;
    const float x2_1 = x_1 * x_1;
    const float x2_2 = x_2 * x_2;
    const float x2_3 = x_3 * x_3;
    const float x2_4 = x_4 * x_4;
    const float x2_5 = x_5 * x_5;
    const float x2_6 = x_6 * x_6;
    const float x2_7 = x_7 * x_7;

    // Evaluate the numerator polynomial p.
    float p_0 = x2_0 * alpha_9 + alpha_7;
    float p_1 = x2_1 * alpha_9 + alpha_7;
    float p_2 = x2_2 * alpha_9 + alpha_7;
    float p_3 = x2_3 * alpha_9 + alpha_7;
    float p_4 = x2_4 * alpha_9 + alpha_7;
    float p_5 = x2_5 * alpha_9 + alpha_7;
    float p_6 = x2_6 * alpha_9 + alpha_7;
    float p_7 = x2_7 * alpha_9 + alpha_7;
    p_0 = x2_0 * p_0 + alpha_5;
    p_1 = x2_1 * p_1 + alpha_5;
    p_2 = x2_2 * p_2 + alpha_5;
    p_3 = x2_3 * p_3 + alpha_5;
    p_4 = x2_4 * p_4 + alpha_5;
    p_5 = x2_5 * p_5 + alpha_5;
    p_6 = x2_6 * p_6 + alpha_5;
    p_7 = x2_7 * p_7 + alpha_5;
    p_0 = x2_0 * p_0 + alpha_3;
    p_1 = x2_1 * p_1 + alpha_3;
    p_2 = x2_2 * p_2 + alpha_3;
    p_3 = x2_3 * p_3 + alpha_3;
    p_4 = x2_4 * p_4 + alpha_3;
    p_5 = x2_5 * p_5 + alpha_3;
    p_6 = x2_6 * p_6 + alpha_3;
    p_7 = x2_7 * p_7 + alpha_3;
    p_0 = x2_0 * p_0 + alpha_1;
    p_1 = x2_1 * p_1 + alpha_1;
    p_2 = x2_2 * p_2 + alpha_1;
    p_3 = x2_3 * p_3 + alpha_1;
    p_4 = x2_4 * p_4 + alpha_1;
    p_5 = x2_5 * p_5 + alpha_1;
    p_6 = x2_6 * p_6 + alpha_1;
    p_7 = x2_7 * p_7 + alpha_1;
    p_0 = x_0 * p_0;
    p_1 = x_1 * p_1;
    p_2 = x_2 * p_2;
    p_3 = x_3 * p_3;
    p_4 = x_4 * p_4;
    p_5 = x_5 * p_5;
    p_6 = x_6 * p_6;
    p_7 = x_7 * p_7;

    // Evaluate the denominator polynomial q.
    float q_0 = x2_0 * beta_6 + beta_4;
    float q_1 = x2_1 * beta_6 + beta_4;
    float q_2 = x2_2 * beta_6 + beta_4;
    float q_3 = x2_3 * beta_6 + beta_4;
    float q_4 = x2_4 * beta_6 + beta_4;
    float q_5 = x2_5 * beta_6 + beta_4;
    float q_6 = x2_6 * beta_6 + beta_4;
    float q_7 = x2_7 * beta_6 + beta_4;
    q_0 = x2_0 * q_0 + beta_2;
    q_1 = x2_1 * q_1 + beta_2;
    q_2 = x2_2 * q_2 + beta_2;
    q_3 = x2_3 * q_3 + beta_2;
    q_4 = x2_4 * q_4 + beta_2;
    q_5 = x2_5 * q_5 + beta_2;
    q_6 = x2_6 * q_6 + beta_2;
    q_7 = x2_7 * q_7 + beta_2;
    q_0 = x2_0 * q_0 + beta_0;
    q_1 = x2_1 * q_1 + beta_0;
    q_2 = x2_2 * q_2 + beta_0;
    q_3 = x2_3 * q_3 + beta_0;
    q_4 = x2_4 * q_4 + beta_0;
    q_5 = x2_5 * q_5 + beta_0;
    q_6 = x2_6 * q_6 + beta_0;
    q_7 = x2_7 * q_7 + beta_0;

    // Divide the numerator by the denominator.
    const float y_0 =  p_0 / q_0;
    const float y_1 =  p_1 / q_1;
    const float y_2 =  p_2 / q_2;
    const float y_3 =  p_3 / q_3;
    const float y_4 =  p_4 / q_4;
    const float y_5 =  p_5 / q_5;
    const float y_6 =  p_6 / q_6;
    const float y_7 =  p_7 / q_7;

    *output++ = y_0;
    *output++ = y_1;
    *output++ = y_2;
    *output++ = y_3;
    *output++ = y_4;
    *output++ = y_5;
    *output++ = y_6;
    *output++ = y_7;
  }
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
