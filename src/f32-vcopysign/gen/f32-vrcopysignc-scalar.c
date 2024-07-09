// Auto-generated file. Do not edit!
//   Template: src/f32-vcopysign/rcopysignc.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/simd/f32-scalar.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_f32_vrcopysignc_ukernel__scalar_u1(
    size_t batch,
    const float* sign,
    const float* mag,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(sign != NULL);
  assert(mag != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vmag = xnn_abs_f32(xnn_set1_f32(*mag));

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vsign = xnn_loadu_f32(sign);
    sign += xnn_simd_size_f32;

    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vrcopysignc_ukernel__scalar_u2(
    size_t batch,
    const float* sign,
    const float* mag,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(sign != NULL);
  assert(mag != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vmag = xnn_abs_f32(xnn_set1_f32(*mag));

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    xnn_simd_f32_t vsign_0 = xnn_loadu_f32(sign);
    xnn_simd_f32_t vsign_1 = xnn_loadu_f32(sign + 1 * xnn_simd_size_f32);
    sign += 2;

    vsign_0 = xnn_and_f32(vsign_0, vsign_mask);
    vsign_1 = xnn_and_f32(vsign_1, vsign_mask);

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign_0, vmag);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign_1, vmag);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 2;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vsign = xnn_loadu_f32(sign);
    sign += xnn_simd_size_f32;

    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vrcopysignc_ukernel__scalar_u4(
    size_t batch,
    const float* sign,
    const float* mag,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(sign != NULL);
  assert(mag != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vmag = xnn_abs_f32(xnn_set1_f32(*mag));

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    xnn_simd_f32_t vsign_0 = xnn_loadu_f32(sign);
    xnn_simd_f32_t vsign_1 = xnn_loadu_f32(sign + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vsign_2 = xnn_loadu_f32(sign + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vsign_3 = xnn_loadu_f32(sign + 3 * xnn_simd_size_f32);
    sign += 4;

    vsign_0 = xnn_and_f32(vsign_0, vsign_mask);
    vsign_1 = xnn_and_f32(vsign_1, vsign_mask);
    vsign_2 = xnn_and_f32(vsign_2, vsign_mask);
    vsign_3 = xnn_and_f32(vsign_3, vsign_mask);

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign_0, vmag);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign_1, vmag);
    xnn_simd_f32_t vy_2 = xnn_or_f32(vsign_2, vmag);
    xnn_simd_f32_t vy_3 = xnn_or_f32(vsign_3, vmag);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 4;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vsign = xnn_loadu_f32(sign);
    sign += xnn_simd_size_f32;

    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}

void xnn_f32_vrcopysignc_ukernel__scalar_u8(
    size_t batch,
    const float* sign,
    const float* mag,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(sign != NULL);
  assert(mag != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vmag = xnn_abs_f32(xnn_set1_f32(*mag));

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    xnn_simd_f32_t vsign_0 = xnn_loadu_f32(sign);
    xnn_simd_f32_t vsign_1 = xnn_loadu_f32(sign + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vsign_2 = xnn_loadu_f32(sign + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vsign_3 = xnn_loadu_f32(sign + 3 * xnn_simd_size_f32);
    xnn_simd_f32_t vsign_4 = xnn_loadu_f32(sign + 4 * xnn_simd_size_f32);
    xnn_simd_f32_t vsign_5 = xnn_loadu_f32(sign + 5 * xnn_simd_size_f32);
    xnn_simd_f32_t vsign_6 = xnn_loadu_f32(sign + 6 * xnn_simd_size_f32);
    xnn_simd_f32_t vsign_7 = xnn_loadu_f32(sign + 7 * xnn_simd_size_f32);
    sign += 8;

    vsign_0 = xnn_and_f32(vsign_0, vsign_mask);
    vsign_1 = xnn_and_f32(vsign_1, vsign_mask);
    vsign_2 = xnn_and_f32(vsign_2, vsign_mask);
    vsign_3 = xnn_and_f32(vsign_3, vsign_mask);
    vsign_4 = xnn_and_f32(vsign_4, vsign_mask);
    vsign_5 = xnn_and_f32(vsign_5, vsign_mask);
    vsign_6 = xnn_and_f32(vsign_6, vsign_mask);
    vsign_7 = xnn_and_f32(vsign_7, vsign_mask);

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign_0, vmag);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign_1, vmag);
    xnn_simd_f32_t vy_2 = xnn_or_f32(vsign_2, vmag);
    xnn_simd_f32_t vy_3 = xnn_or_f32(vsign_3, vmag);
    xnn_simd_f32_t vy_4 = xnn_or_f32(vsign_4, vmag);
    xnn_simd_f32_t vy_5 = xnn_or_f32(vsign_5, vmag);
    xnn_simd_f32_t vy_6 = xnn_or_f32(vsign_6, vmag);
    xnn_simd_f32_t vy_7 = xnn_or_f32(vsign_7, vmag);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    xnn_storeu_f32(output + 4 * xnn_simd_size_f32, vy_4);
    xnn_storeu_f32(output + 5 * xnn_simd_size_f32, vy_5);
    xnn_storeu_f32(output + 6 * xnn_simd_size_f32, vy_6);
    xnn_storeu_f32(output + 7 * xnn_simd_size_f32, vy_7);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vsign = xnn_loadu_f32(sign);
    sign += xnn_simd_size_f32;

    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
}
