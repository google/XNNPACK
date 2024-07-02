// Auto-generated file. Do not edit!
//   Template: src/f32-vcopysign/copysign.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/simd/f32-avx.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_f32_vcopysign_ukernel__avx_u8(
    size_t batch,
    const float* mag,
    const float* sign,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vsign = xnn_loadu_f32(sign);
    sign += xnn_simd_size_f32;

    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vsign = xnn_load_tail_f32(sign, batch >> XNN_LOG2_SIZEOF_FLOAT);
    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vcopysign_ukernel__avx_u16(
    size_t batch,
    const float* mag,
    const float* sign,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    xnn_simd_f32_t vsign_0 = xnn_loadu_f32(sign);
    xnn_simd_f32_t vsign_1 = xnn_loadu_f32(sign + 1 * xnn_simd_size_f32);
    sign += 16;

    vsign_0 = xnn_and_f32(vsign_0, vsign_mask);
    vsign_1 = xnn_and_f32(vsign_1, vsign_mask);

    xnn_simd_f32_t vmag_0 = xnn_abs_f32(xnn_loadu_f32(mag));
    xnn_simd_f32_t vmag_1 = xnn_abs_f32(xnn_loadu_f32(mag + 1 * xnn_simd_size_f32));
    mag += 16;

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign_0, vmag_0);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign_1, vmag_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 16;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vsign = xnn_loadu_f32(sign);
    sign += xnn_simd_size_f32;

    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vsign = xnn_load_tail_f32(sign, batch >> XNN_LOG2_SIZEOF_FLOAT);
    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vcopysign_ukernel__avx_u24(
    size_t batch,
    const float* mag,
    const float* sign,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);

  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    xnn_simd_f32_t vsign_0 = xnn_loadu_f32(sign);
    xnn_simd_f32_t vsign_1 = xnn_loadu_f32(sign + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vsign_2 = xnn_loadu_f32(sign + 2 * xnn_simd_size_f32);
    sign += 24;

    vsign_0 = xnn_and_f32(vsign_0, vsign_mask);
    vsign_1 = xnn_and_f32(vsign_1, vsign_mask);
    vsign_2 = xnn_and_f32(vsign_2, vsign_mask);

    xnn_simd_f32_t vmag_0 = xnn_abs_f32(xnn_loadu_f32(mag));
    xnn_simd_f32_t vmag_1 = xnn_abs_f32(xnn_loadu_f32(mag + 1 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_2 = xnn_abs_f32(xnn_loadu_f32(mag + 2 * xnn_simd_size_f32));
    mag += 24;

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign_0, vmag_0);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign_1, vmag_1);
    xnn_simd_f32_t vy_2 = xnn_or_f32(vsign_2, vmag_2);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    output += 24;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vsign = xnn_loadu_f32(sign);
    sign += xnn_simd_size_f32;

    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vsign = xnn_load_tail_f32(sign, batch >> XNN_LOG2_SIZEOF_FLOAT);
    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vcopysign_ukernel__avx_u32(
    size_t batch,
    const float* mag,
    const float* sign,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    xnn_simd_f32_t vsign_0 = xnn_loadu_f32(sign);
    xnn_simd_f32_t vsign_1 = xnn_loadu_f32(sign + 1 * xnn_simd_size_f32);
    xnn_simd_f32_t vsign_2 = xnn_loadu_f32(sign + 2 * xnn_simd_size_f32);
    xnn_simd_f32_t vsign_3 = xnn_loadu_f32(sign + 3 * xnn_simd_size_f32);
    sign += 32;

    vsign_0 = xnn_and_f32(vsign_0, vsign_mask);
    vsign_1 = xnn_and_f32(vsign_1, vsign_mask);
    vsign_2 = xnn_and_f32(vsign_2, vsign_mask);
    vsign_3 = xnn_and_f32(vsign_3, vsign_mask);

    xnn_simd_f32_t vmag_0 = xnn_abs_f32(xnn_loadu_f32(mag));
    xnn_simd_f32_t vmag_1 = xnn_abs_f32(xnn_loadu_f32(mag + 1 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_2 = xnn_abs_f32(xnn_loadu_f32(mag + 2 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_3 = xnn_abs_f32(xnn_loadu_f32(mag + 3 * xnn_simd_size_f32));
    mag += 32;

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign_0, vmag_0);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign_1, vmag_1);
    xnn_simd_f32_t vy_2 = xnn_or_f32(vsign_2, vmag_2);
    xnn_simd_f32_t vy_3 = xnn_or_f32(vsign_3, vmag_3);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 32;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vsign = xnn_loadu_f32(sign);
    sign += xnn_simd_size_f32;

    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vsign = xnn_load_tail_f32(sign, batch >> XNN_LOG2_SIZEOF_FLOAT);
    vsign = xnn_and_f32(vsign, vsign_mask);

    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
