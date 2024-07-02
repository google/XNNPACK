// Auto-generated file. Do not edit!
//   Template: src/f32-vcopysign/copysignc.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/simd/f32-avx512f.h"

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_f32_vcopysignc_ukernel__avx512f_u16(
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
  assert(xnn_simd_size_f32 == 16);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vsign = xnn_set1_f32(*sign);
  vsign = xnn_and_f32(vsign, vsign_mask);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vcopysignc_ukernel__avx512f_u32(
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
  assert(xnn_simd_size_f32 == 16);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vsign = xnn_set1_f32(*sign);
  vsign = xnn_and_f32(vsign, vsign_mask);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {

    xnn_simd_f32_t vmag_0 = xnn_abs_f32(xnn_loadu_f32(mag));
    xnn_simd_f32_t vmag_1 = xnn_abs_f32(xnn_loadu_f32(mag + 1 * xnn_simd_size_f32));
    mag += 32;

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign, vmag_0);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign, vmag_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 32;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vcopysignc_ukernel__avx512f_u48(
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
  assert(xnn_simd_size_f32 == 16);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vsign = xnn_set1_f32(*sign);
  vsign = xnn_and_f32(vsign, vsign_mask);

  for (; batch >= 48 * sizeof(float); batch -= 48 * sizeof(float)) {

    xnn_simd_f32_t vmag_0 = xnn_abs_f32(xnn_loadu_f32(mag));
    xnn_simd_f32_t vmag_1 = xnn_abs_f32(xnn_loadu_f32(mag + 1 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_2 = xnn_abs_f32(xnn_loadu_f32(mag + 2 * xnn_simd_size_f32));
    mag += 48;

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign, vmag_0);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign, vmag_1);
    xnn_simd_f32_t vy_2 = xnn_or_f32(vsign, vmag_2);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    output += 48;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vcopysignc_ukernel__avx512f_u64(
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
  assert(xnn_simd_size_f32 == 16);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vsign = xnn_set1_f32(*sign);
  vsign = xnn_and_f32(vsign, vsign_mask);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {

    xnn_simd_f32_t vmag_0 = xnn_abs_f32(xnn_loadu_f32(mag));
    xnn_simd_f32_t vmag_1 = xnn_abs_f32(xnn_loadu_f32(mag + 1 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_2 = xnn_abs_f32(xnn_loadu_f32(mag + 2 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_3 = xnn_abs_f32(xnn_loadu_f32(mag + 3 * xnn_simd_size_f32));
    mag += 64;

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign, vmag_0);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign, vmag_1);
    xnn_simd_f32_t vy_2 = xnn_or_f32(vsign, vmag_2);
    xnn_simd_f32_t vy_3 = xnn_or_f32(vsign, vmag_3);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 64;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
