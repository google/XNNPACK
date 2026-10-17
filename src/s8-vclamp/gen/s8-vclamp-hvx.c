// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/s8-vclamp/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/s8-hvx.h"
#include "src/xnnpack/vunary.h"


void xnn_s8_vclamp_ukernel__hvx_u128(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_s8_minmax_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  const xnn_simd_s8_t vmin = xnn_set1_s8(params->scalar.min);
  const xnn_simd_s8_t vmax = xnn_set1_s8(params->scalar.max);

  for (; batch >= xnn_simd_bytes_s8; batch -= xnn_simd_bytes_s8) {
    xnn_simd_s8_t vacc = xnn_loadu_s8(input);
    input += xnn_simd_size_s8;

    vacc = xnn_max_s8(vmin, vacc);
    vacc = xnn_min_s8(vmax, vacc);

    xnn_storeu_s8(output, vacc);
    output += xnn_simd_size_s8;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s8_t vacc = xnn_load_tail_s8(input, batch);

    vacc = xnn_max_s8(vmin, vacc);
    vacc = xnn_min_s8(vmax, vacc);

    xnn_store_tail_s8(output, vacc, batch);
  }
}

void xnn_s8_vclamp_ukernel__hvx_u256(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_s8_minmax_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  const xnn_simd_s8_t vmin = xnn_set1_s8(params->scalar.min);
  const xnn_simd_s8_t vmax = xnn_set1_s8(params->scalar.max);

  for (; batch >= 256 * sizeof(int8_t); batch -= 256 * sizeof(int8_t)) {
    xnn_simd_s8_t vacc0 = xnn_loadu_s8(input + 0 * xnn_simd_size_s8);
    xnn_simd_s8_t vacc1 = xnn_loadu_s8(input + 1 * xnn_simd_size_s8);
    input += 256;

    vacc0 = xnn_max_s8(vmin, vacc0);
    vacc1 = xnn_max_s8(vmin, vacc1);

    vacc0 = xnn_min_s8(vmax, vacc0);
    vacc1 = xnn_min_s8(vmax, vacc1);

    xnn_storeu_s8(output + 0 * xnn_simd_size_s8, vacc0);
    xnn_storeu_s8(output + 1 * xnn_simd_size_s8, vacc1);
    output += 256;
  }
  for (; batch >= xnn_simd_bytes_s8; batch -= xnn_simd_bytes_s8) {
    xnn_simd_s8_t vacc = xnn_loadu_s8(input);
    input += xnn_simd_size_s8;

    vacc = xnn_max_s8(vmin, vacc);
    vacc = xnn_min_s8(vmax, vacc);

    xnn_storeu_s8(output, vacc);
    output += xnn_simd_size_s8;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s8_t vacc = xnn_load_tail_s8(input, batch);

    vacc = xnn_max_s8(vmin, vacc);
    vacc = xnn_min_s8(vmax, vacc);

    xnn_store_tail_s8(output, vacc, batch);
  }
}
