// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/vcvt.h"


void xnn_f16_f32_vcvt_ukernel__hvx_u32(
    size_t batch,
    const xnn_float16* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 32 * sizeof(xnn_float16); batch -= 32 * sizeof(xnn_float16)) {
    const xnn_simd_f32_t vy = xnn_cvt_f32_f16(xnn_loadu_f16(input));
    input += 32;

    xnn_storeu_f32(output, vy);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(xnn_float16));
    assert(batch <= 31 * sizeof(xnn_float16));

    const size_t num_elements = batch >> XNN_LOG2_SIZEOF_FLOAT16;
    const xnn_simd_f32_t vy = xnn_cvt_f32_f16(xnn_loadu_f16(input));
    xnn_store_tail_f32(output, vy, num_elements);
  }
}
