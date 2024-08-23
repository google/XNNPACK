// Auto-generated file. Do not edit!
//   Template: src/u32-vlog/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/math.h"
#include "xnnpack/vlog.h"


void xnn_u32_vlog_ukernel__scalar_x1(
    size_t batch,
    const uint32_t* input,
    uint32_t input_lshift,
    uint32_t output_scale,
    uint16_t* output) {

  assert(batch != 0);
  assert(input != NULL);
  assert(input_lshift < 32);
  assert(output != NULL);


  if XNN_UNLIKELY(batch != 0) {
    do {
      const uint32_t vi = *input++;
      const uint32_t scaled = vi << input_lshift;

      const uint32_t log_value = XNN_LIKELY(scaled != 0) ? math_u32_log32(scaled, output_scale) : 0;

      const uint32_t vout = math_min_u32(log_value, (uint32_t) INT16_MAX);
      *output++ = (uint16_t) vout;
    } while (--batch != 0);
  }
}
