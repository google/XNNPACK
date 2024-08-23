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


void xnn_u32_vlog_ukernel__scalar_x4(
    size_t batch,
    const uint32_t* input,
    uint32_t input_lshift,
    uint32_t output_scale,
    uint16_t* output) {

  assert(batch != 0);
  assert(input != NULL);
  assert(input_lshift < 32);
  assert(output != NULL);

  for (; batch >= 4; batch -= 4) {
    const uint32_t vi0 = input[0];
    const uint32_t vi1 = input[1];
    const uint32_t vi2 = input[2];
    const uint32_t vi3 = input[3];
    input += 4;

    const uint32_t scaled0 = vi0 << input_lshift;
    const uint32_t scaled1 = vi1 << input_lshift;
    const uint32_t scaled2 = vi2 << input_lshift;
    const uint32_t scaled3 = vi3 << input_lshift;

    const uint32_t log_value0 = XNN_LIKELY(scaled0 != 0) ? math_u32_log32(scaled0, output_scale) : 0;

    const uint32_t vout0 = math_min_u32(log_value0, (uint32_t) INT16_MAX);  // signed max value
    output[0] = (uint16_t) vout0;
    const uint32_t log_value1 = XNN_LIKELY(scaled1 != 0) ? math_u32_log32(scaled1, output_scale) : 0;

    const uint32_t vout1 = math_min_u32(log_value1, (uint32_t) INT16_MAX);  // signed max value
    output[1] = (uint16_t) vout1;
    const uint32_t log_value2 = XNN_LIKELY(scaled2 != 0) ? math_u32_log32(scaled2, output_scale) : 0;

    const uint32_t vout2 = math_min_u32(log_value2, (uint32_t) INT16_MAX);  // signed max value
    output[2] = (uint16_t) vout2;
    const uint32_t log_value3 = XNN_LIKELY(scaled3 != 0) ? math_u32_log32(scaled3, output_scale) : 0;

    const uint32_t vout3 = math_min_u32(log_value3, (uint32_t) INT16_MAX);  // signed max value
    output[3] = (uint16_t) vout3;

    output += 4;
  }

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
