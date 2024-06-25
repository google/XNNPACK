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

extern XNN_INTERNAL const uint16_t xnn_table_vlog[129];

#define LOG_SEGMENTS_LOG2 7
#define LOG_SCALE 65536
#define LOG_SCALE_LOG2 16
#define LOG_COEFF 45426

static uint32_t xnn_u32_log32(uint32_t x, uint32_t out_scale) {
  const uint32_t log2x = math_clz_nonzero_u32(x) ^ 31;
  int32_t frac = x - (UINT32_C(1) << log2x);
  frac <<= math_doz_u32(LOG_SCALE_LOG2, log2x);
  frac >>= math_doz_u32(log2x, LOG_SCALE_LOG2);

  const uint32_t base_seg = frac >> (LOG_SCALE_LOG2 - LOG_SEGMENTS_LOG2);
  const uint32_t seg_unit = (UINT32_C(1) << LOG_SCALE_LOG2) >> LOG_SEGMENTS_LOG2;

  const int32_t c0 = xnn_table_vlog[base_seg];
  const int32_t c1 = xnn_table_vlog[base_seg + 1];
  const int32_t seg_base = seg_unit * base_seg;
  const int32_t rel_pos = math_asr_s32((c1 - c0) * (frac - seg_base), LOG_SCALE_LOG2);
  const uint32_t fraction = frac + c0 + rel_pos;
  const uint32_t log2 = (log2x << LOG_SCALE_LOG2) + fraction;
  const uint32_t round = LOG_SCALE >> 1;
  const uint32_t loge = (math_mulext_u32(log2, LOG_COEFF) + round) >> LOG_SCALE_LOG2;

  const uint32_t loge_scaled = (out_scale * loge + round) >> LOG_SCALE_LOG2;
  return loge_scaled;
}

void xnn_u32_vlog_ukernel__scalar_x3(
    size_t batch,
    const uint32_t* input,
    uint32_t input_lshift,
    uint32_t output_scale,
    uint16_t* output) {

  assert(batch != 0);
  assert(input != NULL);
  assert(input_lshift < 32);
  assert(output != NULL);

  for (; batch >= 3; batch -= 3) {
    const uint32_t vi0 = input[0];
    const uint32_t vi1 = input[1];
    const uint32_t vi2 = input[2];
    input += 3;

    const uint32_t scaled0 = vi0 << input_lshift;
    const uint32_t scaled1 = vi1 << input_lshift;
    const uint32_t scaled2 = vi2 << input_lshift;

    const uint32_t log_value0 = XNN_LIKELY(scaled0 != 0) ? xnn_u32_log32(scaled0, output_scale) : 0;

    const uint32_t vout0 = math_min_u32(log_value0, (uint32_t) INT16_MAX);  // signed max value
    output[0] = (uint16_t) vout0;
    const uint32_t log_value1 = XNN_LIKELY(scaled1 != 0) ? xnn_u32_log32(scaled1, output_scale) : 0;

    const uint32_t vout1 = math_min_u32(log_value1, (uint32_t) INT16_MAX);  // signed max value
    output[1] = (uint16_t) vout1;
    const uint32_t log_value2 = XNN_LIKELY(scaled2 != 0) ? xnn_u32_log32(scaled2, output_scale) : 0;

    const uint32_t vout2 = math_min_u32(log_value2, (uint32_t) INT16_MAX);  // signed max value
    output[2] = (uint16_t) vout2;

    output += 3;
  }

  if XNN_UNLIKELY(batch != 0) {
    do {
      const uint32_t vi = *input++;
      const uint32_t scaled = vi << input_lshift;

      const uint32_t log_value = XNN_LIKELY(scaled != 0) ? xnn_u32_log32(scaled, output_scale) : 0;

      const uint32_t vout = math_min_u32(log_value, (uint32_t) INT16_MAX);
      *output++ = (uint16_t) vout;
    } while (--batch != 0);
  }
}
