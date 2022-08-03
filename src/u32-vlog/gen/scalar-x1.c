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

#include <xnnpack/math.h>
#include <xnnpack/vlog.h>

extern XNN_INTERNAL const uint16_t xnn_table_loglut[130];

// Calculate integer logarithm, 32 Bit version
static uint32_t xnn_u32_log32(uint32_t x, uint32_t out_scale) {
  const int log_scale = 65536;
  const int log_scale_log2 = 16;
  const int log_coeff = 45426;
  const uint32_t log2x = math_clz_nonzero_u32(x) ^ 31;  // log2 of x
  assert(log2x < 32);

  // Number of segments in the log lookup table. The table will be log_segments+1
  // in length (with some padding).
  const int log_segments_log2 = 7;

  // Part 1
  uint32_t frac = x - (UINT32_C(1) << log2x);

  // Shift the fractional part into msb of 16 bits
  frac =  XNN_UNPREDICTABLE(log2x < log_scale_log2) ?
      (frac << (log_scale_log2 - log2x)) :
      (frac >> (log2x - log_scale_log2));

  // Part 2
  const uint32_t base_seg = frac >> (log_scale_log2 - log_segments_log2);
  const uint32_t seg_unit = (UINT32_C(1) << log_scale_log2) >> log_segments_log2;

  assert(128 == (1 << log_segments_log2));
  assert(base_seg < (1 << log_segments_log2));

  const uint32_t c0 = xnn_table_loglut[base_seg];
  const uint32_t c1 = xnn_table_loglut[base_seg + 1];
  const uint32_t seg_base = seg_unit * base_seg;
  const uint32_t rel_pos = ((c1 - c0) * (frac - seg_base)) >> log_scale_log2;
  const uint32_t fraction =  frac + c0 + rel_pos;

  const uint32_t log2 = (log2x << log_scale_log2) + fraction;
  const uint32_t round = log_scale / 2;
  const uint32_t loge = (((uint64_t) log_coeff) * log2 + round) >> log_scale_log2;
  // Finally scale to our output scale
  const uint32_t loge_scaled = (out_scale * loge + round) >> log_scale_log2;
  return loge_scaled;
}

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

      const uint32_t log_value = scaled ? xnn_u32_log32(scaled, output_scale) : 0;

      const uint32_t vout = math_min_u32(log_value, (uint32_t) INT16_MAX);
      *output++ = (uint16_t) vout;
    } while (--batch != 0);
  }
}
