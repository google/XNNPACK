// Auto-generated file. Do not edit!
//   Template: src/u32-filterbank-accumulate/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math.h>
#include <xnnpack/filterbank.h>


void xnn_u32_filterbank_accumulate_ukernel__neon_x1(
    size_t rows,
    size_t batch_size,
    const uint32_t* input,
    const uint16_t* input_offset,
    const uint16_t* weight_offset,
    const uint16_t* weight_widths,
    const uint16_t* weights,
    const uint16_t* unweights,
    uint64_t* output) {

  assert(rows != 0);
  assert(batch_size != 0);
  assert(input != NULL);
  assert(input_offset != NULL);
  assert(weight_offset != NULL);
  assert(weight_widths != NULL);
  assert(weights != NULL);
  assert(output != NULL);

  uint64x2_t weight_accumulator = vdupq_n_u64(0);
  uint64x2_t unweight_accumulator = vdupq_n_u64(0);

  do {
    const size_t io = (size_t) *input_offset++;
    const size_t wo = (size_t) *weight_offset++;
    size_t n = (size_t) *weight_widths++;

    const uint32_t* i = input + io;
    const uint16_t* w = weights + wo;
    const uint16_t* u = unweights + wo;

    assert(n != 0);

    do {
      const uint32x2_t vi = vld1_dup_u32(i); ++i;
      const uint16x4_t vw = vld1_dup_u16(w); ++w;
      const uint16x4_t vu = vld1_dup_u16(u); ++u;
      const uint32x2_t vw32 = vget_low_u32(vmovl_u16(vw));
      const uint32x2_t vu32 = vget_low_u32(vmovl_u16(vu));

      weight_accumulator =   vmlal_u32(weight_accumulator,   vi, vw32);
      unweight_accumulator = vmlal_u32(unweight_accumulator, vi, vu32);

    } while (--n != 0);


    vst1_u64(output, vget_low_u64(weight_accumulator));  ++output;
    weight_accumulator = unweight_accumulator;

  } while (--rows != 0);
}
