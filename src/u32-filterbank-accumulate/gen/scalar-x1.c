// Auto-generated file. Do not edit!
//   Template: src/u32-filterbank-accumulate/scalar.c.in
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
#include <xnnpack/filterbank.h>


void xnn_u32_filterbank_accumulate_ukernel__scalar_x1(
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

  uint64_t weight_accumulator = 0;
  uint64_t unweight_accumulator = 0;

  do {
    const size_t io = (size_t) *input_offset++;
    const size_t wo = (size_t) *weight_offset++;
    size_t n = (size_t) *weight_widths++;

    const uint32_t* i = input + io;
    const uint16_t* w = weights + wo;
    const uint16_t* u = unweights + wo;

    assert(n != 0);

    do {
      const uint32_t vi = (uint32_t) *i++;
      const uint32_t vw = (uint32_t) *w++;
      const uint32_t vu = (uint32_t) *u++;

      const uint64_t vwacc = (uint64_t) vi * (uint64_t) vw;
      const uint64_t vuacc = (uint64_t) vi * (uint64_t) vu;

      weight_accumulator += vwacc;
      unweight_accumulator += vuacc;

    } while (--n != 0);

    *output++ = weight_accumulator;
    weight_accumulator = unweight_accumulator;

  } while (--rows != 0);
}
