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

#include "xnnpack/filterbank.h"
#include "xnnpack/math.h"


void xnn_u32_filterbank_accumulate_ukernel__scalar_x1(
    size_t rows,
    const uint32_t* input,
    const uint8_t* weight_widths,
    const uint16_t* weights,
    uint64_t* output) {

  assert(rows != 0);
  assert(input != NULL);
  assert(weight_widths != NULL);
  assert(weights != NULL);
  assert(output != NULL);

  uint64_t weight_accumulator = 0;
  uint64_t unweight_accumulator = 0;

  // compute unweight as initial weight
  size_t n = (size_t) *weight_widths++;
  assert(n != 0);
  do {
    const uint32_t vi = *input++;
    const uint32_t vu = (uint32_t) weights[1];  // unweight
    weights += 2;

    const uint64_t vuacc = math_mulext_u32(vi, vu);

    weight_accumulator += vuacc;

  } while (--n != 0);

  do {
    size_t n = (size_t) *weight_widths++;
    assert(n != 0);
    do {
      const uint32_t vi = *input++;
      const uint32_t vw = (uint32_t) weights[0];  // weight
      const uint32_t vu = (uint32_t) weights[1];  // unweight
      weights += 2;

      const uint64_t vwacc = math_mulext_u32(vi, vw);
      const uint64_t vuacc = math_mulext_u32(vi, vu);

      weight_accumulator += vwacc;
      unweight_accumulator += vuacc;

    } while (--n != 0);

    *output++ = weight_accumulator;
    weight_accumulator = unweight_accumulator;
    unweight_accumulator = 0;

  } while (--rows != 0);
}
