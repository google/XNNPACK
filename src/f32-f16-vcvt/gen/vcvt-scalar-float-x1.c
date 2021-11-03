// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/scalar-float.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/vcvt.h>

#include <fp16.h>


void xnn_f32_f16_vcvt_ukernel__scalar_float_x1(
    size_t n,
    const float* input,
    void* output,
    const void* params)
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float* i = (const float*) input;
  do {
    const float vh = *i++;

    const uint16_t vf = fp16_ieee_from_fp32_value(vh);

    ((uint16_t*) output)[0] = vf;
    output = (uint16_t*) output + 1;

    n -= sizeof(uint16_t);
  } while (n != 0);
}
