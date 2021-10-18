// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/scalar-float.c.in
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


void xnn_f16_f32_vcvt_ukernel__scalar_float_x1(
    size_t n,
    const void* input,
    float* output,
    const void* params)
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  do {
    const uint16_t vh = *i++;

    const float vf = fp16_ieee_to_fp32_value(vh);

    *output++ = vf;

    n -= sizeof(float);
  } while (n != 0);
}
