// Auto-generated file. Do not edit!
//   Template: src/s16-vrshift/scalar.c.in
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
#include <xnnpack/vrshift.h>


void xnn_s16_vrshift_ukernel__scalar_x2(
    size_t c,
    const int16_t* input,
    uint32_t shift,
    int16_t* output) {

  assert(c > 0);
  assert(input != NULL);
  assert(shift < 32);
  assert(output != NULL);

 for (; c >= 2; c -= 2) {
   const uint16_t vi0 = (uint16_t) input[0];
   const uint16_t vi1 = (uint16_t) input[1];
   input += 2;

   const uint16_t vout0 = vi0 << (uint16_t) shift;
   const uint16_t vout1 = vi1 << (uint16_t) shift;

   output[0] = (int16_t) vout0;
   output[1] = (int16_t) vout1;
   output += 2;
 }

 if XNN_UNLIKELY(c != 0) {
   do {
     const uint16_t vi = (uint16_t) *input++;

     const uint16_t vout = vi << (uint16_t) shift;

     *output++ = (int16_t) vout;
   } while (--c != 0);
 }
}
