// Auto-generated file. Do not edit!
//   Template: src/x8-lut/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/lut.h>
#include <xnnpack/common.h>


void xnn_x8_lut_ukernel__scalar_x1(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const uint8_t t[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(n != 0);
  assert(x != NULL);
  assert(y != NULL);

  do {
    const size_t vx = (size_t) *x++;
    const uint32_t vt = (uint32_t) t[vx];
    *y++ = (uint8_t) vt;
    n -= sizeof(uint8_t);
  } while (n != 0);
}
