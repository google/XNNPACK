// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/rmax.h>


void xnn_u8_rmax_ukernel__scalar(
    size_t n,
    const uint8_t* x,
    uint8_t* y)
{
  assert(n != 0);

  uint8_t vmax0 = 0;
  uint8_t vmax1 = 0;
  for (; n >= 2 * sizeof(uint8_t); n -= 2 * sizeof(uint8_t)) {
    const uint8_t vt0 = x[0];
    const uint8_t vt1 = x[1];
    x += 2;

    vmax0 = vt0 > vmax0 ? vt0 : vmax0;
    vmax1 = vt1 > vmax1 ? vt1 : vmax1;
  }
  uint8_t vmax = vmax0 > vmax1 ? vmax0 : vmax1;
  if (n != 0) {
    const uint8_t vt = *x++;
    vmax = vt > vmax ? vt : vmax;
  }
  *y = vmax;
}
