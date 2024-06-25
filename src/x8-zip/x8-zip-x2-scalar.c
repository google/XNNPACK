// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/zip.h"


void xnn_x8_zip_x2_ukernel__scalar(
    size_t n,
    const uint8_t* input,
    uint8_t* output)
{
  assert(n != 0);

  const uint8_t* x = input;
  const uint8_t* y = (const uint8_t*) ((uintptr_t) x + n);
  uint8_t* o = output;

  do {
    const uint8_t vx = *x++;
    const uint8_t vy = *y++;
    o[0] = vx;
    o[1] = vy;
    o += 2;

    n -= sizeof(uint8_t);
  } while (n != 0);
}
