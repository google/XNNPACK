// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_x4_ukernel__scalar(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);
  const uint32_t* z = (const uint32_t*) ((uintptr_t) y + n);
  const uint32_t* w = (const uint32_t*) ((uintptr_t) z + n);
  uint32_t* o = output;

  do {
    const uint32_t vx = *x++;
    const uint32_t vy = *y++;
    const uint32_t vz = *z++;
    const uint32_t vw = *w++;
    o[0] = vx;
    o[1] = vy;
    o[2] = vz;
    o[3] = vw;
    o += 4;

    n -= 4;
  } while (n != 0);
}
