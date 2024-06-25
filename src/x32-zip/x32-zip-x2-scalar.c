// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_x2_ukernel__scalar(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);

  do {
    const uint32_t vx = *x++;
    const uint32_t vy = *y++;
    output[0] = vx;
    output[1] = vy;
    output += 2;

    n -= 4;
  } while (n != 0);
}
