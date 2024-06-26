// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/zip.h"


void xnn_x8_zip_xm_ukernel__scalar(
    size_t n,
    size_t m,
    const uint8_t* input,
    uint8_t* output)
{
  assert(n != 0);
  assert(m >= 4);

  size_t k = n;
  do {
    size_t l = m;
    const uint8_t* input_column = input++;
    do {
      *output++ = *input_column;
      input_column = (uint8_t*) ((uintptr_t) input_column + n);
    } while (--l != 0);
    k -= sizeof(uint8_t);
  } while (k != 0);
}
