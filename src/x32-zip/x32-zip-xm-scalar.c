// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_xm_ukernel__scalar(
    size_t n,
    size_t m,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);
  assert(m >= 4);

  size_t k = n;
  do {
    size_t l = m;
    const uint32_t* input_column = input++;
    do {
      *output++ = *input_column;
      input_column = (uint32_t*) ((uintptr_t) input_column + n);
    } while (--l != 0);
    k -= 4;
  } while (k != 0);
}
