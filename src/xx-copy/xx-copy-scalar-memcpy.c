// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <string.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_xx_copy_ukernel__scalar_memcpy(size_t batch, const void* input, void* output, const void* params) {
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);

  memcpy(output, input, batch);
}
