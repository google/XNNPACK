// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <string.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_xx_copy_ukernel__memcpy(size_t size, const void* input, void* output, const void* params) {
  assert(size != 0);

  memcpy(output, input, size);
}
