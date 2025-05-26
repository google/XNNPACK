// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdlib.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/params.h"


enum xnn_status xnn_delete_operator(xnn_operator_t op)
{
  enum xnn_status status = xnn_destroy_operator(op);
  if (status != xnn_status_success) {
    return status;
  }
  xnn_release_simd_memory(op);
  return status;
}
