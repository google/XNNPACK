// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdlib.h>

#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/log.h"
#include "xnnpack/operator.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/params.h"


enum xnn_status xnn_delete_operator(xnn_operator_t op)
{
  enum xnn_status status = xnn_destroy_operator(op);
  if (status != xnn_status_success) {
    return status;
  }
  xnn_release_simd_memory(op);
  return status;
}
