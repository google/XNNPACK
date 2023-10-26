// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdlib.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>


enum xnn_status xnn_delete_operator(xnn_operator_t op)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to delete operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (op == NULL) {
    return xnn_status_invalid_parameter;
  }

  xnn_release_memory(op->indirection_buffer);
  if (op->weights_cache == NULL) {
    xnn_release_simd_memory(op->packed_weights.pointer);
  }
  if (op->num_post_operation_params != 0) {
    xnn_release_memory(op->post_operation_params);
  }
  xnn_release_simd_memory(op->zero_buffer);
  if (op->zero_buffers) {
    for (size_t i = 1; i < op->batch_size; ++i) {
      xnn_release_simd_memory(op->zero_buffers[i]);
    }
    xnn_release_memory(op->zero_buffers);
  }
  xnn_release_memory(op->pixelwise_buffer);
  xnn_release_memory(op->subconvolution_buffer);
  xnn_release_simd_memory(op->lookup_table);
  xnn_release_simd_memory(op);
  return xnn_status_success;
}
