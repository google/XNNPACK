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


#if XNN_PLATFORM_JIT
static void xnn_release_jit_ukernel(struct xnn_ukernel ukernel)
{
  switch (ukernel.type) {
    case xnn_ukernel_type_gemm:
      xnn_release_code_memory(&ukernel.gemm.general_code_buffer);
      xnn_release_code_memory(&ukernel.gemm.mr1_code_buffer);
      break;
    default:
      break;
      // Do nothing, only GEMMs have JIT kernels now.
  }
}
#endif  // XNN_PLATFORM_JIT

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
  xnn_release_simd_memory(op->packed_weights);
  xnn_release_simd_memory(op->zero_buffer);
  xnn_release_memory(op->pixelwise_buffer);
  xnn_release_memory(op->subconvolution_buffer);
  xnn_release_simd_memory(op->lookup_table);
#if XNN_PLATFORM_JIT
  xnn_release_jit_ukernel(op->ukernel);
#endif  // XNN_PLATFORM_JIT
  xnn_release_simd_memory(op);
  return xnn_status_success;
}
