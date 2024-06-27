// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/enums/microkernel-type.yaml
//   Generator: tools/generate-enum.py

#pragma once

#include "xnnpack/common.h"


#ifdef __cplusplus
extern "C" {
#endif

enum xnn_microkernel_type {
  xnn_microkernel_type_default = 0,
  xnn_microkernel_type_average_pooling,
  xnn_microkernel_type_conv2d_hwc2chw,
  xnn_microkernel_type_dwconv,
  xnn_microkernel_type_gemm,
  xnn_microkernel_type_global_average_pooling,
  xnn_microkernel_type_igemm,
  xnn_microkernel_type_mean,
  xnn_microkernel_type_pixelwise_average_pooling,
  xnn_microkernel_type_spmm,
  xnn_microkernel_type_subconv2d,
  xnn_microkernel_type_transpose,
  xnn_microkernel_type_vmulcaddc,
};

XNN_INTERNAL const char* xnn_microkernel_type_to_string(enum xnn_microkernel_type microkernel_type);

#ifdef __cplusplus
}  // extern "C"
#endif
