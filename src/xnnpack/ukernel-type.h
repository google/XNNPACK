// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/ukernel-strings.yaml
//   Generator: tools/generate-enum.py

#pragma once

enum xnn_ukernel_type {
  xnn_ukernel_type_default = 0,
  xnn_ukernel_type_average_pooling,
  xnn_ukernel_type_conv2d_hwc2chw,
  xnn_ukernel_type_dwconv,
  xnn_ukernel_type_gemm,
  xnn_ukernel_type_igemm,
  xnn_ukernel_type_pixelwise_average_pooling,
  xnn_ukernel_type_spmm,
  xnn_ukernel_type_subconv2d,
  xnn_ukernel_type_vmulcaddc,
};