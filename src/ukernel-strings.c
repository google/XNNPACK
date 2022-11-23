// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/ukernel-strings.yaml
//   Generator: tools/generate-enum.py


#include <assert.h>
#include <stdint.h>

#include <xnnpack/ukernel-type.h>


static const uint8_t offset[10] = {
  0, 8, 24, 39, 46, 51, 57, 83, 88, 98
};

static const char data[] = 
  "Default\0"
  "Average Pooling\0"
  "Conv2D HWC2CHW\0"
  "DWConv\0"
  "GEMM\0"
  "IGEMM\0"
  "Pixelwise Average Pooling\0"
  "SPMM\0"
  "Subconv2D\0"
  "VMulCAddC";

const char* xnn_ukernel_type_to_string(enum xnn_ukernel_type ukernel_type) {
  assert(ukernel_type >= xnn_ukernel_type_default);
  assert(ukernel_type <= xnn_ukernel_type_vmulcaddc);
  return &data[offset[ukernel_type]];
}
