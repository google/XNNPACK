// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/enums/microkernel-type.yaml
//   Generator: tools/generate-enum.py


#include <assert.h>
#include <stdint.h>

#include <xnnpack/microkernel-type.h>


static const uint8_t offset[13] = {
  0, 8, 24, 39, 46, 51, 74, 80, 85, 111, 116, 126, 136
};

static const char data[] =
  "Default\0"
  "Average Pooling\0"
  "Conv2D HWC2CHW\0"
  "DWConv\0"
  "GEMM\0"
  "Global Average Pooling\0"
  "IGEMM\0"
  "Mean\0"
  "Pixelwise Average Pooling\0"
  "SPMM\0"
  "Subconv2D\0"
  "Transpose\0"
  "VMulCAddC";

const char* xnn_microkernel_type_to_string(enum xnn_microkernel_type microkernel_type) {
  assert(microkernel_type >= xnn_microkernel_type_default);
  assert(microkernel_type <= xnn_microkernel_type_vmulcaddc);
  return &data[offset[microkernel_type]];
}
