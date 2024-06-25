// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/enums/allocation-type.yaml
//   Generator: tools/generate-enum.py

#include <assert.h>
#include <stdint.h>

#include "xnnpack/allocation-type.h"

#if XNN_LOG_LEVEL > 0
static const uint8_t offset[6] = {
  0, 8, 15, 25, 34, 45
};

static const char data[] =
  "invalid\0"
  "static\0"
  "workspace\0"
  "external\0"
  "persistent\0"
  "dynamic";

const char* xnn_allocation_type_to_string(enum xnn_allocation_type allocation_type) {
  assert(allocation_type >= xnn_allocation_type_invalid);
  assert(allocation_type <= xnn_allocation_type_dynamic);
  return &data[offset[allocation_type]];
}
#endif  // XNN_LOG_LEVEL > 0
