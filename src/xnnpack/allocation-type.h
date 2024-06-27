// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/enums/allocation-type.yaml
//   Generator: tools/generate-enum.py

#pragma once

#include "xnnpack/common.h"


#ifdef __cplusplus
extern "C" {
#endif

enum xnn_allocation_type {
  xnn_allocation_type_invalid = 0,
  xnn_allocation_type_static,
  xnn_allocation_type_workspace,
  xnn_allocation_type_external,
  xnn_allocation_type_persistent,
  xnn_allocation_type_dynamic,
};

#if XNN_LOG_LEVEL <= 0
  XNN_INLINE static const char* xnn_allocation_type_to_string(enum xnn_allocation_type type) {
    return "<unknown>";
  }
#else
  XNN_INTERNAL const char* xnn_allocation_type_to_string(enum xnn_allocation_type type);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif
