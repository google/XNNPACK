// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "xnnpack/common.h"


#ifdef __cplusplus
extern "C" {
#endif

enum xnn_operator_type {
#define XNN_ENUM_ITEM_0(enum_name, enum_string) enum_name = 0,
#define XNN_ENUM_ITEM(enum_name, enum_string) enum_name,
  #include "xnnpack/operator-type-defs.h"
#undef XNN_ENUM_ITEM_0
#undef XNN_ENUM_ITEM
};

XNN_INTERNAL const char* xnn_operator_type_to_string(enum xnn_operator_type operator_type);

#ifdef __cplusplus
}  // extern "C"
#endif
