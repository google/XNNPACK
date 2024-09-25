// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "xnnpack/common.h"


#ifdef __cplusplus
extern "C" {
#endif

enum xnn_node_type {
#define XNN_ENUM_ITEM_0(enum_name, enum_string) enum_name = 0,
#define XNN_ENUM_ITEM(enum_name, enum_string) enum_name,
  #include "xnnpack/node-type-defs.h"
#undef XNN_ENUM_ITEM_0
#undef XNN_ENUM_ITEM
};

#if XNN_LOG_LEVEL <= 0
  XNN_INLINE static const char* xnn_node_type_to_string(enum xnn_node_type type) {
    return "<unknown>";
  }
#else
  XNN_INTERNAL const char* xnn_node_type_to_string(enum xnn_node_type type);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif
