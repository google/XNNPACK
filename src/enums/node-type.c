// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/node-type.h"

#if XNN_LOG_LEVEL > 0
const char* xnn_node_type_to_string(enum xnn_node_type node_type) {
  switch(node_type) {
  #define XNN_ENUM_ITEM(enum_name, enum_string) case enum_name: return enum_string;
  #include "xnnpack/node-type-defs.h"
  default:
    XNN_UNREACHABLE;
  #undef XNN_ENUM_ITEM
  };
}
#endif  // XNN_LOG_LEVEL > 0