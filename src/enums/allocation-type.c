// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>

#include "xnnpack/allocation-type.h"

#if XNN_LOG_LEVEL > 0
const char* xnn_allocation_type_to_string(enum xnn_allocation_type allocation_type) {
  switch(allocation_type) {
  #define XNN_ENUM_ITEM(enum_name, enum_string) case enum_name: return enum_string;
  #include "xnnpack/allocation-type-defs.h"
  default:
    XNN_UNREACHABLE;
  #undef XNN_ENUM_ITEM
  };
}
#endif  // XNN_LOG_LEVEL > 0
