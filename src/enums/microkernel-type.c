// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>

#include "xnnpack/microkernel-type.h"

const char* xnn_microkernel_type_to_string(enum xnn_microkernel_type microkernel_type) {
  switch(microkernel_type) {
  #define XNN_ENUM_ITEM(enum_name, enum_string) case enum_name: return enum_string;
  #include "xnnpack/microkernel-type-defs.h"
  default:
    XNN_UNREACHABLE;
  #undef XNN_ENUM_ITEM
  };
}
