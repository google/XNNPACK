// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/operator-type.h"

#include <assert.h>

#include "src/xnnpack/common.h"

const char* xnn_operator_type_to_string(enum xnn_operator_type operator_type) {
  switch (operator_type) {
#define XNN_ENUM_ITEM(enum_name, enum_string) \
  case enum_name:                             \
    return enum_string;
#include "src/xnnpack/operator-type-defs.h"
    default:
      XNN_UNREACHABLE;
#undef XNN_ENUM_ITEM
  };
}
