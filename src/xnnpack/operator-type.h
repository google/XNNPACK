// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_SRC_XNNPACK_OPERATOR_TYPE_H_
#define THIRD_PARTY_XNNPACK_SRC_XNNPACK_OPERATOR_TYPE_H_

#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

enum xnn_operator_type {
#define XNN_ENUM_ITEM_0(enum_name, enum_string) enum_name = 0,
#define XNN_ENUM_ITEM(enum_name, enum_string) enum_name,
#include "src/xnnpack/operator-type-defs.h"
#undef XNN_ENUM_ITEM_0
#undef XNN_ENUM_ITEM
};

XNN_INTERNAL const char* xnn_operator_type_to_string(
    enum xnn_operator_type operator_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_XNNPACK_SRC_XNNPACK_OPERATOR_TYPE_H_