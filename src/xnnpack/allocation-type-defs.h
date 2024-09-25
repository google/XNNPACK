// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_ENUM_ITEM_0
#define XNN_ENUM_ITEM_0(enum_name, enum_string) XNN_ENUM_ITEM(enum_name, enum_string)
#define XNN_DEFINED_ENUM_ITEM_0
#endif

XNN_ENUM_ITEM_0(xnn_allocation_type_invalid, "invalid")
XNN_ENUM_ITEM(xnn_allocation_type_static, "static")
XNN_ENUM_ITEM(xnn_allocation_type_workspace, "workspace")
XNN_ENUM_ITEM(xnn_allocation_type_external, "external")
XNN_ENUM_ITEM(xnn_allocation_type_persistent, "persistent")
XNN_ENUM_ITEM(xnn_allocation_type_dynamic, "dynamic")


#ifdef XNN_DEFINED_ENUM_ITEM_0
#undef XNN_DEFINED_ENUM_ITEM_0
#undef XNN_ENUM_ITEM_0
#endif
