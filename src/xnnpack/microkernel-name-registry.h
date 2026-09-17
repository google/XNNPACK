// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root of this source tree.

// Function-pointer -> #ukernel symbol map populated as a side-effect of the
// XNN_INIT_*_UKERNEL macros and consumed by xnn_get_runtime_profiling_info.

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// First-registration wins; later calls with the same `fn_ptr` are no-ops.
void xnn_register_microkernel_name(const void* fn_ptr, const char* name);

// Returns NULL if `fn_ptr` was never registered or is NULL.
const char* xnn_lookup_microkernel_name(const void* fn_ptr);

// Use at every ukernel call site in src/configs/*-config.c so a release
// build with the flag off can drop the `#ukernel` string literals entirely.
#define XNN_REGISTER_UKERNEL_NAME(ukernel) \
  xnn_register_microkernel_name((const void*)(ukernel), #ukernel)

#ifdef __cplusplus
}
#endif
