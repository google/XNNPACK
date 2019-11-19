// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>

#include <xnnpack/common.h>


XNN_INTERNAL void* xnn_allocate(void* context, size_t size);
XNN_INTERNAL void* xnn_reallocate(void* context, void* pointer, size_t size);
XNN_INTERNAL void xnn_deallocate(void* context, void* pointer);
XNN_INTERNAL void* xnn_aligned_allocate(void* context, size_t alignment, size_t size);
XNN_INTERNAL void xnn_aligned_deallocate(void* context, void* pointer);
