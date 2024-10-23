// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, ukernel, size_fn)                          \
  XNN_INTERNAL void ukernel(size_t m, size_t k, size_t mr, size_t kr,      \
                            size_t sr, size_t m_idx_start, const float* x, \
                            size_t x_stride, void* y);                     \
                                                                           \
  XNN_INTERNAL size_t size_fn(size_t m, size_t k, size_t mr, size_t kr,    \
                              size_t sr);

#include "x32-pack-lh/x32-pack-lh.h"

#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
