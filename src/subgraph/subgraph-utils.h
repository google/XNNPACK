// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_SUBGRAPH_SUBGRAPH_UTILS_H_
#define XNNPACK_SRC_SUBGRAPH_SUBGRAPH_UTILS_H_

#include <stdio.h>

#include "include/xnnpack.h"
#include "src/xnnpack/log.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prints a list of the subgraph's values and nodes to the given `out`. We wrap
// the actual function in a macro so that we can get the location of the caller.
#define xnn_subgraph_log(s, o) xnn_subgraph_log_impl(__FILE__, __LINE__, s, o)

// Log level-specific macros.
#if XNN_LOG_LEVEL >= XNN_LOG_DEBUG
#define xnn_subgraph_log_debug(s) \
  xnn_subgraph_log_impl(__FILE__, __LINE__, s, stderr)
#else
#define xnn_subgraph_log_debug(s)
#endif

#if XNN_LOG_LEVEL >= XNN_LOG_INFO
#define xnn_subgraph_log_info(s) \
  xnn_subgraph_log_impl(__FILE__, __LINE__, s, stderr)
#else
#define xnn_subgraph_log_info(s)
#endif

#if XNN_LOG_LEVEL >= XNN_LOG_WARNING
#define xnn_subgraph_log_warning(s) \
  xnn_subgraph_log_impl(__FILE__, __LINE__, s, stderr)
#else
#define xnn_subgraph_log_warning(s)
#endif

#if XNN_LOG_LEVEL >= XNN_LOG_ERROR
#define xnn_subgraph_log_error(s) \
  xnn_subgraph_log_impl(__FILE__, __LINE__, s, stderr)
#else
#define xnn_subgraph_log_error(s)
#endif

// The actual implementation of the subgraph logging function, should not be
// called directly.
void xnn_subgraph_log_impl(const char* filename, size_t line_number,
                           xnn_subgraph_t subgraph, FILE* out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_SUBGRAPH_SUBGRAPH_UTILS_H_
