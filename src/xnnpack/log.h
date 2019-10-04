// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <inttypes.h>

#include <clog.h>

#ifndef XNN_LOG_LEVEL
  #error "Undefined XNN_LOG_LEVEL"
#endif

CLOG_DEFINE_LOG_DEBUG(xnn_log_debug, "XNNPACK", XNN_LOG_LEVEL);
CLOG_DEFINE_LOG_INFO(xnn_log_info, "XNNPACK", XNN_LOG_LEVEL);
CLOG_DEFINE_LOG_WARNING(xnn_log_warning, "XNNPACK", XNN_LOG_LEVEL);
CLOG_DEFINE_LOG_ERROR(xnn_log_error, "XNNPACK", XNN_LOG_LEVEL);
CLOG_DEFINE_LOG_FATAL(xnn_log_fatal, "XNNPACK", XNN_LOG_LEVEL);
