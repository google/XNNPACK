// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "test/subgraph/runtime-flags.h"

#include <gtest/gtest.h>
#if GTEST_HAS_ABSL
#include <absl/flags/flag.h>
#endif

// We only define this if GTest has a separate Abseil install, so we
// can use Abseil's built-in command-line-flag processing.
#if GTEST_HAS_ABSL
ABSL_FLAG(uint32_t, xnn_runtime_flags, 0,
          "Value to pass to xnn_create_runtime for flags");
#endif

extern "C" {

uint32_t xnn_test_runtime_flags() {
#if GTEST_HAS_ABSL
  return absl::GetFlag(FLAGS_xnn_runtime_flags);
#else
  return 0;
#endif
}
}
