// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SPAN_H_
#define XNNPACK_YNNPACK_BASE_SPAN_H_

#include "slinky/base/span.h"

namespace ynn {

// slinky's span is designed to be compatible with C++20's span, use it until
// we can depend on C++20.
using slinky::span;

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SPAN_H_
