// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_REF_COUNT_H_
#define XNNPACK_YNNPACK_BASE_REF_COUNT_H_

#include "slinky/base/ref_count.h"

namespace ynn {

// This is a basic "intrusive" reference counted pointer class.
using slinky::ref_counted;

// This is the helper for implementing reference counting of the above.
using slinky::ref_count;

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_REF_COUNT_H_
