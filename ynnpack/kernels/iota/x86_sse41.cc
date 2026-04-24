// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_sse41.h"

#include <cstddef>
#include <cstdint>

#include "ynnpack/kernels/iota/generic.h"
#include "ynnpack/kernels/iota/iota.h"

namespace ynn {

YNN_DEFINE_IOTA_KERNEL(arch_flag::sse41, ynn_iota_int32_sse41, int32_t);

}  // namespace ynn
