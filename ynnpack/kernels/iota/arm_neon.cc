// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm_neon.h"

#include <cstddef>
#include <cstdint>

#include "ynnpack/kernels/iota/generic.h"
#include "ynnpack/kernels/iota/iota.h"

namespace ynn {

YNN_DEFINE_IOTA_KERNEL(arch_flag::neon, iota_fp32_neon, float, 16);
YNN_DEFINE_IOTA_KERNEL(arch_flag::neon, iota_int32_neon, int32_t, 16);

}  // namespace ynn
