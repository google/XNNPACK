// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx512.h"

#include <cstdint>

#include "ynnpack/kernels/iota/generic.h"
#include "ynnpack/kernels/iota/iota.h"

namespace ynn {

YNN_DEFINE_IOTA_KERNEL(arch_flag::avx512, iota_fp32_avx512, float, 64);
YNN_DEFINE_IOTA_KERNEL(arch_flag::avx512, iota_int32_avx512, int32_t, 64);

}  // namespace ynn
