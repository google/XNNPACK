// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm_neonfma.h"

#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_FMA(arm_neonfma, f32x4, arch_flag::neonfma);

}  // namespace simd
}  // namespace ynn
