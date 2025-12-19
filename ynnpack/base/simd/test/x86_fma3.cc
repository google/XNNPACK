// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_fma3.h"

#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_FMA(x86_fma3, float, 8);

}  // namespace simd
}  // namespace ynn
