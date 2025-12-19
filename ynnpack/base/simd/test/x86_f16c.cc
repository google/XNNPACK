// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_f16c.h"

#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_CONVERT(x86_f16c, float, f16x8);
TEST_CONVERT(x86_f16c, float, f16x16);

}  // namespace simd
}  // namespace ynn
