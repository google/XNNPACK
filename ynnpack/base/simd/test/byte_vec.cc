// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "ynnpack/base/simd/byte_vec.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class byte_vec : public ::testing::Test {};

TEST_LOAD_STORE(byte_vec, u8, 4);
TEST_LOAD_STORE(byte_vec, u8, 8);
TEST_LOAD_STORE(byte_vec, u8, 16);

TEST_PARTIAL_LOAD_STORE(byte_vec, u8, 4);
TEST_PARTIAL_LOAD_STORE(byte_vec, u8, 8);
TEST_PARTIAL_LOAD_STORE(byte_vec, u8, 16);

}  // namespace simd
}  // namespace ynn
