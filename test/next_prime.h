// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __XNNPACK_TEST_NEXT_PRIME_H_
#define __XNNPACK_TEST_NEXT_PRIME_H_

#include <cstddef>

namespace xnnpack {

bool IsPrime(size_t n);
size_t NextPrime(size_t n);

};  // namespace xnnpack

#endif  // __XNNPACK_TEST_NEXT_PRIME_H_
