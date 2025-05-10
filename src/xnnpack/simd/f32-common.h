// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F32_COMMON_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F32_COMMON_H_

#include <stdint.h>

#include "src/xnnpack/common.h"

// rsqrtps and the like do not produce consistent results across
// microarchitectures, so we use this famous (and portable) approximation
// instead: https://en.wikipedia.org/wiki/Fast_inverse_square_root
XNN_ALIGN(64)
static const int32_t approx_reciprocal_sqrt_magic[] = {
    0x5F375A86, 0x5F375A86, 0x5F375A86, 0x5F375A86, 0x5F375A86,
    0x5F375A86, 0x5F375A86, 0x5F375A86, 0x5F375A86, 0x5F375A86,
    0x5F375A86, 0x5F375A86, 0x5F375A86, 0x5F375A86, 0x5F375A86,
    0x5F375A86, 0x5F375A86, 0x5F375A86, 0x5F375A86, 0x5F375A86,
};

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_COMMON_H_
