// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_DOT_ARM64_SME_H_
#define XNNPACK_YNNPACK_KERNELS_DOT_ARM64_SME_H_

#include <cstddef>
#include <cstdint>

namespace ynn {

// Get the size of an SME vector for the given datatype.
size_t sme_vl(float);
size_t sme_vl(int32_t);

#if __APPLE__ && __clang_major__ < 17
#define YNN_DISABLE_SME
#endif

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_ARM64_SME_H_
