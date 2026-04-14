// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dequantize_dot/dequantize_dot.h"

#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

dequantize_dot_kernel_fn get_dequantize_dot_kernel(ynn_type type,
                                                   uint64_t arch_flags) {
#define YNN_DEQUANTIZE_DOT_KERNEL(arch, name, output_type)                     \
  if (type == type_of<output_type>() && is_arch_supported(arch, arch_flags)) { \
    return name;                                                               \
  }
#include "ynnpack/kernels/dequantize_dot/kernels.inc"
#undef YNN_DEQUANTIZE_DOT_KERNEL
  return nullptr;
}

}  // namespace ynn
