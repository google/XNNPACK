// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_ISA_CHECKS_H_
#define XNNPACK_SRC_XNNPACK_ISA_CHECKS_H_

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"

inline size_t get_batch_scale(size_t element_size) {
#if XNN_ARCH_RISCV
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  return hardware_config
             ? std::max<size_t>(1, hardware_config->vlenb / element_size)
             : 1;
#else
  return 1;
#endif
}

template <typename T>
size_t get_batch_scale() {
  return get_batch_scale(sizeof(T));
}

#define TEST_REQUIRES_ARCH_FLAGS(FLAGS)                       \
  do {                                                        \
    const struct xnn_hardware_config* hardware_config =       \
        xnn_init_hardware_config();                           \
    if (hardware_config == nullptr ||                         \
        (hardware_config->arch_flags & (FLAGS)) != (FLAGS)) { \
      GTEST_SKIP();                                           \
    }                                                         \
  } while (0)

#endif  // XNNPACK_SRC_XNNPACK_ISA_CHECKS_H_
