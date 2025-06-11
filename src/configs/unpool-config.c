// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/unpool.h"

static struct xnn_unpool_config x32_unpool_config = {0};

XNN_INIT_ONCE_GUARD(x32_unpool);

static void init_x32_unpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__neon;
    } else {
      x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__scalar;
    }
  #elif XNN_ARCH_ARM64
    x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__neon;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__sse2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__wasmsimd;
  #else
    x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__scalar;
  #endif

}

const struct xnn_unpool_config* xnn_init_x32_unpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x32_unpool);
  return &x32_unpool_config;
}
