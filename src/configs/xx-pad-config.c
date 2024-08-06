// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/pad.h"

static struct xnn_xx_pad_config xx_pad_config = {0};

XNN_INIT_ONCE_GUARD(xx_pad);

static void init_xx_pad_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      xx_pad_config.ukernel = (xnn_pad_ukernel_fn) xnn_xx_pad_ukernel_p16__neon_u16;
      xx_pad_config.row_tile = 1;
    } else if (!XNN_PLATFORM_MOBILE) {
      xx_pad_config.ukernel = (xnn_pad_ukernel_fn) xnn_xx_pad_ukernel_p4__scalar_u16;
      xx_pad_config.row_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    xx_pad_config.ukernel = (xnn_pad_ukernel_fn) xnn_xx_pad_ukernel_p16__neon_u16;
    xx_pad_config.row_tile = 1;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    xx_pad_config.ukernel = (xnn_pad_ukernel_fn) xnn_xx_pad_ukernel_p16__sse2_u16;
    xx_pad_config.row_tile = 1;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    xx_pad_config.ukernel = (xnn_pad_ukernel_fn) xnn_xx_pad_ukernel_p16__wasmsimd_u16;
    xx_pad_config.row_tile = 1;
  #else
    xx_pad_config.ukernel = (xnn_pad_ukernel_fn) xnn_xx_pad_ukernel_p4__scalar_u16;
    xx_pad_config.row_tile = 1;
  #endif
}

const struct xnn_xx_pad_config* xnn_init_xx_pad_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(xx_pad);
  return &xx_pad_config;
}
