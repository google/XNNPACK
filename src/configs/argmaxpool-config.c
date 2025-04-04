// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>


#include "src/xnnpack/argmaxpool.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/microfnptr.h"

static struct xnn_argmaxpool_config f32_argmaxpool_config = {0};

XNN_INIT_ONCE_GUARD(f32_argmaxpool);

static void init_f32_argmaxpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_argmaxpool_config = (struct xnn_argmaxpool_config) {
        .ukernel = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__neon_c4,
        .primary_tile = 9,
      };
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_argmaxpool_config = (struct xnn_argmaxpool_config) {
        .ukernel = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1,
        .primary_tile = 9,
      };
    }
  #elif XNN_ARCH_ARM64
    f32_argmaxpool_config = (struct xnn_argmaxpool_config) {
      .ukernel = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__neon_c4,
      .primary_tile = 9,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_argmaxpool_config = (struct xnn_argmaxpool_config) {
      .ukernel = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4,
      .primary_tile = 9,
    };
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_argmaxpool_config = (struct xnn_argmaxpool_config) {
      .ukernel = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__wasmsimd_c4,
      .primary_tile = 9,
    };
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    f32_argmaxpool_config = (struct xnn_argmaxpool_config) {
      .ukernel = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__rvv_u1v,
      .primary_tile = 9,
    };
  #else
    f32_argmaxpool_config = (struct xnn_argmaxpool_config) {
      .ukernel = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1,
      .primary_tile = 9,
    };
  #endif
}

const struct xnn_argmaxpool_config* xnn_init_f32_argmaxpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_argmaxpool);
  return &f32_argmaxpool_config;
}
