// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>


#include "xnnpack/argmaxpool.h"
#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"

static struct xnn_argmaxpool_config f32_argmaxpool_config[XNN_MAX_F32_ARGMAXPOOL_UKERNELS ] = {0};

XNN_INIT_ONCE_GUARD(f32_argmaxpool);

static void init_f32_argmaxpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_argmaxpool_config[0] = (struct xnn_argmaxpool_config) {
        .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_4x__neon_c4,
        .first_pass_tile_size = 4,
      };
      f32_argmaxpool_config[1] = (struct xnn_argmaxpool_config) {
        .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9x__neon_c4,
        .first_pass_tile_size = 9,
      };
      f32_argmaxpool_config[2] = (struct xnn_argmaxpool_config) {
        .mp = (xnn_argmaxpool_multipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__neon_c4,
        .first_pass_tile_size = 9,
        .remainder_pass_tile_size = 8,
      };
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_argmaxpool_config[0] = (struct xnn_argmaxpool_config) {
        .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_4x__scalar_c1,
        .first_pass_tile_size = 4,
      };
      f32_argmaxpool_config[1] = (struct xnn_argmaxpool_config) {
        .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9x__scalar_c1,
        .first_pass_tile_size = 9,
      };
      f32_argmaxpool_config[2] = (struct xnn_argmaxpool_config) {
        .mp = (xnn_argmaxpool_multipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1,
        .first_pass_tile_size = 9,
        .remainder_pass_tile_size = 8,
      };
    }
  #elif XNN_ARCH_ARM64
    f32_argmaxpool_config[0] = (struct xnn_argmaxpool_config) {
      .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_4x__neon_c4,
      .first_pass_tile_size = 4,
    };
    f32_argmaxpool_config[1] = (struct xnn_argmaxpool_config) {
      .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9x__neon_c4,
      .first_pass_tile_size = 9,
    };
    f32_argmaxpool_config[2] = (struct xnn_argmaxpool_config) {
      .mp = (xnn_argmaxpool_multipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__neon_c4,
      .first_pass_tile_size = 9,
      .remainder_pass_tile_size = 8,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_argmaxpool_config[0] = (struct xnn_argmaxpool_config) {
      .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_4x__sse2_c4,
      .first_pass_tile_size = 4,
    };
    f32_argmaxpool_config[1] = (struct xnn_argmaxpool_config) {
      .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9x__sse2_c4,
      .first_pass_tile_size = 9,
    };
    f32_argmaxpool_config[2] = (struct xnn_argmaxpool_config) {
      .mp = (xnn_argmaxpool_multipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4,
      .first_pass_tile_size = 9,
      .remainder_pass_tile_size = 8,
    };
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_argmaxpool_config[0] = (struct xnn_argmaxpool_config) {
      .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_4x__wasmsimd_c4,
      .first_pass_tile_size = 4,
    };
    f32_argmaxpool_config[1] = (struct xnn_argmaxpool_config) {
      .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9x__wasmsimd_c4,
      .first_pass_tile_size = 9,
    };
    f32_argmaxpool_config[2] = (struct xnn_argmaxpool_config) {
      .mp = (xnn_argmaxpool_multipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__wasmsimd_c4,
      .first_pass_tile_size = 9,
      .remainder_pass_tile_size = 8,
    };
  #else
    f32_argmaxpool_config[0] = (struct xnn_argmaxpool_config) {
      .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_4x__scalar_c1,
      .first_pass_tile_size = 4,
    };
    f32_argmaxpool_config[1] = (struct xnn_argmaxpool_config) {
      .up = (xnn_argmaxpool_unipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9x__scalar_c1,
      .first_pass_tile_size = 9,
    };
    f32_argmaxpool_config[2] = (struct xnn_argmaxpool_config) {
      .mp = (xnn_argmaxpool_multipass_ukernel_fn) xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1,
      .first_pass_tile_size = 9,
      .remainder_pass_tile_size = 8,
    };
  #endif
}

const struct xnn_argmaxpool_config* xnn_init_f32_argmaxpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_argmaxpool);
  return f32_argmaxpool_config;
}
