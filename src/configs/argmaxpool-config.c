// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <pthread.h>
#endif

#include <xnnpack/common.h>
#include <xnnpack/config.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/argmaxpool.h>


static struct xnn_argmaxpool_config f32_argmaxpool_config[XNN_MAX_F32_ARGMAXPOOL_UKERNELS ] = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f32_argmaxpool = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f32_argmaxpool = PTHREAD_ONCE_INIT;
#endif

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
  #elif XNN_ARCH_WASM
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
  #elif XNN_ARCH_RISCV
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
  #elif XNN_ARCH_PPC64
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

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f32_argmaxpool_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_argmaxpool_config();
    return TRUE;
  }
#endif

const struct xnn_argmaxpool_config* xnn_init_f32_argmaxpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_argmaxpool, &init_f32_argmaxpool_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_argmaxpool, &init_f32_argmaxpool_config);
  #endif
  return f32_argmaxpool_config;
}
