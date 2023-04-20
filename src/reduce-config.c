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
#include <xnnpack/reduce.h>


static struct xnn_reduce_config f32_rsum_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f32_rsum = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f32_rsum = PTHREAD_ONCE_INIT;
#endif

static void init_f32_rsum_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__neon_x16_acc4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
        .element_tile = 16,
      };
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__scalar_x4_acc4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
        .element_tile = 4,
      };
    }
  #elif XNN_ARCH_ARM64
    f32_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__neon_x16_acc4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
      .element_tile = 16,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx) {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__avx_x32_acc4,
        .init.f32_scale = xnn_init_f32_scale_avx_params,
        .element_tile = 32,
      };
    } else {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__sse_x16_acc4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
        .element_tile = 16,
      };
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__wasmsimd_x16_acc4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
      .element_tile = 16,
    };
  #elif XNN_ARCH_WASM
    f32_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__scalar_x4_acc4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
      .element_tile = 4,
    };
  #elif XNN_ARCH_RISCV
    f32_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__scalar_x4_acc4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
      .element_tile = 4,
    };
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f32_rsum_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_rsum_config();
    return TRUE;
  }
#endif

const struct xnn_reduce_config* xnn_init_f32_rsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_rsum, &init_f32_rsum_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_rsum, &init_f32_rsum_config);
  #endif
  return &f32_rsum_config;
}
