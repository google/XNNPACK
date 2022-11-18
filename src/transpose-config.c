// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <errno.h>
  #include <pthread.h>
  #include <sys/mman.h>
  #include <unistd.h>
#endif

#ifdef _MSC_VER
  #include <intrin.h>
#endif

#include <xnnpack/config.h>
#include <xnnpack/log.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/transpose.h>
#include <xnnpack/vunary.h>

static struct xnn_transpose_config transpose_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard = PTHREAD_ONCE_INIT;
#endif

static void init_transpose_config(void) {
#if XNN_ARCH_ARM
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config->use_arm_neon) {
    /*************************** TRANSPOSE AArch32 micro-kernels **********************/
      transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__memcpy;

      transpose_config.x8 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__16x16_reuse_dec_zip_neon,
        .tile_size = 32,
      };

      transpose_config.x16 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon,
        .tile_size = 32,
      };

      transpose_config.x32 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__4x4_reuse_dec_zip_neon,
        .tile_size = 32,
      };

      transpose_config.xx = (struct xnn_transpose_subconfig) {
        .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_memcpy,
        .tile_size = 32,
      };

  } else if (!XNN_PLATFORM_MOBILE) {
      transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__memcpy;

      transpose_config.x8 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__2x4_scalar_int,
        .tile_size = 32,
      };

      transpose_config.x16 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__2x4_scalar_int,
        .tile_size = 32,
      };

      transpose_config.x32 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__2x4_scalar_int,
        .tile_size = 32,
      };

      transpose_config.xx = (struct xnn_transpose_subconfig) {
        .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_memcpy,
        .tile_size = 32,
      };
  }

#elif XNN_ARCH_ARM64
  /*************************** TRANSPOSE AArch64 micro-kernels **********************/
    transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__memcpy;

    transpose_config.x8 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__16x16_reuse_dec_zip_neon,
      .tile_size = 32,
    };

    transpose_config.x16 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon,
      .tile_size = 32,
    };

    transpose_config.x32 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__4x4_neon_tbl128,
      .tile_size = 32,
      .init.x32 = (xnn_init_x32_transpose_params_fn) xnn_init_x32_transpose_neon_tbl128_params,
    };

    transpose_config.xx = (struct xnn_transpose_subconfig) {
      .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_memcpy,
      .tile_size = 32,
    };

#elif XNN_ARCH_X86 || XNN_ARCH_X86_64
  /*************************** TRANSPOSE x86 micro-kernels **********************/
    transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__memcpy;

    transpose_config.x8 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__16x16_reuse_mov_sse2,
      .tile_size = 32,
    };

    transpose_config.x16 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2,
      .tile_size = 32,
    };

    transpose_config.x32 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__4x4_sse,
      .tile_size = 32,
    };

    transpose_config.xx = (struct xnn_transpose_subconfig) {
      .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_memcpy,
      .tile_size = 32,
    };

#elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

  /*************************** TRANSPOSE WAsm SIMD micro-kernels***********************/
    transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__memcpy;

    transpose_config.x8 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__16x16_reuse_mov_wasmsimd,
      .tile_size = 32,
    };

    transpose_config.x16 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd,
      .tile_size = 32,
    };

    transpose_config.x32 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__4x4_reuse_mov_wasmsimd,
      .tile_size = 32,
    };

    transpose_config.xx = (struct xnn_transpose_subconfig) {
      .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_memcpy,
      .tile_size = 32,
    };

#elif XNN_ARCH_WASM

  /*************************** TRANSPOSE WAsm micro-kernels**********************/
    transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__memcpy;

    transpose_config.x8 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__2x4_scalar_int,
      .tile_size = 32,
    };

    transpose_config.x16 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__2x4_scalar_int,
      .tile_size = 32,
    };

    transpose_config.x32 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__2x4_scalar_int,
      .tile_size = 32,
    };

    transpose_config.xx = (struct xnn_transpose_subconfig) {
      .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_memcpy,
      .tile_size = 32,
    };

#elif XNN_ARCH_RISCV

  /************************** TRANSPOSE RISC-V micro-kernels *********************/
    transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__memcpy;

    transpose_config.x8 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__2x4_scalar_int,
      .tile_size = 32,
    };

    transpose_config.x16 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__2x4_scalar_int,
      .tile_size = 32,
    };

    transpose_config.x32 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__2x4_scalar_int,
      .tile_size = 32,
    };

    transpose_config.xx = (struct xnn_transpose_subconfig) {
      .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_memcpy,
      .tile_size = 32,
    };

#else
  #error "Unsupported architecture"
#endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_transpose_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_transpose_config();
    return TRUE;
  }
#endif

const struct xnn_transpose_config* xnn_init_transpose_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard, &init_transpose_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard, &init_transpose_config);
  #endif
  return &transpose_config;
}

