// Copyright 2022 Google LLC
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
#include <xnnpack/lut.h>


static struct xnn_x8_lut_config x8_lut_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard = PTHREAD_ONCE_INIT;
#endif

static void init_x8_lut_config(void) {
  #if XNN_ARCH_ARM
    x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u4;
    x8_lut_config.tile_size = 4;
  #elif XNN_ARCH_ARM64
    x8_lut_config.microkernel = xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u64;
    x8_lut_config.tile_size = 64;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      if (hardware_config->use_x86_avx512vbmi) {
        x8_lut_config.microkernel = xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u128;
        x8_lut_config.tile_size = 128;
      } else {
        x8_lut_config.microkernel = xnn_x8_lut_ukernel__avx512skx_vpshufb_u64;
        x8_lut_config.tile_size = 64;
      }
    } else if (hardware_config->use_x86_avx2) {
      x8_lut_config.microkernel = xnn_x8_lut_ukernel__avx2_u128;
      x8_lut_config.tile_size = 128;
    } else if (hardware_config->use_x86_avx) {
      x8_lut_config.microkernel = xnn_x8_lut_ukernel__avx_u64;
      x8_lut_config.tile_size = 64;
    } else {
      // Note: SSSE3 version is usually slower than scalar
      x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u4;
      x8_lut_config.tile_size = 4;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    if (hardware_config->is_x86) {
      #if XNN_ARCH_WASMRELAXEDSIMD
        if (hardware_config->use_wasm_pshufb) {
          x8_lut_config.microkernel = xnn_x8_lut_ukernel__wasmpshufb_u32;
          x8_lut_config.tile_size = 32;
        } else {
          x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u1;
          x8_lut_config.tile_size = 1;
        }
      #else
        x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u1;
        x8_lut_config.tile_size = 1;
      #endif
    } else {
      x8_lut_config.microkernel = xnn_x8_lut_ukernel__wasmsimd_u32;
      x8_lut_config.tile_size = 32;
    }
  #elif XNN_ARCH_WASM
    x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u1;
    x8_lut_config.tile_size = 1;
  #elif XNN_ARCH_RISCV
    x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u4;
    x8_lut_config.tile_size = 4;
  #elif XNN_ARCH_PPC64
    x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u4;
    x8_lut_config.tile_size = 4;
  #else
    #error "Unsupported architecture"
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_x8_lut_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_x8_lut_config();
    return TRUE;
  }
#endif

const struct xnn_x8_lut_config* xnn_init_x8_lut_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard, &init_x8_lut_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard, &init_x8_lut_config);
  #endif
  return &x8_lut_config;
}
