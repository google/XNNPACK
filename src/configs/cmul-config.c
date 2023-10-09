// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <pthread.h>
#endif

#include <xnnpack/common.h>
#include <xnnpack/config.h>
#include <xnnpack/vbinary.h>


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static struct xnn_cmul_config f16_cmul_config = {0};
#endif
static struct xnn_cmul_config f32_cmul_config = {0};


#if XNN_PLATFORM_WINDOWS
  #if XNN_ARCH_ARM || XNN_ARCH_ARM64
    static INIT_ONCE init_guard_f16_cmul = INIT_ONCE_STATIC_INIT;
  #endif
  static INIT_ONCE init_guard_f32_cmul = INIT_ONCE_STATIC_INIT;
#else
  #if XNN_ARCH_ARM || XNN_ARCH_ARM64
    static pthread_once_t init_guard_f16_cmul = PTHREAD_ONCE_INIT;
  #endif
  static pthread_once_t init_guard_f32_cmul = PTHREAD_ONCE_INIT;
#endif


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void init_f16_cmul_config(void) {
      f16_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vcmul_ukernel__neonfp16arith_u16;
      f16_cmul_config.element_tile = 16;
  }
#endif

static void init_f32_cmul_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__neon_u8;
      f32_cmul_config.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__scalar_u4;
      f32_cmul_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__neon_u8;
    f32_cmul_config.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__sse_u8;
    f32_cmul_config.element_tile = 8;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__wasmsimd_u8;
    f32_cmul_config.element_tile = 8;
  #elif XNN_ARCH_WASM
    f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__scalar_u4;
    f32_cmul_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__scalar_u4;
    f32_cmul_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__scalar_u4;
    f32_cmul_config.element_tile = 4;
  #endif
}


#if XNN_PLATFORM_WINDOWS && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static BOOL CALLBACK init_f16_cmul_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_cmul_config();
    return TRUE;
  }
#endif

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f32_cmul_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_cmul_config();
    return TRUE;
  }
#endif


const struct xnn_cmul_config* xnn_init_f16_cmul_config() {
  #if XNN_ARCH_ARM || XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
      return NULL;
    }
    #if XNN_PLATFORM_WINDOWS
      InitOnceExecuteOnce(&init_guard_f16_cmul, &init_f16_cmul_config_windows, NULL, NULL);
    #else
      pthread_once(&init_guard_f16_cmul, &init_f16_cmul_config);
    #endif
    return &f16_cmul_config;
  #else
    return NULL;
  #endif
}

const struct xnn_cmul_config* xnn_init_f32_cmul_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_cmul, &init_f32_cmul_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_cmul, &init_f32_cmul_config);
  #endif
  return &f32_cmul_config;
}
