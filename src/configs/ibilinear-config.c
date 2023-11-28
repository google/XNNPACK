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
#include <xnnpack/ibilinear.h>


static struct xnn_ibilinear_config f16_ibilinear_config = {0};
static struct xnn_ibilinear_config f32_ibilinear_config = {0};
static struct xnn_ibilinear_config s8_ibilinear_config = {0};
static struct xnn_ibilinear_config u8_ibilinear_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_ibilinear = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_ibilinear = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_s8_ibilinear = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_u8_ibilinear = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_ibilinear = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_ibilinear = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_s8_ibilinear = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_u8_ibilinear = PTHREAD_ONCE_INIT;
#endif

static void init_f16_ibilinear_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f16_ibilinear_ukernel__neonfp16arith_c8;
      f16_ibilinear_config.pixel_tile = 1;
      f16_ibilinear_config.channel_tile = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f16_ibilinear_ukernel__neonfp16arith_c8;
      f16_ibilinear_config.pixel_tile = 1;
      f16_ibilinear_config.channel_tile = 8;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f16_ibilinear_ukernel__fma3_c8;
      f16_ibilinear_config.pixel_tile = 1;
      f16_ibilinear_config.channel_tile = 8;
    }
  #endif
}

static void init_f32_ibilinear_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f32_ibilinear_ukernel__neon_c8;
      f32_ibilinear_config.pixel_tile = 1;
      f32_ibilinear_config.channel_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f32_ibilinear_ukernel__scalar_c2;
      f32_ibilinear_config.pixel_tile = 1;
      f32_ibilinear_config.channel_tile = 2;
    }
  #elif XNN_ARCH_ARM64
    f32_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f32_ibilinear_ukernel__neonfma_c8;
    f32_ibilinear_config.pixel_tile = 1;
    f32_ibilinear_config.channel_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f32_ibilinear_ukernel__sse_c8;
    f32_ibilinear_config.pixel_tile = 1;
    f32_ibilinear_config.channel_tile = 8;
  #elif XNN_ARCH_WASMRELAXEDSIMD
    f32_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_c8;
    f32_ibilinear_config.pixel_tile = 1;
    f32_ibilinear_config.channel_tile = 8;
  #elif XNN_ARCH_WASMSIMD
    f32_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f32_ibilinear_ukernel__wasmsimd_c8;
    f32_ibilinear_config.pixel_tile = 1;
    f32_ibilinear_config.channel_tile = 8;
  #elif XNN_ARCH_WASM
    f32_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f32_ibilinear_ukernel__scalar_c2;
    f32_ibilinear_config.pixel_tile = 1;
    f32_ibilinear_config.channel_tile = 2;
  #elif XNN_ARCH_RISCV
    f32_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f32_ibilinear_ukernel__scalar_c2;
    f32_ibilinear_config.pixel_tile = 1;
    f32_ibilinear_config.channel_tile = 2;
  #elif XNN_ARCH_PPC64
    f32_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_f32_ibilinear_ukernel__scalar_c2;
    f32_ibilinear_config.pixel_tile = 1;
    f32_ibilinear_config.channel_tile = 2;
  #endif
}

static void init_s8_ibilinear_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      s8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_s8_ibilinear_ukernel__neon_c8;
      s8_ibilinear_config.pixel_tile = 1;
      s8_ibilinear_config.channel_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      s8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_s8_ibilinear_ukernel__scalar_c1;
      s8_ibilinear_config.pixel_tile = 1;
      s8_ibilinear_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    s8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_s8_ibilinear_ukernel__neon_c16;
    s8_ibilinear_config.pixel_tile = 1;
    s8_ibilinear_config.channel_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_sse4_1) {
      s8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_s8_ibilinear_ukernel__sse41_c16;
      s8_ibilinear_config.pixel_tile = 1;
      s8_ibilinear_config.channel_tile = 16;
    } else {
      s8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_s8_ibilinear_ukernel__sse2_c8;
      s8_ibilinear_config.pixel_tile = 1;
      s8_ibilinear_config.channel_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    s8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c8;
    s8_ibilinear_config.pixel_tile = 1;
    s8_ibilinear_config.channel_tile = 8;
  #elif XNN_ARCH_WASM
    s8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_s8_ibilinear_ukernel__scalar_c1;
    s8_ibilinear_config.pixel_tile = 1;
    s8_ibilinear_config.channel_tile = 1;
  #elif XNN_ARCH_RISCV
    s8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_s8_ibilinear_ukernel__scalar_c1;
    s8_ibilinear_config.pixel_tile = 1;
    s8_ibilinear_config.channel_tile = 1;
  #elif XNN_ARCH_PPC64
    s8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_s8_ibilinear_ukernel__scalar_c1;
    s8_ibilinear_config.pixel_tile = 1;
    s8_ibilinear_config.channel_tile = 1;
  #endif
}

static void init_u8_ibilinear_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      u8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_u8_ibilinear_ukernel__neon_c8;
      u8_ibilinear_config.pixel_tile = 1;
      u8_ibilinear_config.channel_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      u8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_u8_ibilinear_ukernel__scalar_c1;
      u8_ibilinear_config.pixel_tile = 1;
      u8_ibilinear_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    u8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_u8_ibilinear_ukernel__neon_c16;
    u8_ibilinear_config.pixel_tile = 1;
    u8_ibilinear_config.channel_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_sse4_1) {
      u8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_u8_ibilinear_ukernel__sse41_c16;
      u8_ibilinear_config.pixel_tile = 1;
      u8_ibilinear_config.channel_tile = 16;
    } else {
      u8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_u8_ibilinear_ukernel__sse2_c8;
      u8_ibilinear_config.pixel_tile = 1;
      u8_ibilinear_config.channel_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    u8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_u8_ibilinear_ukernel__wasmsimd_dot16x2_c8;
    u8_ibilinear_config.pixel_tile = 1;
    u8_ibilinear_config.channel_tile = 8;
  #elif XNN_ARCH_WASM
    u8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_u8_ibilinear_ukernel__scalar_c1;
    u8_ibilinear_config.pixel_tile = 1;
    u8_ibilinear_config.channel_tile = 1;
  #elif XNN_ARCH_RISCV
    u8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_u8_ibilinear_ukernel__scalar_c1;
    u8_ibilinear_config.pixel_tile = 1;
    u8_ibilinear_config.channel_tile = 1;
  #elif XNN_ARCH_PPC64
    u8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_u8_ibilinear_ukernel__scalar_c1;
    u8_ibilinear_config.pixel_tile = 1;
    u8_ibilinear_config.channel_tile = 1;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_ibilinear_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_ibilinear_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_ibilinear_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_ibilinear_config();
    return TRUE;
  }

  static BOOL CALLBACK init_s8_ibilinear_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_s8_ibilinear_config();
    return TRUE;
  }

  static BOOL CALLBACK init_u8_ibilinear_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_u8_ibilinear_config();
    return TRUE;
  }
#endif

const struct xnn_ibilinear_config* xnn_init_f16_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_ibilinear, &init_f16_ibilinear_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_ibilinear, &init_f16_ibilinear_config);
  #endif
  return &f16_ibilinear_config;
}

const struct xnn_ibilinear_config* xnn_init_f32_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_ibilinear, &init_f32_ibilinear_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_ibilinear, &init_f32_ibilinear_config);
  #endif
  return &f32_ibilinear_config;
}

const struct xnn_ibilinear_config* xnn_init_s8_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_s8_ibilinear, &init_s8_ibilinear_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_s8_ibilinear, &init_s8_ibilinear_config);
  #endif
  return &s8_ibilinear_config;
}

const struct xnn_ibilinear_config* xnn_init_u8_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_u8_ibilinear, &init_u8_ibilinear_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_u8_ibilinear, &init_u8_ibilinear_config);
  #endif
  return &u8_ibilinear_config;
}
