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
#include <xnnpack/maxpool.h>


static struct xnn_maxpool_config f16_maxpool_config = {0};
static struct xnn_maxpool_config f32_maxpool_config = {0};
static struct xnn_maxpool_config s8_maxpool_config = {0};
static struct xnn_maxpool_config u8_maxpool_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_maxpool = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_maxpool = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_s8_maxpool = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_u8_maxpool = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_maxpool = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_maxpool = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_s8_maxpool = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_u8_maxpool = PTHREAD_ONCE_INIT;
#endif

static void init_f16_maxpool_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f16_maxpool_minmax_ukernel_9p8x__neonfp16arith_c8;
      f16_maxpool_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_maxpool_config.first_pass_tile_size = 9;
      f16_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f16_maxpool_minmax_ukernel_9p8x__neonfp16arith_c8;
      f16_maxpool_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_maxpool_config.first_pass_tile_size = 9;
      f16_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f16_maxpool_minmax_ukernel_9p8x__f16c_c8;
      f16_maxpool_config.init.f16 = xnn_init_f16_minmax_avx_params;
      f16_maxpool_config.first_pass_tile_size = 9;
      f16_maxpool_config.remainder_pass_tile_size = 8;
    }
  #endif
}

static void init_f32_maxpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4;
      f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_maxpool_config.first_pass_tile_size = 9;
      f32_maxpool_config.remainder_pass_tile_size = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1;
      f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_maxpool_config.first_pass_tile_size = 9;
      f32_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_ARM64
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_maxpool_config.first_pass_tile_size = 9;
    f32_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_sse_params;
    f32_maxpool_config.first_pass_tile_size = 9;
    f32_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4;
      f32_maxpool_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_maxpool_config.first_pass_tile_size = 9;
      f32_maxpool_config.remainder_pass_tile_size = 8;
    } else {
      f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4;
      f32_maxpool_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_maxpool_config.first_pass_tile_size = 9;
      f32_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_WASM
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_maxpool_config.first_pass_tile_size = 9;
    f32_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_RISCV
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_maxpool_config.first_pass_tile_size = 9;
    f32_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_PPC64
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_maxpool_config.first_pass_tile_size = 9;
    f32_maxpool_config.remainder_pass_tile_size = 8;
  #endif
}

static void init_s8_maxpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__neon_c16;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_neon_params;
      s8_maxpool_config.first_pass_tile_size = 9;
      s8_maxpool_config.remainder_pass_tile_size = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
      s8_maxpool_config.first_pass_tile_size = 9;
      s8_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_ARM64
    s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__neon_c16;
    s8_maxpool_config.init.s8 = xnn_init_s8_minmax_neon_params;
    s8_maxpool_config.first_pass_tile_size = 9;
    s8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_sse4_1) {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__sse41_c16;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_sse4_params;
      s8_maxpool_config.first_pass_tile_size = 9;
      s8_maxpool_config.remainder_pass_tile_size = 8;
    } else {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__sse2_c16;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_sse2_params;
      s8_maxpool_config.first_pass_tile_size = 9;
      s8_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__wasmsimd_c16;
    s8_maxpool_config.init.s8 = xnn_init_s8_minmax_wasmsimd_params;
    s8_maxpool_config.first_pass_tile_size = 9;
    s8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_WASM
    s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1;
    s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
    s8_maxpool_config.first_pass_tile_size = 9;
    s8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_RISCV
    s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1;
    s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
    s8_maxpool_config.first_pass_tile_size = 9;
    s8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_PPC64
    s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1;
    s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
    s8_maxpool_config.first_pass_tile_size = 9;
    s8_maxpool_config.remainder_pass_tile_size = 8;
  #endif
}

static void init_u8_maxpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16;
      u8_maxpool_config.init.u8 = xnn_init_u8_minmax_neon_params;
      u8_maxpool_config.first_pass_tile_size = 9;
      u8_maxpool_config.remainder_pass_tile_size = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1;
      u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
      u8_maxpool_config.first_pass_tile_size = 9;
      u8_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_ARM64
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_neon_params;
    u8_maxpool_config.first_pass_tile_size = 9;
    u8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_sse2_params;
    u8_maxpool_config.first_pass_tile_size = 9;
    u8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__wasmsimd_c16;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_wasmsimd_params;
    u8_maxpool_config.first_pass_tile_size = 9;
    u8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_WASM
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
    u8_maxpool_config.first_pass_tile_size = 9;
    u8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_RISCV
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
    u8_maxpool_config.first_pass_tile_size = 9;
    u8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_PPC64
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
    u8_maxpool_config.first_pass_tile_size = 9;
    u8_maxpool_config.remainder_pass_tile_size = 8;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_maxpool_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_maxpool_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_maxpool_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_maxpool_config();
    return TRUE;
  }

  static BOOL CALLBACK init_s8_maxpool_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_s8_maxpool_config();
    return TRUE;
  }

  static BOOL CALLBACK init_u8_maxpool_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_u8_maxpool_config();
    return TRUE;
  }
#endif

const struct xnn_maxpool_config* xnn_init_f16_maxpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_maxpool, &init_f16_maxpool_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_maxpool, &init_f16_maxpool_config);
  #endif
  return &f16_maxpool_config;
}

const struct xnn_maxpool_config* xnn_init_f32_maxpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_maxpool, &init_f32_maxpool_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_maxpool, &init_f32_maxpool_config);
  #endif
  return &f32_maxpool_config;
}

const struct xnn_maxpool_config* xnn_init_s8_maxpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_s8_maxpool, &init_s8_maxpool_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_s8_maxpool, &init_s8_maxpool_config);
  #endif
  return &s8_maxpool_config;
}

const struct xnn_maxpool_config* xnn_init_u8_maxpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_u8_maxpool, &init_u8_maxpool_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_u8_maxpool, &init_u8_maxpool_config);
  #endif
  return &u8_maxpool_config;
}
