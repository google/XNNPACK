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
#include <xnnpack/spmm.h>


static struct xnn_spmm_config f16_spmm_config = {0};
static struct xnn_spmm_config f32_spmm_config = {0};
static struct xnn_spmm_config f32_spmm2_config = {0};
static struct xnn_spmm_config f32_spmm4_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_spmm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_spmm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_spmm2 = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_spmm4 = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_spmm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_spmm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_spmm2 = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_spmm4 = PTHREAD_ONCE_INIT;
#endif

static void init_f16_spmm_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined;
      f16_spmm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_spmm_config.mr = 32;
      f16_spmm_config.nr = 1;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined;
      f16_spmm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_spmm_config.mr = 32;
      f16_spmm_config.nr = 1;
    }
  #endif
}

static void init_f32_spmm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_32x1__neon;
      f32_spmm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_spmm_config.mr = 32;
      f32_spmm_config.nr = 1;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x1__scalar;
      f32_spmm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_spmm_config.mr = 8;
      f32_spmm_config.nr = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_32x1__neonfma_pipelined;
    f32_spmm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm_config.mr = 32;
    f32_spmm_config.nr = 1;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_32x1__sse;
    f32_spmm_config.init.f32 = xnn_init_f32_minmax_sse_params;
    f32_spmm_config.mr = 32;
    f32_spmm_config.nr = 1;
  #elif XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86;
      f32_spmm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_spmm_config.mr = 32;
      f32_spmm_config.nr = 1;
    } else {
      f32_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm;
      f32_spmm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_spmm_config.mr = 32;
      f32_spmm_config.nr = 1;
    }
  #elif XNN_ARCH_WASMSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86;
      f32_spmm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_spmm_config.mr = 32;
      f32_spmm_config.nr = 1;
    } else {
      f32_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm;
      f32_spmm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_spmm_config.mr = 32;
      f32_spmm_config.nr = 1;
    }
  #elif XNN_ARCH_WASM
    f32_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x1__scalar;
    f32_spmm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm_config.mr = 8;
    f32_spmm_config.nr = 1;
  #elif XNN_ARCH_RISCV
    f32_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x1__scalar;
    f32_spmm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm_config.mr = 8;
    f32_spmm_config.nr = 1;
  #elif XNN_ARCH_PPC64
    f32_spmm_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x1__scalar;
    f32_spmm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm_config.mr = 8;
    f32_spmm_config.nr = 1;
  #endif
}

static void init_f32_spmm2_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_spmm2_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x2__scalar;
      f32_spmm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_spmm2_config.mr = 8;
      f32_spmm2_config.nr = 2;
    }
  #elif XNN_ARCH_ARM64
    f32_spmm2_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_32x2__aarch64_neonfma;
    f32_spmm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm2_config.mr = 32;
    f32_spmm2_config.nr = 2;
  #elif XNN_ARCH_WASM
    f32_spmm2_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x2__scalar;
    f32_spmm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm2_config.mr = 8;
    f32_spmm2_config.nr = 2;
  #elif XNN_ARCH_RISCV
    f32_spmm2_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x2__scalar;
    f32_spmm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm2_config.mr = 8;
    f32_spmm2_config.nr = 2;
  #elif XNN_ARCH_PPC64
    f32_spmm2_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x2__scalar;
    f32_spmm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm2_config.mr = 8;
    f32_spmm2_config.nr = 2;
  #endif
}

static void init_f32_spmm4_config(void) {
  #if XNN_ARCH_ARM
    if (!XNN_PLATFORM_MOBILE) {
      f32_spmm4_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x4__scalar;
      f32_spmm4_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_spmm4_config.mr = 8;
      f32_spmm4_config.nr = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_spmm4_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_32x4__aarch64_neonfma;
    f32_spmm4_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm4_config.mr = 32;
    f32_spmm4_config.nr = 4;
  #elif XNN_ARCH_WASM
    f32_spmm4_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x4__scalar;
    f32_spmm4_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm4_config.mr = 8;
    f32_spmm4_config.nr = 4;
  #elif XNN_ARCH_RISCV
    f32_spmm4_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x4__scalar;
    f32_spmm4_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm4_config.mr = 8;
    f32_spmm4_config.nr = 4;
  #elif XNN_ARCH_PPC64
    f32_spmm4_config.ukernel = (xnn_spmm_ukernel_fn) xnn_f32_spmm_minmax_ukernel_8x4__scalar;
    f32_spmm4_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_spmm4_config.mr = 8;
    f32_spmm4_config.nr = 4;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_spmm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_spmm_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_spmm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_spmm_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_spmm2_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_spmm2_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_spmm4_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_spmm4_config();
    return TRUE;
  }
#endif

const struct xnn_spmm_config* xnn_init_f16_spmm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_spmm, &init_f16_spmm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_spmm, &init_f16_spmm_config);
  #endif
  return &f16_spmm_config;
}

const struct xnn_spmm_config* xnn_init_f32_spmm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_spmm, &init_f32_spmm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_spmm, &init_f32_spmm_config);
  #endif
  return &f32_spmm_config;
}

const struct xnn_spmm_config* xnn_init_f32_spmm2_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_spmm2, &init_f32_spmm2_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_spmm2, &init_f32_spmm2_config);
  #endif
  return &f32_spmm2_config;
}

const struct xnn_spmm_config* xnn_init_f32_spmm4_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_spmm4, &init_f32_spmm4_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_spmm4, &init_f32_spmm4_config);
  #endif
  return &f32_spmm4_config;
}
