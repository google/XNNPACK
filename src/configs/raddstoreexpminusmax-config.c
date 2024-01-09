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
#include <xnnpack/raddstoreexpminusmax.h>


static struct xnn_raddstoreexpminusmax_config f16_raddstoreexpminusmax_config = {0};
static struct xnn_raddstoreexpminusmax_config f32_raddstoreexpminusmax_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_raddstoreexpminusmax = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_raddstoreexpminusmax = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_raddstoreexpminusmax = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_raddstoreexpminusmax = PTHREAD_ONCE_INIT;
#endif

static void init_f16_raddstoreexpminusmax_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32;
      f16_raddstoreexpminusmax_config.init.f16 = xnn_init_f16_expminus_fp16arith_rr2_p2_params;
      f16_raddstoreexpminusmax_config.element_tile = 32;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40;
      f16_raddstoreexpminusmax_config.init.f16 = xnn_init_f16_expminus_fp16arith_rr2_p2_params;
      f16_raddstoreexpminusmax_config.element_tile = 40;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40;
      f16_raddstoreexpminusmax_config.init.f16 = xnn_init_f16_expminus_avx2_rr1_p2_params;
      f16_raddstoreexpminusmax_config.element_tile = 40;
    }
  #endif
}

static void init_f32_raddstoreexpminusmax_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8;
      f32_raddstoreexpminusmax_config.init.f32 = xnn_init_f32_expminus_neon_rr2_lut64_p2_params;
      f32_raddstoreexpminusmax_config.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2;
      f32_raddstoreexpminusmax_config.init.f32 = xnn_init_f32_expminus_scalar_rr2_p5_params;
      f32_raddstoreexpminusmax_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_raddstoreexpminusmax_config.ukernel =
      (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16;
    f32_raddstoreexpminusmax_config.init.f32 = xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params;
    f32_raddstoreexpminusmax_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_raddstoreexpminusmax_config.ukernel =
      (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20_acc2;
    f32_raddstoreexpminusmax_config.init.f32 = xnn_init_f32_expminus_sse2_rr2_p5_params;
    f32_raddstoreexpminusmax_config.element_tile = 20;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f32_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2;
      f32_raddstoreexpminusmax_config.init.f32 = xnn_init_f32_expminus_wasmsimd_rr2_p5_params;
      f32_raddstoreexpminusmax_config.element_tile = 16;
    #else
      f32_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2;
      f32_raddstoreexpminusmax_config.init.f32 = xnn_init_f32_expminus_wasmsimd_rr2_p5_params;
      f32_raddstoreexpminusmax_config.element_tile = 16;
    #endif
  #elif XNN_ARCH_WASM
    f32_raddstoreexpminusmax_config.ukernel =
      (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2;
    f32_raddstoreexpminusmax_config.init.f32 = xnn_init_f32_expminus_scalar_rr2_p5_params;
    f32_raddstoreexpminusmax_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    #if XNN_ENABLE_RISCV_VECTOR
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      f32_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v;
      f32_raddstoreexpminusmax_config.init.f32 = xnn_init_f32_expminus_rvv_rr2_p6_params;
      f32_raddstoreexpminusmax_config.element_tile = hardware_config->vlenb;  // VLENB * (4 / sizeof(float))
    #else
      f32_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2;
      f32_raddstoreexpminusmax_config.init.f32 = xnn_init_f32_expminus_scalar_rr2_p5_params;
      f32_raddstoreexpminusmax_config.element_tile = 4;
    #endif
  #elif XNN_ARCH_PPC64
    f32_raddstoreexpminusmax_config.ukernel =
      (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2;
    f32_raddstoreexpminusmax_config.init.f32 = xnn_init_f32_expminus_scalar_rr2_p5_params;
    f32_raddstoreexpminusmax_config.element_tile = 4;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_raddstoreexpminusmax_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_raddstoreexpminusmax_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_raddstoreexpminusmax_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_raddstoreexpminusmax_config();
    return TRUE;
  }
#endif

static bool is_f16_compatible_config(const struct xnn_hardware_config hardware_config[restrict XNN_MIN_ELEMENTS(1)]) {
  #if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR) || (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR)
    return hardware_config->use_arm_neon_fp16_arith;
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    return hardware_config->use_x86_avx2;
  #else
    return false;
  #endif
}

const struct xnn_raddstoreexpminusmax_config* xnn_init_f16_raddstoreexpminusmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_raddstoreexpminusmax, &init_f16_raddstoreexpminusmax_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_raddstoreexpminusmax, &init_f16_raddstoreexpminusmax_config);
  #endif
  return &f16_raddstoreexpminusmax_config;
}

const struct xnn_raddstoreexpminusmax_config* xnn_init_f32_raddstoreexpminusmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_raddstoreexpminusmax, &init_f32_raddstoreexpminusmax_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_raddstoreexpminusmax, &init_f32_raddstoreexpminusmax_config);
  #endif
  return &f32_raddstoreexpminusmax_config;
}
