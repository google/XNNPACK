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

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"

static struct xnn_reduce_config f16_f32acc_rsum_config = {0};
static struct xnn_reduce_config f16_f32acc_rdsum_config = {0};
static struct xnn_reduce_config f16_rminmax_config = {0};
static struct xnn_reduce_config f32_rminmax_config = {0};
static struct xnn_reduce_config f32_rsum_config = {0};
static struct xnn_reduce_config f32_rdsum_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_f32acc_rsum = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_f32acc_rdsum = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_rminmax = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_rminmax = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_rsum = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_rdsum = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_f32acc_rsum = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_f32acc_rdsum = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_rminmax = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_rminmax = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_rsum = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_rdsum = PTHREAD_ONCE_INIT;
#endif

static void init_f16_f32acc_rsum_config(void) {
  #if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_f32acc_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u32_acc4,
        .init.f16_f32acc_scale = xnn_init_f16_f32acc_scale_scalar_params,
        .element_tile = 32,
      };
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64)
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      f16_f32acc_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f16_f32acc_rsum_ukernel__avx512skx_u64_acc4,
        .init.f16_f32acc_scale = xnn_init_f16_f32acc_scale_scalar_params,
        .element_tile = 64,
      };
    } else if (hardware_config->use_x86_f16c) {
      f16_f32acc_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4,
        .init.f16_f32acc_scale = xnn_init_f16_f32acc_scale_avx_params,
        .element_tile = 32,
      };
    }
  #endif
}

static void init_f16_rminmax_config(void) {
  #if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rminmax_config.ukernel = (xnn_reduce_ukernel_fn) xnn_f16_rminmax_ukernel__neonfp16arith_u32_acc4;
    } else {
      f16_rminmax_config.ukernel = (xnn_reduce_ukernel_fn) xnn_f16_rminmax_ukernel__scalar_u2_acc2;
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->use_x86_avx512fp16) {
        f16_rminmax_config.ukernel = (xnn_reduce_ukernel_fn) xnn_f16_rminmax_ukernel__avx512fp16_u128_acc4;
      } else
    #endif
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      f16_rminmax_config.ukernel = (xnn_reduce_ukernel_fn) xnn_f16_rminmax_ukernel__avx512skx_u64_acc4;
    } else {
      f16_rminmax_config.ukernel = (xnn_reduce_ukernel_fn) xnn_f16_rminmax_ukernel__scalar_u2_acc2;
    }
  #else
    f16_rminmax_config.ukernel = (xnn_reduce_ukernel_fn) xnn_f16_rminmax_ukernel__scalar_u2_acc2;
  #endif
}

static void init_f32_rminmax_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_rminmax_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__neon_u16_acc4,
        .element_tile = 16,
      };
    } else {
      f32_rminmax_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__scalar_u4_acc4,
        .element_tile = 4,
      };
    }
  #elif XNN_ARCH_ARM64
    f32_rminmax_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__neon_u16_acc4,
      .element_tile = 16,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_rminmax_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__avx512f_u64_acc4,
        .element_tile = 64,
      };
    } else if (hardware_config->use_x86_avx) {
      f32_rminmax_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__avx_u32_acc4,
        .init.f32_default = xnn_init_f32_default_avx_params,
        .element_tile = 32,
      };
    } else {
      f32_rminmax_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__sse_u16_acc4,
        .element_tile = 16,
      };
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rminmax_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__wasmsimd_minmax_u16_acc4,
      .element_tile = 16,
    };
  #elif XNN_ARCH_WASM
    f32_rminmax_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__wasm_u4_acc4,
      .element_tile = 4,
    };
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_rminmax_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__rvv_u8v,
      .element_tile = hardware_config->vlenb * 2,  // VLENB * (8 / sizeof(float))
    };
  #else
    f32_rminmax_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__scalar_u4_acc4,
      .element_tile = 4,
    };
  #endif
}

static void init_f32_rsum_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__neon_u16_acc4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
        .element_tile = 16,
      };
    } else {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__scalar_u4_acc4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
        .element_tile = 4,
      };
    }
  #elif XNN_ARCH_ARM64
    f32_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__neon_u16_acc4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
      .element_tile = 16,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__avx512f_u64_acc4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
        .element_tile = 64,
      };
    } else if (hardware_config->use_x86_avx) {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__avx_u32_acc4,
        .init.f32_scale = xnn_init_f32_scale_avx_params,
        .element_tile = 32,
      };
    } else {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__sse_u16_acc4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
        .element_tile = 16,
      };
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__wasmsimd_u16_acc4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
      .element_tile = 16,
    };
  #else
    f32_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__scalar_u4_acc4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
      .element_tile = 4,
    };
  #endif
}

static void init_f16_f32acc_rdsum_config(void) {
  #if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_f32acc_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16,
        .init.f16_f32acc_scale = xnn_init_f16_f32acc_scale_scalar_params,
        .element_tile = 16,
      };
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      f16_f32acc_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64,
        .init.f16_f32acc_scale = xnn_init_f16_f32acc_scale_scalar_params,
        .element_tile = 64,
      };
    } else if (hardware_config->use_x86_f16c) {
      f16_f32acc_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32,
        .init.f16_f32acc_scale = xnn_init_f16_f32acc_scale_avx_params,
        .element_tile = 32,
      };
    }
  #endif
}

static void init_f32_rdsum_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__neon_c16,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
        .element_tile = 16,
      };
    } else {
      f32_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__scalar_c4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
        .element_tile = 4,
      };
    }
  #elif XNN_ARCH_ARM64
    f32_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__neon_c16,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
      .element_tile = 16,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__avx512f_c64,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
        .element_tile = 64,
      };
    } else if (hardware_config->use_x86_avx) {
      f32_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__avx_c32,
        .init.f32_scale = xnn_init_f32_scale_avx_params,
        .element_tile = 32,
      };
    } else {
      f32_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__sse_c16,
        .init.f32_scale = xnn_init_f32_scale_sse_params,
        .element_tile = 16,
      };
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
      .element_tile = 16,
    };
  #else
    f32_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__scalar_c4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
      .element_tile = 4,
    };
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_f32acc_rsum_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_f32acc_rsum_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_f32acc_rdsum_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_f32acc_rdsum_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_rminmax_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_rminmax_config();
    return TRUE;
  }
  static BOOL CALLBACK init_f32_rminmax_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_rminmax_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_rsum_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_rsum_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_rdsum_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_rdsum_config();
    return TRUE;
  }
#endif

const struct xnn_reduce_config* xnn_init_f16_f32acc_rsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_f32acc_rsum, &init_f16_f32acc_rsum_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_f32acc_rsum, &init_f16_f32acc_rsum_config);
  #endif
  return &f16_f32acc_rsum_config;
}

const struct xnn_reduce_config* xnn_init_f16_rminmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_rminmax, &init_f16_rminmax_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_rminmax, &init_f16_rminmax_config);
  #endif
  return &f16_rminmax_config;
}

const struct xnn_reduce_config* xnn_init_f32_rminmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_rminmax, &init_f32_rminmax_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_rminmax, &init_f32_rminmax_config);
  #endif
  return &f32_rminmax_config;
}

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

const struct xnn_reduce_config* xnn_init_f16_f32acc_rdsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_f32acc_rdsum, &init_f16_f32acc_rdsum_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_f32acc_rdsum, &init_f16_f32acc_rdsum_config);
  #endif
  return &f16_f32acc_rdsum_config;
}

const struct xnn_reduce_config* xnn_init_f32_rdsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_rdsum, &init_f32_rdsum_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_rdsum, &init_f32_rdsum_config);
  #endif
  return &f32_rdsum_config;
}
