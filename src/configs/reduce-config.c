// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"

static struct xnn_reduce_config f16_f32acc_rsum_config = {0};
static struct xnn_reduce_config f16_f32acc_rdsum_config = {0};
static struct xnn_reduce_config f16_rminmax_config = {0};
static struct xnn_reduce_config f32_rminmax_config = {0};
static struct xnn_reduce_config f32_rsum_config = {0};
static struct xnn_reduce_config f32_rdsum_config = {0};
static struct xnn_reduce_config qs8_rsum_config = {0};
static struct xnn_reduce_config qs8_rdsum_config = {0};
static struct xnn_reduce_config qu8_rsum_config = {0};
static struct xnn_reduce_config qu8_rdsum_config = {0};

XNN_INIT_ONCE_GUARD(f16_f32acc_rsum);
XNN_INIT_ONCE_GUARD(f16_f32acc_rdsum);
XNN_INIT_ONCE_GUARD(f16_rminmax);
XNN_INIT_ONCE_GUARD(f32_rminmax);
XNN_INIT_ONCE_GUARD(f32_rsum);
XNN_INIT_ONCE_GUARD(f32_rdsum);
XNN_INIT_ONCE_GUARD(qs8_rsum);
XNN_INIT_ONCE_GUARD(qs8_rdsum);
XNN_INIT_ONCE_GUARD(qu8_rsum);
XNN_INIT_ONCE_GUARD(qu8_rdsum);

static void init_qs8_rsum_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ARM_DOTPROD
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          qs8_rsum_config = (struct xnn_reduce_config) {
            .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__neondot_u32_acc2,
          };
        } else
      #endif  // XNN_ENABLE_ARM_DOTPROD
      {
        qs8_rsum_config = (struct xnn_reduce_config) {
          .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__neon_u32_acc2,
        };
      }
    } else {
      qs8_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__scalar_u4,
      };
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
      #if XNN_ENABLE_ARM_DOTPROD
        qs8_rsum_config = (struct xnn_reduce_config) {
          .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__neondot_u32_acc2,
        };
      #endif  // XNN_ENABLE_ARM_DOTPROD
    } else {
      qs8_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__neon_u32_acc2,
      };
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    #if XNN_ENABLE_AVX512VNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
        qs8_rsum_config = (struct xnn_reduce_config) {
          .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__avx512vnni_u128_acc2,
        };
      } else
    #endif
    #if XNN_ENABLE_AVXVNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnni) {
        qs8_rsum_config = (struct xnn_reduce_config) {
          .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__avxvnni_u128_acc2,
        };
      } else
    #endif
    #if XNN_ENABLE_AVX512SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
        qs8_rsum_config = (struct xnn_reduce_config) {
          .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__avx512skx_u128_acc2,
        };
      } else
    #endif
    #if XNN_ENABLE_AVX256SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx256skx) {
          qs8_rsum_config = (struct xnn_reduce_config) {
            .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__avx256skx_u64_acc2,
        };
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      qs8_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__avx2_u64_acc2,
      };
    } else if (hardware_config->use_x86_ssse3) {
      qs8_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__ssse3_u32_acc2,
      };
    }
  #elif XNN_ARCH_WASMRELAXEDSIMD
    qs8_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__wasmrelaxedsimd_u64_acc4,
    };
  #elif XNN_ARCH_WASMSIMD
    qs8_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__wasmsimd_u32_acc4,
    };
  #else
    qs8_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_qs8_rsum_ukernel__scalar_u4,
    };
  #endif

  qs8_rsum_config.init.qs8_reduce = xnn_init_qs8_reduce_minmax_scalar_params;
}

static void init_qs8_rdsum_config(void) {
 #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qs8_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qs8_rdsum_ukernel_7p7x__neon_c32,
      };
    } else {
      qs8_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qs8_rdsum_ukernel_7p7x__scalar_c4,
      };
    }
  #elif XNN_ARCH_ARM64
    qs8_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qs8_rdsum_ukernel_7p7x__neon_c32,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512SKX
      if (hardware_config->use_x86_avx512skx) {
        qs8_rdsum_config = (struct xnn_reduce_config) {
          .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qs8_rdsum_ukernel_7p7x__avx512skx_c64,
        };
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      qs8_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qs8_rdsum_ukernel_7p7x__avx2_c64,
      };
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qs8_rdsum_ukernel_7p7x__sse41_c64,
      };
    } else {
      qs8_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qs8_rdsum_ukernel_7p7x__scalar_c4,
      };
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs8_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qs8_rdsum_ukernel_7p7x__wasmsimd_c32,
    };
  #else
    qs8_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qs8_rdsum_ukernel_7p7x__scalar_c4,
    };
  #endif
}

static void init_qu8_rdsum_config(void) {
 #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    if (hardware_config->use_arm_neon) {
      qu8_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qu8_rdsum_ukernel_7p7x__neon_u32,
      };
    } else {
      qu8_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qu8_rdsum_ukernel_7p7x__scalar_c4,
      };
    }
  #elif XNN_ARCH_ARM64
    qu8_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qu8_rdsum_ukernel_7p7x__neon_u16,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_ssse3) {
      qu8_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64,
      };
    } else {
      qu8_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qu8_rdsum_ukernel_7p7x__scalar_c4,
      };
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qu8_rdsum_ukernel_7p7x__wasmsimd_c32,
    };
  #else
    qu8_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_qu8_rdsum_ukernel_7p7x__scalar_c4,
    };
  #endif
}

static void init_qu8_rsum_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    if (hardware_config->use_arm_neon) {
      qu8_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_qu8_rsum_ukernel__neon_u32_acc2,
      };
    } else {
      qu8_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_qu8_rsum_ukernel__scalar_u4,
      };
    }
  #elif XNN_ARCH_ARM64
    qu8_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_qu8_rsum_ukernel__neon_u32_acc2,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    if (hardware_config->use_x86_avx2) {
      qu8_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_qu8_rsum_ukernel__avx2_u64_acc2,
      };
    } else {
      qu8_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_qu8_rsum_ukernel__sse2_u32_acc2,
      };
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_qu8_rsum_ukernel__wasmsimd_u32_acc4,
    };
  #else
    qu8_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_qu8_rsum_ukernel__scalar_u4,
    };
  #endif

  qu8_rsum_config.init.qu8_reduce = xnn_init_qu8_reduce_minmax_scalar_params;
}

static void init_f16_f32acc_rsum_config(void) {
  #if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_f32acc_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u32_acc4,
        .init.f16_f32acc_scale = xnn_init_f16_f32acc_scale_scalar_params,
      };
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64)
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
        f16_f32acc_rsum_config = (struct xnn_reduce_config) {
          .ukernel = (xnn_reduce_ukernel_fn) xnn_f16_f32acc_rsum_ukernel__avx512skx_u64_acc4,
          .init.f16_f32acc_scale = xnn_init_f16_f32acc_scale_scalar_params,
        };
      } else
    #endif
    if (hardware_config->use_x86_f16c) {
      f16_f32acc_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4,
        .init.f16_f32acc_scale = xnn_init_f16_f32acc_scale_scalar_params,
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
    #if XNN_ENABLE_AVX512FP16 || XNN_ENABLE_AVX512SKX
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
    #endif
    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->use_x86_avx512fp16) {
        f16_rminmax_config.ukernel = (xnn_reduce_ukernel_fn) xnn_f16_rminmax_ukernel__avx512fp16_u128_acc4;
      } else
    #endif
    #if XNN_ENABLE_AVX512SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
        f16_rminmax_config.ukernel = (xnn_reduce_ukernel_fn) xnn_f16_rminmax_ukernel__avx512skx_u64_acc4;
      } else
    #endif
    {
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
      };
    } else {
      f32_rminmax_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__scalar_u4_acc4,
      };
    }
  #elif XNN_ARCH_ARM64
    f32_rminmax_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__neon_u16_acc4,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512F
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
        f32_rminmax_config = (struct xnn_reduce_config) {
          .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__avx512f_u64_acc4,
        };
      } else
    #endif
    if (hardware_config->use_x86_avx) {
      f32_rminmax_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__avx_u32_acc4,
      };
    } else {
      f32_rminmax_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__sse_u16_acc4,
      };
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rminmax_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__wasmsimd_minmax_u16_acc4,
    };
  #elif XNN_ARCH_WASM
    f32_rminmax_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__wasm_u4_acc4,
    };
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_rminmax_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__rvv_u8v,
    };
  #else
    f32_rminmax_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rminmax_ukernel__scalar_u4_acc4,
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
      };
    } else {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__scalar_u4_acc4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
      };
    }
  #elif XNN_ARCH_ARM64
    f32_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__neon_u16_acc4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512F
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
        f32_rsum_config = (struct xnn_reduce_config) {
          .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__avx512f_u64_acc4,
          .init.f32_scale = xnn_init_f32_scale_scalar_params,
        };
      } else
    #endif
    if (hardware_config->use_x86_avx) {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__avx_u32_acc4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
      };
    } else {
      f32_rsum_config = (struct xnn_reduce_config) {
        .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__sse_u16_acc4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
      };
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__wasmsimd_u16_acc4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
    };
  #else
    f32_rsum_config = (struct xnn_reduce_config) {
      .ukernel = (xnn_reduce_ukernel_fn) xnn_f32_rsum_ukernel__scalar_u4_acc4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
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
      };
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
        f16_f32acc_rdsum_config = (struct xnn_reduce_config) {
          .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64,
          .init.f16_f32acc_scale = xnn_init_f16_f32acc_scale_scalar_params,
        };
      } else
    #endif
    if (hardware_config->use_x86_f16c) {
      f16_f32acc_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32,
        .init.f16_f32acc_scale = xnn_init_f16_f32acc_scale_scalar_params,
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
      };
    } else {
      f32_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__scalar_c4,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
      };
    }
  #elif XNN_ARCH_ARM64
    f32_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__neon_c16,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512F
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
        f32_rdsum_config = (struct xnn_reduce_config) {
          .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__avx512f_c64,
          .init.f32_scale = xnn_init_f32_scale_scalar_params,
        };
      } else
    #endif
    if (hardware_config->use_x86_avx) {
      f32_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__avx_c32,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
      };
    } else {
      f32_rdsum_config = (struct xnn_reduce_config) {
        .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__sse_c16,
        .init.f32_scale = xnn_init_f32_scale_scalar_params,
      };
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
    };
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__rvv_u4v,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
    };
  #else
    f32_rdsum_config = (struct xnn_reduce_config) {
      .rd_ukernel = (xnn_rdsum_ukernel_fn) xnn_f32_rdsum_ukernel_7p7x__scalar_c4,
      .init.f32_scale = xnn_init_f32_scale_scalar_params,
    };
  #endif
}

const struct xnn_reduce_config* xnn_init_f16_f32acc_rsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_f32acc_rsum);
  return &f16_f32acc_rsum_config;
}

const struct xnn_reduce_config* xnn_init_f16_rminmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_rminmax);
  return &f16_rminmax_config;
}

const struct xnn_reduce_config* xnn_init_f32_rminmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_rminmax);
  return &f32_rminmax_config;
}

const struct xnn_reduce_config* xnn_init_f32_rsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_rsum);
  return &f32_rsum_config;
}

const struct xnn_reduce_config* xnn_init_f16_f32acc_rdsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_f32acc_rdsum);
  return &f16_f32acc_rdsum_config;
}

const struct xnn_reduce_config* xnn_init_f32_rdsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_rdsum);
  return &f32_rdsum_config;
}

const struct xnn_reduce_config* xnn_init_qs8_rsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_rsum);
  return &qs8_rsum_config;
}

const struct xnn_reduce_config* xnn_init_qs8_rdsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_rdsum);
  return &qs8_rdsum_config;
}

const struct xnn_reduce_config* xnn_init_qu8_rsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_rsum);
  return &qu8_rsum_config;
}

const struct xnn_reduce_config* xnn_init_qu8_rdsum_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_rdsum);
  return &qu8_rdsum_config;
}
