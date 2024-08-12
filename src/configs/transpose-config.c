// Copyright 2022 Google LLC
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
#include "xnnpack/transpose.h"
#include "xnnpack/vunary.h"

static struct xnn_transpose_config transpose_config = {0};

XNN_INIT_ONCE_GUARD(transpose);

static void init_transpose_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    if (hardware_config->use_arm_neon) {
      transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
      transpose_config.x8 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__16x16_reuse_dec_zip_neon,
        .tile_size = 32,
      };
      transpose_config.x16 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon,
        .tile_size = 32,
      };
      transpose_config.x24 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x24_transposec_ukernel__2x2_neon_tbl64,
        .tile_size = 32,
      };
      transpose_config.x32 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__4x4_reuse_dec_zip_neon,
        .tile_size = 32,
      };
      transpose_config.x64 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x64_transposec_ukernel__2x2_reuse_dec_zip_neon,
        .tile_size = 32,
      };
      transpose_config.xx = (struct xnn_transpose_subconfig) {
        .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_scalar_memcpy,
        .tile_size = 32,
      };
    } else if (!XNN_PLATFORM_MOBILE) {
      transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
      transpose_config.x8 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__2x4_scalar_int,
        .tile_size = 32,
      };
      transpose_config.x16 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__2x4_scalar_int,
        .tile_size = 32,
      };
      transpose_config.x24 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x24_transposec_ukernel__1x2_scalar,
        .tile_size = 32,
      };
      transpose_config.x32 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__2x4_scalar_int,
        .tile_size = 32,
      };
      transpose_config.x64 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x64_transposec_ukernel__4x2_scalar_int,
        .tile_size = 32,
      };
      transpose_config.xx = (struct xnn_transpose_subconfig) {
        .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_scalar_memcpy,
        .tile_size = 32,
      };
    }
  #elif XNN_ARCH_ARM64
    transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    transpose_config.x8 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__16x16_reuse_dec_zip_neon,
      .tile_size = 32,
    };
    transpose_config.x16 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon,
      .tile_size = 32,
    };
    transpose_config.x24 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128,
      .tile_size = 32,
    };
    transpose_config.x32 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__4x4_aarch64_neon_tbl128,
      .tile_size = 32,
    };
    transpose_config.x64 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x64_transposec_ukernel__2x2_multi_dec_zip_neon,
      .tile_size = 32,
    };
    transpose_config.xx = (struct xnn_transpose_subconfig) {
      .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_scalar_memcpy,
      .tile_size = 32,
    };
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    transpose_config.xx = (struct xnn_transpose_subconfig) {
      .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_scalar_memcpy,
      .tile_size = 32,
    };
    transpose_config.x8 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__16x16_reuse_mov_sse2,
      .tile_size = 32,
    };
    transpose_config.x16 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2,
      .tile_size = 32,
    };
    transpose_config.x24 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x24_transposec_ukernel__1x2_scalar,
      .tile_size = 32,
    };
    transpose_config.x32 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__4x4_sse,
      .tile_size = 32,
    };
    transpose_config.x64 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x64_transposec_ukernel__2x2_multi_mov_sse2,
      .tile_size = 32,
    };
    if (hardware_config->use_x86_ssse3) {
      transpose_config.x24 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x24_transposec_ukernel__4x4_ssse3,
        .tile_size = 32,
      };
    }
    if (hardware_config->use_x86_avx) {
      transpose_config.x32 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__8x8_reuse_multi_avx,
        .tile_size = 32,
      };
      transpose_config.x64 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x64_transposec_ukernel__4x4_reuse_multi_avx,
        .tile_size = 32,
      };
    }
    if (hardware_config->use_x86_avx2) {
      transpose_config.x8 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__32x32_reuse_switch_avx2,
        .tile_size = 32,
      };
      transpose_config.x16 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2,
        .tile_size = 32,
      };
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    transpose_config.x8 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__16x16_reuse_mov_wasmsimd,
      .tile_size = 32,
    };
    transpose_config.x16 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd,
      .tile_size = 32,
    };
    transpose_config.x24 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x24_transposec_ukernel__1x2_scalar,
      .tile_size = 32,
    };
    transpose_config.x32 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__4x4_reuse_mov_wasmsimd,
      .tile_size = 32,
    };
    transpose_config.x64 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x64_transposec_ukernel__4x2_scalar_int,
      .tile_size = 32,
    };
    transpose_config.xx = (struct xnn_transpose_subconfig) {
      .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_scalar_memcpy,
      .tile_size = 32,
    };
  #else
    transpose_config.copy = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    transpose_config.x8 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x8_transposec_ukernel__2x4_scalar_int,
      .tile_size = 32,
    };
    transpose_config.x16 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x16_transposec_ukernel__2x4_scalar_int,
      .tile_size = 32,
    };
    transpose_config.x24 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x24_transposec_ukernel__1x2_scalar,
      .tile_size = 32,
    };
    #if XNN_ENABLE_RISCV_VECTOR
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->vlenb >= 128) {
        transpose_config.x32 = (struct xnn_transpose_subconfig) {
          .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__32x8_rvv,
          .tile_size = 32,
        };
      } else if (hardware_config->vlenb == 64) {
        transpose_config.x32 = (struct xnn_transpose_subconfig) {
          .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__16x8_rvv,
          .tile_size = 32,
        };
      } else if (hardware_config->vlenb == 32) {
        transpose_config.x32 = (struct xnn_transpose_subconfig) {
          .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__8x8_rvv,
          .tile_size = 32,
        };
      } else if (hardware_config->vlenb == 16) {
        transpose_config.x32 = (struct xnn_transpose_subconfig) {
          .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__4x4_rvv,
          .tile_size = 32,
        };
      } else {
        transpose_config.x32 = (struct xnn_transpose_subconfig) {
          .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__2x4_scalar_int,
          .tile_size = 32,
        };
      }
    #else
      transpose_config.x32 = (struct xnn_transpose_subconfig) {
        .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x32_transposec_ukernel__2x4_scalar_int,
        .tile_size = 32,
      };
    #endif
    transpose_config.x64 = (struct xnn_transpose_subconfig) {
      .const_size_ukernel = (xnn_transposec_ukernel_fn) xnn_x64_transposec_ukernel__4x2_scalar_int,
      .tile_size = 32,
    };
    transpose_config.xx = (struct xnn_transpose_subconfig) {
      .variable_size_ukernel = xnn_xx_transposev_ukernel__1x1_scalar_memcpy,
      .tile_size = 32,
    };
  #endif
}

const struct xnn_transpose_config* xnn_init_transpose_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(transpose);
  return &transpose_config;
}
