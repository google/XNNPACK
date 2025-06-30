// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/transpose.h"
#include "src/xnnpack/vunary.h"

static struct xnn_transpose_config transpose_config = {0};

XNN_INIT_ONCE_GUARD(transpose);

// Macros to log the microkernel names if and when they are registered.
#define XNN_INIT_COPY_UKERNEL(ukernel) \
  (xnn_vunary_ukernel_fn) ukernel;     \
  xnn_log_info("Using copy microkernel '%s'.", #ukernel);

#define XNN_INIT_TRANSPOSEC_UKERNEL(ukernel) \
  (xnn_transposec_ukernel_fn) ukernel;       \
  xnn_log_info("Using transposec microkernel '%s'.", #ukernel);

#define XNN_INIT_TRANSPOSEV_UKERNEL(ukernel) \
  (xnn_transposev_ukernel_fn) ukernel;       \
  xnn_log_info("Using transposev microkernel '%s'.", #ukernel);

static void init_transpose_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    if ((hardware_config->arch_flags & xnn_arch_arm_neon)) {
      transpose_config.copy = XNN_INIT_COPY_UKERNEL(xnn_xx_copy_ukernel__scalar_memcpy);
      transpose_config.x8.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x8_transposec_ukernel__16x16_reuse_dec_zip_neon);
      transpose_config.x8.tile_size = 32;
      transpose_config.x16.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
      transpose_config.x16.tile_size = 32;
      transpose_config.x24.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x24_transposec_ukernel__2x2_neon_tbl64);
      transpose_config.x24.tile_size = 32;
      transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__4x4_reuse_dec_zip_neon);
      transpose_config.x32.tile_size = 32;
      transpose_config.x64.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x64_transposec_ukernel__2x2_reuse_dec_zip_neon);
      transpose_config.x64.tile_size = 32;
      transpose_config.xx.variable_size_ukernel = XNN_INIT_TRANSPOSEV_UKERNEL(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
      transpose_config.xx.tile_size = 32;
    } else {
      transpose_config.copy = XNN_INIT_COPY_UKERNEL(xnn_xx_copy_ukernel__scalar_memcpy);
      transpose_config.x8.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x8_transposec_ukernel__2x4_scalar_int);
      transpose_config.x8.tile_size = 32;
      transpose_config.x16.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x16_transposec_ukernel__2x4_scalar_int);
      transpose_config.x16.tile_size = 32;
      transpose_config.x24.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x24_transposec_ukernel__1x2_scalar);
      transpose_config.x24.tile_size = 32;
      transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__2x4_scalar_int);
      transpose_config.x32.tile_size = 32;
      transpose_config.x64.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x64_transposec_ukernel__4x2_scalar_int);
      transpose_config.x64.tile_size = 32;
      transpose_config.xx.variable_size_ukernel = XNN_INIT_TRANSPOSEV_UKERNEL(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
      transpose_config.xx.tile_size = 32;
    }
  #elif XNN_ARCH_ARM64
    transpose_config.copy = XNN_INIT_COPY_UKERNEL(xnn_xx_copy_ukernel__scalar_memcpy);
    transpose_config.x8.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x8_transposec_ukernel__16x16_reuse_dec_zip_neon);
    transpose_config.x8.tile_size = 32;
    transpose_config.x16.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
    transpose_config.x16.tile_size = 32;
    transpose_config.x24.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128);
    transpose_config.x24.tile_size = 32;
    transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__4x4_aarch64_neon_tbl128);
    transpose_config.x32.tile_size = 32;
    transpose_config.x64.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x64_transposec_ukernel__2x2_multi_dec_zip_neon);
    transpose_config.x64.tile_size = 32;
    transpose_config.xx.variable_size_ukernel = XNN_INIT_TRANSPOSEV_UKERNEL(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
    transpose_config.xx.tile_size = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    transpose_config.copy = XNN_INIT_COPY_UKERNEL(xnn_xx_copy_ukernel__scalar_memcpy);
    transpose_config.xx.variable_size_ukernel = XNN_INIT_TRANSPOSEV_UKERNEL(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
    transpose_config.xx.tile_size = 32;
    transpose_config.x8.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x8_transposec_ukernel__16x16_reuse_mov_sse2);
    transpose_config.x8.tile_size = 32;
    transpose_config.x16.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
    transpose_config.x16.tile_size = 32;
    transpose_config.x24.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x24_transposec_ukernel__1x2_scalar);
    transpose_config.x24.tile_size = 32;
    transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__4x4_sse);
    transpose_config.x32.tile_size = 32;
    transpose_config.x64.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x64_transposec_ukernel__2x2_multi_mov_sse2);
    transpose_config.x64.tile_size = 32;
    if ((hardware_config->arch_flags & xnn_arch_x86_ssse3)) {
      transpose_config.x24.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x24_transposec_ukernel__4x4_ssse3);
      transpose_config.x24.tile_size = 32;
    }
    if ((hardware_config->arch_flags & xnn_arch_x86_avx)) {
      transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__8x8_reuse_multi_avx);
      transpose_config.x32.tile_size = 32;
      transpose_config.x64.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x64_transposec_ukernel__4x4_reuse_multi_avx);
      transpose_config.x64.tile_size = 32;
    }
    if ((hardware_config->arch_flags & xnn_arch_x86_avx2)) {
      transpose_config.x8.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x8_transposec_ukernel__32x32_reuse_switch_avx2);
      transpose_config.x8.tile_size = 32;
      transpose_config.x16.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
      transpose_config.x16.tile_size = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    transpose_config.copy = XNN_INIT_COPY_UKERNEL(xnn_xx_copy_ukernel__scalar_memcpy);
    transpose_config.x8.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x8_transposec_ukernel__16x16_reuse_mov_wasmsimd);
    transpose_config.x8.tile_size = 32;
    transpose_config.x16.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
    transpose_config.x16.tile_size = 32;
    transpose_config.x24.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x24_transposec_ukernel__1x2_scalar);
    transpose_config.x24.tile_size = 32;
    transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__4x4_reuse_mov_wasmsimd);
    transpose_config.x32.tile_size = 32;
    transpose_config.x64.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x64_transposec_ukernel__4x2_scalar_int);
    transpose_config.x64.tile_size = 32;
    transpose_config.xx.variable_size_ukernel = XNN_INIT_TRANSPOSEV_UKERNEL(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
    transpose_config.xx.tile_size = 32;
  #else
    transpose_config.copy = XNN_INIT_COPY_UKERNEL(xnn_xx_copy_ukernel__scalar_memcpy);
    transpose_config.x8.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x8_transposec_ukernel__2x4_scalar_int);
    transpose_config.x8.tile_size = 32;
    transpose_config.x16.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x16_transposec_ukernel__2x4_scalar_int);
    transpose_config.x16.tile_size = 32;
    transpose_config.x24.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x24_transposec_ukernel__1x2_scalar);
    transpose_config.x24.tile_size = 32;
    #if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->vlenb >= 128) {
        transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__32x8_rvv);
        transpose_config.x32.tile_size = 32;
      } else if (hardware_config->vlenb == 64) {
        transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__16x8_rvv);
        transpose_config.x32.tile_size = 32;
      } else if (hardware_config->vlenb == 32) {
        transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__8x8_rvv);
        transpose_config.x32.tile_size = 32;
      } else if (hardware_config->vlenb == 16) {
        transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__4x4_rvv);
        transpose_config.x32.tile_size = 32;
      } else {
        transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__2x4_scalar_int);
        transpose_config.x32.tile_size = 32;
      }
    #elif XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
      transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__32x32_multi_multi_hvx);
      transpose_config.x32.tile_size = 32;
    #else
      transpose_config.x32.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x32_transposec_ukernel__2x4_scalar_int);
      transpose_config.x32.tile_size = 32;
    #endif
    transpose_config.x64.const_size_ukernel = XNN_INIT_TRANSPOSEC_UKERNEL(xnn_x64_transposec_ukernel__4x2_scalar_int);
    transpose_config.x64.tile_size = 32;
    transpose_config.xx.variable_size_ukernel = XNN_INIT_TRANSPOSEV_UKERNEL(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
    transpose_config.xx.tile_size = 32;
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
