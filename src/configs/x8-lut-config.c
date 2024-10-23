// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/init-once.h"
#include "xnnpack/lut.h"


static struct xnn_x8_lut_config x8_lut_config = {0};

XNN_INIT_ONCE_GUARD(x8_lut);

static void init_x8_lut_config(void) {
  #if XNN_ARCH_ARM
    x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u4;
  #elif XNN_ARCH_ARM64
    x8_lut_config.microkernel = xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u64;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    #if XNN_ENABLE_AVX256VBMI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vbmi) {
        x8_lut_config.microkernel = xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u128;
      } else
    #endif
    #if XNN_ENABLE_AVX512SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
        x8_lut_config.microkernel = xnn_x8_lut_ukernel__avx512skx_vpshufb_u64;
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      x8_lut_config.microkernel = xnn_x8_lut_ukernel__avx2_u128;
    } else if (hardware_config->use_x86_avx) {
      x8_lut_config.microkernel = xnn_x8_lut_ukernel__avx_u64;
    } else {
      // Note: SSSE3 version is usually slower than scalar
      x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u4;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);

    if (hardware_config->is_x86) {
      #if XNN_ARCH_WASMRELAXEDSIMD
        if (hardware_config->use_wasm_pshufb) {
          x8_lut_config.microkernel = xnn_x8_lut_ukernel__wasmpshufb_u32;
        } else {
          x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u4;
        }
      #else
        x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u4;
      #endif
    } else {
      x8_lut_config.microkernel = xnn_x8_lut_ukernel__wasmsimd_u32;
    }
  #else
    x8_lut_config.microkernel = xnn_x8_lut_ukernel__scalar_u4;
  #endif
}

const struct xnn_x8_lut_config* xnn_init_x8_lut_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x8_lut);
  return &x8_lut_config;
}
