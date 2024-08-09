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
#include "xnnpack/zip.h"

static struct xnn_zip_config x8_zip_config = {0};
static struct xnn_zip_config x32_zip_config = {0};

XNN_INIT_ONCE_GUARD(x8_zip);
XNN_INIT_ONCE_GUARD(x32_zip);

static void init_x8_zip_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      x8_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x2_ukernel__neon;
      x8_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x3_ukernel__neon;
      x8_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x4_ukernel__neon;
      x8_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x8_zip_xm_ukernel__neon;
    } else if (!XNN_PLATFORM_MOBILE) {
      x8_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x2_ukernel__scalar;
      x8_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x3_ukernel__scalar;
      x8_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x4_ukernel__scalar;
      x8_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x8_zip_xm_ukernel__scalar;
    }
  #elif XNN_ARCH_ARM64
    x8_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x2_ukernel__neon;
    x8_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x3_ukernel__neon;
    x8_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x4_ukernel__neon;
    x8_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x8_zip_xm_ukernel__neon;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    x8_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x2_ukernel__sse2;
    x8_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x3_ukernel__sse2;
    x8_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x4_ukernel__sse2;
    x8_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x8_zip_xm_ukernel__sse2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    x8_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x2_ukernel__scalar;
    x8_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x3_ukernel__scalar;
    x8_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x4_ukernel__scalar;
    x8_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x8_zip_xm_ukernel__scalar;
  #else
    x8_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x2_ukernel__scalar;
    x8_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x3_ukernel__scalar;
    x8_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x8_zip_x4_ukernel__scalar;
    x8_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x8_zip_xm_ukernel__scalar;
  #endif

}

static void init_x32_zip_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      x32_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x2_ukernel__neon;
      x32_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x3_ukernel__neon;
      x32_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x4_ukernel__neon;
      x32_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x32_zip_xm_ukernel__neon;
    } else if (!XNN_PLATFORM_MOBILE) {
      x32_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x2_ukernel__scalar;
      x32_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x3_ukernel__scalar;
      x32_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x4_ukernel__scalar;
      x32_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x32_zip_xm_ukernel__scalar;
    }
  #elif XNN_ARCH_ARM64
    x32_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x2_ukernel__neon;
    x32_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x3_ukernel__neon;
    x32_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x4_ukernel__neon;
    x32_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x32_zip_xm_ukernel__neon;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    x32_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x2_ukernel__sse2;
    x32_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x3_ukernel__sse2;
    x32_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x4_ukernel__sse2;
    x32_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x32_zip_xm_ukernel__sse2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    x32_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x2_ukernel__wasmsimd;
    x32_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x3_ukernel__wasmsimd;
    x32_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x4_ukernel__wasmsimd;
    x32_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x32_zip_xm_ukernel__wasmsimd;
  #else
    x32_zip_config.x2 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x2_ukernel__scalar;
    x32_zip_config.x3 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x3_ukernel__scalar;
    x32_zip_config.x4 = (xnn_zipc_ukernel_fn) xnn_x32_zip_x4_ukernel__scalar;
    x32_zip_config.xm = (xnn_zipv_ukernel_fn) xnn_x32_zip_xm_ukernel__scalar;
  #endif

}

const struct xnn_zip_config* xnn_init_x8_zip_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x8_zip);
  return &x8_zip_config;
}

const struct xnn_zip_config* xnn_init_x32_zip_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(x32_zip);
  return &x32_zip_config;
}
