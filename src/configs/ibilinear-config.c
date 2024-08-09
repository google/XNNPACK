// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"

static struct xnn_ibilinear_config f16_ibilinear_config = {0};
static struct xnn_ibilinear_config f32_ibilinear_config = {0};
static struct xnn_ibilinear_config s8_ibilinear_config = {0};
static struct xnn_ibilinear_config u8_ibilinear_config = {0};

XNN_INIT_ONCE_GUARD(f16_ibilinear);
XNN_INIT_ONCE_GUARD(f32_ibilinear);
XNN_INIT_ONCE_GUARD(s8_ibilinear);
XNN_INIT_ONCE_GUARD(u8_ibilinear);

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
  #else
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
  #else
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
  #else
    u8_ibilinear_config.ukernel = (xnn_ibilinear_ukernel_fn) xnn_u8_ibilinear_ukernel__scalar_c1;
    u8_ibilinear_config.pixel_tile = 1;
    u8_ibilinear_config.channel_tile = 1;
  #endif
}

const struct xnn_ibilinear_config* xnn_init_f16_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_ibilinear);
  return &f16_ibilinear_config;
}

const struct xnn_ibilinear_config* xnn_init_f32_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_ibilinear);
  return &f32_ibilinear_config;
}

const struct xnn_ibilinear_config* xnn_init_s8_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(s8_ibilinear);
  return &s8_ibilinear_config;
}

const struct xnn_ibilinear_config* xnn_init_u8_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(u8_ibilinear);
  return &u8_ibilinear_config;
}
