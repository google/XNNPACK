// Copyright 2023 Google LLC
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
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/vmulcaddc.h"

static struct xnn_vmulcaddc_config f16_vmulcaddc_config = {0};
static struct xnn_vmulcaddc_config f32_vmulcaddc_config = {0};

XNN_INIT_ONCE_GUARD(f16_vmulcaddc);
XNN_INIT_ONCE_GUARD(f32_vmulcaddc);

// Macros to log the microkernel names if and when they are registered.
#define XNN_INIT_VMULCADDC_UKERNEL(ukernel) \
  (xnn_vmulcaddc_ukernel_fn) ukernel;       \
  xnn_log_info("Using vmulcaddc microkernel '%s'.", #ukernel);

static void init_f16_vmulcaddc_config(void) {
  // LINT.IfChange(f16_vmulcaddc_identifier)
  f16_vmulcaddc_config.identifier = xnn_create_config_identifier(xnn_config_name_f16_vmulcaddc, /*version=*/0);
  // LINT.ThenChange(:f16_vmulcaddc_config)
  // LINT.IfChange(f16_vmulcaddc_config)
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if ((hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith)) {
      f16_vmulcaddc_config.ukernel = XNN_INIT_VMULCADDC_UKERNEL(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x);
      f16_vmulcaddc_config.init.f16 = xnn_init_f16_minmax_scalar_params;
      f16_vmulcaddc_config.channel_tile = 8;
      f16_vmulcaddc_config.row_tile = 2;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if ((hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith)) {
      f16_vmulcaddc_config.ukernel = XNN_INIT_VMULCADDC_UKERNEL(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x);
      f16_vmulcaddc_config.init.f16 = xnn_init_f16_minmax_scalar_params;
      f16_vmulcaddc_config.channel_tile = 8;
      f16_vmulcaddc_config.row_tile = 2;
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if ((hardware_config->arch_flags & xnn_arch_x86_avx2)) {
      f16_vmulcaddc_config.ukernel = XNN_INIT_VMULCADDC_UKERNEL(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x);
      f16_vmulcaddc_config.init.f16 = xnn_init_f16_minmax_scalar_params;
      f16_vmulcaddc_config.channel_tile = 8;
      f16_vmulcaddc_config.row_tile = 2;
    }
  #endif
  // LINT.ThenChange(:f16_vmulcaddc_identifier)
}

static void init_f32_vmulcaddc_config(void) {
  // LINT.IfChange(f32_vmulcaddc_identifier)
  f32_vmulcaddc_config.identifier = xnn_create_config_identifier(xnn_config_name_f32_vmulcaddc, /*version=*/0);
  // LINT.ThenChange(:f32_vmulcaddc_config)
  // LINT.IfChange(f32_vmulcaddc_config)
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if ((hardware_config->arch_flags & xnn_arch_arm_neon)) {
      f32_vmulcaddc_config.ukernel = XNN_INIT_VMULCADDC_UKERNEL(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x);
      f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_vmulcaddc_config.channel_tile = 4;
      f32_vmulcaddc_config.row_tile = 2;
    } else {
      f32_vmulcaddc_config.ukernel = XNN_INIT_VMULCADDC_UKERNEL(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x);
      f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_vmulcaddc_config.channel_tile = 1;
      f32_vmulcaddc_config.row_tile = 2;
    }
  #elif XNN_ARCH_ARM64
    f32_vmulcaddc_config.ukernel = XNN_INIT_VMULCADDC_UKERNEL(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x);
    f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_vmulcaddc_config.channel_tile = 4;
    f32_vmulcaddc_config.row_tile = 2;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_vmulcaddc_config.ukernel = XNN_INIT_VMULCADDC_UKERNEL(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x);
    f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_vmulcaddc_config.channel_tile = 4;
    f32_vmulcaddc_config.row_tile = 2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f32_vmulcaddc_config.ukernel = XNN_INIT_VMULCADDC_UKERNEL(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x);
      f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_vmulcaddc_config.channel_tile = 4;
      f32_vmulcaddc_config.row_tile = 2;
    #else
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->is_x86) {
        f32_vmulcaddc_config.ukernel = XNN_INIT_VMULCADDC_UKERNEL(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x);
        f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_vmulcaddc_config.channel_tile = 4;
        f32_vmulcaddc_config.row_tile = 2;
      } else {
        f32_vmulcaddc_config.ukernel = XNN_INIT_VMULCADDC_UKERNEL(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x);
        f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_vmulcaddc_config.channel_tile = 4;
        f32_vmulcaddc_config.row_tile = 2;
      }
    #endif
  #else
    f32_vmulcaddc_config.ukernel = XNN_INIT_VMULCADDC_UKERNEL(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x);
    f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_vmulcaddc_config.channel_tile = 1;
    f32_vmulcaddc_config.row_tile = 2;
  #endif
  // LINT.ThenChange(:f32_vmulcaddc_identifier)
}

const struct xnn_vmulcaddc_config* xnn_init_f16_vmulcaddc_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_vmulcaddc);
  return &f16_vmulcaddc_config;
}

const struct xnn_vmulcaddc_config* xnn_init_f32_vmulcaddc_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_vmulcaddc);
  return &f32_vmulcaddc_config;
}
