// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#if XNN_ENABLE_CPUINFO
  #include <cpuinfo.h>
#endif  // XNN_ENABLE_CPUINFO

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"

static struct xnn_dwconv_config f16_dwconv_config[XNN_MAX_F16_DWCONV_UKERNELS] = {0};
static struct xnn_dwconv_config f32_dwconv_config[XNN_MAX_F32_DWCONV_UKERNELS] = {0};
static struct xnn_dwconv_config qs8_qc8w_dwconv_config[XNN_MAX_QC8_DWCONV_UKERNELS] = {0};
static struct xnn_dwconv_config qs8_dwconv_config[XNN_MAX_QS8_DWCONV_UKERNELS] = {0};
static struct xnn_dwconv_config qu8_dwconv_config[XNN_MAX_QU8_DWCONV_UKERNELS] = {0};

XNN_INIT_ONCE_GUARD(f16_dwconv);
XNN_INIT_ONCE_GUARD(f32_dwconv);
XNN_INIT_ONCE_GUARD(qs8_qc8w_dwconv);
XNN_INIT_ONCE_GUARD(qs8_dwconv);
XNN_INIT_ONCE_GUARD(qu8_dwconv);

static void init_f16_dwconv_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_3p16c__neonfp16arith;
      f16_dwconv_config[0].init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_dwconv_config[0].channel_tile = 16;
      f16_dwconv_config[0].channel_subtile = 16;
      f16_dwconv_config[0].channel_round = 1;
      f16_dwconv_config[0].primary_tile = 3;

      f16_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_4p16c__neonfp16arith;
      f16_dwconv_config[1].init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_dwconv_config[1].channel_tile = 16;
      f16_dwconv_config[1].channel_subtile = 16;
      f16_dwconv_config[1].channel_round = 1;
      f16_dwconv_config[1].primary_tile = 4;

      f16_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_9p8c__neonfp16arith;
      f16_dwconv_config[2].init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_dwconv_config[2].channel_tile = 8;
      f16_dwconv_config[2].channel_subtile = 8;
      f16_dwconv_config[2].channel_round = 1;
      f16_dwconv_config[2].primary_tile = 9;

      f16_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_25p8c__neonfp16arith_acc2;
      f16_dwconv_config[3].init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_dwconv_config[3].channel_tile = 8;
      f16_dwconv_config[3].channel_subtile = 8;
      f16_dwconv_config[3].channel_round = 1;
      f16_dwconv_config[3].primary_tile = 25;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_3p16c__neonfp16arith;
      f16_dwconv_config[0].init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_dwconv_config[0].channel_tile = 16;
      f16_dwconv_config[0].channel_subtile = 16;
      f16_dwconv_config[0].channel_round = 1;
      f16_dwconv_config[0].primary_tile = 3;

      f16_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_4p16c__neonfp16arith;
      f16_dwconv_config[1].init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_dwconv_config[1].channel_tile = 16;
      f16_dwconv_config[1].channel_subtile = 16;
      f16_dwconv_config[1].channel_round = 1;
      f16_dwconv_config[1].primary_tile = 4;

      f16_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_9p16c__neonfp16arith;
      f16_dwconv_config[2].init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_dwconv_config[2].channel_tile = 16;
      f16_dwconv_config[2].channel_subtile = 16;
      f16_dwconv_config[2].channel_round = 1;
      f16_dwconv_config[2].primary_tile = 9;

      f16_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_25p8c__neonfp16arith_acc2;
      f16_dwconv_config[3].init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_dwconv_config[3].channel_tile = 8;
      f16_dwconv_config[3].channel_subtile = 8;
      f16_dwconv_config[3].channel_round = 1;
      f16_dwconv_config[3].primary_tile = 25;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_3p16c__fma3;
      f16_dwconv_config[0].init.f16 = xnn_init_f16_minmax_avx_params;
      f16_dwconv_config[0].channel_tile = 16;
      f16_dwconv_config[0].channel_subtile = 16;
      f16_dwconv_config[0].channel_round = 1;
      f16_dwconv_config[0].primary_tile = 3;

      f16_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_4p16c__fma3;
      f16_dwconv_config[1].init.f16 = xnn_init_f16_minmax_avx_params;
      f16_dwconv_config[1].channel_tile = 16;
      f16_dwconv_config[1].channel_subtile = 16;
      f16_dwconv_config[1].channel_round = 1;
      f16_dwconv_config[1].primary_tile = 4;

      f16_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_9p16c__fma3;
      f16_dwconv_config[2].init.f16 = xnn_init_f16_minmax_avx_params;
      f16_dwconv_config[2].channel_tile = 16;
      f16_dwconv_config[2].channel_subtile = 16;
      f16_dwconv_config[2].channel_round = 1;
      f16_dwconv_config[2].primary_tile = 9;

      f16_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f16_dwconv_minmax_ukernel_25p8c__fma3_acc2;
      f16_dwconv_config[3].init.f16 = xnn_init_f16_minmax_avx_params;
      f16_dwconv_config[3].channel_tile = 8;
      f16_dwconv_config[3].channel_subtile = 8;
      f16_dwconv_config[3].channel_round = 1;
      f16_dwconv_config[3].primary_tile = 25;
    }
  #endif
}

static void init_f32_dwconv_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p8c__neon;
      f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[0].channel_tile = 8,
      f32_dwconv_config[0].channel_subtile = 8,
      f32_dwconv_config[0].channel_round = 1,
      f32_dwconv_config[0].primary_tile = 3,

      f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p8c__neon;
      f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[1].channel_tile = 8,
      f32_dwconv_config[1].channel_subtile = 8,
      f32_dwconv_config[1].channel_round = 1,
      f32_dwconv_config[1].primary_tile = 4,

      f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p8c__neon;
      f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[2].channel_tile = 8;
      f32_dwconv_config[2].channel_subtile = 8;
      f32_dwconv_config[2].channel_round = 1;
      f32_dwconv_config[2].primary_tile = 9;

      #if XNN_ENABLE_DWCONV_MULTIPASS
        f32_dwconv_config[3].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_dwconv_config[3].channel_tile = 4;
        f32_dwconv_config[3].channel_subtile = 4;
        f32_dwconv_config[3].channel_round = 4;
        f32_dwconv_config[3].primary_tile = 8;
        f32_dwconv_config[3].middle_tile = 8;
        f32_dwconv_config[3].last_tile = 9;
      #else
        f32_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_25p8c__neon_acc2;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_dwconv_config[3].channel_tile = 8;
        f32_dwconv_config[3].channel_subtile = 8;
        f32_dwconv_config[3].channel_round = 1;
        f32_dwconv_config[3].primary_tile = 25;
      #endif  // XNN_ENABLE_DWCONV_MULTIPASS
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p1c__scalar_acc2;
      f32_dwconv_config[0].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_3p1c__scalar_acc2;
      f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[0].channel_tile = 1;
      f32_dwconv_config[0].channel_subtile = 1;
      f32_dwconv_config[0].channel_round = 1;
      f32_dwconv_config[0].primary_tile = 3;

      f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p1c__scalar_acc2;
      f32_dwconv_config[1].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_4p1c__scalar_acc2;
      f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[1].channel_tile = 1;
      f32_dwconv_config[1].channel_subtile = 1;
      f32_dwconv_config[1].channel_round = 1;
      f32_dwconv_config[1].primary_tile = 4;

      f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p1c__scalar_acc2;
      f32_dwconv_config[2].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_9p1c__scalar_acc2;
      f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[2].channel_tile = 1;
      f32_dwconv_config[2].channel_subtile = 1;
      f32_dwconv_config[2].channel_round = 1;
      f32_dwconv_config[2].primary_tile = 9;

      #if XNN_ENABLE_DWCONV_MULTIPASS
        f32_dwconv_config[3].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_dwconv_config[3].channel_tile = 4;
        f32_dwconv_config[3].channel_subtile = 1;
        f32_dwconv_config[3].channel_round = 1;
        f32_dwconv_config[3].primary_tile = 2;
        f32_dwconv_config[3].middle_tile = 2;
        f32_dwconv_config[3].last_tile = 2;
      #else
        f32_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_25p1c__scalar_acc2;
        f32_dwconv_config[3].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_25p1c__scalar_acc2;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_dwconv_config[3].channel_tile = 1;
        f32_dwconv_config[3].primary_tile = 25;
      #endif  // XNN_ENABLE_DWCONV_MULTIPASS
    }
  #elif XNN_ARCH_ARM64
    f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p8c__neonfma;
    f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_dwconv_config[0].channel_tile = 8;
    f32_dwconv_config[0].channel_subtile = 8;
    f32_dwconv_config[0].channel_round = 1;
    f32_dwconv_config[0].primary_tile = 3;

    f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p8c__neonfma;
    f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_dwconv_config[1].channel_tile = 8;
    f32_dwconv_config[1].channel_subtile = 8;
    f32_dwconv_config[1].channel_round = 1;
    f32_dwconv_config[1].primary_tile = 4;

    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
      f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p8c__neonfma;
      f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[2].channel_tile = 8;
      f32_dwconv_config[2].channel_subtile = 8;
      f32_dwconv_config[2].channel_round = 1;
      f32_dwconv_config[2].primary_tile = 9;
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_kryo:
          f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p8c__neonfma;
          f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_dwconv_config[2].channel_tile = 8;
          f32_dwconv_config[2].channel_subtile = 8;
          f32_dwconv_config[2].channel_round = 1;
          f32_dwconv_config[2].primary_tile = 9;
          break;
        #if XNN_ENABLE_ASSEMBLY
          case cpuinfo_uarch_cortex_a53:
          case cpuinfo_uarch_cortex_a55r0:
          case cpuinfo_uarch_cortex_a55:
            f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p4c__asm_aarch64_neonfma_cortex_a55;
            f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_dwconv_config[2].channel_tile = 4;
            f32_dwconv_config[2].channel_subtile = 4;
            f32_dwconv_config[2].channel_round = 1;
            f32_dwconv_config[2].primary_tile = 9;
            break;
        #endif  // XNN_ENABLE_ASSEMBLY
        default:
          f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p8c__neonfma;
          f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_dwconv_config[2].channel_tile = 8;
          f32_dwconv_config[2].channel_subtile = 8;
          f32_dwconv_config[2].channel_round = 1;
          f32_dwconv_config[2].primary_tile = 9;
          break;
      }
    #endif  // XNN_PLATFORM_IOS && XNN_PLATFORM_MAC

    #if XNN_ENABLE_DWCONV_MULTIPASS
      f32_dwconv_config[3].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2;
      f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[3].channel_tile = 8;
      f32_dwconv_config[3].channel_subtile = 4;
      f32_dwconv_config[3].channel_round = 4;
      f32_dwconv_config[3].primary_tile = 5;
      f32_dwconv_config[3].middle_tile = 5;
      f32_dwconv_config[3].last_tile = 5;
    #else
      f32_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_25p8c__neonfma_acc2;
      f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[3].channel_tile = 8;
      f32_dwconv_config[3].channel_subtile = 8;
      f32_dwconv_config[3].channel_round = 1;
      f32_dwconv_config[3].primary_tile = 25;
    #endif  // XNN_ENABLE_DWCONV_MULTIPASS
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p16c__avx512f;
      f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[0].channel_tile = 16;
      f32_dwconv_config[0].channel_subtile = 16;
      f32_dwconv_config[0].channel_round = 1;
      f32_dwconv_config[0].primary_tile = 3;

      f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p16c__avx512f;
      f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[1].channel_tile = 16;
      f32_dwconv_config[1].channel_subtile = 16;
      f32_dwconv_config[1].channel_round = 1;
      f32_dwconv_config[1].primary_tile = 4;

      f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p16c__avx512f;
      f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[2].channel_tile = 16;
      f32_dwconv_config[2].channel_subtile = 16;
      f32_dwconv_config[2].channel_round = 1;
      f32_dwconv_config[2].primary_tile = 9;

      // Multipass microkernel "acc" value should match unipass and also match across different hardware config.
      // Accumulation (FMA) can produce different results, which results in tests only failing on certain platforms.
      #if XNN_ENABLE_DWCONV_MULTIPASS
        f32_dwconv_config[3].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_dwconv_config[3].channel_tile = 32;
        f32_dwconv_config[3].channel_subtile = 16;
        f32_dwconv_config[3].channel_round = 1;
        f32_dwconv_config[3].primary_tile = 5;
        f32_dwconv_config[3].middle_tile = 5;
        f32_dwconv_config[3].last_tile = 5;
      #else
        f32_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_25p16c__avx512f;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_dwconv_config[3].channel_tile = 16;
        f32_dwconv_config[3].channel_subtile = 16;
        f32_dwconv_config[3].channel_round = 1;
        f32_dwconv_config[3].primary_tile = 25;
      #endif  // XNN_ENABLE_DWCONV_MULTIPASS
    } else if (hardware_config->use_x86_fma3) {
      f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p16c__fma3;
      f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_avx_params;
      f32_dwconv_config[0].channel_tile = 16;
      f32_dwconv_config[0].channel_subtile = 16;
      f32_dwconv_config[0].channel_round = 1;
      f32_dwconv_config[0].primary_tile = 3;

      f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p16c__fma3;
      f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_avx_params;
      f32_dwconv_config[1].channel_tile = 16;
      f32_dwconv_config[1].channel_subtile = 16;
      f32_dwconv_config[1].channel_round = 1;
      f32_dwconv_config[1].primary_tile = 4;

      f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p16c__fma3;
      f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_avx_params;
      f32_dwconv_config[2].channel_tile = 16;
      f32_dwconv_config[2].channel_subtile = 16;
      f32_dwconv_config[2].channel_round = 1;
      f32_dwconv_config[2].primary_tile = 9;

      #if XNN_ENABLE_DWCONV_MULTIPASS
        f32_dwconv_config[3].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_avx_params;
        f32_dwconv_config[3].channel_tile = 8;
        f32_dwconv_config[3].channel_subtile = 8;
        f32_dwconv_config[3].channel_round = 4;
        f32_dwconv_config[3].primary_tile = 5;
        f32_dwconv_config[3].middle_tile = 5;
        f32_dwconv_config[3].last_tile = 5;
      #else
        f32_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_25p8c__fma3;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_avx_params;
        f32_dwconv_config[3].channel_tile = 8;
        f32_dwconv_config[3].channel_subtile = 8;
        f32_dwconv_config[3].channel_round = 1;
        f32_dwconv_config[3].primary_tile = 25;
      #endif  // XNN_ENABLE_DWCONV_MULTIPASS
    } else if (hardware_config->use_x86_avx) {
      f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p16c__avx;
      f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_avx_params;
      f32_dwconv_config[0].channel_tile = 16;
      f32_dwconv_config[0].channel_subtile = 16;
      f32_dwconv_config[0].channel_round = 1;
      f32_dwconv_config[0].primary_tile = 3;

      f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p16c__avx;
      f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_avx_params;
      f32_dwconv_config[1].channel_tile = 16;
      f32_dwconv_config[1].channel_subtile = 16;
      f32_dwconv_config[1].channel_round = 1;
      f32_dwconv_config[1].primary_tile = 4;

      f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p16c__avx;
      f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_avx_params;
      f32_dwconv_config[2].channel_tile = 16;
      f32_dwconv_config[2].channel_subtile = 16;
      f32_dwconv_config[2].channel_round = 1;
      f32_dwconv_config[2].primary_tile = 9;

      #if XNN_ENABLE_DWCONV_MULTIPASS
        f32_dwconv_config[3].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_avx_params;
        f32_dwconv_config[3].channel_tile = 8;
        f32_dwconv_config[3].channel_subtile = 8;
        f32_dwconv_config[3].channel_round = 4;
        f32_dwconv_config[3].primary_tile = 6;
        f32_dwconv_config[3].middle_tile = 6;
        f32_dwconv_config[3].last_tile = 7;
      #else
        f32_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_25p8c__avx;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_avx_params;
        f32_dwconv_config[3].channel_tile = 8;
        f32_dwconv_config[3].channel_subtile = 8;
        f32_dwconv_config[3].channel_round = 1;
        f32_dwconv_config[3].primary_tile = 25;
      #endif  // XNN_ENABLE_DWCONV_MULTIPASS
    } else {
      f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p8c__sse;
      f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_sse_params;
      f32_dwconv_config[0].channel_tile = 8;
      f32_dwconv_config[0].channel_subtile = 8;
      f32_dwconv_config[0].channel_round = 1;
      f32_dwconv_config[0].primary_tile = 3;

      f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p8c__sse;
      f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_sse_params;
      f32_dwconv_config[1].channel_tile = 8;
      f32_dwconv_config[1].channel_subtile = 8;
      f32_dwconv_config[1].channel_round = 1;
      f32_dwconv_config[1].primary_tile = 4;

      f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p8c__sse;
      f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_sse_params;
      f32_dwconv_config[2].channel_tile = 8;
      f32_dwconv_config[2].channel_subtile = 8;
      f32_dwconv_config[2].channel_round = 1;
      f32_dwconv_config[2].primary_tile = 9;

      #if XNN_ENABLE_DWCONV_MULTIPASS
        f32_dwconv_config[3].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_sse_params;
        f32_dwconv_config[3].channel_tile = 16;
        f32_dwconv_config[3].channel_subtile = 4;
        f32_dwconv_config[3].channel_round = 4;
        f32_dwconv_config[3].primary_tile = 8;
        f32_dwconv_config[3].middle_tile = 8;
        f32_dwconv_config[3].last_tile = 9;
      #else
        f32_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_25p8c__sse;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_sse_params;
        f32_dwconv_config[3].channel_tile = 8;
        f32_dwconv_config[3].channel_subtile = 8;
        f32_dwconv_config[3].channel_round = 1;
        f32_dwconv_config[3].primary_tile = 25;
      #endif  // XNN_ENABLE_DWCONV_MULTIPASS
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p8c__wasmrelaxedsimd_fma;
      f32_dwconv_config[0].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma;
      f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_dwconv_config[0].channel_tile = 8;
      f32_dwconv_config[0].channel_subtile = 8;
      f32_dwconv_config[0].channel_round = 1;
      f32_dwconv_config[0].primary_tile = 3;

      f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p8c__wasmrelaxedsimd_fma;
      f32_dwconv_config[1].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma;
      f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_dwconv_config[1].channel_tile = 8;
      f32_dwconv_config[1].channel_subtile = 8;
      f32_dwconv_config[1].channel_round = 1;
      f32_dwconv_config[1].primary_tile = 4;

      f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p8c__wasmrelaxedsimd_fma;
      f32_dwconv_config[2].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma;
      f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_dwconv_config[2].channel_tile = 8;
      f32_dwconv_config[2].channel_subtile = 8;
      f32_dwconv_config[2].channel_round = 1;
      f32_dwconv_config[2].primary_tile = 9;
    #else
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->is_x86) {
        f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p8c__wasmsimd_x86;
        f32_dwconv_config[0].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_3p8c__wasmsimd;
        f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_dwconv_config[0].channel_tile = 8;
        f32_dwconv_config[0].channel_subtile = 8;
        f32_dwconv_config[0].channel_round = 1;
        f32_dwconv_config[0].primary_tile = 3;

        f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p8c__wasmsimd_x86;
        f32_dwconv_config[1].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_4p8c__wasmsimd;
        f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_dwconv_config[1].channel_tile = 8;
        f32_dwconv_config[1].channel_subtile = 8;
        f32_dwconv_config[1].channel_round = 1;
        f32_dwconv_config[1].primary_tile = 4;

        #if XNN_ENABLE_DWCONV_MULTIPASS
          f32_dwconv_config[2].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3f3m3l8c4s4r__wasmsimd_x86;
          f32_dwconv_config[2].linear.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_ukernel_3f3m3l8c4s4r__wasmsimd;
          f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
          f32_dwconv_config[2].channel_tile = 8;
          f32_dwconv_config[2].channel_subtile = 4;
          f32_dwconv_config[2].channel_round = 4;
          f32_dwconv_config[2].primary_tile = 3;
          f32_dwconv_config[2].middle_tile = 3;
          f32_dwconv_config[2].last_tile = 3;
        #else
          f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p8c__wasmsimd_x86;
          f32_dwconv_config[2].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_9p8c__wasmsimd;
          f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
          f32_dwconv_config[2].channel_tile = 8;
          f32_dwconv_config[2].channel_subtile = 8;
          f32_dwconv_config[2].channel_round = 1;
          f32_dwconv_config[2].primary_tile = 9;
        #endif  // XNN_ENABLE_DWCONV_MULTIPASS
      } else {
        f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p4c__wasmsimd_arm;
        f32_dwconv_config[0].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_3p4c__wasmsimd;
        f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_dwconv_config[0].channel_tile = 4;
        f32_dwconv_config[0].channel_subtile = 4;
        f32_dwconv_config[0].channel_round = 1;
        f32_dwconv_config[0].primary_tile = 3;

        f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p4c__wasmsimd_arm;
        f32_dwconv_config[1].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_4p4c__wasmsimd;
        f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_dwconv_config[1].channel_tile = 4;
        f32_dwconv_config[1].channel_subtile = 4;
        f32_dwconv_config[1].channel_round = 1;
        f32_dwconv_config[1].primary_tile = 4;

        #if XNN_ENABLE_DWCONV_MULTIPASS
          f32_dwconv_config[2].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3f3m3l4c4s4r__wasmsimd_arm;
          f32_dwconv_config[2].linear.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_ukernel_3f3m3l4c4s4r__wasmsimd;
          f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
          f32_dwconv_config[2].channel_tile = 4;
          f32_dwconv_config[2].channel_subtile = 4;
          f32_dwconv_config[2].channel_round = 4;
          f32_dwconv_config[2].primary_tile = 3;
          f32_dwconv_config[2].middle_tile = 3;
          f32_dwconv_config[2].last_tile = 3;
        #else
          f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p4c__wasmsimd_arm;
          f32_dwconv_config[2].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_9p4c__wasmsimd;
          f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
          f32_dwconv_config[2].channel_tile = 4;
          f32_dwconv_config[2].channel_subtile = 4;
          f32_dwconv_config[2].channel_round = 1;
          f32_dwconv_config[2].primary_tile = 9;
        #endif  // XNN_ENABLE_DWCONV_MULTIPASS
      }
    #endif

    #if XNN_ARCH_WASMRELAXEDSIMD
      #if XNN_ENABLE_DWCONV_MULTIPASS
        f32_dwconv_config[3].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma;
        f32_dwconv_config[3].linear.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_dwconv_config[3].channel_tile = 4;
        f32_dwconv_config[3].channel_subtile = 4;
        f32_dwconv_config[3].channel_round = 4;
        f32_dwconv_config[3].primary_tile = 5;
        f32_dwconv_config[3].middle_tile = 5;
        f32_dwconv_config[3].last_tile = 5;
      #else
        f32_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd_fma;
        f32_dwconv_config[3].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_dwconv_config[3].channel_tile = 8;
        f32_dwconv_config[3].channel_subtile = 8;
        f32_dwconv_config[3].channel_round = 1;
        f32_dwconv_config[3].primary_tile = 25;
      #endif  // XNN_ENABLE_DWCONV_MULTIPASS
    #else
      #if XNN_ENABLE_DWCONV_MULTIPASS
        f32_dwconv_config[3].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm;
        f32_dwconv_config[3].linear.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_dwconv_config[3].channel_tile = 4;
        f32_dwconv_config[3].channel_subtile = 4;
        f32_dwconv_config[3].channel_round = 4;
        f32_dwconv_config[3].primary_tile = 5;
        f32_dwconv_config[3].middle_tile = 5;
        f32_dwconv_config[3].last_tile = 5;
      #else
        f32_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_25p4c__wasmsimd_arm;
        f32_dwconv_config[3].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_25p4c__wasmsimd;
        f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_dwconv_config[3].channel_tile = 4;
        f32_dwconv_config[3].channel_subtile = 4;
        f32_dwconv_config[3].channel_round = 1;
        f32_dwconv_config[3].primary_tile = 25;
      #endif  // XNN_ENABLE_DWCONV_MULTIPASS
    #endif
  #elif XNN_ARCH_WASM
    f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p1c__wasm_acc2;
    f32_dwconv_config[0].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_3p1c__scalar_acc2;
    f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_dwconv_config[0].channel_tile = 1;
    f32_dwconv_config[0].channel_subtile = 1;
    f32_dwconv_config[0].channel_round = 1;
    f32_dwconv_config[0].primary_tile = 3;

    f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p1c__wasm_acc2;
    f32_dwconv_config[1].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_4p1c__scalar_acc2;
    f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_dwconv_config[1].channel_tile = 1;
    f32_dwconv_config[1].channel_subtile = 1;
    f32_dwconv_config[1].channel_round = 1;
    f32_dwconv_config[1].primary_tile = 4;

    f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p1c__wasm_acc2;
    f32_dwconv_config[2].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_9p1c__scalar_acc2;
    f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_dwconv_config[2].channel_tile = 1;
    f32_dwconv_config[2].channel_subtile = 1;
    f32_dwconv_config[2].channel_round = 1;
    f32_dwconv_config[2].primary_tile = 9;

    #if XNN_ENABLE_DWCONV_MULTIPASS
      f32_dwconv_config[3].minmax.multipass = (xnn_dwconv_multipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2;
      f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[3].channel_tile = 1;
      f32_dwconv_config[3].channel_subtile = 1;
      f32_dwconv_config[3].channel_round = 1;
      f32_dwconv_config[3].primary_tile = 5;
      f32_dwconv_config[3].middle_tile = 5;
      f32_dwconv_config[3].last_tile = 5;
    #else
      f32_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_25p1c__wasm_acc2;
      f32_dwconv_config[3].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_25p1c__scalar_acc2;
      f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_dwconv_config[3].channel_tile = 1;
      f32_dwconv_config[3].channel_subtile = 1;
      f32_dwconv_config[3].channel_round = 1;
      f32_dwconv_config[3].primary_tile = 25;
    #endif  // XNN_ENABLE_DWCONV_MULTIPASS
  #else
    f32_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_3p1c__scalar_acc2;
    f32_dwconv_config[0].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_3p1c__scalar_acc2;
    f32_dwconv_config[0].init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_dwconv_config[0].channel_tile = 1;
    f32_dwconv_config[0].channel_subtile = 1;
    f32_dwconv_config[0].channel_round = 1;
    f32_dwconv_config[0].primary_tile = 3;

    f32_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_4p1c__scalar_acc2;
    f32_dwconv_config[1].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_4p1c__scalar_acc2;
    f32_dwconv_config[1].init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_dwconv_config[1].channel_tile = 1;
    f32_dwconv_config[1].channel_subtile = 1;
    f32_dwconv_config[1].channel_round = 1;
    f32_dwconv_config[1].primary_tile = 4;

    f32_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_9p1c__scalar_acc2;
    f32_dwconv_config[2].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_9p1c__scalar_acc2;
    f32_dwconv_config[2].init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_dwconv_config[2].channel_tile = 1;
    f32_dwconv_config[2].channel_subtile = 1;
    f32_dwconv_config[2].channel_round = 1;
    f32_dwconv_config[2].primary_tile = 9;

    f32_dwconv_config[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_minmax_ukernel_25p1c__scalar_acc2;
    f32_dwconv_config[3].linear.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_f32_dwconv_ukernel_25p1c__scalar_acc2;
    f32_dwconv_config[3].init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_dwconv_config[3].channel_tile = 1;
    f32_dwconv_config[3].channel_subtile = 1;
    f32_dwconv_config[3].channel_round = 1;
    f32_dwconv_config[3].primary_tile = 25;
  #endif
}

static void init_qs8_qc8w_dwconv_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_v8) {
        qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__asm_aarch32_neonv8_mla8_cortex_a35;
        qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
        qs8_qc8w_dwconv_config[0].channel_tile = 16;
        qs8_qc8w_dwconv_config[0].channel_subtile = 16;
        qs8_qc8w_dwconv_config[0].channel_round = 1;
        qs8_qc8w_dwconv_config[0].primary_tile = 3;
        qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neonv8_mla8_ld64;
        qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
        qs8_qc8w_dwconv_config[1].channel_tile = 16;
        qs8_qc8w_dwconv_config[1].channel_subtile = 16;
        qs8_qc8w_dwconv_config[1].channel_round = 1;
        qs8_qc8w_dwconv_config[1].primary_tile = 9;
        qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__neonv8_mla8_ld64;
        qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
        qs8_qc8w_dwconv_config[2].channel_tile = 8;
        qs8_qc8w_dwconv_config[2].channel_subtile = 8;
        qs8_qc8w_dwconv_config[2].channel_round = 1;
        qs8_qc8w_dwconv_config[2].primary_tile = 25;
      } else {
        qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__neon_mla8_ld128;
        qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params;
        qs8_qc8w_dwconv_config[0].channel_tile = 16;
        qs8_qc8w_dwconv_config[0].channel_subtile = 16;
        qs8_qc8w_dwconv_config[0].channel_round = 1;
        qs8_qc8w_dwconv_config[0].primary_tile = 3;
        qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neon_mla8_ld64;
        qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params;
        qs8_qc8w_dwconv_config[1].channel_tile = 16;
        qs8_qc8w_dwconv_config[1].channel_subtile = 16;
        qs8_qc8w_dwconv_config[1].channel_round = 1;
        qs8_qc8w_dwconv_config[1].primary_tile = 9;
        qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__neon_mla8_ld64;
        qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params;
        qs8_qc8w_dwconv_config[2].channel_tile = 8;
        qs8_qc8w_dwconv_config[2].channel_subtile = 8;
        qs8_qc8w_dwconv_config[2].channel_round = 1;
        qs8_qc8w_dwconv_config[2].primary_tile = 25;
      }
    } else {
      qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p1c__scalar_fmagic;
      qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params;
      qs8_qc8w_dwconv_config[0].channel_tile = 1;
      qs8_qc8w_dwconv_config[0].channel_subtile = 1;
      qs8_qc8w_dwconv_config[0].channel_round = 1;
      qs8_qc8w_dwconv_config[0].primary_tile = 3;
      qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p1c__scalar_fmagic;
      qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params;
      qs8_qc8w_dwconv_config[1].channel_tile = 1;
      qs8_qc8w_dwconv_config[1].channel_subtile = 1;
      qs8_qc8w_dwconv_config[1].channel_round = 1;
      qs8_qc8w_dwconv_config[1].primary_tile = 9;
      qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p1c__scalar_fmagic;
      qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params;
      qs8_qc8w_dwconv_config[2].channel_tile = 1;
      qs8_qc8w_dwconv_config[2].channel_subtile = 1;
      qs8_qc8w_dwconv_config[2].channel_round = 1;
      qs8_qc8w_dwconv_config[2].primary_tile = 25;
    }
  #elif XNN_ARCH_ARM64
    qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__neonv8_mla8_ld128;
    qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
    qs8_qc8w_dwconv_config[0].channel_tile = 16;
    qs8_qc8w_dwconv_config[0].channel_subtile = 16;
    qs8_qc8w_dwconv_config[0].channel_round = 1;
    qs8_qc8w_dwconv_config[0].primary_tile = 3;
    qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neonv8_mla8_ld64;
    qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
    qs8_qc8w_dwconv_config[1].channel_tile = 16;
    qs8_qc8w_dwconv_config[1].channel_subtile = 16;
    qs8_qc8w_dwconv_config[1].channel_round = 1;
    qs8_qc8w_dwconv_config[1].primary_tile = 9;
    qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neonv8_mla8_ld64;
    qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
    qs8_qc8w_dwconv_config[2].channel_tile = 16;
    qs8_qc8w_dwconv_config[2].channel_subtile = 16;
    qs8_qc8w_dwconv_config[2].channel_round = 1;
    qs8_qc8w_dwconv_config[2].primary_tile = 25;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p32c__avx512skx_mul32;
      qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params;
      qs8_qc8w_dwconv_config[0].channel_tile = 32;
      qs8_qc8w_dwconv_config[0].channel_subtile = 32;
      qs8_qc8w_dwconv_config[0].channel_round = 1;
      qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32;
      qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params;
      qs8_qc8w_dwconv_config[1].channel_tile = 32;
      qs8_qc8w_dwconv_config[1].channel_subtile = 32;
      qs8_qc8w_dwconv_config[1].channel_round = 1;
      qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__avx512skx_mul32;
      qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params;
      qs8_qc8w_dwconv_config[2].channel_tile = 32;
      qs8_qc8w_dwconv_config[2].channel_subtile = 32;
      qs8_qc8w_dwconv_config[2].channel_round = 1;
    } else if (hardware_config->use_x86_avx2) {
      qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__avx2_mul32;
      qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params;
      qs8_qc8w_dwconv_config[0].channel_tile = 16;
      qs8_qc8w_dwconv_config[0].channel_subtile = 16;
      qs8_qc8w_dwconv_config[0].channel_round = 1;
      qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul32;
      qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params;
      qs8_qc8w_dwconv_config[1].channel_tile = 16;
      qs8_qc8w_dwconv_config[1].channel_subtile = 16;
      qs8_qc8w_dwconv_config[1].channel_round = 1;
      qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul32;
      qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params;
      qs8_qc8w_dwconv_config[2].channel_tile = 16;
      qs8_qc8w_dwconv_config[2].channel_subtile = 16;
      qs8_qc8w_dwconv_config[2].channel_round = 1;
    } else if (hardware_config->use_x86_avx) {
      qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__avx_mul16_add16;
      qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params;
      qs8_qc8w_dwconv_config[0].channel_tile = 16;
      qs8_qc8w_dwconv_config[0].channel_subtile = 16;
      qs8_qc8w_dwconv_config[0].channel_round = 1;
      qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__avx_mul16_add16;
      qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params;
      qs8_qc8w_dwconv_config[1].channel_tile = 16;
      qs8_qc8w_dwconv_config[1].channel_subtile = 16;
      qs8_qc8w_dwconv_config[1].channel_round = 1;
      qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__avx_mul16_add16;
      qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params;
      qs8_qc8w_dwconv_config[2].channel_tile = 16;
      qs8_qc8w_dwconv_config[2].channel_subtile = 16;
      qs8_qc8w_dwconv_config[2].channel_round = 1;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__sse41_mul16;
      qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params;
      qs8_qc8w_dwconv_config[0].channel_tile = 8;
      qs8_qc8w_dwconv_config[0].channel_subtile = 8;
      qs8_qc8w_dwconv_config[0].channel_round = 1;
      qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul16;
      qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params;
      qs8_qc8w_dwconv_config[1].channel_tile = 8;
      qs8_qc8w_dwconv_config[1].channel_subtile = 8;
      qs8_qc8w_dwconv_config[1].channel_round = 1;
      qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul16;
      qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params;
      qs8_qc8w_dwconv_config[2].channel_tile = 8;
      qs8_qc8w_dwconv_config[2].channel_subtile = 8;
      qs8_qc8w_dwconv_config[2].channel_round = 1;
    } else {
      qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__sse2_mul16;
      qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params;
      qs8_qc8w_dwconv_config[0].channel_tile = 8;
      qs8_qc8w_dwconv_config[0].channel_subtile = 8;
      qs8_qc8w_dwconv_config[0].channel_round = 1;
      qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__sse2_mul16;
      qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params;
      qs8_qc8w_dwconv_config[1].channel_tile = 8;
      qs8_qc8w_dwconv_config[1].channel_subtile = 8;
      qs8_qc8w_dwconv_config[1].channel_round = 1;
      qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__sse2_mul16;
      qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params;
      qs8_qc8w_dwconv_config[2].channel_tile = 8;
      qs8_qc8w_dwconv_config[2].channel_subtile = 8;
      qs8_qc8w_dwconv_config[2].channel_round = 1;
    }
    qs8_qc8w_dwconv_config[0].primary_tile = 3;
    qs8_qc8w_dwconv_config[1].primary_tile = 9;
    qs8_qc8w_dwconv_config[2].primary_tile = 25;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__wasmsimd_mul16_add16;
    qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params;
    qs8_qc8w_dwconv_config[0].channel_tile = 16;
    qs8_qc8w_dwconv_config[0].channel_subtile = 16;
    qs8_qc8w_dwconv_config[0].channel_round = 1;
    qs8_qc8w_dwconv_config[0].primary_tile = 3;
    qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__wasmsimd_mul16_add16;
    qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params;
    qs8_qc8w_dwconv_config[1].channel_tile = 16;
    qs8_qc8w_dwconv_config[1].channel_subtile = 16;
    qs8_qc8w_dwconv_config[1].channel_round = 1;
    qs8_qc8w_dwconv_config[1].primary_tile = 9;
    qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__wasmsimd_mul16_add16;
    qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params;
    qs8_qc8w_dwconv_config[2].channel_tile = 16;
    qs8_qc8w_dwconv_config[2].channel_subtile = 16;
    qs8_qc8w_dwconv_config[2].channel_round = 1;
    qs8_qc8w_dwconv_config[2].primary_tile = 25;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p2c__scalar_imagic;
      qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params;
      qs8_qc8w_dwconv_config[0].channel_tile = 2;
      qs8_qc8w_dwconv_config[0].channel_subtile = 2;
      qs8_qc8w_dwconv_config[0].channel_round = 1;
      qs8_qc8w_dwconv_config[0].primary_tile = 3;
      qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p2c__scalar_imagic;
      qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params;
      qs8_qc8w_dwconv_config[1].channel_tile = 2;
      qs8_qc8w_dwconv_config[1].channel_subtile = 2;
      qs8_qc8w_dwconv_config[1].channel_round = 1;
      qs8_qc8w_dwconv_config[1].primary_tile = 9;
      qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p1c__scalar_imagic;
      qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params;
      qs8_qc8w_dwconv_config[2].channel_tile = 1;
      qs8_qc8w_dwconv_config[2].channel_subtile = 1;
      qs8_qc8w_dwconv_config[2].channel_round = 1;
      qs8_qc8w_dwconv_config[2].primary_tile = 25;
    } else {
      qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p2c__wasm_fmagic;
      qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params;
      qs8_qc8w_dwconv_config[0].channel_tile = 2;
      qs8_qc8w_dwconv_config[0].channel_subtile = 2;
      qs8_qc8w_dwconv_config[0].channel_round = 1;
      qs8_qc8w_dwconv_config[0].primary_tile = 3;
      qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p2c__wasm_fmagic;
      qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params;
      qs8_qc8w_dwconv_config[1].channel_tile = 2;
      qs8_qc8w_dwconv_config[1].channel_subtile = 2;
      qs8_qc8w_dwconv_config[1].channel_round = 1;
      qs8_qc8w_dwconv_config[1].primary_tile = 9;
      qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p2c__wasm_fmagic;
      qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params;
      qs8_qc8w_dwconv_config[2].channel_tile = 2;
      qs8_qc8w_dwconv_config[2].channel_subtile = 2;
      qs8_qc8w_dwconv_config[2].channel_round = 1;
      qs8_qc8w_dwconv_config[2].primary_tile = 25;
    }
  #else
    qs8_qc8w_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p2c__scalar_lrintf;
    qs8_qc8w_dwconv_config[0].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params;
    qs8_qc8w_dwconv_config[0].channel_tile = 2;
    qs8_qc8w_dwconv_config[0].channel_subtile = 2;
    qs8_qc8w_dwconv_config[0].channel_round = 1;
    qs8_qc8w_dwconv_config[0].primary_tile = 3;
    qs8_qc8w_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p2c__scalar_lrintf;
    qs8_qc8w_dwconv_config[1].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params;
    qs8_qc8w_dwconv_config[1].channel_tile = 2;
    qs8_qc8w_dwconv_config[1].channel_subtile = 2;
    qs8_qc8w_dwconv_config[1].channel_round = 1;
    qs8_qc8w_dwconv_config[1].primary_tile = 9;
    qs8_qc8w_dwconv_config[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p2c__scalar_lrintf;
    qs8_qc8w_dwconv_config[2].init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params;
    qs8_qc8w_dwconv_config[2].channel_tile = 2;
    qs8_qc8w_dwconv_config[2].channel_subtile = 2;
    qs8_qc8w_dwconv_config[2].channel_round = 1;
    qs8_qc8w_dwconv_config[2].primary_tile = 25;
  #endif
}

static void init_qs8_dwconv_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64;
      qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
      qs8_dwconv_config[0].channel_tile = 16;
      qs8_dwconv_config[0].channel_subtile = 16;
      qs8_dwconv_config[0].channel_round = 1;
      qs8_dwconv_config[0].primary_tile = 9;
      qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64;
      qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
      qs8_dwconv_config[1].channel_tile = 8;
      qs8_dwconv_config[1].channel_subtile = 8;
      qs8_dwconv_config[1].channel_round = 1;
      qs8_dwconv_config[1].primary_tile = 25;
    } else {
      qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_9p1c__scalar_fmagic;
      qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params;
      qs8_dwconv_config[0].channel_tile = 1;
      qs8_dwconv_config[0].channel_subtile = 1;
      qs8_dwconv_config[0].channel_round = 1;
      qs8_dwconv_config[0].primary_tile = 9;
      qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_25p1c__scalar_fmagic;
      qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params;
      qs8_dwconv_config[1].channel_tile = 1;
      qs8_dwconv_config[1].channel_subtile = 1;
      qs8_dwconv_config[1].channel_round = 1;
      qs8_dwconv_config[1].primary_tile = 25;
    }
  #elif XNN_ARCH_ARM64
    qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64;
    qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
    qs8_dwconv_config[0].channel_tile = 16;
    qs8_dwconv_config[0].channel_subtile = 16;
    qs8_dwconv_config[0].channel_round = 1;
    qs8_dwconv_config[0].primary_tile = 9;
    qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64;
    qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
    qs8_dwconv_config[1].channel_tile = 16;
    qs8_dwconv_config[1].channel_subtile = 16;
    qs8_dwconv_config[1].channel_round = 1;
    qs8_dwconv_config[1].primary_tile = 25;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32;
      qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx512_params;
      qs8_dwconv_config[0].channel_tile = 32;
      qs8_dwconv_config[0].channel_subtile = 32;
      qs8_dwconv_config[0].channel_round = 1;
      qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_25p32c__avx512skx_mul32;
      qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx512_params;
      qs8_dwconv_config[1].channel_tile = 32;
      qs8_dwconv_config[1].channel_subtile = 32;
      qs8_dwconv_config[1].channel_round = 1;
    } else if (hardware_config->use_x86_avx2) {
      qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul32;
      qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx2_params;
      qs8_dwconv_config[0].channel_tile = 16;
      qs8_dwconv_config[0].channel_subtile = 16;
      qs8_dwconv_config[0].channel_round = 1;
      qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul32;
      qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx2_params;
      qs8_dwconv_config[1].channel_tile = 16;
      qs8_dwconv_config[1].channel_subtile = 16;
      qs8_dwconv_config[1].channel_round = 1;
    } else if (hardware_config->use_x86_avx) {
      qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx_mul16_add16;
      qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      qs8_dwconv_config[0].channel_tile = 16;
      qs8_dwconv_config[0].channel_subtile = 16;
      qs8_dwconv_config[0].channel_round = 1;
      qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__avx_mul16_add16;
      qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      qs8_dwconv_config[1].channel_tile = 16;
      qs8_dwconv_config[1].channel_subtile = 16;
      qs8_dwconv_config[1].channel_round = 1;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul16_add16;
      qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      qs8_dwconv_config[0].channel_tile = 8;
      qs8_dwconv_config[0].channel_subtile = 8;
      qs8_dwconv_config[0].channel_round = 1;
      qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul16_add16;
      qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      qs8_dwconv_config[1].channel_tile = 8;
      qs8_dwconv_config[1].channel_subtile = 8;
      qs8_dwconv_config[1].channel_round = 1;
    } else {
      qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__sse2_mul16_add16;
      qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse2_params;
      qs8_dwconv_config[0].channel_tile = 8;
      qs8_dwconv_config[0].channel_subtile = 8;
      qs8_dwconv_config[0].channel_round = 1;
      qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_25p8c__sse2_mul16_add16;
      qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse2_params;
      qs8_dwconv_config[1].channel_tile = 8;
      qs8_dwconv_config[1].channel_subtile = 8;
      qs8_dwconv_config[1].channel_round = 1;
    }
    qs8_dwconv_config[0].primary_tile = 9;
    qs8_dwconv_config[1].primary_tile = 25;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__wasmsimd_mul16_add16;
    qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_wasmsimd_params;
    qs8_dwconv_config[0].channel_tile = 16;
    qs8_dwconv_config[0].channel_subtile = 16;
    qs8_dwconv_config[0].channel_round = 1;
    qs8_dwconv_config[0].primary_tile = 9;
    qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__wasmsimd_mul16_add16;
    qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_wasmsimd_params;
    qs8_dwconv_config[1].channel_tile = 16;
    qs8_dwconv_config[1].channel_subtile = 16;
    qs8_dwconv_config[1].channel_round = 1;
    qs8_dwconv_config[1].primary_tile = 25;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_9p2c__scalar_imagic;
      qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params;
      qs8_dwconv_config[0].channel_tile = 2;
      qs8_dwconv_config[0].channel_subtile = 2;
      qs8_dwconv_config[0].channel_round = 1;
      qs8_dwconv_config[0].primary_tile = 9;
      qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_25p1c__scalar_imagic;
      qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params;
      qs8_dwconv_config[1].channel_tile = 1;
      qs8_dwconv_config[1].channel_subtile = 1;
      qs8_dwconv_config[1].channel_round = 1;
      qs8_dwconv_config[1].primary_tile = 25;
    } else {
      qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_9p2c__wasm_fmagic;
      qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params;
      qs8_dwconv_config[0].channel_tile = 2;
      qs8_dwconv_config[0].channel_subtile = 2;
      qs8_dwconv_config[0].channel_round = 1;
      qs8_dwconv_config[0].primary_tile = 9;
      qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_25p2c__wasm_fmagic;
      qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params;
      qs8_dwconv_config[1].channel_tile = 2;
      qs8_dwconv_config[1].channel_subtile = 2;
      qs8_dwconv_config[1].channel_round = 1;
      qs8_dwconv_config[1].primary_tile = 25;
    }
  #else
    qs8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_9p2c__scalar_lrintf;
    qs8_dwconv_config[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params;
    qs8_dwconv_config[0].channel_tile = 2;
    qs8_dwconv_config[0].channel_subtile = 2;
    qs8_dwconv_config[0].channel_round = 1;
    qs8_dwconv_config[0].primary_tile = 9;
    qs8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qs8_dwconv_minmax_fp32_ukernel_25p2c__scalar_lrintf;
    qs8_dwconv_config[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params;
    qs8_dwconv_config[1].channel_tile = 2;
    qs8_dwconv_config[1].channel_subtile = 2;
    qs8_dwconv_config[1].channel_round = 1;
    qs8_dwconv_config[1].primary_tile = 25;
  #endif
}

static void init_qu8_dwconv_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8;
      qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
      qu8_dwconv_config[0].channel_tile = 16;
      qu8_dwconv_config[0].channel_subtile = 16;
      qu8_dwconv_config[0].channel_round = 1;
      qu8_dwconv_config[0].primary_tile = 9;
      qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8;
      qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
      qu8_dwconv_config[1].channel_tile = 8;
      qu8_dwconv_config[1].channel_subtile = 8;
      qu8_dwconv_config[1].channel_round = 1;
      qu8_dwconv_config[1].primary_tile = 25;
    } else {
      qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_9p1c__scalar_fmagic;
      qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      qu8_dwconv_config[0].channel_tile = 1;
      qu8_dwconv_config[0].channel_subtile = 1;
      qu8_dwconv_config[0].channel_round = 1;
      qu8_dwconv_config[0].primary_tile = 9;
      qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_25p1c__scalar_fmagic;
      qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      qu8_dwconv_config[1].channel_tile = 1;
      qu8_dwconv_config[1].channel_subtile = 1;
      qu8_dwconv_config[1].channel_round = 1;
      qu8_dwconv_config[1].primary_tile = 25;
    }
  #elif XNN_ARCH_ARM64
    qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8;
    qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
    qu8_dwconv_config[0].channel_tile = 16;
    qu8_dwconv_config[0].channel_subtile = 16;
    qu8_dwconv_config[0].channel_round = 1;
    qu8_dwconv_config[0].primary_tile = 9;
    qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8;
    qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
    qu8_dwconv_config[1].channel_tile = 8;
    qu8_dwconv_config[1].channel_subtile = 8;
    qu8_dwconv_config[1].channel_round = 1;
    qu8_dwconv_config[1].primary_tile = 25;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32;
      qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx512_params;
      qu8_dwconv_config[0].channel_tile = 32;
      qu8_dwconv_config[0].channel_subtile = 32;
      qu8_dwconv_config[0].channel_round = 1;
      qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_25p32c__avx512skx_mul32;
      qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx512_params;
      qu8_dwconv_config[1].channel_tile = 32;
      qu8_dwconv_config[1].channel_subtile = 32;
      qu8_dwconv_config[1].channel_round = 1;
    } else if (hardware_config->use_x86_avx2) {
      qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul32;
      qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx2_params;
      qu8_dwconv_config[0].channel_tile = 16;
      qu8_dwconv_config[0].channel_subtile = 16;
      qu8_dwconv_config[0].channel_round = 1;
      qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul32;
      qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx2_params;
      qu8_dwconv_config[1].channel_tile = 16;
      qu8_dwconv_config[1].channel_subtile = 16;
      qu8_dwconv_config[1].channel_round = 1;
    } else if (hardware_config->use_x86_avx) {
      qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__avx_mul16;
      qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_dwconv_config[0].channel_tile = 16;
      qu8_dwconv_config[0].channel_subtile = 16;
      qu8_dwconv_config[0].channel_round = 1;
      qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__avx_mul16;
      qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_dwconv_config[1].channel_tile = 16;
      qu8_dwconv_config[1].channel_subtile = 16;
      qu8_dwconv_config[1].channel_round = 1;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul16;
      qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_dwconv_config[0].channel_tile = 8;
      qu8_dwconv_config[0].channel_subtile = 8;
      qu8_dwconv_config[0].channel_round = 1;
      qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul16;
      qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_dwconv_config[1].channel_tile = 8;
      qu8_dwconv_config[1].channel_subtile = 8;
      qu8_dwconv_config[1].channel_round = 1;
    } else {
      qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__sse2_mul16;
      qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_dwconv_config[0].channel_tile = 8;
      qu8_dwconv_config[0].channel_subtile = 8;
      qu8_dwconv_config[0].channel_round = 1;
      qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__sse2_mul16;
      qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_dwconv_config[1].channel_tile = 8;
      qu8_dwconv_config[1].channel_subtile = 8;
      qu8_dwconv_config[1].channel_round = 1;
    }
    qu8_dwconv_config[0].primary_tile = 9;
    qu8_dwconv_config[1].primary_tile = 25;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__wasmsimd_mul16;
    qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_wasmsimd_params;
    qu8_dwconv_config[0].channel_tile = 8;
    qu8_dwconv_config[0].channel_subtile = 8;
    qu8_dwconv_config[0].channel_round = 1;
    qu8_dwconv_config[0].primary_tile = 9;
    qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__wasmsimd_mul16;
    qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_wasmsimd_params;
    qu8_dwconv_config[1].channel_tile = 8;
    qu8_dwconv_config[1].channel_subtile = 8;
    qu8_dwconv_config[1].channel_round = 1;
    qu8_dwconv_config[1].primary_tile = 25;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_9p2c__scalar_imagic;
      qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params;
      qu8_dwconv_config[0].channel_tile = 2;
      qu8_dwconv_config[0].channel_subtile = 2;
      qu8_dwconv_config[0].channel_round = 1;
      qu8_dwconv_config[0].primary_tile = 9;
      qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_25p1c__scalar_imagic;
      qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params;
      qu8_dwconv_config[1].channel_tile = 1;
      qu8_dwconv_config[1].channel_subtile = 1;
      qu8_dwconv_config[1].channel_round = 1;
      qu8_dwconv_config[1].primary_tile = 25;
    } else {
      qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_9p2c__wasm_fmagic;
      qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      qu8_dwconv_config[0].channel_tile = 2;
      qu8_dwconv_config[0].channel_subtile = 2;
      qu8_dwconv_config[0].channel_round = 1;
      qu8_dwconv_config[0].primary_tile = 9;
      qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_25p2c__wasm_fmagic;
      qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      qu8_dwconv_config[1].channel_tile = 2;
      qu8_dwconv_config[1].channel_subtile = 2;
      qu8_dwconv_config[1].channel_round = 1;
      qu8_dwconv_config[1].primary_tile = 25;
    }
  #else
    qu8_dwconv_config[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_9p2c__scalar_lrintf;
    qu8_dwconv_config[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params;
    qu8_dwconv_config[0].channel_tile = 2;
    qu8_dwconv_config[0].channel_subtile = 2;
    qu8_dwconv_config[0].channel_round = 1;
    qu8_dwconv_config[0].primary_tile = 9;
    qu8_dwconv_config[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_fn) xnn_qu8_dwconv_minmax_fp32_ukernel_25p2c__scalar_lrintf;
    qu8_dwconv_config[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params;
    qu8_dwconv_config[1].channel_tile = 2;
    qu8_dwconv_config[1].channel_subtile = 2;
    qu8_dwconv_config[1].channel_round = 1;
    qu8_dwconv_config[1].primary_tile = 25;
  #endif
}

struct xnn_dwconv_config* xnn_init_f16_dwconv_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_dwconv);
  return f16_dwconv_config;
}

struct xnn_dwconv_config* xnn_init_f32_dwconv_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_dwconv);
  return f32_dwconv_config;
}

struct xnn_dwconv_config* xnn_init_qs8_qc8w_dwconv_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_qc8w_dwconv);
  return qs8_qc8w_dwconv_config;
}

struct xnn_dwconv_config* xnn_init_qs8_dwconv_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_dwconv);
  return qs8_dwconv_config;
}

struct xnn_dwconv_config* xnn_init_qu8_dwconv_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_dwconv);
  return qu8_dwconv_config;
}
