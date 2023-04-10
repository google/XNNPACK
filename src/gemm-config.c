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

#ifndef __EMSCRIPTEN__
  #include <cpuinfo.h>
#endif

#include <xnnpack/common.h>
#include <xnnpack/config.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/pack.h>
#include <xnnpack/packw.h>

#define XNN_MR_TO_INDEX(MR) (MR-1)


static struct xnn_gemm_config f16_gemm_config = {0};
static struct xnn_gemm_config f32_gemm_config = {0};
static struct xnn_gemm_config f32_gemm2_config = {0};
static struct xnn_gemm_config qc8_gemm_config = {0};
static struct xnn_gemm_config qs8_gemm_config = {0};
static struct xnn_gemm_config qu8_gemm_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_gemm2 = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qc8_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qs8_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qu8_gemm = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_gemm2 = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qc8_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qs8_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qu8_gemm = PTHREAD_ONCE_INIT;
#endif

static void init_f16_gemm_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
      f16_gemm_config.mr = 6;
      f16_gemm_config.nr = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
        #if XNN_ENABLE_ASSEMBLY
          switch (cpuinfo_get_core(0)->uarch) {
            case cpuinfo_uarch_cortex_a55:
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55);
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
              f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4;
              f16_gemm_config.mr = 6;
              f16_gemm_config.nr = 16;
              #if XNN_ENABLE_JIT
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55);
              #endif
              break;
            case cpuinfo_uarch_cortex_a55r0:
            case cpuinfo_uarch_cortex_a75:
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0);
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
              f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4;
              f16_gemm_config.mr = 6;
              f16_gemm_config.nr = 16;
              #if XNN_ENABLE_JIT
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0);
              #endif
              break;
            case cpuinfo_uarch_exynos_m5:
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
              f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4;
              f16_gemm_config.mr = 4;
              f16_gemm_config.nr = 16;
              #if XNN_ENABLE_JIT
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_4x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64);
              #endif
              break;
            case cpuinfo_uarch_exynos_m4:
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
              f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4;
              f16_gemm_config.mr = 6;
              f16_gemm_config.nr = 16;
              #if XNN_ENABLE_JIT
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64);
              #endif
              break;
            default:
            case cpuinfo_uarch_cortex_a76:
            case cpuinfo_uarch_cortex_a77:
            case cpuinfo_uarch_cortex_a78:
            case cpuinfo_uarch_cortex_x1:
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75);
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
              f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4;
              f16_gemm_config.mr = 6;
              f16_gemm_config.nr = 16;
              #if XNN_ENABLE_JIT
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75);
              #endif
              break;
          }

          #if XNN_MAX_UARCH_TYPES > 1
          {
            /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
            const uint32_t mr = f16_gemm_config.mr;
            const uint32_t nr = f16_gemm_config.nr;
            for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
              const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
              if (uarch_info == NULL) {
                /* No more microarchitectures in the system */
                break;
              }

              switch (uarch_info->uarch) {
                case cpuinfo_uarch_cortex_a55:
                  if (mr == 6 && nr == 16) {
                    f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55;
                    f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55;
                    #if XNN_ENABLE_JIT
                      f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55);
                      f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55);
                    #endif
                  }
                  break;
                case cpuinfo_uarch_cortex_a55r0:
                case cpuinfo_uarch_cortex_a75:
                  if (mr == 6 && nr == 16) {
                    f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0;
                    f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0;
                    #if XNN_ENABLE_JIT
                      f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0);
                      f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0);
                    #endif
                  }
                  break;
                default:
                  break;
              }
            }
          }
          #endif  // XNN_MAX_UARCH_TYPES > 1
        #else  // XNN_ENABLE_ASSEMBLY
          f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64);
          f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64);
          f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64);
          f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64);
          f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
          f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4;
          f16_gemm_config.mr = 6;
          f16_gemm_config.nr = 16;
        #endif  // XNN_ENABLE_ASSEMBLY
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_4x16__avx2_broadcast);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast);
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__avx2_broadcast);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast);
      f16_gemm_config.init.f16 = xnn_init_f16_minmax_avx_params;
      f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f16_gemm_goi_w;
      f16_gemm_config.mr = 4;
      f16_gemm_config.nr = 16;
    }
  #endif
}

static void init_f32_gemm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        switch (cpuinfo_get_uarch(0)->uarch) {
          case cpuinfo_uarch_cortex_a5:
          case cpuinfo_uarch_cortex_a7:
          case cpuinfo_uarch_krait:
          case cpuinfo_uarch_kryo:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a53:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a53);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a55r0:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a32:
          case cpuinfo_uarch_cortex_a35:
          case cpuinfo_uarch_cortex_a55:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;

          case cpuinfo_uarch_cortex_a57:
          case cpuinfo_uarch_cortex_a72:
          case cpuinfo_uarch_cortex_a73:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a75);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a75);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;

          default:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            #if XNN_ENABLE_JIT
              f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75);
              f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75);
            #endif
            break;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = f32_gemm_config.mr;
          const uint32_t nr = f32_gemm_config.nr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a53;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53;
                  #if XNN_ENABLE_JIT
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_prfm_cortex_a53;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_prfm_cortex_a53;
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_prfm_cortex_a53;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_prfm_cortex_a53;
                  #endif
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  #if XNN_ENABLE_JIT
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53;
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53;
                  #endif
                }
                break;
              case cpuinfo_uarch_cortex_a55:
                if (mr == 4 && nr == 8) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  #if XNN_ENABLE_JIT
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53;
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a55;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55;
                  #endif
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld128);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
        f32_gemm_config.mr = 4;
        f32_gemm_config.nr = 8;
      #endif  // XNN_ENABLE_ASSEMBLY
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x4__scalar);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x4__scalar);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__scalar);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__scalar);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x4__scalar);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x4__scalar);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__scalar);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__scalar);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x4__scalar);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_ARM64
    #if XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_cortex_a72:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a75);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a75);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
          f32_gemm_config.mr = 4;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a57:
        case cpuinfo_uarch_cortex_a75:
        case cpuinfo_uarch_cortex_a76:
        case cpuinfo_uarch_exynos_m3:
        case cpuinfo_uarch_exynos_m4:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a75);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a75);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a75);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a75);
          #endif
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          #if XNN_ENABLE_JIT
            f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
            f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
            #if XNN_ENABLE_GEMM_M_SPECIALIZATION
              f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75);
              f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75);
            #endif
            f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75);
            f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75);
          #endif
          break;
        case cpuinfo_uarch_exynos_m1:
        case cpuinfo_uarch_exynos_m2:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8s4__neonfma);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8s4__neonfma);
          #endif
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8s4__neonfma);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          f32_gemm_config.log2_sr = 2;
          break;
        case cpuinfo_uarch_cortex_a53:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a53);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a53);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a53);
          #endif
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a53);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a53);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a55r0:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53);
          #endif
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a35:
        case cpuinfo_uarch_cortex_a55:
        case cpuinfo_uarch_kryo:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55);
          #endif
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a73:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a77:
        case cpuinfo_uarch_exynos_m5:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
          f32_gemm_config.mr = 4;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a78:
        case cpuinfo_uarch_cortex_x1:
        default:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128);
          #endif
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          #if XNN_ENABLE_JIT
            f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_ld64);
            #if XNN_ENABLE_GEMM_M_SPECIALIZATION
              f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_ld128);
              f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128);
            #endif
            f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128);
            f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128);
          #endif
          break;
      }
      #if XNN_MAX_UARCH_TYPES > 1
      {
        /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
        const uint32_t mr = f32_gemm_config.mr;
        const uint32_t nr = f32_gemm_config.nr;
        const uint32_t log2_sr = f32_gemm_config.log2_sr;
        for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
          const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
          if (uarch_info == NULL) {
            /* No more microarchitectures in the system */
            break;
          }

          switch (uarch_info->uarch) {
            case cpuinfo_uarch_cortex_a53:
              if (mr == 6 && nr == 8 && log2_sr == 0) {
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a53;
                #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a53;
                #endif
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a53;
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a53;
                  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a53;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a53;
                  #endif
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a53;
                #endif
              } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a53;
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a53;
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a53;
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a53;
                #endif
              }
              break;
            case cpuinfo_uarch_cortex_a55r0:
              if (mr == 6 && nr == 8 && log2_sr == 0) {
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53;
                #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                #endif
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53;
                  #endif
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53;
                #endif
              } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53;
                #endif
              }
              break;
            case cpuinfo_uarch_cortex_a55:
              if (mr == 6 && nr == 8 && log2_sr == 0) {
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55;
                #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                #endif
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a55;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55;
                #endif
              } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a55;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55;
                #endif
              }
              break;
            default:
              break;
          }
        }
      }
      #endif  // XNN_MAX_UARCH_TYPES > 1
    #else  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a75);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a75);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
        f32_gemm_config.mr = 6;
        f32_gemm_config.nr = 8;
      #else  // !XNN_ENABLE_ASSEMBLY
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4;
        f32_gemm_config.mr = 6;
        f32_gemm_config.nr = 8;
       #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_7x16__avx512f_broadcast);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_prfm_x4;
      f32_gemm_config.mr = 7;
      f32_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_fma3) {
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_zen:
        case cpuinfo_uarch_dhyana:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x16s4__fma3_broadcast);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_avx_params;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4;
          f32_gemm_config.mr = 4;
          f32_gemm_config.nr = 16;
          f32_gemm_config.log2_sr = 2;
          break;
        default:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_avx_params;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4;
          f32_gemm_config.mr = 5;
          f32_gemm_config.nr = 16;
          break;
      }
    } else if (hardware_config->use_x86_avx) {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_5x16__avx_broadcast);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x16__avx_broadcast);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_avx_params;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4;
      f32_gemm_config.mr = 5;
      f32_gemm_config.nr = 16;
    } else {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__sse_load1);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__sse_load1);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__sse_load1);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__sse_load1);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_sse_params;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
      #else
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_splat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x8__wasmsimd_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x8__wasmsimd_splat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmsimd_splat);
      #endif

      f32_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 8;
    } else {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4;
        f32_gemm_config.mr = 6;
        f32_gemm_config.nr = 8;
      #else
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_splat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4;
        f32_gemm_config.mr = 5;
        f32_gemm_config.nr = 8;
      #endif
    }
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_2x4__scalar);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_2x4__scalar);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_2x4__scalar);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_2x4__scalar);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_2x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_2x4__scalar);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4;
      f32_gemm_config.mr = 2;
      f32_gemm_config.nr = 4;
    } else {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x4__wasm);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x4__wasm);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x4__wasm);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x4__wasm);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x4__scalar);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_RISCV
    f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x4__scalar);
    f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x4__scalar);
    f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__scalar);
    f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__scalar);
    f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x4__scalar);
    f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x4__scalar);
    f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__scalar);
    f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__scalar);
    f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x4__scalar);
    f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x4__scalar);
    f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
    f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
    f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4;
    f32_gemm_config.mr = 4;
    f32_gemm_config.nr = 4;
  #endif
}

static void init_f32_gemm2_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64);
      f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64);
      f32_gemm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_prfm_x2;
      f32_gemm2_config.mr = 4;
      f32_gemm2_config.nr = 2;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__scalar);
      f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__scalar);
      f32_gemm2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2__scalar);
      f32_gemm2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2__scalar);
      f32_gemm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4;
      f32_gemm2_config.mr = 4;
      f32_gemm2_config.nr = 2;
    }
  #elif XNN_ARCH_ARM64
    #if XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_prfm_cortex_a75);
      f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_prfm_cortex_a75);
      f32_gemm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_prfm_x2;
      f32_gemm2_config.mr = 4;
      f32_gemm2_config.nr = 2;
    #else  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_prfm_cortex_a75);
        f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_prfm_cortex_a75);
        f32_gemm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_prfm_x2;
        f32_gemm2_config.mr = 4;
        f32_gemm2_config.nr = 2;
      #else  // !XNN_ENABLE_ASSEMBLY
        f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64);
        f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64);
        f32_gemm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_prfm_x2;
        f32_gemm2_config.mr = 4;
        f32_gemm2_config.nr = 2;
       #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__sse);
    f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__sse);
    f32_gemm2_config.init.f32 = xnn_init_f32_minmax_sse_params;
    f32_gemm2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4;
    f32_gemm2_config.mr = 4;
    f32_gemm2_config.nr = 2;
    f32_gemm2_config.log2_kr = 2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
      #else
        f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_x86);
        f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_x86);
        f32_gemm2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmsimd);
        f32_gemm2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmsimd);
      #endif

      f32_gemm2_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_gemm2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4;
      f32_gemm2_config.mr = 4;
      f32_gemm2_config.nr = 2;
      f32_gemm2_config.log2_kr = 2;
    } else {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
      #else
        f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_arm);
        f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_arm);
        f32_gemm2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmsimd);
        f32_gemm2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmsimd);
      #endif

      f32_gemm2_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_gemm2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4;
      f32_gemm2_config.mr = 4;
      f32_gemm2_config.nr = 2;
      f32_gemm2_config.log2_kr = 2;
    }
  #elif XNN_ARCH_WASM
    f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__wasm);
    f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__wasm);
    f32_gemm2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2__scalar);
    f32_gemm2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2__scalar);
    f32_gemm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4;
    f32_gemm2_config.mr = 4;
    f32_gemm2_config.nr = 2;
  #elif XNN_ARCH_RISCV
    f32_gemm2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__scalar);
    f32_gemm2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__scalar);
    f32_gemm2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2__scalar);
    f32_gemm2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2__scalar);
    f32_gemm2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4;
    f32_gemm2_config.mr = 4;
    f32_gemm2_config.nr = 2;
  #endif
}

static void init_qc8_gemm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            switch (cpuinfo_get_uarch(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55);
                qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55);
                qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c4__neondot);
                qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c4__neondot);
                qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
                qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
                qc8_gemm_config.mr = 4;
                qc8_gemm_config.nr = 8;
                qc8_gemm_config.log2_kr = 2;
                break;
              default:
                qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64);
                qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64);
                qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c4__neondot);
                qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c4__neondot);
                qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
                qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
                qc8_gemm_config.mr = 4;
                qc8_gemm_config.nr = 8;
                qc8_gemm_config.log2_kr = 2;
                break;
            }
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          switch (cpuinfo_get_uarch(0)->uarch) {
            case cpuinfo_uarch_cortex_a5:
            case cpuinfo_uarch_cortex_a7:
            case cpuinfo_uarch_krait:
            case cpuinfo_uarch_kryo:
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neon_params;
              qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qc8_gemm_config.mr = 4;
              qc8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a32:
            case cpuinfo_uarch_cortex_a35:
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
              qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qc8_gemm_config.mr = 4;
              qc8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a57:
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a53);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a53);
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35);
              qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
              qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qc8_gemm_config.mr = 4;
              qc8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a55r0:
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53);
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
              qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qc8_gemm_config.mr = 4;
              qc8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a72:
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
              qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
              qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qc8_gemm_config.mr = 2;
              qc8_gemm_config.nr = 8;
              qc8_gemm_config.log2_kr = 1;
              qc8_gemm_config.log2_sr = 2;
              break;
            case cpuinfo_uarch_exynos_m1:
            case cpuinfo_uarch_exynos_m2:
            case cpuinfo_uarch_exynos_m3:
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_prfm_ld64);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_prfm_ld64);
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35);
              qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
              qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qc8_gemm_config.mr = 4;
              qc8_gemm_config.nr = 8;
              break;

            default:
              if (hardware_config->use_arm_neon_v8) {
                qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64);
                qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64);
                qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
                qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
                qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
                qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
                qc8_gemm_config.mr = 4;
                qc8_gemm_config.nr = 8;
              } else {
                qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
                qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
                qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
                qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
                qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neon_params;
                qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
                qc8_gemm_config.mr = 4;
                qc8_gemm_config.nr = 8;
              }
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qc8_gemm_config.mr;
          const uint32_t nr = qc8_gemm_config.nr;
          const uint32_t log2_kr = qc8_gemm_config.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 8 && log2_kr == 2 && hardware_config->use_arm_neon_dot) {
                    qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                    qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                    qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c4__neondot;
                    qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c4__neondot;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
                break;
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a53;
                  qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a53;
                  qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35;
                  qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_prfm_cortex_a35;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53;
                  qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53;
                  qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35;
                  qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35;
                }
                break;

              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x8c4__neondot);
            qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x8c4__neondot);
            qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c4__neondot);
            qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c4__neondot);
            qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
            qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qc8_gemm_config.mr = 4;
            qc8_gemm_config.nr = 8;
            qc8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_v8) {
          qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
          qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qc8_gemm_config.mr = 2;
          qc8_gemm_config.nr = 8;
          qc8_gemm_config.log2_kr = 1;
          qc8_gemm_config.log2_sr = 2;
        } else {
          qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal);
          qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal);
          qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal);
          qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal);
          qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neon_params;
          qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qc8_gemm_config.mr = 2;
          qc8_gemm_config.nr = 8;
          qc8_gemm_config.log2_kr = 1;
          qc8_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    } else {
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_armsimd32_params;
      qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qc8_gemm_config.mr = 2;
      qc8_gemm_config.nr = 2;
      qc8_gemm_config.log2_kr = 2;
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
            qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qc8_gemm_config.mr = 4;
            qc8_gemm_config.nr = 16;
            qc8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
          qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
          qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
          qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
          qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
          qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qc8_gemm_config.mr = 2;
          qc8_gemm_config.nr = 8;
          qc8_gemm_config.log2_kr = 3;
        }
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__neondot);
            qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__neondot);
            qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
            qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qc8_gemm_config.mr = 4;
            qc8_gemm_config.nr = 16;
            qc8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
          qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qc8_gemm_config.mr = 2;
          qc8_gemm_config.nr = 8;
          qc8_gemm_config.log2_kr = 1;
          qc8_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            switch (cpuinfo_get_core(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                break;
              case cpuinfo_uarch_cortex_x1:
              case cpuinfo_uarch_cortex_a78:
                qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                break;
              default:
                qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64);
                qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64);
                break;
            }
            qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
            qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qc8_gemm_config.mr = 4;
            qc8_gemm_config.nr = 16;
            qc8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          switch (cpuinfo_get_core(0)->uarch) {
            case cpuinfo_uarch_cortex_a35:
            case cpuinfo_uarch_kryo:
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64);
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
              qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qc8_gemm_config.mr = 4;
              qc8_gemm_config.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a55r0:
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53);
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
              qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qc8_gemm_config.mr = 4;
              qc8_gemm_config.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a72:
            case cpuinfo_uarch_cortex_a73:
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm);
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm);
              qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
              qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qc8_gemm_config.mr = 2;
              qc8_gemm_config.nr = 8;
              qc8_gemm_config.log2_kr = 3;
              break;

            default:
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
              qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
              qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
              qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
              qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qc8_gemm_config.mr = 2;
              qc8_gemm_config.nr = 8;
              qc8_gemm_config.log2_kr = 3;
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qc8_gemm_config.mr;
          const uint32_t nr = qc8_gemm_config.nr;
          const uint32_t log2_kr = qc8_gemm_config.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a53:
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 2 && nr == 8 && log2_kr == 3) {
                  qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)].function[i] = (xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53;
                  qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)].function[i] = (xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53;
                  qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53;
                  qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53;
                }
                break;

              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 16 && log2_kr == 2 && hardware_config->use_arm_neon_dot) {
                    qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                    qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                    qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c4__neondot;
                    qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c4__neondot;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__neondot);
            qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__neondot);
            qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
            qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qc8_gemm_config.mr = 4;
            qc8_gemm_config.nr = 16;
            qc8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_neonv8_params;
          qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qc8_gemm_config.mr = 2;
          qc8_gemm_config.nr = 8;
          qc8_gemm_config.log2_kr = 1;
          qc8_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_avx512_params;
      qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qc8_gemm_config.mr = 4;
      qc8_gemm_config.nr = 16;
      qc8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_xop) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_sse4_params;
      qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qc8_gemm_config.mr = 2;
      qc8_gemm_config.nr = 4;
      qc8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx2) {
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_3x8c8__avx2);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_3x8c8__avx2);
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c8__avx2);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c8__avx2);
      qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_avx2_params;
      qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qc8_gemm_config.mr = 3;
      qc8_gemm_config.nr = 8;
      qc8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx) {
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_sse4_params;
      qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qc8_gemm_config.mr = 2;
      qc8_gemm_config.nr = 4;
      qc8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_sse4_1) {
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_sse4_params;
      qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qc8_gemm_config.mr = 3;
      qc8_gemm_config.nr = 4;
      qc8_gemm_config.log2_kr = 3;
    } else {
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_sse2_params;
      qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qc8_gemm_config.mr = 3;
      qc8_gemm_config.nr = 4;
      qc8_gemm_config.log2_kr = 3;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->use_wasm_sdot) {
        qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x4c16__wasmsdot);
        qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x4c16__wasmsdot);
        qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot);
        qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c16__wasmsdot);
        qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_wasmsimd_params;
        qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
        qc8_gemm_config.mr = 4;
        qc8_gemm_config.nr = 4;
        qc8_gemm_config.log2_kr = 4;
      } else {
        qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
        qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
        qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
        qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
        qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_wasmsimd_params;
        qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
        qc8_gemm_config.mr = 4;
        qc8_gemm_config.nr = 4;
        qc8_gemm_config.log2_kr = 1;
        qc8_gemm_config.log2_sr = 2;
      }
    #else
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_wasmsimd_params;
      qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qc8_gemm_config.mr = 4;
      qc8_gemm_config.nr = 4;
      qc8_gemm_config.log2_kr = 1;
      qc8_gemm_config.log2_sr = 2;
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_scalar_imagic_params;
      qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qc8_gemm_config.mr = 2;
      qc8_gemm_config.nr = 2;
    } else {
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_scalar_fmagic_params;
      qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qc8_gemm_config.mr = 4;
      qc8_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_RISCV
    qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qc8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qc8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qc8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qc8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qc8_gemm_config.init.qc8 = xnn_init_qc8_conv_minmax_fp32_scalar_lrintf_params;
    qc8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
    qc8_gemm_config.mr = 3;
    qc8_gemm_config.nr = 4;
  #endif
}

static void init_qs8_gemm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            switch (cpuinfo_get_uarch(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55);
                qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55);
                qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
                qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
                qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
                qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
                qs8_gemm_config.mr = 4;
                qs8_gemm_config.nr = 8;
                qs8_gemm_config.log2_kr = 2;
                break;
              default:
                qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_ld64);
                qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_ld64);
                qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
                qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
                qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
                qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
                qs8_gemm_config.mr = 4;
                qs8_gemm_config.nr = 8;
                qs8_gemm_config.log2_kr = 2;
                break;
            }
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          switch (cpuinfo_get_uarch(0)->uarch) {
            case cpuinfo_uarch_cortex_a5:
            case cpuinfo_uarch_cortex_a7:
            case cpuinfo_uarch_krait:
            case cpuinfo_uarch_kryo:
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_gemm_config.mr = 4;
              qs8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a32:
            case cpuinfo_uarch_cortex_a35:
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_gemm_config.mr = 4;
              qs8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a57:
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53);
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_gemm_config.mr = 4;
              qs8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a55r0:
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53);
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_gemm_config.mr = 4;
              qs8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a72:
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
              qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_gemm_config.mr = 2;
              qs8_gemm_config.nr = 8;
              qs8_gemm_config.log2_kr = 1;
              qs8_gemm_config.log2_sr = 2;
              break;
            case cpuinfo_uarch_exynos_m1:
            case cpuinfo_uarch_exynos_m2:
            case cpuinfo_uarch_exynos_m3:
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64);
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_gemm_config.mr = 4;
              qs8_gemm_config.nr = 8;
              break;
            default:
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_gemm_config.mr = 4;
              qs8_gemm_config.nr = 8;
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qs8_gemm_config.mr;
          const uint32_t nr = qs8_gemm_config.nr;
          const uint32_t log2_kr = qs8_gemm_config.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 8 && log2_kr == 2 && hardware_config->use_arm_neon_dot) {
                    qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                    qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                    qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot;
                    qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c4__neondot;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
                break;
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53;
                  qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53;
                  qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7;
                  qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53;
                  qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53;
                  qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7;
                  qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neondot);
            qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neondot);
            qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
            qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
            qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
            qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_gemm_config.mr = 4;
            qs8_gemm_config.nr = 8;
            qs8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qs8_gemm_config.mr = 2;
          qs8_gemm_config.nr = 8;
          qs8_gemm_config.log2_kr = 1;
          qs8_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    } else {
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_armsimd32_params;
      qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_gemm_config.mr = 2;
      qs8_gemm_config.nr = 2;
      qs8_gemm_config.log2_kr = 2;
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
            qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
            qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
            qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_gemm_config.mr = 4;
            qs8_gemm_config.nr = 16;
            qs8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal);
          qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal);
          qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal);
          qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal);
          qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qs8_gemm_config.mr = 2;
          qs8_gemm_config.nr = 8;
          qs8_gemm_config.log2_kr = 3;
        }
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neondot);
            qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
            qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot);
            qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
            qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
            qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_gemm_config.mr = 4;
            qs8_gemm_config.nr = 16;
            qs8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qs8_gemm_config.mr = 2;
          qs8_gemm_config.nr = 8;
          qs8_gemm_config.log2_kr = 1;
          qs8_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            switch (cpuinfo_get_core(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                break;
              case cpuinfo_uarch_cortex_x1:
              case cpuinfo_uarch_cortex_a78:
                qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                break;
              default:
                qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64);
                qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64);
                break;
            }
            qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
            qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
            qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
            qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_gemm_config.mr = 4;
            qs8_gemm_config.nr = 16;
            qs8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          switch (cpuinfo_get_core(0)->uarch) {
            case cpuinfo_uarch_cortex_a35:
            case cpuinfo_uarch_kryo:
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64);
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
              qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_gemm_config.mr = 4;
              qs8_gemm_config.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a55r0:
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53);
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
              qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_gemm_config.mr = 4;
              qs8_gemm_config.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a72:
            case cpuinfo_uarch_cortex_a73:
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_gemm_config.mr = 2;
              qs8_gemm_config.nr = 8;
              qs8_gemm_config.log2_kr = 3;
              break;

            default:
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal);
              qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal);
              qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal);
              qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_gemm_config.mr = 2;
              qs8_gemm_config.nr = 8;
              qs8_gemm_config.log2_kr = 3;
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qs8_gemm_config.mr;
          const uint32_t nr = qs8_gemm_config.nr;
          const uint32_t log2_kr = qs8_gemm_config.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a53:
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 2 && nr == 8 && log2_kr == 3) {
                  qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53;
                  qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53;
                  qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53;
                  qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm_cortex_a53;
                }
                break;

              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 16 && log2_kr == 2 && hardware_config->use_arm_neon_dot) {
                    qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                    qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                    qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot;
                    qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neondot;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neondot);
            qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
            qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot);
            qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
            qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
            qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_gemm_config.mr = 4;
            qs8_gemm_config.nr = 16;
            qs8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qs8_gemm_config.mr = 2;
          qs8_gemm_config.nr = 8;
          qs8_gemm_config.log2_kr = 1;
          qs8_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx512_params;
      qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_gemm_config.mr = 4;
      qs8_gemm_config.nr = 16;
      qs8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_xop) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_gemm_config.mr = 2;
      qs8_gemm_config.nr = 4;
      qs8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx2) {
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2);
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2);
      qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx2_params;
      qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_gemm_config.mr = 3;
      qs8_gemm_config.nr = 8;
      qs8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx) {
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_gemm_config.mr = 2;
      qs8_gemm_config.nr = 4;
      qs8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_gemm_config.mr = 3;
      qs8_gemm_config.nr = 4;
      qs8_gemm_config.log2_kr = 3;
    } else {
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse2_params;
      qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_gemm_config.mr = 3;
      qs8_gemm_config.nr = 4;
      qs8_gemm_config.log2_kr = 3;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->use_wasm_sdot) {
        qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_4x4c16__wasmsdot);
        qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_4x4c16__wasmsdot);
        qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot);
        qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c16__wasmsdot);
        qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_wasmsimd_params;
        qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
        qs8_gemm_config.mr = 4;
        qs8_gemm_config.nr = 4;
        qs8_gemm_config.log2_kr = 4;
      } else {
        qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_wasmsimd_params;
        qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
        qs8_gemm_config.mr = 4;
        qs8_gemm_config.nr = 4;
        qs8_gemm_config.log2_kr = 1;
        qs8_gemm_config.log2_sr = 2;
      }
    #else
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_wasmsimd_params;
      qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_gemm_config.mr = 4;
      qs8_gemm_config.nr = 4;
      qs8_gemm_config.log2_kr = 1;
      qs8_gemm_config.log2_sr = 2;
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params;
      qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_gemm_config.mr = 2;
      qs8_gemm_config.nr = 2;
    } else {
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params;
      qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_gemm_config.mr = 4;
      qs8_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_RISCV
    qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qs8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qs8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qs8_gemm_config.init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params;
    qs8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
    qs8_gemm_config.mr = 3;
    qs8_gemm_config.nr = 4;
  #endif
}

static void init_qu8_gemm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__neondot);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__neondot);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 8;
            qu8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          switch (cpuinfo_get_uarch(0)->uarch) {
            case cpuinfo_uarch_cortex_a5:
            case cpuinfo_uarch_cortex_a7:
            case cpuinfo_uarch_krait:
            case cpuinfo_uarch_kryo:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a32:
            case cpuinfo_uarch_cortex_a35:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a57:
            case cpuinfo_uarch_cortex_a72:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a55r0:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_exynos_m1:
            case cpuinfo_uarch_exynos_m2:
            case cpuinfo_uarch_exynos_m3:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_ld64);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 8;
              break;
            default:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 8;
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qu8_gemm_config.mr;
          const uint32_t nr = qu8_gemm_config.nr;
          const uint32_t log2_kr = qu8_gemm_config.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a53;
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_prfm_cortex_a7;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53;
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__neondot);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__neondot);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 8;
            qu8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane);
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
          qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
          qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
          qu8_gemm_config.mr = 3;
          qu8_gemm_config.nr = 8;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    } else {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_armsimd32_params;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 2;
      qu8_gemm_config.nr = 2;
      qu8_gemm_config.log2_kr = 2;
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_ASSEMBLY
      if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
        #if XNN_ENABLE_ARM_DOTPROD
          switch (cpuinfo_get_core(0)->uarch) {
            case cpuinfo_uarch_cortex_a55:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 16;
              qu8_gemm_config.log2_kr = 2;
              break;
            default:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 16;
              qu8_gemm_config.log2_kr = 2;
              break;
          }
        #endif  // XNN_ENABLE_ARM_DOTPROD
      } else {
        switch (cpuinfo_get_core(0)->uarch) {
          case cpuinfo_uarch_cortex_a53:
          case cpuinfo_uarch_cortex_a55r0:
          case cpuinfo_uarch_kryo:
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_prfm_cortex_a53);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_prfm_cortex_a53);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 16;
            break;

          case cpuinfo_uarch_cortex_a57:
          case cpuinfo_uarch_cortex_a72:
          case cpuinfo_uarch_cortex_a73:
          case cpuinfo_uarch_cortex_a75:
          case cpuinfo_uarch_cortex_a76:
          case cpuinfo_uarch_exynos_m1:
          case cpuinfo_uarch_exynos_m2:
          case cpuinfo_uarch_exynos_m3:
          case cpuinfo_uarch_exynos_m4:
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_prfm_cortex_a75);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_prfm_cortex_a75);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 16;
            break;

          default:
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 16;
            break;
        }
      }
      #if XNN_MAX_UARCH_TYPES > 1
      {
        /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
        const uint32_t mr = qu8_gemm_config.mr;
        const uint32_t nr = qu8_gemm_config.nr;
        const uint32_t log2_kr = qu8_gemm_config.log2_kr;
        for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
          const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
          if (uarch_info == NULL) {
            /* No more microarchitectures in the system */
            break;
          }

          switch (uarch_info->uarch) {
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a55r0:
              if (mr == 4 && nr == 16 && log2_kr == 0) {
                qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_prfm_cortex_a53;
                qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_prfm_cortex_a53;
              }
              break;

            case cpuinfo_uarch_cortex_a55:
              #if XNN_ENABLE_ARM_DOTPROD
                if (mr == 4 && nr == 16 && log2_kr == 2 && hardware_config->use_arm_neon_dot) {
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                }
              #endif  // XNN_ENABLE_ARM_DOTPROD
              break;
            default:
              break;
          }
        }
      }
      #endif  // XNN_MAX_UARCH_TYPES > 1
    #else  // !XNN_ENABLE_ASSEMBLY
      if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
        #if XNN_ENABLE_ARM_DOTPROD
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__neondot);
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
          qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
          qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
          qu8_gemm_config.mr = 4;
          qu8_gemm_config.nr = 16;
          qu8_gemm_config.log2_kr = 2;
        #endif  // XNN_ENABLE_ARM_DOTPROD
      } else {
        qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane);
        qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane);
        qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
        qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
        qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
        qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
        qu8_gemm_config.mr = 4;
        qu8_gemm_config.nr = 16;
      }
    #endif  // XNN_ENABLE_ASSEMBLY
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx512_params;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 4;
      qu8_gemm_config.nr = 16;
      qu8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_xop) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 2;
      qu8_gemm_config.nr = 4;
      qu8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx2) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x8c8__avx2);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x8c8__avx2);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx2_params;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 3;
      qu8_gemm_config.nr = 8;
      qu8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 2;
      qu8_gemm_config.nr = 4;
      qu8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 3;
      qu8_gemm_config.nr = 4;
      qu8_gemm_config.log2_kr = 3;
    } else {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 3;
      qu8_gemm_config.nr = 4;
      qu8_gemm_config.log2_kr = 3;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_wasmsimd_params;
    qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
    qu8_gemm_config.mr = 4;
    qu8_gemm_config.nr = 4;
    qu8_gemm_config.log2_kr = 1;
    qu8_gemm_config.log2_sr = 2;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 2;
      qu8_gemm_config.nr = 2;
    } else {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 4;
      qu8_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_RISCV
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params;
    qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
    qu8_gemm_config.mr = 3;
    qu8_gemm_config.nr = 4;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_gemm_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_gemm_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_gemm2_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_gemm_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qc8_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qc8_gemm_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qs8_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qs8_gemm_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qu8_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qu8_gemm_config();
    return TRUE;
  }
#endif

struct xnn_gemm_config* xnn_init_f16_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_gemm, &init_f16_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_gemm, &init_f16_gemm_config);
  #endif
  return &f16_gemm_config;
}

struct xnn_gemm_config* xnn_init_f32_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_gemm, &init_f32_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_gemm, &init_f32_gemm_config);
  #endif
  return &f32_gemm_config;
}

struct xnn_gemm_config* xnn_init_f32_gemm2_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_gemm2, &init_f32_gemm2_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_gemm2, &init_f32_gemm2_config);
  #endif
  return &f32_gemm2_config;
}

struct xnn_gemm_config* xnn_init_qc8_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qc8_gemm, &init_qc8_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qc8_gemm, &init_qc8_gemm_config);
  #endif
  return &qc8_gemm_config;
}

struct xnn_gemm_config* xnn_init_qs8_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qs8_gemm, &init_qs8_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qs8_gemm, &init_qs8_gemm_config);
  #endif
  return &qs8_gemm_config;
}

struct xnn_gemm_config* xnn_init_qu8_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qu8_gemm, &init_qu8_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qu8_gemm, &init_qu8_gemm_config);
  #endif
  return &qu8_gemm_config;
}
