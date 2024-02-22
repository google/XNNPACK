// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#if _WIN32
  #include <windows.h>
#else
  #include <pthread.h>
#endif

#if XNN_ENABLE_CPUINFO
  #include <cpuinfo.h>
#endif  // XNN_ENABLE_CPUINFO

#include <experiments-config.h>
#include <xnnpack/common.h>
#include <xnnpack/config.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/pack.h>
#include <xnnpack/packw.h>

#if XNN_ARCH_WASMSIMD
  #include <emscripten.h>
#endif

#define XNN_MR_TO_INDEX(MR) (MR-1)


static struct xnn_gemm_config f16_gemm_config = {0};
static struct xnn_gemm_config f32_gemm_config = {0};
static struct xnn_gemm_config f32_gemm_nr2_config = {0};
static struct xnn_gemm_config f32_qc4w_gemm_config = {0};
static struct xnn_gemm_config f32_qc8w_gemm_config = {0};
static struct xnn_gemm_config qd8_f16_qc8w_gemm_config = {0};
static struct xnn_gemm_config qd8_f16_qc4w_gemm_config = {0};
static struct xnn_gemm_config qd8_f32_qc4w_gemm_config = {0};
static struct xnn_gemm_config qd8_f32_qc8w_gemm_config = {0};
static struct xnn_gemm_config qs8_qc8w_gemm_config = {0};
static struct xnn_gemm_config qu8_gemm_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_gemm_nr2 = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_qc4w_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_qc8w_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qd8_f16_qc4w_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qd8_f16_qc8w_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qd8_f32_qc4w_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qd8_f32_qc8w_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qs8_qc8w_gemm = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qu8_gemm = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_gemm_nr2 = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_qc4w_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_qc8w_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qd8_f16_qc4w_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qd8_f16_qc8w_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qd8_f32_qc4w_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qd8_f32_qc8w_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qs8_qc8w_gemm = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qu8_gemm = PTHREAD_ONCE_INIT;
#endif

static void init_f16_gemm_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
      f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm;
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
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55);
              f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
              f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
              f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
              f16_gemm_config.mr = 6;
              f16_gemm_config.nr = 16;
              #if XNN_ENABLE_JIT
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55);
              #endif
              break;
            case cpuinfo_uarch_cortex_a55r0:
            case cpuinfo_uarch_cortex_a75:
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0);
              f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
              f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
              f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
              f16_gemm_config.mr = 6;
              f16_gemm_config.nr = 16;
              #if XNN_ENABLE_JIT
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0);
              #endif
              break;
            case cpuinfo_uarch_exynos_m5:
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
              f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
              f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
              f16_gemm_config.mr = 4;
              f16_gemm_config.nr = 16;
              #if XNN_ENABLE_JIT
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_4x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64);
              #endif
              break;
            case cpuinfo_uarch_exynos_m4:
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
              f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
              f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
              f16_gemm_config.mr = 6;
              f16_gemm_config.nr = 16;
              #if XNN_ENABLE_JIT
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64);
              #endif
              break;
            default:
            case cpuinfo_uarch_cortex_a76:
            case cpuinfo_uarch_cortex_a77:
            case cpuinfo_uarch_cortex_a78:
            case cpuinfo_uarch_cortex_a510:
            case cpuinfo_uarch_cortex_a710:
            case cpuinfo_uarch_cortex_a715:
            case cpuinfo_uarch_cortex_x1:
            case cpuinfo_uarch_cortex_x2:
            case cpuinfo_uarch_cortex_x3:
            case cpuinfo_uarch_neoverse_n1:
            case cpuinfo_uarch_neoverse_n2:
            case cpuinfo_uarch_neoverse_v1:
            case cpuinfo_uarch_neoverse_v2:
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
              f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75);
              f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
              f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
              f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
              f16_gemm_config.mr = 6;
              f16_gemm_config.nr = 16;
              #if XNN_ENABLE_JIT
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
                f16_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75);
                f16_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64);
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
          f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64);
          f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64);
          f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64);
          f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64);
          f16_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
          f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
          f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
          f16_gemm_config.mr = 6;
          f16_gemm_config.nr = 16;
        #endif  // XNN_ENABLE_ASSEMBLY
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__avx2_broadcast);
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_4x16__avx2_broadcast);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast);
      f16_gemm_config.init.f16 = xnn_init_f16_minmax_avx_params;
      f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
      f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f16_gemm_goi_w;
      f16_gemm_config.mr = 4;
      f16_gemm_config.nr = 16;
    }
  #endif
}

#if XNN_ARCH_WASMSIMD
  EM_JS(int, hardware_concurrency, (void), {
    return self.navigator.hardwareConcurrency;
  });
  // A cpu with more than `kCoreCountThresholdForAdaptiveAvxOptimization` is
  // assumed to support AVX instructions.
  const int kCoreCountThresholdForAdaptiveAvxOptimization = 4;
#endif

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
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a53:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a55r0:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a32:
          case cpuinfo_uarch_cortex_a35:
          case cpuinfo_uarch_cortex_a55:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;

          case cpuinfo_uarch_cortex_a57:
          case cpuinfo_uarch_cortex_a72:
          case cpuinfo_uarch_cortex_a73:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;

          default:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
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
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm;
                  #if XNN_ENABLE_JIT
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm;
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm;
                  #endif
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53;
                  #if XNN_ENABLE_JIT
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53;
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53;
                  #endif
                }
                break;
              case cpuinfo_uarch_cortex_a55:
                if (mr == 4 && nr == 8) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55;
                  #if XNN_ENABLE_JIT
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53;
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a55;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53;
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
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld128);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
        f32_gemm_config.mr = 4;
        f32_gemm_config.nr = 8;
      #endif  // XNN_ENABLE_ASSEMBLY
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x4__scalar);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__scalar);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x4__scalar);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__scalar);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x4__scalar);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__scalar);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x4__scalar);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__scalar);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x4__scalar);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_ARM64
    #if XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_cortex_a72:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
          f32_gemm_config.mr = 4;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a57:
        case cpuinfo_uarch_cortex_a75:
        case cpuinfo_uarch_cortex_a76:
        case cpuinfo_uarch_exynos_m3:
        case cpuinfo_uarch_exynos_m4:
        case cpuinfo_uarch_neoverse_n1:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm);
          #endif
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          #if XNN_ENABLE_JIT
            f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm);
            #if XNN_ENABLE_GEMM_M_SPECIALIZATION
              f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm);
              f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm);
            #endif
          #endif
          break;
        case cpuinfo_uarch_exynos_m1:
        case cpuinfo_uarch_exynos_m2:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8s4__neonfma);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8s4__neonfma);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8s4__neonfma);
          #endif
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4_prfm;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          f32_gemm_config.log2_sr = 2;
          break;
        case cpuinfo_uarch_cortex_a53:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm);
          #endif
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a55r0:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53);
          #endif
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a35:
        case cpuinfo_uarch_cortex_a55:
        case cpuinfo_uarch_kryo:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55);
          #endif
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a73:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a77:
        case cpuinfo_uarch_exynos_m5:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
          f32_gemm_config.mr = 4;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_x3:
        case cpuinfo_uarch_neoverse_v2:
          // TODO(fbarchard): Implement asm with indexed inputs
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          break;
        case cpuinfo_uarch_cortex_a78:
        case cpuinfo_uarch_cortex_a510:
        case cpuinfo_uarch_cortex_a710:
        case cpuinfo_uarch_cortex_a715:
        case cpuinfo_uarch_cortex_x1:
        case cpuinfo_uarch_cortex_x2:
        case cpuinfo_uarch_neoverse_n2:
        case cpuinfo_uarch_neoverse_v1:
        default:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
          #if XNN_ENABLE_GEMM_M_SPECIALIZATION
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128);
          #endif
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
          f32_gemm_config.mr = 6;
          f32_gemm_config.nr = 8;
          #if XNN_ENABLE_JIT
            f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_ld64);
            f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128);
            // TODO: f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_ld64);
            f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128);
            #if XNN_ENABLE_GEMM_M_SPECIALIZATION
              f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_ld128);
              f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128);
            #endif
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
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm;
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm;
                #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm;
                #endif
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm;
                  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm;
                  #endif
                #endif
              } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm;
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm;
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm;
                #endif
              }
              break;
            case cpuinfo_uarch_cortex_a55r0:
              if (mr == 6 && nr == 8 && log2_sr == 0) {
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53;
                #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                #endif
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53;
                  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                    f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53;
                    f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53;
                  #endif
                #endif
              } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53;
                #endif
              }
              break;
            case cpuinfo_uarch_cortex_a55:
              if (mr == 6 && nr == 8 && log2_sr == 0) {
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55;
                #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                #endif
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a55;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55;
                #endif
              } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                #if XNN_ENABLE_JIT
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_jit_gemm_code_generator_fn) xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a55;
                  f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_jit_igemm_code_generator_fn) xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53;
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
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
        f32_gemm_config.mr = 6;
        f32_gemm_config.nr = 8;
      #else  // !XNN_ENABLE_ASSEMBLY
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
        f32_gemm_config.mr = 6;
        f32_gemm_config.nr = 8;
       #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_7x16__avx512f_broadcast);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4_prfm;
      f32_gemm_config.mr = 7;
      f32_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_fma3) {
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_zen:
        case cpuinfo_uarch_dhyana:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x16s4__fma3_broadcast);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_avx_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4;
          f32_gemm_config.mr = 4;
          f32_gemm_config.nr = 16;
          f32_gemm_config.log2_sr = 2;
          break;
        default:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast_prfm);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_avx_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4;
          f32_gemm_config.mr = 5;
          f32_gemm_config.nr = 16;
          break;
      }
    } else if (hardware_config->use_x86_avx) {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x16__avx_broadcast);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_5x16__avx_broadcast);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_avx_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4;
      f32_gemm_config.mr = 5;
      f32_gemm_config.nr = 16;
    } else {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__sse_load1);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__sse_load1);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__sse_load1);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__sse_load1);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_sse_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      #if XNN_ARCH_WASMRELAXEDSIMD
        #if XNN_ENABLE_JIT
          f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1);
          f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1);
        #endif
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
      #else
        #if XNN_ENABLE_JIT
          f32_gemm_config.generator.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1);
          f32_gemm_config.generator.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1);
        #endif

        if (xnn_get_experiment_config()->adaptive_avx_optimization && hardware_concurrency() > kCoreCountThresholdForAdaptiveAvxOptimization) {
          f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat);
          f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x8__wasmsimd_loadsplat);
          f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat);
          f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x8__wasmsimd_loadsplat);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat);
          f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat);
          f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_loadsplat);
          f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat);
          f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_loadsplat);
        } else {
          f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmsimd_splat);
          f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x8__wasmsimd_splat);
          f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmsimd_splat);
          f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x8__wasmsimd_splat);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_splat);
          f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat);
          f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_splat);
          f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat);
          f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_splat);
        }
      #endif

      f32_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 8;
    } else {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4;
        f32_gemm_config.mr = 6;
        f32_gemm_config.nr = 8;
      #else
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4;
        f32_gemm_config.mr = 5;
        f32_gemm_config.nr = 8;
      #endif
    }
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_2x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_2x4__scalar);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_2x4__scalar);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_2x4__scalar);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_2x4__scalar);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_2x4__scalar);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4;
      f32_gemm_config.mr = 2;
      f32_gemm_config.nr = 4;
    } else {
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x4__scalar);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x4__wasm);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x4__wasm);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x4__wasm);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x4__wasm);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_RISCV
    f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
    f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x4__scalar);
    f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
    f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x4__scalar);
    f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__scalar);
    f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x4__scalar);
    f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__scalar);
    f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x4__scalar);
    f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__scalar);
    f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x4__scalar);
    f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__scalar);
    f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x4__scalar);
    f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
    f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4;
    f32_gemm_config.mr = 4;
    f32_gemm_config.nr = 4;
  #elif XNN_ARCH_PPC64
    f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
    f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x4__scalar);
    f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
    f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x4__scalar);
    f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__scalar);
    f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x4__scalar);
    f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__scalar);
    f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x4__scalar);
    f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__scalar);
    f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x4__scalar);
    f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__scalar);
    f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x4__scalar);
    f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
    f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4;
    f32_gemm_config.mr = 4;
    f32_gemm_config.nr = 4;
  #endif
}

static void init_f32_gemm_nr2_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64);
      f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64);
      f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm;
      f32_gemm_nr2_config.mr = 4;
      f32_gemm_nr2_config.nr = 2;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2__scalar);
      f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2__scalar);
      f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__scalar);
      f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__scalar);
      f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4;
      f32_gemm_nr2_config.mr = 4;
      f32_gemm_nr2_config.nr = 2;
    }
  #elif XNN_ARCH_ARM64
    #if XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128);
      f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm);
      f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm;
      f32_gemm_nr2_config.mr = 4;
      f32_gemm_nr2_config.nr = 2;
    #else  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm);
        f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm;
        f32_gemm_nr2_config.mr = 4;
        f32_gemm_nr2_config.nr = 2;
      #else  // !XNN_ENABLE_ASSEMBLY
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64);
        f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm;
        f32_gemm_nr2_config.mr = 4;
        f32_gemm_nr2_config.nr = 2;
       #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__sse);
    f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__sse);
    f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_sse_params;
    f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
    f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4;
    f32_gemm_nr2_config.mr = 4;
    f32_gemm_nr2_config.nr = 2;
    f32_gemm_nr2_config.log2_kr = 2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
      #else
        f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmsimd);
        f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmsimd);
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_x86);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_x86);
      #endif

      f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4;
      f32_gemm_nr2_config.mr = 4;
      f32_gemm_nr2_config.nr = 2;
      f32_gemm_nr2_config.log2_kr = 2;
    } else {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
      #else
        f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmsimd);
        f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmsimd);
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_arm);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_arm);
      #endif

      f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4;
      f32_gemm_nr2_config.mr = 4;
      f32_gemm_nr2_config.nr = 2;
      f32_gemm_nr2_config.log2_kr = 2;
    }
  #elif XNN_ARCH_WASM
    f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2__scalar);
    f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2__scalar);
    f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__wasm);
    f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__wasm);
    f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
    f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4;
    f32_gemm_nr2_config.mr = 4;
    f32_gemm_nr2_config.nr = 2;
  #elif XNN_ARCH_RISCV
    f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2__scalar);
    f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2__scalar);
    f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__scalar);
    f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__scalar);
    f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
    f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4;
    f32_gemm_nr2_config.mr = 4;
    f32_gemm_nr2_config.nr = 2;
  #elif XNN_ARCH_PPC64
    f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2__scalar);
    f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2__scalar);
    f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__scalar);
    f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__scalar);
    f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
    f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4;
    f32_gemm_nr2_config.mr = 4;
    f32_gemm_nr2_config.nr = 2;
  #endif
}

static void init_f32_qc4w_gemm_config(void) {
    f32_qc4w_gemm_config.planes = 1;
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x8__neon_lane_ld64);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x8__neon_lane_ld64);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 4;
      f32_qc4w_gemm_config.nr = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x4__scalar);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x4__scalar);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 4;
      f32_qc4w_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_ARM64
    #if XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      switch (cpuinfo_get_core(0)->uarch) {
        // TODO(fbarchard): fill in microkernels.
        case cpuinfo_uarch_cortex_a72:
        case cpuinfo_uarch_cortex_a57:
        case cpuinfo_uarch_cortex_a75:
        case cpuinfo_uarch_cortex_a76:
        case cpuinfo_uarch_exynos_m3:
        case cpuinfo_uarch_exynos_m4:
        case cpuinfo_uarch_exynos_m1:
        case cpuinfo_uarch_exynos_m2:
        case cpuinfo_uarch_cortex_a53:
        case cpuinfo_uarch_cortex_a55r0:
        case cpuinfo_uarch_cortex_a35:
        case cpuinfo_uarch_cortex_a55:
        case cpuinfo_uarch_kryo:
        case cpuinfo_uarch_cortex_a73:
        case cpuinfo_uarch_cortex_a77:
        case cpuinfo_uarch_exynos_m5:
        case cpuinfo_uarch_cortex_a78:
        case cpuinfo_uarch_cortex_a510:
        case cpuinfo_uarch_cortex_a710:
        case cpuinfo_uarch_cortex_a715:
        case cpuinfo_uarch_cortex_x1:
        case cpuinfo_uarch_cortex_x2:
        case cpuinfo_uarch_cortex_x3:
        case cpuinfo_uarch_neoverse_n1:
        case cpuinfo_uarch_neoverse_n2:
        case cpuinfo_uarch_neoverse_v1:
        case cpuinfo_uarch_neoverse_v2:
        default:
          f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128);
          f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128);
          f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
          f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
          f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
          f32_qc4w_gemm_config.mr = 6;
          f32_qc4w_gemm_config.nr = 8;
      }
      #if XNN_MAX_UARCH_TYPES > 1
        /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
        const uint32_t mr = f32_gemm_config.mr;
        const uint32_t nr = f32_gemm_config.nr;
        const uint32_t log2_sr = f32_gemm_config.log2_sr;
        // TODO(fbarchard): fill in with microkernels.
        (void) mr;
        (void) nr;
        (void) log2_sr;
        for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
          const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
          if (uarch_info == NULL) {
            /* No more microarchitectures in the system */
            break;
          }
          switch (uarch_info->uarch) {
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a55r0:
            case cpuinfo_uarch_cortex_a55:
            default:
              break;
          }
        }
      #endif  // XNN_MAX_UARCH_TYPES > 1
    #else  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128);
        f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128);
        f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
        f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
        f32_qc4w_gemm_config.mr = 6;
        f32_qc4w_gemm_config.nr = 8;
      #else  // !XNN_ENABLE_ASSEMBLY
        f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128);
        f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128);
        f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
        f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
        f32_qc4w_gemm_config.mr = 6;
        f32_qc4w_gemm_config.nr = 8;
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x32__avx512skx_broadcast);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_7x32__avx512skx_broadcast);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_avx512_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 7;
      f32_qc4w_gemm_config.nr = 32;
    } else if (hardware_config->use_x86_avx2) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x16__avx2_broadcast);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_3x16__avx2_broadcast);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_avx_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 3;
      f32_qc4w_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_fma3) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x16__fma3_broadcast);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_3x16__fma3_broadcast);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_avx_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 3;
      f32_qc4w_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x16__avx_broadcast);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_3x16__avx_broadcast);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_avx_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 3;
      f32_qc4w_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x8__sse41_dup);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x8__sse41_dup);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_sse_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 4;
      f32_qc4w_gemm_config.nr = 8;
    } else {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x8__sse2_dup);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x8__sse2_dup);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_sse_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 4;
      f32_qc4w_gemm_config.nr = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x4__wasm);
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x4__wasm);
    f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
    f32_qc4w_gemm_config.mr = 4;
    f32_qc4w_gemm_config.nr = 4;
  #elif XNN_ARCH_WASM
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x4__wasm);
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x4__wasm);
    f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
    f32_qc4w_gemm_config.mr = 4;
    f32_qc4w_gemm_config.nr = 4;
  #elif XNN_ARCH_RISCV
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x4__scalar);
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x4__scalar);
    f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
    f32_qc4w_gemm_config.mr = 4;
    f32_qc4w_gemm_config.nr = 4;
  #elif XNN_ARCH_PPC64
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x4__scalar);
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x4__scalar);
    f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
    f32_qc4w_gemm_config.mr = 4;
    f32_qc4w_gemm_config.nr = 4;
  #endif
}

static void init_f32_qc8w_gemm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2;
      f32_qc8w_gemm_config.mr = 4;
      f32_qc8w_gemm_config.nr = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2;
      f32_qc8w_gemm_config.mr = 4;
      f32_qc8w_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_ARM64
    #if XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      switch (cpuinfo_get_core(0)->uarch) {
        // TODO(fbarchard): fill in microkernels.
        case cpuinfo_uarch_cortex_a72:
        case cpuinfo_uarch_cortex_a57:
        case cpuinfo_uarch_cortex_a75:
        case cpuinfo_uarch_cortex_a76:
        case cpuinfo_uarch_exynos_m3:
        case cpuinfo_uarch_exynos_m4:
        case cpuinfo_uarch_exynos_m1:
        case cpuinfo_uarch_exynos_m2:
        case cpuinfo_uarch_cortex_a53:
        case cpuinfo_uarch_cortex_a55r0:
        case cpuinfo_uarch_cortex_a35:
        case cpuinfo_uarch_cortex_a55:
        case cpuinfo_uarch_kryo:
        case cpuinfo_uarch_cortex_a73:
        case cpuinfo_uarch_cortex_a77:
        case cpuinfo_uarch_exynos_m5:
        case cpuinfo_uarch_cortex_a78:
        case cpuinfo_uarch_cortex_x1:
        case cpuinfo_uarch_neoverse_n1:
        case cpuinfo_uarch_neoverse_v1:
        default:
          f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4);
          f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
          f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
          f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2;
          f32_qc8w_gemm_config.mr = 6;
          f32_qc8w_gemm_config.nr = 8;
      }
      #if XNN_MAX_UARCH_TYPES > 1
        /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
        const uint32_t mr = f32_gemm_config.mr;
        const uint32_t nr = f32_gemm_config.nr;
        const uint32_t log2_sr = f32_gemm_config.log2_sr;
        // TODO(fbarchard): fill in with microkernels.
        (void) mr;
        (void) nr;
        (void) log2_sr;
        for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
          const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
          if (uarch_info == NULL) {
            /* No more microarchitectures in the system */
            break;
          }
          switch (uarch_info->uarch) {
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a55r0:
            case cpuinfo_uarch_cortex_a55:
            default:
              break;
          }
        }
      #endif  // XNN_MAX_UARCH_TYPES > 1
    #else  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
        f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
        f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2;
        f32_qc8w_gemm_config.mr = 6;
        f32_qc8w_gemm_config.nr = 8;
      #else  // !XNN_ENABLE_ASSEMBLY
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64);
        f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
        f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2;
        f32_qc8w_gemm_config.mr = 6;
        f32_qc8w_gemm_config.nr = 8;
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x32__avx512skx_broadcast);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_7x32__avx512skx_broadcast);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2;
      f32_qc8w_gemm_config.mr = 7;
      f32_qc8w_gemm_config.nr = 32;
    } else if (hardware_config->use_x86_avx2) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx2_broadcast);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx2_broadcast);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_avx_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2;
      f32_qc8w_gemm_config.mr = 5;
      f32_qc8w_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_fma3) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x16__fma3_broadcast);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_5x16__fma3_broadcast);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_avx_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2;
      f32_qc8w_gemm_config.mr = 5;
      f32_qc8w_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx_broadcast);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx_broadcast);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_avx_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2;
      f32_qc8w_gemm_config.mr = 5;
      f32_qc8w_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse41_dup);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse41_dup);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_sse_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2;
      f32_qc8w_gemm_config.mr = 4;
      f32_qc8w_gemm_config.nr = 8;
    } else {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse2_dup);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse2_dup);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_sse_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2;
      f32_qc8w_gemm_config.mr = 4;
      f32_qc8w_gemm_config.nr = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      #if XNN_ARCH_WASMRELAXEDSIMD
        #if XNN_ENABLE_JIT
          // TODO(b/302273500) Implement WASM JIT relaxed simd qc8w kernels
        #endif
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
      #else
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_1x8__wasmsimd_splat);
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_4x8__wasmsimd_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmsimd_splat);
      #endif

      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2;
      f32_qc8w_gemm_config.mr = 4;
      f32_qc8w_gemm_config.nr = 8;
    } else {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
        f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2;
        f32_qc8w_gemm_config.mr = 6;
        f32_qc8w_gemm_config.nr = 8;
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_1x8__wasmsimd_splat);
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_5x8__wasmsimd_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_5x8__wasmsimd_splat);
      #else
        f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
        f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2;
        f32_qc8w_gemm_config.mr = 5;
        f32_qc8w_gemm_config.nr = 8;
      #endif
    }
  #elif XNN_ARCH_WASM
    f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm);
    f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm);
    f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
    f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2;
    f32_qc8w_gemm_config.mr = 4;
    f32_qc8w_gemm_config.nr = 4;
  #elif XNN_ARCH_RISCV
    f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar);
    f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar);
    f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
    f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2;
    f32_qc8w_gemm_config.mr = 4;
    f32_qc8w_gemm_config.nr = 4;
  #elif XNN_ARCH_PPC64
    f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar);
    f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar);
    f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
    f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2;
    f32_qc8w_gemm_config.mr = 4;
    f32_qc8w_gemm_config.nr = 4;
  #endif
}

static void init_qd8_f16_qc4w_gemm_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
        #if XNN_ENABLE_ARM_I8MM
          qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm);
          qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm);
          qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
          qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
          qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
          qd8_f16_qc4w_gemm_config.mr = 6;
          qd8_f16_qc4w_gemm_config.nr = 8;
          qd8_f16_qc4w_gemm_config.log2_kr = 3;
          qd8_f16_qc4w_gemm_config.planes = 2;
        #endif  // XNN_ENABLE_ARM_I8MM
      } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
        #if XNN_ENABLE_ARM_DOTPROD
          qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
          qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith);
          qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
          qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
          qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
          qd8_f16_qc4w_gemm_config.mr = 4;
          qd8_f16_qc4w_gemm_config.nr = 16;
          qd8_f16_qc4w_gemm_config.log2_kr = 2;
          qd8_f16_qc4w_gemm_config.planes = 2;
        #endif  // XNN_ENABLE_ARM_DOTPROD
      } else {
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane);
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane);
        qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
        qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
        qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
        qd8_f16_qc4w_gemm_config.mr = 6;
        qd8_f16_qc4w_gemm_config.nr = 16;
        qd8_f16_qc4w_gemm_config.planes = 2;
      }
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
      #if XNN_ENABLE_ARM_I8MM
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm);
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm);
        qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
        qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
        qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
        qd8_f16_qc4w_gemm_config.mr = 4;
        qd8_f16_qc4w_gemm_config.nr = 16;
        qd8_f16_qc4w_gemm_config.log2_kr = 3;
        qd8_f16_qc4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_I8MM
    } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
      #if XNN_ENABLE_ARM_DOTPROD
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith);
        qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
        qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
        qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
        qd8_f16_qc4w_gemm_config.mr = 4;
        qd8_f16_qc4w_gemm_config.nr = 16;
        qd8_f16_qc4w_gemm_config.log2_kr = 2;
        qd8_f16_qc4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_DOTPROD
    } else {
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane);
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane);
        qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
        qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
        qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
        qd8_f16_qc4w_gemm_config.mr = 6;
        qd8_f16_qc4w_gemm_config.nr = 16;
        qd8_f16_qc4w_gemm_config.planes = 2;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnnigfni) {
      qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm);
      qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni_prfm);
      qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_avxvnni_params;
      qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f16_qc4w_gemm_config.mr = 7;
      qd8_f16_qc4w_gemm_config.nr = 8;
      qd8_f16_qc4w_gemm_config.log2_kr = 3;
      qd8_f16_qc4w_gemm_config.planes = 2;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
      qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnni_prfm);
      qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnni_prfm);
      qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_avxvnni_params;
      qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f16_qc4w_gemm_config.mr = 7;
      qd8_f16_qc4w_gemm_config.nr = 8;
      qd8_f16_qc4w_gemm_config.log2_kr = 3;
      qd8_f16_qc4w_gemm_config.planes = 2;
    #if XNN_ENABLE_AVXVNNI
      } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnni) {
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_avxvnni_params;
        qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
        qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
        qd8_f16_qc4w_gemm_config.mr = 5;
        qd8_f16_qc4w_gemm_config.nr = 8;
        qd8_f16_qc4w_gemm_config.log2_kr = 3;
        qd8_f16_qc4w_gemm_config.planes = 2;
    #endif
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512skx);
      qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512skx);
      qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_avx_params;
      qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f16_qc4w_gemm_config.mr = 5;
      qd8_f16_qc4w_gemm_config.nr = 8;
      qd8_f16_qc4w_gemm_config.log2_kr = 3;
      qd8_f16_qc4w_gemm_config.planes = 2;
    } else if (hardware_config->use_x86_avx2) {
      qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2);
      qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2);
      qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_avx_params;
      qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f16_qc4w_gemm_config.mr = 3;
      qd8_f16_qc4w_gemm_config.nr = 8;
      qd8_f16_qc4w_gemm_config.log2_kr = 3;
      qd8_f16_qc4w_gemm_config.planes = 2;
    }
  #endif
}

static void init_qd8_f32_qc4w_gemm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
        #if XNN_ENABLE_ARM_I8MM
          qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm);
          qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm);
          qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
          qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
          qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
          qd8_f32_qc4w_gemm_config.mr = 6;
          qd8_f32_qc4w_gemm_config.nr = 8;
          qd8_f32_qc4w_gemm_config.log2_kr = 3;
          qd8_f32_qc4w_gemm_config.planes = 2;
        #endif  // XNN_ENABLE_ARM_I8MM
      } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
        #if XNN_ENABLE_ARM_DOTPROD
          qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot);
          qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot);
          qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
          qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
          qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
          qd8_f32_qc4w_gemm_config.mr = 4;
          qd8_f32_qc4w_gemm_config.nr = 16;
          qd8_f32_qc4w_gemm_config.log2_kr = 2;
          qd8_f32_qc4w_gemm_config.planes = 2;
        #endif  // XNN_ENABLE_ARM_DOTPROD
      } else {
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane);
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane);
        qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
        qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
        qd8_f32_qc4w_gemm_config.mr = 6;
        qd8_f32_qc4w_gemm_config.nr = 16;
        qd8_f32_qc4w_gemm_config.planes = 2;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 4;
      qd8_f32_qc4w_gemm_config.nr = 4;
      qd8_f32_qc4w_gemm_config.planes = 2;
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
      #if XNN_ENABLE_ARM_I8MM
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm);
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm);
        qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
        qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
        qd8_f32_qc4w_gemm_config.mr = 4;
        qd8_f32_qc4w_gemm_config.nr = 16;
        qd8_f32_qc4w_gemm_config.log2_kr = 3;
        qd8_f32_qc4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_I8MM
    } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
      #if XNN_ENABLE_ARM_DOTPROD
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot);
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot);
        qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
        qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
        qd8_f32_qc4w_gemm_config.mr = 4;
        qd8_f32_qc4w_gemm_config.nr = 16;
        qd8_f32_qc4w_gemm_config.log2_kr = 2;
        qd8_f32_qc4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_DOTPROD
    } else {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 6;
      qd8_f32_qc4w_gemm_config.nr = 16;
      qd8_f32_qc4w_gemm_config.planes = 2;
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnnigfni) {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni_prfm);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_avx512vnni_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 7;
      qd8_f32_qc4w_gemm_config.nr = 16;
      qd8_f32_qc4w_gemm_config.log2_kr = 3;
      qd8_f32_qc4w_gemm_config.planes = 2;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_avx512vnni_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 7;
      qd8_f32_qc4w_gemm_config.nr = 16;
      qd8_f32_qc4w_gemm_config.log2_kr = 3;
      qd8_f32_qc4w_gemm_config.planes = 2;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnnigfni) {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni_prfm);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_avx512vnni_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 7;
      qd8_f32_qc4w_gemm_config.nr = 8;
      qd8_f32_qc4w_gemm_config.log2_kr = 3;
      qd8_f32_qc4w_gemm_config.planes = 2;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnni_prfm);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnni_prfm);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_avx512vnni_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 7;
      qd8_f32_qc4w_gemm_config.nr = 8;
      qd8_f32_qc4w_gemm_config.log2_kr = 3;
      qd8_f32_qc4w_gemm_config.planes = 2;
    #if XNN_ENABLE_AVXVNNI
      } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnni) {
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_avxvnni_params;
        qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
        qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
        qd8_f32_qc4w_gemm_config.mr = 5;
        qd8_f32_qc4w_gemm_config.nr = 8;
        qd8_f32_qc4w_gemm_config.log2_kr = 3;
        qd8_f32_qc4w_gemm_config.planes = 2;
    #endif
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 7;
      qd8_f32_qc4w_gemm_config.nr = 16;
      qd8_f32_qc4w_gemm_config.log2_kr = 3;
      qd8_f32_qc4w_gemm_config.planes = 2;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_xop) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__xop_ld128);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__xop_ld128);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_sse_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 4;
      qd8_f32_qc4w_gemm_config.nr = 4;
      qd8_f32_qc4w_gemm_config.log2_kr = 3;
      qd8_f32_qc4w_gemm_config.planes = 1;
    } else if (hardware_config->use_x86_avx2) {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_avx_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 3;
      qd8_f32_qc4w_gemm_config.nr = 8;
      qd8_f32_qc4w_gemm_config.log2_kr = 3;
      qd8_f32_qc4w_gemm_config.planes = 2;
    } else if (hardware_config->use_x86_avx) {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld128);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__avx_ld128);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_sse_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 4;
      qd8_f32_qc4w_gemm_config.nr = 4;
      qd8_f32_qc4w_gemm_config.log2_kr = 3;
      qd8_f32_qc4w_gemm_config.planes = 1;
    } else if (hardware_config->use_x86_sse4_1) {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld128);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld128);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_sse_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 3;
      qd8_f32_qc4w_gemm_config.nr = 4;
      qd8_f32_qc4w_gemm_config.log2_kr = 3;
      qd8_f32_qc4w_gemm_config.planes = 1;
    } else {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld128);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld128);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_sse_params;
      qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
      qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
      qd8_f32_qc4w_gemm_config.mr = 4;
      qd8_f32_qc4w_gemm_config.nr = 4;
      qd8_f32_qc4w_gemm_config.log2_kr = 3;
      qd8_f32_qc4w_gemm_config.planes = 1;
    }
  #elif XNN_ARCH_WASMRELAXEDSIMD || XNN_ARCH_WASMSIMD
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64);
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64);
    qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_wasmsimd_params;
    qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
    qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
    qd8_f32_qc4w_gemm_config.mr = 4;
    qd8_f32_qc4w_gemm_config.nr = 4;
    qd8_f32_qc4w_gemm_config.log2_kr = 3;
    qd8_f32_qc4w_gemm_config.planes = 2;
  #elif XNN_ARCH_WASM
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm);
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm);
    qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
    qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
    qd8_f32_qc4w_gemm_config.mr = 4;
    qd8_f32_qc4w_gemm_config.nr = 4;
    qd8_f32_qc4w_gemm_config.planes = 2;
  #elif XNN_ARCH_RISCV
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar);
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar);
    qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
    qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
    qd8_f32_qc4w_gemm_config.mr = 4;
    qd8_f32_qc4w_gemm_config.nr = 4;
    qd8_f32_qc4w_gemm_config.planes = 2;
  #elif XNN_ARCH_PPC64
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar);
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar);
    qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
    qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
    qd8_f32_qc4w_gemm_config.mr = 4;
    qd8_f32_qc4w_gemm_config.nr = 4;
    qd8_f32_qc4w_gemm_config.planes = 2;
  #endif
}

static void init_qd8_f16_qc8w_gemm_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
            qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 8;
            qd8_f16_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
            qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 8;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
          qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 1;
          qd8_f16_qc8w_gemm_config.log2_sr = 2;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qd8_f16_qc8w_gemm_config.mr;
          const uint32_t nr = qd8_f16_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qd8_f16_qc8w_gemm_config.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 8 && log2_kr == 2 && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
                    qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c4__neondotfp16arith;
                    qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55;
                    qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith;
                    qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
            qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 8;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
          qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 1;
          qd8_f16_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
            qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
            qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
          qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 3;
        }
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
            qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
            qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
          qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 1;
          qd8_f16_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
            qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
           switch (cpuinfo_get_core(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55);
                qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55);
                break;
              default:
                qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128);
                qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128);
                break;
            }
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
            qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
          qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 1;
          qd8_f16_qc8w_gemm_config.log2_sr = 2;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qd8_f16_qc8w_gemm_config.mr;
          const uint32_t nr = qd8_f16_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qd8_f16_qc8w_gemm_config.log2_kr;
          // Avoid unused warnings.
          (void) mr;
          (void) nr;
          (void) log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 16 && log2_kr == 2 && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
                    qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith;
                    qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55;
                    qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c4__neondotfp16arith;
                    qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55;
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
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
            qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
            qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
          qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 1;
          qd8_f16_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
      qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni_prfm);
      qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni_prfm);
      qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx512vnni_prfm);
      qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_7x8c8__avx512vnni_prfm);
      qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_avxvnni_params;
      qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f16_qc8w_gemm_config.mr = 7;
      qd8_f16_qc8w_gemm_config.nr = 8;
      qd8_f16_qc8w_gemm_config.log2_kr = 3;
    #if XNN_ENABLE_AVXVNNI
      } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnni) {
        // AVX VNNI should be checked before AVX512SKX as it performs better with VNNI microkernels
        qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_avxvnni_params;
        qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
        qd8_f16_qc8w_gemm_config.mr = 5;
        qd8_f16_qc8w_gemm_config.nr = 8;
        qd8_f16_qc8w_gemm_config.log2_kr = 3;
    #endif
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx512skx);
      qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx512skx);
      qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx512skx);
      qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avx512skx);
      qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_avx_params;
      qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f16_qc8w_gemm_config.mr = 5;
      qd8_f16_qc8w_gemm_config.nr = 8;
      qd8_f16_qc8w_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx2) {
      qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx2);
      qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avx2);
      qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2);
      qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_3x8c8__avx2);
      qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_avx_params;
      qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f16_qc8w_gemm_config.mr = 3;
      qd8_f16_qc8w_gemm_config.nr = 8;
      qd8_f16_qc8w_gemm_config.log2_kr = 3;
    }
  #endif
}

static void init_qd8_f32_qc8w_gemm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 8;
            qd8_f32_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c4__neondot);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 8;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f32_qc8w_gemm_config.mr = 2;
          qd8_f32_qc8w_gemm_config.nr = 8;
          qd8_f32_qc8w_gemm_config.log2_kr = 1;
          qd8_f32_qc8w_gemm_config.log2_sr = 2;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qd8_f32_qc8w_gemm_config.mr;
          const uint32_t nr = qd8_f32_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qd8_f32_qc8w_gemm_config.log2_kr;
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
                    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot;
                    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c4__neondot;
                    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
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
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c4__neondot);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 8;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f32_qc8w_gemm_config.mr = 2;
          qd8_f32_qc8w_gemm_config.nr = 8;
          qd8_f32_qc8w_gemm_config.log2_kr = 1;
          qd8_f32_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    } else {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar);
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 1;
      qd8_f32_qc8w_gemm_config.nr = 2;
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f32_qc8w_gemm_config.mr = 2;
          qd8_f32_qc8w_gemm_config.nr = 8;
          qd8_f32_qc8w_gemm_config.log2_kr = 3;
        }
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__neondot);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f32_qc8w_gemm_config.mr = 2;
          qd8_f32_qc8w_gemm_config.nr = 8;
          qd8_f32_qc8w_gemm_config.log2_kr = 1;
          qd8_f32_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
           switch (cpuinfo_get_core(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                break;
              default:
                qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                break;
            }
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16__neon_mlal_lane);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16__neon_mlal_lane);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f32_qc8w_gemm_config.mr = 4;
          qd8_f32_qc8w_gemm_config.nr = 16;
          return;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qd8_f32_qc8w_gemm_config.mr;
          const uint32_t nr = qd8_f32_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qd8_f32_qc8w_gemm_config.log2_kr;
          // Avoid unused warnings.
          (void) mr;
          (void) nr;
          (void) log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 16 && log2_kr == 2 && hardware_config->use_arm_neon_dot) {
                    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot;
                    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot;
                    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
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
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__neondot);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qd8_f32_qc8w_gemm_config.mr = 2;
          qd8_f32_qc8w_gemm_config.nr = 8;
          qd8_f32_qc8w_gemm_config.log2_kr = 1;
          qd8_f32_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512vnni_prfm);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c8__avx512vnni_prfm);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_avx512vnni_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 7;
      qd8_f32_qc8w_gemm_config.nr = 16;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni_prfm);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni_prfm);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx512vnni_prfm);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x8c8__avx512vnni_prfm);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_avxvnni_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 7;
      qd8_f32_qc8w_gemm_config.nr = 8;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    #if XNN_ENABLE_AVXVNNI
      } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnni) {
        // AVX VNNI should be checked before AVX512SKX as it performs better with VNNI microkernels
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_avxvnni_params;
        qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
        qd8_f32_qc8w_gemm_config.mr = 5;
        qd8_f32_qc8w_gemm_config.nr = 8;
        qd8_f32_qc8w_gemm_config.log2_kr = 3;
    #endif
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512skx_prfm);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c8__avx512skx_prfm);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 7;
      qd8_f32_qc8w_gemm_config.nr = 16;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_xop) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__xop_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__xop_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__xop_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__xop_ld64);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_sse_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 2;
      qd8_f32_qc8w_gemm_config.nr = 4;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx2) {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx2);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx2);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx2);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__avx2);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_avx_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 3;
      qd8_f32_qc8w_gemm_config.nr = 8;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx) {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld128);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld128);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__avx_ld128);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__avx_ld128);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_sse_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 2;
      qd8_f32_qc8w_gemm_config.nr = 4;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_sse4_1) {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse41_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse41_ld64);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_sse_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 3;
      qd8_f32_qc8w_gemm_config.nr = 4;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    } else {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse2_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse2_ld64);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_sse_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 3;
      qd8_f32_qc8w_gemm_config.nr = 4;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    }
  #elif XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_wasm_sdot) {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c16__wasmsdot);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c16__wasmsdot);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c16__wasmsdot);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c16__wasmsdot);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 4;
      qd8_f32_qc8w_gemm_config.nr = 4;
      qd8_f32_qc8w_gemm_config.log2_kr = 4;
    } else {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 4;
      qd8_f32_qc8w_gemm_config.nr = 4;
      qd8_f32_qc8w_gemm_config.log2_kr = 1;
      qd8_f32_qc8w_gemm_config.log2_sr = 2;
    }
  #elif XNN_ARCH_WASMSIMD
    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
    qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
    qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
    qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
    qd8_f32_qc8w_gemm_config.mr = 4;
    qd8_f32_qc8w_gemm_config.nr = 4;
    qd8_f32_qc8w_gemm_config.log2_kr = 1;
    qd8_f32_qc8w_gemm_config.log2_sr = 2;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x2__scalar);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 2;
      qd8_f32_qc8w_gemm_config.nr = 2;
    } else {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__wasm);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__wasm);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4__wasm);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4__wasm);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qd8_f32_qc8w_gemm_config.mr = 4;
      qd8_f32_qc8w_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_RISCV
    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__scalar);
    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__scalar);
    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4__scalar);
    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4__scalar);
    qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
    qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
    qd8_f32_qc8w_gemm_config.mr = 4;
    qd8_f32_qc8w_gemm_config.nr = 4;
  #elif XNN_ARCH_PPC64
    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__scalar);
    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__scalar);
    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4__scalar);
    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4__scalar);
    qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
    qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
    qd8_f32_qc8w_gemm_config.mr = 4;
    qd8_f32_qc8w_gemm_config.nr = 4;
  #endif
}

static void init_qs8_qc8w_gemm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            switch (cpuinfo_get_uarch(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot);
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55);
                qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
                qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
                qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
                qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
                qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
                qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
                qs8_qc8w_gemm_config.mr = 4;
                qs8_qc8w_gemm_config.nr = 8;
                qs8_qc8w_gemm_config.log2_kr = 2;
                break;
              default:
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot);
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64);
                qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
                qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
                qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
                qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
                qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
                qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
                qs8_qc8w_gemm_config.mr = 4;
                qs8_qc8w_gemm_config.nr = 8;
                qs8_qc8w_gemm_config.log2_kr = 2;
                break;
            }
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          switch (cpuinfo_get_uarch(0)->uarch) {
            case cpuinfo_uarch_cortex_a5:
            case cpuinfo_uarch_cortex_a7:
            case cpuinfo_uarch_krait:
            case cpuinfo_uarch_kryo:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params;
              qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
              qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a32:
            case cpuinfo_uarch_cortex_a35:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
              qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a57:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
              qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a55r0:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
              qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a72:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
              qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 2;
              qs8_qc8w_gemm_config.nr = 8;
              qs8_qc8w_gemm_config.log2_kr = 1;
              qs8_qc8w_gemm_config.log2_sr = 2;
              break;
            case cpuinfo_uarch_exynos_m1:
            case cpuinfo_uarch_exynos_m2:
            case cpuinfo_uarch_exynos_m3:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64_prfm);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
              qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 8;
              break;

            default:
              if (hardware_config->use_arm_neon_v8) {
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64);
                qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
                qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
                qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
                qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
                qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
                qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
                qs8_qc8w_gemm_config.mr = 4;
                qs8_qc8w_gemm_config.nr = 8;
              } else {
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
                qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params;
                qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
                qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
                qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
                qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
                qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
                qs8_qc8w_gemm_config.mr = 4;
                qs8_qc8w_gemm_config.nr = 8;
              }
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qs8_qc8w_gemm_config.mr;
          const uint32_t nr = qs8_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qs8_qc8w_gemm_config.log2_kr;
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
                    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot;
                    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot;
                    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
                break;
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm;
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35;
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53;
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
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__neondot);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 8;
            qs8_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_v8) {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
          qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 2;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 1;
          qs8_qc8w_gemm_config.log2_sr = 2;
        } else {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params;
          qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 2;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 1;
          qs8_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    } else {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 2;
      qs8_qc8w_gemm_config.nr = 2;
      qs8_qc8w_gemm_config.log2_kr = 2;
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
          qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 2;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 3;
        }
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__neondot);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
          qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 2;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 1;
          qs8_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            switch (cpuinfo_get_core(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                break;
              default:
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                break;
            }
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          switch (cpuinfo_get_core(0)->uarch) {
            case cpuinfo_uarch_cortex_a35:
            case cpuinfo_uarch_kryo:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
              qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a55r0:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
              qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a72:
            case cpuinfo_uarch_cortex_a73:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
              qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 2;
              qs8_qc8w_gemm_config.nr = 8;
              qs8_qc8w_gemm_config.log2_kr = 3;
              break;

            default:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
              qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 2;
              qs8_qc8w_gemm_config.nr = 8;
              qs8_qc8w_gemm_config.log2_kr = 3;
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qs8_qc8w_gemm_config.mr;
          const uint32_t nr = qs8_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qs8_qc8w_gemm_config.log2_kr;
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
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm;
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm;
                }
                break;

              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 16 && log2_kr == 2 && hardware_config->use_arm_neon_dot) {
                    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot;
                    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot;
                    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
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
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM && XNN_ENABLE_ARM_DOTPROD
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__neondot);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
            qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
          qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
          qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 2;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 1;
          qs8_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni_prfm);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c8__avx512vnni_prfm);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_to_qu8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_to_qu8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_to_qu8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_to_qu8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_to_qu8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 7;
      qs8_qc8w_gemm_config.nr = 16;
      qs8_qc8w_gemm_config.log2_kr = 3;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni_prfm);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx512vnni_prfm);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx512vnni_prfm);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x8c8__avx512vnni_prfm);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_to_qu8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_to_qu8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_to_qu8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_to_qu8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_to_qu8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 7;
      qs8_qc8w_gemm_config.nr = 8;
      qs8_qc8w_gemm_config.log2_kr = 3;
    #if XNN_ENABLE_AVXVNNI
      } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnni) {
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni_prfm);
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni_prfm);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni_prfm);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x8c8__avxvnni_prfm);
        qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params;
        qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_to_qu8_gemm_gio_w;
        qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_to_qu8_gemm_goi_w;
        qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_to_qu8_conv_goki_w;
        qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_to_qu8_conv_kgo_w;
        qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_to_qu8_deconv_goki_w;
        qs8_qc8w_gemm_config.mr = 5;
        qs8_qc8w_gemm_config.nr = 8;
        qs8_qc8w_gemm_config.log2_kr = 3;
    #endif
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 7;
      qs8_qc8w_gemm_config.nr = 16;
      qs8_qc8w_gemm_config.log2_kr = 3;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512skx);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx512skx);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx512skx);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__avx512skx);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 4;  // TODO: upgrade to 5x8 prfm when supported
      qs8_qc8w_gemm_config.nr = 8;
      qs8_qc8w_gemm_config.log2_kr = 3;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_xop) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 2;
      qs8_qc8w_gemm_config.nr = 4;
      qs8_qc8w_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx2) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx2);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx2);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__avx2);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 3;
      qs8_qc8w_gemm_config.nr = 8;
      qs8_qc8w_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 2;
      qs8_qc8w_gemm_config.nr = 4;
      qs8_qc8w_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 3;
      qs8_qc8w_gemm_config.nr = 4;
      qs8_qc8w_gemm_config.log2_kr = 3;
    } else {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 3;
      qs8_qc8w_gemm_config.nr = 4;
      qs8_qc8w_gemm_config.log2_kr = 3;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->use_wasm_sdot) {
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot);
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmsdot);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c16__wasmsdot);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c16__wasmsdot);
        qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params;
        qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
        qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
        qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
        qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
        qs8_qc8w_gemm_config.mr = 4;
        qs8_qc8w_gemm_config.nr = 4;
        qs8_qc8w_gemm_config.log2_kr = 4;
      } else {
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params;
        qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
        qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
        qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
        qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
        qs8_qc8w_gemm_config.mr = 4;
        qs8_qc8w_gemm_config.nr = 4;
        qs8_qc8w_gemm_config.log2_kr = 1;
        qs8_qc8w_gemm_config.log2_sr = 2;
      }
    #else
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 4;
      qs8_qc8w_gemm_config.nr = 4;
      qs8_qc8w_gemm_config.log2_kr = 1;
      qs8_qc8w_gemm_config.log2_sr = 2;
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 2;
      qs8_qc8w_gemm_config.nr = 2;
    } else {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params;
      qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
      qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 4;
      qs8_qc8w_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_RISCV
    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params;
    qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
    qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
    qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
    qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
    qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
    qs8_qc8w_gemm_config.mr = 3;
    qs8_qc8w_gemm_config.nr = 4;
  #elif XNN_ARCH_PPC64
    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params;
    qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
    qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
    qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
    qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
    qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
    qs8_qc8w_gemm_config.mr = 3;
    qs8_qc8w_gemm_config.nr = 4;
  #endif
}

static void init_qu8_gemm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_udot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__neondot);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__neondot);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
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
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a32:
            case cpuinfo_uarch_cortex_a35:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a57:
            case cpuinfo_uarch_cortex_a72:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a55r0:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_exynos_m1:
            case cpuinfo_uarch_exynos_m2:
            case cpuinfo_uarch_exynos_m3:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 8;
              break;
            default:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
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
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm;
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7;
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_udot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__neondot);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__neondot);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
            qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 8;
            qu8_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane);
          qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
          qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
          qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
          qu8_gemm_config.mr = 3;
          qu8_gemm_config.nr = 8;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    } else {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_armsimd32_params;
      qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 2;
      qu8_gemm_config.nr = 2;
      qu8_gemm_config.log2_kr = 2;
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_ASSEMBLY
      if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
        #if XNN_ENABLE_ARM_I8MM
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c8__neoni8mm);
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c8__neoni8mm);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c8__neoni8mm);
          qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
          qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
          qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
          qu8_gemm_config.mr = 4;
          qu8_gemm_config.nr = 16;
          qu8_gemm_config.log2_kr = 3;
        #endif  // XNN_ENABLE_ARM_I8MM
      } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
        #if XNN_ENABLE_ARM_DOTPROD
          switch (cpuinfo_get_core(0)->uarch) {
            case cpuinfo_uarch_cortex_a55:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
              qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
              qu8_gemm_config.mr = 4;
              qu8_gemm_config.nr = 16;
              qu8_gemm_config.log2_kr = 2;
              break;
            default:
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
              qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
              qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128);
              qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
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
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
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
          case cpuinfo_uarch_neoverse_n1:
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75_prfm);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75_prfm);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
            qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 16;
            break;

          default:
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
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
                qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm;
                qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm;
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
      if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
        #if XNN_ENABLE_ARM_I8MM
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c8__neoni8mm);
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c8__neoni8mm);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c8__neoni8mm);
          qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
          qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
          qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
          qu8_gemm_config.mr = 4;
          qu8_gemm_config.nr = 16;
          qu8_gemm_config.log2_kr = 3;
        #endif  // XNN_ENABLE_ARM_I8MM
      } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
        #if XNN_ENABLE_ARM_DOTPROD
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__neondot);
          qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
          qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
          qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
          qu8_gemm_config.mr = 4;
          qu8_gemm_config.nr = 16;
          qu8_gemm_config.log2_kr = 2;
        #endif  // XNN_ENABLE_ARM_DOTPROD
      } else {
        qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
        qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane);
        qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
        qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane);
        qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
        qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
        qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
        qu8_gemm_config.mr = 4;
        qu8_gemm_config.nr = 16;
      }
    #endif  // XNN_ENABLE_ASSEMBLY
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx512_params;
      qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 7;
      qu8_gemm_config.nr = 16;
      qu8_gemm_config.log2_kr = 3;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx512skx);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_4x8c8__avx512skx);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x8c8__avx512skx);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_4x8c8__avx512skx);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx512_params;
      qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 4;  // TODO: upgrade to 5x8 prfm when supported
      qu8_gemm_config.nr = 8;
      qu8_gemm_config.log2_kr = 3;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_xop) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 2;
      qu8_gemm_config.nr = 4;
      qu8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx2) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x8c8__avx2);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x8c8__avx2);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx2_params;
      qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 3;
      qu8_gemm_config.nr = 8;
      qu8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 2;
      qu8_gemm_config.nr = 4;
      qu8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 3;
      qu8_gemm_config.nr = 4;
      qu8_gemm_config.log2_kr = 3;
    } else {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 3;
      qu8_gemm_config.nr = 4;
      qu8_gemm_config.log2_kr = 3;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_wasmsimd_params;
    qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
    qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
    qu8_gemm_config.mr = 4;
    qu8_gemm_config.nr = 4;
    qu8_gemm_config.log2_kr = 1;
    qu8_gemm_config.log2_sr = 2;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params;
      qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 2;
      qu8_gemm_config.nr = 2;
    } else {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
      qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
      qu8_gemm_config.mr = 4;
      qu8_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_RISCV
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params;
    qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
    qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
    qu8_gemm_config.mr = 3;
    qu8_gemm_config.nr = 4;
  #elif XNN_ARCH_PPC64
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params;
    qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
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

  static BOOL CALLBACK init_f32_gemm_nr2_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_gemm_nr2_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_qc4w_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_qc4w_gemm_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_qc8w_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_qc8w_gemm_config();
    return TRUE;
  }

 static BOOL CALLBACK init_qd8_f16_qc8w_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qd8_f16_qc8w_gemm_config();
    return TRUE;
 }

 static BOOL CALLBACK init_qd8_f16_qc4w_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qd8_f16_qc4w_gemm_config();
    return TRUE;
  }

 static BOOL CALLBACK init_qd8_f32_qc4w_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qd8_f32_qc4w_gemm_config();
    return TRUE;
  }

 static BOOL CALLBACK init_qd8_f32_qc8w_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qd8_f32_qc8w_gemm_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qs8_qc8w_gemm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qs8_qc8w_gemm_config();
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

struct xnn_gemm_config* xnn_init_f32_gemm_nr2_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_gemm_nr2, &init_f32_gemm_nr2_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_gemm_nr2, &init_f32_gemm_nr2_config);
  #endif
  return &f32_gemm_nr2_config;
}

struct xnn_gemm_config* xnn_init_f32_qc4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_qc4w_gemm, &init_f32_qc4w_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_qc4w_gemm, &init_f32_qc4w_gemm_config);
  #endif
  return &f32_qc4w_gemm_config;
}

struct xnn_gemm_config* xnn_init_f32_qc8w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_qc8w_gemm, &init_f32_qc8w_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_qc8w_gemm, &init_f32_qc8w_gemm_config);
  #endif
  return &f32_qc8w_gemm_config;
}

struct xnn_gemm_config* xnn_init_qd8_f16_qc8w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qd8_f16_qc8w_gemm, &init_qd8_f16_qc8w_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qd8_f16_qc8w_gemm, &init_qd8_f16_qc8w_gemm_config);
  #endif
  return &qd8_f16_qc8w_gemm_config;
}

struct xnn_gemm_config* xnn_init_qd8_f16_qc4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qd8_f16_qc4w_gemm, &init_qd8_f16_qc4w_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qd8_f16_qc4w_gemm, &init_qd8_f16_qc4w_gemm_config);
  #endif
  return &qd8_f16_qc4w_gemm_config;
}

struct xnn_gemm_config* xnn_init_qd8_f32_qc4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qd8_f32_qc4w_gemm, &init_qd8_f32_qc4w_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qd8_f32_qc4w_gemm, &init_qd8_f32_qc4w_gemm_config);
  #endif
  return &qd8_f32_qc4w_gemm_config;
}

struct xnn_gemm_config* xnn_init_qd8_f32_qc8w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qd8_f32_qc8w_gemm, &init_qd8_f32_qc8w_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qd8_f32_qc8w_gemm, &init_qd8_f32_qc8w_gemm_config);
  #endif
  return &qd8_f32_qc8w_gemm_config;
}

struct xnn_gemm_config* xnn_init_qs8_qc8w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qs8_qc8w_gemm, &init_qs8_qc8w_gemm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qs8_qc8w_gemm, &init_qs8_qc8w_gemm_config);
  #endif
  return &qs8_qc8w_gemm_config;
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
