// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <pthread.h>
#endif

#ifdef _MSC_VER
  #include <intrin.h>
#endif

#ifndef __EMSCRIPTEN__
  #include <cpuinfo.h>
#endif

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/argmaxpool.h>
#include <xnnpack/avgpool.h>
#include <xnnpack/common.h>
#include <xnnpack/conv.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/depthtospace.h>
#include <xnnpack/gavgpool.h>
#include <xnnpack/gemm.h>
#include <xnnpack/fill.h>
#include <xnnpack/ibilinear.h>
#include <xnnpack/igemm.h>
#include <xnnpack/log.h>
#include <xnnpack/lut.h>
#include <xnnpack/maxpool.h>
#include <xnnpack/pad.h>
#include <xnnpack/params.h>
#include <xnnpack/params-init.h>
#include <xnnpack/pavgpool.h>
#include <xnnpack/prelu.h>
#include <xnnpack/raddstoreexpminusmax.h>
#include <xnnpack/rmax.h>
#include <xnnpack/spmm.h>
#include <xnnpack/unpool.h>
#include <xnnpack/vaddsub.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vcvt.h>
#include <xnnpack/vmul.h>
#include <xnnpack/vmulcaddc.h>
#include <xnnpack/vunary.h>
#include <xnnpack/zip.h>

#ifndef XNN_ENABLE_ASSEMBLY
  #define XNN_ENABLE_ASSEMBLY 1
#endif

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard = PTHREAD_ONCE_INIT;
#endif

static const struct xnn_allocator* volatile init_allocator = NULL;

struct xnn_parameters xnn_params = {
  .init_flags = 0
};

static void init(void) {
#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  // Unlike most other architectures, on x86/x86-64 when floating-point instructions
  // have no NaN arguments, but produce NaN output, the output NaN has sign bit set.
  // We use it to distinguish x86/x86-64 from other architectures, by doing subtraction
  // of two infinities (must produce NaN per IEEE 754 standard).
  static const volatile float inf = INFINITY;
  const bool is_wasm_x86 = signbit(inf - inf);
#endif
  uint32_t init_flags = XNN_INIT_FLAG_XNNPACK;

#if XNN_ARCH_ARM
  #if XNN_PLATFORM_MOBILE
    if (!cpuinfo_has_arm_neon()) {
      xnn_log_error("XNNPACK initialization failed: NEON is not supported");
      return;
    }
  #else
    if (!cpuinfo_has_arm_vfpv2() && !cpuinfo_has_arm_vfpv3()) {
      xnn_log_error("XNNPACK initialization failed: VFP is not supported");
      return;
    }
  #endif

  if (cpuinfo_has_arm_neon()) {
    /**************************** QC8 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_QC8_OPERATORS
      init_flags |= XNN_INIT_FLAG_QC8;

      #if XNN_ENABLE_ASSEMBLY
        if (!XNN_PLATFORM_IOS && cpuinfo_has_arm_neon_dot()) {
          switch (cpuinfo_get_uarch(0)->uarch) {
            case cpuinfo_uarch_cortex_a55:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8c4__aarch32_neondot_cortex_a55);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8c4__aarch32_neondot_cortex_a55);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c4__neondot);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c4__neondot);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
              xnn_params.qc8.gemm.mr = 4;
              xnn_params.qc8.gemm.nr = 8;
              xnn_params.qc8.gemm.log2_kr = 2;
              break;
            default:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8c4__aarch32_neondot_ld64);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8c4__aarch32_neondot_ld64);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c4__neondot);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c4__neondot);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
              xnn_params.qc8.gemm.mr = 4;
              xnn_params.qc8.gemm.nr = 8;
              xnn_params.qc8.gemm.log2_kr = 2;
              break;
          }
        } else {
          switch (cpuinfo_get_uarch(0)->uarch) {
            case cpuinfo_uarch_cortex_a7:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a7);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neon_params;
              xnn_params.qc8.gemm.mr = 4;
              xnn_params.qc8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a35:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a7);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__aarch32_neon_mlal_lane_ld64);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neon_params;
              xnn_params.qc8.gemm.mr = 4;
              xnn_params.qc8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a57:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_prfm_cortex_a53);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_prfm_cortex_a53);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
              xnn_params.qc8.gemm.mr = 4;
              xnn_params.qc8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a55r0:
            case cpuinfo_uarch_kryo:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_cortex_a53);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_cortex_a53);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
              xnn_params.qc8.gemm.mr = 4;
              xnn_params.qc8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a72:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
              xnn_params.qc8.gemm.mr = 2;
              xnn_params.qc8.gemm.nr = 8;
              xnn_params.qc8.gemm.log2_kr = 1;
              xnn_params.qc8.gemm.log2_sr = 2;
              break;
            case cpuinfo_uarch_exynos_m1:
            case cpuinfo_uarch_exynos_m2:
            case cpuinfo_uarch_exynos_m3:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_prfm_ld64);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_prfm_ld64);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
              xnn_params.qc8.gemm.mr = 4;
              xnn_params.qc8.gemm.nr = 8;
              break;

            default:
              if (cpuinfo_has_arm_neon_v8()) {
                xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64);
                xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64);
                xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane);
                xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane);
                xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
                xnn_params.qc8.gemm.mr = 4;
                xnn_params.qc8.gemm.nr = 8;
              } else {
                xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__aarch32_neon_mlal_lane_ld64);
                xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__aarch32_neon_mlal_lane_ld64);
                xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane);
                xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane);
                xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neon_params;
                xnn_params.qc8.gemm.mr = 4;
                xnn_params.qc8.gemm.nr = 8;
              }
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = xnn_params.qc8.gemm.mr;
          const uint32_t nr = xnn_params.qc8.gemm.nr;
          const uint32_t log2_kr = xnn_params.qc8.gemm.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                if (mr == 4 && nr == 8 && log2_kr == 2 && cpuinfo_has_arm_neon_dot()) {
                  xnn_params.qc8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8c4__aarch32_neondot_cortex_a55;
                  xnn_params.qc8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8c4__aarch32_neondot_cortex_a55;
                  xnn_params.qc8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c4__neondot;
                  xnn_params.qc8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c4__neondot;
                }
                break;
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  xnn_params.qc8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_prfm_cortex_a53;
                  xnn_params.qc8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_prfm_cortex_a53;
                  xnn_params.qc8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane;
                  xnn_params.qc8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  xnn_params.qc8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_cortex_a53;
                  xnn_params.qc8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_cortex_a53;
                  xnn_params.qc8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane;
                  xnn_params.qc8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane;
                }
                break;

              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        if (!XNN_PLATFORM_IOS && cpuinfo_has_arm_neon_dot()) {
          xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x8c4__neondot);
          xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x8c4__neondot);
          xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c4__neondot);
          xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c4__neondot);
          xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
          xnn_params.qc8.gemm.mr = 4;
          xnn_params.qc8.gemm.nr = 8;
          xnn_params.qc8.gemm.log2_kr = 2;
        } else if (cpuinfo_has_arm_v8()) {
          xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
          xnn_params.qc8.gemm.mr = 2;
          xnn_params.qc8.gemm.nr = 8;
          xnn_params.qc8.gemm.log2_kr = 1;
          xnn_params.qc8.gemm.log2_sr = 2;
        } else {
          xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal);
          xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal);
          xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal);
          xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal);
          xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neon_params;
          xnn_params.qc8.gemm.mr = 2;
          xnn_params.qc8.gemm.nr = 8;
          xnn_params.qc8.gemm.log2_kr = 1;
          xnn_params.qc8.gemm.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY

      if (cpuinfo_has_arm_neon_v8()) {
        xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neonv8_mla8_ld64;
        xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_neonv8_params;
        xnn_params.qc8.dwconv[0].channel_tile = 16;
        xnn_params.qc8.dwconv[0].primary_tile = 9;
        xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__neonv8_mla8_ld64;
        xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_neonv8_params;
        xnn_params.qc8.dwconv[1].channel_tile = 8;
        xnn_params.qc8.dwconv[1].primary_tile = 25;
      } else {
        xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neon_mla8_ld64;
        xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_neon_params;
        xnn_params.qc8.dwconv[0].channel_tile = 16;
        xnn_params.qc8.dwconv[0].primary_tile = 9;
        xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__neon_mla8_ld64;
        xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_neon_params;
        xnn_params.qc8.dwconv[1].channel_tile = 8;
        xnn_params.qc8.dwconv[1].primary_tile = 25;
      }
    #endif  // XNN_NO_QC8_OPERATORS

    /**************************** QS8 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_QS8_OPERATORS
      init_flags |= XNN_INIT_FLAG_QS8;

      #if XNN_ENABLE_ASSEMBLY
        if (!XNN_PLATFORM_IOS && cpuinfo_has_arm_neon_dot()) {
          switch (cpuinfo_get_uarch(0)->uarch) {
            case cpuinfo_uarch_cortex_a55:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__aarch32_neondot_cortex_a55);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__aarch32_neondot_cortex_a55);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 4;
              xnn_params.qs8.gemm.nr = 8;
              xnn_params.qs8.gemm.log2_kr = 2;
              break;
            default:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__aarch32_neondot_ld64);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__aarch32_neondot_ld64);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 4;
              xnn_params.qs8.gemm.nr = 8;
              xnn_params.qs8.gemm.log2_kr = 2;
              break;
          }
        } else {
          switch (cpuinfo_get_uarch(0)->uarch) {
            case cpuinfo_uarch_cortex_a7:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a7);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 4;
              xnn_params.qs8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a35:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a7);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 4;
              xnn_params.qs8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a57:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 4;
              xnn_params.qs8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a55r0:
            case cpuinfo_uarch_kryo:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a53);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a53);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 4;
              xnn_params.qs8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a72:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 2;
              xnn_params.qs8.gemm.nr = 8;
              xnn_params.qs8.gemm.log2_kr = 1;
              xnn_params.qs8.gemm.log2_sr = 2;
              break;
            case cpuinfo_uarch_exynos_m1:
            case cpuinfo_uarch_exynos_m2:
            case cpuinfo_uarch_exynos_m3:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 4;
              xnn_params.qs8.gemm.nr = 8;
              break;
            default:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 4;
              xnn_params.qs8.gemm.nr = 8;
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = xnn_params.qs8.gemm.mr;
          const uint32_t nr = xnn_params.qs8.gemm.nr;
          const uint32_t log2_kr = xnn_params.qs8.gemm.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                if (mr == 4 && nr == 8 && log2_kr == 2 && cpuinfo_has_arm_neon_dot()) {
                  xnn_params.qs8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__aarch32_neondot_cortex_a55;
                  xnn_params.qs8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__aarch32_neondot_cortex_a55;
                  xnn_params.qs8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot;
                  xnn_params.qs8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c4__neondot;
                }
                break;
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  xnn_params.qs8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53;
                  xnn_params.qs8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53;
                  xnn_params.qs8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane;
                  xnn_params.qs8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  xnn_params.qs8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a53;
                  xnn_params.qs8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a53;
                  xnn_params.qs8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane;
                  xnn_params.qs8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        if (!XNN_PLATFORM_IOS && cpuinfo_has_arm_neon_dot()) {
          xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neondot);
          xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neondot);
          xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
          xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
          xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          xnn_params.qs8.gemm.mr = 4;
          xnn_params.qs8.gemm.nr = 8;
          xnn_params.qs8.gemm.log2_kr = 2;
        } else {
          xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          xnn_params.qs8.gemm.mr = 2;
          xnn_params.qs8.gemm.nr = 8;
          xnn_params.qs8.gemm.log2_kr = 1;
          xnn_params.qs8.gemm.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY

      xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64;
      xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
      xnn_params.qs8.dwconv[0].channel_tile = 16;
      xnn_params.qs8.dwconv[0].primary_tile = 9;
      xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64;
      xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
      xnn_params.qs8.dwconv[1].channel_tile = 8;
      xnn_params.qs8.dwconv[1].primary_tile = 25;

      xnn_params.qs8.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qs8_gavgpool_minmax_rndnu_ukernel_7x__neon_c8,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qs8_gavgpool_minmax_rndnu_ukernel_7p7x__neon_c8,
        .init.qs8 = xnn_init_qs8_avgpool_minmax_rndnu_neon_params,
        .update.qs8 = xnn_update_qs8_avgpool_minmax_rndnu_neon_params,
        .row_tile = 7,
        .channel_tile = 8,
      };

      xnn_params.qs8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__neon_ld64_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16,
        .init.qs8_addsub = xnn_init_qs8_add_minmax_neon_params,
        .element_tile = 16,
      };
      xnn_params.qs8.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmul_minmax_rndnu_ukernel__neon_ld64_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_rndnu_ukernel__neon_ld64_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_rndnu_ukernel__neon_ld64_x16,
        .init.qs8_mul = xnn_init_qs8_mul_minmax_rndnu_neon_params,
        .element_tile = 16,
      };
    #endif  // XNN_NO_QS8_OPERATORS

    /*************************** QU8 AArch32 micro-kernels ***************************/
    #ifndef XNN_NO_QU8_OPERATORS
      init_flags |= XNN_INIT_FLAG_QU8;

      #if XNN_ENABLE_ASSEMBLY
        if (!XNN_PLATFORM_IOS && cpuinfo_has_arm_neon_dot()) {
          xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__neondot);
          xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__neondot);
          xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
          xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
          xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
          xnn_params.qu8.gemm.mr = 4;
          xnn_params.qu8.gemm.nr = 8;
          xnn_params.qu8.gemm.log2_kr = 2;
        } else {
          switch (cpuinfo_get_uarch(0)->uarch) {
            case cpuinfo_uarch_cortex_a7:
              xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a7);
              xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64);
              xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              xnn_params.qu8.gemm.mr = 4;
              xnn_params.qu8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a35:
              xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a7);
              xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64);
              xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              xnn_params.qu8.gemm.mr = 4;
              xnn_params.qu8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a57:
            case cpuinfo_uarch_cortex_a72:
              xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53);
              xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53);
              xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              xnn_params.qu8.gemm.mr = 4;
              xnn_params.qu8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a55r0:
            case cpuinfo_uarch_kryo:
              xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a53);
              xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a53);
              xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              xnn_params.qu8.gemm.mr = 4;
              xnn_params.qu8.gemm.nr = 8;
              break;
            case cpuinfo_uarch_exynos_m1:
            case cpuinfo_uarch_exynos_m2:
            case cpuinfo_uarch_exynos_m3:
              xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64);
              xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64);
              xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              xnn_params.qu8.gemm.mr = 4;
              xnn_params.qu8.gemm.nr = 8;
              break;
            default:
              xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64);
              xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64);
              xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
              xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
              xnn_params.qu8.gemm.mr = 4;
              xnn_params.qu8.gemm.nr = 8;
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = xnn_params.qu8.gemm.mr;
          const uint32_t nr = xnn_params.qu8.gemm.nr;
          const uint32_t log2_kr = xnn_params.qu8.gemm.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  xnn_params.qu8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53;
                  xnn_params.qu8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53;
                  xnn_params.qu8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane;
                  xnn_params.qu8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  xnn_params.qu8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a53;
                  xnn_params.qu8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a53;
                  xnn_params.qu8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane;
                  xnn_params.qu8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        if (!XNN_PLATFORM_IOS && cpuinfo_has_arm_neon_dot()) {
          xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__neondot);
          xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__neondot);
          xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot);
          xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot);
          xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
          xnn_params.qu8.gemm.mr = 4;
          xnn_params.qu8.gemm.nr = 8;
          xnn_params.qu8.gemm.log2_kr = 2;
        } else {
          xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane);
          xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane);
          xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
          xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
          xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
          xnn_params.qu8.gemm.mr = 3;
          xnn_params.qu8.gemm.nr = 8;
        }
      #endif  // XNN_ENABLE_ASSEMBLY

      xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8;
      xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
      xnn_params.qu8.dwconv[0].channel_tile = 16;
      xnn_params.qu8.dwconv[0].primary_tile = 9;
      xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8;
      xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
      xnn_params.qu8.dwconv[1].channel_tile = 8;
      xnn_params.qu8.dwconv[1].primary_tile = 25;

      xnn_params.qu8.avgpool = (struct avgpool_parameters) {
        .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9x__neon_c8,
        .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9p8x__neon_c8,
        .init.qu8 = xnn_init_qu8_avgpool_minmax_neon_params,
        .primary_tile = 9,
        .incremental_tile = 8,
        .channel_tile = 8,
      };
      xnn_params.qu8.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qu8_gavgpool_minmax_rndnu_ukernel_7x__neon_c8,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qu8_gavgpool_minmax_rndnu_ukernel_7p7x__neon_c8,
        .init.qu8 = xnn_init_qu8_avgpool_minmax_rndnu_neon_params,
        .update.qu8 = xnn_update_qu8_avgpool_minmax_rndnu_neon_params,
        .row_tile = 7,
        .channel_tile = 8,
      };
      xnn_params.qu8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__neon_ld64_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__neon_ld64_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__neon_ld64_x16,
        .init.qu8_addsub = xnn_init_qu8_add_minmax_neon_params,
        .element_tile = 8,
      };
      xnn_params.qu8.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmul_minmax_rndnu_ukernel__neon_ld64_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_x16,
        .init.qu8_mul = xnn_init_qu8_mul_minmax_rndnu_neon_params,
        .element_tile = 16,
      };
    #endif  // XNN_NO_QU8_OPERATORS

    /**************************** S8 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_S8_OPERATORS
      init_flags |= XNN_INIT_FLAG_S8;

      xnn_params.s8.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_s8_vclamp_ukernel__neon_x64,
        .init.s8_minmax = xnn_init_s8_minmax_neon_params,
        .element_tile = 64,
      };
      xnn_params.s8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_s8_ibilinear_ukernel__neon_c8,
        .pixel_tile = 1,
        .channel_tile = 8,
      };
      xnn_params.s8.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_s8_maxpool_minmax_ukernel_9p8x__neon_c16,
        .init.s8 = xnn_init_s8_minmax_neon_params,
        .mr = 9,
        .qr = 8,
      };
    #endif  // XNN_NO_S8_OPERATORS

    /**************************** U8 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_U8_OPERATORS
      init_flags |= XNN_INIT_FLAG_U8;

      xnn_params.u8.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_u8_vclamp_ukernel__neon_x64,
        .init.u8_minmax = xnn_init_u8_minmax_neon_params,
        .element_tile = 64,
      };
      xnn_params.u8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_u8_ibilinear_ukernel__neon_c8,
        .pixel_tile = 1,
        .channel_tile = 8,
      };
      xnn_params.u8.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16,
        .init.u8 = xnn_init_u8_minmax_neon_params,
        .mr = 9,
        .qr = 8,
      };
      xnn_params.u8.rmax = xnn_u8_rmax_ukernel__neon;
      xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    #endif  // XNN_NO_U8_OPERATORS

    /**************************** X8 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_X8_OPERATORS
      init_flags |= XNN_INIT_FLAG_X8;

      xnn_params.x8.lut = xnn_x8_lut_ukernel__scalar_x4;
      xnn_params.x8.zip = (struct zip_parameters) {
        .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__neon,
        .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__neon,
        .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__neon,
        .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__neon,
      };
    #endif  // XNN_NO_X8_OPERATORS

    /**************************** F32 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_F32_OPERATORS
      init_flags |= XNN_INIT_FLAG_F32;

      #if XNN_ENABLE_ASSEMBLY
        switch (cpuinfo_get_uarch(0)->uarch) {
          case cpuinfo_uarch_cortex_a5:
          case cpuinfo_uarch_cortex_a7:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a7);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a7);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 4;
            xnn_params.f32.gemm.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a53:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_prfm_cortex_a53);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_prfm_cortex_a53);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 4;
            xnn_params.f32.gemm.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a55r0:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a53);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a53);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 4;
            xnn_params.f32.gemm.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a35:
          case cpuinfo_uarch_cortex_a55:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a55);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a55);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 4;
            xnn_params.f32.gemm.nr = 8;
            break;

          case cpuinfo_uarch_cortex_a57:
          case cpuinfo_uarch_cortex_a72:
          case cpuinfo_uarch_cortex_a73:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_prfm_cortex_a75);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_prfm_cortex_a75);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 4;
            xnn_params.f32.gemm.nr = 8;
            break;

          case cpuinfo_uarch_krait:
          default:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a75);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a75);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 4;
            xnn_params.f32.gemm.nr = 8;
            break;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = xnn_params.f32.gemm.mr;
          const uint32_t nr = xnn_params.f32.gemm.nr;
          const uint32_t log2_sr = xnn_params.f32.gemm.log2_sr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8 && log2_sr == 0) {
                  xnn_params.f32.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_prfm_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_prfm_cortex_a53;
                  xnn_params.f32.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64;
                  xnn_params.f32.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8 && log2_sr == 0) {
                  xnn_params.f32.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a53;
                  xnn_params.f32.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64;
                  xnn_params.f32.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64;
                }
                break;
              case cpuinfo_uarch_cortex_a55:
                if (mr == 4 && nr == 8 && log2_sr == 0) {
                  xnn_params.f32.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a55;
                  xnn_params.f32.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a55;
                  xnn_params.f32.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64;
                  xnn_params.f32.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128);
        xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld128);
        xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64);
        xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64);
        xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
        xnn_params.f32.gemm.mr = 4;
        xnn_params.f32.gemm.nr = 8;
      #endif  // XNN_ENABLE_ASSEMBLY
      xnn_params.f32.gemm2.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64);
      xnn_params.f32.gemm2.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64);
      xnn_params.f32.gemm2.init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.gemm2.mr = 4;
      xnn_params.f32.gemm2.nr = 2;

      xnn_params.f32.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x3__neon;
      xnn_params.f32.dwconv[0].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[0].channel_tile = 8,
      xnn_params.f32.dwconv[0].primary_tile = 3,

      xnn_params.f32.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x4__neon;
      xnn_params.f32.dwconv[1].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[1].channel_tile = 8,
      xnn_params.f32.dwconv[1].primary_tile = 4,

      xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x9__neon;
      xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[2].channel_tile = 8;
      xnn_params.f32.dwconv[2].primary_tile = 9;

      xnn_params.f32.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2;
      xnn_params.f32.dwconv[3].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[3].channel_tile = 8;
      xnn_params.f32.dwconv[3].primary_tile = 25;

      xnn_params.f32.avgpool = (struct avgpool_parameters) {
        .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9x__neon_c4,
        .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9p8x__neon_c4,
        .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
        .primary_tile = 9,
        .incremental_tile = 8,
        .channel_tile = 4,
      };
      xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
        .unipass = (xnn_pavgpool_unipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9x__neon_c4,
        .multipass = (xnn_pavgpool_multipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9p8x__neon_c4,
        .primary_tile = 9,
        .incremental_tile = 8,
        .channel_tile = 4,
      };
      xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4,
        .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
        .update.f32 = xnn_update_f32_scaleminmax_scalar_params,
        .row_tile = 7,
        .channel_tile = 4,
      };
      xnn_params.f32.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4,
        .init.f32 = xnn_init_f32_minmax_scalar_params,
        .mr = 9,
        .qr = 8,
      };
      xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
        .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__neon_c4,
        .mr = 4,
      };
      xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
        .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__neon_c4,
        .mr = 9,
      };
      xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
        .mp = (xnn_argmaxpool_multipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__neon_c4,
        .mr = 9,
        .qr = 8,
      };
      xnn_params.f32.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_f32_ibilinear_ukernel__neon_c8,
        .pixel_tile = 1,
        .channel_tile = 8,
      };
      xnn_params.f32.abs = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vabs_ukernel__neon_x8,
        .element_tile = 8,
      };
      xnn_params.f32.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vclamp_ukernel__neon_x8,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 8,
      };
      if (cpuinfo_has_arm_neon_fma()) {
        xnn_params.f32.elu = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__neonfma_rr1_p6_x8,
          .init.f32_elu = xnn_init_f32_elu_neonfma_rr1_p6_params,
          .element_tile = 8,
        };
      } else {
        xnn_params.f32.elu = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x8,
          .init.f32_elu = xnn_init_f32_elu_neon_rr2_lut16_p3_params,
          .element_tile = 8,
        };
      }
      xnn_params.f32.hswish = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vhswish_ukernel__neon_x16,
        .init.f32_hswish = xnn_init_f32_hswish_scalar_params,
        .element_tile = 16,
      };
      xnn_params.f32.lrelu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vlrelu_ukernel__neon_x8,
        .init.f32_lrelu = xnn_init_f32_lrelu_scalar_params,
        .element_tile = 8,
      };
      xnn_params.f32.neg = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vneg_ukernel__neon_x8,
        .element_tile = 8,
      };
      if (cpuinfo_has_arm_neon_v8()) {
        xnn_params.f32.rndne = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__neonv8_x8,
          .element_tile = 8,
        };
        xnn_params.f32.rndz = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__neonv8_x8,
          .element_tile = 8,
        };
        xnn_params.f32.rndu = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__neonv8_x8,
          .element_tile = 8,
        };
        xnn_params.f32.rndd = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__neonv8_x8,
          .element_tile = 8,
        };
      } else {
        xnn_params.f32.rndne = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__neon_x8,
          .element_tile = 8,
        };
        xnn_params.f32.rndz = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__neon_x8,
          .element_tile = 8,
        };
        xnn_params.f32.rndu = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__neon_x8,
          .element_tile = 8,
        };
        xnn_params.f32.rndd = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__neon_x8,
          .element_tile = 8,
        };
      }
      xnn_params.f32.sigmoid = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8,
        .init.f32_sigmoid = xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params,
        .element_tile = 8,
      };
      xnn_params.f32.sqr = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqr_ukernel__neon_x8,
        .element_tile = 8,
      };
      xnn_params.f32.sqrt = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqrt_ukernel__scalar_sqrt_x1,
        .element_tile = 1,
      };
      xnn_params.f32.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__neon_2x8,
        .row_tile = 2,
        .channel_tile = 8,
      };
      xnn_params.f32.raddstoreexpminusmax = (struct raddstoreexpminusmax_parameters) {
        .ukernel = xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x8,
        .init = xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
        .element_tile = 8,
      };
      xnn_params.f32.rmax = xnn_f32_rmax_ukernel__neon;
      xnn_params.f32.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_minmax_ukernel__neon_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__neon_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__neon_x8,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 8,
      };
      xnn_params.f32.vdiv = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_minmax_ukernel__scalar_x2,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_minmax_ukernel__scalar_x2,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_minmax_ukernel__scalar_x2,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 2,
      };
      xnn_params.f32.vmax = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__neon_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__neon_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__neon_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vmin = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__neon_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__neon_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__neon_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_minmax_ukernel__neon_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__neon_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__neon_x8,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 8,
      };
      xnn_params.f32.vsub = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_minmax_ukernel__neon_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_minmax_ukernel__neon_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_minmax_ukernel__neon_x8,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 8,
      };
      xnn_params.f32.vsqrdiff = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiff_ukernel__neon_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__neon_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__neon_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
        .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x,
        .init.f32 = xnn_init_f32_minmax_scalar_params,
        .channel_tile = 4,
        .row_tile = 2,
      };
      #ifndef XNN_NO_NCHW_OPERATORS
        init_flags |= XNN_INIT_FLAG_CHW_OPT;

        xnn_params.f32.spmm = (struct spmm_parameters) {
          .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_32x1__neon,
          .mr = 32,
          .nr = 1,
        };
        xnn_params.f32.conv_hwc2chw_3x3c3s2 = (struct conv_hwc2chw_parameters) {
          .ukernel_with_symm_padding =
            (xnn_conv_hwc2chw_ukernel_function) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2,
          .output_channel_tile = 4,
          .output_height_tile = 2,
          .output_width_tile = 2,
        };
        xnn_params.f32.dwconv2d_chw_3x3 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_2x4,
          .output_width_tile = 4,
          .output_height_tile = 2,
        };
        xnn_params.f32.dwconv2d_chw_3x3s2 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_1x4,
          .output_width_tile = 4,
          .output_height_tile = 1,
        };
        xnn_params.f32.dwconv2d_chw_5x5 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4,
          .output_width_tile = 4,
          .output_height_tile = 1,
        };
        xnn_params.f32.dwconv2d_chw_5x5s2 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4,
          .output_width_tile = 4,
          .output_height_tile = 1,
        };
        xnn_params.f32.gavgpool_cw = (struct gavgpool_cw_parameters) {
          .ukernel = (xnn_gavgpool_cw_ukernel_function) xnn_f32_gavgpool_cw_ukernel__neon_x4,
          .channel_tile = 4,
        };
        xnn_params.f32.ibilinear_chw = (struct ibilinear_chw_parameters) {
          .ukernel = (xnn_ibilinear_chw_ukernel_function) xnn_f32_ibilinear_chw_ukernel__neon_p8,
          .channel_tile = 1,
          .pixel_tile = 8,
        };
      #endif  // XNN_NO_NCHW_OPERATORS
    #endif  // XNN_NO_F32_OPERATORS

    /*************************** VCVT AArch32 micro-kernels ***************************/
    #ifndef XNN_NO_VCVT_OPERATORS
      init_flags |= XNN_INIT_FLAG_VCVT;

      if (cpuinfo_has_arm_neon_fp16()) {
        xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__neonfp16_x16,
          .element_tile = 16,
        };
        xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__neonfp16_x16,
          .element_tile = 16,
        };
      } else {
        xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__neon_int16_x16,
          .init.f16_f32_cvt = xnn_init_f16_f32_cvt_neon_params,
          .element_tile = 16,
        };
        xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__neon_x8,
          .init.f32_f16_cvt = xnn_init_f32_f16_cvt_neon_params,
          .element_tile = 8,
        };
      }
      if (cpuinfo_has_arm_neon_v8()) {
        xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_qs8_vcvt_ukernel__neonv8_x32,
          .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_neonv8_params,
          .element_tile = 32,
        };
        xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__neonv8_x32,
          .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_neonv8_params,
          .element_tile = 32,
        };
      } else {
        xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_qs8_vcvt_ukernel__neon_x32,
          .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_neon_params,
          .element_tile = 32,
        };
        xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__neon_x32,
          .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_neon_params,
          .element_tile = 32,
        };
      }
      xnn_params.vcvt.qs8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qs8_f32_vcvt_ukernel__neon_x32,
        .init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_neon_params,
        .element_tile = 32,
      };
      xnn_params.vcvt.qu8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qu8_f32_vcvt_ukernel__neon_x32,
        .init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_neon_params,
        .element_tile = 32,
      };
    #endif  // XNN_NO_VCVT_OPERATORS

    /**************************** X32 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_X32_OPERATORS
      init_flags |= XNN_INIT_FLAG_X32;

      xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__neon;
      xnn_params.x32.zip = (struct zip_parameters) {
        .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__neon,
        .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__neon,
        .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__neon,
        .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__neon,
      };
      #ifndef XNN_NO_NCHW_OPERATORS
        xnn_params.x32.depthtospace2d_chw2hwc = (struct depthtospace2d_chw2hwc_parameters) {
          .ukernel = (xnn_depthtospace2d_chw2hwc_ukernel_function) xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar,
          .channel_tile = 1,
          .pixel_tile = 1,
        };
      #endif  // XNN_NO_NCHW_OPERATORS
    #endif  // XNN_NO_X32_OPERATORS

    /**************************** XX AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_XX_OPERATORS
      init_flags |= XNN_INIT_FLAG_XX;

      xnn_params.xx.copy = (xnn_univector_ukernel_function) xnn_xx_copy_ukernel__memcpy;
      xnn_params.xx.fill = (struct fill_parameters) {
        .ukernel = (xnn_fill_ukernel_function) xnn_xx_fill_ukernel__neon_x64,
        .row_tile = 1,
      };
      xnn_params.xx.pad = (struct pad_parameters) {
        .ukernel = (xnn_pad_ukernel_function) xnn_xx_pad_ukernel__neon,
        .row_tile = 1,
      };
    #endif  // XNN_NO_XX_OPERATORS

  } else if (!XNN_PLATFORM_MOBILE) {

    /*************************** QS8 AArch32 Pre-NEON micro-kernels ***************************/
    #ifndef XNN_NO_QS8_OPERATORS
      init_flags |= XNN_INIT_FLAG_QS8;

      xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic);
      xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_2x2__scalar_fmagic);
      xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic);
      xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic);
      xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qs8.gemm.mr = 2;
      xnn_params.qs8.gemm.nr = 2;

      xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up1x9__scalar_fmagic;
      xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qs8.dwconv[0].channel_tile = 1;
      xnn_params.qs8.dwconv[0].primary_tile = 9;
      xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up1x25__scalar_fmagic;
      xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qs8.dwconv[1].channel_tile = 1;
      xnn_params.qs8.dwconv[1].primary_tile = 25;

      xnn_params.qs8.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1,
        .init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params,
        .update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_scalar_imagic_params,
        .row_tile = 7,
        .channel_tile = 1,
      };
      xnn_params.qs8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__scalar_x1,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__scalar_x1,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__scalar_x1,
        .init.qs8_addsub = xnn_init_qs8_add_minmax_scalar_params,
        .element_tile = 1,
      };
      xnn_params.qs8.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmul_minmax_fp32_ukernel__scalar_x4,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_x4,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_x4,
        .init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_scalar_params,
        .element_tile = 4,
      };
    #endif  // XNN_NO_QS8_OPERATORS

    /*************************** QU8 AArch32 Pre-NEON micro-kernels ***************************/
    #ifndef XNN_NO_QU8_OPERATORS
      init_flags |= XNN_INIT_FLAG_QU8;

      xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic);
      xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_fmagic);
      xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic);
      xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic);
      xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qu8.gemm.mr = 2;
      xnn_params.qu8.gemm.nr = 2;

      xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up1x9__scalar_fmagic;
      xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qu8.dwconv[0].channel_tile = 1;
      xnn_params.qu8.dwconv[0].primary_tile = 9;
      xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up1x25__scalar_fmagic;
      xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qu8.dwconv[1].channel_tile = 1;
      xnn_params.qu8.dwconv[1].primary_tile = 25;

      xnn_params.qu8.avgpool = (struct avgpool_parameters) {
        .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9x__scalar_c1,
        .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9p8x__scalar_c1,
        .init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params,
        .primary_tile = 9,
        .incremental_tile = 8,
        .channel_tile = 1,
      };
      xnn_params.qu8.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1,
        .init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params,
        .update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_scalar_imagic_params,
        .row_tile = 7,
        .channel_tile = 1,
      };
      xnn_params.qu8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__scalar_x1,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__scalar_x1,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__scalar_x1,
        .init.qu8_addsub = xnn_init_qu8_add_minmax_scalar_params,
        .element_tile = 1,
      };
      xnn_params.qu8.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmul_minmax_fp32_ukernel__scalar_x4,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_x4,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_x4,
        .init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_scalar_params,
        .element_tile = 4,
      };
    #endif  // XNN_NO_QU8_OPERATORS

    /**************************** S8 AArch32 Pre-NEON micro-kernels ****************************/
    #ifndef XNN_NO_S8_OPERATORS
      init_flags |= XNN_INIT_FLAG_S8;

      xnn_params.s8.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_s8_vclamp_ukernel__scalar_x4,
        .init.s8_minmax = xnn_init_s8_minmax_scalar_params,
        .element_tile = 4,
      };
      xnn_params.s8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_s8_ibilinear_ukernel__scalar_c1,
        .pixel_tile = 1,
        .channel_tile = 1,
      };
      xnn_params.s8.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1,
        .init.s8 = xnn_init_s8_minmax_scalar_params,
        .mr = 9,
        .qr = 8,
      };
    #endif  // XNN_NO_S8_OPERATORS

    /**************************** U8 AArch32 Pre-NEON micro-kernels ****************************/
    #ifndef XNN_NO_U8_OPERATORS
      init_flags |= XNN_INIT_FLAG_U8;

      xnn_params.u8.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_u8_vclamp_ukernel__scalar_x4,
        .init.u8_minmax = xnn_init_u8_minmax_scalar_params,
        .element_tile = 4,
      };
      xnn_params.u8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_u8_ibilinear_ukernel__scalar_c1,
        .pixel_tile = 1,
        .channel_tile = 1,
      };
      xnn_params.u8.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1,
        .init.u8 = xnn_init_u8_minmax_scalar_params,
        .mr = 9,
        .qr = 8,
      };
      xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
      xnn_params.u8.rmax = xnn_u8_rmax_ukernel__scalar;
    #endif  // XNN_NO_U8_OPERATORS

    /**************************** X8 AArch32 Pre-NEON micro-kernels ****************************/
    #ifndef XNN_NO_X8_OPERATORS
      init_flags |= XNN_INIT_FLAG_X8;

      xnn_params.x8.lut = xnn_x8_lut_ukernel__scalar_x4;
      xnn_params.x8.zip = (struct zip_parameters) {
        .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__scalar,
        .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__scalar,
        .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__scalar,
        .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__scalar,
      };
    #endif  // XNN_NO_X8_OPERATORS

    /**************************** F32 AArch32 Pre-NEON micro-kernels ****************************/
    #ifndef XNN_NO_F32_OPERATORS
      init_flags |= XNN_INIT_FLAG_F32;

      xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x4__scalar);
      xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x4__scalar);
      xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x4__scalar);
      xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x4__scalar);
      xnn_params.f32.gemm.relu.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_4x4__scalar);
      xnn_params.f32.gemm.relu.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_4x4__scalar);
      xnn_params.f32.gemm.relu.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_1x4__scalar);
      xnn_params.f32.gemm.relu.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_1x4__scalar);
      xnn_params.f32.gemm.linear.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x4__scalar);
      xnn_params.f32.gemm.linear.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x4__scalar);
      xnn_params.f32.gemm.linear.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x4__scalar);
      xnn_params.f32.gemm.linear.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x4__scalar);
      xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.gemm.mr = 4;
      xnn_params.f32.gemm.nr = 4;

      xnn_params.f32.gemm2.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x2__scalar);
      xnn_params.f32.gemm2.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x2__scalar),
      xnn_params.f32.gemm2.linear.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x2__scalar);
      xnn_params.f32.gemm2.linear.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2__scalar),
      xnn_params.f32.gemm2.init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.gemm2.mr = 4;
      xnn_params.f32.gemm2.nr = 2;

      xnn_params.f32.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x3__scalar_acc2;
      xnn_params.f32.dwconv[0].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x3__scalar_acc2;
      xnn_params.f32.dwconv[0].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[0].channel_tile = 1;
      xnn_params.f32.dwconv[0].primary_tile = 3;

      xnn_params.f32.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2;
      xnn_params.f32.dwconv[1].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x4__scalar_acc2;
      xnn_params.f32.dwconv[1].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[1].channel_tile = 1;
      xnn_params.f32.dwconv[1].primary_tile = 4;

      xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2;
      xnn_params.f32.dwconv[2].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x9__scalar_acc2;
      xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[2].channel_tile = 1;
      xnn_params.f32.dwconv[2].primary_tile = 9;

      xnn_params.f32.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2;
      xnn_params.f32.dwconv[3].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x25__scalar_acc2;
      xnn_params.f32.dwconv[3].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[3].channel_tile = 1;
      xnn_params.f32.dwconv[3].primary_tile = 25;

      xnn_params.f32.avgpool = (struct avgpool_parameters) {
        .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9x__scalar_c1,
        .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9p8x__scalar_c1,
        .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
        .primary_tile = 9,
        .incremental_tile = 8,
        .channel_tile = 1,
      };
      xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
        .unipass = (xnn_pavgpool_unipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9x__scalar_c1,
        .multipass = (xnn_pavgpool_multipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9p8x__scalar_c1,
        .primary_tile = 9,
        .incremental_tile = 8,
        .channel_tile = 1,
      };
      xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1,
        .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
        .update.f32 = xnn_update_f32_scaleminmax_scalar_params,
        .row_tile = 7,
        .channel_tile = 1,
      };
      xnn_params.f32.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1,
        .init.f32 = xnn_init_f32_minmax_scalar_params,
        .mr = 9,
        .qr = 8,
      };
      xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
        .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__scalar_c1,
        .mr = 4,
      };
      xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
        .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__scalar_c1,
        .mr = 9,
      };
      xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
        .mp = (xnn_argmaxpool_multipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1,
        .mr = 9,
        .qr = 8,
      };
      xnn_params.f32.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_f32_ibilinear_ukernel__scalar_c2,
        .pixel_tile = 1,
        .channel_tile = 2,
      };
      xnn_params.f32.abs = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vabs_ukernel__scalar_x4,
        .element_tile = 4,
      };
      xnn_params.f32.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vclamp_ukernel__scalar_x4,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 4,
      };
      xnn_params.f32.elu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4,
        .init.f32_elu = xnn_init_f32_elu_scalar_rr2_lut16_p3_params,
        .element_tile = 4,
      };
      xnn_params.f32.hswish = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vhswish_ukernel__scalar_x4,
        .init.f32_hswish = xnn_init_f32_hswish_scalar_params,
        .element_tile = 4,
      };
      xnn_params.f32.lrelu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vlrelu_ukernel__scalar_x4,
        .init.f32_lrelu = xnn_init_f32_lrelu_scalar_params,
        .element_tile = 4,
      };
      xnn_params.f32.neg = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vneg_ukernel__scalar_x4,
        .element_tile = 4,
      };
      xnn_params.f32.rndne = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__scalar_libm_x1,
        .element_tile = 1,
      };
      xnn_params.f32.rndz = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__scalar_libm_x1,
        .element_tile = 1,
      };
      xnn_params.f32.rndu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__scalar_libm_x1,
        .element_tile = 1,
      };
      xnn_params.f32.rndd = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__scalar_libm_x1,
        .element_tile = 1,
      };
      xnn_params.f32.sigmoid = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x2,
        .init.f32_sigmoid = xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params,
        .element_tile = 2,
      };
      xnn_params.f32.sqr = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqr_ukernel__scalar_x4,
        .element_tile = 4,
      };
      xnn_params.f32.sqrt = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqrt_ukernel__scalar_sqrt_x1,
        .element_tile = 1,
      };
      xnn_params.f32.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__scalar_2x4,
        .row_tile = 4,
        .channel_tile = 4,
      };
      xnn_params.f32.raddstoreexpminusmax = (struct raddstoreexpminusmax_parameters) {
        .ukernel = xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x4_acc2,
        .init = xnn_init_f32_expminus_scalar_rr2_p5_params,
        .element_tile = 4,
      };
      xnn_params.f32.rmax = xnn_f32_rmax_ukernel__scalar;
      xnn_params.f32.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_minmax_ukernel__scalar_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__scalar_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__scalar_x8,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 8,
      };
      xnn_params.f32.vdiv = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_minmax_ukernel__scalar_x2,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_minmax_ukernel__scalar_x2,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_minmax_ukernel__scalar_x2,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 2,
      };
      xnn_params.f32.vmax = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__scalar_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__scalar_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__scalar_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vmin = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__scalar_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__scalar_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__scalar_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_minmax_ukernel__scalar_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__scalar_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__scalar_x8,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 8,
      };
      xnn_params.f32.vsub = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_minmax_ukernel__scalar_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_minmax_ukernel__scalar_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_minmax_ukernel__scalar_x8,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 8,
      };
      xnn_params.f32.vsqrdiff = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiff_ukernel__scalar_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__scalar_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__scalar_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
        .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x,
        .init.f32 = xnn_init_f32_minmax_scalar_params,
        .channel_tile = 1,
        .row_tile = 2,
      };
      #ifndef XNN_NO_NCHW_OPERATORS
        init_flags |= XNN_INIT_FLAG_CHW_OPT;

        xnn_params.f32.spmm = (struct spmm_parameters) {
          .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_8x1__scalar,
          .mr = 8,
          .nr = 1,
        };
        xnn_params.f32.spmm2 = (struct spmm_parameters) {
          .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_8x2__scalar,
          .mr = 8,
          .nr = 2,
        };
        xnn_params.f32.spmm4 = (struct spmm_parameters) {
          .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_8x4__scalar,
          .mr = 8,
          .nr = 4,
        };
        xnn_params.f32.conv_hwc2chw_3x3c3s2 = (struct conv_hwc2chw_parameters) {
          .ukernel_with_symm_padding =
            (xnn_conv_hwc2chw_ukernel_function) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1,
          .output_channel_tile = 4,
          .output_height_tile = 1,
          .output_width_tile = 1,
        };
        xnn_params.f32.dwconv2d_chw_3x3 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_4x1,
          .output_width_tile = 1,
          .output_height_tile = 4,
        };
        xnn_params.f32.dwconv2d_chw_3x3s2 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_2x1_acc2,
          .output_width_tile = 1,
          .output_height_tile = 2,
        };
        xnn_params.f32.dwconv2d_chw_5x5 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_2x1_acc2,
          .output_width_tile = 1,
          .output_height_tile = 2,
        };
        xnn_params.f32.dwconv2d_chw_5x5s2 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_2x1_acc2,
          .output_width_tile = 1,
          .output_height_tile = 2,
        };
        xnn_params.f32.gavgpool_cw = (struct gavgpool_cw_parameters) {
          .ukernel = (xnn_gavgpool_cw_ukernel_function) xnn_f32_gavgpool_cw_ukernel__scalar_x1,
          .channel_tile = 1,
        };
        xnn_params.f32.ibilinear_chw = (struct ibilinear_chw_parameters) {
          .ukernel = (xnn_ibilinear_chw_ukernel_function) xnn_f32_ibilinear_chw_ukernel__scalar_p4,
          .channel_tile = 1,
          .pixel_tile = 4,
        };
      #endif  // XNN_NO_NCHW_OPERATORS
    #endif  // XNN_NO_F32_OPERATORS

    /*************************** VCVT AArch32 Pre-NEON micro-kernels ***************************/
    #ifndef XNN_NO_VCVT_OPERATORS
      init_flags |= XNN_INIT_FLAG_VCVT;

      xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__scalar_x4,
        .init.f16_f32_cvt = xnn_init_f16_f32_cvt_scalar_params,
        .element_tile = 4,
      };
      xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__scalar_fabsf_x2,
        .init.f32_f16_cvt = xnn_init_f32_f16_cvt_scalar_fabsf_params,
        .element_tile = 2,
      };
      xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qs8_vcvt_ukernel__scalar_imagic_x4,
        .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_imagic_params,
        .element_tile = 4,
      };
      xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__scalar_imagic_x4,
        .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_imagic_params,
        .element_tile = 4,
      };
      xnn_params.vcvt.qs8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qs8_f32_vcvt_ukernel__scalar_x4,
        .init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params,
        .element_tile = 4,
      };
      xnn_params.vcvt.qu8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qu8_f32_vcvt_ukernel__scalar_x4,
        .init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params,
        .element_tile = 4,
      };
    #endif  // XNN_NO_VCVT_OPERATORS

    /**************************** X32 AArch32 Pre-NEON micro-kernels ****************************/
    #ifndef XNN_NO_X32_OPERATORS
      init_flags |= XNN_INIT_FLAG_X32;

      xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__scalar;
      xnn_params.x32.zip = (struct zip_parameters) {
        .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__scalar,
        .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__scalar,
        .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__scalar,
        .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__scalar,
      };
      #ifndef XNN_NO_NCHW_OPERATORS
        xnn_params.x32.depthtospace2d_chw2hwc = (struct depthtospace2d_chw2hwc_parameters) {
          .ukernel = (xnn_depthtospace2d_chw2hwc_ukernel_function) xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar,
          .channel_tile = 1,
          .pixel_tile = 1,
        };
      #endif  // XNN_NO_NCHW_OPERATORS
    #endif  // XNN_NO_X32_OPERATORS

    /**************************** XX AArch32 Pre-NEON micro-kernels ****************************/
    #ifndef XNN_NO_XX_OPERATORS
      init_flags |= XNN_INIT_FLAG_XX;

      xnn_params.xx.copy = (xnn_univector_ukernel_function) xnn_xx_copy_ukernel__memcpy;
      xnn_params.xx.fill = (struct fill_parameters) {
        .ukernel = (xnn_fill_ukernel_function) xnn_xx_fill_ukernel__scalar_x16,
        .row_tile = 1,
      };
      xnn_params.xx.pad = (struct pad_parameters) {
        .ukernel = (xnn_pad_ukernel_function) xnn_xx_pad_ukernel__scalar,
        .row_tile = 1,
      };
    #endif  // XNN_NO_XX_OPERATORS
  }

#elif XNN_ARCH_ARM64

  /**************************** QC8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_QC8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QC8;

    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        if (cpuinfo_has_arm_neon_dot()) {
          xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__aarch64_neondot_ld128);
          xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c4__neondot);
          xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__aarch64_neondot_ld128);
          xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c4__neondot);
          xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
          xnn_params.qc8.gemm.mr = 4;
          xnn_params.qc8.gemm.nr = 16;
          xnn_params.qc8.gemm.log2_kr = 2;
        } else {
          xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c8__aarch64_neon_mlal);
          xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c8__aarch64_neon_mlal);
          xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c8__aarch64_neon_mlal);
          xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c8__aarch64_neon_mlal);
          xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
          xnn_params.qc8.gemm.mr = 2;
          xnn_params.qc8.gemm.nr = 8;
          xnn_params.qc8.gemm.log2_kr = 3;
        }
      #else  // !XNN_ENABLE_ASSEMBLY
        if (cpuinfo_has_arm_neon_dot()) {
          xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__neondot);
          xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c4__neondot);
          xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__neondot);
          xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c4__neondot);
          xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
          xnn_params.qc8.gemm.mr = 4;
          xnn_params.qc8.gemm.nr = 16;
          xnn_params.qc8.gemm.log2_kr = 2;
        } else {
          xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
          xnn_params.qc8.gemm.mr = 2;
          xnn_params.qc8.gemm.nr = 8;
          xnn_params.qc8.gemm.log2_kr = 1;
          xnn_params.qc8.gemm.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        if (cpuinfo_has_arm_neon_dot()) {
          switch (cpuinfo_get_core(0)->uarch) {
            case cpuinfo_uarch_cortex_a55:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__aarch64_neondot_cortex_a55);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__aarch64_neondot_cortex_a55);
              break;
            case cpuinfo_uarch_cortex_x1:
            case cpuinfo_uarch_cortex_a78:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__aarch64_neondot_ld128);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__aarch64_neondot_ld128);
              break;
            default:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__aarch64_neondot_ld64);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__aarch64_neondot_ld64);
              break;
          }
          xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c4__neondot);
          xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c4__neondot);
          xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
          xnn_params.qc8.gemm.mr = 4;
          xnn_params.qc8.gemm.nr = 16;
          xnn_params.qc8.gemm.log2_kr = 2;
        } else {
          switch (cpuinfo_get_core(0)->uarch) {
            case cpuinfo_uarch_cortex_a35:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x16__aarch64_neon_mlal_lane_ld64);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x16__aarch64_neon_mlal_lane_ld64);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
              xnn_params.qc8.gemm.mr = 4;
              xnn_params.qc8.gemm.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a55r0:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
              xnn_params.qc8.gemm.mr = 4;
              xnn_params.qc8.gemm.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a72:
            case cpuinfo_uarch_cortex_a73:
            case cpuinfo_uarch_kryo:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c8__aarch64_neon_mlal_prfm);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c8__aarch64_neon_mlal_prfm);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c8__aarch64_neon_mlal_prfm);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c8__aarch64_neon_mlal_prfm);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
              xnn_params.qc8.gemm.mr = 2;
              xnn_params.qc8.gemm.nr = 8;
              xnn_params.qc8.gemm.log2_kr = 3;
              break;

            default:
              xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c8__aarch64_neon_mlal);
              xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c8__aarch64_neon_mlal);
              xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c8__aarch64_neon_mlal);
              xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c8__aarch64_neon_mlal);
              xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
              xnn_params.qc8.gemm.mr = 2;
              xnn_params.qc8.gemm.nr = 8;
              xnn_params.qc8.gemm.log2_kr = 3;
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = xnn_params.qc8.gemm.mr;
          const uint32_t nr = xnn_params.qc8.gemm.nr;
          const uint32_t log2_kr = xnn_params.qc8.gemm.log2_kr;
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
                  xnn_params.qc8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c8__aarch64_neon_mlal_prfm_cortex_a53;
                  xnn_params.qc8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c8__aarch64_neon_mlal_prfm_cortex_a53;
                  xnn_params.qc8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c8__aarch64_neon_mlal_prfm_cortex_a53;
                  xnn_params.qc8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c8__aarch64_neon_mlal_prfm_cortex_a53;
                }
                break;

              case cpuinfo_uarch_cortex_a55:
                if (mr == 4 && nr == 16 && log2_kr == 2 && cpuinfo_has_arm_neon_dot()) {
                  xnn_params.qc8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__aarch64_neondot_cortex_a55;
                  xnn_params.qc8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__aarch64_neondot_cortex_a55;
                  xnn_params.qc8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c4__neondot;
                  xnn_params.qc8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c4__neondot;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // !XNN_ENABLE_ASSEMBLY
        if (cpuinfo_has_arm_neon_dot()) {
          xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c4__neondot);
          xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c4__neondot);
          xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c4__neondot);
          xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c4__neondot);
          xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
          xnn_params.qc8.gemm.mr = 4;
          xnn_params.qc8.gemm.nr = 16;
          xnn_params.qc8.gemm.log2_kr = 2;
        } else {
          xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_neonv8_params;
          xnn_params.qc8.gemm.mr = 2;
          xnn_params.qc8.gemm.nr = 8;
          xnn_params.qc8.gemm.log2_kr = 1;
          xnn_params.qc8.gemm.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC

    xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neonv8_mla8_ld64;
    xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_neonv8_params;
    xnn_params.qc8.dwconv[0].channel_tile = 16;
    xnn_params.qc8.dwconv[0].primary_tile = 9;
    xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__neonv8_mla8_ld64;
    xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_neonv8_params;
    xnn_params.qc8.dwconv[1].channel_tile = 16;
    xnn_params.qc8.dwconv[1].primary_tile = 25;
  #endif  // XNN_NO_QC8_OPERATORS

  /**************************** QS8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QS8;

    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        if (cpuinfo_has_arm_neon_dot()) {
          xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld128);
          xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
          xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld128);
          xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
          xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          xnn_params.qs8.gemm.mr = 4;
          xnn_params.qs8.gemm.nr = 16;
          xnn_params.qs8.gemm.log2_kr = 2;
        } else {
          xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__aarch64_neon_mlal);
          xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__aarch64_neon_mlal);
          xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__aarch64_neon_mlal);
          xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neon_mlal);
          xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          xnn_params.qs8.gemm.mr = 2;
          xnn_params.qs8.gemm.nr = 8;
          xnn_params.qs8.gemm.log2_kr = 3;
        }
      #else  // !XNN_ENABLE_ASSEMBLY
        if (cpuinfo_has_arm_neon_dot()) {
          xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neondot);
          xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
          xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot);
          xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
          xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          xnn_params.qs8.gemm.mr = 4;
          xnn_params.qs8.gemm.nr = 16;
          xnn_params.qs8.gemm.log2_kr = 2;
        } else {
          xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          xnn_params.qs8.gemm.mr = 2;
          xnn_params.qs8.gemm.nr = 8;
          xnn_params.qs8.gemm.log2_kr = 1;
          xnn_params.qs8.gemm.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        if (cpuinfo_has_arm_neon_dot()) {
          switch (cpuinfo_get_core(0)->uarch) {
            case cpuinfo_uarch_cortex_a55:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_cortex_a55);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_cortex_a55);
              break;
            case cpuinfo_uarch_cortex_x1:
            case cpuinfo_uarch_cortex_a78:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld128);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld128);
              break;
            default:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld64);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld64);
              break;
          }
          xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
          xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
          xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          xnn_params.qs8.gemm.mr = 4;
          xnn_params.qs8.gemm.nr = 16;
          xnn_params.qs8.gemm.log2_kr = 2;
        } else {
          switch (cpuinfo_get_core(0)->uarch) {
            case cpuinfo_uarch_cortex_a35:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 4;
              xnn_params.qs8.gemm.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a55r0:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 4;
              xnn_params.qs8.gemm.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a72:
            case cpuinfo_uarch_cortex_a73:
            case cpuinfo_uarch_kryo:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__aarch64_neon_mlal_prfm);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__aarch64_neon_mlal_prfm);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__aarch64_neon_mlal_prfm);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neon_mlal_prfm);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 2;
              xnn_params.qs8.gemm.nr = 8;
              xnn_params.qs8.gemm.log2_kr = 3;
              break;

            default:
              xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__aarch64_neon_mlal);
              xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__aarch64_neon_mlal);
              xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__aarch64_neon_mlal);
              xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neon_mlal);
              xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
              xnn_params.qs8.gemm.mr = 2;
              xnn_params.qs8.gemm.nr = 8;
              xnn_params.qs8.gemm.log2_kr = 3;
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = xnn_params.qs8.gemm.mr;
          const uint32_t nr = xnn_params.qs8.gemm.nr;
          const uint32_t log2_kr = xnn_params.qs8.gemm.log2_kr;
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
                  xnn_params.qs8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__aarch64_neon_mlal_prfm_cortex_a53;
                  xnn_params.qs8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__aarch64_neon_mlal_prfm_cortex_a53;
                  xnn_params.qs8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__aarch64_neon_mlal_prfm_cortex_a53;
                  xnn_params.qs8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neon_mlal_prfm_cortex_a53;
                }
                break;

              case cpuinfo_uarch_cortex_a55:
                if (mr == 4 && nr == 16 && log2_kr == 2 && cpuinfo_has_arm_neon_dot()) {
                  xnn_params.qs8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_cortex_a55;
                  xnn_params.qs8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_cortex_a55;
                  xnn_params.qs8.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot;
                  xnn_params.qs8.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neondot;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // !XNN_ENABLE_ASSEMBLY
        if (cpuinfo_has_arm_neon_dot()) {
          xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neondot);
          xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
          xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot);
          xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
          xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          xnn_params.qs8.gemm.mr = 4;
          xnn_params.qs8.gemm.nr = 16;
          xnn_params.qs8.gemm.log2_kr = 2;
        } else {
          xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal);
          xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
          xnn_params.qs8.gemm.mr = 2;
          xnn_params.qs8.gemm.nr = 8;
          xnn_params.qs8.gemm.log2_kr = 1;
          xnn_params.qs8.gemm.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC

    xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64;
    xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
    xnn_params.qs8.dwconv[0].channel_tile = 16;
    xnn_params.qs8.dwconv[0].primary_tile = 9;
    xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64;
    xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_rndnu_neon_params;
    xnn_params.qs8.dwconv[1].channel_tile = 16;
    xnn_params.qs8.dwconv[1].primary_tile = 25;

    xnn_params.qs8.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qs8_gavgpool_minmax_rndnu_ukernel_7x__neon_c8,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qs8_gavgpool_minmax_rndnu_ukernel_7p7x__neon_c8,
      .init.qs8 = xnn_init_qs8_avgpool_minmax_rndnu_neon_params,
      .update.qs8 = xnn_update_qs8_avgpool_minmax_rndnu_neon_params,
      .row_tile = 7,
      .channel_tile = 8,
    };

    xnn_params.qs8.vadd = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__neon_ld64_x32,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32,
      .init.qs8_addsub = xnn_init_qs8_add_minmax_neon_params,
      .element_tile = 32,
    };
    xnn_params.qs8.vmul = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmul_minmax_rndnu_ukernel__neon_ld64_x16,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_rndnu_ukernel__neon_ld64_x16,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_rndnu_ukernel__neon_ld64_x16,
      .init.qs8_mul = xnn_init_qs8_mul_minmax_rndnu_neon_params,
      .element_tile = 16,
    };
  #endif  // XNN_NO_QS8_OPERATORS

  /**************************** QU8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_QU8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QU8;

    #if XNN_ENABLE_ASSEMBLY
      if (cpuinfo_has_arm_neon_dot()) {
        switch (cpuinfo_get_core(0)->uarch) {
          case cpuinfo_uarch_cortex_a55:
            xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_cortex_a55);
            xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_cortex_a55);
            xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
            xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
            xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            xnn_params.qu8.gemm.mr = 4;
            xnn_params.qu8.gemm.nr = 16;
            xnn_params.qu8.gemm.log2_kr = 2;
            break;
          default:
            xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld128);
            xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld128);
            xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
            xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
            xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            xnn_params.qu8.gemm.mr = 4;
            xnn_params.qu8.gemm.nr = 16;
            xnn_params.qu8.gemm.log2_kr = 2;
            break;
        }
      } else {
        switch (cpuinfo_get_core(0)->uarch) {
          case cpuinfo_uarch_cortex_a53:
          case cpuinfo_uarch_cortex_a55r0:
            xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53);
            xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53);
            xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            xnn_params.qu8.gemm.mr = 4;
            xnn_params.qu8.gemm.nr = 16;
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
            xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75);
            xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75);
            xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            xnn_params.qu8.gemm.mr = 4;
            xnn_params.qu8.gemm.nr = 16;
            break;

          case cpuinfo_uarch_kryo:
            xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane);
            xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane);
            xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            xnn_params.qu8.gemm.mr = 4;
            xnn_params.qu8.gemm.nr = 16;
            break;

          default:
            xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a75);
            xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a75);
            xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
            xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            xnn_params.qu8.gemm.mr = 4;
            xnn_params.qu8.gemm.nr = 16;
            break;
        }
      }
      #if XNN_MAX_UARCH_TYPES > 1
      {
        /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
        const uint32_t mr = xnn_params.qu8.gemm.mr;
        const uint32_t nr = xnn_params.qu8.gemm.nr;
        const uint32_t log2_kr = xnn_params.qu8.gemm.log2_kr;
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
                xnn_params.qu8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53;
                xnn_params.qu8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53;
              }
              break;

            case cpuinfo_uarch_cortex_a55:
              if (mr == 4 && nr == 16 && log2_kr == 2 && cpuinfo_has_arm_neon_dot()) {
                xnn_params.qu8.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_cortex_a55;
                xnn_params.qu8.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_cortex_a55;
              }
              break;
            default:
              break;
          }
        }
      }
      #endif  // XNN_MAX_UARCH_TYPES > 1
    #else  // !XNN_ENABLE_ASSEMBLY
      if (cpuinfo_has_arm_neon_dot()) {
        xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot);
        xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__neondot);
        xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot);
        xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot);
        xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
        xnn_params.qu8.gemm.mr = 4;
        xnn_params.qu8.gemm.nr = 16;
        xnn_params.qu8.gemm.log2_kr = 2;
      } else {
        xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane);
        xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane);
        xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
        xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
        xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
        xnn_params.qu8.gemm.mr = 4;
        xnn_params.qu8.gemm.nr = 16;
      }
    #endif  // XNN_ENABLE_ASSEMBLY

    xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8;
    xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
    xnn_params.qu8.dwconv[0].channel_tile = 16;
    xnn_params.qu8.dwconv[0].primary_tile = 9;
    xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8;
    xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
    xnn_params.qu8.dwconv[1].channel_tile = 8;
    xnn_params.qu8.dwconv[1].primary_tile = 25;

    xnn_params.qu8.avgpool = (struct avgpool_parameters) {
      .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9x__neon_c8,
      .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9p8x__neon_c8,
      .init.qu8 = xnn_init_qu8_avgpool_minmax_neon_params,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 8,
    };
    xnn_params.qu8.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qu8_gavgpool_minmax_rndnu_ukernel_7x__neon_c8,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qu8_gavgpool_minmax_rndnu_ukernel_7p7x__neon_c8,
      .init.qu8 = xnn_init_qu8_avgpool_minmax_rndnu_neon_params,
      .update.qu8 = xnn_update_qu8_avgpool_minmax_rndnu_neon_params,
      .row_tile = 7,
      .channel_tile = 8,
    };
    xnn_params.qu8.vadd = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__neon_ld64_x32,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__neon_ld64_x32,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__neon_ld64_x32,
      .init.qu8_addsub = xnn_init_qu8_add_minmax_neon_params,
      .element_tile = 8,
    };
    xnn_params.qu8.vmul = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmul_minmax_rndnu_ukernel__neon_ld64_x16,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_x16,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_x16,
      .init.qu8_mul = xnn_init_qu8_mul_minmax_rndnu_neon_params,
      .element_tile = 16,
    };
  #endif  // XNN_NO_QU8_OPERATORS

  /**************************** S8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_S8_OPERATORS
    init_flags |= XNN_INIT_FLAG_S8;

    xnn_params.s8.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_s8_vclamp_ukernel__neon_x64,
      .init.s8_minmax = xnn_init_s8_minmax_neon_params,
      .element_tile = 64,
    };
    xnn_params.s8.ibilinear = (struct ibilinear_parameters) {
      .ukernel = (xnn_ibilinear_ukernel_function) xnn_s8_ibilinear_ukernel__neon_c16,
      .pixel_tile = 1,
      .channel_tile = 16,
    };
    xnn_params.s8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_s8_maxpool_minmax_ukernel_9p8x__neon_c16,
      .init.s8 = xnn_init_s8_minmax_neon_params,
      .mr = 9,
      .qr = 8,
    };
  #endif  // XNN_NO_S8_OPERATORS

  /**************************** U8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_U8_OPERATORS
    init_flags |= XNN_INIT_FLAG_U8;

    xnn_params.u8.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_u8_vclamp_ukernel__neon_x64,
      .init.u8_minmax = xnn_init_u8_minmax_neon_params,
      .element_tile = 64,
    };
    xnn_params.u8.ibilinear = (struct ibilinear_parameters) {
      .ukernel = (xnn_ibilinear_ukernel_function) xnn_u8_ibilinear_ukernel__neon_c16,
      .pixel_tile = 1,
      .channel_tile = 16,
    };
    xnn_params.u8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16,
      .init.u8 = xnn_init_u8_minmax_neon_params,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    xnn_params.u8.rmax = xnn_u8_rmax_ukernel__neon;
  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_X8_OPERATORS
    init_flags |= XNN_INIT_FLAG_X8;

    xnn_params.x8.lut = xnn_x8_lut_ukernel__neon_tbx128x4_x64;
    xnn_params.x8.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__neon,
      .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__neon,
      .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__neon,
      .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__neon,
    };
  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F16 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_F16_OPERATORS
    if (cpuinfo_has_arm_neon_fp16_arith()) {
      init_flags |= XNN_INIT_FLAG_F16 | XNN_INIT_FLAG_F16_NATIVE;
      xnn_params.f16.gemm.mr = 6;
      xnn_params.f16.gemm.nr = 16;

      #if XNN_ENABLE_ASSEMBLY
        switch (cpuinfo_get_core(0)->uarch) {
          case cpuinfo_uarch_cortex_a55:
            xnn_params.f16.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55);
            break;

          case cpuinfo_uarch_cortex_a75:
          case cpuinfo_uarch_cortex_x1:
            xnn_params.f16.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a75);
            break;

          default:
            xnn_params.f16.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
            break;
        }
        xnn_params.f16.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);

        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = xnn_params.f16.gemm.mr;
          const uint32_t nr = xnn_params.f16.gemm.nr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                if (mr == 6 && nr == 16) {
                  xnn_params.f16.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55;
                }
                break;

              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 6 && nr == 16) {
                  xnn_params.f16.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64;
                }
                break;

              /* Cortex A75 is the medium core Exynos 9820 (M4) */
              case cpuinfo_uarch_cortex_a75:
                if (mr == 6 && nr == 16) {
                  xnn_params.f16.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a75;
                }
                break;

              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        xnn_params.f16.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64);
        xnn_params.f16.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64);
      #endif  // XNN_ENABLE_ASSEMBLY
      xnn_params.f16.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64);
      xnn_params.f16.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64);
      xnn_params.f16.gemm.init.f16 = xnn_init_f16_scaleminmax_neon_params;

      xnn_params.f16.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith;
      xnn_params.f16.dwconv[0].init.f16 = xnn_init_f16_minmax_neon_params;
      xnn_params.f16.dwconv[0].channel_tile = 16;
      xnn_params.f16.dwconv[0].primary_tile = 3;

      xnn_params.f16.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith;
      xnn_params.f16.dwconv[1].init.f16 = xnn_init_f16_minmax_neon_params;
      xnn_params.f16.dwconv[1].channel_tile = 16;
      xnn_params.f16.dwconv[1].primary_tile = 4;

      xnn_params.f16.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith;
      xnn_params.f16.dwconv[2].init.f16 = xnn_init_f16_minmax_neon_params;
      xnn_params.f16.dwconv[2].channel_tile = 16;
      xnn_params.f16.dwconv[2].primary_tile = 9;

      xnn_params.f16.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2;
      xnn_params.f16.dwconv[3].init.f16 = xnn_init_f16_minmax_neon_params;
      xnn_params.f16.dwconv[3].channel_tile = 8;
      xnn_params.f16.dwconv[3].primary_tile = 25;

      xnn_params.f16.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8,
        .init.f16 = xnn_init_f16_scaleminmax_neon_params,
        .update.f16 = xnn_update_f16_scaleminmax_neon_params,
        .row_tile = 7,
        .channel_tile = 8,
      };

      xnn_params.f16.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_f16_maxpool_minmax_ukernel_9p8x__neonfp16arith_c8,
        .init.f16 = xnn_init_f16_minmax_neon_params,
        .mr = 9,
        .qr = 8,
      };
      xnn_params.f16.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_f16_ibilinear_ukernel__neonfp16arith_c8,
        .pixel_tile = 1,
        .channel_tile = 8,
      };

      xnn_params.f16.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f16_prelu_ukernel__neonfp16arith_2x16,
        .row_tile = 2,
        .channel_tile = 16,
      };

      xnn_params.f16.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vadd_minmax_ukernel__neonfp16arith_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vaddc_minmax_ukernel__neonfp16arith_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vaddc_minmax_ukernel__neonfp16arith_x16,
        .init.f16_minmax = xnn_init_f16_minmax_neon_params,
        .element_tile = 16,
      };
      xnn_params.f16.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vmul_minmax_ukernel__neonfp16arith_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vmulc_minmax_ukernel__neonfp16arith_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vmulc_minmax_ukernel__neonfp16arith_x16,
        .init.f16_minmax = xnn_init_f16_minmax_neon_params,
        .element_tile = 16,
      };
      xnn_params.f16.vmulcaddc = (struct vmulcaddc_parameters) {
        .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x,
        .init.f16 = xnn_init_f16_minmax_neon_params,
        .channel_tile = 8,
        .row_tile = 2,
      };

      xnn_params.f16.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_vclamp_ukernel__neonfp16arith_x16,
        .init.f16_minmax = xnn_init_f16_minmax_neon_params,
        .element_tile = 16,
      };
      xnn_params.f16.hswish = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_vhswish_ukernel__neonfp16arith_x16,
        .init.f16_hswish = xnn_init_f16_hswish_neon_params,
        .element_tile = 16,
      };
      xnn_params.f16.lrelu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_vlrelu_ukernel__neonfp16arith_x16,
        .init.f16_lrelu = xnn_init_f16_lrelu_neon_params,
        .element_tile = 16,
      };
    }
  #endif  // XNN_NO_F16_OPERATORS

  /**************************** F32 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_F32_OPERATORS
    init_flags |= XNN_INIT_FLAG_F32;

    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75);
        xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75);
        xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
        xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
        xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
        xnn_params.f32.gemm.mr = 6;
        xnn_params.f32.gemm.nr = 8;
      #else  // !XNN_ENABLE_ASSEMBLY
        xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__neonfma_lane_ld64);
        xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__neonfma_lane_ld64);
        xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neonfma_lane_ld64);
        xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64);
        xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
        xnn_params.f32.gemm.mr = 6;
        xnn_params.f32.gemm.nr = 8;
       #endif  // XNN_ENABLE_ASSEMBLY
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        switch (cpuinfo_get_core(0)->uarch) {
          case cpuinfo_uarch_cortex_a57:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a75);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a75);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 6;
            xnn_params.f32.gemm.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a72:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 4;
            xnn_params.f32.gemm.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a75:
          case cpuinfo_uarch_cortex_a76:
          case cpuinfo_uarch_exynos_m3:
          case cpuinfo_uarch_exynos_m4:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 6;
            xnn_params.f32.gemm.nr = 8;
            #if XNN_ENABLE_JIT
              xnn_params.f32.gemm.generator.gemm = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75);
              xnn_params.f32.gemm.generator.igemm = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75);
              xnn_params.f32.gemm.generator.gemm1 = xnn_init_hmp_gemm_codegen(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
              xnn_params.f32.gemm.generator.igemm1 = xnn_init_hmp_igemm_codegen(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
            #endif
            break;
          case cpuinfo_uarch_exynos_m1:
          case cpuinfo_uarch_exynos_m2:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8s4__neonfma);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8s4__neonfma);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 6;
            xnn_params.f32.gemm.nr = 8;
            xnn_params.f32.gemm.log2_sr = 2;
            break;
          case cpuinfo_uarch_cortex_a53:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a53);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a53);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 6;
            xnn_params.f32.gemm.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a55r0:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a53);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a53);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 6;
            xnn_params.f32.gemm.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a35:
          case cpuinfo_uarch_cortex_a55:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a55);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a55);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 6;
            xnn_params.f32.gemm.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a73:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a73);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a73);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 6;
            xnn_params.f32.gemm.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a77:
          case cpuinfo_uarch_exynos_m5:
          case cpuinfo_uarch_kryo:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a75);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a75);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 4;
            xnn_params.f32.gemm.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a78:
          case cpuinfo_uarch_cortex_x1:
          default:
            xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_ld128);
            xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_ld128);
            xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neonfma_lane_ld64);
            xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64);
            xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.gemm.mr = 6;
            xnn_params.f32.gemm.nr = 8;
            break;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = xnn_params.f32.gemm.mr;
          const uint32_t nr = xnn_params.f32.gemm.nr;
          const uint32_t log2_sr = xnn_params.f32.gemm.log2_sr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a53:
                if (mr == 6 && nr == 8 && log2_sr == 0) {
                  xnn_params.f32.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a53;
                  xnn_params.f32.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                  xnn_params.f32.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_prfm_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_prfm_cortex_a53;
                  xnn_params.f32.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 6 && nr == 8 && log2_sr == 0) {
                  xnn_params.f32.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a53;
                  xnn_params.f32.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                  xnn_params.f32.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a53;
                  xnn_params.f32.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                }
                break;
              case cpuinfo_uarch_cortex_a55:
                if (mr == 6 && nr == 8 && log2_sr == 0) {
                  xnn_params.f32.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a55;
                  xnn_params.f32.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a55;
                  xnn_params.f32.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                  xnn_params.f32.gemm.minmax.gemm.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a55;
                  xnn_params.f32.gemm.minmax.igemm.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a55;
                  xnn_params.f32.gemm.minmax.gemm1.function[i] = (xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                  xnn_params.f32.gemm.minmax.igemm1.function[i] = (xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // !XNN_ENABLE_ASSEMBLY
        xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_6x8__neonfma_lane_ld64);
        xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_6x8__neonfma_lane_ld64);
        xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__neonfma_lane_ld64);
        xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64);
        xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
        xnn_params.f32.gemm.mr = 6;
        xnn_params.f32.gemm.nr = 8;
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC
    xnn_params.f32.gemm2.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x2__neonfma_lane_ld64);
    xnn_params.f32.gemm2.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x2__neonfma_lane_ld64);
    xnn_params.f32.gemm2.init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.gemm2.mr = 4;
    xnn_params.f32.gemm2.nr = 2;

    xnn_params.f32.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x3__neonfma;
    xnn_params.f32.dwconv[0].init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.dwconv[0].channel_tile = 8;
    xnn_params.f32.dwconv[0].primary_tile = 3;

    xnn_params.f32.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma;
    xnn_params.f32.dwconv[1].init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.dwconv[1].channel_tile = 8;
    xnn_params.f32.dwconv[1].primary_tile = 4;

    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC
      xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma;
      xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[2].channel_tile = 8;
      xnn_params.f32.dwconv[2].primary_tile = 9;
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_kryo:
          xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma;
          xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_scalar_params;
          xnn_params.f32.dwconv[2].channel_tile = 8;
          xnn_params.f32.dwconv[2].primary_tile = 9;
          break;
        #if XNN_ENABLE_ASSEMBLY
          case cpuinfo_uarch_cortex_a53:
          case cpuinfo_uarch_cortex_a55r0:
          case cpuinfo_uarch_cortex_a55:
            xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55;
            xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_scalar_params;
            xnn_params.f32.dwconv[2].channel_tile = 4;
            xnn_params.f32.dwconv[2].primary_tile = 9;
            break;
        #endif  // XNN_ENABLE_ASSEMBLY
        default:
          xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma;
          xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_scalar_params;
          xnn_params.f32.dwconv[2].channel_tile = 8;
          xnn_params.f32.dwconv[2].primary_tile = 9;
          break;
      }
    #endif  // XNN_PLATFORM_IOS && XNN_PLATFORM_MAC

    xnn_params.f32.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2;
    xnn_params.f32.dwconv[3].init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.dwconv[3].channel_tile = 8;
    xnn_params.f32.dwconv[3].primary_tile = 25;

    xnn_params.f32.avgpool = (struct avgpool_parameters) {
      .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9x__neon_c4,
      .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9p8x__neon_c4,
      .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 4,
    };
    xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
      .unipass = (xnn_pavgpool_unipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9x__neon_c4,
      .multipass = (xnn_pavgpool_multipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9p8x__neon_c4,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 4,
    };
    xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4,
      .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
      .update.f32 = xnn_update_f32_scaleminmax_scalar_params,
      .row_tile = 7,
      .channel_tile = 4,
    };
    xnn_params.f32.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4,
      .init.f32 = xnn_init_f32_minmax_scalar_params,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__neon_c4,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__neon_c4,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_multipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__neon_c4,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.ibilinear = (struct ibilinear_parameters) {
      .ukernel = (xnn_ibilinear_ukernel_function) xnn_f32_ibilinear_ukernel__neonfma_c8,
      .pixel_tile = 1,
      .channel_tile = 8,
    };
    xnn_params.f32.abs = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vabs_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vclamp_ukernel__neon_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.elu = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x16,
      .init.f32_elu = xnn_init_f32_elu_neonfma_rr1_lut16_p3_params,
      .element_tile = 16,
    };
    xnn_params.f32.hswish = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vhswish_ukernel__neon_x16,
      .init.f32_hswish = xnn_init_f32_hswish_scalar_params,
      .element_tile = 16,
    };
    xnn_params.f32.lrelu = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vlrelu_ukernel__neon_x8,
      .init.f32_lrelu = xnn_init_f32_lrelu_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.neg = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vneg_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.rndne = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__neonv8_x8,
      .element_tile = 8,
    };
    xnn_params.f32.rndz = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__neonv8_x8,
      .element_tile = 8,
    };
    xnn_params.f32.rndu = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__neonv8_x8,
      .element_tile = 8,
    };
    xnn_params.f32.rndd = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__neonv8_x8,
      .element_tile = 8,
    };
    xnn_params.f32.sigmoid = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16,
      .init.f32_sigmoid = xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params,
      .element_tile = 16,
    };
    xnn_params.f32.sqr = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqr_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.sqrt = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqrt_ukernel__neon_sqrt_x4,
      .element_tile = 4,
    };
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__neon_2x8,
      .row_tile = 2,
      .channel_tile = 8,
    };
    xnn_params.f32.raddstoreexpminusmax = (struct raddstoreexpminusmax_parameters) {
      .ukernel = xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x16,
      .init = xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
      .element_tile = 16,
    };
    xnn_params.f32.rmax = xnn_f32_rmax_ukernel__neon;
    xnn_params.f32.vadd = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_minmax_ukernel__neon_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__neon_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__neon_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.vdiv = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_minmax_ukernel__neon_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_minmax_ukernel__neon_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_minmax_ukernel__neon_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.vmax = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__neon_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__neon_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmin = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__neon_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__neon_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmul = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_minmax_ukernel__neon_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__neon_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__neon_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.vsub = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_minmax_ukernel__neon_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_minmax_ukernel__neon_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_minmax_ukernel__neon_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.vsqrdiff = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiff_ukernel__neon_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__neon_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x,
      .init.f32 = xnn_init_f32_minmax_scalar_params,
      .channel_tile = 4,
      .row_tile = 2,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
      init_flags |= XNN_INIT_FLAG_CHW_OPT;

      xnn_params.f32.spmm = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_32x1__neonfma_pipelined,
        .mr = 32,
        .nr = 1,
      };
      xnn_params.f32.spmm2 = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_32x2__neonfma,
        .mr = 32,
        .nr = 2,
      };
      xnn_params.f32.spmm4 = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_32x4__neonfma,
        .mr = 32,
        .nr = 4,
      };
      xnn_params.f32.conv_hwc2chw_3x3c3s2 = (struct conv_hwc2chw_parameters) {
        .ukernel_with_symm_padding =
          (xnn_conv_hwc2chw_ukernel_function) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neonfma_2x2,
        .output_channel_tile = 4,
        .output_height_tile = 2,
        .output_width_tile = 2,
      };
      xnn_params.f32.dwconv2d_chw_3x3 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3p1__neonfma_3x4,
        .output_width_tile = 4,
        .output_height_tile = 3,
      };
      xnn_params.f32.dwconv2d_chw_3x3s2 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neonfma_2x4_acc2,
        .output_width_tile = 4,
        .output_height_tile = 2,
      };
      xnn_params.f32.dwconv2d_chw_5x5 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_4x4,
        .output_width_tile = 4,
        .output_height_tile = 4,
      };
      xnn_params.f32.dwconv2d_chw_5x5s2 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neonfma_1x4_acc2,
        .output_width_tile = 4,
        .output_height_tile = 1,
      };
      xnn_params.f32.gavgpool_cw = (struct gavgpool_cw_parameters) {
        .ukernel = (xnn_gavgpool_cw_ukernel_function) xnn_f32_gavgpool_cw_ukernel__neon_x4,
        .channel_tile = 4,
      };
      xnn_params.f32.ibilinear_chw = (struct ibilinear_chw_parameters) {
        .ukernel = (xnn_ibilinear_chw_ukernel_function) xnn_f32_ibilinear_chw_ukernel__neonfma_p8,
        .channel_tile = 1,
        .pixel_tile = 8,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /*************************** VCVT AArch64 micro-kernels ***************************/
  #ifndef XNN_NO_VCVT_OPERATORS
    init_flags |= XNN_INIT_FLAG_VCVT;

    xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__neonfp16_x16,
      .element_tile = 16,
    };
    xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__neonfp16_x16,
      .element_tile = 16,
    };
    xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_qs8_vcvt_ukernel__neonv8_x32,
      .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_neonv8_params,
      .element_tile = 32,
    };
    xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__neonv8_x32,
      .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_neonv8_params,
      .element_tile = 32,
    };
    xnn_params.vcvt.qs8_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_qs8_f32_vcvt_ukernel__neon_x32,
      .init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_neon_params,
      .element_tile = 32,
    };
    xnn_params.vcvt.qu8_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_qu8_f32_vcvt_ukernel__neon_x32,
      .init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_neon_params,
      .element_tile = 32,
    };
  #endif  // XNN_NO_VCVT_OPERATORS

  /**************************** X32 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_X32_OPERATORS
    init_flags |= XNN_INIT_FLAG_X32;

    xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__neon;
    xnn_params.x32.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__neon,
      .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__neon,
      .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__neon,
      .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__neon,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
      xnn_params.x32.depthtospace2d_chw2hwc = (struct depthtospace2d_chw2hwc_parameters) {
        .ukernel = (xnn_depthtospace2d_chw2hwc_ukernel_function) xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar,
        .channel_tile = 1,
        .pixel_tile = 1,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_X32_OPERATORS

  /**************************** XX AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_XX_OPERATORS
    init_flags |= XNN_INIT_FLAG_XX;

    xnn_params.xx.copy = (xnn_univector_ukernel_function) xnn_xx_copy_ukernel__memcpy;
    xnn_params.xx.fill = (struct fill_parameters) {
      .ukernel = (xnn_fill_ukernel_function) xnn_xx_fill_ukernel__neon_x64,
      .row_tile = 1,
    };
    xnn_params.xx.pad = (struct pad_parameters) {
      .ukernel = (xnn_pad_ukernel_function) xnn_xx_pad_ukernel__neon,
      .row_tile = 1,
    };
  #endif

#elif XNN_ARCH_X86 || XNN_ARCH_X86_64
  if (!cpuinfo_has_x86_sse2()) {
    xnn_log_error("XNNPACK initialization failed: SSE2 is not supported");
    return;
  }

  /**************************** QC8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_QC8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QC8;

    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_avx512_params;
      xnn_params.qc8.gemm.mr = 4;
      xnn_params.qc8.gemm.nr = 16;
      xnn_params.qc8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_xop()) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_sse4_params;
      xnn_params.qc8.gemm.mr = 2;
      xnn_params.qc8.gemm.nr = 4;
      xnn_params.qc8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_3x8c8__avx2);
      xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_3x8c8__avx2);
      xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x8c8__avx2);
      xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x8c8__avx2);
      xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_avx2_params;
      xnn_params.qc8.gemm.mr = 3;
      xnn_params.qc8.gemm.nr = 8;
      xnn_params.qc8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_sse4_params;
      xnn_params.qc8.gemm.mr = 2;
      xnn_params.qc8.gemm.nr = 4;
      xnn_params.qc8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_sse4_params;
      xnn_params.qc8.gemm.mr = 3;
      xnn_params.qc8.gemm.nr = 4;
      xnn_params.qc8.gemm.log2_kr = 3;
    } else {
      xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_sse2_params;
      xnn_params.qc8.gemm.mr = 3;
      xnn_params.qc8.gemm.nr = 4;
      xnn_params.qc8.gemm.log2_kr = 3;
    }

    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up32x9__avx512skx_mul32;
      xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_avx512_params;
      xnn_params.qc8.dwconv[0].channel_tile = 32;
      xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up32x25__avx512skx_mul32;
      xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_avx512_params;
      xnn_params.qc8.dwconv[1].channel_tile = 32;
    } else if (cpuinfo_has_x86_xop()) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__xop_mul16_add16;
      xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_sse4_params;
      xnn_params.qc8.dwconv[0].channel_tile = 16;
      xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__xop_mul16_add16;
      xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_sse4_params;
      xnn_params.qc8.dwconv[1].channel_tile = 16;
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul32;
      xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_avx2_params;
      xnn_params.qc8.dwconv[0].channel_tile = 16;
      xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul32;
      xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_avx2_params;
      xnn_params.qc8.dwconv[1].channel_tile = 16;
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16_add16;
      xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_sse4_params;
      xnn_params.qc8.dwconv[0].channel_tile = 16;
      xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul16_add16;
      xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_sse4_params;
      xnn_params.qc8.dwconv[1].channel_tile = 16;
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul16;
      xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_sse4_params;
      xnn_params.qc8.dwconv[0].channel_tile = 8;
      xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__sse41_mul16;
      xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_sse4_params;
      xnn_params.qc8.dwconv[1].channel_tile = 8;
    } else if (cpuinfo_has_x86_sse2()) {
      xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__sse2_mul16;
      xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_sse2_params;
      xnn_params.qc8.dwconv[0].channel_tile = 8;
      xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__sse2_mul16;
      xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_sse2_params;
      xnn_params.qc8.dwconv[1].channel_tile = 8;
    }
    xnn_params.qc8.dwconv[0].primary_tile = 9;
    xnn_params.qc8.dwconv[1].primary_tile = 25;
  #endif  // XNN_NO_QC8_OPERATORS

  /**************************** QS8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QS8;

    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx512_params;
      xnn_params.qs8.gemm.mr = 4;
      xnn_params.qs8.gemm.nr = 16;
      xnn_params.qs8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_xop()) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      xnn_params.qs8.gemm.mr = 2;
      xnn_params.qs8.gemm.nr = 4;
      xnn_params.qs8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2);
      xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2);
      xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2);
      xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2);
      xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx2_params;
      xnn_params.qs8.gemm.mr = 3;
      xnn_params.qs8.gemm.nr = 8;
      xnn_params.qs8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      xnn_params.qs8.gemm.mr = 2;
      xnn_params.qs8.gemm.nr = 4;
      xnn_params.qs8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      xnn_params.qs8.gemm.mr = 3;
      xnn_params.qs8.gemm.nr = 4;
      xnn_params.qs8.gemm.log2_kr = 3;
    } else {
      xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse2_params;
      xnn_params.qs8.gemm.mr = 3;
      xnn_params.qs8.gemm.nr = 4;
      xnn_params.qs8.gemm.log2_kr = 3;
    }

    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up32x9__avx512skx_mul32;
      xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx512_params;
      xnn_params.qs8.dwconv[0].channel_tile = 32;
      xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up32x25__avx512skx_mul32;
      xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx512_params;
      xnn_params.qs8.dwconv[1].channel_tile = 32;
    } else if (cpuinfo_has_x86_xop()) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__xop_mul16_add16;
      xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      xnn_params.qs8.dwconv[0].channel_tile = 16;
      xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__xop_mul16_add16;
      xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      xnn_params.qs8.dwconv[1].channel_tile = 16;
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul32;
      xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx2_params;
      xnn_params.qs8.dwconv[0].channel_tile = 16;
      xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul32;
      xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_avx2_params;
      xnn_params.qs8.dwconv[1].channel_tile = 16;
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16_add16;
      xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      xnn_params.qs8.dwconv[0].channel_tile = 16;
      xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul16_add16;
      xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      xnn_params.qs8.dwconv[1].channel_tile = 16;
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul16_add16;
      xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      xnn_params.qs8.dwconv[0].channel_tile = 8;
      xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__sse41_mul16_add16;
      xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse4_params;
      xnn_params.qs8.dwconv[1].channel_tile = 8;
    } else if (cpuinfo_has_x86_sse2()) {
      xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__sse2_mul16_add16;
      xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse2_params;
      xnn_params.qs8.dwconv[0].channel_tile = 8;
      xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__sse2_mul16_add16;
      xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_sse2_params;
      xnn_params.qs8.dwconv[1].channel_tile = 8;
    }
    xnn_params.qs8.dwconv[0].primary_tile = 9;
    xnn_params.qs8.dwconv[1].primary_tile = 25;

    if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qs8.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8,
        .init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_sse4_params,
        .update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_sse4_params,
        .row_tile = 7,
        .channel_tile = 8,
      };
    } else {
      xnn_params.qs8.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8,
        .init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_sse2_params,
        .update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_sse2_params,
        .row_tile = 7,
        .channel_tile = 8,
      };
    }

    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.qs8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__avx512skx_mul32_ld128_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_x16,
        .init.qs8_addsub = xnn_init_qs8_add_minmax_avx512_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_xop()) {
      xnn_params.qs8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__xop_mul32_ld32_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8,
        .init.qs8_addsub = xnn_init_qs8_add_minmax_sse4_mul32_params,
        .element_tile = 8,
      };
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.qs8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16,
        .init.qs8_addsub = xnn_init_qs8_add_minmax_avx2_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.qs8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__avx_mul32_ld32_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__avx_mul32_ld32_x8,
        .init.qs8_addsub = xnn_init_qs8_add_minmax_sse4_mul32_params,
        .element_tile = 8,
      };
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qs8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8,
        .init.qs8_addsub = xnn_init_qs8_add_minmax_sse4_mul16_params,
        .element_tile = 8,
      };
    } else {
      xnn_params.qs8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8,
        .init.qs8_addsub = xnn_init_qs8_add_minmax_sse2_params,
        .element_tile = 8,
      };
    }
    if (cpuinfo_has_x86_avx()) {
      xnn_params.qs8.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16,
        .init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_sse4_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qs8.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x16,
        .init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_sse4_params,
        .element_tile = 16,
      };
    } else {
      xnn_params.qs8.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x8,
        .init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_sse2_params,
        .element_tile = 8,
      };
    }
  #endif  // XNN_NO_QS8_OPERATORS

  /**************************** QU8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_QU8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QU8;

    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx);
      xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx);
      xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx512_params;
      xnn_params.qu8.gemm.mr = 4;
      xnn_params.qu8.gemm.nr = 16;
      xnn_params.qu8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_xop()) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64);
      xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64);
      xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.gemm.mr = 2;
      xnn_params.qu8.gemm.nr = 4;
      xnn_params.qu8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2);
      xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_3x8c8__avx2);
      xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2);
      xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x8c8__avx2);
      xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx2_params;
      xnn_params.qu8.gemm.mr = 3;
      xnn_params.qu8.gemm.nr = 8;
      xnn_params.qu8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.gemm.mr = 2;
      xnn_params.qu8.gemm.nr = 4;
      xnn_params.qu8.gemm.log2_kr = 3;
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.gemm.mr = 3;
      xnn_params.qu8.gemm.nr = 4;
      xnn_params.qu8.gemm.log2_kr = 3;
    } else {
      xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.gemm.mr = 3;
      xnn_params.qu8.gemm.nr = 4;
      xnn_params.qu8.gemm.log2_kr = 3;
    }

    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up32x9__avx512skx_mul32;
      xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx512_params;
      xnn_params.qu8.dwconv[0].channel_tile = 32;
      xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up32x25__avx512skx_mul32;
      xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx512_params;
      xnn_params.qu8.dwconv[1].channel_tile = 32;
    } else if (cpuinfo_has_x86_xop()) {
      // XOP should be checked before AVX2: AMD Excavator supports both, but performs better with XOP microkernels
      xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__xop_mul32;
      xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.dwconv[0].channel_tile = 16;
      xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__xop_mul32;
      xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.dwconv[1].channel_tile = 16;
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul32;
      xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx2_params;
      xnn_params.qu8.dwconv[0].channel_tile = 16;
      xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul32;
      xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_avx2_params;
      xnn_params.qu8.dwconv[1].channel_tile = 16;
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16;
      xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.dwconv[0].channel_tile = 16;
      xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul16;
      xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.dwconv[1].channel_tile = 16;
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul16;
      xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.dwconv[0].channel_tile = 8;
      xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__sse41_mul16;
      xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.dwconv[1].channel_tile = 8;
    } else if (cpuinfo_has_x86_sse2()) {
      xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__sse2_mul16;
      xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.dwconv[0].channel_tile = 8;
      xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__sse2_mul16;
      xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_sse2_params;
      xnn_params.qu8.dwconv[1].channel_tile = 8;
    }
    xnn_params.qu8.dwconv[0].primary_tile = 9;
    xnn_params.qu8.dwconv[1].primary_tile = 25;

    xnn_params.qu8.avgpool = (struct avgpool_parameters) {
      .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9x__sse2_c8,
      .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9p8x__sse2_c8,
      .init.qu8 = xnn_init_qu8_avgpool_minmax_sse2_params,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 8,
    };
    if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qu8.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8,
        .init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_sse4_params,
        .update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_sse4_params,
        .row_tile = 7,
        .channel_tile = 8,
      };
    } else {
      xnn_params.qu8.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8,
        .init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_sse2_params,
        .update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_sse2_params,
        .row_tile = 7,
        .channel_tile = 8,
      };
    }

    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.qu8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__avx512skx_mul32_ld128_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_x16,
        .init.qu8_addsub = xnn_init_qu8_add_minmax_avx512_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_xop()) {
      xnn_params.qu8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__xop_mul32_ld32_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__xop_mul32_ld32_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__xop_mul32_ld32_x8,
        .init.qu8_addsub = xnn_init_qu8_add_minmax_sse4_params,
        .element_tile = 8,
      };
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.qu8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__avx2_mul32_ld64_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16,
        .init.qu8_addsub = xnn_init_qu8_add_minmax_avx2_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.qu8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__avx_mul32_ld32_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__avx_mul32_ld32_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__avx_mul32_ld32_x8,
        .init.qu8_addsub = xnn_init_qu8_add_minmax_sse4_params,
        .element_tile = 8,
      };
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qu8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__sse41_mul16_ld64_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8,
        .init.qu8_addsub = xnn_init_qu8_add_minmax_sse2_params,
        .element_tile = 8,
      };
    } else {
      xnn_params.qu8.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__sse2_mul16_ld64_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8,
        .init.qu8_addsub = xnn_init_qu8_add_minmax_sse2_params,
        .element_tile = 8,
      };
    }
    if (cpuinfo_has_x86_avx()) {
      xnn_params.qu8.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16,
        .init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_sse2_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.qu8.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x16,
        .init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_sse2_params,
        .element_tile = 16,
      };
    } else {
      xnn_params.qu8.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x8,
        .init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_sse2_params,
        .element_tile = 8,
      };
    }
  #endif  // XNN_NO_QU8_OPERATORS

  /**************************** U8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_S8_OPERATORS
    init_flags |= XNN_INIT_FLAG_S8;

    if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.s8.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_s8_vclamp_ukernel__sse41_x64,
        .init.s8_minmax = xnn_init_s8_minmax_sse4_params,
        .element_tile = 64,
      };
      xnn_params.s8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_s8_ibilinear_ukernel__sse41_c16,
        .pixel_tile = 1,
        .channel_tile = 16,
      };
      xnn_params.s8.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_s8_maxpool_minmax_ukernel_9p8x__sse41_c16,
        .init.s8 = xnn_init_s8_minmax_sse4_params,
        .mr = 9,
        .qr = 8,
      };
    } else {
      xnn_params.s8.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_s8_vclamp_ukernel__sse2_x64,
        .init.s8_minmax = xnn_init_s8_minmax_sse2_params,
        .element_tile = 64,
      };
      xnn_params.s8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_s8_ibilinear_ukernel__sse2_c8,
        .pixel_tile = 1,
        .channel_tile = 8,
      };
      xnn_params.s8.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_s8_maxpool_minmax_ukernel_9p8x__sse2_c16,
        .init.s8 = xnn_init_s8_minmax_sse2_params,
        .mr = 9,
        .qr = 8,
      };
    }
  #endif  // XNN_NO_S8_OPERATORS

  /**************************** U8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_U8_OPERATORS
    init_flags |= XNN_INIT_FLAG_U8;

    xnn_params.u8.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_u8_vclamp_ukernel__sse2_x64,
      .init.u8_minmax = xnn_init_u8_minmax_sse2_params,
      .element_tile = 64,
    };
    if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.u8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_u8_ibilinear_ukernel__sse41_c16,
        .pixel_tile = 1,
        .channel_tile = 16,
      };
    } else {
      xnn_params.u8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_u8_ibilinear_ukernel__sse2_c8,
        .pixel_tile = 1,
        .channel_tile = 8,
      };
    }
    xnn_params.u8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16,
      .init.u8 = xnn_init_u8_minmax_sse2_params,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    xnn_params.u8.rmax = xnn_u8_rmax_ukernel__sse2;
  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_X8_OPERATORS
    init_flags |= XNN_INIT_FLAG_X8;

    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.x8.lut = xnn_x8_lut_ukernel__avx512skx_vpshufb_x64;
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.x8.lut = xnn_x8_lut_ukernel__avx2_x128;
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.x8.lut = xnn_x8_lut_ukernel__avx_x64;
    } else {
      // Note: SSSE3 version is usually slower than scalar
      xnn_params.x8.lut = xnn_x8_lut_ukernel__scalar_x4;
    }
    xnn_params.x8.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__sse2,
      .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__sse2,
      .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__sse2,
      .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__sse2,
    };
  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F16 x86 micro-kernels ****************************/
  #ifndef XNN_NO_F16_OPERATORS
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx2()) {
      init_flags |= XNN_INIT_FLAG_F16;

      xnn_params.f16.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f16_gemm_minmax_ukernel_4x16__avx2_broadcast);
      xnn_params.f16.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast);
      xnn_params.f16.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f16_gemm_minmax_ukernel_1x16__avx2_broadcast);
      xnn_params.f16.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast);
      xnn_params.f16.gemm.init.f16 = xnn_init_f16_scaleminmax_avx_params;
      xnn_params.f16.gemm.mr = 4;
      xnn_params.f16.gemm.nr = 16;

      xnn_params.f16.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f16_dwconv_minmax_ukernel_up16x3__fma3;
      xnn_params.f16.dwconv[0].init.f16 = xnn_init_f16_minmax_avx_params;
      xnn_params.f16.dwconv[0].channel_tile = 16;
      xnn_params.f16.dwconv[0].primary_tile = 3;

      xnn_params.f16.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f16_dwconv_minmax_ukernel_up16x4__fma3;
      xnn_params.f16.dwconv[1].init.f16 = xnn_init_f16_minmax_avx_params;
      xnn_params.f16.dwconv[1].channel_tile = 16;
      xnn_params.f16.dwconv[1].primary_tile = 4;

      xnn_params.f16.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f16_dwconv_minmax_ukernel_up16x9__fma3;
      xnn_params.f16.dwconv[2].init.f16 = xnn_init_f16_minmax_avx_params;
      xnn_params.f16.dwconv[2].channel_tile = 16;
      xnn_params.f16.dwconv[2].primary_tile = 9;

      xnn_params.f16.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2;
      xnn_params.f16.dwconv[3].init.f16 = xnn_init_f16_minmax_avx_params;
      xnn_params.f16.dwconv[3].channel_tile = 8;
      xnn_params.f16.dwconv[3].primary_tile = 25;

      xnn_params.f16.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8,
        .init.f16 = xnn_init_f16_scaleminmax_avx_params,
        .update.f16 = xnn_update_f16_scaleminmax_avx_params,
        .row_tile = 7,
        .channel_tile = 8,
      };

      xnn_params.f16.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_f16_maxpool_minmax_ukernel_9p8x__f16c_c8,
        .init.f16 = xnn_init_f16_minmax_avx_params,
        .mr = 9,
        .qr = 8,
      };
      xnn_params.f16.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_f16_ibilinear_ukernel__fma3_c8,
        .pixel_tile = 1,
        .channel_tile = 8,
      };

      xnn_params.f16.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f16_prelu_ukernel__f16c_2x16,
        .row_tile = 2,
        .channel_tile = 16,
      };

      xnn_params.f16.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vadd_minmax_ukernel__f16c_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vaddc_minmax_ukernel__f16c_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vaddc_minmax_ukernel__f16c_x16,
        .init.f16_minmax = xnn_init_f16_minmax_avx_params,
        .element_tile = 16,
      };
      xnn_params.f16.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vmul_minmax_ukernel__f16c_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vmulc_minmax_ukernel__f16c_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f16_vmulc_minmax_ukernel__f16c_x16,
        .init.f16_minmax = xnn_init_f16_minmax_avx_params,
        .element_tile = 16,
      };
      xnn_params.f16.vmulcaddc = (struct vmulcaddc_parameters) {
        .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x,
        .init.f16 = xnn_init_f16_minmax_avx_params,
        .channel_tile = 8,
        .row_tile = 2,
      };

      xnn_params.f16.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_vclamp_ukernel__f16c_x16,
        .init.f16_minmax = xnn_init_f16_minmax_avx_params,
        .element_tile = 16,
      };
      xnn_params.f16.hswish = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_vhswish_ukernel__f16c_x16,
        .init.f16_hswish = xnn_init_f16_hswish_avx_params,
        .element_tile = 16,
      };
      xnn_params.f16.lrelu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_vlrelu_ukernel__f16c_x16,
        .init.f16_lrelu = xnn_init_f16_lrelu_avx_params,
        .element_tile = 16,
      };
    }
  #endif  // XNN_NO_F16_OPERATORS

  /**************************** F32 x86 micro-kernels ****************************/
  #ifndef XNN_NO_F32_OPERATORS
    init_flags |= XNN_INIT_FLAG_F32;

    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast);
      xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_7x16__avx512f_broadcast);
      xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast);
      xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast);
      xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.gemm.mr = 7;
      xnn_params.f32.gemm.nr = 16;
    } else if (cpuinfo_has_x86_fma3()) {
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_zen:
        case cpuinfo_uarch_dhyana:
          xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast);
          xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x16s4__fma3_broadcast);
          xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast);
          xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast);
          xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_avx_params;
          xnn_params.f32.gemm.mr = 4;
          xnn_params.f32.gemm.nr = 16;
          xnn_params.f32.gemm.log2_sr = 2;
          break;
        default:
          xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast);
          xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast);
          xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast);
          xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast);
          xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_avx_params;
          xnn_params.f32.gemm.mr = 5;
          xnn_params.f32.gemm.nr = 16;
          break;
      }
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast);
      xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_5x16__avx_broadcast);
      xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast);
      xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x16__avx_broadcast);
      xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_avx_params;
      xnn_params.f32.gemm.mr = 5;
      xnn_params.f32.gemm.nr = 16;
    } else {
      xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__sse_load1);
      xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__sse_load1);
      xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__sse_load1);
      xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__sse_load1);
      xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_sse_params;
      xnn_params.f32.gemm.mr = 4;
      xnn_params.f32.gemm.nr = 8;
    }
    xnn_params.f32.gemm2.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x2c4__sse);
    xnn_params.f32.gemm2.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x2c4__sse);
    xnn_params.f32.gemm2.init.f32 = xnn_init_f32_minmax_sse_params;
    xnn_params.f32.gemm2.mr = 4;
    xnn_params.f32.gemm2.nr = 2;
    xnn_params.f32.gemm2.log2_kr = 2;

    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up16x3__avx512f;
      xnn_params.f32.dwconv[0].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[0].channel_tile = 16;
      xnn_params.f32.dwconv[0].primary_tile = 3;

      xnn_params.f32.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f;
      xnn_params.f32.dwconv[1].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[1].channel_tile = 16;
      xnn_params.f32.dwconv[1].primary_tile = 4;

      xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f;
      xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[2].channel_tile = 16;
      xnn_params.f32.dwconv[2].primary_tile = 9;

      xnn_params.f32.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f;
      xnn_params.f32.dwconv[3].init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.dwconv[3].channel_tile = 16;
      xnn_params.f32.dwconv[3].primary_tile = 25;
    } else if (cpuinfo_has_x86_fma3()) {
      xnn_params.f32.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up16x3__fma3;
      xnn_params.f32.dwconv[0].init.f32 = xnn_init_f32_minmax_avx_params;
      xnn_params.f32.dwconv[0].channel_tile = 16;
      xnn_params.f32.dwconv[0].primary_tile = 3;

      xnn_params.f32.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up16x4__fma3;
      xnn_params.f32.dwconv[1].init.f32 = xnn_init_f32_minmax_avx_params;
      xnn_params.f32.dwconv[1].channel_tile = 16;
      xnn_params.f32.dwconv[1].primary_tile = 4;

      xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up16x9__fma3;
      xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_avx_params;
      xnn_params.f32.dwconv[2].channel_tile = 16;
      xnn_params.f32.dwconv[2].primary_tile = 9;

      xnn_params.f32.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x25__fma3;
      xnn_params.f32.dwconv[3].init.f32 = xnn_init_f32_minmax_avx_params;
      xnn_params.f32.dwconv[3].channel_tile = 8;
      xnn_params.f32.dwconv[3].primary_tile = 25;
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up16x3__avx;
      xnn_params.f32.dwconv[0].init.f32 = xnn_init_f32_minmax_avx_params;
      xnn_params.f32.dwconv[0].channel_tile = 16;
      xnn_params.f32.dwconv[0].primary_tile = 3;

      xnn_params.f32.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up16x4__avx;
      xnn_params.f32.dwconv[1].init.f32 = xnn_init_f32_minmax_avx_params;
      xnn_params.f32.dwconv[1].channel_tile = 16;
      xnn_params.f32.dwconv[1].primary_tile = 4;

      xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up16x9__avx;
      xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_avx_params;
      xnn_params.f32.dwconv[2].channel_tile = 16;
      xnn_params.f32.dwconv[2].primary_tile = 9;

      xnn_params.f32.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x25__avx;
      xnn_params.f32.dwconv[3].init.f32 = xnn_init_f32_minmax_avx_params;
      xnn_params.f32.dwconv[3].channel_tile = 8;
      xnn_params.f32.dwconv[3].primary_tile = 25;
    } else {
      xnn_params.f32.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x3__sse;
      xnn_params.f32.dwconv[0].init.f32 = xnn_init_f32_minmax_sse_params;
      xnn_params.f32.dwconv[0].channel_tile = 8;
      xnn_params.f32.dwconv[0].primary_tile = 3;

      xnn_params.f32.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x4__sse;
      xnn_params.f32.dwconv[1].init.f32 = xnn_init_f32_minmax_sse_params;
      xnn_params.f32.dwconv[1].channel_tile = 8;
      xnn_params.f32.dwconv[1].primary_tile = 4;

      xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x9__sse;
      xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_sse_params;
      xnn_params.f32.dwconv[2].channel_tile = 8;
      xnn_params.f32.dwconv[2].primary_tile = 9;

      xnn_params.f32.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x25__sse;
      xnn_params.f32.dwconv[3].init.f32 = xnn_init_f32_minmax_sse_params;
      xnn_params.f32.dwconv[3].channel_tile = 8;
      xnn_params.f32.dwconv[3].primary_tile = 25;
    }
    xnn_params.f32.avgpool = (struct avgpool_parameters) {
      .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9x__sse_c4,
      .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9p8x__sse_c4,
      .init.f32 = xnn_init_f32_scaleminmax_sse_params,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 4,
    };
    xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
      .unipass = (xnn_pavgpool_unipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9x__sse_c4,
      .multipass = (xnn_pavgpool_multipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9p8x__sse_c4,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 4,
    };
    xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4,
      .init.f32 = xnn_init_f32_scaleminmax_sse_params,
      .update.f32 = xnn_update_f32_scaleminmax_sse_params,
      .row_tile = 7,
      .channel_tile = 4,
    };
    xnn_params.f32.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4,
      .init.f32 = xnn_init_f32_minmax_sse_params,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__sse2_c4,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__sse2_c4,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_multipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.ibilinear = (struct ibilinear_parameters) {
      .ukernel = (xnn_ibilinear_ukernel_function) xnn_f32_ibilinear_ukernel__sse_c8,
      .pixel_tile = 1,
      .channel_tile = 8,
    };
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.abs = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vabs_ukernel__avx512f_x16,
        .init.f32_abs = xnn_init_f32_abs_avx512_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.abs = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vabs_ukernel__avx_x16,
        .init.f32_abs = xnn_init_f32_abs_avx_params,
        .element_tile = 16,
      };
    } else {
      xnn_params.f32.abs = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vabs_ukernel__sse_x8,
        .init.f32_abs = xnn_init_f32_abs_sse_params,
        .element_tile = 8,
      };
    }
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vclamp_ukernel__avx512f_x16,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vclamp_ukernel__avx_x16,
        .init.f32_minmax = xnn_init_f32_minmax_avx_params,
        .element_tile = 16,
      };
    } else {
      xnn_params.f32.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vclamp_ukernel__sse_x8,
        .init.f32_minmax = xnn_init_f32_minmax_sse_params,
        .element_tile = 8,
      };
    }
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.elu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64,
        .init.f32_elu = xnn_init_f32_elu_avx512_rr1_lut16_p3_params,
        .element_tile = 64,
      };
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.f32.elu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56,
        .init.f32_elu = xnn_init_f32_elu_avx2_rr1_lut4_p4_params,
        .element_tile = 56,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.elu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32,
        .init.f32_elu = xnn_init_f32_elu_avx_rr2_lut4_p4_params,
        .element_tile = 32,
      };
    } else {
      xnn_params.f32.elu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x12,
        .init.f32_elu = xnn_init_f32_elu_sse2_rr2_lut16_p3_params,
        .element_tile = 12,
      };
    }
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.hswish = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vhswish_ukernel__avx512f_x16,
        .init.f32_hswish = xnn_init_f32_hswish_avx512_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_fma3()) {
      xnn_params.f32.hswish = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vhswish_ukernel__fma3_x16,
        .init.f32_hswish = xnn_init_f32_hswish_avx_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.hswish = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vhswish_ukernel__avx_x16,
        .init.f32_hswish = xnn_init_f32_hswish_avx_params,
        .element_tile = 16,
      };
    } else {
      xnn_params.f32.hswish = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vhswish_ukernel__sse_x8,
        .init.f32_hswish = xnn_init_f32_hswish_sse_params,
        .element_tile = 8,
      };
    }
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.lrelu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vlrelu_ukernel__avx512f_x16,
        .init.f32_lrelu = xnn_init_f32_lrelu_scalar_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.lrelu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vlrelu_ukernel__avx_x16,
        .init.f32_lrelu = xnn_init_f32_lrelu_avx_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.f32.lrelu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vlrelu_ukernel__sse41_x8,
        .init.f32_lrelu = xnn_init_f32_lrelu_sse_params,
        .element_tile = 8,
      };
    } else {
      xnn_params.f32.lrelu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vlrelu_ukernel__sse_x8,
        .init.f32_lrelu = xnn_init_f32_lrelu_sse_params,
        .element_tile = 8,
      };
    }
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.neg = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vneg_ukernel__avx512f_x16,
        .init.f32_neg = xnn_init_f32_neg_avx512_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.neg = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vneg_ukernel__avx_x16,
        .init.f32_neg = xnn_init_f32_neg_avx_params,
        .element_tile = 16,
      };
    } else {
      xnn_params.f32.neg = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vneg_ukernel__sse_x8,
        .init.f32_neg = xnn_init_f32_neg_sse_params,
        .element_tile = 8,
      };
    }
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.rndne = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__avx512f_x16,
        .element_tile = 16,
      };
      xnn_params.f32.rndz = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__avx512f_x16,
        .element_tile = 16,
      };
      xnn_params.f32.rndu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__avx512f_x16,
        .element_tile = 16,
      };
      xnn_params.f32.rndd = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__avx512f_x16,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.rndne = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__avx_x16,
        .init.f32_rnd = xnn_init_f32_rnd_avx_params,
        .element_tile = 16,
      };
      xnn_params.f32.rndz = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__avx_x16,
        .init.f32_rnd = xnn_init_f32_rnd_avx_params,
        .element_tile = 16,
      };
      xnn_params.f32.rndu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__avx_x16,
        .init.f32_rnd = xnn_init_f32_rnd_avx_params,
        .element_tile = 16,
      };
      xnn_params.f32.rndd = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__avx_x16,
        .init.f32_rnd = xnn_init_f32_rnd_avx_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.f32.rndne = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__sse41_x8,
        .element_tile = 8,
      };
      xnn_params.f32.rndz = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__sse41_x8,
        .element_tile = 8,
      };
      xnn_params.f32.rndu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__sse41_x8,
        .element_tile = 8,
      };
      xnn_params.f32.rndd = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__sse41_x8,
        .element_tile = 8,
      };
    } else {
      xnn_params.f32.rndne = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__sse2_x8,
        .init.f32_rnd = xnn_init_f32_rnd_sse2_params,
        .element_tile = 8,
      };
      xnn_params.f32.rndz = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__sse2_x8,
        .init.f32_rnd = xnn_init_f32_rnd_sse2_params,
        .element_tile = 8,
      };
      xnn_params.f32.rndu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__sse2_x8,
        .init.f32_rnd = xnn_init_f32_rnd_sse2_params,
        .element_tile = 8,
      };
      xnn_params.f32.rndd = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__sse2_x8,
        .init.f32_rnd = xnn_init_f32_rnd_sse2_params,
        .element_tile = 8,
      };
    }
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.sigmoid = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x64,
        .init.f32_sigmoid = xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params,
        .element_tile = 64,
      };
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.f32.sigmoid = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x40,
        .init.f32_sigmoid = xnn_init_f32_sigmoid_avx2_rr1_p5_params,
        .element_tile = 40,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.sigmoid = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x40,
        .init.f32_sigmoid = xnn_init_f32_sigmoid_avx_rr2_p5_params,
        .element_tile = 40,
      };
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.f32.sigmoid = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x8,
        .init.f32_sigmoid = xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params,
        .element_tile = 8,
      };
    } else {
      xnn_params.f32.sigmoid = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x8,
        .init.f32_sigmoid = xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params,
        .element_tile = 8,
      };
    }
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.sqr = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqr_ukernel__avx512f_x16,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.sqr = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqr_ukernel__avx_x16,
        .init.f32_default = xnn_init_f32_default_avx_params,
        .element_tile = 16,
      };
    } else {
      xnn_params.f32.sqr = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqr_ukernel__sse_x8,
        .element_tile = 8,
      };
    }
    if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.sqrt = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqrt_ukernel__avx_sqrt_x8,
        .init.f32_sqrt = xnn_init_f32_sqrt_avx_params,
        .element_tile = 8,
      };
    } else {
      xnn_params.f32.sqrt = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqrt_ukernel__sse_sqrt_x4,
        .element_tile = 4,
      };
    }
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__avx512f_2x16,
        .row_tile = 2,
        .channel_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__avx_2x16,
        .row_tile = 2,
        .channel_tile = 16,
      };
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.f32.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__sse41_2x8,
        .row_tile = 2,
        .channel_tile = 8,
      };
    } else {
      xnn_params.f32.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__sse2_2x8,
        .row_tile = 2,
        .channel_tile = 8,
      };
    }
    xnn_params.f32.raddstoreexpminusmax = (struct raddstoreexpminusmax_parameters) {
      .ukernel = xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x20_acc2,
      .init = xnn_init_f32_expminus_sse2_rr2_p5_params,
      .element_tile = 20,
    };
    xnn_params.f32.rmax = xnn_f32_rmax_ukernel__sse;
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_minmax_ukernel__avx512f_x32,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__avx512f_x32,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__avx512f_x32,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 32,
      };
      xnn_params.f32.vdiv = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_minmax_ukernel__avx512f_x32,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_minmax_ukernel__avx512f_x32,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_minmax_ukernel__avx512f_x32,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 32,
      };
      xnn_params.f32.vmax = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__avx512f_x32,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__avx512f_x32,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__avx512f_x32,
        .element_tile = 32,
      };
      xnn_params.f32.vmin = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__avx512f_x32,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__avx512f_x32,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__avx512f_x32,
        .element_tile = 32,
      };
      xnn_params.f32.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_minmax_ukernel__avx512f_x32,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__avx512f_x32,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__avx512f_x32,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 32,
      };
      xnn_params.f32.vsub = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_minmax_ukernel__avx512f_x32,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_minmax_ukernel__avx512f_x32,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_minmax_ukernel__avx512f_x32,
        .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
        .element_tile = 32,
      };
      xnn_params.f32.vsqrdiff = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiff_ukernel__avx512f_x32,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__avx512f_x32,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__avx512f_x32,
        .element_tile = 32,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.f32.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_minmax_ukernel__avx_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__avx_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__avx_x16,
        .init.f32_minmax = xnn_init_f32_minmax_avx_params,
        .element_tile = 16,
      };
      xnn_params.f32.vdiv = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_minmax_ukernel__avx_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_minmax_ukernel__avx_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_minmax_ukernel__avx_x16,
        .init.f32_minmax = xnn_init_f32_minmax_avx_params,
        .element_tile = 16,
      };
      xnn_params.f32.vmax = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__avx_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__avx_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__avx_x16,
        .init.f32_default = xnn_init_f32_default_avx_params,
        .element_tile = 16,
      };
      xnn_params.f32.vmin = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__avx_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__avx_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__avx_x16,
        .init.f32_default = xnn_init_f32_default_avx_params,
        .element_tile = 16,
      };
      xnn_params.f32.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_minmax_ukernel__avx_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__avx_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__avx_x16,
        .init.f32_minmax = xnn_init_f32_minmax_avx_params,
        .element_tile = 16,
      };
      xnn_params.f32.vsub = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_minmax_ukernel__avx_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_minmax_ukernel__avx_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_minmax_ukernel__avx_x16,
        .init.f32_minmax = xnn_init_f32_minmax_avx_params,
        .element_tile = 16,
      };
      xnn_params.f32.vsqrdiff = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiff_ukernel__avx_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__avx_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__avx_x16,
        .init.f32_default = xnn_init_f32_default_avx_params,
        .element_tile = 16,
      };
    } else {
      xnn_params.f32.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_minmax_ukernel__sse_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__sse_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__sse_x8,
        .init.f32_minmax = xnn_init_f32_minmax_sse_params,
        .element_tile = 8,
      };
      xnn_params.f32.vdiv = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_minmax_ukernel__sse_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_minmax_ukernel__sse_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_minmax_ukernel__sse_x8,
        .init.f32_minmax = xnn_init_f32_minmax_sse_params,
        .element_tile = 8,
      };
      xnn_params.f32.vmax = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__sse_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__sse_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__sse_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vmin = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__sse_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__sse_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__sse_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_minmax_ukernel__sse_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__sse_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__sse_x8,
        .init.f32_minmax = xnn_init_f32_minmax_sse_params,
        .element_tile = 8,
      };
      xnn_params.f32.vsub = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_minmax_ukernel__sse_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_minmax_ukernel__sse_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_minmax_ukernel__sse_x8,
        .init.f32_minmax = xnn_init_f32_minmax_sse_params,
        .element_tile = 8,
      };
      xnn_params.f32.vsqrdiff = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiff_ukernel__sse_x8,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__sse_x8,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__sse_x8,
        .element_tile = 8,
      };
    }
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x,
      .init.f32 = xnn_init_f32_minmax_sse_params,
      .channel_tile = 4,
      .row_tile = 2,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
      // Sparse microkernels on x86 currently target only SSE, and on processors
      // with AVX ISA dense inference is expected to be faster than sparse.
      if (!cpuinfo_has_x86_avx()) {
        init_flags |= XNN_INIT_FLAG_CHW_OPT;
      }

      xnn_params.f32.spmm = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_32x1__sse,
        .mr = 32,
        .nr = 1,
      };
      xnn_params.f32.conv_hwc2chw_3x3c3s2 = (struct conv_hwc2chw_parameters) {
        .ukernel_with_symm_padding =
          (xnn_conv_hwc2chw_ukernel_function) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2,
        .output_channel_tile = 4,
        .output_height_tile = 2,
        .output_width_tile = 2,
      };
      if (cpuinfo_has_x86_ssse3()) {
        xnn_params.f32.dwconv2d_chw_3x3 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_2x4_acc2,
          .output_width_tile = 4,
          .output_height_tile = 2,
        };
      } else {
        xnn_params.f32.dwconv2d_chw_3x3 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_2x4_acc2,
          .output_width_tile = 4,
          .output_height_tile = 2,
        };
      }
      xnn_params.f32.dwconv2d_chw_3x3s2 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4_acc3,
        .output_width_tile = 4,
        .output_height_tile = 1,
      };
      xnn_params.f32.dwconv2d_chw_5x5 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_4x4,
        .output_width_tile = 4,
        .output_height_tile = 4,
      };
      xnn_params.f32.dwconv2d_chw_5x5s2 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_2x4,
        .output_width_tile = 4,
        .output_height_tile = 2,
      };
      xnn_params.f32.gavgpool_cw = (struct gavgpool_cw_parameters) {
        .ukernel = (xnn_gavgpool_cw_ukernel_function) xnn_f32_gavgpool_cw_ukernel__sse_x4,
        .channel_tile = 4,
      };
      xnn_params.f32.ibilinear_chw = (struct ibilinear_chw_parameters) {
        .ukernel = (xnn_ibilinear_chw_ukernel_function) xnn_f32_ibilinear_chw_ukernel__sse_p8,
        .channel_tile = 1,
        .pixel_tile = 8,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /*************************** VCVT x86 micro-kernels ***************************/
  #ifndef XNN_NO_VCVT_OPERATORS
    init_flags |= XNN_INIT_FLAG_VCVT;

    if (cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__avx512skx_x16,
        .element_tile = 16,
      };
      xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__avx512skx_x16,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_f16c()) {
      xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__f16c_x16,
        .element_tile = 16,
      };
      xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__f16c_x16,
        .init.f32_f16_cvt = xnn_init_f32_f16_cvt_f16c_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__avx_int16_x16,
        .init.f16_f32_cvt = xnn_init_f16_f32_cvt_sse_int16_params,
        .element_tile = 16,
      };
      xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__avx_x24,
        .init.f32_f16_cvt = xnn_init_f32_f16_cvt_sse2_params,
        .element_tile = 24,
      };
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__sse41_int16_x16,
        .init.f16_f32_cvt = xnn_init_f16_f32_cvt_sse_int16_params,
        .element_tile = 16,
      };
      xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__sse41_x8,
        .init.f32_f16_cvt = xnn_init_f32_f16_cvt_sse2_params,
        .element_tile = 8,
      };
    } else {
      xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__sse2_int16_x32,
        .init.f16_f32_cvt = xnn_init_f16_f32_cvt_sse_int16_params,
        .element_tile = 32,
      };
      xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__sse2_x16,
        .init.f32_f16_cvt = xnn_init_f32_f16_cvt_sse2_params,
        .element_tile = 16,
      };
    }
    if (cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qs8_vcvt_ukernel__avx512skx_x128,
        .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_avx512_params,
        .element_tile = 128,
      };
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qs8_vcvt_ukernel__avx2_x64,
        .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_avx2_params,
        .element_tile = 64,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qs8_vcvt_ukernel__avx_x32,
        .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_avx_params,
        .element_tile = 32,
      };
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qs8_vcvt_ukernel__sse41_x32,
        .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_sse4_params,
        .element_tile = 32,
      };
    } else {
      xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qs8_vcvt_ukernel__sse2_x32,
        .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_sse2_params,
        .element_tile = 32,
      };
    }
    if (cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__avx512skx_x128,
        .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_avx512_params,
        .element_tile = 128,
      };
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__avx2_x64,
        .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_avx2_params,
        .element_tile = 64,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__avx_x32,
        .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_avx_params,
        .element_tile = 32,
      };
    } else {
      xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__sse2_x32,
        .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_sse2_params,
        .element_tile = 32,
      };
    }
    if (cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl()) {
      xnn_params.vcvt.qs8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qs8_f32_vcvt_ukernel__avx512skx_x32,
        .init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_avx512_params,
        .element_tile = 32,
      };
      xnn_params.vcvt.qu8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qu8_f32_vcvt_ukernel__avx512skx_x32,
        .init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_avx512_params,
        .element_tile = 32,
      };
    } else if (cpuinfo_has_x86_avx2()) {
      xnn_params.vcvt.qs8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qs8_f32_vcvt_ukernel__avx2_x16,
        .init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_avx_params,
        .element_tile = 16,
      };
      xnn_params.vcvt.qu8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qu8_f32_vcvt_ukernel__avx2_x16,
        .init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_avx_params,
        .element_tile = 16,
      };
    } else if (cpuinfo_has_x86_avx()) {
      xnn_params.vcvt.qs8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qs8_f32_vcvt_ukernel__avx_x32,
        .init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_avx_params,
        .element_tile = 32,
      };
      xnn_params.vcvt.qu8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qu8_f32_vcvt_ukernel__avx_x32,
        .init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_avx_params,
        .element_tile = 32,
      };
    } else if (cpuinfo_has_x86_sse4_1()) {
      xnn_params.vcvt.qs8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qs8_f32_vcvt_ukernel__sse41_x16,
        .init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_sse4_params,
        .element_tile = 16,
      };
      xnn_params.vcvt.qu8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qu8_f32_vcvt_ukernel__sse41_x16,
        .init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_sse4_params,
        .element_tile = 16,
      };
    } else {
      xnn_params.vcvt.qs8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qs8_f32_vcvt_ukernel__sse2_x32,
        .init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_sse2_params,
        .element_tile = 32,
      };
      xnn_params.vcvt.qu8_to_f32 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_qu8_f32_vcvt_ukernel__sse2_x32,
        .init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_sse2_params,
        .element_tile = 32,
      };
    }
  #endif  // XNN_NO_VCVT_OPERATORS

  /**************************** X32 x86 micro-kernels ****************************/
  #ifndef XNN_NO_X32_OPERATORS
    init_flags |= XNN_INIT_FLAG_X32;

    xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__sse2;
    xnn_params.x32.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__sse2,
      .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__sse2,
      .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__sse2,
      .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__sse2,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
      xnn_params.x32.depthtospace2d_chw2hwc = (struct depthtospace2d_chw2hwc_parameters) {
        .ukernel = (xnn_depthtospace2d_chw2hwc_ukernel_function) xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar,
        .channel_tile = 1,
        .pixel_tile = 1,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_X32_OPERATORS

  /**************************** XX x86 micro-kernels ****************************/
  #ifndef XNN_NO_XX_OPERATORS
    init_flags |= XNN_INIT_FLAG_XX;

    xnn_params.xx.copy = (xnn_univector_ukernel_function) xnn_xx_copy_ukernel__memcpy;
    xnn_params.xx.fill = (struct fill_parameters) {
      .ukernel = (xnn_fill_ukernel_function) xnn_xx_fill_ukernel__sse2_x64,
      .row_tile = 1,
    };
    xnn_params.xx.pad = (struct pad_parameters) {
      .ukernel = (xnn_pad_ukernel_function) xnn_xx_pad_ukernel__sse2,
      .row_tile = 1,
    };
  #endif

#elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

  /**************************** QC8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QC8;

    #if defined(XNN_WASMSIMD_VERSION) && (XNN_WASMSIMD_VERSION >= 88)
      xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_wasmsimd_params;
      xnn_params.qc8.gemm.mr = 4;
      xnn_params.qc8.gemm.nr = 4;
      xnn_params.qc8.gemm.log2_kr = 1;
      xnn_params.qc8.gemm.log2_sr = 2;
    #else
      xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_mul16_ld64);
      xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_mul16_ld64);
      xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul16_ld64);
      xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul16_ld64);
      xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_wasmsimd_params;
      xnn_params.qc8.gemm.mr = 3;
      xnn_params.qc8.gemm.nr = 4;
      xnn_params.qc8.gemm.log2_kr = 3;
    #endif

    xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__wasmsimd_mul16_add16;
    xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_wasmsimd_params;
    xnn_params.qc8.dwconv[0].channel_tile = 16;
    xnn_params.qc8.dwconv[0].primary_tile = 9;
    xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__wasmsimd_mul16_add16;
    xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_wasmsimd_params;
    xnn_params.qc8.dwconv[1].channel_tile = 16;
    xnn_params.qc8.dwconv[1].primary_tile = 25;
  #endif  // XNN_NO_QC8_OPERATORS

  /**************************** QS8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QS8;

    #if defined(XNN_WASMSIMD_VERSION) && (XNN_WASMSIMD_VERSION >= 88)
      xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_wasmsimd_params;
      xnn_params.qs8.gemm.mr = 4;
      xnn_params.qs8.gemm.nr = 4;
      xnn_params.qs8.gemm.log2_kr = 1;
      xnn_params.qs8.gemm.log2_sr = 2;
    #else  // XNN_WASMSIMD_VERSION >= 88
      xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_mul16_ld64);
      xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_mul16_ld64);
      xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul16_ld64);
      xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul16_ld64);
      xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_wasmsimd_params;
      xnn_params.qs8.gemm.mr = 3;
      xnn_params.qs8.gemm.nr = 4;
      xnn_params.qs8.gemm.log2_kr = 3;
    #endif  // XNN_WASMSIMD_VERSION >= 88

    xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__wasmsimd_mul16_add16;
    xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_wasmsimd_params;
    xnn_params.qs8.dwconv[0].channel_tile = 16;
    xnn_params.qs8.dwconv[0].primary_tile = 9;
    xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__wasmsimd_mul16_add16;
    xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_wasmsimd_params;
    xnn_params.qs8.dwconv[1].channel_tile = 16;
    xnn_params.qs8.dwconv[1].primary_tile = 25;

    xnn_params.qs8.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16,
      .init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params,
      .update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_wasmsimd_params,
      .row_tile = 7,
      .channel_tile = 16,
    };

    xnn_params.qs8.vadd = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__wasmsimd_x32,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32,
      .init.qs8_addsub = xnn_init_qs8_add_minmax_wasmsimd_params,
      .element_tile = 32,
    };
    xnn_params.qs8.vmul = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmul_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8,
      .init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_wasmsimd_params,
      .element_tile = 8,
    };
  #endif  // XNN_NO_QS8_OPERATORS

  /**************************** QU8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_QU8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QU8;

    #if defined(XNN_WASMSIMD_VERSION) && (XNN_WASMSIMD_VERSION >= 88)
      xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_wasmsimd_params;
      xnn_params.qu8.gemm.mr = 4;
      xnn_params.qu8.gemm.nr = 4;
      xnn_params.qu8.gemm.log2_kr = 1;
      xnn_params.qu8.gemm.log2_sr = 2;
    #else  // XNN_WASMSIMD_VERSION >= 88
      xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_mul32_ld64);
      xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_mul32_ld64);
      xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul32_ld64);
      xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul32_ld64);
      xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_wasmsimd_params;
      xnn_params.qu8.gemm.mr = 3;
      xnn_params.qu8.gemm.nr = 4;
      xnn_params.qu8.gemm.log2_kr = 3;
    #endif  // XNN_WASMSIMD_VERSION >= 88

    xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__wasmsimd_mul16;
    xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_wasmsimd_params;
    xnn_params.qu8.dwconv[0].channel_tile = 8;
    xnn_params.qu8.dwconv[0].primary_tile = 9;
    xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__wasmsimd_mul16;
    xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_wasmsimd_params;
    xnn_params.qu8.dwconv[1].channel_tile = 8;
    xnn_params.qu8.dwconv[1].primary_tile = 25;

    xnn_params.qu8.avgpool = (struct avgpool_parameters) {
      .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9x__scalar_c1,
      .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9p8x__scalar_c1,
      .init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 1,
    };
    xnn_params.qu8.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16,
      .init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_wasmsimd_params,
      .update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_wasmsimd_params,
      .row_tile = 7,
      .channel_tile = 16,
    };

    xnn_params.qu8.vadd = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__wasmsimd_x32,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__wasmsimd_x32,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__wasmsimd_x32,
      .init.qu8_addsub = xnn_init_qu8_add_minmax_wasmsimd_params,
      .element_tile = 32,
    };
    xnn_params.qu8.vmul = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmul_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8,
      .init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_wasmsimd_params,
      .element_tile = 8,
    };
  #endif  // XNN_NO_QU8_OPERATORS

  /**************************** S8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_S8_OPERATORS
    init_flags |= XNN_INIT_FLAG_S8;

    xnn_params.s8.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_s8_vclamp_ukernel__wasmsimd_x64,
      .init.s8_minmax = xnn_init_s8_minmax_wasmsimd_params,
      .element_tile = 64,
    };
    #if defined(XNN_WASMSIMD_VERSION) && (XNN_WASMSIMD_VERSION >= 88)
      xnn_params.s8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c8,
        .pixel_tile = 1,
        .channel_tile = 8,
      };
    #else  // XNN_WASMSIMD_VERSION >= 88
      xnn_params.s8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c8,
        .pixel_tile = 1,
        .channel_tile = 8,
      };
    #endif  // XNN_WASMSIMD_VERSION >= 88
    xnn_params.s8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_s8_maxpool_minmax_ukernel_9p8x__wasmsimd_c16,
      .init.s8 = xnn_init_s8_minmax_wasmsimd_params,
      .mr = 9,
      .qr = 8,
    };
  #endif  // XNN_NO_S8_OPERATORS

  /**************************** U8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_U8_OPERATORS
    init_flags |= XNN_INIT_FLAG_U8;

    xnn_params.u8.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_u8_vclamp_ukernel__wasmsimd_x64,
      .init.u8_minmax = xnn_init_u8_minmax_wasmsimd_params,
      .element_tile = 64,
    };
    #if defined(XNN_WASMSIMD_VERSION) && (XNN_WASMSIMD_VERSION >= 88)
      xnn_params.u8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_u8_ibilinear_ukernel__wasmsimd_dot16x2_c8,
        .pixel_tile = 1,
        .channel_tile = 8,
      };
    #else  // XNN_WASMSIMD_VERSION >= 88
      xnn_params.u8.ibilinear = (struct ibilinear_parameters) {
        .ukernel = (xnn_ibilinear_ukernel_function) xnn_u8_ibilinear_ukernel__wasmsimd_mul32_c8,
        .pixel_tile = 1,
        .channel_tile = 8,
      };
    #endif  // XNN_WASMSIMD_VERSION >= 88
    xnn_params.u8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_minmax_ukernel_9p8x__wasmsimd_c16,
      .init.u8 = xnn_init_u8_minmax_wasmsimd_params,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    xnn_params.u8.rmax = xnn_u8_rmax_ukernel__scalar;
  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_X8_OPERATORS
    init_flags |= XNN_INIT_FLAG_X8;

    xnn_params.x8.lut = xnn_x8_lut_ukernel__scalar_x4;
    xnn_params.x8.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__scalar,
      .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__scalar,
      .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__scalar,
      .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__scalar,
    };
  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F32 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_F32_OPERATORS
    init_flags |= XNN_INIT_FLAG_F32;

    if (is_wasm_x86) {
      xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat);
      xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_splat);
      xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat);
      xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat);
      xnn_params.f32.gemm.relu.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_splat);
      xnn_params.f32.gemm.relu.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_splat);
      xnn_params.f32.gemm.relu.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat);
      xnn_params.f32.gemm.relu.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat);
      xnn_params.f32.gemm.linear.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__wasmsimd_splat);
      xnn_params.f32.gemm.linear.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__wasmsimd_splat);
      xnn_params.f32.gemm.linear.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__wasmsimd_splat);
      xnn_params.f32.gemm.linear.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__wasmsimd_splat);
      xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      xnn_params.f32.gemm.mr = 4;
      xnn_params.f32.gemm.nr = 8;

      xnn_params.f32.gemm2.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_x86);
      xnn_params.f32.gemm2.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_x86);
      xnn_params.f32.gemm2.linear.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x2c4__wasmsimd);
      xnn_params.f32.gemm2.linear.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2c4__wasmsimd);
      xnn_params.f32.gemm2.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      xnn_params.f32.gemm2.mr = 4;
      xnn_params.f32.gemm2.nr = 2;
      xnn_params.f32.gemm2.log2_kr = 2;
    } else {
      xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat);
      xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_splat);
      xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat);
      xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat);
      xnn_params.f32.gemm.relu.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_splat);
      xnn_params.f32.gemm.relu.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_splat);
      xnn_params.f32.gemm.relu.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat);
      xnn_params.f32.gemm.relu.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat);
      xnn_params.f32.gemm.linear.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_5x8__wasmsimd_splat);
      xnn_params.f32.gemm.linear.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_5x8__wasmsimd_splat);
      xnn_params.f32.gemm.linear.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__wasmsimd_splat);
      xnn_params.f32.gemm.linear.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__wasmsimd_splat);
      xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      xnn_params.f32.gemm.mr = 5;
      xnn_params.f32.gemm.nr = 8;

      xnn_params.f32.gemm2.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_arm);
      xnn_params.f32.gemm2.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_arm);
      xnn_params.f32.gemm2.linear.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x2c4__wasmsimd);
      xnn_params.f32.gemm2.linear.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2c4__wasmsimd);
      xnn_params.f32.gemm2.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      xnn_params.f32.gemm2.mr = 4;
      xnn_params.f32.gemm2.nr = 2;
      xnn_params.f32.gemm2.log2_kr = 2;
    }

    if (is_wasm_x86) {
      xnn_params.f32.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x3__wasmsimd_x86;
      xnn_params.f32.dwconv[0].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up8x3__wasmsimd;
      xnn_params.f32.dwconv[0].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      xnn_params.f32.dwconv[0].channel_tile = 8;
      xnn_params.f32.dwconv[0].primary_tile = 3;

      xnn_params.f32.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86;
      xnn_params.f32.dwconv[1].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up8x4__wasmsimd;
      xnn_params.f32.dwconv[1].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      xnn_params.f32.dwconv[1].channel_tile = 8;
      xnn_params.f32.dwconv[1].primary_tile = 4;

      xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86;
      xnn_params.f32.dwconv[2].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up8x9__wasmsimd;
      xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      xnn_params.f32.dwconv[2].channel_tile = 8;
      xnn_params.f32.dwconv[2].primary_tile = 9;
    } else {
      xnn_params.f32.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up4x3__wasmsimd_arm;
      xnn_params.f32.dwconv[0].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up4x3__wasmsimd;
      xnn_params.f32.dwconv[0].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      xnn_params.f32.dwconv[0].channel_tile = 4;
      xnn_params.f32.dwconv[0].primary_tile = 3;

      xnn_params.f32.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm;
      xnn_params.f32.dwconv[1].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up4x4__wasmsimd;
      xnn_params.f32.dwconv[1].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      xnn_params.f32.dwconv[1].channel_tile = 4;
      xnn_params.f32.dwconv[1].primary_tile = 4;

      xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm;
      xnn_params.f32.dwconv[2].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up4x9__wasmsimd;
      xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      xnn_params.f32.dwconv[2].channel_tile = 4;
      xnn_params.f32.dwconv[2].primary_tile = 9;
    }

    xnn_params.f32.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm;
    xnn_params.f32.dwconv[3].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up4x25__wasmsimd;
    xnn_params.f32.dwconv[3].init.f32 = xnn_init_f32_minmax_wasmsimd_params;
    xnn_params.f32.dwconv[3].channel_tile = 4;
    xnn_params.f32.dwconv[3].primary_tile = 25;

    if (is_wasm_x86) {
      xnn_params.f32.avgpool = (struct avgpool_parameters) {
        .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9x__wasmsimd_x86_c4,
        .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9p8x__wasmsimd_x86_c4,
        .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
        .primary_tile = 9,
        .incremental_tile = 8,
        .channel_tile = 4,
      };
      xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
        .unipass = (xnn_pavgpool_unipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9x__wasmsimd_x86_c4,
        .multipass = (xnn_pavgpool_multipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9p8x__wasmsimd_x86_c4,
        .primary_tile = 9,
        .incremental_tile = 8,
        .channel_tile = 4,
      };
      xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4,
        .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
        .update.f32 = xnn_update_f32_scaleminmax_scalar_params,
        .row_tile = 7,
        .channel_tile = 4,
      };
    } else {
      xnn_params.f32.avgpool = (struct avgpool_parameters) {
        .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9x__wasmsimd_arm_c4,
        .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9p8x__wasmsimd_arm_c4,
        .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
        .primary_tile = 9,
        .incremental_tile = 8,
        .channel_tile = 4,
      };
      xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
        .unipass = (xnn_pavgpool_unipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9x__wasmsimd_arm_c4,
        .multipass = (xnn_pavgpool_multipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9p8x__wasmsimd_arm_c4,
        .primary_tile = 9,
        .incremental_tile = 8,
        .channel_tile = 4,
      };
      xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
        .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4,
        .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4,
        .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
        .update.f32 = xnn_update_f32_scaleminmax_scalar_params,
        .row_tile = 7,
        .channel_tile = 4,
      };
    }
    if (is_wasm_x86) {
      xnn_params.f32.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4,
        .init.f32 = xnn_init_f32_minmax_wasmsimd_params,
        .mr = 9,
        .qr = 8,
      };
    } else {
      xnn_params.f32.maxpool = (struct maxpool_parameters) {
        .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4,
        .init.f32 = xnn_init_f32_minmax_wasmsimd_params,
        .mr = 9,
        .qr = 8,
      };
    }
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__wasmsimd_c4,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__wasmsimd_c4,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_multipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__wasmsimd_c4,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.ibilinear = (struct ibilinear_parameters) {
      .ukernel = (xnn_ibilinear_ukernel_function) xnn_f32_ibilinear_ukernel__wasmsimd_c8,
      .pixel_tile = 1,
      .channel_tile = 8,
    };
    xnn_params.f32.abs = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vabs_ukernel__wasmsimd_x8,
      .init.f32_abs = xnn_init_f32_abs_wasmsimd_params,
      .element_tile = 16,
    };
    if (is_wasm_x86) {
      xnn_params.f32.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vclamp_ukernel__wasmsimd_x86_x8,
        .init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params,
        .element_tile = 8,
      };
    } else {
      xnn_params.f32.clamp = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vclamp_ukernel__wasmsimd_arm_x8,
        .init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params,
        .element_tile = 8,
      };
    }
    if (is_wasm_x86) {
      xnn_params.f32.elu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x20,
        .init.f32_elu = xnn_init_f32_elu_wasmsimd_rr2_p6_params,
        .element_tile = 20,
      };
    } else {
      xnn_params.f32.elu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x20,
        .init.f32_elu = xnn_init_f32_elu_wasmsimd_rr2_p6_params,
        .element_tile = 20,
      };
    }
    xnn_params.f32.hswish = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vhswish_ukernel__wasmsimd_x16,
      .init.f32_hswish = xnn_init_f32_hswish_wasmsimd_params,
      .element_tile = 16,
    };
    if (is_wasm_x86) {
      xnn_params.f32.lrelu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vlrelu_ukernel__wasmsimd_minmax_x8,
        .init.f32_lrelu = xnn_init_f32_lrelu_wasmsimd_params,
        .element_tile = 8,
      };
    } else {
      xnn_params.f32.lrelu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vlrelu_ukernel__wasmsimd_bitselect_x8,
        .init.f32_lrelu = xnn_init_f32_lrelu_wasmsimd_params,
        .element_tile = 8,
      };
    }
    xnn_params.f32.neg = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vneg_ukernel__wasmsimd_x8,
      .init.f32_neg = xnn_init_f32_neg_wasmsimd_params,
      .element_tile = 16,
    };
    xnn_params.f32.relu = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrelu_ukernel__wasmsimd_x16,
      .element_tile = 16,
    };
    #if defined(XNN_WASMSIMD_VERSION) && (XNN_WASMSIMD_VERSION >= 91)
      xnn_params.f32.rndne = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__wasmsimd_native_x8,
        .element_tile = 8,
      };
      xnn_params.f32.rndz = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__wasmsimd_native_x8,
        .element_tile = 8,
      };
      xnn_params.f32.rndu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__wasmsimd_native_x8,
        .element_tile = 8,
      };
      xnn_params.f32.rndd = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__wasmsimd_native_x8,
        .element_tile = 8,
      };
    #else  // XNN_WASMSIMD_VERSION >= 91
      xnn_params.f32.rndne = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__wasmsimd_addsub_x8,
        .init.f32_rnd = xnn_init_f32_rnd_wasmsimd_params,
        .element_tile = 8,
      };
      if (is_wasm_x86) {
        xnn_params.f32.rndz = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__wasmsimd_addsub_x8,
          .init.f32_rnd = xnn_init_f32_rnd_wasmsimd_params,
          .element_tile = 8,
        };
      } else {
        xnn_params.f32.rndz = (struct vunary_parameters) {
          .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__wasmsimd_cvt_x8,
          .init.f32_rnd = xnn_init_f32_rnd_wasmsimd_params,
          .element_tile = 8,
        };
      }
      xnn_params.f32.rndu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__wasmsimd_addsub_x8,
        .init.f32_rnd = xnn_init_f32_rnd_wasmsimd_params,
        .element_tile = 8,
      };
      xnn_params.f32.rndd = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__wasmsimd_addsub_x8,
        .init.f32_rnd = xnn_init_f32_rnd_wasmsimd_params,
        .element_tile = 8,
      };
    #endif  // XNN_WASMSIMD_VERSION >= 91
    xnn_params.f32.sigmoid = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x16,
      .init.f32_sigmoid = xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params,
      .element_tile = 16,
    };
    xnn_params.f32.sqr = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqr_ukernel__wasmsimd_x8,
      .element_tile = 16,
    };
    xnn_params.f32.sqrt = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_x8,
      .element_tile = 8,
    };
    if (is_wasm_x86) {
      xnn_params.f32.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__wasmsimd_minmax_2x8,
        .row_tile = 2,
        .channel_tile = 8,
      };
    } else {
      xnn_params.f32.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__wasmsimd_bitselect_2x8,
        .row_tile = 2,
        .channel_tile = 8,
      };
    }
    xnn_params.f32.raddstoreexpminusmax = (struct raddstoreexpminusmax_parameters) {
      .ukernel = xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x16_acc2,
      .init = xnn_init_f32_expminus_wasmsimd_rr2_p5_params,
      .element_tile = 16,
    };
    if (is_wasm_x86) {
      xnn_params.f32.rmax = xnn_f32_rmax_ukernel__wasmsimd_x86;
      xnn_params.f32.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_minmax_ukernel__wasmsimd_x86_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__wasmsimd_x86_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__wasmsimd_x86_x16,
        .linear.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_ukernel__wasmsimd_x16,
        .linear.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__wasmsimd_x16,
        .linear.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__wasmsimd_x16,
        .init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params,
        .element_tile = 16,
      };
      xnn_params.f32.vdiv = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_minmax_ukernel__wasmsimd_x86_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_minmax_ukernel__wasmsimd_x86_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_x16,
        .linear.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_ukernel__wasmsimd_x16,
        .linear.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_ukernel__wasmsimd_x16,
        .linear.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_ukernel__wasmsimd_x16,
        .init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params,
        .element_tile = 16,
      };
      xnn_params.f32.vmax = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__wasmsimd_x86_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__wasmsimd_x86_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__wasmsimd_x86_x16,
        .element_tile = 16,
      };
      xnn_params.f32.vmin = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__wasmsimd_x86_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__wasmsimd_x86_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__wasmsimd_x86_x16,
        .element_tile = 16,
      };
      xnn_params.f32.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_minmax_ukernel__wasmsimd_x86_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__wasmsimd_x86_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__wasmsimd_x86_x16,
        .linear.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_ukernel__wasmsimd_x16,
        .linear.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__wasmsimd_x16,
        .linear.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__wasmsimd_x16,
        .init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params,
        .element_tile = 16,
      };
      xnn_params.f32.vsub = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_minmax_ukernel__wasmsimd_x86_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_minmax_ukernel__wasmsimd_x86_x16,
        .linear.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_ukernel__wasmsimd_x16,
        .linear.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_ukernel__wasmsimd_x16,
        .linear.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_ukernel__wasmsimd_x16,
        .init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params,
        .element_tile = 16,
      };
    } else {
      xnn_params.f32.rmax = xnn_f32_rmax_ukernel__wasmsimd_arm;
      xnn_params.f32.vadd = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_minmax_ukernel__wasmsimd_arm_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__wasmsimd_arm_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__wasmsimd_arm_x16,
        .linear.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_ukernel__wasmsimd_x16,
        .linear.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__wasmsimd_x16,
        .linear.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__wasmsimd_x16,
        .init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params,
        .element_tile = 16,
      };
      xnn_params.f32.vdiv = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_minmax_ukernel__wasmsimd_arm_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_minmax_ukernel__wasmsimd_arm_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_x16,
        .linear.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_ukernel__wasmsimd_x16,
        .linear.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_ukernel__wasmsimd_x16,
        .linear.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_ukernel__wasmsimd_x16,
        .init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params,
        .element_tile = 16,
      };
      xnn_params.f32.vmax = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__wasmsimd_arm_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__wasmsimd_arm_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__wasmsimd_arm_x16,
        .element_tile = 16,
      };
      xnn_params.f32.vmin = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__wasmsimd_arm_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__wasmsimd_arm_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__wasmsimd_arm_x16,
        .element_tile = 16,
      };
      xnn_params.f32.vmul = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_minmax_ukernel__wasmsimd_arm_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__wasmsimd_arm_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__wasmsimd_arm_x16,
        .linear.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_ukernel__wasmsimd_x16,
        .linear.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__wasmsimd_x16,
        .linear.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__wasmsimd_x16,
        .init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params,
        .element_tile = 16,
      };
      xnn_params.f32.vsub = (struct vbinary_parameters) {
        .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_x16,
        .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_minmax_ukernel__wasmsimd_arm_x16,
        .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_minmax_ukernel__wasmsimd_arm_x16,
        .linear.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_ukernel__wasmsimd_x16,
        .linear.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_ukernel__wasmsimd_x16,
        .linear.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_ukernel__wasmsimd_x16,
        .init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params,
        .element_tile = 16,
      };
    }
    xnn_params.f32.vsqrdiff = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiff_ukernel__wasmsimd_x16,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__wasmsimd_x16,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__wasmsimd_x16,
      .element_tile = 16,
    };
    if (is_wasm_x86) {
      xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
        .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x,
        .init.f32 = xnn_init_f32_minmax_wasmsimd_params,
        .channel_tile = 4,
        .row_tile = 2,
      };
    } else {
      xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
        .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x,
        .init.f32 = xnn_init_f32_minmax_wasmsimd_params,
        .channel_tile = 4,
        .row_tile = 2,
      };
    }
    #ifndef XNN_NO_NCHW_OPERATORS
      init_flags |= XNN_INIT_FLAG_CHW_OPT;

      if (is_wasm_x86) {
        xnn_params.f32.spmm = (struct spmm_parameters) {
          .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86,
          .mr = 32,
          .nr = 1,
        };
      } else {
        xnn_params.f32.spmm = (struct spmm_parameters) {
          .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm,
          .mr = 32,
          .nr = 1,
        };
      }
      xnn_params.f32.conv_hwc2chw_3x3c3s2 = (struct conv_hwc2chw_parameters) {
        .ukernel_with_symm_padding =
          (xnn_conv_hwc2chw_ukernel_function) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2,
        .output_channel_tile = 4,
        .output_height_tile = 2,
        .output_width_tile = 2,
      };
      if (is_wasm_x86) {
        xnn_params.f32.dwconv2d_chw_3x3 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_2x4,
          .output_width_tile = 4,
          .output_height_tile = 2,
        };
        xnn_params.f32.dwconv2d_chw_3x3s2 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_1x4_acc2,
          .output_width_tile = 4,
          .output_height_tile = 1,
        };
        xnn_params.f32.dwconv2d_chw_5x5 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_3x4,
          .output_width_tile = 4,
          .output_height_tile = 3,
        };
        xnn_params.f32.dwconv2d_chw_5x5s2 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4_acc2,
          .output_width_tile = 4,
          .output_height_tile = 1,
        };
      } else {
        xnn_params.f32.dwconv2d_chw_3x3 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_2x4,
          .output_width_tile = 4,
          .output_height_tile = 2,
        };
        xnn_params.f32.dwconv2d_chw_3x3s2 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_1x4_acc4,
          .output_width_tile = 4,
          .output_height_tile = 1,
        };
        xnn_params.f32.dwconv2d_chw_5x5 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_3x4,
          .output_width_tile = 4,
          .output_height_tile = 3,
        };
        xnn_params.f32.dwconv2d_chw_5x5s2 = (struct dwconv2d_chw_parameters) {
          .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4_acc2,
          .output_width_tile = 4,
          .output_height_tile = 1,
        };
      }
      if (is_wasm_x86) {
        xnn_params.f32.gavgpool_cw = (struct gavgpool_cw_parameters) {
          .ukernel = (xnn_gavgpool_cw_ukernel_function) xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_x4,
          .channel_tile = 4,
        };
      } else {
        xnn_params.f32.gavgpool_cw = (struct gavgpool_cw_parameters) {
          .ukernel = (xnn_gavgpool_cw_ukernel_function) xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_x4,
          .channel_tile = 4,
        };
      }
      xnn_params.f32.ibilinear_chw = (struct ibilinear_chw_parameters) {
        .ukernel = (xnn_ibilinear_chw_ukernel_function) xnn_f32_ibilinear_chw_ukernel__wasmsimd_p8,
        .channel_tile = 1,
        .pixel_tile = 8,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /*************************** VCVT WAsm SIMD micro-kernels***************************/
  #ifndef XNN_NO_VCVT_OPERATORS
    init_flags |= XNN_INIT_FLAG_VCVT;

    xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x16,
      .init.f16_f32_cvt = xnn_init_f16_f32_cvt_wasmsimd_int16_params,
      .element_tile = 16,
    };
    xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__wasmsimd_x24,
      .init.f32_f16_cvt = xnn_init_f32_f16_cvt_wasmsimd_params,
      .element_tile = 24,
    };
    xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x32,
      .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_wasmsimd_magic_params,
      .element_tile = 32,
    };
    xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_x32,
      .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_wasmsimd_magic_params,
      .element_tile = 32,
    };
    xnn_params.vcvt.qs8_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_qs8_f32_vcvt_ukernel__wasmsimd_x32,
      .init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_wasmsimd_params,
      .element_tile = 32,
    };
    xnn_params.vcvt.qu8_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_qu8_f32_vcvt_ukernel__wasmsimd_x32,
      .init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_wasmsimd_params,
      .element_tile = 32,
    };
  #endif  // XNN_NO_VCVT_OPERATORS

  /**************************** X32 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_X32_OPERATORS
    init_flags |= XNN_INIT_FLAG_X32;

    xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__wasmsimd;
    xnn_params.x32.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__wasmsimd,
      .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__wasmsimd,
      .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__wasmsimd,
      .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__wasmsimd,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
      xnn_params.x32.depthtospace2d_chw2hwc = (struct depthtospace2d_chw2hwc_parameters) {
        .ukernel = (xnn_depthtospace2d_chw2hwc_ukernel_function) xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar,
        .channel_tile = 1,
        .pixel_tile = 1,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_X32_OPERATORS

  /**************************** XX WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_XX_OPERATORS
    init_flags |= XNN_INIT_FLAG_XX;

    xnn_params.xx.copy = (xnn_univector_ukernel_function) xnn_xx_copy_ukernel__memcpy;
    xnn_params.xx.fill = (struct fill_parameters) {
      .ukernel = (xnn_fill_ukernel_function) xnn_xx_fill_ukernel__wasmsimd_x64,
      .row_tile = 1,
    };
    xnn_params.xx.pad = (struct pad_parameters) {
      .ukernel = (xnn_pad_ukernel_function) xnn_xx_pad_ukernel__wasmsimd,
      .row_tile = 1,
    };
  #endif

#elif XNN_ARCH_WASM

  /**************************** QC8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_QC8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QC8;

    if (is_wasm_x86) {
      xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_scalar_imagic_params;
      xnn_params.qc8.gemm.mr = 2;
      xnn_params.qc8.gemm.nr = 2;
    } else {
      xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_scalar_fmagic_params;
      xnn_params.qc8.gemm.mr = 4;
      xnn_params.qc8.gemm.nr = 4;
    }

    if (is_wasm_x86) {
      xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up2x9__scalar_imagic;
      xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_scalar_imagic_params;
      xnn_params.qc8.dwconv[0].channel_tile = 2;
      xnn_params.qc8.dwconv[0].primary_tile = 9;
      xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up1x25__scalar_imagic;
      xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_scalar_imagic_params;
      xnn_params.qc8.dwconv[1].channel_tile = 1;
      xnn_params.qc8.dwconv[1].primary_tile = 25;
    } else {
      xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up2x9__wasm_fmagic;
      xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_scalar_fmagic_params;
      xnn_params.qc8.dwconv[0].channel_tile = 2;
      xnn_params.qc8.dwconv[0].primary_tile = 9;
      xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up2x25__wasm_fmagic;
      xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_scalar_fmagic_params;
      xnn_params.qc8.dwconv[1].channel_tile = 2;
      xnn_params.qc8.dwconv[1].primary_tile = 25;
    }
  #endif  // XNN_NO_QC8_OPERATORS

  /**************************** QS8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QS8;

    if (is_wasm_x86) {
      xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params;
      xnn_params.qs8.gemm.mr = 2;
      xnn_params.qs8.gemm.nr = 2;
    } else {
      xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qs8.gemm.mr = 4;
      xnn_params.qs8.gemm.nr = 4;
    }

    if (is_wasm_x86) {
      xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up2x9__scalar_imagic;
      xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params;
      xnn_params.qs8.dwconv[0].channel_tile = 2;
      xnn_params.qs8.dwconv[0].primary_tile = 9;
      xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up1x25__scalar_imagic;
      xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params;
      xnn_params.qs8.dwconv[1].channel_tile = 1;
      xnn_params.qs8.dwconv[1].primary_tile = 25;
    } else {
      xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up2x9__wasm_fmagic;
      xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qs8.dwconv[0].channel_tile = 2;
      xnn_params.qs8.dwconv[0].primary_tile = 9;
      xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up2x25__wasm_fmagic;
      xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qs8.dwconv[1].channel_tile = 2;
      xnn_params.qs8.dwconv[1].primary_tile = 25;
    }

    xnn_params.qs8.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4,
      .init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params,
      .update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_scalar_imagic_params,
      .row_tile = 7,
      .channel_tile = 4,
    };

    xnn_params.qs8.vadd = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__scalar_x4,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__scalar_x4,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__scalar_x4,
      .init.qs8_addsub = xnn_init_qs8_add_minmax_scalar_params,
      .element_tile = 4,
    };
    xnn_params.qs8.vmul = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmul_minmax_fp32_ukernel__scalar_x4,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_x4,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_x4,
      .init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_scalar_params,
      .element_tile = 4,
    };
  #endif  // XNN_NO_QS8_OPERATORS

  /**************************** QU8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_QU8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QU8;

    if (is_wasm_x86) {
      xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params;
      xnn_params.qu8.gemm.mr = 2;
      xnn_params.qu8.gemm.nr = 2;
    } else {
      xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qu8.gemm.mr = 4;
      xnn_params.qu8.gemm.nr = 4;
    }

    if (is_wasm_x86) {
      xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up2x9__scalar_imagic;
      xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params;
      xnn_params.qu8.dwconv[0].channel_tile = 2;
      xnn_params.qu8.dwconv[0].primary_tile = 9;
      xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up1x25__scalar_imagic;
      xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params;
      xnn_params.qu8.dwconv[1].channel_tile = 1;
      xnn_params.qu8.dwconv[1].primary_tile = 25;
    } else {
      xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up2x9__wasm_fmagic;
      xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qu8.dwconv[0].channel_tile = 2;
      xnn_params.qu8.dwconv[0].primary_tile = 9;
      xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up2x25__wasm_fmagic;
      xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params;
      xnn_params.qu8.dwconv[1].channel_tile = 2;
      xnn_params.qu8.dwconv[1].primary_tile = 25;
    }

    xnn_params.qu8.avgpool = (struct avgpool_parameters) {
      .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9x__scalar_c1,
      .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9p8x__scalar_c1,
      .init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 1,
    };
    xnn_params.qu8.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4,
      .init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params,
      .update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_scalar_imagic_params,
      .row_tile = 7,
      .channel_tile = 4,
    };

    xnn_params.qu8.vadd = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__scalar_x4,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__scalar_x4,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__scalar_x4,
      .init.qu8_addsub = xnn_init_qu8_add_minmax_scalar_params,
      .element_tile = 4,
    };
    xnn_params.qu8.vmul = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmul_minmax_fp32_ukernel__scalar_x4,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_x4,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_x4,
      .init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_scalar_params,
      .element_tile = 4,
    };
  #endif  // XNN_NO_QU8_OPERATORS

  /**************************** S8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_S8_OPERATORS
    init_flags |= XNN_INIT_FLAG_S8;

    xnn_params.s8.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_s8_vclamp_ukernel__scalar_x4,
      .init.s8_minmax = xnn_init_s8_minmax_scalar_params,
      .element_tile = 4,
    };
    xnn_params.s8.ibilinear = (struct ibilinear_parameters) {
      .ukernel = (xnn_ibilinear_ukernel_function) xnn_s8_ibilinear_ukernel__scalar_c1,
      .pixel_tile = 1,
      .channel_tile = 1,
    };
    xnn_params.s8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1,
      .init.s8 = xnn_init_s8_minmax_scalar_params,
      .mr = 9,
      .qr = 8,
    };
  #endif  // XNN_NO_S8_OPERATORS

  /**************************** U8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_U8_OPERATORS
    init_flags |= XNN_INIT_FLAG_U8;

    xnn_params.u8.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_u8_vclamp_ukernel__scalar_x4,
      .init.u8_minmax = xnn_init_u8_minmax_scalar_params,
      .element_tile = 4,
    };
    xnn_params.u8.ibilinear = (struct ibilinear_parameters) {
      .ukernel = (xnn_ibilinear_ukernel_function) xnn_u8_ibilinear_ukernel__scalar_c1,
      .pixel_tile = 1,
      .channel_tile = 1,
    };
    xnn_params.u8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1,
      .init.u8 = xnn_init_u8_minmax_scalar_params,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    xnn_params.u8.rmax = xnn_u8_rmax_ukernel__scalar;
  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_X8_OPERATORS
    init_flags |= XNN_INIT_FLAG_X8;

    xnn_params.x8.lut = xnn_x8_lut_ukernel__scalar_x4;
    xnn_params.x8.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__scalar,
      .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__scalar,
      .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__scalar,
      .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__scalar,
    };
  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F32 WAsm micro-kernels****************************/
  #ifndef XNN_NO_F32_OPERATORS
    init_flags |= XNN_INIT_FLAG_F32;

    if (is_wasm_x86) {
      xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_2x4__scalar);
      xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_2x4__scalar);
      xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x4__wasm);
      xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x4__wasm);
      xnn_params.f32.gemm.relu.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_2x4__scalar);
      xnn_params.f32.gemm.relu.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_2x4__scalar);
      xnn_params.f32.gemm.relu.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_1x4__wasm);
      xnn_params.f32.gemm.relu.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_1x4__wasm);
      xnn_params.f32.gemm.linear.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_2x4__scalar);
      xnn_params.f32.gemm.linear.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_2x4__scalar);
      xnn_params.f32.gemm.linear.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x4__wasm);
      xnn_params.f32.gemm.linear.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x4__wasm);
      xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.gemm.mr = 2;
      xnn_params.f32.gemm.nr = 4;
    } else {
      xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x4__wasm);
      xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x4__wasm);
      xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x4__wasm);
      xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x4__wasm);
      xnn_params.f32.gemm.relu.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_4x4__wasm);
      xnn_params.f32.gemm.relu.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_4x4__wasm);
      xnn_params.f32.gemm.relu.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_1x4__wasm);
      xnn_params.f32.gemm.relu.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_1x4__wasm);
      xnn_params.f32.gemm.linear.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x4__wasm);
      xnn_params.f32.gemm.linear.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x4__wasm);
      xnn_params.f32.gemm.linear.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x4__wasm);
      xnn_params.f32.gemm.linear.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x4__wasm);
      xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
      xnn_params.f32.gemm.mr = 4;
      xnn_params.f32.gemm.nr = 4;
    }
    xnn_params.f32.gemm2.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x2__wasm);
    xnn_params.f32.gemm2.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x2__wasm),
    xnn_params.f32.gemm2.linear.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x2__wasm);
    xnn_params.f32.gemm2.linear.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2__wasm),
    xnn_params.f32.gemm2.init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.gemm2.mr = 4;
    xnn_params.f32.gemm2.nr = 2;

    xnn_params.f32.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x3__wasm_acc2;
    xnn_params.f32.dwconv[0].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x3__wasm_acc2;
    xnn_params.f32.dwconv[0].init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.dwconv[0].channel_tile = 1;
    xnn_params.f32.dwconv[0].primary_tile = 3;

    xnn_params.f32.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2;
    xnn_params.f32.dwconv[1].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x4__wasm_acc2;
    xnn_params.f32.dwconv[1].init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.dwconv[1].channel_tile = 1;
    xnn_params.f32.dwconv[1].primary_tile = 4;

    xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2;
    xnn_params.f32.dwconv[2].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x9__wasm_acc2;
    xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.dwconv[2].channel_tile = 1;
    xnn_params.f32.dwconv[2].primary_tile = 9;

    xnn_params.f32.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2;
    xnn_params.f32.dwconv[3].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x25__wasm_acc2;
    xnn_params.f32.dwconv[3].init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.dwconv[3].channel_tile = 1;
    xnn_params.f32.dwconv[3].primary_tile = 25;

    xnn_params.f32.avgpool = (struct avgpool_parameters) {
      .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9x__wasm_c1,
      .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9p8x__wasm_c1,
      .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 1,
    };
    xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
      .unipass = (xnn_pavgpool_unipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9x__wasm_c1,
      .multipass = (xnn_pavgpool_multipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9p8x__wasm_c1,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 1,
    };
    xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1,
      .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
      .update.f32 = xnn_update_f32_scaleminmax_scalar_params,
      .row_tile = 7,
      .channel_tile = 1,
    };
    xnn_params.f32.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1,
      .init.f32 = xnn_init_f32_minmax_scalar_params,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__scalar_c1,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__scalar_c1,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_multipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.ibilinear = (struct ibilinear_parameters) {
      .ukernel = (xnn_ibilinear_ukernel_function) xnn_f32_ibilinear_ukernel__scalar_c2,
      .pixel_tile = 1,
      .channel_tile = 2,
    };
    xnn_params.f32.abs = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vabs_ukernel__scalar_x4,
      .element_tile = 4,
    };
    xnn_params.f32.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vclamp_ukernel__wasm_x4,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 4,
    };
    if (is_wasm_x86) {
      xnn_params.f32.hswish = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vhswish_ukernel__scalar_x4,
        .init.f32_hswish = xnn_init_f32_hswish_scalar_params,
        .element_tile = 4,
      };
    } else {
      xnn_params.f32.hswish = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vhswish_ukernel__wasm_x4,
        .init.f32_hswish = xnn_init_f32_hswish_scalar_params,
        .element_tile = 4,
      };
    }
    if (is_wasm_x86) {
      xnn_params.f32.elu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x2,
        .init.f32_elu = xnn_init_f32_elu_scalar_rr2_lut16_p3_params,
        .element_tile = 2,
      };
    } else {
      xnn_params.f32.elu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__wasm_rr2_p6_x6,
        .init.f32_elu = xnn_init_f32_elu_scalar_rr2_p6_params,
        .element_tile = 6,
      };
    }
    xnn_params.f32.lrelu = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vlrelu_ukernel__scalar_x4,
      .init.f32_lrelu = xnn_init_f32_lrelu_scalar_params,
      .element_tile = 4,
    };
    xnn_params.f32.neg = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vneg_ukernel__scalar_x4,
      .element_tile = 4,
    };
    if (is_wasm_x86) {
      xnn_params.f32.relu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrelu_ukernel__scalar_x8,
        .element_tile = 8,
      };
    } else {
      xnn_params.f32.relu = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrelu_ukernel__wasm_x8,
        .element_tile = 8,
      };
    }
    xnn_params.f32.rndne = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__scalar_libm_x4,
      .element_tile = 4,
    };
    xnn_params.f32.rndz = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__scalar_libm_x4,
      .element_tile = 4,
    };
    xnn_params.f32.rndu = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__scalar_libm_x4,
      .element_tile = 4,
    };
    xnn_params.f32.rndd = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__scalar_libm_x4,
      .element_tile = 4,
    };
    xnn_params.f32.sigmoid = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x2,
      .init.f32_sigmoid = xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params,
      .element_tile = 2,
    };
    xnn_params.f32.sqr = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqr_ukernel__scalar_x4,
      .element_tile = 4,
    };
    xnn_params.f32.sqrt = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqrt_ukernel__scalar_sqrt_x1,
      .element_tile = 1,
    };
    if (is_wasm_x86) {
      xnn_params.f32.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__scalar_2x4,
        .row_tile = 2,
        .channel_tile = 4,
      };
    } else {
      xnn_params.f32.prelu = (struct prelu_parameters) {
        .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__wasm_2x4,
        .row_tile = 2,
        .channel_tile = 4,
      };
    }
    xnn_params.f32.raddstoreexpminusmax = (struct raddstoreexpminusmax_parameters) {
      .ukernel = xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x4_acc2,
      .init = xnn_init_f32_expminus_scalar_rr2_p5_params,
      .element_tile = 4,
    };
    xnn_params.f32.rmax = xnn_f32_rmax_ukernel__scalar;
    xnn_params.f32.vadd = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_minmax_ukernel__wasm_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__wasm_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__wasm_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.vdiv = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_minmax_ukernel__wasm_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_minmax_ukernel__wasm_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_minmax_ukernel__wasm_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.vmax = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__wasm_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__wasm_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__wasm_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmin = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__wasm_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__wasm_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__wasm_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmul = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_minmax_ukernel__wasm_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__wasm_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__wasm_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.vsub = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_minmax_ukernel__wasm_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_minmax_ukernel__wasm_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_minmax_ukernel__wasm_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.vsqrdiff = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiff_ukernel__scalar_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__scalar_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__scalar_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x,
      .init.f32 = xnn_init_f32_minmax_scalar_params,
      .channel_tile = 1,
      .row_tile = 2,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
      init_flags |= XNN_INIT_FLAG_CHW_OPT;

      xnn_params.f32.spmm = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_8x1__scalar,
        .mr = 8,
        .nr = 1,
      };
      xnn_params.f32.spmm2 = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_8x2__scalar,
        .mr = 8,
        .nr = 2,
      };
      xnn_params.f32.spmm4 = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_8x4__scalar,
        .mr = 8,
        .nr = 4,
      };
      xnn_params.f32.conv_hwc2chw_3x3c3s2 = (struct conv_hwc2chw_parameters) {
        .ukernel_with_symm_padding =
          (xnn_conv_hwc2chw_ukernel_function) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1,
        .output_channel_tile = 4,
        .output_height_tile = 1,
        .output_width_tile = 1,
      };
      xnn_params.f32.dwconv2d_chw_3x3 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1_acc2,
        .output_width_tile = 1,
        .output_height_tile = 2,
      };
      xnn_params.f32.dwconv2d_chw_3x3s2 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc2,
        .output_width_tile = 1,
        .output_height_tile = 1,
      };
      xnn_params.f32.dwconv2d_chw_5x5 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc5,
        .output_width_tile = 1,
        .output_height_tile = 1,
      };
      xnn_params.f32.dwconv2d_chw_5x5s2 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc5,
        .output_width_tile = 1,
        .output_height_tile = 1,
      };
      xnn_params.f32.gavgpool_cw = (struct gavgpool_cw_parameters) {
        .ukernel = (xnn_gavgpool_cw_ukernel_function) xnn_f32_gavgpool_cw_ukernel__scalar_x1,
        .channel_tile = 1,
      };
      xnn_params.f32.ibilinear_chw = (struct ibilinear_chw_parameters) {
        .ukernel = (xnn_ibilinear_chw_ukernel_function) xnn_f32_ibilinear_chw_ukernel__scalar_p4,
        .channel_tile = 1,
        .pixel_tile = 4,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /*************************** VCVT WAsm micro-kernels***************************/
  #ifndef XNN_NO_VCVT_OPERATORS
    init_flags |= XNN_INIT_FLAG_VCVT;

    xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__scalar_x1,
      .init.f16_f32_cvt = xnn_init_f16_f32_cvt_scalar_params,
      .element_tile = 1,
    };
    xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__scalar_bitcast_x4,
      .init.f32_f16_cvt = xnn_init_f32_f16_cvt_scalar_bitcast_params,
      .element_tile = 4,
    };
    if (is_wasm_x86) {
      xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__scalar_imagic_x1,
        .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_imagic_params,
        .element_tile = 1,
      };
      xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__scalar_imagic_x1,
        .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_imagic_params,
        .element_tile = 1,
      };
    } else {
      xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_x4,
        .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_fmagic_params,
        .element_tile = 4,
      };
      xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
        .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_x4,
        .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_fmagic_params,
        .element_tile = 4,
      };
    }
    xnn_params.vcvt.qs8_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_qs8_f32_vcvt_ukernel__scalar_x1,
      .init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params,
      .element_tile = 1,
    };
    xnn_params.vcvt.qu8_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_qu8_f32_vcvt_ukernel__scalar_x1,
      .init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params,
      .element_tile = 1,
    };
  #endif  // XNN_NO_VCVT_OPERATORS

  /**************************** X32 WAsm micro-kernels****************************/
  #ifndef XNN_NO_X32_OPERATORS
    init_flags |= XNN_INIT_FLAG_X32;

    xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__scalar;
    xnn_params.x32.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__scalar,
      .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__scalar,
      .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__scalar,
      .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__scalar,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
      xnn_params.x32.depthtospace2d_chw2hwc = (struct depthtospace2d_chw2hwc_parameters) {
        .ukernel = (xnn_depthtospace2d_chw2hwc_ukernel_function) xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar,
        .channel_tile = 1,
        .pixel_tile = 1,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_X32_OPERATORS

  /**************************** XX WAsm micro-kernels****************************/
  #ifndef XNN_NO_XX_OPERATORS
    init_flags |= XNN_INIT_FLAG_XX;

    xnn_params.xx.copy = (xnn_univector_ukernel_function) xnn_xx_copy_ukernel__memcpy;
    xnn_params.xx.fill = (struct fill_parameters) {
      .ukernel = (xnn_fill_ukernel_function) xnn_xx_fill_ukernel__scalar_x16,
      .row_tile = 1,
    };
    xnn_params.xx.pad = (struct pad_parameters) {
      .ukernel = (xnn_pad_ukernel_function) xnn_xx_pad_ukernel__scalar,
      .row_tile = 1,
    };
  #endif

#elif XNN_ARCH_RISCV

  /************************** QC8 RISC-V micro-kernels **************************/
  #ifndef XNN_NO_QC8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QC8;

    xnn_params.qc8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    xnn_params.qc8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    xnn_params.qc8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qc8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    xnn_params.qc8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qc8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    xnn_params.qc8.gemm.init.qc8 = xnn_init_qs8_minmax_scalar_lrintf_params;
    xnn_params.qc8.gemm.mr = 3;
    xnn_params.qc8.gemm.nr = 4;

    xnn_params.qc8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up2x9__scalar_lrintf;
    xnn_params.qc8.dwconv[0].init.qc8 = xnn_init_qs8_minmax_scalar_lrintf_params;
    xnn_params.qc8.dwconv[0].channel_tile = 2;
    xnn_params.qc8.dwconv[0].primary_tile = 9;
    xnn_params.qc8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qc8_dwconv_minmax_fp32_ukernel_up2x25__scalar_lrintf;
    xnn_params.qc8.dwconv[1].init.qc8 = xnn_init_qs8_minmax_scalar_lrintf_params;
    xnn_params.qc8.dwconv[1].channel_tile = 2;
    xnn_params.qc8.dwconv[1].primary_tile = 25;
  #endif  // XNN_NO_QS8_OPERATORS

  /************************** QS8 RISC-V micro-kernels **************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QS8;

    xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qs8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qs8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    xnn_params.qs8.gemm.init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params;
    xnn_params.qs8.gemm.mr = 3;
    xnn_params.qs8.gemm.nr = 4;

    xnn_params.qs8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up2x9__scalar_lrintf;
    xnn_params.qs8.dwconv[0].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params;
    xnn_params.qs8.dwconv[0].channel_tile = 2;
    xnn_params.qs8.dwconv[0].primary_tile = 9;
    xnn_params.qs8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qs8_dwconv_minmax_fp32_ukernel_up2x25__scalar_lrintf;
    xnn_params.qs8.dwconv[1].init.qs8 = xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params;
    xnn_params.qs8.dwconv[1].channel_tile = 2;
    xnn_params.qs8.dwconv[1].primary_tile = 25;

    xnn_params.qs8.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1,
      .init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params,
      .update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_scalar_imagic_params,
      .row_tile = 7,
      .channel_tile = 1,
    };

    xnn_params.qs8.vadd = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vadd_minmax_ukernel__scalar_x4,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__scalar_x4,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vaddc_minmax_ukernel__scalar_x4,
      .init.qs8_addsub = xnn_init_qs8_add_minmax_scalar_params,
      .element_tile = 4,
    };
    xnn_params.qs8.vmul = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmul_minmax_fp32_ukernel__scalar_x4,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_x4,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_x4,
      .init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_scalar_params,
      .element_tile = 4,
    };
  #endif  // XNN_NO_QS8_OPERATORS

  /************************** QU8 RISC-V micro-kernels **************************/
  #ifndef XNN_NO_QU8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QU8;

    xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    xnn_params.qu8.gemm.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params;
    xnn_params.qu8.gemm.mr = 3;
    xnn_params.qu8.gemm.nr = 4;

    xnn_params.qu8.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up2x9__scalar_lrintf;
    xnn_params.qu8.dwconv[0].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params;
    xnn_params.qu8.dwconv[0].channel_tile = 2;
    xnn_params.qu8.dwconv[0].primary_tile = 9;
    xnn_params.qu8.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_qu8_dwconv_minmax_fp32_ukernel_up2x25__scalar_lrintf;
    xnn_params.qu8.dwconv[1].init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params;
    xnn_params.qu8.dwconv[1].channel_tile = 2;
    xnn_params.qu8.dwconv[1].primary_tile = 25;

    xnn_params.qu8.avgpool = (struct avgpool_parameters) {
      .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9x__scalar_c1,
      .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_qu8_avgpool_minmax_ukernel_9p8x__scalar_c1,
      .init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 1,
    };
    xnn_params.qu8.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1,
      .init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params,
      .update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_scalar_imagic_params,
      .row_tile = 7,
      .channel_tile = 1,
    };

    xnn_params.qu8.vadd = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vadd_minmax_ukernel__scalar_x4,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__scalar_x4,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vaddc_minmax_ukernel__scalar_x4,
      .init.qu8_addsub = xnn_init_qu8_add_minmax_scalar_params,
      .element_tile = 4,
    };
    xnn_params.qu8.vmul = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmul_minmax_fp32_ukernel__scalar_x4,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_x4,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_x4,
      .init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_scalar_params,
      .element_tile = 4,
    };
  #endif  // XNN_NO_QU8_OPERATORS

  /************************** S8 RISC-V micro-kernels ***************************/
  #ifndef XNN_NO_S8_OPERATORS
    init_flags |= XNN_INIT_FLAG_S8;

    xnn_params.s8.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_s8_vclamp_ukernel__scalar_x4,
      .init.s8_minmax = xnn_init_s8_minmax_scalar_params,
      .element_tile = 4,
    };
    xnn_params.s8.ibilinear = (struct ibilinear_parameters) {
      .ukernel = (xnn_ibilinear_ukernel_function) xnn_s8_ibilinear_ukernel__scalar_c1,
      .pixel_tile = 1,
      .channel_tile = 1,
    };
    xnn_params.s8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1,
      .init.s8 = xnn_init_s8_minmax_scalar_params,
      .mr = 9,
      .qr = 8,
    };
  #endif  // XNN_NO_S8_OPERATORS

  /************************** U8 RISC-V micro-kernels ***************************/
  #ifndef XNN_NO_U8_OPERATORS
    init_flags |= XNN_INIT_FLAG_U8;

    xnn_params.u8.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_u8_vclamp_ukernel__scalar_x4,
      .init.u8_minmax = xnn_init_u8_minmax_scalar_params,
      .element_tile = 4,
    };
    xnn_params.u8.ibilinear = (struct ibilinear_parameters) {
      .ukernel = (xnn_ibilinear_ukernel_function) xnn_u8_ibilinear_ukernel__scalar_c1,
      .pixel_tile = 1,
      .channel_tile = 1,
    };
    xnn_params.u8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1,
      .init.u8 = xnn_init_u8_minmax_scalar_params,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    xnn_params.u8.rmax = xnn_u8_rmax_ukernel__scalar;
  #endif  // XNN_NO_U8_OPERATORS

  /************************** X8 RISC-V micro-kernels ***************************/
  #ifndef XNN_NO_X8_OPERATORS
    init_flags |= XNN_INIT_FLAG_X8;

    xnn_params.x8.lut = xnn_x8_lut_ukernel__scalar_x4;
    xnn_params.x8.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__scalar,
      .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__scalar,
      .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__scalar,
      .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__scalar,
    };
  #endif  // XNN_NO_X8_OPERATORS

  /************************** F32 RISC-V micro-kernels **************************/
  #ifndef XNN_NO_F32_OPERATORS
    init_flags |= XNN_INIT_FLAG_F32;

    xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x4__scalar);
    xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x4__scalar);
    xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_1x4__scalar);
    xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_1x4__scalar);
    xnn_params.f32.gemm.relu.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_4x4__scalar);
    xnn_params.f32.gemm.relu.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_4x4__scalar);
    xnn_params.f32.gemm.relu.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_relu_ukernel_1x4__scalar);
    xnn_params.f32.gemm.relu.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_relu_ukernel_1x4__scalar);
    xnn_params.f32.gemm.linear.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x4__scalar);
    xnn_params.f32.gemm.linear.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x4__scalar);
    xnn_params.f32.gemm.linear.gemm1 = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x4__scalar);
    xnn_params.f32.gemm.linear.igemm1 = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x4__scalar);
    xnn_params.f32.gemm.init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.gemm.mr = 4;
    xnn_params.f32.gemm.nr = 4;

    xnn_params.f32.gemm2.minmax.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_minmax_ukernel_4x2__scalar);
    xnn_params.f32.gemm2.minmax.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_minmax_ukernel_4x2__scalar),
    xnn_params.f32.gemm2.linear.gemm = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x2__scalar);
    xnn_params.f32.gemm2.linear.igemm = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2__scalar),
    xnn_params.f32.gemm2.init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.gemm2.mr = 4;
    xnn_params.f32.gemm2.nr = 2;

    xnn_params.f32.dwconv[0].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x3__scalar_acc2;
    xnn_params.f32.dwconv[0].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x3__scalar_acc2;
    xnn_params.f32.dwconv[0].init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.dwconv[0].channel_tile = 1;
    xnn_params.f32.dwconv[0].primary_tile = 3;

    xnn_params.f32.dwconv[1].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2;
    xnn_params.f32.dwconv[1].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x4__scalar_acc2;
    xnn_params.f32.dwconv[1].init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.dwconv[1].channel_tile = 1;
    xnn_params.f32.dwconv[1].primary_tile = 4;

    xnn_params.f32.dwconv[2].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2;
    xnn_params.f32.dwconv[2].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x9__scalar_acc2;
    xnn_params.f32.dwconv[2].init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.dwconv[2].channel_tile = 1;
    xnn_params.f32.dwconv[2].primary_tile = 9;

    xnn_params.f32.dwconv[3].minmax.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2;
    xnn_params.f32.dwconv[3].linear.unipass = (xnn_dwconv_unipass_ukernel_function) xnn_f32_dwconv_ukernel_up1x25__scalar_acc2;
    xnn_params.f32.dwconv[3].init.f32 = xnn_init_f32_minmax_scalar_params;
    xnn_params.f32.dwconv[3].channel_tile = 1;
    xnn_params.f32.dwconv[3].primary_tile = 25;

    xnn_params.f32.avgpool = (struct avgpool_parameters) {
      .unipass = (xnn_avgpool_unipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9x__scalar_c1,
      .multipass = (xnn_avgpool_multipass_ukernel_function) xnn_f32_avgpool_minmax_ukernel_9p8x__scalar_c1,
      .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 1,
    };
    xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
      .unipass = (xnn_pavgpool_unipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9x__scalar_c1,
      .multipass = (xnn_pavgpool_multipass_ukernel_function) xnn_f32_pavgpool_minmax_ukernel_9p8x__scalar_c1,
      .primary_tile = 9,
      .incremental_tile = 8,
      .channel_tile = 1,
    };
    xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
      .unipass = (xnn_gavgpool_unipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1,
      .multipass = (xnn_gavgpool_multipass_ukernel_function) xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1,
      .init.f32 = xnn_init_f32_scaleminmax_scalar_params,
      .update.f32 = xnn_update_f32_scaleminmax_scalar_params,
      .row_tile = 7,
      .channel_tile = 1,
    };
    xnn_params.f32.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1,
      .init.f32 = xnn_init_f32_minmax_scalar_params,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__scalar_c1,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_unipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__scalar_c1,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_multipass_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.ibilinear = (struct ibilinear_parameters) {
      .ukernel = (xnn_ibilinear_ukernel_function) xnn_f32_ibilinear_ukernel__scalar_c2,
      .pixel_tile = 1,
      .channel_tile = 2,
    };
    xnn_params.f32.abs = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vabs_ukernel__scalar_x4,
      .element_tile = 4,
    };
    xnn_params.f32.clamp = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vclamp_ukernel__scalar_x4,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 4,
    };
    xnn_params.f32.elu = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4,
      .init.f32_elu = xnn_init_f32_elu_scalar_rr2_lut16_p3_params,
      .element_tile = 4,
    };
    xnn_params.f32.hswish = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vhswish_ukernel__scalar_x4,
      .init.f32_hswish = xnn_init_f32_hswish_scalar_params,
      .element_tile = 4,
    };
    xnn_params.f32.lrelu = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vlrelu_ukernel__scalar_x4,
      .init.f32_lrelu = xnn_init_f32_lrelu_scalar_params,
      .element_tile = 4,
    };
    xnn_params.f32.neg = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vneg_ukernel__scalar_x4,
      .element_tile = 4,
    };
    xnn_params.f32.rndne = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndne_ukernel__scalar_libm_x1,
      .element_tile = 1,
    };
    xnn_params.f32.rndz = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndz_ukernel__scalar_libm_x1,
      .element_tile = 1,
    };
    xnn_params.f32.rndu = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndu_ukernel__scalar_libm_x1,
      .element_tile = 1,
    };
    xnn_params.f32.rndd = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vrndd_ukernel__scalar_libm_x1,
      .element_tile = 1,
    };
    xnn_params.f32.sigmoid = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x2,
      .init.f32_sigmoid = xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params,
      .element_tile = 2,
    };
    xnn_params.f32.sqr = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqr_ukernel__scalar_x4,
      .element_tile = 4,
    };
    xnn_params.f32.sqrt = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_vsqrt_ukernel__scalar_sqrt_x1,
      .element_tile = 1,
    };
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__scalar_2x4,
      .row_tile = 4,
      .channel_tile = 4,
    };
    xnn_params.f32.raddstoreexpminusmax = (struct raddstoreexpminusmax_parameters) {
      .ukernel = xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x4_acc2,
      .init = xnn_init_f32_expminus_scalar_rr2_p5_params,
      .element_tile = 4,
    };
    xnn_params.f32.rmax = xnn_f32_rmax_ukernel__scalar;
    xnn_params.f32.vadd = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_minmax_ukernel__scalar_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__scalar_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_minmax_ukernel__scalar_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.vdiv = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_minmax_ukernel__scalar_x2,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_minmax_ukernel__scalar_x2,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_minmax_ukernel__scalar_x2,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 2,
    };
    xnn_params.f32.vmax = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__scalar_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__scalar_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__scalar_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmin = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__scalar_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__scalar_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__scalar_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmul = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_minmax_ukernel__scalar_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__scalar_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_minmax_ukernel__scalar_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.vsub = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_minmax_ukernel__scalar_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_minmax_ukernel__scalar_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_minmax_ukernel__scalar_x8,
      .init.f32_minmax = xnn_init_f32_minmax_scalar_params,
      .element_tile = 8,
    };
    xnn_params.f32.vsqrdiff = (struct vbinary_parameters) {
      .minmax.op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiff_ukernel__scalar_x8,
      .minmax.opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__scalar_x8,
      .minmax.ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsqrdiffc_ukernel__scalar_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x,
      .init.f32 = xnn_init_f32_minmax_scalar_params,
      .channel_tile = 1,
      .row_tile = 2,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
      init_flags |= XNN_INIT_FLAG_CHW_OPT;

      xnn_params.f32.spmm = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_8x1__scalar,
        .mr = 8,
        .nr = 1,
      };
      xnn_params.f32.spmm2 = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_8x2__scalar,
        .mr = 8,
        .nr = 2,
      };
      xnn_params.f32.spmm4 = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_minmax_ukernel_8x4__scalar,
        .mr = 8,
        .nr = 4,
      };
      xnn_params.f32.conv_hwc2chw_3x3c3s2 = (struct conv_hwc2chw_parameters) {
        .ukernel_with_symm_padding =
          (xnn_conv_hwc2chw_ukernel_function) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1,
        .output_channel_tile = 4,
        .output_height_tile = 1,
        .output_width_tile = 1,
      };
      xnn_params.f32.dwconv2d_chw_3x3 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1_acc2,
        .output_width_tile = 1,
        .output_height_tile = 2,
      };
      xnn_params.f32.dwconv2d_chw_3x3s2 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc2,
        .output_width_tile = 1,
        .output_height_tile = 1,
      };
      xnn_params.f32.dwconv2d_chw_5x5 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc5,
        .output_width_tile = 1,
        .output_height_tile = 1,
      };
      xnn_params.f32.dwconv2d_chw_5x5s2 = (struct dwconv2d_chw_parameters) {
        .ukernel = (xnn_dwconv2d_chw_ukernel_function) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc5,
        .output_width_tile = 1,
        .output_height_tile = 1,
      };
      xnn_params.f32.gavgpool_cw = (struct gavgpool_cw_parameters) {
        .ukernel = (xnn_gavgpool_cw_ukernel_function) xnn_f32_gavgpool_cw_ukernel__scalar_x1,
        .channel_tile = 1,
      };
      xnn_params.f32.ibilinear_chw = (struct ibilinear_chw_parameters) {
        .ukernel = (xnn_ibilinear_chw_ukernel_function) xnn_f32_ibilinear_chw_ukernel__scalar_p4,
        .channel_tile = 1,
        .pixel_tile = 4,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /************************** VCVT RISC-V micro-kernels *************************/
  #ifndef XNN_NO_VCVT_OPERATORS
    init_flags |= XNN_INIT_FLAG_VCVT;

    xnn_params.vcvt.f16_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f16_f32_vcvt_ukernel__scalar_x4,
      .init.f16_f32_cvt = xnn_init_f16_f32_cvt_scalar_params,
      .element_tile = 4,
    };
    xnn_params.vcvt.f32_to_f16 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_f16_vcvt_ukernel__scalar_fabsf_x2,
      .init.f32_f16_cvt = xnn_init_f32_f16_cvt_scalar_fabsf_params,
      .element_tile = 2,
    };
    xnn_params.vcvt.f32_to_qs8 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_x4,
      .init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_lrintf_params,
      .element_tile = 4,
    };
    xnn_params.vcvt.f32_to_qu8 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_x4,
      .init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_lrintf_params,
      .element_tile = 4,
    };
    xnn_params.vcvt.qs8_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_qs8_f32_vcvt_ukernel__scalar_x4,
      .init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params,
      .element_tile = 4,
    };
    xnn_params.vcvt.qu8_to_f32 = (struct vunary_parameters) {
      .ukernel = (xnn_univector_ukernel_function) xnn_qu8_f32_vcvt_ukernel__scalar_x4,
      .init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params,
      .element_tile = 4,
    };
  #endif  // XNN_NO_VCVT_OPERATORS

  /************************** X32 RISC-V micro-kernels **************************/
  #ifndef XNN_NO_X32_OPERATORS
    init_flags |= XNN_INIT_FLAG_X32;

    xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__scalar;
    xnn_params.x32.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__scalar,
      .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__scalar,
      .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__scalar,
      .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__scalar,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
      xnn_params.x32.depthtospace2d_chw2hwc = (struct depthtospace2d_chw2hwc_parameters) {
        .ukernel = (xnn_depthtospace2d_chw2hwc_ukernel_function) xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar,
        .channel_tile = 1,
        .pixel_tile = 1,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_X32_OPERATORS

  /************************** XX RISC-V micro-kernels ***************************/
  #ifndef XNN_NO_XX_OPERATORS
    init_flags |= XNN_INIT_FLAG_XX;

    xnn_params.xx.copy = (xnn_univector_ukernel_function) xnn_xx_copy_ukernel__memcpy;
    xnn_params.xx.fill = (struct fill_parameters) {
      .ukernel = (xnn_fill_ukernel_function) xnn_xx_fill_ukernel__scalar_x16,
      .row_tile = 1,
    };
    xnn_params.xx.pad = (struct pad_parameters) {
      .ukernel = (xnn_pad_ukernel_function) xnn_xx_pad_ukernel__scalar,
      .row_tile = 1,
    };
  #endif  // XNN_NO_XX_OPERATORS

#else
  #error "Unsupported architecture"
#endif

  memcpy(&xnn_params.allocator, init_allocator, sizeof(struct xnn_allocator));
  xnn_params.init_flags = init_flags;
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init();
    return TRUE;
  }
#endif

enum xnn_status xnn_initialize(const struct xnn_allocator* allocator) {
  #if !XNN_PLATFORM_WEB && !XNN_ARCH_RISCV
    if (!cpuinfo_initialize()) {
      return xnn_status_out_of_memory;
    }
  #endif  // !XNN_PLATFORM_WEB && !XNN_ARCH_RISCV
  if (allocator == NULL) {
    allocator = &xnn_default_allocator;
  }
  #ifdef _MSC_VER
    _InterlockedCompareExchangePointer((PVOID volatile*) &init_allocator, (PVOID) allocator, NULL);
  #else
    __sync_bool_compare_and_swap(&init_allocator, NULL, allocator);
  #endif
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard, &init_windows, NULL, NULL);
  #else
    pthread_once(&init_guard, &init);
  #endif
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) != 0) {
    return xnn_status_success;
  } else {
    return xnn_status_unsupported_hardware;
  }
}

enum xnn_status xnn_deinitialize(void) {
  #if !XNN_PLATFORM_WEB && !XNN_ARCH_RISCV
    cpuinfo_deinitialize();
  #endif  // !XNN_PLATFORM_WEB && !XNN_ARCH_RISCV
  return xnn_status_success;
}
