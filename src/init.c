// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <pthread.h>

#ifndef __EMSCRIPTEN__
  #include <cpuinfo.h>
#endif

#include <xnnpack.h>
#include <xnnpack/argmaxpool.h>
#include <xnnpack/avgpool.h>
#include <xnnpack/clamp.h>
#include <xnnpack/common.h>
#include <xnnpack/conv.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/gavgpool.h>
#include <xnnpack/gemm.h>
#include <xnnpack/hswish.h>
#include <xnnpack/igemm.h>
#include <xnnpack/log.h>
#include <xnnpack/lut.h>
#include <xnnpack/maxpool.h>
#include <xnnpack/pad.h>
#include <xnnpack/params.h>
#include <xnnpack/pavgpool.h>
#include <xnnpack/prelu.h>
#include <xnnpack/rmax.h>
#include <xnnpack/spmm.h>
#include <xnnpack/unpool.h>
#include <xnnpack/vadd.h>
#include <xnnpack/vmulcaddc.h>
#include <xnnpack/zip.h>

#ifndef XNN_ENABLE_ASSEMBLY
  #define XNN_ENABLE_ASSEMBLY 1
#endif

static pthread_once_t init_guard = PTHREAD_ONCE_INIT;

struct xnn_parameters xnn_params = {
  .initialized = false
};

#if XNN_ARCH_PNACL || XNN_ARCH_ASMJS || XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  extern uint32_t xnn_stub_wasm_f32_sub(uint32_t a, uint32_t b);
#endif
#if XNN_ARCH_PNACL || XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  extern uint32_t xnn_stub_wasm_f32_min(uint32_t a, uint32_t b);
#endif

static void init(void) {
#if XNN_ARCH_ARM
  if (!cpuinfo_has_arm_neon()) {
    xnn_log_error("XNNPACK initialization failed: NEON is not supported");
    return;
  }

  /**************************** Q8 micro-kernels ****************************/
  #ifndef XNN_NO_Q8_OPERATORS
    xnn_params.q8.gemm = (struct gemm_parameters) {
      .gemm = (xnn_gemm_ukernel_function) xnn_q8_gemm_ukernel_4x8__neon,
      .igemm = (xnn_igemm_ukernel_function) xnn_q8_igemm_ukernel_4x8__neon,
      .mr = 4,
      .nr = 8,
    };

    #if XNN_ENABLE_ASSEMBLY
      xnn_params.q8.dwconv[0] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_q8_dwconv_ukernel_up8x9__aarch32_neon,
        .cr = 8,
        .mr = 9,
      };
    #else
      xnn_params.q8.dwconv[0] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_q8_dwconv_ukernel_up8x9__neon,
        .cr = 8,
        .mr = 9,
      };
    #endif
    xnn_params.q8.avgpool = (struct avgpool_parameters) {
      .up = (xnn_avgpool_up_ukernel_function) xnn_q8_avgpool_ukernel_up9__neon,
      .mp = (xnn_avgpool_mp_ukernel_function) xnn_q8_avgpool_ukernel_mp9p8q__neon,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.q8.gavgpool = (struct gavgpool_parameters) {
      .up = (xnn_gavgpool_up_ukernel_function) xnn_q8_gavgpool_ukernel_up7__neon,
      .mp = (xnn_gavgpool_mp_ukernel_function) xnn_q8_gavgpool_ukernel_mp7p7q__neon,
      .mr = 7,
    };
    xnn_params.q8.vadd = (xnn_vadd_ukernel_function) xnn_q8_vadd_ukernel__neon;
  #endif  // XNN_NO_Q8_OPERATORS

  /**************************** U8 micro-kernels ****************************/
  #ifndef XNN_NO_U8_OPERATORS
    xnn_params.u8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_ukernel_9p8q__neon,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.u8.clamp = (xnn_univector_ukernel_function) xnn_u8_clamp_ukernel__neon;
    xnn_params.u8.rmax = xnn_u8_rmax_ukernel__neon;
    xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 micro-kernels ****************************/
  #ifndef XNN_NO_X8_OPERATORS
    xnn_params.x8.lut = xnn_x8_lut_ukernel__scalar;
    xnn_params.x8.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__neon,
      .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__neon,
      .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__neon,
      .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__neon,
    };
  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F32 micro-kernels ****************************/
  #ifndef XNN_NO_F32_OPERATORS
    xnn_params.f32.gemm = (struct gemm_parameters) {
      .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__neon_ld128,
      .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__neon_ld128,
      .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__neon_ld64,
      .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__neon_ld64,
      .mr = 4,
      .nr = 8,
    };
    xnn_params.f32.gemm2 = (struct gemm_parameters) {
      .gemm = NULL,
      .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2__neon_ld64,
      .mr = 4,
      .nr = 2,
    };
    xnn_params.f32.dwconv[0] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x4__psimd,
      .cr = 4,
      .mr = 4,
    };
    xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x9__neon,
      .cr = 4,
      .mr = 9,
    };
    xnn_params.f32.dwconv[2] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x25__psimd,
      .cr = 4,
      .mr = 25,
    };
    xnn_params.f32.avgpool = (struct avgpool_parameters) {
      .up = (xnn_avgpool_up_ukernel_function) xnn_f32_avgpool_ukernel_up9__neon,
      .mp = (xnn_avgpool_mp_ukernel_function) xnn_f32_avgpool_ukernel_mp9p8q__neon,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
      .up = (xnn_pavgpool_up_ukernel_function) xnn_f32_pavgpool_ukernel_up9__neon,
      .mp = (xnn_pavgpool_mp_ukernel_function) xnn_f32_pavgpool_ukernel_mp9p8q__neon,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
      .up = (xnn_gavgpool_up_ukernel_function) xnn_f32_gavgpool_ukernel_up7__neon,
      .mp = (xnn_gavgpool_mp_ukernel_function) xnn_f32_gavgpool_ukernel_mp7p7q__neon,
      .mr = 7,
    };
    xnn_params.f32.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_ukernel_9p8q__psimd,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_up4__psimd,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_up9__psimd,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_mp_ukernel_function) xnn_f32_argmaxpool_ukernel_mp9p8q__psimd,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__neon;
    xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__neon;
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel_x4__psimd,
      .mr = 4,
    };
    xnn_params.f32.vadd = (xnn_vadd_ukernel_function) xnn_f32_vadd_ukernel__psimd;
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_ukernel_c4__neon_x2,
      .cr = 4,
      .mr = 2,
    };
  #endif  // XNN_NO_F32_OPERATORS

  /**************************** X32 micro-kernels ****************************/
  #ifndef XNN_NO_X32_OPERATORS
    xnn_params.x32.pad = (struct pad_parameters) {
      .ukernel = xnn_x32_pad_x2__neon,
      .mr = 2,
    };
    xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__psimd;
    xnn_params.x32.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__neon,
      .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__neon,
      .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__neon,
      .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__neon,
    };
  #endif  // XNN_NO_X32_OPERATORS

#elif XNN_ARCH_ARM64

  /**************************** Q8 micro-kernels ****************************/
  #ifndef XNN_NO_Q8_OPERATORS
    xnn_params.q8.gemm = (struct gemm_parameters) {
      .gemm = (xnn_gemm_ukernel_function) xnn_q8_gemm_ukernel_8x8__neon,
      .igemm = (xnn_igemm_ukernel_function) xnn_q8_igemm_ukernel_8x8__neon,
      .mr = 8,
      .nr = 8,
    };
    xnn_params.q8.dwconv[0] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_q8_dwconv_ukernel_up8x9__neon,
      .cr = 8,
      .mr = 9,
    };
    xnn_params.q8.avgpool = (struct avgpool_parameters) {
      .up = (xnn_avgpool_up_ukernel_function) xnn_q8_avgpool_ukernel_up9__neon,
      .mp = (xnn_avgpool_mp_ukernel_function) xnn_q8_avgpool_ukernel_mp9p8q__neon,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.q8.gavgpool = (struct gavgpool_parameters) {
      .up = (xnn_gavgpool_up_ukernel_function) xnn_q8_gavgpool_ukernel_up7__neon,
      .mp = (xnn_gavgpool_mp_ukernel_function) xnn_q8_gavgpool_ukernel_mp7p7q__neon,
      .mr = 7,
    };
    xnn_params.q8.vadd = (xnn_vadd_ukernel_function) xnn_q8_vadd_ukernel__neon;
  #endif  // XNN_NO_Q8_OPERATORS

  /**************************** U8 micro-kernels ****************************/
  #ifndef XNN_NO_U8_OPERATORS
    xnn_params.u8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_ukernel_9p8q__neon,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.u8.clamp = (xnn_univector_ukernel_function) xnn_u8_clamp_ukernel__neon;
    xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    xnn_params.u8.rmax = xnn_u8_rmax_ukernel__neon;
  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 micro-kernels ****************************/
  #ifndef XNN_NO_X8_OPERATORS
    xnn_params.x8.lut = xnn_x8_lut_ukernel__scalar;
    xnn_params.x8.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__neon,
      .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__neon,
      .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__neon,
      .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__neon,
    };
  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F32 micro-kernels ****************************/
  #ifndef XNN_NO_F32_OPERATORS
    #if XNN_ENABLE_ASSEMBLY
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_kryo:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a57,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .mr = 4,
            .nr = 8,
          };
          break;
        case cpuinfo_uarch_cortex_a57:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a57,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a57,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a57,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a57,
            .mr = 6,
            .nr = 8,
          };
          break;
        case cpuinfo_uarch_cortex_a72:
        case cpuinfo_uarch_cortex_a76:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .mr = 4,
            .nr = 8,
          };
          break;
        case cpuinfo_uarch_cortex_a75:
        case cpuinfo_uarch_mongoose_m1:
        case cpuinfo_uarch_mongoose_m2:
        case cpuinfo_uarch_meerkat_m3:
        case (cpuinfo_uarch_meerkat_m3 + 1):
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .mr = 6,
            .nr = 8,
          };
          break;
        case cpuinfo_uarch_cortex_a53:
        case cpuinfo_uarch_cortex_a55:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x12__aarch64_neonfma_cortex_a53,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x12__aarch64_neonfma_cortex_a53,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x12__aarch64_neonfma_cortex_a53,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x12__aarch64_neonfma_cortex_a53,
            .mr = 4,
            .nr = 12,
          };
          break;
        case cpuinfo_uarch_cortex_a73:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a73,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a73,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .mr = 6,
            .nr = 8,
          };
          break;
        default:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_6x8__neonfma_ld64,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_6x8__neonfma_ld64,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .mr = 6,
            .nr = 8,
          };
          break;
      }
    #else  // XNN_ENABLE_ASSEMBLY
      xnn_params.f32.gemm = (struct gemm_parameters) {
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_6x8__neonfma_ld64,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_6x8__neonfma_ld64,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__neonfma_ld64,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__neonfma_ld64,
        .mr = 6,
        .nr = 8,
      };
    #endif

    xnn_params.f32.gemm2 = (struct gemm_parameters) {
      .gemm = NULL,
      .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2__neonfma_ld64,
      .mr = 4,
      .nr = 2,
    };
    xnn_params.f32.dwconv[0] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x4__psimd,
      .cr = 4,
      .mr = 4,
    };
    switch (cpuinfo_get_core(0)->uarch) {
      case cpuinfo_uarch_kryo:
        xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
          .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x9__neonfma,
          .cr = 4,
          .mr = 9,
        };
        break;
#if XNN_ENABLE_ASSEMBLY
      case cpuinfo_uarch_cortex_a53:
      case cpuinfo_uarch_cortex_a55:
        xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
          .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55,
          .cr = 4,
          .mr = 9,
        };
        break;
#endif
      default:
        xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
          .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up8x9__neonfma,
          .cr = 8,
          .mr = 9,
        };
        break;
    }
    xnn_params.f32.dwconv[2] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x25__psimd,
      .cr = 4,
      .mr = 25,
    };
    xnn_params.f32.avgpool = (struct avgpool_parameters) {
      .up = (xnn_avgpool_up_ukernel_function) xnn_f32_avgpool_ukernel_up9__neon,
      .mp = (xnn_avgpool_mp_ukernel_function) xnn_f32_avgpool_ukernel_mp9p8q__neon,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
      .up = (xnn_pavgpool_up_ukernel_function) xnn_f32_pavgpool_ukernel_up9__neon,
      .mp = (xnn_pavgpool_mp_ukernel_function) xnn_f32_pavgpool_ukernel_mp9p8q__neon,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
      .up = (xnn_gavgpool_up_ukernel_function) xnn_f32_gavgpool_ukernel_up7__neon,
      .mp = (xnn_gavgpool_mp_ukernel_function) xnn_f32_gavgpool_ukernel_mp7p7q__neon,
      .mr = 7,
    };
    xnn_params.f32.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_ukernel_9p8q__psimd,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_up4__psimd,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_up9__psimd,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_mp_ukernel_function) xnn_f32_argmaxpool_ukernel_mp9p8q__psimd,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__neon;
    xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__neonfma;
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel_x4__psimd,
      .mr = 4,
    };
    xnn_params.f32.vadd = (xnn_vadd_ukernel_function) xnn_f32_vadd_ukernel__psimd;
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2,
      .cr = 4,
      .mr = 2,
    };
    #ifndef XNN_NO_SPNCHW_OPERATORS
      xnn_params.f32.spmm = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_ukernel_16x1__neonfma_pipelined,
        .mr = 16,
        .nr = 1,
      };
      xnn_params.f32.spmm2 = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_ukernel_16x2__neonfma,
        .mr = 16,
        .nr = 2,
      };
      xnn_params.f32.spmm4 = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_ukernel_16x4__neonfma,
        .mr = 16,
        .nr = 4,
      };
      xnn_params.f32.hwc2spchw_dconv3x3c3s2 = (struct hwc2spchw_dconv_parameters) {
        .ukernel_with_symm_padding =
          (xnn_conv_hwc2spchw_ukernel_function) xnn_f32_conv_hwc2spchw_ukernel_3x3s2p1c3x4__neonfma_2x2,
        .output_channel_tile = 4,
        .output_height_tile = 2,
        .output_width_tile = 2,
      };
      xnn_params.f32.spchw_dwconv3x3 = (struct spchw_dwconv_parameters) {
        .ukernel = (xnn_dwconv_spchw_ukernel_function) xnn_f32_dwconv_spchw_ukernel_3x3p1__neonfma,
        .input_width_tile = 4,
        .output_width_tile = 4,
        .output_height_tile = 3,
      };
      xnn_params.f32.spchw_dwconv3x3s2 = (struct spchw_dwconv_parameters) {
        .ukernel = (xnn_dwconv_spchw_ukernel_function) xnn_f32_dwconv_spchw_ukernel_3x3s2p1__neonfma,
        .input_width_tile = 4,
        .output_width_tile = 4,
        .output_height_tile = 1,
      };
      xnn_params.f32.spchw_gavgpool = (struct spchw_gavgpool_parameters) {
        .ukernel = (xnn_gavgpool_spchw_ukernel_function) xnn_f32_gavgpool_spchw_ukernel__neon_x4,
        .channel_tile = 4,
      };
    #endif  // XNN_NO_SPNCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /**************************** X32 micro-kernels ****************************/
  #ifndef XNN_NO_X32_OPERATORS
    xnn_params.x32.pad = (struct pad_parameters) {
      .ukernel = xnn_x32_pad_x2__neon,
      .mr = 2,
    };
    xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__psimd;
    xnn_params.x32.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__neon,
      .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__neon,
      .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__neon,
      .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__neon,
    };
  #endif  // XNN_NO_X32_OPERATORS

#elif XNN_ARCH_X86 || XNN_ARCH_X86_64
  if (!cpuinfo_has_x86_sse2()) {
    xnn_log_error("XNNPACK initialization failed: SSE2 is not supported");
    return;
  }

  /**************************** Q8 micro-kernels ****************************/
  #ifndef XNN_NO_Q8_OPERATORS
    xnn_params.q8.gemm = (struct gemm_parameters) {
      .gemm = (xnn_gemm_ukernel_function) xnn_q8_gemm_ukernel_4x4c2__sse2,
      .igemm = (xnn_igemm_ukernel_function) xnn_q8_igemm_ukernel_4x4c2__sse2,
      .mr = 4,
      .nr = 4,
      .log2_kr = 1,
    };
    xnn_params.q8.dwconv[0] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_q8_dwconv_ukernel_up8x9__sse2,
      .cr = 8,
      .mr = 9,
    };
    xnn_params.q8.avgpool = (struct avgpool_parameters) {
      .up = (xnn_avgpool_up_ukernel_function) xnn_q8_avgpool_ukernel_up9__sse2,
      .mp = (xnn_avgpool_mp_ukernel_function) xnn_q8_avgpool_ukernel_mp9p8q__sse2,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.q8.gavgpool = (struct gavgpool_parameters) {
      .up = (xnn_gavgpool_up_ukernel_function) xnn_q8_gavgpool_ukernel_up7__sse2,
      .mp = (xnn_gavgpool_mp_ukernel_function) xnn_q8_gavgpool_ukernel_mp7p7q__sse2,
      .mr = 7,
    };
    xnn_params.q8.vadd = (xnn_vadd_ukernel_function) xnn_q8_vadd_ukernel__sse2;
  #endif  // XNN_NO_Q8_OPERATORS

  /**************************** U8 micro-kernels ****************************/
  #ifndef XNN_NO_U8_OPERATORS
    xnn_params.u8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_ukernel_9p8q__sse2,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.u8.clamp = (xnn_univector_ukernel_function) xnn_u8_clamp_ukernel__sse2;
    xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    xnn_params.u8.rmax = xnn_u8_rmax_ukernel__sse2;
  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 micro-kernels ****************************/
  #ifndef XNN_NO_X8_OPERATORS
    xnn_params.x8.lut = xnn_x8_lut_ukernel__scalar;
    xnn_params.x8.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__sse2,
      .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__sse2,
      .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__sse2,
      .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__sse2,
    };
  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F32 micro-kernels ****************************/
  #ifndef XNN_NO_F32_OPERATORS
    xnn_params.f32.gemm = (struct gemm_parameters) {
      .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__sse_load1,
      .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__sse_load1,
      .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__sse_load1,
      .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__sse_load1,
      .mr = 4,
      .nr = 8,
    };
    xnn_params.f32.gemm2 = (struct gemm_parameters) {
      .gemm = NULL,
      .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2c4__sse,
      .mr = 4,
      .nr = 2,
      .log2_kr = 2,
    };
    xnn_params.f32.dwconv[0] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x4__sse,
      .cr = 4,
      .mr = 4,
    };
    xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x9__sse,
      .cr = 4,
      .mr = 9,
    };
    xnn_params.f32.dwconv[2] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x25__sse,
      .cr = 4,
      .mr = 25,
    };
    xnn_params.f32.avgpool = (struct avgpool_parameters) {
      .up = (xnn_avgpool_up_ukernel_function) xnn_f32_avgpool_ukernel_up9__sse,
      .mp = (xnn_avgpool_mp_ukernel_function) xnn_f32_avgpool_ukernel_mp9p8q__sse,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
      .up = (xnn_pavgpool_up_ukernel_function) xnn_f32_pavgpool_ukernel_up9__sse,
      .mp = (xnn_pavgpool_mp_ukernel_function) xnn_f32_pavgpool_ukernel_mp9p8q__sse,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
      .up = (xnn_gavgpool_up_ukernel_function) xnn_f32_gavgpool_ukernel_up7__sse,
      .mp = (xnn_gavgpool_mp_ukernel_function) xnn_f32_gavgpool_ukernel_mp7p7q__sse,
      .mr = 7,
    };
    xnn_params.f32.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_ukernel_9p8q__sse,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_up4__sse2,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_up9__sse2,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_mp_ukernel_function) xnn_f32_argmaxpool_ukernel_mp9p8q__sse2,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__sse;
    xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__sse;
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel_x4__sse,
      .mr = 4,
    };
    xnn_params.f32.vadd = (xnn_vadd_ukernel_function) xnn_f32_vadd_ukernel__sse;
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_ukernel_c4__sse_x2,
      .cr = 4,
      .mr = 2,
    };
    #ifndef XNN_NO_SPNCHW_OPERATORS
      xnn_params.f32.spmm = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_ukernel_4x1__sse,
        .mr = 4,
        .nr = 1,
      };
      xnn_params.f32.spchw_dwconv3x3 = (struct spchw_dwconv_parameters) {
        .ukernel = (xnn_dwconv_spchw_ukernel_function) xnn_f32_dwconv_spchw_ukernel_3x3p1__sse,
        .input_width_tile = 4,
        .output_width_tile = 4,
        .output_height_tile = 1,
      };
      xnn_params.f32.spchw_dwconv3x3s2 = (struct spchw_dwconv_parameters) {
        .ukernel = (xnn_dwconv_spchw_ukernel_function) xnn_f32_dwconv_spchw_ukernel_3x3s2p1__sse,
        .input_width_tile = 4,
        .output_width_tile = 4,
        .output_height_tile = 1,
      };
      xnn_params.f32.spchw_gavgpool = (struct spchw_gavgpool_parameters) {
        .ukernel = (xnn_gavgpool_spchw_ukernel_function) xnn_f32_gavgpool_spchw_ukernel__sse_x4,
        .channel_tile = 4,
      };
    #endif  // XNN_NO_SPNCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /**************************** X32 micro-kernels ****************************/
  #ifndef XNN_NO_X32_OPERATORS
    xnn_params.x32.pad = (struct pad_parameters) {
      .ukernel = xnn_x32_pad_x2__sse2,
      .mr = 2,
    };
    xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__psimd;
    xnn_params.x32.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__sse2,
      .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__sse2,
      .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__sse2,
      .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__sse2,
    };
  #endif  // XNN_NO_X32_OPERATORS

#elif XNN_ARCH_PNACL || XNN_ARCH_WASMSIMD
  // Unlike most other architectures, on x86/x86-64 when floating-point instructions
  // have no NaN arguments, but produce NaN output, the output NaN has sign bit set.
  // We use it to distinguish x86/x86-64 from other architectures, by doing subtraction
  // of two infinities (must produce NaN per IEEE 754 standard).
  static volatile uint32_t minus_inf = UINT32_C(0xFF800000);
  const bool is_wasm_x86 = (int32_t) xnn_stub_wasm_f32_sub(minus_inf, minus_inf) < 0;

  /**************************** Q8 micro-kernels ****************************/
  #ifndef XNN_NO_Q8_OPERATORS
    xnn_params.q8.gemm = (struct gemm_parameters) {
      .gemm = (xnn_gemm_ukernel_function) xnn_q8_gemm_ukernel_2x2__scalar,
      .igemm = (xnn_igemm_ukernel_function) xnn_q8_igemm_ukernel_2x2__scalar,
      .mr = 2,
      .nr = 2,
    };
    xnn_params.q8.dwconv[0] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_q8_dwconv_ukernel_up1x9__scalar,
      .cr = 1,
      .mr = 9,
    };
    xnn_params.q8.avgpool = (struct avgpool_parameters) {
      .up = (xnn_avgpool_up_ukernel_function) xnn_q8_avgpool_ukernel_up9__scalar,
      .mp = (xnn_avgpool_mp_ukernel_function) xnn_q8_avgpool_ukernel_mp9p8q__scalar,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.q8.gavgpool = (struct gavgpool_parameters) {
      .up = (xnn_gavgpool_up_ukernel_function) xnn_q8_gavgpool_ukernel_up7__scalar,
      .mp = (xnn_gavgpool_mp_ukernel_function) xnn_q8_gavgpool_ukernel_mp7p7q__scalar,
      .mr = 7,
    };
    xnn_params.q8.vadd = (xnn_vadd_ukernel_function) xnn_q8_vadd_ukernel__scalar;
  #endif  // XNN_NO_Q8_OPERATORS

  /**************************** U8 micro-kernels ****************************/
  #ifndef XNN_NO_U8_OPERATORS
    xnn_params.u8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_ukernel_9p8q__scalar,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.u8.clamp = (xnn_univector_ukernel_function) xnn_u8_clamp_ukernel__scalar;
    xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    xnn_params.u8.rmax = xnn_u8_rmax_ukernel__scalar;
  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 micro-kernels ****************************/
  #ifndef XNN_NO_X8_OPERATORS
    xnn_params.x8.lut = xnn_x8_lut_ukernel__scalar;
    xnn_params.x8.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__scalar,
      .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__scalar,
      .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__scalar,
      .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__scalar,
    };
  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F32 micro-kernels ****************************/
  #ifndef XNN_NO_F32_OPERATORS
    if (is_wasm_x86) {
      xnn_params.f32.gemm = (struct gemm_parameters) {
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8s4__psimd,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8s4__psimd,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8s4__psimd,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8s4__psimd,
        .mr = 4,
        .nr = 8,
        .log2_sr = 2,
      };
    } else {
      xnn_params.f32.gemm = (struct gemm_parameters) {
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_6x8s4__psimd,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_6x8s4__psimd,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_igemm_ukernel_1x8s4__psimd,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8s4__psimd,
        .mr = 6,
        .nr = 8,
        .log2_sr = 2,
      };
    }
    xnn_params.f32.gemm2 = (struct gemm_parameters) {
      .gemm = NULL,
      .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2c4__psimd,
      .mr = 4,
      .nr = 2,
      .log2_kr = 2,
    };
    xnn_params.f32.dwconv[0] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x4__psimd,
      .cr = 4,
      .mr = 4,
    };
    xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x9__psimd,
      .cr = 4,
      .mr = 9,
    };
    xnn_params.f32.dwconv[2] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x25__psimd,
      .cr = 4,
      .mr = 25,
    };
    xnn_params.f32.avgpool = (struct avgpool_parameters) {
      .up = (xnn_avgpool_up_ukernel_function) xnn_f32_avgpool_ukernel_up9__psimd,
      .mp = (xnn_avgpool_mp_ukernel_function) xnn_f32_avgpool_ukernel_mp9p8q__psimd,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
      .up = (xnn_pavgpool_up_ukernel_function) xnn_f32_pavgpool_ukernel_up9__psimd,
      .mp = (xnn_pavgpool_mp_ukernel_function) xnn_f32_pavgpool_ukernel_mp9p8q__psimd,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
      .up = (xnn_gavgpool_up_ukernel_function) xnn_f32_gavgpool_ukernel_up7__psimd,
      .mp = (xnn_gavgpool_mp_ukernel_function) xnn_f32_gavgpool_ukernel_mp7p7q__psimd,
      .mr = 7,
    };
    xnn_params.f32.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_ukernel_9p8q__psimd,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_up4__psimd,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_up9__psimd,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_mp_ukernel_function) xnn_f32_argmaxpool_ukernel_mp9p8q__psimd,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__psimd;
    xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__psimd;
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel_x4__psimd,
      .mr = 4,
    };
    xnn_params.f32.vadd = (xnn_vadd_ukernel_function) xnn_f32_vadd_ukernel__psimd;
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_ukernel_c4__psimd_x2,
      .cr = 4,
      .mr = 2,
    };
  #endif  // XNN_NO_F32_OPERATORS

  /**************************** X32 micro-kernels ****************************/
  #ifndef XNN_NO_X32_OPERATORS
    xnn_params.x32.pad = (struct pad_parameters) {
      .ukernel = xnn_x32_pad_x2__psimd,
      .mr = 2,
    };
    xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__psimd;
    xnn_params.x32.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__psimd,
      .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__psimd,
      .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__psimd,
      .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__psimd,
    };
  #endif  // XNN_NO_X32_OPERATORS

#elif XNN_ARCH_WASM || XNN_ARCH_ASMJS
  // Unlike most other architectures, on x86/x86-64 when floating-point instructions
  // have no NaN arguments, but produce NaN output, the output NaN has sign bit set.
  // We use it to distinguish x86/x86-64 from other architectures, by doing subtraction
  // of two infinities (must produce NaN per IEEE 754 standard).
  static volatile uint32_t minus_inf = UINT32_C(0xFF800000);
  const bool is_wasm_x86 = (int32_t) xnn_stub_wasm_f32_sub(minus_inf, minus_inf) < 0;

  /**************************** Q8 micro-kernels ****************************/
  #ifndef XNN_NO_Q8_OPERATORS
    xnn_params.q8.gemm = (struct gemm_parameters) {
      .gemm = (xnn_gemm_ukernel_function) xnn_q8_gemm_ukernel_2x2__scalar,
      .igemm = (xnn_igemm_ukernel_function) xnn_q8_igemm_ukernel_2x2__scalar,
      .mr = 2,
      .nr = 2,
    };
    xnn_params.q8.dwconv[0] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_q8_dwconv_ukernel_up1x9__scalar,
      .cr = 1,
      .mr = 9,
    };
    xnn_params.q8.avgpool = (struct avgpool_parameters) {
      .up = (xnn_avgpool_up_ukernel_function) xnn_q8_avgpool_ukernel_up9__scalar,
      .mp = (xnn_avgpool_mp_ukernel_function) xnn_q8_avgpool_ukernel_mp9p8q__scalar,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.q8.gavgpool = (struct gavgpool_parameters) {
      .up = (xnn_gavgpool_up_ukernel_function) xnn_q8_gavgpool_ukernel_up7__scalar,
      .mp = (xnn_gavgpool_mp_ukernel_function) xnn_q8_gavgpool_ukernel_mp7p7q__scalar,
      .mr = 7,
    };
    xnn_params.q8.vadd = (xnn_vadd_ukernel_function) xnn_q8_vadd_ukernel__scalar;
  #endif  // XNN_NO_Q8_OPERATORS

  /**************************** U8 micro-kernels ****************************/
  #ifndef XNN_NO_U8_OPERATORS
    xnn_params.u8.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_ukernel_9p8q__scalar,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.u8.clamp = (xnn_univector_ukernel_function) xnn_u8_clamp_ukernel__scalar;
    xnn_params.u8.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    xnn_params.u8.rmax = xnn_u8_rmax_ukernel__scalar;
  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 micro-kernels ****************************/
  #ifndef XNN_NO_X8_OPERATORS
    xnn_params.x8.lut = xnn_x8_lut_ukernel__scalar;
    xnn_params.x8.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x8_zip_x2_ukernel__scalar,
      .x3 = (xnn_zipc_ukernel_function) xnn_x8_zip_x3_ukernel__scalar,
      .x4 = (xnn_zipc_ukernel_function) xnn_x8_zip_x4_ukernel__scalar,
      .xm = (xnn_zipv_ukernel_function) xnn_x8_zip_xm_ukernel__scalar,
    };
  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F32 micro-kernels ****************************/
  #ifndef XNN_NO_F32_OPERATORS
    if (is_wasm_x86) {
      xnn_params.f32.gemm = (struct gemm_parameters) {
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_2x4__scalar,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_2x4__scalar,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x4__scalar,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x4__scalar,
        .mr = 2,
        .nr = 4,
      };
    } else {
      xnn_params.f32.gemm = (struct gemm_parameters) {
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x4__scalar,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x4__scalar,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x4__scalar,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x4__scalar,
        .mr = 4,
        .nr = 4,
      };
    }
    xnn_params.f32.gemm2 = (struct gemm_parameters) {
      .gemm = NULL,
      .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2__scalar,
      .mr = 4,
      .nr = 2,
    };
    xnn_params.f32.dwconv[0] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up1x4__scalar,
      .cr = 1,
      .mr = 4,
    };
    xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up1x9__scalar,
      .cr = 1,
      .mr = 9,
    };
    xnn_params.f32.dwconv[2] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up1x25__scalar,
      .cr = 1,
      .mr = 25,
    };
    xnn_params.f32.avgpool = (struct avgpool_parameters) {
      .up = (xnn_avgpool_up_ukernel_function) xnn_f32_avgpool_ukernel_up9__scalar,
      .mp = (xnn_avgpool_mp_ukernel_function) xnn_f32_avgpool_ukernel_mp9p8q__scalar,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
      .up = (xnn_pavgpool_up_ukernel_function) xnn_f32_pavgpool_ukernel_up9__scalar,
      .mp = (xnn_pavgpool_mp_ukernel_function) xnn_f32_pavgpool_ukernel_mp9p8q__scalar,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
      .up = (xnn_gavgpool_up_ukernel_function) xnn_f32_gavgpool_ukernel_up7__scalar,
      .mp = (xnn_gavgpool_mp_ukernel_function) xnn_f32_gavgpool_ukernel_mp7p7q__scalar,
      .mr = 7,
    };
    xnn_params.f32.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_ukernel_9p8q__scalar,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_up4__scalar,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_up9__scalar,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_mp_ukernel_function) xnn_f32_argmaxpool_ukernel_mp9p8q__scalar,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__scalar;
    xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__scalar;
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel_x4__scalar,
      .mr = 4,
    };
    xnn_params.f32.vadd = (xnn_vadd_ukernel_function) xnn_f32_vadd_ukernel__scalar;
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_ukernel_c1__scalar_x2,
      .cr = 1,
      .mr = 2,
    };
    #ifndef XNN_NO_SPNCHW_OPERATORS
      xnn_params.f32.spmm = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_ukernel_4x1__scalar,
        .mr = 4,
        .nr = 1,
      };
    #endif  // XNN_NO_SPNCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /**************************** X32 micro-kernels ****************************/
  #ifndef XNN_NO_X32_OPERATORS
    xnn_params.x32.pad = (struct pad_parameters) {
      .ukernel = xnn_x32_pad_x2__scalar,
      .mr = 2,
    };
    xnn_params.x32.unpool = (xnn_unpool_ukernel_function) xnn_x32_unpool_ukernel__scalar;
    xnn_params.x32.zip = (struct zip_parameters) {
      .x2 = (xnn_zipc_ukernel_function) xnn_x32_zip_x2_ukernel__scalar,
      .x3 = (xnn_zipc_ukernel_function) xnn_x32_zip_x3_ukernel__scalar,
      .x4 = (xnn_zipc_ukernel_function) xnn_x32_zip_x4_ukernel__scalar,
      .xm = (xnn_zipv_ukernel_function) xnn_x32_zip_xm_ukernel__scalar,
    };
  #endif  // XNN_NO_X32_OPERATORS

#else
  #error "Unsupported architecture"
#endif
  xnn_params.initialized = true;
}

enum xnn_status xnn_initialize(void) {
  #ifndef __EMSCRIPTEN__
    if (!cpuinfo_initialize()) {
      return xnn_status_out_of_memory;
    }
  #endif
  pthread_once(&init_guard, &init);
  if (xnn_params.initialized) {
    return xnn_status_success;
  } else {
    return xnn_status_unsupported_hardware;
  }
}

enum xnn_status xnn_deinitialize(void) {
  #ifndef __EMSCRIPTEN__
    cpuinfo_deinitialize();
  #endif
  return xnn_status_success;
}
