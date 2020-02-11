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
#include <string.h>

#include <pthread.h>

#ifndef __EMSCRIPTEN__
  #include <cpuinfo.h>
#endif

#include <xnnpack.h>
#include <xnnpack/argmaxpool.h>
#include <xnnpack/avgpool.h>
#include <xnnpack/bilinear.h>
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
#include <xnnpack/memory.h>
#include <xnnpack/pad.h>
#include <xnnpack/params.h>
#include <xnnpack/pavgpool.h>
#include <xnnpack/prelu.h>
#include <xnnpack/raddstoreexpminusmax.h>
#include <xnnpack/rmax.h>
#include <xnnpack/spmm.h>
#include <xnnpack/unpool.h>
#include <xnnpack/vadd.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vmulcaddc.h>
#include <xnnpack/vunary.h>
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
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_ukernel_9p8x__neon_c16,
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
    #if XNN_ENABLE_ASSEMBLY
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_cortex_a53:
        case cpuinfo_uarch_cortex_a55:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__aarch32_neon_ld64,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__neon_lane_ld64,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__neon_lane_ld64,
            .mr = 4,
            .nr = 8,
          };
          break;

        case cpuinfo_uarch_cortex_a57:
        case cpuinfo_uarch_cortex_a72:
        case cpuinfo_uarch_cortex_a73:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__aarch32_neon_pld_cortex_a75,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__aarch32_neon_pld_cortex_a75,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__neon_lane_ld64,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__neon_lane_ld64,
            .mr = 4,
            .nr = 8,
          };
          break;

        default:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__neon_lane_ld64,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__neon_lane_ld64,
            .mr = 4,
            .nr = 8,
          };
          break;
      }
    #else  // XNN_ENABLE_ASSEMBLY
      xnn_params.f32.gemm = (struct gemm_parameters) {
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__neon_lane_ld128,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__neon_lane_ld128,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__neon_lane_ld64,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__neon_lane_ld64,
        .mr = 4,
        .nr = 8,
      };
    #endif  // XNN_ENABLE_ASSEMBLY
    xnn_params.f32.gemm2 = (struct gemm_parameters) {
      .gemm = NULL,
      .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2__neon_lane_ld64,
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
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_ukernel_9p8x__psimd_c4,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__psimd_c4,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__psimd_c4,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_mp_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.bilinear = (struct bilinear_parameters) {
      .ukernel = (xnn_bilinear_ukernel_function) xnn_f32_bilinear_ukernel__neon_c8,
      .pixel_tile = 1,
      .channel_tile = 8,
    };
    xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__neon;
    xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__neon_x8;
    xnn_params.f32.sigmoid = (xnn_univector_ukernel_function) xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8;
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__neon_2x8,
      .row_tile = 2,
      .channel_tile = 8,
    };
    xnn_params.f32.raddstoreexpminusmax = xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8;
    xnn_params.f32.rmax = xnn_f32_rmax_ukernel__neon;
    xnn_params.f32.vadd = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_ukernel__neon_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__neon_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vdiv = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_ukernel__scalar_x2,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_ukernel__scalar_x2,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_ukernel__scalar_x2,
      .element_tile = 2,
    };
    xnn_params.f32.vmax = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__neon_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__neon_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmin = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__neon_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__neon_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmul = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_ukernel__neon_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__neon_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vsub = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_ukernel__neon_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_ukernel__neon_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_ukernel_c4__neon_2x,
      .channel_tile = 4,
      .row_tile = 2,
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
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_ukernel_9p8x__neon_c16,
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
        case cpuinfo_uarch_cortex_a76:
        case cpuinfo_uarch_exynos_m3:
        case cpuinfo_uarch_exynos_m4:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
            .mr = 6,
            .nr = 8,
          };
          break;
        case cpuinfo_uarch_exynos_m1:
        case cpuinfo_uarch_exynos_m2:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_6x8s4__neonfma,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_6x8s4__neonfma,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8s4__neonfma,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8s4__neonfma,
            .mr = 6,
            .nr = 8,
            .log2_sr = 2,
          };
          break;

        case cpuinfo_uarch_cortex_a53:
        case cpuinfo_uarch_cortex_a55:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53,
            .mr = 6,
            .nr = 8,
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
        case cpuinfo_uarch_cortex_a77:
        case cpuinfo_uarch_exynos_m5:
        case cpuinfo_uarch_kryo:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a57,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a57,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a57,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a57,
            .mr = 4,
            .nr = 8,
          };
          break;
      }
    #else  // XNN_ENABLE_ASSEMBLY
      xnn_params.f32.gemm = (struct gemm_parameters) {
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_6x8__neonfma_lane_ld64,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_6x8__neonfma_lane_ld64,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__neonfma_lane_ld64,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__neonfma_lane_ld64,
        .mr = 6,
        .nr = 8,
      };
    #endif  // XNN_ENABLE_ASSEMBLY

    xnn_params.f32.gemm2 = (struct gemm_parameters) {
      .gemm = NULL,
      .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2__neonfma_lane_ld64,
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
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_ukernel_9p8x__psimd_c4,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__psimd_c4,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__psimd_c4,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_mp_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.bilinear = (struct bilinear_parameters) {
      .ukernel = (xnn_bilinear_ukernel_function) xnn_f32_bilinear_ukernel__neonfma_c8,
      .pixel_tile = 1,
      .channel_tile = 8,
    };
    xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__neon;
    xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__neonfma_x8;
    xnn_params.f32.sigmoid = (xnn_univector_ukernel_function) xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16;
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__neon_2x8,
      .row_tile = 2,
      .channel_tile = 8,
    };
    xnn_params.f32.raddstoreexpminusmax = xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16;
    xnn_params.f32.rmax = xnn_f32_rmax_ukernel__neon;
    xnn_params.f32.vadd = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_ukernel__neon_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__neon_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vdiv = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_ukernel__neon_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_ukernel__neon_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmax = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__neon_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__neon_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmin = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__neon_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__neon_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmul = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_ukernel__neon_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__neon_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vsub = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_ukernel__neon_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_ukernel__neon_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_ukernel__neon_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_ukernel_c4__neonfma_2x,
      .channel_tile = 4,
      .row_tile = 2,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
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
      xnn_params.f32.spchw_dwconv5x5 = (struct spchw_dwconv_parameters) {
        .ukernel = (xnn_dwconv_spchw_ukernel_function) xnn_f32_dwconv_spchw_ukernel_5x5p2__neonfma,
        .input_width_tile = 4,
        .output_width_tile = 4,
        .output_height_tile = 3,
      };
      xnn_params.f32.spchw_dwconv5x5s2 = (struct spchw_dwconv_parameters) {
        .ukernel = (xnn_dwconv_spchw_ukernel_function) xnn_f32_dwconv_spchw_ukernel_5x5s2p2__neonfma,
        .input_width_tile = 4,
        .output_width_tile = 4,
        .output_height_tile = 1,
      };
      xnn_params.f32.spchw_gavgpool = (struct spchw_gavgpool_parameters) {
        .ukernel = (xnn_gavgpool_spchw_ukernel_function) xnn_f32_gavgpool_spchw_ukernel__neon_x4,
        .channel_tile = 4,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
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
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_ukernel_9p8x__sse2_c16,
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
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.gemm = (struct gemm_parameters) {
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_7x16__avx512f_broadcast,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_7x16__avx512f_broadcast,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x16__avx512f_broadcast,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x16__avx512f_broadcast,
        .mr = 7,
        .nr = 16,
      };
    } else if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_fma3()) {
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_zen:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x16s4__fma3_broadcast,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x16s4__fma3_broadcast,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x16s4__fma3_broadcast,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x16s4__fma3_broadcast,
            .mr = 4,
            .nr = 16,
            .log2_sr = 2,
          };
          break;
        default:
          xnn_params.f32.gemm = (struct gemm_parameters) {
            .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_5x16__fma3_broadcast,
            .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_5x16__fma3_broadcast,
            .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x16__fma3_broadcast,
            .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x16__fma3_broadcast,
            .mr = 5,
            .nr = 16,
          };
          break;
      }
    } else if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx()) {
      xnn_params.f32.gemm = (struct gemm_parameters) {
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_5x16__avx_broadcast,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_5x16__avx_broadcast,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x16__avx_broadcast,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x16__avx_broadcast,
        .mr = 5,
        .nr = 16,
      };
    } else {
      xnn_params.f32.gemm = (struct gemm_parameters) {
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__sse_load1,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__sse_load1,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__sse_load1,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__sse_load1,
        .mr = 4,
        .nr = 8,
      };
    }
    xnn_params.f32.gemm2 = (struct gemm_parameters) {
      .gemm = NULL,
      .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2c4__sse,
      .mr = 4,
      .nr = 2,
      .log2_kr = 2,
    };
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.dwconv[0] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up16x4__avx512f,
        .cr = 16,
        .mr = 4,
      };
      xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up16x9__avx512f,
        .cr = 16,
        .mr = 9,
      };
      xnn_params.f32.dwconv[2] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up16x25__avx512f,
        .cr = 16,
        .mr = 25,
      };
    } else if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_fma3()) {
      xnn_params.f32.dwconv[0] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up16x4__fma3,
        .cr = 16,
        .mr = 4,
      };
      xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up16x9__fma3,
        .cr = 16,
        .mr = 9,
      };
      xnn_params.f32.dwconv[2] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up8x25__fma3,
        .cr = 8,
        .mr = 25,
      };
    } else if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx()) {
      xnn_params.f32.dwconv[0] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up16x4__avx,
        .cr = 16,
        .mr = 4,
      };
      xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up16x9__avx,
        .cr = 16,
        .mr = 9,
      };
      xnn_params.f32.dwconv[2] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up8x25__avx,
        .cr = 8,
        .mr = 25,
      };
    } else {
      xnn_params.f32.dwconv[0] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up8x4__sse,
        .cr = 8,
        .mr = 4,
      };
      xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up8x9__sse,
        .cr = 8,
        .mr = 9,
      };
      xnn_params.f32.dwconv[2] = (struct dwconv_parameters) {
        .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up8x25__sse,
        .cr = 8,
        .mr = 25,
      };
    }
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
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_ukernel_9p8x__sse_c4,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__sse2_c4,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__sse2_c4,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_mp_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.bilinear = (struct bilinear_parameters) {
      .ukernel = (xnn_bilinear_ukernel_function) xnn_f32_bilinear_ukernel__sse_c8,
      .pixel_tile = 1,
      .channel_tile = 8,
    };
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__avx512f;
    } else if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx()) {
      xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__avx;
    } else {
      xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__sse;
    }
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__avx512f_x32;
    } else if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_fma3()) {
      xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__fma3_x16;
    } else if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx()) {
      xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__avx_x16;
    } else {
      xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__sse_x8;
    }
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx2()) {
      xnn_params.f32.sigmoid = (xnn_univector_ukernel_function) xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x40;
    } else {
      xnn_params.f32.sigmoid = (xnn_univector_ukernel_function) xnn_f32_sigmoid_ukernel__sse2_p5_div_x16;
    }
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__sse2_2x8,
      .row_tile = 2,
      .channel_tile = 8,
    };
    xnn_params.f32.raddstoreexpminusmax = xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20_acc2;
    xnn_params.f32.rmax = xnn_f32_rmax_ukernel__sse;
    if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx512f()) {
      xnn_params.f32.vadd = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_ukernel__avx512f_x32,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__avx512f_x32,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__avx512f_x32,
        .element_tile = 32,
      };
      xnn_params.f32.vdiv = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_ukernel__avx512f_x32,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_ukernel__avx512f_x32,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_ukernel__avx512f_x32,
        .element_tile = 32,
      };
      xnn_params.f32.vmax = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__avx512f_x32,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__avx512f_x32,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__avx512f_x32,
        .element_tile = 32,
      };
      xnn_params.f32.vmin = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__avx512f_x32,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__avx512f_x32,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__avx512f_x32,
        .element_tile = 32,
      };
      xnn_params.f32.vmul = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_ukernel__avx512f_x32,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__avx512f_x32,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__avx512f_x32,
        .element_tile = 32,
      };
      xnn_params.f32.vsub = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_ukernel__avx512f_x32,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_ukernel__avx512f_x32,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_ukernel__avx512f_x32,
        .element_tile = 32,
      };
    } else if (!XNN_PLATFORM_MOBILE && cpuinfo_has_x86_avx()) {
      xnn_params.f32.vadd = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_ukernel__avx_x16,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__avx_x16,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__avx_x16,
        .element_tile = 16,
      };
      xnn_params.f32.vdiv = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_ukernel__avx_x16,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_ukernel__avx_x16,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_ukernel__avx_x16,
        .element_tile = 16,
      };
      xnn_params.f32.vmax = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__avx_x16,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__avx_x16,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__avx_x16,
        .element_tile = 16,
      };
      xnn_params.f32.vmin = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__avx_x16,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__avx_x16,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__avx_x16,
        .element_tile = 16,
      };
      xnn_params.f32.vmul = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_ukernel__avx_x16,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__avx_x16,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__avx_x16,
        .element_tile = 16,
      };
      xnn_params.f32.vsub = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_ukernel__avx_x16,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_ukernel__avx_x16,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_ukernel__avx_x16,
        .element_tile = 16,
      };
    } else {
      xnn_params.f32.vadd = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_ukernel__sse_x8,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__sse_x8,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__sse_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vdiv = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_ukernel__sse_x8,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_ukernel__sse_x8,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_ukernel__sse_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vmax = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__sse_x8,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__sse_x8,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__sse_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vmin = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__sse_x8,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__sse_x8,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__sse_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vmul = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_ukernel__sse_x8,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__sse_x8,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__sse_x8,
        .element_tile = 8,
      };
      xnn_params.f32.vsub = (struct vbinary_parameters) {
        .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_ukernel__sse_x8,
        .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_ukernel__sse_x8,
        .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_ukernel__sse_x8,
        .element_tile = 8,
      };
    }
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_ukernel_c4__sse_2x,
      .channel_tile = 4,
      .row_tile = 2,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
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
    #endif  // XNN_NO_NCHW_OPERATORS
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
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_ukernel_9p8x__scalar_c1,
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
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x8__psimd_splat,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x8__psimd_splat,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x8__psimd_splat,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x8__psimd_splat,
        .mr = 4,
        .nr = 8,
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
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x4__psimd_acc2,
      .cr = 4,
      .mr = 4,
    };
    xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x9__psimd_acc2,
      .cr = 4,
      .mr = 9,
    };
    xnn_params.f32.dwconv[2] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up4x25__psimd_acc2,
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
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_ukernel_9p8x__psimd_c4,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__psimd_c4,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__psimd_c4,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_mp_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.bilinear = (struct bilinear_parameters) {
      .ukernel = (xnn_bilinear_ukernel_function) xnn_f32_bilinear_ukernel__psimd_c8,
      .pixel_tile = 1,
      .channel_tile = 8,
    };
    xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__psimd;
    xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__psimd_x8;
    xnn_params.f32.sigmoid = (xnn_univector_ukernel_function) xnn_f32_sigmoid_ukernel__psimd_p5_div_x16;
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__psimd_2x8,
      .row_tile = 2,
      .channel_tile = 8,
    };
    xnn_params.f32.raddstoreexpminusmax = xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16_acc2;
    xnn_params.f32.rmax = xnn_f32_rmax_ukernel__psimd;
    xnn_params.f32.vadd = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_ukernel__psimd_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__psimd_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__psimd_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vdiv = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_ukernel__psimd_x4,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_ukernel__psimd_x4,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_ukernel__psimd_x4,
      .element_tile = 4,
    };
    xnn_params.f32.vmax = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__psimd_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__psimd_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__psimd_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmin = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__psimd_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__psimd_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__psimd_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmul = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_ukernel__psimd_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__psimd_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__psimd_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vsub = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_ukernel__psimd_x8,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_ukernel__psimd_x8,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_ukernel__psimd_x8,
      .element_tile = 8,
    };
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_ukernel_c4__psimd_2x,
      .channel_tile = 4,
      .row_tile = 2,
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
      .ukernel = (xnn_maxpool_ukernel_function) xnn_u8_maxpool_ukernel_9p8x__scalar_c1,
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
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x4__wasm,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x4__wasm,
        .mr = 2,
        .nr = 4,
      };
    } else {
      xnn_params.f32.gemm = (struct gemm_parameters) {
        .gemm = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_4x4__wasm,
        .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x4__wasm,
        .gemm1 = (xnn_gemm_ukernel_function) xnn_f32_gemm_ukernel_1x4__wasm,
        .igemm1 = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_1x4__wasm,
        .mr = 4,
        .nr = 4,
      };
    }
    xnn_params.f32.gemm2 = (struct gemm_parameters) {
      .gemm = NULL,
      .igemm = (xnn_igemm_ukernel_function) xnn_f32_igemm_ukernel_4x2__wasm,
      .mr = 4,
      .nr = 2,
    };
    xnn_params.f32.dwconv[0] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up1x4__wasm_acc2,
      .cr = 1,
      .mr = 4,
    };
    xnn_params.f32.dwconv[1] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up1x9__wasm_acc2,
      .cr = 1,
      .mr = 9,
    };
    xnn_params.f32.dwconv[2] = (struct dwconv_parameters) {
      .up = (xnn_dwconv_up_ukernel_function) xnn_f32_dwconv_ukernel_up1x25__wasm_acc2,
      .cr = 1,
      .mr = 25,
    };
    xnn_params.f32.avgpool = (struct avgpool_parameters) {
      .up = (xnn_avgpool_up_ukernel_function) xnn_f32_avgpool_ukernel_up9__wasm,
      .mp = (xnn_avgpool_mp_ukernel_function) xnn_f32_avgpool_ukernel_mp9p8q__wasm,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.pavgpool = (struct pavgpool_parameters) {
      .up = (xnn_pavgpool_up_ukernel_function) xnn_f32_pavgpool_ukernel_up9__wasm,
      .mp = (xnn_pavgpool_mp_ukernel_function) xnn_f32_pavgpool_ukernel_mp9p8q__wasm,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.gavgpool = (struct gavgpool_parameters) {
      .up = (xnn_gavgpool_up_ukernel_function) xnn_f32_gavgpool_ukernel_up7__wasm,
      .mp = (xnn_gavgpool_mp_ukernel_function) xnn_f32_gavgpool_ukernel_mp7p7q__wasm,
      .mr = 7,
    };
    xnn_params.f32.maxpool = (struct maxpool_parameters) {
      .ukernel = (xnn_maxpool_ukernel_function) xnn_f32_maxpool_ukernel_9p8x__wasm_c1,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.argmaxpool[0] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_4x__scalar_c1,
      .mr = 4,
    };
    xnn_params.f32.argmaxpool[1] = (struct argmaxpool_parameters) {
      .up = (xnn_argmaxpool_up_ukernel_function) xnn_f32_argmaxpool_ukernel_9x__scalar_c1,
      .mr = 9,
    };
    xnn_params.f32.argmaxpool[2] = (struct argmaxpool_parameters) {
      .mp = (xnn_argmaxpool_mp_ukernel_function) xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1,
      .mr = 9,
      .qr = 8,
    };
    xnn_params.f32.bilinear = (struct bilinear_parameters) {
      .ukernel = (xnn_bilinear_ukernel_function) xnn_f32_bilinear_ukernel__scalar_c2,
      .pixel_tile = 1,
      .channel_tile = 2,
    };
    xnn_params.f32.clamp = (xnn_univector_ukernel_function) xnn_f32_clamp_ukernel__wasm;
    xnn_params.f32.hswish = (xnn_univector_ukernel_function) xnn_f32_hswish_ukernel__wasm_x4;
    xnn_params.f32.sigmoid = (xnn_univector_ukernel_function) xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x2;
    xnn_params.f32.prelu = (struct prelu_parameters) {
      .ukernel = (xnn_prelu_ukernel_function) xnn_f32_prelu_ukernel__wasm_2x4,
      .row_tile = 4,
      .channel_tile = 4,
    };
    xnn_params.f32.raddstoreexpminusmax = xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4_acc2;
    xnn_params.f32.rmax = xnn_f32_rmax_ukernel__scalar;
    xnn_params.f32.vadd = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vadd_ukernel__wasm_x4,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__wasm_x4,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vaddc_ukernel__wasm_x4,
      .element_tile = 8,
    };
    xnn_params.f32.vdiv = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdiv_ukernel__wasm_x2,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vdivc_ukernel__wasm_x2,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrdivc_ukernel__wasm_x2,
      .element_tile = 2,
    };
    xnn_params.f32.vmax = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmax_ukernel__wasm_x4,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__wasm_x4,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmaxc_ukernel__wasm_x4,
      .element_tile = 8,
    };
    xnn_params.f32.vmin = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmin_ukernel__wasm_x4,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__wasm_x4,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vminc_ukernel__wasm_x4,
      .element_tile = 8,
    };
    xnn_params.f32.vmul = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmul_ukernel__wasm_x4,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__wasm_x4,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vmulc_ukernel__wasm_x4,
      .element_tile = 8,
    };
    xnn_params.f32.vsub = (struct vbinary_parameters) {
      .op_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsub_ukernel__wasm_x4,
      .opc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vsubc_ukernel__wasm_x4,
      .ropc_ukernel = (xnn_vbinary_ukernel_function) xnn_f32_vrsubc_ukernel__wasm_x4,
      .element_tile = 8,
    };
    xnn_params.f32.vmulcaddc = (struct vmulcaddc_parameters) {
      .ukernel = (xnn_vmulcaddc_ukernel_function) xnn_f32_vmulcaddc_ukernel_c1__wasm_2x,
      .channel_tile = 1,
      .row_tile = 2,
    };
    #ifndef XNN_NO_NCHW_OPERATORS
      xnn_params.f32.spmm = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_ukernel_8x1__scalar,
        .mr = 8,
        .nr = 1,
      };
      xnn_params.f32.spmm2 = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_ukernel_8x2__scalar,
        .mr = 8,
        .nr = 2,
      };
      xnn_params.f32.spmm4 = (struct spmm_parameters) {
        .ukernel = (xnn_spmm_ukernel_function) xnn_f32_spmm_ukernel_8x4__scalar,
        .mr = 8,
        .nr = 4,
      };
      xnn_params.f32.hwc2spchw_dconv3x3c3s2 = (struct hwc2spchw_dconv_parameters) {
        .ukernel_with_symm_padding =
          (xnn_conv_hwc2spchw_ukernel_function) xnn_f32_conv_hwc2spchw_ukernel_3x3s2p1c3x4__scalar_1x1,
        .output_channel_tile = 4,
        .output_height_tile = 1,
        .output_width_tile = 1,
      };
      xnn_params.f32.spchw_dwconv3x3 = (struct spchw_dwconv_parameters) {
        .ukernel = (xnn_dwconv_spchw_ukernel_function) xnn_f32_dwconv_spchw_ukernel_3x3p1__scalar,
        .input_width_tile = 1,
        .output_width_tile = 1,
        .output_height_tile = 1,
      };
      xnn_params.f32.spchw_dwconv3x3s2 = (struct spchw_dwconv_parameters) {
        .ukernel = (xnn_dwconv_spchw_ukernel_function) xnn_f32_dwconv_spchw_ukernel_3x3s2p1__scalar,
        .input_width_tile = 1,
        .output_width_tile = 1,
        .output_height_tile = 1,
      };
      xnn_params.f32.spchw_dwconv5x5 = (struct spchw_dwconv_parameters) {
        .ukernel = (xnn_dwconv_spchw_ukernel_function) xnn_f32_dwconv_spchw_ukernel_5x5p2__scalar,
        .input_width_tile = 1,
        .output_width_tile = 1,
        .output_height_tile = 1,
      };
      xnn_params.f32.spchw_dwconv5x5s2 = (struct spchw_dwconv_parameters) {
        .ukernel = (xnn_dwconv_spchw_ukernel_function) xnn_f32_dwconv_spchw_ukernel_5x5s2p2__scalar,
        .input_width_tile = 1,
        .output_width_tile = 1,
        .output_height_tile = 1,
      };
      xnn_params.f32.spchw_gavgpool = (struct spchw_gavgpool_parameters) {
        .ukernel = (xnn_gavgpool_spchw_ukernel_function) xnn_f32_gavgpool_spchw_ukernel__scalar_x1,
        .channel_tile = 1,
      };
    #endif  // XNN_NO_NCHW_OPERATORS
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

enum xnn_status xnn_initialize(const struct xnn_allocator* allocator) {
  #ifndef __EMSCRIPTEN__
    if (!cpuinfo_initialize()) {
      return xnn_status_out_of_memory;
    }
  #endif
  pthread_once(&init_guard, &init);
  if (xnn_params.initialized) {
    if (allocator != NULL) {
      memcpy(&xnn_params.allocator, allocator, sizeof(struct xnn_allocator));
    } else {
      xnn_params.allocator.allocate = &xnn_allocate;
      xnn_params.allocator.reallocate = &xnn_reallocate;
      xnn_params.allocator.deallocate = &xnn_deallocate;
      xnn_params.allocator.aligned_allocate = &xnn_aligned_allocate;
      xnn_params.allocator.aligned_deallocate = &xnn_aligned_deallocate;
    }
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
