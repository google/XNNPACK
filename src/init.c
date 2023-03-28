// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <errno.h>
  #include <pthread.h>
  #include <sys/mman.h>
  #include <unistd.h>
#endif

#ifdef _MSC_VER
  #include <intrin.h>
#endif

#ifndef __EMSCRIPTEN__
  #include <cpuinfo.h>
#endif

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/avgpool.h>
#include <xnnpack/common.h>
#include <xnnpack/config.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>
#include <xnnpack/microparams-init.h>


#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard = PTHREAD_ONCE_INIT;
#endif

#define XNN_MR_TO_INDEX(MR) (MR-1)

#ifndef XNN_ENABLE_ASSEMBLY
  #error "XNN_ENABLE_ASSEMBLY is not defined"
#endif

#ifndef XNN_ENABLE_GEMM_M_SPECIALIZATION
  #error "XNN_ENABLE_GEMM_M_SPECIALIZATION is not defined"
#endif

static const struct xnn_allocator* volatile init_allocator = NULL;

static void init(void) {
  uint32_t init_flags = XNN_INIT_FLAG_XNNPACK;

#if XNN_ARCH_ARM
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  assert(hardware_config != NULL);
  if (hardware_config->use_arm_neon) {
    /**************************** QC8 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_QC8_OPERATORS
      // TODO(zhin): remove these init flags after removing checks in operators.
      init_flags |= XNN_INIT_FLAG_QC8;

    #endif  // XNN_NO_QC8_OPERATORS

    /**************************** QS8 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_QS8_OPERATORS
      init_flags |= XNN_INIT_FLAG_QS8;

    #endif  // XNN_NO_QS8_OPERATORS

    /*************************** QU8 AArch32 micro-kernels ***************************/
    #ifndef XNN_NO_QU8_OPERATORS
      init_flags |= XNN_INIT_FLAG_QU8;

    #endif  // XNN_NO_QU8_OPERATORS

    /**************************** S8 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_S8_OPERATORS
      init_flags |= XNN_INIT_FLAG_S8;

    #endif  // XNN_NO_S8_OPERATORS

    /**************************** U8 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_U8_OPERATORS
      init_flags |= XNN_INIT_FLAG_U8;

    #endif  // XNN_NO_U8_OPERATORS

    /**************************** X8 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_X8_OPERATORS
      init_flags |= XNN_INIT_FLAG_X8;

    #endif  // XNN_NO_X8_OPERATORS

    /**************************** F16 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_F16_OPERATORS
      #if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
        if (hardware_config->use_arm_neon_fp16_arith) {
          init_flags |= XNN_INIT_FLAG_F16 | XNN_INIT_FLAG_F16_NATIVE;

          #ifndef XNN_NO_NCHW_OPERATORS
            init_flags |= XNN_INIT_FLAG_CHW_OPT;

          #endif  // XNN_NO_NCHW_OPERATORS
        }
      #endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    #endif  // XNN_NO_F16_OPERATORS

    /**************************** F32 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_F32_OPERATORS
      init_flags |= XNN_INIT_FLAG_F32;


      #ifndef XNN_NO_NCHW_OPERATORS
        init_flags |= XNN_INIT_FLAG_CHW_OPT;

      #endif  // XNN_NO_NCHW_OPERATORS
    #endif  // XNN_NO_F32_OPERATORS

    /*************************** VCVT AArch32 micro-kernels ***************************/
    #ifndef XNN_NO_VCVT_OPERATORS
      init_flags |= XNN_INIT_FLAG_VCVT;
    #endif  // XNN_NO_VCVT_OPERATORS

    /**************************** X32 AArch32 micro-kernels ****************************/
    #ifndef XNN_NO_X32_OPERATORS
      init_flags |= XNN_INIT_FLAG_X32;

    #endif  // XNN_NO_X32_OPERATORS

  } else if (!XNN_PLATFORM_MOBILE) {

    /*************************** QC8 AArch32 Pre-NEON micro-kernels ***************************/
    #ifndef XNN_NO_QC8_OPERATORS
      init_flags |= XNN_INIT_FLAG_QC8;

    #endif  // XNN_NO_QS8_OPERATORS

    /*************************** QS8 AArch32 Pre-NEON micro-kernels ***************************/
    #ifndef XNN_NO_QS8_OPERATORS
      init_flags |= XNN_INIT_FLAG_QS8;

    #endif  // XNN_NO_QS8_OPERATORS

    /*************************** QU8 AArch32 Pre-NEON micro-kernels ***************************/
    #ifndef XNN_NO_QU8_OPERATORS
      init_flags |= XNN_INIT_FLAG_QU8;

    #endif  // XNN_NO_QU8_OPERATORS

    /**************************** S8 AArch32 Pre-NEON micro-kernels ****************************/
    #ifndef XNN_NO_S8_OPERATORS
      init_flags |= XNN_INIT_FLAG_S8;

    #endif  // XNN_NO_S8_OPERATORS

    /**************************** U8 AArch32 Pre-NEON micro-kernels ****************************/
    #ifndef XNN_NO_U8_OPERATORS
      init_flags |= XNN_INIT_FLAG_U8;

    #endif  // XNN_NO_U8_OPERATORS

    /**************************** X8 AArch32 Pre-NEON micro-kernels ****************************/
    #ifndef XNN_NO_X8_OPERATORS
      init_flags |= XNN_INIT_FLAG_X8;

    #endif  // XNN_NO_X8_OPERATORS

    /**************************** F32 AArch32 Pre-NEON micro-kernels ****************************/
    #ifndef XNN_NO_F32_OPERATORS
      init_flags |= XNN_INIT_FLAG_F32;

      #ifndef XNN_NO_NCHW_OPERATORS
        init_flags |= XNN_INIT_FLAG_CHW_OPT;

      #endif  // XNN_NO_NCHW_OPERATORS
    #endif  // XNN_NO_F32_OPERATORS

    /*************************** VCVT AArch32 Pre-NEON micro-kernels ***************************/
    #ifndef XNN_NO_VCVT_OPERATORS
      init_flags |= XNN_INIT_FLAG_VCVT;
    #endif  // XNN_NO_VCVT_OPERATORS

    /**************************** X32 AArch32 Pre-NEON micro-kernels ****************************/
    #ifndef XNN_NO_X32_OPERATORS
      init_flags |= XNN_INIT_FLAG_X32;

    #endif  // XNN_NO_X32_OPERATORS

  }

#elif XNN_ARCH_ARM64
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  assert(hardware_config != NULL);

  /**************************** QC8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_QC8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QC8;

  #endif  // XNN_NO_QC8_OPERATORS

  /**************************** QS8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QS8;

  #endif  // XNN_NO_QS8_OPERATORS

  /**************************** QU8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_QU8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QU8;


  #endif  // XNN_NO_QU8_OPERATORS

  /**************************** S8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_S8_OPERATORS
    init_flags |= XNN_INIT_FLAG_S8;
  #endif  // XNN_NO_S8_OPERATORS

  /**************************** U8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_U8_OPERATORS
    init_flags |= XNN_INIT_FLAG_U8;

  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_X8_OPERATORS
    init_flags |= XNN_INIT_FLAG_X8;

  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F16 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_F16_OPERATORS
    #if XNN_ENABLE_ARM_FP16_VECTOR
      if (hardware_config->use_arm_neon_fp16_arith) {
        init_flags |= XNN_INIT_FLAG_F16 | XNN_INIT_FLAG_F16_NATIVE;

        #ifndef XNN_NO_NCHW_OPERATORS
          init_flags |= XNN_INIT_FLAG_CHW_OPT;

        #endif  // XNN_NO_NCHW_OPERATORS
      }
    #endif  // XNN_ENABLE_ARM_FP16_VECTOR
  #endif  // XNN_NO_F16_OPERATORS

  /**************************** F32 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_F32_OPERATORS
    init_flags |= XNN_INIT_FLAG_F32;

    #ifndef XNN_NO_NCHW_OPERATORS
      init_flags |= XNN_INIT_FLAG_CHW_OPT;

    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /*************************** VCVT AArch64 micro-kernels ***************************/
  #ifndef XNN_NO_VCVT_OPERATORS
    init_flags |= XNN_INIT_FLAG_VCVT;
  #endif  // XNN_NO_VCVT_OPERATORS

  /**************************** X32 AArch64 micro-kernels ****************************/
  #ifndef XNN_NO_X32_OPERATORS
    init_flags |= XNN_INIT_FLAG_X32;

  #endif  // XNN_NO_X32_OPERATORS

#elif XNN_ARCH_X86 || XNN_ARCH_X86_64
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  assert(hardware_config != NULL);
  /**************************** QC8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_QC8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QC8;

  #endif  // XNN_NO_QC8_OPERATORS

  /**************************** QS8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QS8;

  #endif  // XNN_NO_QS8_OPERATORS

  /**************************** QU8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_QU8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QU8;

  #endif  // XNN_NO_QU8_OPERATORS

  /**************************** U8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_S8_OPERATORS
    init_flags |= XNN_INIT_FLAG_S8;

  #endif  // XNN_NO_S8_OPERATORS

  /**************************** U8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_U8_OPERATORS
    init_flags |= XNN_INIT_FLAG_U8;

  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 x86 micro-kernels ****************************/
  #ifndef XNN_NO_X8_OPERATORS
    init_flags |= XNN_INIT_FLAG_X8;

  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F16 x86 micro-kernels ****************************/
  #ifndef XNN_NO_F16_OPERATORS
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx2) {
      init_flags |= XNN_INIT_FLAG_F16;

    }
  #endif  // XNN_NO_F16_OPERATORS

  /**************************** F32 x86 micro-kernels ****************************/
  #ifndef XNN_NO_F32_OPERATORS
    init_flags |= XNN_INIT_FLAG_F32;

    #ifndef XNN_NO_NCHW_OPERATORS
      // Sparse microkernels on x86 currently target only SSE, and on processors
      // with AVX ISA dense inference is expected to be faster than sparse.
      if (!hardware_config->use_x86_avx) {
        init_flags |= XNN_INIT_FLAG_CHW_OPT;
      }

    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /*************************** VCVT x86 micro-kernels ***************************/
  #ifndef XNN_NO_VCVT_OPERATORS
    init_flags |= XNN_INIT_FLAG_VCVT;
  #endif  // XNN_NO_VCVT_OPERATORS

  /**************************** X32 x86 micro-kernels ****************************/
  #ifndef XNN_NO_X32_OPERATORS
    init_flags |= XNN_INIT_FLAG_X32;

  #endif  // XNN_NO_X32_OPERATORS

#elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  /**************************** QC8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QC8;

  #endif  // XNN_NO_QC8_OPERATORS

  /**************************** QS8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QS8;

  #endif  // XNN_NO_QS8_OPERATORS

  /**************************** QU8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_QU8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QU8;

  #endif  // XNN_NO_QU8_OPERATORS

  /**************************** S8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_S8_OPERATORS
    init_flags |= XNN_INIT_FLAG_S8;

  #endif  // XNN_NO_S8_OPERATORS

  /**************************** U8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_U8_OPERATORS
    init_flags |= XNN_INIT_FLAG_U8;

  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_X8_OPERATORS
    init_flags |= XNN_INIT_FLAG_X8;

  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F32 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_F32_OPERATORS
    init_flags |= XNN_INIT_FLAG_F32;

    #ifndef XNN_NO_NCHW_OPERATORS
      init_flags |= XNN_INIT_FLAG_CHW_OPT;

    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /*************************** VCVT WAsm SIMD micro-kernels***************************/
  #ifndef XNN_NO_VCVT_OPERATORS
    init_flags |= XNN_INIT_FLAG_VCVT;
  #endif  // XNN_NO_VCVT_OPERATORS

  /**************************** X32 WAsm SIMD micro-kernels****************************/
  #ifndef XNN_NO_X32_OPERATORS
    init_flags |= XNN_INIT_FLAG_X32;

  #endif  // XNN_NO_X32_OPERATORS

#elif XNN_ARCH_WASM
  /**************************** QC8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_QC8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QC8;

  #endif  // XNN_NO_QC8_OPERATORS

  /**************************** QS8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QS8;

  #endif  // XNN_NO_QS8_OPERATORS

  /**************************** QU8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_QU8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QU8;

  #endif  // XNN_NO_QU8_OPERATORS

  /**************************** S8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_S8_OPERATORS
    init_flags |= XNN_INIT_FLAG_S8;

  #endif  // XNN_NO_S8_OPERATORS

  /**************************** U8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_U8_OPERATORS
    init_flags |= XNN_INIT_FLAG_U8;

  #endif  // XNN_NO_U8_OPERATORS

  /**************************** X8 WAsm micro-kernels****************************/
  #ifndef XNN_NO_X8_OPERATORS
    init_flags |= XNN_INIT_FLAG_X8;

  #endif  // XNN_NO_X8_OPERATORS

  /**************************** F32 WAsm micro-kernels****************************/
  #ifndef XNN_NO_F32_OPERATORS
    init_flags |= XNN_INIT_FLAG_F32;

    #ifndef XNN_NO_NCHW_OPERATORS
      init_flags |= XNN_INIT_FLAG_CHW_OPT;

    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /*************************** VCVT WAsm micro-kernels***************************/
  #ifndef XNN_NO_VCVT_OPERATORS
    init_flags |= XNN_INIT_FLAG_VCVT;
  #endif  // XNN_NO_VCVT_OPERATORS

  /**************************** X32 WAsm micro-kernels****************************/
  #ifndef XNN_NO_X32_OPERATORS
    init_flags |= XNN_INIT_FLAG_X32;

  #endif  // XNN_NO_X32_OPERATORS

#elif XNN_ARCH_RISCV

  /************************** QC8 RISC-V micro-kernels **************************/
  #ifndef XNN_NO_QC8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QC8;

  #endif  // XNN_NO_QS8_OPERATORS

  /************************** QS8 RISC-V micro-kernels **************************/
  #ifndef XNN_NO_QS8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QS8;

  #endif  // XNN_NO_QS8_OPERATORS

  /************************** QU8 RISC-V micro-kernels **************************/
  #ifndef XNN_NO_QU8_OPERATORS
    init_flags |= XNN_INIT_FLAG_QU8;

  #endif  // XNN_NO_QU8_OPERATORS

  /************************** S8 RISC-V micro-kernels ***************************/
  #ifndef XNN_NO_S8_OPERATORS
    init_flags |= XNN_INIT_FLAG_S8;

  #endif  // XNN_NO_S8_OPERATORS

  /************************** U8 RISC-V micro-kernels ***************************/
  #ifndef XNN_NO_U8_OPERATORS
    init_flags |= XNN_INIT_FLAG_U8;

  #endif  // XNN_NO_U8_OPERATORS

  /************************** X8 RISC-V micro-kernels ***************************/
  #ifndef XNN_NO_X8_OPERATORS
    init_flags |= XNN_INIT_FLAG_X8;

  #endif  // XNN_NO_X8_OPERATORS

  /************************** F32 RISC-V micro-kernels **************************/
  #ifndef XNN_NO_F32_OPERATORS
    init_flags |= XNN_INIT_FLAG_F32;

    #ifndef XNN_NO_NCHW_OPERATORS
      init_flags |= XNN_INIT_FLAG_CHW_OPT;

    #endif  // XNN_NO_NCHW_OPERATORS
  #endif  // XNN_NO_F32_OPERATORS

  /************************** VCVT RISC-V micro-kernels *************************/
  #ifndef XNN_NO_VCVT_OPERATORS
    init_flags |= XNN_INIT_FLAG_VCVT;
  #endif  // XNN_NO_VCVT_OPERATORS

  /************************** X32 RISC-V micro-kernels **************************/
  #ifndef XNN_NO_X32_OPERATORS
    init_flags |= XNN_INIT_FLAG_X32;

  #endif  // XNN_NO_X32_OPERATORS

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
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    xnn_log_error("XNNPACK initialization failed: hardware not supported");
    return xnn_status_unsupported_hardware;
  }

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
  return xnn_status_success;
}
