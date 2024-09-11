// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_vsqrdiff_ukernel__neonfp16arith_u8, 8, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_vsqrdiff_ukernel__neonfp16arith_u16, 16, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_fp16_arith, xnn_f16_vsqrdiff_ukernel__fp16arith_u1, 1, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_fp16_arith, xnn_f16_vsqrdiff_ukernel__fp16arith_u2, 2, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_fp16_arith, xnn_f16_vsqrdiff_ukernel__fp16arith_u4, 4, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512fp16, xnn_f16_vsqrdiff_ukernel__avx512fp16_u32, 32, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512fp16, xnn_f16_vsqrdiff_ukernel__avx512fp16_u64, 64, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_f16c, xnn_f16_vsqrdiff_ukernel__f16c_u8, 8, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_f16c, xnn_f16_vsqrdiff_ukernel__f16c_u16, 16, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
