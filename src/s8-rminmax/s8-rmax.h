// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, datatype_out, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, datatype_out)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, datatype_out) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, datatype_out, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_rmax_ukernel__scalar_u1, 1, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_rmax_ukernel__scalar_u2_acc2, 2, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_rmax_ukernel__scalar_u3_acc3, 3, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_rmax_ukernel__scalar_u4_acc2, 4, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_rmax_ukernel__scalar_u4_acc4, 4, false, int8_t, int8_t, void*, NULL)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_s8_rmax_ukernel__neon_u16, 16, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_s8_rmax_ukernel__neon_u32_acc2, 32, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_s8_rmax_ukernel__neon_u48_acc3, 48, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_s8_rmax_ukernel__neon_u64_acc2, 64, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_s8_rmax_ukernel__neon_u64_acc4, 64, false, int8_t, int8_t, void*, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_s8_rmax_ukernel__sse41_u16, 16, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_s8_rmax_ukernel__sse41_u32_acc2, 32, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_s8_rmax_ukernel__sse41_u48_acc3, 48, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_s8_rmax_ukernel__sse41_u64_acc2, 64, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_s8_rmax_ukernel__sse41_u64_acc4, 64, false, int8_t, int8_t, void*, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_rmax_ukernel__wasmsimd_u16, 16, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_rmax_ukernel__wasmsimd_u32_acc2, 32, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_rmax_ukernel__wasmsimd_u48_acc3, 48, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_rmax_ukernel__wasmsimd_u64_acc2, 64, false, int8_t, int8_t, void*, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_rmax_ukernel__wasmsimd_u64_acc4, 64, false, int8_t, int8_t, void*, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hvx, xnn_s8_rmax_ukernel__hvx_u256_acc2, 256, false, int8_t, int8_t, void*, NULL)
#endif  // XNN_ARCH_HEXAGON && XNN_ENABLE_HVX


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
