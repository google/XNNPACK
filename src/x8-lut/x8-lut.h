// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_size, datatype, params_type, init_params)                   \
  XNN_UKERNEL(arch_flags, ukernel, batch_size, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, batch_size, datatype)                                                         \
  XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_size, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

XNN_UKERNEL_WITH_PARAMS(0, xnn_x8_lut_ukernel__scalar_u1, 1, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(0, xnn_x8_lut_ukernel__scalar_u2, 2, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(0, xnn_x8_lut_ukernel__scalar_u4, 4, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(0, xnn_x8_lut_ukernel__scalar_u8, 8, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(0, xnn_x8_lut_ukernel__scalar_u16, 16, uint8_t, void, nullptr)

#if XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm64, xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u16, 16, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm64, xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u32, 32, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm64, xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u48, 48, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm64, xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u64, 64, uint8_t, void, nullptr)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__ssse3_u16, 16, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__ssse3_u32, 32, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx_u16, 16, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx_u32, 32, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx_u48, 48, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx_u64, 64, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx2_u32, 32, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx2_u64, 64, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx2_u96, 96, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx2_u128, 128, uint8_t, void, nullptr)
#endif

#if XNN_ENABLE_AVX512 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx512skx_vpshufb_u64, 64, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx512skx_vpshufb_u128, 128, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx512skx_vpshufb_u192, 192, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx512skx_vpshufb_u256, 256, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u64, 64, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u128, 128, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u192, 192, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86, xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u256, 256, uint8_t, void, nullptr)
#endif

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(xnn_arch_wasmsimd, xnn_x8_lut_ukernel__wasmsimd_u16, 16, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_wasmsimd, xnn_x8_lut_ukernel__wasmsimd_u32, 32, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_wasmsimd, xnn_x8_lut_ukernel__wasmsimd_u48, 48, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_wasmsimd, xnn_x8_lut_ukernel__wasmsimd_u64, 64, uint8_t, void, nullptr)
#endif

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(xnn_arch_wasmsimd, xnn_x8_lut_ukernel__wasmpshufb_u16, 16, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_wasmsimd, xnn_x8_lut_ukernel__wasmpshufb_u32, 32, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_wasmsimd, xnn_x8_lut_ukernel__wasmpshufb_u48, 48, uint8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_wasmsimd, xnn_x8_lut_ukernel__wasmpshufb_u64, 64, uint8_t, void, nullptr)
#endif

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif
#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
