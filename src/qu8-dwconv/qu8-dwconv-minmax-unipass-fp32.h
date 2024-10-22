// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Arguments are:
// XNN_DWCONV_UNIPASS(arch, name, c_block, pipelined, cr, kr, datatype, weights_type,params_type, init_fn)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__neon_mul16, 8, false, 8, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_v8, xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__neonv8_mul16, 8, false, 8, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neonv8_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__neon_mul16, 16, false, 16, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_v8, xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__neonv8_mul16, 16, false, 16, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neonv8_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qu8_dwconv_minmax_fp32_ukernel_9p32c__neon_mul16, 32, false, 32, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_v8, xnn_qu8_dwconv_minmax_fp32_ukernel_9p32c__neonv8_mul16, 32, false, 32, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neonv8_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__neon_mul16, 8, false, 8, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_v8, xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__neonv8_mul16, 8, false, 8, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neonv8_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__neon_mul16, 16, false, 16, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_v8, xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__neonv8_mul16, 16, false, 16, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neonv8_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon, xnn_qu8_dwconv_minmax_fp32_ukernel_25p32c__neon_mul16, 32, false, 32, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neon_params)
XNN_DWCONV_UNIPASS(xnn_arch_arm_neon_v8, xnn_qu8_dwconv_minmax_fp32_ukernel_25p32c__neonv8_mul16, 32, false, 32, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_neonv8_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__sse2_mul16, 8, false, 8, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_sse4_1, xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul16, 8, false, 8, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_sse4_1, xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul32, 8, false, 8, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__sse2_mul16, 16, false, 16, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_sse4_1, xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__sse41_mul16, 16, false, 16, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_sse4_1, xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__sse41_mul32, 16, false, 16, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__sse2_mul16, 8, false, 8, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_sse4_1, xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul16, 8, false, 8, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_sse4_1, xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul32, 8, false, 8, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__sse2_mul16, 16, false, 16, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_sse4_1, xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__sse41_mul16, 16, false, 16, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_sse4_1, xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__sse41_mul32, 16, false, 16, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx, xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__avx_mul16, 8, false, 8, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx, xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__avx_mul32, 8, false, 8, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx2, xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__avx2_mul32, 8, false, 8, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx, xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__avx_mul16, 16, false, 16, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx, xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__avx_mul32, 16, false, 16, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx2, xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul32, 16, false, 16, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx2, xnn_qu8_dwconv_minmax_fp32_ukernel_9p32c__avx2_mul32, 32, false, 32, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx, xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__avx_mul16, 8, false, 8, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx, xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__avx_mul32, 8, false, 8, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx2, xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__avx2_mul32, 8, false, 8, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx, xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__avx_mul16, 16, false, 16, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx, xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__avx_mul32, 16, false, 16, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx2, xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul32, 16, false, 16, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx2, xnn_qu8_dwconv_minmax_fp32_ukernel_25p32c__avx2_mul32, 32, false, 32, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx512skx, xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__avx512skx_mul32, 16, false, 16, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx512skx, xnn_qu8_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32, 32, false, 32, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx512skx, xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__avx512skx_mul32, 16, false, 16, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(xnn_arch_x86_avx512skx, xnn_qu8_dwconv_minmax_fp32_ukernel_25p32c__avx512skx_mul32, 32, false, 32, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__wasmsimd_mul16, 8, false, 8, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__wasmsimd_mul16, 16, false, 16, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__wasmsimd_mul16, 8, false, 8, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p16c__wasmsimd_mul16, 16, false, 16, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p1c__wasm_fmagic, 1, false, 1, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p2c__wasm_fmagic, 2, false, 2, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p4c__wasm_fmagic, 4, false, 4, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p1c__wasm_fmagic, 1, false, 1, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p2c__wasm_fmagic, 2, false, 2, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p4c__wasm_fmagic, 4, false, 4, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p1c__scalar_fmagic, 1, false, 1, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p1c__scalar_imagic, 1, false, 1, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p1c__scalar_lrintf, 1, false, 1, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p2c__scalar_fmagic, 2, false, 2, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p2c__scalar_imagic, 2, false, 2, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p2c__scalar_lrintf, 2, false, 2, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p4c__scalar_fmagic, 4, false, 4, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p4c__scalar_imagic, 4, false, 4, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_9p4c__scalar_lrintf, 4, false, 4, 9, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p1c__scalar_fmagic, 1, false, 1, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p1c__scalar_imagic, 1, false, 1, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p1c__scalar_lrintf, 1, false, 1, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p2c__scalar_fmagic, 2, false, 2, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p2c__scalar_imagic, 2, false, 2, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p2c__scalar_lrintf, 2, false, 2, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p4c__scalar_fmagic, 4, false, 4, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p4c__scalar_imagic, 4, false, 4, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)
XNN_DWCONV_UNIPASS(0, xnn_qu8_dwconv_minmax_fp32_ukernel_25p4c__scalar_lrintf, 4, false, 4, 25, uint8_t, void, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_fp32_scalar_params)

