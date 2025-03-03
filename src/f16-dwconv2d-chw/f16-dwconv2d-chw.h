// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_1x8, 3, 3, 1, 1, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_1x8_acc2, 3, 3, 1, 1, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_1x8_acc3, 3, 3, 1, 1, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_1x8_acc4, 3, 3, 1, 1, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_2x8, 3, 3, 1, 1, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_2x8_acc2, 3, 3, 1, 1, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_3x8, 3, 3, 1, 1, 3, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_4x8, 3, 3, 1, 1, 4, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_5x8, 3, 3, 1, 1, 5, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_6x8, 3, 3, 1, 1, 6, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x8, 3, 3, 2, 1, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x8_acc2, 3, 3, 2, 1, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x8_acc3, 3, 3, 2, 1, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x8_acc4, 3, 3, 2, 1, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_2x8, 3, 3, 2, 1, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_2x8_acc2, 3, 3, 2, 1, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_3x8, 3, 3, 2, 1, 3, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_4x8, 3, 3, 2, 1, 4, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x8, 5, 5, 1, 2, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x8_acc2, 5, 5, 1, 2, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x8_acc3, 5, 5, 1, 2, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x8_acc4, 5, 5, 1, 2, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x8_acc5, 5, 5, 1, 2, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_2x8, 5, 5, 1, 2, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_2x8_acc2, 5, 5, 1, 2, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_2x8_acc3, 5, 5, 1, 2, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_3x8, 5, 5, 1, 2, 3, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_3x8_acc2, 5, 5, 1, 2, 3, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_4x8, 5, 5, 1, 2, 4, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_4x8_acc2, 5, 5, 1, 2, 4, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_5x8, 5, 5, 1, 2, 5, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x8, 5, 5, 2, 2, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x8_acc2, 5, 5, 2, 2, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x8_acc3, 5, 5, 2, 2, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x8_acc4, 5, 5, 2, 2, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x8_acc5, 5, 5, 2, 2, 1, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_2x8, 5, 5, 2, 2, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_2x8_acc2, 5, 5, 2, 2, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_2x8_acc3, 5, 5, 2, 2, 2, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_3x8, 5, 5, 2, 2, 3, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_3x8_acc2, 5, 5, 2, 2, 3, 8, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
