// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vtanh/rvvfp16arith.c.in
//   Generator: tools/xngen
//
/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025 The RVVMF Authors (Valentin Volokitin)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  tanh.c                                       *
 *   Contains: intrinsic function tanh for f64, f32, f16 *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the hyperbolic tangent of input vector V     *
 *                                                       *
 * Algorithm:                                            *
 *    1) Piecewise polynomial approximation on segments: *
 *       f64 [0, 0x1.30fc1931f09c9p+4] - 94,             *
 *       f32 [0, 0x1.205966p+3] - 83,                    *
 *       f16 [0, 0x1.0a4p+2] - 10                        *
 *    2) For efficiency, some sections are divided into  *
 *       2 (fp16), 4 (fp64) or 8 (fp32) equal sections   *
 *    3) Polynomial degrees: f64 - 13, f32 - 5, f16 - 5  *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/vunary.h"

#define RVVMF_EXP_AS_FP16(x) (*(_Float16*)(&(x)))


static uint16_t RVVMF_FP16_tanhsp_LOOK_UP_TABLE [88] = {
0x00000000, 0x00000000, 0x00003c00, 0x0000009f, 0x0000b557, 0x00001b71, 0x00002fb4,
0x00000000, 0x000034d8, 0x000081a2, 0x00003b44, 0x0000b465, 0x0000ad82, 0x00002df6,
0x0000ce13, 0x0000b500, 0x00003696, 0x00008359, 0x00003aa5, 0x0000b578, 0x0000ae6e,
0x00003105, 0x0000c687, 0x0000b700, 0x00003870, 0x000085c1, 0x0000398a, 0x0000b625,
0x0000a160, 0x00003072, 0x0000b632, 0x0000b900, 0x000039a2, 0x00008a70, 0x00003809,
0x0000b5ae, 0x00002ed8, 0x00002ab3, 0x0000bcd6, 0x0000bb00, 0x00003ac9, 0x00000892,
0x0000347d, 0x0000b39d, 0x00002e99, 0x0000a1aa, 0x0000291e, 0x0000bd00, 0x00003b88,
0x00008207, 0x00002f49, 0x0000aedb, 0x00002bef, 0x0000a615, 0x00002351, 0x0000bf00,
0x00003be5, 0x00008aa2, 0x000026cf, 0x0000a6b6, 0x00002458, 0x0000a040, 0x00001a36,
0x0000c100, 0x00003bfc, 0x0000084a, 0x00001b75, 0x00009b6f, 0x000018ef, 0x00009521,
0x00001017, 0x0000c300, 0x00003bff, 0x00008599, 0x000014b3, 0x000094bb, 0x00001194,
0x0000117e, 0x00002048, 0x0000c414, 0x00003c00, 0x00000000, 0x00000000, 0x00000000,
0x00000000, 0x00000000, 0x00000000, 0x00000000};

#if 1 != 8
vfloat16m1_t __riscv_vtanh_f16m1(vfloat16m1_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e16m1(avl);
    vuint16m1_t ix = __riscv_vand_vx_u16m1(
                 __riscv_vreinterpret_v_f16m1_u16m1(x), 0x7fff, vl);
    
    vuint16m1_t index = __riscv_vsrl_vx_u16m1(ix, 9, vl);
    index = __riscv_vsub_vx_u16m1(index, 25, vl);

#if 1 == 1
    vbool16_t mask = __riscv_vmsltu_vx_u16m1_b16(ix, 0x3400, vl);
#endif
#if 1 == 2
    vbool8_t mask = __riscv_vmsltu_vx_u16m2_b8(ix, 0x3400, vl);
#endif
#if 1 == 4
    vbool4_t mask = __riscv_vmsltu_vx_u16m4_b4(ix, 0x3400, vl);
#endif
    index = __riscv_vmerge_vxm_u16m1(index, 0x0000, mask, vl);
     
    // 0x1.0a4p+2f16
#if 1 == 1
    mask = __riscv_vmsgtu_vx_u16m1_b16(ix, 0x4429, vl);
#endif
#if 1 == 2
    mask = __riscv_vmsgtu_vx_u16m2_b8(ix, 0x4429, vl);
#endif
#if 1 == 4
    mask = __riscv_vmsgtu_vx_u16m4_b4(ix, 0x4429, vl);
#endif
    vfloat16m1_t y = __riscv_vreinterpret_v_u16m1_f16m1(
                        __riscv_vmerge_vxm_u16m1(ix, 0x0000, mask, vl));
    index = __riscv_vmerge_vxm_u16m1(index, 10, mask, vl);
    
    index = __riscv_vsll_vx_u16m1(index, 4, vl);
            
    vfloat16m1_t p0H = __riscv_vloxei16_v_f16m1((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE, index, vl);
    vfloat16m1_t p0L = __riscv_vloxei16_v_f16m1((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 1, index, vl);
    vfloat16m1_t  p1 = __riscv_vloxei16_v_f16m1((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 2, index, vl);
    vfloat16m1_t  p2 = __riscv_vloxei16_v_f16m1((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 3, index, vl);
    vfloat16m1_t  p3 = __riscv_vloxei16_v_f16m1((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 4, index, vl);
    vfloat16m1_t  p4 = __riscv_vloxei16_v_f16m1((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 5, index, vl);
    vfloat16m1_t  p5 = __riscv_vloxei16_v_f16m1((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 6, index, vl);
    vfloat16m1_t x_m = __riscv_vloxei16_v_f16m1((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 7, index, vl);
    
    y = __riscv_vfadd_vv_f16m1(y, x_m, vl);
    
    vfloat16m1_t px = __riscv_vfmadd_vv_f16m1(y, p5, p4, vl);
    px = __riscv_vfmadd_vv_f16m1(px, y, p3, vl);
    px = __riscv_vfmadd_vv_f16m1(px, y, p2, vl);
    px = __riscv_vfmadd_vv_f16m1(px, y, p1, vl);
    px = __riscv_vfmadd_vv_f16m1(px, y, p0L, vl);
    px = __riscv_vfadd_vv_f16m1(px, p0H, vl);
    
    vuint16m1_t signx = __riscv_vand_vx_u16m1(
                __riscv_vreinterpret_v_f16m1_u16m1(x), 0x8000, vl);
    px = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vor_vv_u16m1(
                __riscv_vreinterpret_v_f16m1_u16m1(px), signx, vl));

#ifndef __FAST_MATH__
#if 1 == 1
    vbool16_t mask_sNaN = __riscv_vmsgtu_vx_u16m1_b16 (ix, 0x7c00, vl);
    px = __riscv_vmerge_vvm_f16m1(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b16(mask_sNaN,
                  __riscv_vmsltu_vx_u16m1_b16(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        px = __riscv_vfmerge_vfm_f16m1(px, x1, mask_sNaN, vl);
    }
#endif
#if 1 == 2
    vbool8_t mask_sNaN = __riscv_vmsgtu_vx_u16m2_b8 (ix, 0x7c00, vl);
    px = __riscv_vmerge_vvm_f16m2(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b8(mask_sNaN,
                  __riscv_vmsltu_vx_u16m2_b8(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_sNaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        px = __riscv_vfmerge_vfm_f16m2(px, x1, mask_sNaN, vl);
    }
#endif
#if 1 == 4
    vbool4_t mask_sNaN = __riscv_vmsgtu_vx_u16m4_b4 (ix, 0x7c00, vl);
    px = __riscv_vmerge_vvm_f16m4(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b4(mask_sNaN,
                  __riscv_vmsltu_vx_u16m4_b4(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b4(mask_sNaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        px = __riscv_vfmerge_vfm_f16m4(px, x1, mask_sNaN, vl);
    }
#endif

#endif

    return px;
}
#else

vfloat16m4_t __riscv_vtanh_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vtanh_f16m8(vfloat16m8_t x, size_t avl)
{  
    vfloat16m8_t res;
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vtanh_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e16m4(avl-vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vtanh_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}
#endif

void xnn_f16_vtanh_ukernel__rvvfp16arith_tanh_u1v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_HALF;
  do {
    const size_t n = __riscv_vsetvl_e16m1(batch);
    vfloat16m1_t vx = __riscv_vle16_v_f16m1((_Float16*)input, n);
    input += n;
    vfloat16m1_t vacc = __riscv_vtanh_f16m1(vx, n);
    __riscv_vse16_v_f16m1((_Float16*)output, vacc, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
