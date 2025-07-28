// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vtanh/rvvfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2025 RVVPL and Lobachevsky State University of Nizhny Novgorod
// Code adapted from https://github.com/rvvpl/rvvmf
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/vunary.h"

#define RVVMF_EXP_AS_FP16(x) (*(_Float16*)(&(x)))


#if 2 == 1
uint16_t RVVMF_FP16_tanhsp_LOOK_UP_TABLE [88] = {
0x0000, 0x0000, 0x3c00, 0x009f, 0xb557, 0x1b71, 0x2fb4,
0x0000, 0x34d8, 0x81a2, 0x3b44, 0xb465, 0xad82, 0x2df6,
0xce13, 0xb500, 0x3696, 0x8359, 0x3aa5, 0xb578, 0xae6e,
0x3105, 0xc687, 0xb700, 0x3870, 0x85c1, 0x398a, 0xb625,
0xa160, 0x3072, 0xb632, 0xb900, 0x39a2, 0x8a70, 0x3809,
0xb5ae, 0x2ed8, 0x2ab3, 0xbcd6, 0xbb00, 0x3ac9, 0x0892,
0x347d, 0xb39d, 0x2e99, 0xa1aa, 0x291e, 0xbd00, 0x3b88,
0x8207, 0x2f49, 0xaedb, 0x2bef, 0xa615, 0x2351, 0xbf00,
0x3be5, 0x8aa2, 0x26cf, 0xa6b6, 0x2458, 0xa040, 0x1a36,
0xc100, 0x3bfc, 0x084a, 0x1b75, 0x9b6f, 0x18ef, 0x9521,
0x1017, 0xc300, 0x3bff, 0x8599, 0x14b3, 0x94bb, 0x1194,
0x117e, 0x2048, 0xc414, 0x3c00, 0x0000, 0x0000, 0x0000,
0x0000, 0x0000, 0x0000, 0x0000};
#else
extern uint16_t RVVMF_FP16_tanhsp_LOOK_UP_TABLE [88];
#endif

#if 2 != 8
vfloat16m2_t __riscv_vtanh_f16m2(vfloat16m2_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e16m2(avl);
    vuint16m2_t ix = __riscv_vand_vx_u16m2(
                 __riscv_vreinterpret_v_f16m2_u16m2(x), 0x7fff, vl);
    
    vuint16m2_t index = __riscv_vsrl_vx_u16m2(ix, 9, vl);
    index = __riscv_vsub_vx_u16m2(index, 25, vl);

    vbool8_t mask = __riscv_vmsltu_vx_u16m2_b8(ix, 0x3400, vl);
    index = __riscv_vmerge_vxm_u16m2(index, 0x0000, mask, vl);
     
    // 0x1.0a4p+2f16
    mask = __riscv_vmsgtu_vx_u16m2_b8(ix, 0x4429, vl);
    vfloat16m2_t y = __riscv_vreinterpret_v_u16m2_f16m2(
                        __riscv_vmerge_vxm_u16m2(ix, 0x0000, mask, vl));
    index = __riscv_vmerge_vxm_u16m2(index, 10, mask, vl);
    
    index = __riscv_vsll_vx_u16m2(index, 4, vl);
            
    vfloat16m2_t p0H = __riscv_vloxei16_v_f16m2((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE, index, vl);
    vfloat16m2_t p0L = __riscv_vloxei16_v_f16m2((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 1, index, vl);
    vfloat16m2_t  p1 = __riscv_vloxei16_v_f16m2((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 2, index, vl);
    vfloat16m2_t  p2 = __riscv_vloxei16_v_f16m2((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 3, index, vl);
    vfloat16m2_t  p3 = __riscv_vloxei16_v_f16m2((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 4, index, vl);
    vfloat16m2_t  p4 = __riscv_vloxei16_v_f16m2((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 5, index, vl);
    vfloat16m2_t  p5 = __riscv_vloxei16_v_f16m2((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 6, index, vl);
    vfloat16m2_t x_m = __riscv_vloxei16_v_f16m2((_Float16*)RVVMF_FP16_tanhsp_LOOK_UP_TABLE + 7, index, vl);
    
    y = __riscv_vfadd_vv_f16m2(y, x_m, vl);
    
    vfloat16m2_t px = __riscv_vfmadd_vv_f16m2(y, p5, p4, vl);
    px = __riscv_vfmadd_vv_f16m2(px, y, p3, vl);
    px = __riscv_vfmadd_vv_f16m2(px, y, p2, vl);
    px = __riscv_vfmadd_vv_f16m2(px, y, p1, vl);
    px = __riscv_vfmadd_vv_f16m2(px, y, p0L, vl);
    px = __riscv_vfadd_vv_f16m2(px, p0H, vl);
    
    vuint16m2_t signx = __riscv_vand_vx_u16m2(
                __riscv_vreinterpret_v_f16m2_u16m2(x), 0x8000, vl);
    px = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vor_vv_u16m2(
                __riscv_vreinterpret_v_f16m2_u16m2(px), signx, vl));

#ifndef __FAST_MATH__
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

void xnn_f16_vtanh_ukernel__rvvfp16arith_tanh_u2v(
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
    const size_t n = __riscv_vsetvl_e16m2(batch);
    vfloat16m2_t vx = __riscv_vle16_v_f16m2((_Float16*)input, n);
    input += n;
    vfloat16m2_t vacc = __riscv_vtanh_f16m2(vx, n);
    __riscv_vse16_v_f16m2((_Float16*)output, vacc, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
