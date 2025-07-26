// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vexp/rvvfp16arith.c.in
//   Generator: tools/xngen
//
/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025 The RVVMF Authors (Elena Panova)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  exp.c                                        *
 *   Contains: intrinsic rvv 1.0 function exp            *
 *      for float16, accuracy=0.501 ulp                  *
 *      in domain [underflow, overflow]                  *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the e-base exponent of input vector V        *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction to a small interval near 0   *
 *    2) Additional reduction using the look-up table    *
 *       of size 2^k (k=3)                               *
 *    3) Polynomial degree: 3                            *
 *    4) Reconstruction of the result                    *
 *                                                       *
 *                                                       *
 *********************************************************
*/


#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/vunary.h"

#if 4 == 1
  #define RVV_BOOL vbool16_t
#endif

#if 4 == 2
  #define RVV_BOOL vbool8_t
#endif

#if 4 == 4
  #define RVV_BOOL vbool4_t
#endif

#if 4 == 8
  #define RVV_BOOL vbool2_t
#endif



/* ---------- UTILS ---------- */

#ifndef forceinline 
    #if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
        #define forceinline __attribute__((always_inline)) inline
    #else
        #define forceinline inline
    #endif
#endif

typedef _Float16 FLOAT16_T;

#define RVVMF_EXP_AS_FP16(x) (*(FLOAT16_T*)(&(x)))

forceinline vfloat16m4_t calc_polynom_deg_1_f16m4(vfloat16m4_t x, FLOAT16_T a0, FLOAT16_T a1, size_t vl)
{ 
    return __riscv_vfmadd_vf_f16m4(x, a1, __riscv_vfmv_v_f_f16m4(a0, vl), vl);
}

forceinline void fast_2_sum_vv_f16m4(vfloat16m4_t a, vfloat16m4_t b, vfloat16m4_t* sh, vfloat16m4_t* sl, size_t vl)
{
    *sh = __riscv_vfadd_vv_f16m4(a, b, vl);
    *sl = __riscv_vfsub_vv_f16m4(b, __riscv_vfsub_vv_f16m4(*sh, a, vl), vl);
}
    
forceinline void fast_2_sum_fv_f16m4(FLOAT16_T a, vfloat16m4_t b, vfloat16m4_t* sh, vfloat16m4_t* sl, size_t vl)
{
    *sh = __riscv_vfadd_vf_f16m4(b, a, vl);
    *sl = __riscv_vfsub_vv_f16m4(b, __riscv_vfsub_vf_f16m4(*sh, a, vl), vl);
}
        
forceinline void mul21_vv_f16m4(vfloat16m4_t ah, vfloat16m4_t al, vfloat16m4_t bh, vfloat16m4_t bl, vfloat16m4_t* rh, size_t vl)
{
    vfloat16m4_t zh, zl;
    zh = __riscv_vfmul_vv_f16m4(ah, bh, vl);
    zl = __riscv_vfmsub_vv_f16m4(ah, bh, zh, vl);
    zl = __riscv_vfadd_vv_f16m4(zl, __riscv_vfmadd_vv_f16m4(ah, bl, __riscv_vfmul_vv_f16m4(al, bh, vl), vl), vl);
    *rh = __riscv_vfadd_vv_f16m4(zh, zl, vl);
}

forceinline void fma12_vv_f16m4(vfloat16m4_t ah, vfloat16m4_t bh, vfloat16m4_t ch, vfloat16m4_t* zh, vfloat16m4_t* zl, size_t vl)
{ 
    *zh = __riscv_vfmadd_vv_f16m4(ah, bh, ch, vl);
    *zl = __riscv_vfmadd_vv_f16m4(ah, bh, __riscv_vfsub_vv_f16m4(ch, *zh, vl), vl);
}

forceinline void fma12_ver2p2_vf_f16m4(vfloat16m4_t ah, FLOAT16_T bh, vfloat16m4_t ch, vfloat16m4_t* zh, vfloat16m4_t* zl, size_t vl)
{
    vfloat16m4_t sh, sl;
    *zh = __riscv_vfmadd_vf_f16m4(ah, bh, ch, vl);
    fast_2_sum_vv_f16m4(ch, __riscv_vfneg_v_f16m4(*zh, vl), &sh, &sl, vl);
    
    *zl = __riscv_vfmadd_vf_f16m4(ah, bh, sh, vl);
    *zl = __riscv_vfadd_vv_f16m4(*zl, sl, vl);
}

/* ---------- DATA ---------- */
#if 4 == 1

uint16_t ZERO_F16 = 0x0; 
uint16_t ONE_F16 = 0x3c00;

uint16_t EXP_EXPM1_OVERFLOW_THRESHOLD_F16 = 0x498b;  // 0x1.62cp3f16 
uint16_t EXP_SUBNORMAL_THRESHOLD_F16 = 0xc8da;       // -0x1.368p3f16 
uint16_t EXP_ZERO_THRESHOLD_F16 = 0xcc55;            // -0x1.154p4f16 
uint16_t EXP_UNDERFLOW_VALUE_F16 = 0x0;              // 0.0f16 

size_t TABLE_SIZE_DEG_F16 = 3;
uint16_t MASK_FI_BIT_F16 = 0x0007;
uint16_t MASK_HI_BIT_F16 = 0x01ff;
uint16_t MAGIC_CONST_1_F16 = 0x6600;  // 1536.0f16 
uint16_t INV_LOG2_2K_F16 = 0x49c5;    // 0x1.714p3f16 
uint16_t M_LOG2_2K_H_F16 = 0xad80;    // -0x1.6p-4f16 
uint16_t M_LOG2_2K_L_F16 = 0x91c8;    // -0x1.72p-11f16 
uint16_t M_LOG2_2K_LL_F16 = 0x8003;   // -0x1.8p-23f16 

uint16_t RVVMF_EXP_LOOK_UP_TABLE_HIGH_F16[8] = {
    0x3c00, 0x3c5d, 0x3cc2, 0x3d30,
    0x3da8, 0x3e2b, 0x3eba, 0x3f56
};
uint16_t RVVMF_EXP_LOOK_UP_TABLE_LOW_F16[8] = {
    0x0, 0x8d1f, 0x8c08, 0x8253,
    0x8f3, 0xb2a, 0x8fd, 0x18e
};

uint16_t EXP_POL_COEFF_2_F16 = 0x3800;  // 0x1p-1f16 
uint16_t EXP_POL_COEFF_3_F16 = 0x3154;  // 0x1.55p-3f16 

#else
extern uint16_t ZERO_F16; 
extern uint16_t ONE_F16;
extern uint16_t EXP_EXPM1_OVERFLOW_THRESHOLD_F16;  // 0x1.62cp3f16 
extern uint16_t EXP_SUBNORMAL_THRESHOLD_F16;       // -0x1.368p3f16 
extern uint16_t EXP_ZERO_THRESHOLD_F16;            // -0x1.154p4f16 
extern uint16_t EXP_UNDERFLOW_VALUE_F16;              // 0.0f16 
extern size_t TABLE_SIZE_DEG_F16;
extern uint16_t MASK_FI_BIT_F16;
extern uint16_t MASK_HI_BIT_F16;
extern uint16_t MAGIC_CONST_1_F16;  // 1536.0f16 
extern uint16_t INV_LOG2_2K_F16;    // 0x1.714p3f16 
extern uint16_t M_LOG2_2K_H_F16;    // -0x1.6p-4f16 
extern uint16_t M_LOG2_2K_L_F16;    // -0x1.72p-11f16 
extern uint16_t M_LOG2_2K_LL_F16;   // -0x1.8p-23f16 
extern uint16_t RVVMF_EXP_LOOK_UP_TABLE_HIGH_F16[8];
extern uint16_t RVVMF_EXP_LOOK_UP_TABLE_LOW_F16[8];
extern uint16_t EXP_POL_COEFF_2_F16;  // 0x1p-1f16 
extern uint16_t EXP_POL_COEFF_3_F16;  // 0x1.55p-3f16 

#endif


/* ---------- EXP IMPLEMENTATION ---------- */

#if 4 == 1
forceinline void check_special_cases_f16m4(vfloat16m4_t* x, vfloat16m4_t* special, RVV_BOOL* specialMask,
    FLOAT16_T overflowThreshold, size_t vl)
{
    uint16_t pinf = 0x7c00;    
    RVV_BOOL mask;
    // check +inf
    *specialMask = __riscv_vmfeq_vf_f16m4_b16(*x, RVVMF_EXP_AS_FP16(pinf), vl);
    *special = __riscv_vfmerge_vfm_f16m4(*x, RVVMF_EXP_AS_FP16(pinf), *specialMask, vl);
    // check overflow
    mask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f16m4_b16(*x, overflowThreshold, vl),
        __riscv_vmflt_vf_f16m4_b16(*x, RVVMF_EXP_AS_FP16(pinf), vl), vl);
    *special = __riscv_vfmerge_vfm_f16m4(*special, RVVMF_EXP_AS_FP16(pinf), mask, vl);
    *specialMask = __riscv_vmor_mm_b16(*specialMask, mask, vl);  
    /* if (__riscv_vcpop_m_b16(mask, vl))
        volatile double exception = DBL_MAX*2.0; */  // FE_OVERFLOW
    
    // NaNs, overflow, -inf -- automatically
    *x = __riscv_vfmerge_vfm_f16m4(*x, RVVMF_EXP_AS_FP16(ZERO_F16), *specialMask, vl);
}
#endif

#if 4 == 2
forceinline void check_special_cases_f16m4(vfloat16m4_t* x, vfloat16m4_t* special, RVV_BOOL* specialMask,
    FLOAT16_T overflowThreshold, size_t vl)
{
    uint16_t pinf = 0x7c00;    
    RVV_BOOL mask;
    // check +inf
    *specialMask = __riscv_vmfeq_vf_f16m4_b8(*x, RVVMF_EXP_AS_FP16(pinf), vl);
    *special = __riscv_vfmerge_vfm_f16m4(*x, RVVMF_EXP_AS_FP16(pinf), *specialMask, vl);
    // check overflow
    mask = __riscv_vmand_mm_b8(__riscv_vmfgt_vf_f16m4_b8(*x, overflowThreshold, vl),
        __riscv_vmflt_vf_f16m4_b8(*x, RVVMF_EXP_AS_FP16(pinf), vl), vl);
    *special = __riscv_vfmerge_vfm_f16m4(*special, RVVMF_EXP_AS_FP16(pinf), mask, vl);
    *specialMask = __riscv_vmor_mm_b8(*specialMask, mask, vl);  
    /* if (__riscv_vcpop_m_b8(mask, vl))
        volatile double exception = DBL_MAX*2.0; */  // FE_OVERFLOW
    
    // NaNs, overflow, -inf -- automatically
    *x = __riscv_vfmerge_vfm_f16m4(*x, RVVMF_EXP_AS_FP16(ZERO_F16), *specialMask, vl);
}
#endif

#if 4 == 4
forceinline void check_special_cases_f16m4(vfloat16m4_t* x, vfloat16m4_t* special, RVV_BOOL* specialMask,
    FLOAT16_T overflowThreshold, size_t vl)
{
    uint16_t pinf = 0x7c00;    
    RVV_BOOL mask;
    // check +inf
    *specialMask = __riscv_vmfeq_vf_f16m4_b4(*x, RVVMF_EXP_AS_FP16(pinf), vl);
    *special = __riscv_vfmerge_vfm_f16m4(*x, RVVMF_EXP_AS_FP16(pinf), *specialMask, vl);
    // check overflow
    mask = __riscv_vmand_mm_b4(__riscv_vmfgt_vf_f16m4_b4(*x, overflowThreshold, vl),
        __riscv_vmflt_vf_f16m4_b4(*x, RVVMF_EXP_AS_FP16(pinf), vl), vl);
    *special = __riscv_vfmerge_vfm_f16m4(*special, RVVMF_EXP_AS_FP16(pinf), mask, vl);
    *specialMask = __riscv_vmor_mm_b4(*specialMask, mask, vl);  
    /* if (__riscv_vcpop_m_b4(mask, vl))
        volatile double exception = DBL_MAX*2.0; */  // FE_OVERFLOW
    
    // NaNs, overflow, -inf -- automatically
    *x = __riscv_vfmerge_vfm_f16m4(*x, RVVMF_EXP_AS_FP16(ZERO_F16), *specialMask, vl);
}
#endif

#if 4 == 8
forceinline void check_special_cases_f16m4(vfloat16m4_t* x, vfloat16m4_t* special, RVV_BOOL* specialMask,
    FLOAT16_T overflowThreshold, size_t vl)
{
    uint16_t pinf = 0x7c00;    
    RVV_BOOL mask;
    // check +inf
    *specialMask = __riscv_vmfeq_vf_f16m4_b2(*x, RVVMF_EXP_AS_FP16(pinf), vl);
    *special = __riscv_vfmerge_vfm_f16m4(*x, RVVMF_EXP_AS_FP16(pinf), *specialMask, vl);
    // check overflow
    mask = __riscv_vmand_mm_b2(__riscv_vmfgt_vf_f16m4_b2(*x, overflowThreshold, vl),
        __riscv_vmflt_vf_f16m4_b2(*x, RVVMF_EXP_AS_FP16(pinf), vl), vl);
    *special = __riscv_vfmerge_vfm_f16m4(*special, RVVMF_EXP_AS_FP16(pinf), mask, vl);
    *specialMask = __riscv_vmor_mm_b2(*specialMask, mask, vl);  
    /* if (__riscv_vcpop_m_b2(mask, vl))
        volatile double exception = DBL_MAX*2.0; */  // FE_OVERFLOW
    
    // NaNs, overflow, -inf -- automatically
    *x = __riscv_vfmerge_vfm_f16m4(*x, RVVMF_EXP_AS_FP16(ZERO_F16), *specialMask, vl);
}
#endif


forceinline void do_exp_argument_reduction_hl_f16m4(vfloat16m4_t x,
    vfloat16m4_t* yh, vfloat16m4_t* yl, vuint16m4_t* ei, vuint16m4_t* fi, size_t vl)
{
    vfloat16m4_t vmagicConst1 = __riscv_vfmv_v_f_f16m4(RVVMF_EXP_AS_FP16(MAGIC_CONST_1_F16), vl);
    vfloat16m4_t h = __riscv_vfmadd_vf_f16m4(x, RVVMF_EXP_AS_FP16(INV_LOG2_2K_F16), vmagicConst1, vl);
    vuint16m4_t hi = __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(h), MASK_HI_BIT_F16, vl);
    *fi = __riscv_vand_vx_u16m4(hi, MASK_FI_BIT_F16, vl);
    *ei = __riscv_vsrl_vx_u16m4(hi, TABLE_SIZE_DEG_F16, vl);
    h = __riscv_vfsub_vv_f16m4(h, vmagicConst1, vl);
    fma12_ver2p2_vf_f16m4(h, RVVMF_EXP_AS_FP16(M_LOG2_2K_L_F16),
        __riscv_vfmadd_vf_f16m4(h, RVVMF_EXP_AS_FP16(M_LOG2_2K_H_F16), x, vl), yh, yl, vl);
    *yl = __riscv_vfmadd_vf_f16m4(h, RVVMF_EXP_AS_FP16(M_LOG2_2K_LL_F16), *yl, vl);
    fast_2_sum_vv_f16m4(*yh, *yl, yh, yl, vl);
}

forceinline void get_table_values_hl_f16m4(
    vuint16m4_t* index, vfloat16m4_t* th, vfloat16m4_t* tl, size_t vl)
{
    *index = __riscv_vmul_vx_u16m4(*index, (uint16_t)sizeof(FLOAT16_T), vl);
    *th = __riscv_vloxei16_v_f16m4((FLOAT16_T*)RVVMF_EXP_LOOK_UP_TABLE_HIGH_F16, *index, vl);
    *tl = __riscv_vloxei16_v_f16m4((FLOAT16_T*)RVVMF_EXP_LOOK_UP_TABLE_LOW_F16, *index, vl);
}

forceinline void calculate_exp_polynom_hl_f16m4(vfloat16m4_t yh, vfloat16m4_t yl, vfloat16m4_t* ph, vfloat16m4_t* pl, size_t vl)
{
    vfloat16m4_t sqryh = __riscv_vfmul_vv_f16m4(yh, yh, vl);
    vfloat16m4_t r = calc_polynom_deg_1_f16m4(yh, RVVMF_EXP_AS_FP16(EXP_POL_COEFF_2_F16),
        RVVMF_EXP_AS_FP16(EXP_POL_COEFF_3_F16), vl); 
    fma12_vv_f16m4(sqryh, r, yh, ph, pl, vl);
    *pl = __riscv_vfadd_vv_f16m4(*pl, yl, vl);
}

forceinline void update_exponent_f16m4(vuint16m4_t ei, vfloat16m4_t* res, size_t vl)
{
    *res = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vadd_vv_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(*res), __riscv_vsll_vx_u16m4(ei, (size_t)10, vl), vl));
}

forceinline void update_exponent_with_subnormal_f16m4(FLOAT16_T subnormalThreshold, vfloat16m4_t x,
    vuint16m4_t ei, vfloat16m4_t* res, size_t vl)
{
#ifndef __FAST_MATH__
    uint16_t ninf = 0xfc00;
#if 4 == 1
    RVV_BOOL subnormalMask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f16m1_b16(x, RVVMF_EXP_AS_FP16(ninf), vl),
        __riscv_vmflt_vf_f16m4_b16(x, subnormalThreshold, vl), vl);
#endif
#if 4 == 2
    RVV_BOOL subnormalMask = __riscv_vmand_mm_b8(__riscv_vmfgt_vf_f16m2_b8(x, RVVMF_EXP_AS_FP16(ninf), vl),
        __riscv_vmflt_vf_f16m4_b8(x, subnormalThreshold, vl), vl);
#endif
#if 4 == 4
    RVV_BOOL subnormalMask = __riscv_vmand_mm_b4(__riscv_vmfgt_vf_f16m4_b4(x, RVVMF_EXP_AS_FP16(ninf), vl),
        __riscv_vmflt_vf_f16m4_b4(x, subnormalThreshold, vl), vl);
#endif
#if 4 == 8
    RVV_BOOL subnormalMask = __riscv_vmand_mm_b2(__riscv_vmfgt_vf_f16m8_b2(x, RVVMF_EXP_AS_FP16(ninf), vl),
        __riscv_vmflt_vf_f16m4_b2(x, subnormalThreshold, vl), vl);
#endif
    /* if (__riscv_vcpop_m_b16(subnormalMask, vl))
        volatile double exception = nextafter(DBL_MIN/(double((uint16_t)1 << 52)), 0.0) */ // FE_UNDERFLOW
    
    vfloat16m4_t subnormalRes;
    vuint16m4_t shiftNum = __riscv_vreinterpret_v_i16m4_u16m4(__riscv_vneg_v_i16m4(__riscv_vreinterpret_v_u16m4_i16m4(ei), vl));
    shiftNum = __riscv_vadd_vx_u16m4(__riscv_vand_vx_u16m4(shiftNum, (uint16_t)0x003f, vl), (uint16_t)1, vl);
    shiftNum = __riscv_vsll_vx_u16m4(shiftNum, (size_t)10, vl);
    subnormalRes = __riscv_vfadd_vv_f16m4(*res, __riscv_vreinterpret_v_u16m4_f16m4(shiftNum), vl);
    subnormalRes = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vx_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(subnormalRes), (uint16_t)0x83ff, vl));
#endif

    update_exponent_f16m4(ei, res, vl);
    
#ifndef __FAST_MATH__
    *res = __riscv_vmerge_vvm_f16m4(*res, subnormalRes, subnormalMask, vl);
#endif   
}

forceinline void reconstruct_exp_hl_hl_f16m4(vfloat16m4_t x, vuint16m4_t ei, vfloat16m4_t th, vfloat16m4_t tl,
    vfloat16m4_t pm4h, vfloat16m4_t pm4l, vfloat16m4_t* res, FLOAT16_T subnormalThreshold, size_t vl)
{
    vfloat16m4_t sh, sl;
    fast_2_sum_fv_f16m4(RVVMF_EXP_AS_FP16(ONE_F16), pm4h, &sh, &sl, vl);
    sl = __riscv_vfadd_vv_f16m4(sl, pm4l, vl);
    mul21_vv_f16m4(th, tl, sh, sl, res, vl);
    update_exponent_with_subnormal_f16m4(subnormalThreshold, x, ei, res, vl);
}

forceinline void update_underflow_f16m4(vfloat16m4_t x, vfloat16m4_t* res,
    FLOAT16_T underflowThreshold, FLOAT16_T underflowValue, size_t vl)
{
#if 4 == 1
    RVV_BOOL underflowMask = __riscv_vmflt_vf_f16m1_b16(x, underflowThreshold, vl);
#endif
#if 4 == 2
    RVV_BOOL underflowMask = __riscv_vmflt_vf_f16m2_b8(x, underflowThreshold, vl);
#endif
#if 4 == 4
    RVV_BOOL underflowMask = __riscv_vmflt_vf_f16m4_b4(x, underflowThreshold, vl);
#endif
#if 4 == 8
    RVV_BOOL underflowMask = __riscv_vmflt_vf_f16m8_b2(x, underflowThreshold, vl);
#endif
    *res = __riscv_vfmerge_vfm_f16m4(*res, underflowValue, underflowMask, vl);
}

forceinline void set_pos_sign_f16m4(vfloat16m4_t* res, size_t vl)
{
    uint16_t signMask = 0x7fff;
    *res = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vx_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(*res), signMask, vl));
}

vfloat16m4_t __riscv_vexp_f16m4(vfloat16m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m4(avl);

    vfloat16m4_t res, yh, yl, th, tl, pm4h, pm4l;
    vuint16m4_t ei, fi;
    
#ifndef __FAST_MATH__
    FLOAT16_T zeroThreshold = RVVMF_EXP_AS_FP16(EXP_ZERO_THRESHOLD_F16);
    vfloat16m4_t special;
    RVV_BOOL specialMask;
    check_special_cases_f16m4(&x, &special, &specialMask, RVVMF_EXP_AS_FP16(EXP_EXPM1_OVERFLOW_THRESHOLD_F16), vl);
#else
    FLOAT16_T zeroThreshold = RVVMF_EXP_AS_FP16(EXP_SUBNORMAL_THRESHOLD_F16);    
#endif
    
    do_exp_argument_reduction_hl_f16m4(x, &yh, &yl, &ei, &fi, vl);
    get_table_values_hl_f16m4(&fi, &th, &tl, vl);
    calculate_exp_polynom_hl_f16m4(yh, yl, &pm4h, &pm4l, vl);
    reconstruct_exp_hl_hl_f16m4(x, ei, th, tl, pm4h, pm4l, &res, RVVMF_EXP_AS_FP16(EXP_SUBNORMAL_THRESHOLD_F16), vl);
    update_underflow_f16m4(x, &res, zeroThreshold, RVVMF_EXP_AS_FP16(EXP_UNDERFLOW_VALUE_F16), vl);
    set_pos_sign_f16m4(&res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m4(res, special, specialMask, vl);
#endif

    return res;
}




void xnn_f16_vexp_ukernel__rvvfp16arith_exp_u4v(
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
    const size_t n = __riscv_vsetvl_e16m4(batch);
    vfloat16m4_t vx = __riscv_vle16_v_f16m4((FLOAT16_T*)input, n);
    input += n;
    vfloat16m4_t vacc = __riscv_vexp_f16m4(vx, n);
    __riscv_vse16_v_f16m4((FLOAT16_T*)output, vacc, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
