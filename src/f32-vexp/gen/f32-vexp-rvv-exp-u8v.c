// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vexp/rvv.c.in
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
 *      for float32, accuracy=0.501 ulp                  *
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
 *       of size 2^k (k=4)                               *
 *    3) Polynomial degree: 4                            *
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


#include <stdint.h>
#include <float.h>
#include <math.h>

#if 8 == 1
  #define RVV_BOOL vbool32_t
#endif

#if 8 == 2
  #define RVV_BOOL vbool16_t
#endif

#if 8 == 4
  #define RVV_BOOL vbool8_t
#endif

#if 8 == 8
  #define RVV_BOOL vbool4_t
#endif

/* ---------- UTILS ---------- */

#ifndef forceinline 
    #if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
        #define forceinline __attribute__((always_inline)) inline
    #else
        #define forceinline inline
    #endif
#endif

#define RVVMF_EXP_AS_FP32(x) (*(float*)(&(x)))

forceinline vfloat32m8_t calc_polynom_deg_2_f32m8(vfloat32m8_t x, float a0, float a1, float a2, size_t vl)
{ 
    return __riscv_vfmadd_vv_f32m8(x, __riscv_vfmadd_vf_f32m8(x, a2,
        __riscv_vfmv_v_f_f32m8(a1, vl), vl), __riscv_vfmv_v_f_f32m8(a0, vl), vl);
}

forceinline void fast_2_sum_vv_f32m8(vfloat32m8_t a, vfloat32m8_t b, vfloat32m8_t* sh, vfloat32m8_t* sl, size_t vl)
{
    *sh = __riscv_vfadd_vv_f32m8(a, b, vl);
    *sl = __riscv_vfsub_vv_f32m8(b, __riscv_vfsub_vv_f32m8(*sh, a, vl), vl);
}
    
forceinline void fast_2_sum_fv_f32m8(float a, vfloat32m8_t b, vfloat32m8_t* sh, vfloat32m8_t* sl, size_t vl)
{
    *sh = __riscv_vfadd_vf_f32m8(b, a, vl);
    *sl = __riscv_vfsub_vv_f32m8(b, __riscv_vfsub_vf_f32m8(*sh, a, vl), vl);
}
        
forceinline void mul21_vv_f32m8(vfloat32m8_t ah, vfloat32m8_t al, vfloat32m8_t bh, vfloat32m8_t bl, vfloat32m8_t* rh, size_t vl)
{
    vfloat32m8_t zh, zl;
    zh = __riscv_vfmul_vv_f32m8(ah, bh, vl);
    zl = __riscv_vfmsub_vv_f32m8(ah, bh, zh, vl);
    zl = __riscv_vfadd_vv_f32m8(zl, __riscv_vfmadd_vv_f32m8(ah, bl, __riscv_vfmul_vv_f32m8(al, bh, vl), vl), vl);
    *rh = __riscv_vfadd_vv_f32m8(zh, zl, vl);
}

forceinline void fma12_vv_f32m8(vfloat32m8_t ah, vfloat32m8_t bh, vfloat32m8_t ch, vfloat32m8_t* zh, vfloat32m8_t* zl, size_t vl)
{ 
    *zh = __riscv_vfmadd_vv_f32m8(ah, bh, ch, vl);
    *zl = __riscv_vfmadd_vv_f32m8(ah, bh, __riscv_vfsub_vv_f32m8(ch, *zh, vl), vl);
}

forceinline void fma12_vf_f32m8(vfloat32m8_t ah, float bh, vfloat32m8_t ch, vfloat32m8_t* zh, vfloat32m8_t* zl, size_t vl)
{ 
    *zh = __riscv_vfmadd_vf_f32m8(ah, bh, ch, vl);
    *zl = __riscv_vfmadd_vf_f32m8(ah, bh, __riscv_vfsub_vv_f32m8(ch, *zh, vl), vl);
}

/* ---------- DATA ---------- */
#if 8 == 1

uint32_t ZERO_F32 = 0x0;        // 0.0f
uint32_t ONE_F32  = 0x3f800000;  // 1.0f

uint32_t EXP_EXPM1_OVERFLOW_THRESHOLD_F32 = 0x42b17217;  // 0x1.62e42ep6f
uint32_t EXP_SUBNORMAL_THRESHOLD_F32      = 0xc2aeac4f;       // -0x1.5d589ep6f
uint32_t EXP_ZERO_THRESHOLD_F32           = 0xc2cff1b4;            // -0x1.9fe368p6f
uint32_t EXP_UNDERFLOW_VALUE_F32          = 0x0;                  // 0.0f

size_t  TABLE_SIZE_DEG_F32 = 4;
uint32_t MASK_FI_BIT_F32 = 0x0000000f;
uint32_t MASK_HI_BIT_F32 = 0x00001fff;
uint32_t MAGIC_CONST_1_F32 = 0x4b400000;  // 12582912.0f
uint32_t INV_LOG2_2K_F32 = 0x41b8aa3b;    // 0x1.715476p4f
uint32_t M_LOG2_2K_H_F32 = 0xbd317000;    // -0x1.62ep-5f
uint32_t M_LOG2_2K_L_F32 = 0xb605fdf4;    // -0x1.0bfbe8p-19f
uint32_t M_LOG2_2K_LL_F32 = 0xa9e7bcd6;   // -0x1.cf79acp-44f

uint32_t RVVMF_EXP_LOOK_UP_TABLE_HIGH_F32[16] = {
    0x3f800000, 0x3f85aac3, 0x3f8b95c2, 0x3f91c3d3,
    0x3f9837f0, 0x3f9ef532, 0x3fa5fed7, 0x3fad583f,
    0x3fb504f3, 0x3fbd08a4, 0x3fc5672a, 0x3fce248c,
    0x3fd744fd, 0x3fe0ccdf, 0x3feac0c7, 0x3ff5257d
};
uint32_t RVVMF_EXP_LOOK_UP_TABLE_LOW_F32[16] = {
    0x0, 0x334f9891, 0xb260aba1, 0x33675624,
    0x33231b71, 0x33412342, 0xb32c9d5e, 0xb22deaf6,
    0x32cfe77a, 0xb3414fe8, 0x320aa837, 0x3228fc24,
    0xb2d4a58a, 0xb21eab59, 0xb24116de, 0x32292436
};

uint32_t EXP_POL_COEFF_2_F32 = 0x3f000000;  // 0x1p-1f
uint32_t EXP_POL_COEFF_3_F32 = 0x3e2aab6f;  // 0x1.5556dep-3f
uint32_t EXP_POL_COEFF_4_F32 = 0x3d2aab4b;  // 0x1.555696p-5f
#else

extern uint32_t ZERO_F32;
extern uint32_t ONE_F32 ;
extern uint32_t EXP_EXPM1_OVERFLOW_THRESHOLD_F32;
extern uint32_t EXP_SUBNORMAL_THRESHOLD_F32     ;
extern uint32_t EXP_ZERO_THRESHOLD_F32          ;
extern uint32_t EXP_UNDERFLOW_VALUE_F32         ;

extern size_t  TABLE_SIZE_DEG_F32;
extern uint32_t MASK_FI_BIT_F32;
extern uint32_t MASK_HI_BIT_F32;
extern uint32_t MAGIC_CONST_1_F32;  // 12582912.0f
extern uint32_t INV_LOG2_2K_F32;    // 0x1.715476p4f
extern uint32_t M_LOG2_2K_H_F32;    // -0x1.62ep-5f
extern uint32_t M_LOG2_2K_L_F32;    // -0x1.0bfbe8p-19f
extern uint32_t M_LOG2_2K_LL_F32;   // -0x1.cf79acp-44f

extern uint32_t RVVMF_EXP_LOOK_UP_TABLE_HIGH_F32[16];
extern  uint32_t RVVMF_EXP_LOOK_UP_TABLE_LOW_F32[16];

extern uint32_t EXP_POL_COEFF_2_F32;  // 0x1p-1f
extern uint32_t EXP_POL_COEFF_3_F32;  // 0x1.5556dep-3f
extern uint32_t EXP_POL_COEFF_4_F32;  // 0x1.555696p-5f


#endif

/* ---------- EXP IMPLEMENTATION ---------- */

#if 8 == 1
forceinline void check_special_cases_f32m1(vfloat32m1_t* x, vfloat32m1_t* special, RVV_BOOL* specialMask,
    float overflowThreshold, size_t vl)
{
    uint32_t pinf = 0x7f800000;    
    RVV_BOOL mask;
    // check +inf
    *specialMask = __riscv_vmfeq_vf_f32m1_b32(*x, RVVMF_EXP_AS_FP32(pinf), vl);
    *special = __riscv_vfmerge_vfm_f32m1(*x, RVVMF_EXP_AS_FP32(pinf), *specialMask, vl);
    // check overflow
    mask = __riscv_vmand_mm_b32(__riscv_vmfgt_vf_f32m8_b32(*x, overflowThreshold, vl),
        __riscv_vmflt_vf_f32m1_b32(*x, RVVMF_EXP_AS_FP32(pinf), vl), vl);
    *special = __riscv_vfmerge_vfm_f32m1(*special, RVVMF_EXP_AS_FP32(pinf), mask, vl);
    *specialMask = __riscv_vmor_mm_b32(*specialMask, mask, vl);  
    /* if (__riscv_vcpop_m_b32(mask, vl))
        volatile double exception = DBL_MAX*2.0; */  // FE_OVERFLOW
    
    // NaNs, overflow, -inf -- automatically
    *x = __riscv_vfmerge_vfm_f32m1(*x, RVVMF_EXP_AS_FP32(ZERO_F32), *specialMask, vl);
}
#endif

#if 8 == 2
forceinline void check_special_cases_f32m2(vfloat32m2_t* x, vfloat32m2_t* special, RVV_BOOL* specialMask,
    float overflowThreshold, size_t vl)
{
    uint32_t pinf = 0x7f800000;    
    RVV_BOOL mask;
    // check +inf
    *specialMask = __riscv_vmfeq_vf_f32m2_b16(*x, RVVMF_EXP_AS_FP32(pinf), vl);
    *special = __riscv_vfmerge_vfm_f32m2(*x, RVVMF_EXP_AS_FP32(pinf), *specialMask, vl);
    // check overflow
    mask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f32m2_b16(*x, overflowThreshold, vl),
        __riscv_vmflt_vf_f32m2_b16(*x, RVVMF_EXP_AS_FP32(pinf), vl), vl);
    *special = __riscv_vfmerge_vfm_f32m2(*special, RVVMF_EXP_AS_FP32(pinf), mask, vl);
    *specialMask = __riscv_vmor_mm_b16(*specialMask, mask, vl);  
    /* if (__riscv_vcpop_m_b32(mask, vl))
        volatile double exception = DBL_MAX*2.0; */  // FE_OVERFLOW
    
    // NaNs, overflow, -inf -- automatically
    *x = __riscv_vfmerge_vfm_f32m2(*x, RVVMF_EXP_AS_FP32(ZERO_F32), *specialMask, vl);
}
#endif

#if 8 == 4
forceinline void check_special_cases_f32m4(vfloat32m4_t* x, vfloat32m4_t* special, RVV_BOOL* specialMask,
    float overflowThreshold, size_t vl)
{
    uint32_t pinf = 0x7f800000;    
    RVV_BOOL mask;
    // check +inf
    *specialMask = __riscv_vmfeq_vf_f32m4_b8(*x, RVVMF_EXP_AS_FP32(pinf), vl);
    *special = __riscv_vfmerge_vfm_f32m4(*x, RVVMF_EXP_AS_FP32(pinf), *specialMask, vl);
    // check overflow
    mask = __riscv_vmand_mm_b8(__riscv_vmfgt_vf_f32m4_b8(*x, overflowThreshold, vl),
        __riscv_vmflt_vf_f32m4_b8(*x, RVVMF_EXP_AS_FP32(pinf), vl), vl);
    *special = __riscv_vfmerge_vfm_f32m4(*special, RVVMF_EXP_AS_FP32(pinf), mask, vl);
    *specialMask = __riscv_vmor_mm_b8(*specialMask, mask, vl);  
    /* if (__riscv_vcpop_m_b32(mask, vl))
        volatile double exception = DBL_MAX*2.0; */  // FE_OVERFLOW
    
    // NaNs, overflow, -inf -- automatically
    *x = __riscv_vfmerge_vfm_f32m4(*x, RVVMF_EXP_AS_FP32(ZERO_F32), *specialMask, vl);
}
#endif
#if 8 == 8
forceinline void check_special_cases_f32m8(vfloat32m8_t* x, vfloat32m8_t* special, RVV_BOOL* specialMask,
    float overflowThreshold, size_t vl)
{
    uint32_t pinf = 0x7f800000;    
    RVV_BOOL mask;
    // check +inf
    *specialMask = __riscv_vmfeq_vf_f32m8_b4(*x, RVVMF_EXP_AS_FP32(pinf), vl);
    *special = __riscv_vfmerge_vfm_f32m8(*x, RVVMF_EXP_AS_FP32(pinf), *specialMask, vl);
    // check overflow
    mask = __riscv_vmand_mm_b4(__riscv_vmfgt_vf_f32m8_b4(*x, overflowThreshold, vl),
        __riscv_vmflt_vf_f32m8_b4(*x, RVVMF_EXP_AS_FP32(pinf), vl), vl);
    *special = __riscv_vfmerge_vfm_f32m8(*special, RVVMF_EXP_AS_FP32(pinf), mask, vl);
    *specialMask = __riscv_vmor_mm_b4(*specialMask, mask, vl);  
    /* if (__riscv_vcpop_m_b32(mask, vl))
        volatile double exception = DBL_MAX*2.0; */  // FE_OVERFLOW
    
    // NaNs, overflow, -inf -- automatically
    *x = __riscv_vfmerge_vfm_f32m8(*x, RVVMF_EXP_AS_FP32(ZERO_F32), *specialMask, vl);
}
#endif


forceinline void do_exp_argument_reduction_hl_f32m8(vfloat32m8_t x,
    vfloat32m8_t* yh, vfloat32m8_t* yl, vuint32m8_t* ei, vuint32m8_t* fi, size_t vl)
{
    vfloat32m8_t vmagicConst1 = __riscv_vfmv_v_f_f32m8(RVVMF_EXP_AS_FP32(MAGIC_CONST_1_F32), vl);
    vfloat32m8_t h = __riscv_vfmadd_vf_f32m8(x, RVVMF_EXP_AS_FP32(INV_LOG2_2K_F32), vmagicConst1, vl);
    vuint32m8_t hi = __riscv_vand_vx_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(h), MASK_HI_BIT_F32, vl);
    *fi = __riscv_vand_vx_u32m8(hi, MASK_FI_BIT_F32, vl);
    *ei = __riscv_vsrl_vx_u32m8(hi, TABLE_SIZE_DEG_F32, vl);
    h = __riscv_vfsub_vv_f32m8(h, vmagicConst1, vl);
    fma12_vf_f32m8(h, RVVMF_EXP_AS_FP32(M_LOG2_2K_L_F32),
        __riscv_vfmadd_vf_f32m8(h, RVVMF_EXP_AS_FP32(M_LOG2_2K_H_F32), x, vl), yh, yl, vl);
    *yl = __riscv_vfmadd_vf_f32m8(h, RVVMF_EXP_AS_FP32(M_LOG2_2K_LL_F32), *yl, vl);
    fast_2_sum_vv_f32m8(*yh, *yl, yh, yl, vl);
}

forceinline void get_table_values_hl_f32m8(
    vuint32m8_t* index, vfloat32m8_t* th, vfloat32m8_t* tl, size_t vl)
{
    *index = __riscv_vmul_vx_u32m8(*index, (uint32_t)sizeof(float), vl);
    *th = __riscv_vloxei32_v_f32m8((float*)RVVMF_EXP_LOOK_UP_TABLE_HIGH_F32, *index, vl);
    *tl = __riscv_vloxei32_v_f32m8((float*)RVVMF_EXP_LOOK_UP_TABLE_LOW_F32, *index, vl);
}

forceinline void calculate_exp_polynom_hl_f32m8(vfloat32m8_t yh, vfloat32m8_t yl, vfloat32m8_t* ph, vfloat32m8_t* pl, size_t vl)
{
    vfloat32m8_t sqryh = __riscv_vfmul_vv_f32m8(yh, yh, vl);
    vfloat32m8_t r = calc_polynom_deg_2_f32m8(yh, RVVMF_EXP_AS_FP32(EXP_POL_COEFF_2_F32),
        RVVMF_EXP_AS_FP32(EXP_POL_COEFF_3_F32), RVVMF_EXP_AS_FP32(EXP_POL_COEFF_4_F32), vl);
    fma12_vv_f32m8(sqryh, r, yh, ph, pl, vl);
    *pl = __riscv_vfadd_vv_f32m8(*pl, yl, vl);
}

forceinline void update_exponent_f32m8(vuint32m8_t ei, vfloat32m8_t* res, size_t vl)
{
    *res = __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vadd_vv_u32m8(
        __riscv_vreinterpret_v_f32m8_u32m8(*res), __riscv_vsll_vx_u32m8(ei, (size_t)23, vl), vl));
}

forceinline void update_exponent_with_subnormal_f32m8(float subnormalThreshold, vfloat32m8_t x,
    vuint32m8_t ei, vfloat32m8_t* res, size_t vl)
{
#ifndef __FAST_MATH__
    uint32_t ninf = 0xff800000;
#if 8 == 1
    RVV_BOOL subnormalMask = __riscv_vmand_mm_b32(__riscv_vmfgt_vf_f32m1_b32(x, RVVMF_EXP_AS_FP32(ninf), vl),
        __riscv_vmflt_vf_f32m8_b32(x, subnormalThreshold, vl), vl);
#endif
#if 8 == 2
    RVV_BOOL subnormalMask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f32m2_b16(x, RVVMF_EXP_AS_FP32(ninf), vl),
        __riscv_vmflt_vf_f32m8_b16(x, subnormalThreshold, vl), vl);
#endif
#if 8 == 4
    RVV_BOOL subnormalMask = __riscv_vmand_mm_b8(__riscv_vmfgt_vf_f32m4_b8(x, RVVMF_EXP_AS_FP32(ninf), vl),
        __riscv_vmflt_vf_f32m4_b8(x, subnormalThreshold, vl), vl);
#endif
#if 8 == 8
    RVV_BOOL subnormalMask = __riscv_vmand_mm_b4(__riscv_vmfgt_vf_f32m8_b4(x, RVVMF_EXP_AS_FP32(ninf), vl),
        __riscv_vmflt_vf_f32m8_b4(x, subnormalThreshold, vl), vl);
#endif
    /* if (__riscv_vcpop_m_b32(subnormalMask, vl))
        volatile double exception = nextafter(DBL_MIN/(double((uint64_t)1 << 52)), 0.0) */ // FE_UNDERFLOW
    
    vfloat32m8_t subnormalRes;
    vuint32m8_t shiftNum = __riscv_vreinterpret_v_i32m8_u32m8(__riscv_vneg_v_i32m8(__riscv_vreinterpret_v_u32m8_i32m8(ei), vl));
    shiftNum = __riscv_vadd_vx_u32m8(__riscv_vand_vx_u32m8(shiftNum, (uint32_t)0x000001ff, vl), (uint32_t)1, vl);
    shiftNum = __riscv_vsll_vx_u32m8(shiftNum, (size_t)23, vl);
    subnormalRes = __riscv_vfadd_vv_f32m8(*res, __riscv_vreinterpret_v_u32m8_f32m8(shiftNum), vl);
    subnormalRes = __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vand_vx_u32m8(
        __riscv_vreinterpret_v_f32m8_u32m8(subnormalRes), (uint32_t)0x807fffff, vl));
#endif

    update_exponent_f32m8(ei, res, vl);
    
#ifndef __FAST_MATH__
    *res = __riscv_vmerge_vvm_f32m8(*res, subnormalRes, subnormalMask, vl);
#endif   
}

forceinline void reconstruct_exp_hl_hl_f32m8(vfloat32m8_t x, vuint32m8_t ei, vfloat32m8_t th, vfloat32m8_t tl,
    vfloat32m8_t pm8h, vfloat32m8_t pm8l, vfloat32m8_t* res, float subnormalThreshold, size_t vl)
{
    vfloat32m8_t sh, sl;
    fast_2_sum_fv_f32m8(RVVMF_EXP_AS_FP32(ONE_F32), pm8h, &sh, &sl, vl);
    sl = __riscv_vfadd_vv_f32m8(sl, pm8l, vl);
    mul21_vv_f32m8(th, tl, sh, sl, res, vl);
    update_exponent_with_subnormal_f32m8(subnormalThreshold, x, ei, res, vl);
}

forceinline void update_underflow_f32m8(vfloat32m8_t x, vfloat32m8_t* res,
    float underflowThreshold, float underflowValue, size_t vl)
{
#if 8 == 1
    RVV_BOOL underflowMask = __riscv_vmflt_vf_f32m8_b32(x, underflowThreshold, vl);
#endif
#if 8 == 2
    RVV_BOOL underflowMask = __riscv_vmflt_vf_f32m8_b16(x, underflowThreshold, vl);
#endif
#if 8 == 4
    RVV_BOOL underflowMask = __riscv_vmflt_vf_f32m8_b8(x, underflowThreshold, vl);
#endif
#if 8 == 8
    RVV_BOOL underflowMask = __riscv_vmflt_vf_f32m8_b4(x, underflowThreshold, vl);
#endif
    *res = __riscv_vfmerge_vfm_f32m8(*res, underflowValue, underflowMask, vl);
}

vfloat32m8_t __riscv_vexp_f32m8(vfloat32m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m8(avl);

    vfloat32m8_t res, yh, yl, th, tl, pm8h, pm8l;
    vuint32m8_t ei, fi;
    
#ifndef __FAST_MATH__
    float zeroThreshold = RVVMF_EXP_AS_FP32(EXP_ZERO_THRESHOLD_F32);
    vfloat32m8_t special;
    RVV_BOOL specialMask;
    check_special_cases_f32m8(&x, &special, &specialMask, RVVMF_EXP_AS_FP32(EXP_EXPM1_OVERFLOW_THRESHOLD_F32), vl);
#else
    float zeroThreshold = RVVMF_EXP_AS_FP32(EXP_SUBNORMAL_THRESHOLD_F32);    
#endif
    
    do_exp_argument_reduction_hl_f32m8(x, &yh, &yl, &ei, &fi, vl);
    get_table_values_hl_f32m8(&fi, &th, &tl, vl);
    calculate_exp_polynom_hl_f32m8(yh, yl, &pm8h, &pm8l, vl);
    reconstruct_exp_hl_hl_f32m8(x, ei, th, tl, pm8h, pm8l, &res, RVVMF_EXP_AS_FP32(EXP_SUBNORMAL_THRESHOLD_F32), vl);
    update_underflow_f32m8(x, &res, zeroThreshold, RVVMF_EXP_AS_FP32(EXP_UNDERFLOW_VALUE_F32), vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m8(res, special, specialMask, vl);
#endif

    return res;
}





void xnn_f32_vexp_ukernel__rvv_exp_u8v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t n = __riscv_vsetvl_e32m8(batch);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(input, n);
    input += n;
    vfloat32m8_t vacc = __riscv_vexp_f32m8(vx, n);
    __riscv_vse32_v_f32m8(output, vacc, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
