// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/prefetch.h"
#include "xnnpack/spmm.h"


void xnn_f32_spmm_minmax_ukernel_4x2__rvv(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  mc >>= XNN_LOG2_SIZEOF_FLOAT;
  const float min = params->scalar.min;
  const float max = params->scalar.max;
  size_t max_vector_length = __riscv_vsetvlmax_e32m1();
  size_t vector_length = max_vector_length < 4 ? max_vector_length : 4;
  size_t output_decrement = output_stride * nc - 4 * sizeof(float);
  while (mc >= 4) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 2) {
      const float *input_ptr;
      const float *w_ptr;
      const int32_t *dmap_ptr;
      for (int i = 0; i < 4; i += vector_length) {
        input_ptr = input + i;
        w_ptr = w;
        dmap_ptr = dmap;
        uint32_t nnz = *nnzmap;
        vfloat32m1_t acc0_f32v = __riscv_vfmv_v_f_f32m1(*w_ptr++, vector_length);
        vfloat32m1_t acc1_f32v = __riscv_vfmv_v_f_f32m1(*w_ptr++, vector_length);
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap_ptr++;
            vfloat32m1_t input_f32v = __riscv_vle32_v_f32m1(input_ptr, vector_length);
            acc0_f32v = __riscv_vfmacc_vf_f32m1(acc0_f32v, *w_ptr++, input_f32v, vector_length);
            acc1_f32v = __riscv_vfmacc_vf_f32m1(acc1_f32v, *w_ptr++, input_f32v, vector_length);
            input_ptr = (const float*) ((uintptr_t) input_ptr + (uintptr_t) diff);
          } while (--nnz != 0);
        }
        vfloat32m1_t out0_f32v = __riscv_vfmin_vf_f32m1(acc0_f32v, max, vector_length);
        out0_f32v = __riscv_vfmax_vf_f32m1(out0_f32v, min, vector_length);
        __riscv_vse32_v_f32m1(output + i, out0_f32v, vector_length);
        vfloat32m1_t out1_f32v = __riscv_vfmin_vf_f32m1(acc1_f32v, max, vector_length);
        out1_f32v = __riscv_vfmax_vf_f32m1(out1_f32v, min, vector_length);
        __riscv_vse32_v_f32m1((float*) ((uintptr_t) output + output_stride * 1) + i, out1_f32v, vector_length);
      }
      w = w_ptr;
      dmap = dmap_ptr;
      nnzmap++;
      input = input_ptr - (4 - vector_length);
      output = (float*) ((uintptr_t) output + output_stride * 2);
      n -= 2;
    }
    // fall back loop with nr = 1
    if XNN_UNLIKELY(n != 0) {
      do {
        const float *input_ptr;
        const float *w_ptr;
        const int32_t *dmap_ptr;
        for (int i = 0; i < 4; i += vector_length) {
          input_ptr = input + i;
          w_ptr = w;
          dmap_ptr = dmap;
          vfloat32m1_t acc_f32v = __riscv_vfmv_v_f_f32m1(*w_ptr++, vector_length);
          uint32_t nnz = *nnzmap;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap_ptr++;
              vfloat32m1_t input_f32v = __riscv_vle32_v_f32m1(input_ptr, vector_length);
              acc_f32v = __riscv_vfmacc_vf_f32m1(acc_f32v, *w_ptr++, input_f32v, vector_length);
              input_ptr = (const float*) ((uintptr_t) input_ptr + (uintptr_t) diff);
            } while (--nnz != 0);
          }
          vfloat32m1_t out_f32v = __riscv_vfmin_vf_f32m1(acc_f32v, max, vector_length);
          out_f32v = __riscv_vfmax_vf_f32m1(out_f32v, min, vector_length);
          __riscv_vse32_v_f32m1(output + i, out_f32v, vector_length);
        }
        w = w_ptr;
        dmap = dmap_ptr;
        nnzmap++;
        input = input_ptr - (4 - vector_length);
        output = (float*) ((uintptr_t) output + output_stride);
      } while(--n != 0);
    }
    output = (float*) ((uintptr_t) output - output_decrement);
    input += 4;
    mc -= 4;
  }
  if XNN_UNLIKELY(mc != 0) {
    vector_length = max_vector_length < 2 ? max_vector_length : 2;
    output_decrement += 2 * sizeof(float);
    if (mc & 2) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while(n >= 2) {
        const float *input_ptr;
        const float *w_ptr;
        const int32_t *dmap_ptr;
        for (int i = 0; i < 2; i += vector_length) {
          input_ptr = input + i;
          w_ptr = w;
          dmap_ptr = dmap;
          uint32_t nnz = *nnzmap;
          vfloat32m1_t acc0_f32v = __riscv_vfmv_v_f_f32m1(*w_ptr++, vector_length);
          vfloat32m1_t acc1_f32v = __riscv_vfmv_v_f_f32m1(*w_ptr++, vector_length);
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap_ptr++;
              vfloat32m1_t input_f32v = __riscv_vle32_v_f32m1(input_ptr, vector_length);
              acc0_f32v = __riscv_vfmacc_vf_f32m1(acc0_f32v, *w_ptr++, input_f32v, vector_length);
              acc1_f32v = __riscv_vfmacc_vf_f32m1(acc1_f32v, *w_ptr++, input_f32v, vector_length);
              input_ptr = (const float*) ((uintptr_t) input_ptr + (uintptr_t) diff);
            } while (--nnz != 0);
          }
          vfloat32m1_t out0_f32v = __riscv_vfmin_vf_f32m1(acc0_f32v, max, vector_length);
          out0_f32v = __riscv_vfmax_vf_f32m1(out0_f32v, min, vector_length);
          __riscv_vse32_v_f32m1(output + i, out0_f32v, vector_length);
          vfloat32m1_t out1_f32v = __riscv_vfmin_vf_f32m1(acc1_f32v, max, vector_length);
          out1_f32v = __riscv_vfmax_vf_f32m1(out1_f32v, min, vector_length);
          __riscv_vse32_v_f32m1((float*) ((uintptr_t) output + output_stride * 1) + i, out1_f32v, vector_length);
        }
        w = w_ptr;
        dmap = dmap_ptr;
        nnzmap++;
        input = input_ptr - (2 - vector_length);
        output = (float*) ((uintptr_t) output + output_stride * 2);
        n -= 2;
      }
      // fall back loop with nr = 1
      if XNN_UNLIKELY(n != 0) {
        do {
          const float *input_ptr;
          const float *w_ptr;
          const int32_t *dmap_ptr;
          for (int i = 0; i < 2; i += vector_length) {
            input_ptr = input + i;
            w_ptr = w;
            dmap_ptr = dmap;
            vfloat32m1_t acc_f32v = __riscv_vfmv_v_f_f32m1(*w_ptr++, vector_length);
            uint32_t nnz = *nnzmap;
            if XNN_LIKELY(nnz != 0) {
              do {
                const intptr_t diff = *dmap_ptr++;
                vfloat32m1_t input_f32v = __riscv_vle32_v_f32m1(input_ptr, vector_length);
                acc_f32v = __riscv_vfmacc_vf_f32m1(acc_f32v, *w_ptr++, input_f32v, vector_length);
                input_ptr = (const float*) ((uintptr_t) input_ptr + (uintptr_t) diff);
              } while (--nnz != 0);
            }
            vfloat32m1_t out_f32v = __riscv_vfmin_vf_f32m1(acc_f32v, max, vector_length);
            out_f32v = __riscv_vfmax_vf_f32m1(out_f32v, min, vector_length);
            __riscv_vse32_v_f32m1(output + i, out_f32v, vector_length);
          }
          w = w_ptr;
          dmap = dmap_ptr;
          nnzmap++;
          input = input_ptr - (2 - vector_length);
          output = (float*) ((uintptr_t) output + output_stride);
        } while(--n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 2;
      mc -= 2;
    }
    vector_length = max_vector_length < 1 ? max_vector_length : 1;
    output_decrement += 1 * sizeof(float);
    if (mc & 1) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while(n >= 2) {
        const float *input_ptr;
        const float *w_ptr;
        const int32_t *dmap_ptr;
        for (int i = 0; i < 1; i += vector_length) {
          input_ptr = input + i;
          w_ptr = w;
          dmap_ptr = dmap;
          uint32_t nnz = *nnzmap;
          vfloat32m1_t acc0_f32v = __riscv_vfmv_v_f_f32m1(*w_ptr++, vector_length);
          vfloat32m1_t acc1_f32v = __riscv_vfmv_v_f_f32m1(*w_ptr++, vector_length);
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap_ptr++;
              vfloat32m1_t input_f32v = __riscv_vle32_v_f32m1(input_ptr, vector_length);
              acc0_f32v = __riscv_vfmacc_vf_f32m1(acc0_f32v, *w_ptr++, input_f32v, vector_length);
              acc1_f32v = __riscv_vfmacc_vf_f32m1(acc1_f32v, *w_ptr++, input_f32v, vector_length);
              input_ptr = (const float*) ((uintptr_t) input_ptr + (uintptr_t) diff);
            } while (--nnz != 0);
          }
          vfloat32m1_t out0_f32v = __riscv_vfmin_vf_f32m1(acc0_f32v, max, vector_length);
          out0_f32v = __riscv_vfmax_vf_f32m1(out0_f32v, min, vector_length);
          __riscv_vse32_v_f32m1(output + i, out0_f32v, vector_length);
          vfloat32m1_t out1_f32v = __riscv_vfmin_vf_f32m1(acc1_f32v, max, vector_length);
          out1_f32v = __riscv_vfmax_vf_f32m1(out1_f32v, min, vector_length);
          __riscv_vse32_v_f32m1((float*) ((uintptr_t) output + output_stride * 1) + i, out1_f32v, vector_length);
        }
        w = w_ptr;
        dmap = dmap_ptr;
        nnzmap++;
        input = input_ptr - (1 - vector_length);
        output = (float*) ((uintptr_t) output + output_stride * 2);
        n -= 2;
      }
      // fall back loop with nr = 1
      if XNN_UNLIKELY(n != 0) {
        do {
          const float *input_ptr;
          const float *w_ptr;
          const int32_t *dmap_ptr;
          for (int i = 0; i < 1; i += vector_length) {
            input_ptr = input + i;
            w_ptr = w;
            dmap_ptr = dmap;
            vfloat32m1_t acc_f32v = __riscv_vfmv_v_f_f32m1(*w_ptr++, vector_length);
            uint32_t nnz = *nnzmap;
            if XNN_LIKELY(nnz != 0) {
              do {
                const intptr_t diff = *dmap_ptr++;
                vfloat32m1_t input_f32v = __riscv_vle32_v_f32m1(input_ptr, vector_length);
                acc_f32v = __riscv_vfmacc_vf_f32m1(acc_f32v, *w_ptr++, input_f32v, vector_length);
                input_ptr = (const float*) ((uintptr_t) input_ptr + (uintptr_t) diff);
              } while (--nnz != 0);
            }
            vfloat32m1_t out_f32v = __riscv_vfmin_vf_f32m1(acc_f32v, max, vector_length);
            out_f32v = __riscv_vfmax_vf_f32m1(out_f32v, min, vector_length);
            __riscv_vse32_v_f32m1(output + i, out_f32v, vector_length);
          }
          w = w_ptr;
          dmap = dmap_ptr;
          nnzmap++;
          input = input_ptr - (1 - vector_length);
          output = (float*) ((uintptr_t) output + output_stride);
        } while(--n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 1;
      mc -= 1;
    }
  }
}
