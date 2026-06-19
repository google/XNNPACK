// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>
#include <stdio.h>

#ifdef MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>

static void unpoison_gemm_output(
    size_t mr,
    size_t nr,
    void* c,
    size_t cm_stride,
    size_t sizeof_c) {
  for (size_t i = 0; i < mr; ++i) {
    __msan_unpoison((void*) ((uintptr_t) c + i * cm_stride), nr * sizeof_c);
  }
}

// For kernels implemented in assembly, these functions are "shadow
// implementations" that can be tail called to unpoison any outputs of the
// kernel.
// We could also insert checks that the inputs are initialized here...

// Since we don't know the datatype of the output from the function signature,
// we have a different version of this for each size of an element of c.

// These get used even for `xnn_dqgemm_ukernel_fn` functions, but since this
// argument gets passed on the stack, we can ignore it.
void xnn_gemm_ukernel_msan_sizeof_c_1(
    size_t mr,
    size_t nr,
    size_t k,
    const void* a,
    size_t a_stride,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const void* params
    /*const struct xnn_qd8_quantization_params* quantization_params*/) {
  unpoison_gemm_output(mr, nr, c, cm_stride, 1);
}

void xnn_gemm_ukernel_msan_sizeof_c_4(
    size_t mr,
    size_t nr,
    size_t k,
    const void* a,
    size_t a_stride,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const void* params
    /*const struct xnn_qd8_quantization_params* quantization_params*/) {
  unpoison_gemm_output(mr, nr, c, cm_stride, 4);
}

#endif  // MEMORY_SANITIZER