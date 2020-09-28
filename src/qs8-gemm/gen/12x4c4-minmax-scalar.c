// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRxNRc4-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/gemm.h>

#include <xnnpack/scalar-utils.h>

// This kernel is a scalar model for a kernel using ARMv8.2 dot-product
// instructions.
//
// XNN_DISABLE_TSAN is used because this kernel reads up to 3 bytes past the
// bounds of the `a` matrix region, which may be a race condition with
// another thread. We deem this acceptable because the values that are
// read out of bounds do not affect the result, and the the compiler can't know
// about this undefined behavior.
void xnn_qs8_gemm_minmax_ukernel_12x4c4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN {
  assert(mr != 0);
  assert(mr <= 12);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  int8_t* c6 = (int8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  int8_t* c7 = (int8_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const int8_t* a8 = (const int8_t*) ((uintptr_t) a7 + a_stride);
  int8_t* c8 = (int8_t*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }
  const int8_t* a9 = (const int8_t*) ((uintptr_t) a8 + a_stride);
  int8_t* c9 = (int8_t*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    a9 = a8;
    c9 = c8;
  }
  const int8_t* a10 = (const int8_t*) ((uintptr_t) a9 + a_stride);
  int8_t* c10 = (int8_t*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    a10 = a9;
    c10 = c9;
  }
  const int8_t* a11 = (const int8_t*) ((uintptr_t) a10 + a_stride);
  int8_t* c11 = (int8_t*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 12) {
    a11 = a10;
    c11 = c10;
  }

  // Loop over groups of 4 columns.
  do {
    // `vaccMN` is the accumulator at row `M`, column `N`.
    // Initialize accumulators with bias. 4 bias values are loaded from the
    // weight matrix, at the start of the group of 4 columns.
    int32_t bias0 = ((const int32_t*)w)[0];
    int32_t vacc00 = bias0;
    int32_t vacc10 = bias0;
    int32_t vacc20 = bias0;
    int32_t vacc30 = bias0;
    int32_t vacc40 = bias0;
    int32_t vacc50 = bias0;
    int32_t vacc60 = bias0;
    int32_t vacc70 = bias0;
    int32_t vacc80 = bias0;
    int32_t vacc90 = bias0;
    int32_t vacc100 = bias0;
    int32_t vacc110 = bias0;
    int32_t bias1 = ((const int32_t*)w)[1];
    int32_t vacc01 = bias1;
    int32_t vacc11 = bias1;
    int32_t vacc21 = bias1;
    int32_t vacc31 = bias1;
    int32_t vacc41 = bias1;
    int32_t vacc51 = bias1;
    int32_t vacc61 = bias1;
    int32_t vacc71 = bias1;
    int32_t vacc81 = bias1;
    int32_t vacc91 = bias1;
    int32_t vacc101 = bias1;
    int32_t vacc111 = bias1;
    int32_t bias2 = ((const int32_t*)w)[2];
    int32_t vacc02 = bias2;
    int32_t vacc12 = bias2;
    int32_t vacc22 = bias2;
    int32_t vacc32 = bias2;
    int32_t vacc42 = bias2;
    int32_t vacc52 = bias2;
    int32_t vacc62 = bias2;
    int32_t vacc72 = bias2;
    int32_t vacc82 = bias2;
    int32_t vacc92 = bias2;
    int32_t vacc102 = bias2;
    int32_t vacc112 = bias2;
    int32_t bias3 = ((const int32_t*)w)[3];
    int32_t vacc03 = bias3;
    int32_t vacc13 = bias3;
    int32_t vacc23 = bias3;
    int32_t vacc33 = bias3;
    int32_t vacc43 = bias3;
    int32_t vacc53 = bias3;
    int32_t vacc63 = bias3;
    int32_t vacc73 = bias3;
    int32_t vacc83 = bias3;
    int32_t vacc93 = bias3;
    int32_t vacc103 = bias3;
    int32_t vacc113 = bias3;

    w = (const void*)((uintptr_t)w + 4 * sizeof(int32_t));

    // Inner accumulation loop along the 4 columns.
    // Handle 4 rows at each iteration: this is key to modelling what an
    // actual kernel using ARMv8.2 dot-product instructions would look like.
    size_t k = 0;
    while (k < kc) {
      // Load a 12x4 block of activations.
      int32_t va00 = *a0++;
      int32_t va01 = *a0++;
      int32_t va02 = *a0++;
      int32_t va03 = *a0++;
      int32_t va10 = *a1++;
      int32_t va11 = *a1++;
      int32_t va12 = *a1++;
      int32_t va13 = *a1++;
      int32_t va20 = *a2++;
      int32_t va21 = *a2++;
      int32_t va22 = *a2++;
      int32_t va23 = *a2++;
      int32_t va30 = *a3++;
      int32_t va31 = *a3++;
      int32_t va32 = *a3++;
      int32_t va33 = *a3++;
      int32_t va40 = *a4++;
      int32_t va41 = *a4++;
      int32_t va42 = *a4++;
      int32_t va43 = *a4++;
      int32_t va50 = *a5++;
      int32_t va51 = *a5++;
      int32_t va52 = *a5++;
      int32_t va53 = *a5++;
      int32_t va60 = *a6++;
      int32_t va61 = *a6++;
      int32_t va62 = *a6++;
      int32_t va63 = *a6++;
      int32_t va70 = *a7++;
      int32_t va71 = *a7++;
      int32_t va72 = *a7++;
      int32_t va73 = *a7++;
      int32_t va80 = *a8++;
      int32_t va81 = *a8++;
      int32_t va82 = *a8++;
      int32_t va83 = *a8++;
      int32_t va90 = *a9++;
      int32_t va91 = *a9++;
      int32_t va92 = *a9++;
      int32_t va93 = *a9++;
      int32_t va100 = *a10++;
      int32_t va101 = *a10++;
      int32_t va102 = *a10++;
      int32_t va103 = *a10++;
      int32_t va110 = *a11++;
      int32_t va111 = *a11++;
      int32_t va112 = *a11++;
      int32_t va113 = *a11++;

      // Load a 4x4 block of weights.
      int32_t vb00 = ((const int8_t*)w)[0];
      int32_t vb10 = ((const int8_t*)w)[1];
      int32_t vb20 = ((const int8_t*)w)[2];
      int32_t vb30 = ((const int8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(int8_t));
      int32_t vb01 = ((const int8_t*)w)[0];
      int32_t vb11 = ((const int8_t*)w)[1];
      int32_t vb21 = ((const int8_t*)w)[2];
      int32_t vb31 = ((const int8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(int8_t));
      int32_t vb02 = ((const int8_t*)w)[0];
      int32_t vb12 = ((const int8_t*)w)[1];
      int32_t vb22 = ((const int8_t*)w)[2];
      int32_t vb32 = ((const int8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(int8_t));
      int32_t vb03 = ((const int8_t*)w)[0];
      int32_t vb13 = ((const int8_t*)w)[1];
      int32_t vb23 = ((const int8_t*)w)[2];
      int32_t vb33 = ((const int8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(int8_t));

      // Multiply-accumulate: 12x4 * 4x4 --> 12x4. The inner size 4 here means
      // we're computing 4D dot-products, which makes this a model for
      // a ARMv8.2 dot-product kernel.
      vacc00 += va00 * vb00;
      vacc00 += va01 * vb10;
      vacc00 += va02 * vb20;
      vacc00 += va03 * vb30;
      vacc01 += va00 * vb01;
      vacc01 += va01 * vb11;
      vacc01 += va02 * vb21;
      vacc01 += va03 * vb31;
      vacc02 += va00 * vb02;
      vacc02 += va01 * vb12;
      vacc02 += va02 * vb22;
      vacc02 += va03 * vb32;
      vacc03 += va00 * vb03;
      vacc03 += va01 * vb13;
      vacc03 += va02 * vb23;
      vacc03 += va03 * vb33;
      vacc10 += va10 * vb00;
      vacc10 += va11 * vb10;
      vacc10 += va12 * vb20;
      vacc10 += va13 * vb30;
      vacc11 += va10 * vb01;
      vacc11 += va11 * vb11;
      vacc11 += va12 * vb21;
      vacc11 += va13 * vb31;
      vacc12 += va10 * vb02;
      vacc12 += va11 * vb12;
      vacc12 += va12 * vb22;
      vacc12 += va13 * vb32;
      vacc13 += va10 * vb03;
      vacc13 += va11 * vb13;
      vacc13 += va12 * vb23;
      vacc13 += va13 * vb33;
      vacc20 += va20 * vb00;
      vacc20 += va21 * vb10;
      vacc20 += va22 * vb20;
      vacc20 += va23 * vb30;
      vacc21 += va20 * vb01;
      vacc21 += va21 * vb11;
      vacc21 += va22 * vb21;
      vacc21 += va23 * vb31;
      vacc22 += va20 * vb02;
      vacc22 += va21 * vb12;
      vacc22 += va22 * vb22;
      vacc22 += va23 * vb32;
      vacc23 += va20 * vb03;
      vacc23 += va21 * vb13;
      vacc23 += va22 * vb23;
      vacc23 += va23 * vb33;
      vacc30 += va30 * vb00;
      vacc30 += va31 * vb10;
      vacc30 += va32 * vb20;
      vacc30 += va33 * vb30;
      vacc31 += va30 * vb01;
      vacc31 += va31 * vb11;
      vacc31 += va32 * vb21;
      vacc31 += va33 * vb31;
      vacc32 += va30 * vb02;
      vacc32 += va31 * vb12;
      vacc32 += va32 * vb22;
      vacc32 += va33 * vb32;
      vacc33 += va30 * vb03;
      vacc33 += va31 * vb13;
      vacc33 += va32 * vb23;
      vacc33 += va33 * vb33;
      vacc40 += va40 * vb00;
      vacc40 += va41 * vb10;
      vacc40 += va42 * vb20;
      vacc40 += va43 * vb30;
      vacc41 += va40 * vb01;
      vacc41 += va41 * vb11;
      vacc41 += va42 * vb21;
      vacc41 += va43 * vb31;
      vacc42 += va40 * vb02;
      vacc42 += va41 * vb12;
      vacc42 += va42 * vb22;
      vacc42 += va43 * vb32;
      vacc43 += va40 * vb03;
      vacc43 += va41 * vb13;
      vacc43 += va42 * vb23;
      vacc43 += va43 * vb33;
      vacc50 += va50 * vb00;
      vacc50 += va51 * vb10;
      vacc50 += va52 * vb20;
      vacc50 += va53 * vb30;
      vacc51 += va50 * vb01;
      vacc51 += va51 * vb11;
      vacc51 += va52 * vb21;
      vacc51 += va53 * vb31;
      vacc52 += va50 * vb02;
      vacc52 += va51 * vb12;
      vacc52 += va52 * vb22;
      vacc52 += va53 * vb32;
      vacc53 += va50 * vb03;
      vacc53 += va51 * vb13;
      vacc53 += va52 * vb23;
      vacc53 += va53 * vb33;
      vacc60 += va60 * vb00;
      vacc60 += va61 * vb10;
      vacc60 += va62 * vb20;
      vacc60 += va63 * vb30;
      vacc61 += va60 * vb01;
      vacc61 += va61 * vb11;
      vacc61 += va62 * vb21;
      vacc61 += va63 * vb31;
      vacc62 += va60 * vb02;
      vacc62 += va61 * vb12;
      vacc62 += va62 * vb22;
      vacc62 += va63 * vb32;
      vacc63 += va60 * vb03;
      vacc63 += va61 * vb13;
      vacc63 += va62 * vb23;
      vacc63 += va63 * vb33;
      vacc70 += va70 * vb00;
      vacc70 += va71 * vb10;
      vacc70 += va72 * vb20;
      vacc70 += va73 * vb30;
      vacc71 += va70 * vb01;
      vacc71 += va71 * vb11;
      vacc71 += va72 * vb21;
      vacc71 += va73 * vb31;
      vacc72 += va70 * vb02;
      vacc72 += va71 * vb12;
      vacc72 += va72 * vb22;
      vacc72 += va73 * vb32;
      vacc73 += va70 * vb03;
      vacc73 += va71 * vb13;
      vacc73 += va72 * vb23;
      vacc73 += va73 * vb33;
      vacc80 += va80 * vb00;
      vacc80 += va81 * vb10;
      vacc80 += va82 * vb20;
      vacc80 += va83 * vb30;
      vacc81 += va80 * vb01;
      vacc81 += va81 * vb11;
      vacc81 += va82 * vb21;
      vacc81 += va83 * vb31;
      vacc82 += va80 * vb02;
      vacc82 += va81 * vb12;
      vacc82 += va82 * vb22;
      vacc82 += va83 * vb32;
      vacc83 += va80 * vb03;
      vacc83 += va81 * vb13;
      vacc83 += va82 * vb23;
      vacc83 += va83 * vb33;
      vacc90 += va90 * vb00;
      vacc90 += va91 * vb10;
      vacc90 += va92 * vb20;
      vacc90 += va93 * vb30;
      vacc91 += va90 * vb01;
      vacc91 += va91 * vb11;
      vacc91 += va92 * vb21;
      vacc91 += va93 * vb31;
      vacc92 += va90 * vb02;
      vacc92 += va91 * vb12;
      vacc92 += va92 * vb22;
      vacc92 += va93 * vb32;
      vacc93 += va90 * vb03;
      vacc93 += va91 * vb13;
      vacc93 += va92 * vb23;
      vacc93 += va93 * vb33;
      vacc100 += va100 * vb00;
      vacc100 += va101 * vb10;
      vacc100 += va102 * vb20;
      vacc100 += va103 * vb30;
      vacc101 += va100 * vb01;
      vacc101 += va101 * vb11;
      vacc101 += va102 * vb21;
      vacc101 += va103 * vb31;
      vacc102 += va100 * vb02;
      vacc102 += va101 * vb12;
      vacc102 += va102 * vb22;
      vacc102 += va103 * vb32;
      vacc103 += va100 * vb03;
      vacc103 += va101 * vb13;
      vacc103 += va102 * vb23;
      vacc103 += va103 * vb33;
      vacc110 += va110 * vb00;
      vacc110 += va111 * vb10;
      vacc110 += va112 * vb20;
      vacc110 += va113 * vb30;
      vacc111 += va110 * vb01;
      vacc111 += va111 * vb11;
      vacc111 += va112 * vb21;
      vacc111 += va113 * vb31;
      vacc112 += va110 * vb02;
      vacc112 += va111 * vb12;
      vacc112 += va112 * vb22;
      vacc112 += va113 * vb32;
      vacc113 += va110 * vb03;
      vacc113 += va111 * vb13;
      vacc113 += va112 * vb23;
      vacc113 += va113 * vb33;

      k += 4 * sizeof(int8_t);
    }
    // End of accumulation loop. The variable `k` contains the amount by which
    // we advanced the `va` pointers, so we rewind by this amount now.
    a0 = (const int8_t*)((uintptr_t)a0 - k);
    a1 = (const int8_t*)((uintptr_t)a1 - k);
    a2 = (const int8_t*)((uintptr_t)a2 - k);
    a3 = (const int8_t*)((uintptr_t)a3 - k);
    a4 = (const int8_t*)((uintptr_t)a4 - k);
    a5 = (const int8_t*)((uintptr_t)a5 - k);
    a6 = (const int8_t*)((uintptr_t)a6 - k);
    a7 = (const int8_t*)((uintptr_t)a7 - k);
    a8 = (const int8_t*)((uintptr_t)a8 - k);
    a9 = (const int8_t*)((uintptr_t)a9 - k);
    a10 = (const int8_t*)((uintptr_t)a10 - k);
    a11 = (const int8_t*)((uintptr_t)a11 - k);

    // Post-accumulation work

    const int32_t vmultiplier = params->scalar.multiplier;
    const int64_t vq31rounding = INT64_C(0x40000000);
    const int32_t vremainder_mask = params->scalar.remainder_mask;
    const uint32_t vshift = params->scalar.shift;
    const int32_t vremainder_threshold = params->scalar.remainder_threshold;
    const int32_t voutput_min = params->scalar.output_min_less_zero_point;
    const int32_t voutput_max = params->scalar.output_max_less_zero_point;
    const int32_t voutput_zero_point = params->scalar.output_zero_point;

    const int64_t vproduct00 = (int64_t)vacc00 * (int64_t)vmultiplier;
    const int64_t vproduct01 = (int64_t)vacc01 * (int64_t)vmultiplier;
    const int64_t vproduct02 = (int64_t)vacc02 * (int64_t)vmultiplier;
    const int64_t vproduct03 = (int64_t)vacc03 * (int64_t)vmultiplier;
    const int64_t vproduct10 = (int64_t)vacc10 * (int64_t)vmultiplier;
    const int64_t vproduct11 = (int64_t)vacc11 * (int64_t)vmultiplier;
    const int64_t vproduct12 = (int64_t)vacc12 * (int64_t)vmultiplier;
    const int64_t vproduct13 = (int64_t)vacc13 * (int64_t)vmultiplier;
    const int64_t vproduct20 = (int64_t)vacc20 * (int64_t)vmultiplier;
    const int64_t vproduct21 = (int64_t)vacc21 * (int64_t)vmultiplier;
    const int64_t vproduct22 = (int64_t)vacc22 * (int64_t)vmultiplier;
    const int64_t vproduct23 = (int64_t)vacc23 * (int64_t)vmultiplier;
    const int64_t vproduct30 = (int64_t)vacc30 * (int64_t)vmultiplier;
    const int64_t vproduct31 = (int64_t)vacc31 * (int64_t)vmultiplier;
    const int64_t vproduct32 = (int64_t)vacc32 * (int64_t)vmultiplier;
    const int64_t vproduct33 = (int64_t)vacc33 * (int64_t)vmultiplier;
    const int64_t vproduct40 = (int64_t)vacc40 * (int64_t)vmultiplier;
    const int64_t vproduct41 = (int64_t)vacc41 * (int64_t)vmultiplier;
    const int64_t vproduct42 = (int64_t)vacc42 * (int64_t)vmultiplier;
    const int64_t vproduct43 = (int64_t)vacc43 * (int64_t)vmultiplier;
    const int64_t vproduct50 = (int64_t)vacc50 * (int64_t)vmultiplier;
    const int64_t vproduct51 = (int64_t)vacc51 * (int64_t)vmultiplier;
    const int64_t vproduct52 = (int64_t)vacc52 * (int64_t)vmultiplier;
    const int64_t vproduct53 = (int64_t)vacc53 * (int64_t)vmultiplier;
    const int64_t vproduct60 = (int64_t)vacc60 * (int64_t)vmultiplier;
    const int64_t vproduct61 = (int64_t)vacc61 * (int64_t)vmultiplier;
    const int64_t vproduct62 = (int64_t)vacc62 * (int64_t)vmultiplier;
    const int64_t vproduct63 = (int64_t)vacc63 * (int64_t)vmultiplier;
    const int64_t vproduct70 = (int64_t)vacc70 * (int64_t)vmultiplier;
    const int64_t vproduct71 = (int64_t)vacc71 * (int64_t)vmultiplier;
    const int64_t vproduct72 = (int64_t)vacc72 * (int64_t)vmultiplier;
    const int64_t vproduct73 = (int64_t)vacc73 * (int64_t)vmultiplier;
    const int64_t vproduct80 = (int64_t)vacc80 * (int64_t)vmultiplier;
    const int64_t vproduct81 = (int64_t)vacc81 * (int64_t)vmultiplier;
    const int64_t vproduct82 = (int64_t)vacc82 * (int64_t)vmultiplier;
    const int64_t vproduct83 = (int64_t)vacc83 * (int64_t)vmultiplier;
    const int64_t vproduct90 = (int64_t)vacc90 * (int64_t)vmultiplier;
    const int64_t vproduct91 = (int64_t)vacc91 * (int64_t)vmultiplier;
    const int64_t vproduct92 = (int64_t)vacc92 * (int64_t)vmultiplier;
    const int64_t vproduct93 = (int64_t)vacc93 * (int64_t)vmultiplier;
    const int64_t vproduct100 = (int64_t)vacc100 * (int64_t)vmultiplier;
    const int64_t vproduct101 = (int64_t)vacc101 * (int64_t)vmultiplier;
    const int64_t vproduct102 = (int64_t)vacc102 * (int64_t)vmultiplier;
    const int64_t vproduct103 = (int64_t)vacc103 * (int64_t)vmultiplier;
    const int64_t vproduct110 = (int64_t)vacc110 * (int64_t)vmultiplier;
    const int64_t vproduct111 = (int64_t)vacc111 * (int64_t)vmultiplier;
    const int64_t vproduct112 = (int64_t)vacc112 * (int64_t)vmultiplier;
    const int64_t vproduct113 = (int64_t)vacc113 * (int64_t)vmultiplier;

    const int32_t vq31product00 = (int32_t)(uint32_t)((uint64_t)(vproduct00 + vq31rounding) >> 31);
    const int32_t vq31product01 = (int32_t)(uint32_t)((uint64_t)(vproduct01 + vq31rounding) >> 31);
    const int32_t vq31product02 = (int32_t)(uint32_t)((uint64_t)(vproduct02 + vq31rounding) >> 31);
    const int32_t vq31product03 = (int32_t)(uint32_t)((uint64_t)(vproduct03 + vq31rounding) >> 31);
    const int32_t vq31product10 = (int32_t)(uint32_t)((uint64_t)(vproduct10 + vq31rounding) >> 31);
    const int32_t vq31product11 = (int32_t)(uint32_t)((uint64_t)(vproduct11 + vq31rounding) >> 31);
    const int32_t vq31product12 = (int32_t)(uint32_t)((uint64_t)(vproduct12 + vq31rounding) >> 31);
    const int32_t vq31product13 = (int32_t)(uint32_t)((uint64_t)(vproduct13 + vq31rounding) >> 31);
    const int32_t vq31product20 = (int32_t)(uint32_t)((uint64_t)(vproduct20 + vq31rounding) >> 31);
    const int32_t vq31product21 = (int32_t)(uint32_t)((uint64_t)(vproduct21 + vq31rounding) >> 31);
    const int32_t vq31product22 = (int32_t)(uint32_t)((uint64_t)(vproduct22 + vq31rounding) >> 31);
    const int32_t vq31product23 = (int32_t)(uint32_t)((uint64_t)(vproduct23 + vq31rounding) >> 31);
    const int32_t vq31product30 = (int32_t)(uint32_t)((uint64_t)(vproduct30 + vq31rounding) >> 31);
    const int32_t vq31product31 = (int32_t)(uint32_t)((uint64_t)(vproduct31 + vq31rounding) >> 31);
    const int32_t vq31product32 = (int32_t)(uint32_t)((uint64_t)(vproduct32 + vq31rounding) >> 31);
    const int32_t vq31product33 = (int32_t)(uint32_t)((uint64_t)(vproduct33 + vq31rounding) >> 31);
    const int32_t vq31product40 = (int32_t)(uint32_t)((uint64_t)(vproduct40 + vq31rounding) >> 31);
    const int32_t vq31product41 = (int32_t)(uint32_t)((uint64_t)(vproduct41 + vq31rounding) >> 31);
    const int32_t vq31product42 = (int32_t)(uint32_t)((uint64_t)(vproduct42 + vq31rounding) >> 31);
    const int32_t vq31product43 = (int32_t)(uint32_t)((uint64_t)(vproduct43 + vq31rounding) >> 31);
    const int32_t vq31product50 = (int32_t)(uint32_t)((uint64_t)(vproduct50 + vq31rounding) >> 31);
    const int32_t vq31product51 = (int32_t)(uint32_t)((uint64_t)(vproduct51 + vq31rounding) >> 31);
    const int32_t vq31product52 = (int32_t)(uint32_t)((uint64_t)(vproduct52 + vq31rounding) >> 31);
    const int32_t vq31product53 = (int32_t)(uint32_t)((uint64_t)(vproduct53 + vq31rounding) >> 31);
    const int32_t vq31product60 = (int32_t)(uint32_t)((uint64_t)(vproduct60 + vq31rounding) >> 31);
    const int32_t vq31product61 = (int32_t)(uint32_t)((uint64_t)(vproduct61 + vq31rounding) >> 31);
    const int32_t vq31product62 = (int32_t)(uint32_t)((uint64_t)(vproduct62 + vq31rounding) >> 31);
    const int32_t vq31product63 = (int32_t)(uint32_t)((uint64_t)(vproduct63 + vq31rounding) >> 31);
    const int32_t vq31product70 = (int32_t)(uint32_t)((uint64_t)(vproduct70 + vq31rounding) >> 31);
    const int32_t vq31product71 = (int32_t)(uint32_t)((uint64_t)(vproduct71 + vq31rounding) >> 31);
    const int32_t vq31product72 = (int32_t)(uint32_t)((uint64_t)(vproduct72 + vq31rounding) >> 31);
    const int32_t vq31product73 = (int32_t)(uint32_t)((uint64_t)(vproduct73 + vq31rounding) >> 31);
    const int32_t vq31product80 = (int32_t)(uint32_t)((uint64_t)(vproduct80 + vq31rounding) >> 31);
    const int32_t vq31product81 = (int32_t)(uint32_t)((uint64_t)(vproduct81 + vq31rounding) >> 31);
    const int32_t vq31product82 = (int32_t)(uint32_t)((uint64_t)(vproduct82 + vq31rounding) >> 31);
    const int32_t vq31product83 = (int32_t)(uint32_t)((uint64_t)(vproduct83 + vq31rounding) >> 31);
    const int32_t vq31product90 = (int32_t)(uint32_t)((uint64_t)(vproduct90 + vq31rounding) >> 31);
    const int32_t vq31product91 = (int32_t)(uint32_t)((uint64_t)(vproduct91 + vq31rounding) >> 31);
    const int32_t vq31product92 = (int32_t)(uint32_t)((uint64_t)(vproduct92 + vq31rounding) >> 31);
    const int32_t vq31product93 = (int32_t)(uint32_t)((uint64_t)(vproduct93 + vq31rounding) >> 31);
    const int32_t vq31product100 = (int32_t)(uint32_t)((uint64_t)(vproduct100 + vq31rounding) >> 31);
    const int32_t vq31product101 = (int32_t)(uint32_t)((uint64_t)(vproduct101 + vq31rounding) >> 31);
    const int32_t vq31product102 = (int32_t)(uint32_t)((uint64_t)(vproduct102 + vq31rounding) >> 31);
    const int32_t vq31product103 = (int32_t)(uint32_t)((uint64_t)(vproduct103 + vq31rounding) >> 31);
    const int32_t vq31product110 = (int32_t)(uint32_t)((uint64_t)(vproduct110 + vq31rounding) >> 31);
    const int32_t vq31product111 = (int32_t)(uint32_t)((uint64_t)(vproduct111 + vq31rounding) >> 31);
    const int32_t vq31product112 = (int32_t)(uint32_t)((uint64_t)(vproduct112 + vq31rounding) >> 31);
    const int32_t vq31product113 = (int32_t)(uint32_t)((uint64_t)(vproduct113 + vq31rounding) >> 31);

    const int32_t vremainder00 = (vq31product00 & vremainder_mask) - (int32_t)(vq31product00 < 0);
    const int32_t vremainder01 = (vq31product01 & vremainder_mask) - (int32_t)(vq31product01 < 0);
    const int32_t vremainder02 = (vq31product02 & vremainder_mask) - (int32_t)(vq31product02 < 0);
    const int32_t vremainder03 = (vq31product03 & vremainder_mask) - (int32_t)(vq31product03 < 0);
    const int32_t vremainder10 = (vq31product10 & vremainder_mask) - (int32_t)(vq31product10 < 0);
    const int32_t vremainder11 = (vq31product11 & vremainder_mask) - (int32_t)(vq31product11 < 0);
    const int32_t vremainder12 = (vq31product12 & vremainder_mask) - (int32_t)(vq31product12 < 0);
    const int32_t vremainder13 = (vq31product13 & vremainder_mask) - (int32_t)(vq31product13 < 0);
    const int32_t vremainder20 = (vq31product20 & vremainder_mask) - (int32_t)(vq31product20 < 0);
    const int32_t vremainder21 = (vq31product21 & vremainder_mask) - (int32_t)(vq31product21 < 0);
    const int32_t vremainder22 = (vq31product22 & vremainder_mask) - (int32_t)(vq31product22 < 0);
    const int32_t vremainder23 = (vq31product23 & vremainder_mask) - (int32_t)(vq31product23 < 0);
    const int32_t vremainder30 = (vq31product30 & vremainder_mask) - (int32_t)(vq31product30 < 0);
    const int32_t vremainder31 = (vq31product31 & vremainder_mask) - (int32_t)(vq31product31 < 0);
    const int32_t vremainder32 = (vq31product32 & vremainder_mask) - (int32_t)(vq31product32 < 0);
    const int32_t vremainder33 = (vq31product33 & vremainder_mask) - (int32_t)(vq31product33 < 0);
    const int32_t vremainder40 = (vq31product40 & vremainder_mask) - (int32_t)(vq31product40 < 0);
    const int32_t vremainder41 = (vq31product41 & vremainder_mask) - (int32_t)(vq31product41 < 0);
    const int32_t vremainder42 = (vq31product42 & vremainder_mask) - (int32_t)(vq31product42 < 0);
    const int32_t vremainder43 = (vq31product43 & vremainder_mask) - (int32_t)(vq31product43 < 0);
    const int32_t vremainder50 = (vq31product50 & vremainder_mask) - (int32_t)(vq31product50 < 0);
    const int32_t vremainder51 = (vq31product51 & vremainder_mask) - (int32_t)(vq31product51 < 0);
    const int32_t vremainder52 = (vq31product52 & vremainder_mask) - (int32_t)(vq31product52 < 0);
    const int32_t vremainder53 = (vq31product53 & vremainder_mask) - (int32_t)(vq31product53 < 0);
    const int32_t vremainder60 = (vq31product60 & vremainder_mask) - (int32_t)(vq31product60 < 0);
    const int32_t vremainder61 = (vq31product61 & vremainder_mask) - (int32_t)(vq31product61 < 0);
    const int32_t vremainder62 = (vq31product62 & vremainder_mask) - (int32_t)(vq31product62 < 0);
    const int32_t vremainder63 = (vq31product63 & vremainder_mask) - (int32_t)(vq31product63 < 0);
    const int32_t vremainder70 = (vq31product70 & vremainder_mask) - (int32_t)(vq31product70 < 0);
    const int32_t vremainder71 = (vq31product71 & vremainder_mask) - (int32_t)(vq31product71 < 0);
    const int32_t vremainder72 = (vq31product72 & vremainder_mask) - (int32_t)(vq31product72 < 0);
    const int32_t vremainder73 = (vq31product73 & vremainder_mask) - (int32_t)(vq31product73 < 0);
    const int32_t vremainder80 = (vq31product80 & vremainder_mask) - (int32_t)(vq31product80 < 0);
    const int32_t vremainder81 = (vq31product81 & vremainder_mask) - (int32_t)(vq31product81 < 0);
    const int32_t vremainder82 = (vq31product82 & vremainder_mask) - (int32_t)(vq31product82 < 0);
    const int32_t vremainder83 = (vq31product83 & vremainder_mask) - (int32_t)(vq31product83 < 0);
    const int32_t vremainder90 = (vq31product90 & vremainder_mask) - (int32_t)(vq31product90 < 0);
    const int32_t vremainder91 = (vq31product91 & vremainder_mask) - (int32_t)(vq31product91 < 0);
    const int32_t vremainder92 = (vq31product92 & vremainder_mask) - (int32_t)(vq31product92 < 0);
    const int32_t vremainder93 = (vq31product93 & vremainder_mask) - (int32_t)(vq31product93 < 0);
    const int32_t vremainder100 = (vq31product100 & vremainder_mask) - (int32_t)(vq31product100 < 0);
    const int32_t vremainder101 = (vq31product101 & vremainder_mask) - (int32_t)(vq31product101 < 0);
    const int32_t vremainder102 = (vq31product102 & vremainder_mask) - (int32_t)(vq31product102 < 0);
    const int32_t vremainder103 = (vq31product103 & vremainder_mask) - (int32_t)(vq31product103 < 0);
    const int32_t vremainder110 = (vq31product110 & vremainder_mask) - (int32_t)(vq31product110 < 0);
    const int32_t vremainder111 = (vq31product111 & vremainder_mask) - (int32_t)(vq31product111 < 0);
    const int32_t vremainder112 = (vq31product112 & vremainder_mask) - (int32_t)(vq31product112 < 0);
    const int32_t vremainder113 = (vq31product113 & vremainder_mask) - (int32_t)(vq31product113 < 0);

    int32_t vout00 = asr_s32(vq31product00, vshift) + (int32_t)(vremainder00 > vremainder_threshold);
    int32_t vout01 = asr_s32(vq31product01, vshift) + (int32_t)(vremainder01 > vremainder_threshold);
    int32_t vout02 = asr_s32(vq31product02, vshift) + (int32_t)(vremainder02 > vremainder_threshold);
    int32_t vout03 = asr_s32(vq31product03, vshift) + (int32_t)(vremainder03 > vremainder_threshold);
    int32_t vout10 = asr_s32(vq31product10, vshift) + (int32_t)(vremainder10 > vremainder_threshold);
    int32_t vout11 = asr_s32(vq31product11, vshift) + (int32_t)(vremainder11 > vremainder_threshold);
    int32_t vout12 = asr_s32(vq31product12, vshift) + (int32_t)(vremainder12 > vremainder_threshold);
    int32_t vout13 = asr_s32(vq31product13, vshift) + (int32_t)(vremainder13 > vremainder_threshold);
    int32_t vout20 = asr_s32(vq31product20, vshift) + (int32_t)(vremainder20 > vremainder_threshold);
    int32_t vout21 = asr_s32(vq31product21, vshift) + (int32_t)(vremainder21 > vremainder_threshold);
    int32_t vout22 = asr_s32(vq31product22, vshift) + (int32_t)(vremainder22 > vremainder_threshold);
    int32_t vout23 = asr_s32(vq31product23, vshift) + (int32_t)(vremainder23 > vremainder_threshold);
    int32_t vout30 = asr_s32(vq31product30, vshift) + (int32_t)(vremainder30 > vremainder_threshold);
    int32_t vout31 = asr_s32(vq31product31, vshift) + (int32_t)(vremainder31 > vremainder_threshold);
    int32_t vout32 = asr_s32(vq31product32, vshift) + (int32_t)(vremainder32 > vremainder_threshold);
    int32_t vout33 = asr_s32(vq31product33, vshift) + (int32_t)(vremainder33 > vremainder_threshold);
    int32_t vout40 = asr_s32(vq31product40, vshift) + (int32_t)(vremainder40 > vremainder_threshold);
    int32_t vout41 = asr_s32(vq31product41, vshift) + (int32_t)(vremainder41 > vremainder_threshold);
    int32_t vout42 = asr_s32(vq31product42, vshift) + (int32_t)(vremainder42 > vremainder_threshold);
    int32_t vout43 = asr_s32(vq31product43, vshift) + (int32_t)(vremainder43 > vremainder_threshold);
    int32_t vout50 = asr_s32(vq31product50, vshift) + (int32_t)(vremainder50 > vremainder_threshold);
    int32_t vout51 = asr_s32(vq31product51, vshift) + (int32_t)(vremainder51 > vremainder_threshold);
    int32_t vout52 = asr_s32(vq31product52, vshift) + (int32_t)(vremainder52 > vremainder_threshold);
    int32_t vout53 = asr_s32(vq31product53, vshift) + (int32_t)(vremainder53 > vremainder_threshold);
    int32_t vout60 = asr_s32(vq31product60, vshift) + (int32_t)(vremainder60 > vremainder_threshold);
    int32_t vout61 = asr_s32(vq31product61, vshift) + (int32_t)(vremainder61 > vremainder_threshold);
    int32_t vout62 = asr_s32(vq31product62, vshift) + (int32_t)(vremainder62 > vremainder_threshold);
    int32_t vout63 = asr_s32(vq31product63, vshift) + (int32_t)(vremainder63 > vremainder_threshold);
    int32_t vout70 = asr_s32(vq31product70, vshift) + (int32_t)(vremainder70 > vremainder_threshold);
    int32_t vout71 = asr_s32(vq31product71, vshift) + (int32_t)(vremainder71 > vremainder_threshold);
    int32_t vout72 = asr_s32(vq31product72, vshift) + (int32_t)(vremainder72 > vremainder_threshold);
    int32_t vout73 = asr_s32(vq31product73, vshift) + (int32_t)(vremainder73 > vremainder_threshold);
    int32_t vout80 = asr_s32(vq31product80, vshift) + (int32_t)(vremainder80 > vremainder_threshold);
    int32_t vout81 = asr_s32(vq31product81, vshift) + (int32_t)(vremainder81 > vremainder_threshold);
    int32_t vout82 = asr_s32(vq31product82, vshift) + (int32_t)(vremainder82 > vremainder_threshold);
    int32_t vout83 = asr_s32(vq31product83, vshift) + (int32_t)(vremainder83 > vremainder_threshold);
    int32_t vout90 = asr_s32(vq31product90, vshift) + (int32_t)(vremainder90 > vremainder_threshold);
    int32_t vout91 = asr_s32(vq31product91, vshift) + (int32_t)(vremainder91 > vremainder_threshold);
    int32_t vout92 = asr_s32(vq31product92, vshift) + (int32_t)(vremainder92 > vremainder_threshold);
    int32_t vout93 = asr_s32(vq31product93, vshift) + (int32_t)(vremainder93 > vremainder_threshold);
    int32_t vout100 = asr_s32(vq31product100, vshift) + (int32_t)(vremainder100 > vremainder_threshold);
    int32_t vout101 = asr_s32(vq31product101, vshift) + (int32_t)(vremainder101 > vremainder_threshold);
    int32_t vout102 = asr_s32(vq31product102, vshift) + (int32_t)(vremainder102 > vremainder_threshold);
    int32_t vout103 = asr_s32(vq31product103, vshift) + (int32_t)(vremainder103 > vremainder_threshold);
    int32_t vout110 = asr_s32(vq31product110, vshift) + (int32_t)(vremainder110 > vremainder_threshold);
    int32_t vout111 = asr_s32(vq31product111, vshift) + (int32_t)(vremainder111 > vremainder_threshold);
    int32_t vout112 = asr_s32(vq31product112, vshift) + (int32_t)(vremainder112 > vremainder_threshold);
    int32_t vout113 = asr_s32(vq31product113, vshift) + (int32_t)(vremainder113 > vremainder_threshold);

    vout00 = vout00 < voutput_min ? voutput_min : vout00;
    vout01 = vout01 < voutput_min ? voutput_min : vout01;
    vout02 = vout02 < voutput_min ? voutput_min : vout02;
    vout03 = vout03 < voutput_min ? voutput_min : vout03;
    vout10 = vout10 < voutput_min ? voutput_min : vout10;
    vout11 = vout11 < voutput_min ? voutput_min : vout11;
    vout12 = vout12 < voutput_min ? voutput_min : vout12;
    vout13 = vout13 < voutput_min ? voutput_min : vout13;
    vout20 = vout20 < voutput_min ? voutput_min : vout20;
    vout21 = vout21 < voutput_min ? voutput_min : vout21;
    vout22 = vout22 < voutput_min ? voutput_min : vout22;
    vout23 = vout23 < voutput_min ? voutput_min : vout23;
    vout30 = vout30 < voutput_min ? voutput_min : vout30;
    vout31 = vout31 < voutput_min ? voutput_min : vout31;
    vout32 = vout32 < voutput_min ? voutput_min : vout32;
    vout33 = vout33 < voutput_min ? voutput_min : vout33;
    vout40 = vout40 < voutput_min ? voutput_min : vout40;
    vout41 = vout41 < voutput_min ? voutput_min : vout41;
    vout42 = vout42 < voutput_min ? voutput_min : vout42;
    vout43 = vout43 < voutput_min ? voutput_min : vout43;
    vout50 = vout50 < voutput_min ? voutput_min : vout50;
    vout51 = vout51 < voutput_min ? voutput_min : vout51;
    vout52 = vout52 < voutput_min ? voutput_min : vout52;
    vout53 = vout53 < voutput_min ? voutput_min : vout53;
    vout60 = vout60 < voutput_min ? voutput_min : vout60;
    vout61 = vout61 < voutput_min ? voutput_min : vout61;
    vout62 = vout62 < voutput_min ? voutput_min : vout62;
    vout63 = vout63 < voutput_min ? voutput_min : vout63;
    vout70 = vout70 < voutput_min ? voutput_min : vout70;
    vout71 = vout71 < voutput_min ? voutput_min : vout71;
    vout72 = vout72 < voutput_min ? voutput_min : vout72;
    vout73 = vout73 < voutput_min ? voutput_min : vout73;
    vout80 = vout80 < voutput_min ? voutput_min : vout80;
    vout81 = vout81 < voutput_min ? voutput_min : vout81;
    vout82 = vout82 < voutput_min ? voutput_min : vout82;
    vout83 = vout83 < voutput_min ? voutput_min : vout83;
    vout90 = vout90 < voutput_min ? voutput_min : vout90;
    vout91 = vout91 < voutput_min ? voutput_min : vout91;
    vout92 = vout92 < voutput_min ? voutput_min : vout92;
    vout93 = vout93 < voutput_min ? voutput_min : vout93;
    vout100 = vout100 < voutput_min ? voutput_min : vout100;
    vout101 = vout101 < voutput_min ? voutput_min : vout101;
    vout102 = vout102 < voutput_min ? voutput_min : vout102;
    vout103 = vout103 < voutput_min ? voutput_min : vout103;
    vout110 = vout110 < voutput_min ? voutput_min : vout110;
    vout111 = vout111 < voutput_min ? voutput_min : vout111;
    vout112 = vout112 < voutput_min ? voutput_min : vout112;
    vout113 = vout113 < voutput_min ? voutput_min : vout113;

    vout00 = vout00 > voutput_max ? voutput_max : vout00;
    vout01 = vout01 > voutput_max ? voutput_max : vout01;
    vout02 = vout02 > voutput_max ? voutput_max : vout02;
    vout03 = vout03 > voutput_max ? voutput_max : vout03;
    vout10 = vout10 > voutput_max ? voutput_max : vout10;
    vout11 = vout11 > voutput_max ? voutput_max : vout11;
    vout12 = vout12 > voutput_max ? voutput_max : vout12;
    vout13 = vout13 > voutput_max ? voutput_max : vout13;
    vout20 = vout20 > voutput_max ? voutput_max : vout20;
    vout21 = vout21 > voutput_max ? voutput_max : vout21;
    vout22 = vout22 > voutput_max ? voutput_max : vout22;
    vout23 = vout23 > voutput_max ? voutput_max : vout23;
    vout30 = vout30 > voutput_max ? voutput_max : vout30;
    vout31 = vout31 > voutput_max ? voutput_max : vout31;
    vout32 = vout32 > voutput_max ? voutput_max : vout32;
    vout33 = vout33 > voutput_max ? voutput_max : vout33;
    vout40 = vout40 > voutput_max ? voutput_max : vout40;
    vout41 = vout41 > voutput_max ? voutput_max : vout41;
    vout42 = vout42 > voutput_max ? voutput_max : vout42;
    vout43 = vout43 > voutput_max ? voutput_max : vout43;
    vout50 = vout50 > voutput_max ? voutput_max : vout50;
    vout51 = vout51 > voutput_max ? voutput_max : vout51;
    vout52 = vout52 > voutput_max ? voutput_max : vout52;
    vout53 = vout53 > voutput_max ? voutput_max : vout53;
    vout60 = vout60 > voutput_max ? voutput_max : vout60;
    vout61 = vout61 > voutput_max ? voutput_max : vout61;
    vout62 = vout62 > voutput_max ? voutput_max : vout62;
    vout63 = vout63 > voutput_max ? voutput_max : vout63;
    vout70 = vout70 > voutput_max ? voutput_max : vout70;
    vout71 = vout71 > voutput_max ? voutput_max : vout71;
    vout72 = vout72 > voutput_max ? voutput_max : vout72;
    vout73 = vout73 > voutput_max ? voutput_max : vout73;
    vout80 = vout80 > voutput_max ? voutput_max : vout80;
    vout81 = vout81 > voutput_max ? voutput_max : vout81;
    vout82 = vout82 > voutput_max ? voutput_max : vout82;
    vout83 = vout83 > voutput_max ? voutput_max : vout83;
    vout90 = vout90 > voutput_max ? voutput_max : vout90;
    vout91 = vout91 > voutput_max ? voutput_max : vout91;
    vout92 = vout92 > voutput_max ? voutput_max : vout92;
    vout93 = vout93 > voutput_max ? voutput_max : vout93;
    vout100 = vout100 > voutput_max ? voutput_max : vout100;
    vout101 = vout101 > voutput_max ? voutput_max : vout101;
    vout102 = vout102 > voutput_max ? voutput_max : vout102;
    vout103 = vout103 > voutput_max ? voutput_max : vout103;
    vout110 = vout110 > voutput_max ? voutput_max : vout110;
    vout111 = vout111 > voutput_max ? voutput_max : vout111;
    vout112 = vout112 > voutput_max ? voutput_max : vout112;
    vout113 = vout113 > voutput_max ? voutput_max : vout113;

    vout00 += voutput_zero_point;
    vout01 += voutput_zero_point;
    vout02 += voutput_zero_point;
    vout03 += voutput_zero_point;
    vout10 += voutput_zero_point;
    vout11 += voutput_zero_point;
    vout12 += voutput_zero_point;
    vout13 += voutput_zero_point;
    vout20 += voutput_zero_point;
    vout21 += voutput_zero_point;
    vout22 += voutput_zero_point;
    vout23 += voutput_zero_point;
    vout30 += voutput_zero_point;
    vout31 += voutput_zero_point;
    vout32 += voutput_zero_point;
    vout33 += voutput_zero_point;
    vout40 += voutput_zero_point;
    vout41 += voutput_zero_point;
    vout42 += voutput_zero_point;
    vout43 += voutput_zero_point;
    vout50 += voutput_zero_point;
    vout51 += voutput_zero_point;
    vout52 += voutput_zero_point;
    vout53 += voutput_zero_point;
    vout60 += voutput_zero_point;
    vout61 += voutput_zero_point;
    vout62 += voutput_zero_point;
    vout63 += voutput_zero_point;
    vout70 += voutput_zero_point;
    vout71 += voutput_zero_point;
    vout72 += voutput_zero_point;
    vout73 += voutput_zero_point;
    vout80 += voutput_zero_point;
    vout81 += voutput_zero_point;
    vout82 += voutput_zero_point;
    vout83 += voutput_zero_point;
    vout90 += voutput_zero_point;
    vout91 += voutput_zero_point;
    vout92 += voutput_zero_point;
    vout93 += voutput_zero_point;
    vout100 += voutput_zero_point;
    vout101 += voutput_zero_point;
    vout102 += voutput_zero_point;
    vout103 += voutput_zero_point;
    vout110 += voutput_zero_point;
    vout111 += voutput_zero_point;
    vout112 += voutput_zero_point;
    vout113 += voutput_zero_point;

    if XNN_LIKELY (nc >= 4) {
      // Main case where there the 4 columns fit in the destination.
      c0[0] = (int8_t) vout00;
      c0[1] = (int8_t) vout01;
      c0[2] = (int8_t) vout02;
      c0[3] = (int8_t) vout03;
      c1[0] = (int8_t) vout10;
      c1[1] = (int8_t) vout11;
      c1[2] = (int8_t) vout12;
      c1[3] = (int8_t) vout13;
      c2[0] = (int8_t) vout20;
      c2[1] = (int8_t) vout21;
      c2[2] = (int8_t) vout22;
      c2[3] = (int8_t) vout23;
      c3[0] = (int8_t) vout30;
      c3[1] = (int8_t) vout31;
      c3[2] = (int8_t) vout32;
      c3[3] = (int8_t) vout33;
      c4[0] = (int8_t) vout40;
      c4[1] = (int8_t) vout41;
      c4[2] = (int8_t) vout42;
      c4[3] = (int8_t) vout43;
      c5[0] = (int8_t) vout50;
      c5[1] = (int8_t) vout51;
      c5[2] = (int8_t) vout52;
      c5[3] = (int8_t) vout53;
      c6[0] = (int8_t) vout60;
      c6[1] = (int8_t) vout61;
      c6[2] = (int8_t) vout62;
      c6[3] = (int8_t) vout63;
      c7[0] = (int8_t) vout70;
      c7[1] = (int8_t) vout71;
      c7[2] = (int8_t) vout72;
      c7[3] = (int8_t) vout73;
      c8[0] = (int8_t) vout80;
      c8[1] = (int8_t) vout81;
      c8[2] = (int8_t) vout82;
      c8[3] = (int8_t) vout83;
      c9[0] = (int8_t) vout90;
      c9[1] = (int8_t) vout91;
      c9[2] = (int8_t) vout92;
      c9[3] = (int8_t) vout93;
      c10[0] = (int8_t) vout100;
      c10[1] = (int8_t) vout101;
      c10[2] = (int8_t) vout102;
      c10[3] = (int8_t) vout103;
      c11[0] = (int8_t) vout110;
      c11[1] = (int8_t) vout111;
      c11[2] = (int8_t) vout112;
      c11[3] = (int8_t) vout113;

      // Advance to the next 4 columns.
      c0 = (int8_t*)((uintptr_t)c0 + cn_stride);
      c1 = (int8_t*)((uintptr_t)c1 + cn_stride);
      c2 = (int8_t*)((uintptr_t)c2 + cn_stride);
      c3 = (int8_t*)((uintptr_t)c3 + cn_stride);
      c4 = (int8_t*)((uintptr_t)c4 + cn_stride);
      c5 = (int8_t*)((uintptr_t)c5 + cn_stride);
      c6 = (int8_t*)((uintptr_t)c6 + cn_stride);
      c7 = (int8_t*)((uintptr_t)c7 + cn_stride);
      c8 = (int8_t*)((uintptr_t)c8 + cn_stride);
      c9 = (int8_t*)((uintptr_t)c9 + cn_stride);
      c10 = (int8_t*)((uintptr_t)c10 + cn_stride);
      c11 = (int8_t*)((uintptr_t)c11 + cn_stride);

      nc -= 4;
    } else {
      // Final case where not all of the 4 columns fit in the destination.
      if (nc > 0) {
        c0[0] = vout00;
        c1[0] = vout10;
        c2[0] = vout20;
        c3[0] = vout30;
        c4[0] = vout40;
        c5[0] = vout50;
        c6[0] = vout60;
        c7[0] = vout70;
        c8[0] = vout80;
        c9[0] = vout90;
        c10[0] = vout100;
        c11[0] = vout110;
      }
      if (nc > 1) {
        c0[1] = vout01;
        c1[1] = vout11;
        c2[1] = vout21;
        c3[1] = vout31;
        c4[1] = vout41;
        c5[1] = vout51;
        c6[1] = vout61;
        c7[1] = vout71;
        c8[1] = vout81;
        c9[1] = vout91;
        c10[1] = vout101;
        c11[1] = vout111;
      }
      if (nc > 2) {
        c0[2] = vout02;
        c1[2] = vout12;
        c2[2] = vout22;
        c3[2] = vout32;
        c4[2] = vout42;
        c5[2] = vout52;
        c6[2] = vout62;
        c7[2] = vout72;
        c8[2] = vout82;
        c9[2] = vout92;
        c10[2] = vout102;
        c11[2] = vout112;
      }
      if (nc > 3) {
        c0[3] = vout03;
        c1[3] = vout13;
        c2[3] = vout23;
        c3[3] = vout33;
        c4[3] = vout43;
        c5[3] = vout53;
        c6[3] = vout63;
        c7[3] = vout73;
        c8[3] = vout83;
        c9[3] = vout93;
        c10[3] = vout103;
        c11[3] = vout113;
      }

      nc = 0;
    }
  } while (nc != 0);
}
