// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_gemm_minmax_ukernel_8x16c4__aarch64_neondot
        stp     d7, d6, [sp, -160]!
        stp     d5, d4, [sp, 16]
        stp     d11, d10, [sp, 32]
        stp     d9, d8, [sp, 48]
        stp     x28, x27, [sp, 64]
        stp     x26, x25, [sp, 80]
        stp     x24, x23, [sp, 96]
        stp     x22, x21, [sp, 112]
        stp     x20, x19, [sp, 128]
        stp     x29, x30, [sp, 144]
        sub     sp, sp, 0x300
        add     x9, x3, x4
        add     x8, x6, x7
        cmp     x0, 2
        mov     x21, x2
        sub     x2, x2, 8
        csel    x8, x6, x8, lo
        csel    x25, x3, x9, lo
        cmp     x0, 3
        lsl     x9, x2, 4
        add     x10, x25, x4
        add     x11, x8, x7
        and     x14, x9, 0xffffffffffffff80
        csel    x9, x8, x11, lo
        csel    x17, x25, x10, lo
        cmp     x0, 4
        add     x11, x17, x4
        add     x10, x9, x7
        csel    x10, x9, x10, lo
        csel    x26, x17, x11, lo
        cmp     x0, 5
        add     x12, x26, x4
        add     x11, x10, x7
        csel    x11, x10, x11, lo
        csel    x19, x26, x12, lo
        cmp     x0, 6
        add     x13, x19, x4
        add     x12, x11, x7
        csel    x12, x11, x12, lo
        csel    x27, x19, x13, lo
        ldr     x15, [sp, 936]
        ldr     x16, [sp, 928]
        cmp     x0, 7
        add     x20, x27, x4
        add     x13, x12, x7
        csel    x13, x12, x13, lo
        csel    x20, x27, x20, lo
        add     x14, x14, 0x80
        and     x22, x2, 0xfffffffffffffff8
        cmp     x0, 8
        str     x14, [sp, 8]
        add     x4, x20, x4
        add     x14, x13, x7
        add     x0, x22, 8
        csel    x14, x14, x13, eq
        csel    x28, x4, x20, eq
        sub     x4, x2, x22
1:
        ldp     q19, q17, [x5]
        ldp     q9, q22, [x5, 32]
        cmp     x21, 8
        add     x5, x5, 64
        str     q22, [sp, 576]
        b.cc    3f
        mov     x29, xzr
        add     x7, x25, x0
        add     x22, x26, x0
        add     x23, x27, x0
        add     x24, x28, x0
        add     x30, x5, 64
        stp     q9, q19, [sp, 416]
        mov     v16.16b, v19.16b
        mov     v15.16b, v17.16b
        str     q9, [sp, 464]
        stp     q9, q22, [sp, 384]
        mov     v14.16b, v19.16b
        mov     v6.16b, v17.16b
        stp     q9, q22, [sp, 672]
        mov     v3.16b, v19.16b
        mov     v5.16b, v17.16b
        stp     q9, q22, [sp, 624]
        mov     v10.16b, v19.16b
        mov     v2.16b, v17.16b
        stp     q22, q9, [sp, 352]
        mov     v20.16b, v19.16b
        mov     v1.16b, v17.16b
        mov     v0.16b, v9.16b
        mov     v9.16b, v19.16b
        mov     v24.16b, v22.16b
        mov     x2, x21
        stp     q17, q22, [sp, 480]
        str     q22, [sp, 656]
        stp     q17, q19, [sp, 736]
        mov     v19.16b, v0.16b
        str     q0, [sp, 448]
        str     q0, [sp, 704]
2:
        str     q24, [sp, 720]
        ldr     d24, [x3, x29]
        ldp     q18, q21, [x30, -64]
        ldr     q30, [sp, 752]
        ldr     d31, [x19, x29]
        ldr     d0, [x27, x29]
        ldr     d29, [x20, x29]
        ldr     d4, [x28, x29]
        sdot    v30.4s, v18.16b, v24.4b[0]
        str     q30, [sp, 752]
        ldr     q30, [sp, 736]
        sdot    v3.4s, v18.16b, v31.4b[0]
        sdot    v10.4s, v18.16b, v0.4b[0]
        sdot    v1.4s, v21.16b, v29.4b[0]
        sdot    v9.4s, v18.16b, v4.4b[0]
        ldr     d25, [x25, x29]
        ldr     d26, [x17, x29]
        ldr     d27, [x26, x29]
        ldp     q23, q22, [x30, -32]
        stp     q1, q10, [sp, 592]
        stp     q3, q9, [sp, 512]
        ldp     q9, q10, [sp, 624]
        sdot    v30.4s, v21.16b, v24.4b[0]
        str     q30, [sp, 736]
        sdot    v20.4s, v18.16b, v29.4b[0]
        mov     v30.16b, v29.16b
        stp     q29, q4, [sp, 288]
        sdot    v9.4s, v23.16b, v29.4b[0]
        ldr     q29, [sp, 656]
        sdot    v14.4s, v18.16b, v27.4b[0]
        sdot    v5.4s, v21.16b, v31.4b[0]
        sdot    v2.4s, v21.16b, v0.4b[0]
        mov     v12.16b, v6.16b
        stp     q2, q14, [sp, 320]
        stp     q5, q20, [sp, 544]
        ldp     q19, q14, [sp, 448]
        ldp     q13, q6, [sp, 400]
        ldp     q2, q3, [sp, 672]
        ldp     q20, q1, [sp, 368]
        sdot    v29.4s, v22.16b, v30.4b[0]
        ldr     q30, [sp, 704]
        sdot    v19.4s, v23.16b, v24.4b[0]
        sdot    v6.4s, v23.16b, v25.4b[0]
        sdot    v14.4s, v23.16b, v26.4b[0]
        sdot    v2.4s, v23.16b, v27.4b[0]
        sdot    v1.4s, v23.16b, v31.4b[0]
        sdot    v20.4s, v23.16b, v0.4b[0]
        sdot    v30.4s, v23.16b, v4.4b[0]
        ldr     q23, [sp, 720]
        ldr     q28, [sp, 576]
        ldr     q5, [sp, 496]
        ldr     q8, [sp, 352]
        ldr     q7, [sp, 432]
        ldr     q11, [sp, 480]
        sdot    v23.4s, v22.16b, v4.4b[0]
        sdot    v28.4s, v22.16b, v24.4b[0]
        sdot    v5.4s, v22.16b, v25.4b[0]
        sdot    v13.4s, v22.16b, v26.4b[0]
        sdot    v3.4s, v22.16b, v27.4b[0]
        sdot    v10.4s, v22.16b, v31.4b[0]
        sdot    v8.4s, v22.16b, v0.4b[0]
        str     q23, [sp, 720]
        ldp     q23, q22, [x30, 32]
        sdot    v7.4s, v18.16b, v25.4b[0]
        sdot    v11.4s, v21.16b, v25.4b[0]
        sdot    v16.4s, v18.16b, v26.4b[0]
        sdot    v15.4s, v21.16b, v26.4b[0]
        sdot    v12.4s, v21.16b, v27.4b[0]
        sdot    v17.4s, v21.16b, v4.4b[0]
        ldp     q21, q18, [x30]
        ldr     q4, [sp, 752]
        sdot    v28.4s, v22.16b, v24.4b[1]
        str     q28, [sp, 576]
        sdot    v1.4s, v23.16b, v31.4b[1]
        sdot    v4.4s, v21.16b, v24.4b[1]
        sdot    v10.4s, v22.16b, v31.4b[1]
        sdot    v20.4s, v23.16b, v0.4b[1]
        ldr     q28, [sp, 288]
        str     q4, [sp, 752]
        ldr     q4, [sp, 736]
        str     q1, [sp, 384]
        str     q10, [sp, 640]
        ldp     q1, q10, [sp, 592]
        str     q20, [sp, 368]
        ldr     q20, [sp, 560]
        sdot    v11.4s, v18.16b, v25.4b[1]
        sdot    v5.4s, v22.16b, v25.4b[1]
        sdot    v14.4s, v23.16b, v26.4b[1]
        sdot    v2.4s, v23.16b, v27.4b[1]
        sdot    v3.4s, v22.16b, v27.4b[1]
        sdot    v9.4s, v23.16b, v28.4b[1]
        sdot    v4.4s, v18.16b, v24.4b[1]
        sdot    v19.4s, v23.16b, v24.4b[1]
        ldr     q24, [sp, 720]
        stp     q11, q5, [sp, 480]
        ldr     q5, [sp, 544]
        str     q14, [sp, 464]
        ldr     q14, [sp, 336]
        stp     q2, q3, [sp, 672]
        ldr     q3, [sp, 512]
        ldr     q2, [sp, 320]
        sdot    v20.4s, v21.16b, v28.4b[1]
        sdot    v1.4s, v18.16b, v28.4b[1]
        str     q9, [sp, 624]
        ldr     q9, [sp, 528]
        sdot    v29.4s, v22.16b, v28.4b[1]
        ldr     q28, [sp, 304]
        sdot    v6.4s, v23.16b, v25.4b[1]
        sdot    v13.4s, v22.16b, v26.4b[1]
        sdot    v12.4s, v18.16b, v27.4b[1]
        sub     x2, x2, 8
        sdot    v7.4s, v21.16b, v25.4b[1]
        sdot    v16.4s, v21.16b, v26.4b[1]
        sdot    v15.4s, v18.16b, v26.4b[1]
        stp     q13, q6, [sp, 400]
        sdot    v14.4s, v21.16b, v27.4b[1]
        mov     v6.16b, v12.16b
        sdot    v3.4s, v21.16b, v31.4b[1]
        sdot    v5.4s, v18.16b, v31.4b[1]
        sdot    v10.4s, v21.16b, v0.4b[1]
        sdot    v2.4s, v18.16b, v0.4b[1]
        sdot    v8.4s, v22.16b, v0.4b[1]
        sdot    v9.4s, v21.16b, v28.4b[1]
        sdot    v17.4s, v18.16b, v28.4b[1]
        sdot    v30.4s, v23.16b, v28.4b[1]
        sdot    v24.4s, v22.16b, v28.4b[1]
        add     x29, x29, 8
        cmp     x2, 7
        add     x30, x30, 0x80
        str     q4, [sp, 736]
        stp     q7, q19, [sp, 432]
        str     q8, [sp, 352]
        str     q29, [sp, 656]
        str     q30, [sp, 704]
        b.hi    2b
        ldr     x2, [sp, 8]
        mov     v29.16b, v17.16b
        mov     v30.16b, v9.16b
        stp     q1, q10, [sp, 592]
        ldp     q12, q21, [sp, 352]
        ldp     q9, q26, [sp, 448]
        ldp     q17, q19, [sp, 736]
        ldp     q10, q18, [sp, 384]
        mov     v23.16b, v5.16b
        ldp     q11, q5, [sp, 672]
        mov     v27.16b, v14.16b
        mov     v14.16b, v15.16b
        ldp     q7, q22, [sp, 480]
        ldp     q4, q15, [sp, 416]
        add     x3, x3, x0
        add     x17, x17, x0
        add     x19, x19, x0
        add     x20, x20, x0
        add     x5, x5, x2
        mov     x29, x4
        mov     v8.16b, v2.16b
        mov     v1.16b, v3.16b
        mov     v2.16b, v6.16b
        str     q20, [sp, 560]
        b       4f
3:
        mov     x29, x21
        mov     v24.16b, v22.16b
        str     q9, [sp, 704]
        mov     v29.16b, v17.16b
        mov     v30.16b, v19.16b
        stp     q22, q22, [sp, 640]
        stp     q19, q9, [sp, 608]
        str     q17, [sp, 592]
        str     q19, [sp, 560]
        mov     v12.16b, v22.16b
        mov     v21.16b, v9.16b
        mov     v8.16b, v17.16b
        mov     v10.16b, v9.16b
        mov     v23.16b, v17.16b
        mov     v1.16b, v19.16b
        mov     v5.16b, v22.16b
        mov     v11.16b, v9.16b
        mov     v2.16b, v17.16b
        mov     v27.16b, v19.16b
        mov     v18.16b, v22.16b
        mov     v26.16b, v9.16b
        mov     v14.16b, v17.16b
        mov     v16.16b, v19.16b
        mov     v4.16b, v9.16b
        mov     v7.16b, v17.16b
        mov     v15.16b, v19.16b
        mov     x24, x28
        mov     x23, x27
        mov     x22, x26
        mov     x7, x25
4:
        str     q29, [sp, 736]
        str     q30, [sp, 528]
        cbnz    x29, 6f
5:
        str     q1, [sp, 512]
        mov     x2, x15
        ld1r    {v30.4s}, [x2], 4
        ldr     q0, [sp, 576]
        str     q23, [sp, 544]
        stp     q11, q5, [sp, 672]
        sqrdmulh        v28.4s, v19.4s, v30.4s
        str     q28, [sp, 720]
        sqrdmulh        v28.4s, v17.4s, v30.4s
        str     q28, [sp, 496]
        sqrdmulh        v28.4s, v9.4s, v30.4s
        str     q28, [sp, 480]
        sqrdmulh        v28.4s, v0.4s, v30.4s
        str     q28, [sp, 464]
        sqrdmulh        v28.4s, v15.4s, v30.4s
        str     q28, [sp, 448]
        sqrdmulh        v28.4s, v7.4s, v30.4s
        str     q28, [sp, 432]
        sqrdmulh        v28.4s, v4.4s, v30.4s
        str     q28, [sp, 416]
        sqrdmulh        v28.4s, v22.4s, v30.4s
        str     q28, [sp, 400]
        sqrdmulh        v28.4s, v16.4s, v30.4s
        str     q28, [sp, 384]
        sqrdmulh        v28.4s, v14.4s, v30.4s
        str     q28, [sp, 368]
        sqrdmulh        v28.4s, v26.4s, v30.4s
        str     q28, [sp, 352]
        sqrdmulh        v28.4s, v18.4s, v30.4s
        str     q28, [sp, 336]
        sqrdmulh        v28.4s, v27.4s, v30.4s
        str     q28, [sp, 320]
        sqrdmulh        v28.4s, v2.4s, v30.4s
        str     q28, [sp, 304]
        sqrdmulh        v28.4s, v11.4s, v30.4s
        str     q28, [sp, 272]
        sqrdmulh        v28.4s, v5.4s, v30.4s
        str     q28, [sp, 288]
        sqrdmulh        v28.4s, v1.4s, v30.4s
        str     q28, [sp, 256]
        sqrdmulh        v28.4s, v23.4s, v30.4s
        mov     v23.16b, v15.16b
        ldp     q1, q15, [sp, 624]
        ldr     q11, [sp, 608]
        str     q28, [sp, 240]
        sqrdmulh        v28.4s, v10.4s, v30.4s
        str     q28, [sp, 224]
        sqrdmulh        v28.4s, v15.4s, v30.4s
        mov     v5.16b, v27.16b
        ldr     q27, [sp, 560]
        str     q28, [sp, 208]
        sqrdmulh        v28.4s, v11.4s, v30.4s
        str     q28, [sp, 192]
        sqrdmulh        v28.4s, v8.4s, v30.4s
        str     q28, [sp, 176]
        sqrdmulh        v28.4s, v21.4s, v30.4s
        str     q28, [sp, 160]
        sqrdmulh        v28.4s, v12.4s, v30.4s
        sqrdmulh        v29.4s, v27.4s, v30.4s
        stp     q29, q28, [sp, 128]
        ldr     q29, [sp, 592]
        ldr     q6, [sp, 656]
        mov     v3.16b, v19.16b
        mov     v20.16b, v7.16b
        sqrdmulh        v19.4s, v29.4s, v30.4s
        str     q19, [sp, 96]
        sqrdmulh        v19.4s, v1.4s, v30.4s
        str     q19, [sp, 112]
        sqrdmulh        v19.4s, v6.4s, v30.4s
        str     q19, [sp, 80]
        mov     v19.16b, v4.16b
        mov     v4.16b, v14.16b
        mov     v14.16b, v22.16b
        ldr     q22, [sp, 736]
        ldr     q7, [sp, 528]
        ldr     q28, [sp, 704]
        ld1r    {v25.4s}, [x2]
        sqrdmulh        v31.4s, v22.4s, v30.4s
        mov     v13.16b, v17.16b
        mov     v17.16b, v26.16b
        sqrdmulh        v26.4s, v7.4s, v30.4s
        str     q31, [sp, 752]
        sqrdmulh        v31.4s, v28.4s, v30.4s
        sqrdmulh        v30.4s, v24.4s, v30.4s
        str     q30, [sp, 48]
        cmeq    v30.4s, v25.4s, 0
        str     q26, [sp, 64]
        mov     v26.16b, v20.16b
        mvn     v30.16b, v30.16b
        mov     v28.16b, v27.16b
        and     v27.16b, v26.16b, v30.16b
        mov     v26.16b, v2.16b
        and     v2.16b, v4.16b, v30.16b
        and     v4.16b, v26.16b, v30.16b
        ldr     q26, [sp, 672]
        and     v20.16b, v3.16b, v30.16b
        and     v3.16b, v9.16b, v30.16b
        and     v9.16b, v5.16b, v30.16b
        and     v5.16b, v26.16b, v30.16b
        ldr     q26, [sp, 688]
        and     v28.16b, v28.16b, v30.16b
        str     q31, [sp, 16]
        mov     v31.16b, v24.16b
        and     v26.16b, v26.16b, v30.16b
        str     q26, [sp, 688]
        ldr     q26, [sp, 512]
        mov     v24.16b, v1.16b
        str     q28, [sp, 560]
        and     v28.16b, v29.16b, v30.16b
        and     v26.16b, v26.16b, v30.16b
        str     q26, [sp, 672]
        ldr     q26, [sp, 544]
        str     q25, [sp, 32]
        mov     v1.16b, v16.16b
        and     v16.16b, v14.16b, v30.16b
        and     v26.16b, v26.16b, v30.16b
        str     q26, [sp, 576]
        and     v26.16b, v21.16b, v30.16b
        str     q26, [sp, 640]
        and     v26.16b, v12.16b, v30.16b
        stp     q28, q26, [sp, 592]
        and     v28.16b, v24.16b, v30.16b
        str     q28, [sp, 624]
        and     v28.16b, v6.16b, v30.16b
        str     q28, [sp, 656]
        and     v28.16b, v22.16b, v30.16b
        str     q28, [sp, 736]
        ldp     q28, q24, [sp, 704]
        and     v14.16b, v17.16b, v30.16b
        ldp     q25, q26, [sp, 320]
        and     v0.16b, v0.16b, v30.16b
        and     v28.16b, v28.16b, v30.16b
        str     q28, [sp, 704]
        ssra    v24.4s, v20.4s, 31
        ldp     q17, q20, [sp, 448]
        ldp     q29, q28, [sp, 384]
        and     v23.16b, v23.16b, v30.16b
        and     v18.16b, v18.16b, v30.16b
        ldp     q21, q22, [sp, 480]
        ssra    v20.4s, v0.4s, 31
        ssra    v17.4s, v23.4s, 31
        ssra    v28.4s, v16.4s, 31
        ssra    v26.4s, v18.4s, 31
        ldp     q18, q23, [sp, 288]
        ldp     q0, q16, [sp, 672]
        and     v13.16b, v13.16b, v30.16b
        ssra    v22.4s, v13.4s, 31
        ldp     q13, q12, [sp, 416]
        ssra    v18.4s, v16.4s, 31
        ldr     q16, [sp, 256]
        and     v1.16b, v1.16b, v30.16b
        and     v19.16b, v19.16b, v30.16b
        and     v10.16b, v10.16b, v30.16b
        and     v15.16b, v15.16b, v30.16b
        and     v11.16b, v11.16b, v30.16b
        and     v8.16b, v8.16b, v30.16b
        and     v7.16b, v7.16b, v30.16b
        and     v31.16b, v31.16b, v30.16b
        ssra    v12.4s, v27.4s, 31
        ssra    v29.4s, v1.4s, 31
        ldp     q27, q30, [sp, 352]
        ssra    v16.4s, v0.4s, 31
        ldr     q0, [sp, 240]
        ldr     q1, [sp, 576]
        ssra    v21.4s, v3.4s, 31
        ssra    v30.4s, v2.4s, 31
        ssra    v27.4s, v14.4s, 31
        ssra    v0.4s, v1.4s, 31
        ldp     q2, q1, [sp, 208]
        ldp     q14, q3, [sp, 176]
        ssra    v23.4s, v4.4s, 31
        ssra    v25.4s, v9.4s, 31
        ssra    v2.4s, v15.4s, 31
        ssra    v14.4s, v8.4s, 31
        ldr     q15, [sp, 160]
        ldp     q4, q8, [sp, 624]
        ssra    v3.4s, v11.4s, 31
        ldp     q11, q9, [sp, 592]
        ssra    v1.4s, v10.4s, 31
        ssra    v15.4s, v8.4s, 31
        ldr     q8, [sp, 144]
        ldr     q10, [sp, 560]
        ssra    v13.4s, v19.4s, 31
        ldr     q19, [sp, 272]
        ssra    v8.4s, v9.4s, 31
        ldr     q9, [sp, 128]
        add     x2, x15, 8
        ssra    v19.4s, v5.4s, 31
        ldr     q5, [sp, 656]
        ssra    v9.4s, v10.4s, 31
        ldr     q10, [sp, 96]
        cmp     x1, 0xf
        ssra    v10.4s, v11.4s, 31
        ldr     q11, [sp, 112]
        ssra    v11.4s, v4.4s, 31
        ldr     q4, [sp, 80]
        ssra    v4.4s, v5.4s, 31
        ldr     q5, [sp, 64]
        ssra    v5.4s, v7.4s, 31
        ldp     q6, q7, [sp, 736]
        ssra    v7.4s, v6.4s, 31
        str     q7, [sp, 752]
        ldr     q7, [sp, 16]
        ldr     q6, [sp, 704]
        ssra    v7.4s, v6.4s, 31
        ldr     q6, [sp, 48]
        ssra    v6.4s, v31.4s, 31
        ldr     q31, [sp, 32]
        srshl   v28.4s, v28.4s, v31.4s
        str     q28, [sp, 576]
        srshl   v28.4s, v18.4s, v31.4s
        str     q28, [sp, 592]
        srshl   v28.4s, v16.4s, v31.4s
        str     q28, [sp, 608]
        srshl   v28.4s, v0.4s, v31.4s
        str     q28, [sp, 624]
        srshl   v28.4s, v1.4s, v31.4s
        str     q28, [sp, 640]
        srshl   v28.4s, v2.4s, v31.4s
        str     q28, [sp, 656]
        srshl   v28.4s, v3.4s, v31.4s
        str     q28, [sp, 672]
        srshl   v28.4s, v9.4s, v31.4s
        str     q28, [sp, 688]
        srshl   v28.4s, v10.4s, v31.4s
        str     q28, [sp, 704]
        srshl   v28.4s, v11.4s, v31.4s
        str     q28, [sp, 720]
        srshl   v28.4s, v4.4s, v31.4s
        str     q28, [sp, 736]
        ldr     q28, [sp, 752]
        srshl   v24.4s, v24.4s, v31.4s
        srshl   v21.4s, v21.4s, v31.4s
        srshl   v17.4s, v17.4s, v31.4s
        srshl   v22.4s, v22.4s, v31.4s
        srshl   v20.4s, v20.4s, v31.4s
        srshl   v12.4s, v12.4s, v31.4s
        srshl   v13.4s, v13.4s, v31.4s
        srshl   v29.4s, v29.4s, v31.4s
        srshl   v30.4s, v30.4s, v31.4s
        srshl   v27.4s, v27.4s, v31.4s
        srshl   v26.4s, v26.4s, v31.4s
        srshl   v25.4s, v25.4s, v31.4s
        srshl   v23.4s, v23.4s, v31.4s
        srshl   v19.4s, v19.4s, v31.4s
        srshl   v14.4s, v14.4s, v31.4s
        srshl   v15.4s, v15.4s, v31.4s
        srshl   v8.4s, v8.4s, v31.4s
        srshl   v4.4s, v5.4s, v31.4s
        srshl   v11.4s, v28.4s, v31.4s
        srshl   v10.4s, v7.4s, v31.4s
        srshl   v9.4s, v6.4s, v31.4s
        sqxtn   v31.4h, v24.4s
        sqxtn   v24.4h, v21.4s
        sqxtn   v21.4h, v17.4s
        ld1r    {v28.8h}, [x2]
        sqxtn2  v31.8h, v22.4s
        sqxtn2  v24.8h, v20.4s
        sqxtn2  v21.8h, v12.4s
        ldr     q22, [sp, 576]
        sqxtn   v18.4h, v13.4s
        sqxtn   v17.4h, v29.4s
        sqxtn   v16.4h, v27.4s
        sqxtn   v0.4h, v25.4s
        sqxtn2  v18.8h, v22.4s
        sqxtn2  v17.8h, v30.4s
        sqxtn2  v16.8h, v26.4s
        sqxtn2  v0.8h, v23.4s
        ldr     q29, [sp, 592]
        sqxtn   v1.4h, v19.4s
        sqxtn   v26.4h, v4.4s
        sqadd   v31.8h, v31.8h, v28.8h
        sqxtn2  v1.8h, v29.4s
        ldr     q29, [sp, 608]
        sqadd   v25.8h, v21.8h, v28.8h
        sqadd   v27.8h, v24.8h, v28.8h
        sqadd   v24.8h, v18.8h, v28.8h
        sqxtn   v2.4h, v29.4s
        ldr     q29, [sp, 624]
        sqadd   v22.8h, v17.8h, v28.8h
        sqadd   v21.8h, v16.8h, v28.8h
        sqadd   v19.8h, v0.8h, v28.8h
        sqxtn2  v2.8h, v29.4s
        ldr     q29, [sp, 640]
        sqadd   v18.8h, v1.8h, v28.8h
        sqadd   v17.8h, v2.8h, v28.8h
        add     x2, x15, 10
        sqxtn   v3.4h, v29.4s
        ldr     q29, [sp, 656]
        sqxtn2  v3.8h, v29.4s
        ldp     q29, q30, [sp, 672]
        sqadd   v16.8h, v3.8h, v28.8h
        sqxtn   v12.4h, v29.4s
        sqxtn   v29.4h, v15.4s
        sqxtn2  v12.8h, v14.4s
        sqxtn2  v29.8h, v8.4s
        sqxtn   v20.4h, v30.4s
        ldr     q30, [sp, 704]
        sqadd   v0.8h, v12.8h, v28.8h
        sqadd   v29.8h, v29.8h, v28.8h
        sqxtn2  v20.8h, v30.4s
        ldr     q30, [sp, 720]
        sqadd   v20.8h, v20.8h, v28.8h
        sqxtn   v23.4h, v30.4s
        ldr     q30, [sp, 736]
        sqxtn2  v23.8h, v30.4s
        sqxtn   v30.4h, v10.4s
        sqxtn2  v26.8h, v11.4s
        sqxtn2  v30.8h, v9.4s
        sqadd   v23.8h, v23.8h, v28.8h
        sqadd   v26.8h, v26.8h, v28.8h
        sqadd   v28.8h, v30.8h, v28.8h
        sqxtn   v30.8b, v31.8h
        sqxtn   v31.8b, v25.8h
        sqxtn2  v30.16b, v27.8h
        sqxtn2  v31.16b, v24.8h
        sqxtn   v27.8b, v22.8h
        sqxtn   v25.8b, v19.8h
        sqxtn   v24.8b, v17.8h
        sqxtn   v22.8b, v0.8h
        sqxtn2  v27.16b, v21.8h
        sqxtn2  v25.16b, v18.8h
        sqxtn2  v24.16b, v16.8h
        sqxtn2  v22.16b, v29.8h
        sqxtn   v29.8b, v20.8h
        sqxtn   v26.8b, v26.8h
        sqxtn2  v29.16b, v23.8h
        sqxtn2  v26.16b, v28.8h
        ld1r    {v28.16b}, [x2]
        add     x2, x15, 11
        ld1r    {v23.16b}, [x2]
        smax    v30.16b, v30.16b, v28.16b
        smax    v21.16b, v31.16b, v28.16b
        smax    v27.16b, v27.16b, v28.16b
        smax    v25.16b, v25.16b, v28.16b
        smax    v20.16b, v24.16b, v28.16b
        smax    v22.16b, v22.16b, v28.16b
        smax    v19.16b, v29.16b, v28.16b
        smax    v18.16b, v26.16b, v28.16b
        smin    v31.16b, v30.16b, v23.16b
        smin    v24.16b, v21.16b, v23.16b
        smin    v30.16b, v27.16b, v23.16b
        smin    v25.16b, v25.16b, v23.16b
        smin    v29.16b, v20.16b, v23.16b
        smin    v26.16b, v22.16b, v23.16b
        smin    v28.16b, v19.16b, v23.16b
        smin    v27.16b, v18.16b, v23.16b
        b.ls    8f
        sub     x28, x24, x21
        sub     x20, x20, x21
        sub     x27, x23, x21
        sub     x19, x19, x21
        sub     x26, x22, x21
        sub     x17, x17, x21
        sub     x25, x7, x21
        sub     x3, x3, x21
        str     q31, [x6]
        str     q24, [x8]
        str     q30, [x9]
        str     q25, [x10]
        str     q29, [x11]
        str     q26, [x12]
        str     q28, [x13]
        str     q27, [x14]
        add     x6, x6, x16
        add     x8, x8, x16
        add     x9, x9, x16
        add     x10, x10, x16
        add     x11, x11, x16
        add     x12, x12, x16
        add     x13, x13, x16
        subs    x1, x1, 0x10
        add     x14, x14, x16
        b.ne    1b
        b       12f
6:
        str     q26, [sp, 464]
        mov     v3.16b, v27.16b
        ldr     d26, [x22]
        ldr     d6, [x19]
        ldp     q31, q27, [x5]
        mov     v20.16b, v2.16b
        mov     v0.16b, v9.16b
        ldr     d29, [x7]
        sdot    v20.4s, v27.16b, v26.4b[0]
        sdot    v23.4s, v27.16b, v6.4b[0]
        ldr     d9, [x17]
        ldr     d28, [x23]
        str     q20, [sp, 688]
        str     q23, [sp, 544]
        ldp     q23, q20, [sp, 592]
        ldr     d25, [x20]
        sdot    v7.4s, v27.16b, v29.4b[0]
        sdot    v16.4s, v31.16b, v9.4b[0]
        sdot    v20.4s, v31.16b, v28.4b[0]
        ldr     d30, [x3]
        ldr     d2, [x24]
        stp     q7, q22, [sp, 480]
        str     q16, [sp, 720]
        str     q20, [sp, 608]
        ldr     q16, [sp, 560]
        ldr     q20, [sp, 528]
        ldr     q22, [sp, 736]
        sdot    v23.4s, v27.16b, v25.4b[0]
        sdot    v19.4s, v31.16b, v30.4b[0]
        sdot    v17.4s, v27.16b, v30.4b[0]
        sdot    v15.4s, v31.16b, v29.4b[0]
        sdot    v14.4s, v27.16b, v9.4b[0]
        sdot    v3.4s, v31.16b, v26.4b[0]
        sdot    v1.4s, v31.16b, v6.4b[0]
        sdot    v8.4s, v27.16b, v28.4b[0]
        sdot    v16.4s, v31.16b, v25.4b[0]
        sdot    v20.4s, v31.16b, v2.4b[0]
        str     q23, [sp, 592]
        sdot    v22.4s, v27.16b, v2.4b[0]
        ldp     q31, q27, [x5, 32]
        ldr     q23, [sp, 624]
        ldr     q7, [sp, 464]
        str     q14, [sp, 752]
        sdot    v0.4s, v31.16b, v30.4b[0]
        sdot    v23.4s, v31.16b, v25.4b[0]
        str     q23, [sp, 624]
        ldr     q23, [sp, 704]
        sdot    v4.4s, v31.16b, v29.4b[0]
        sdot    v7.4s, v31.16b, v9.4b[0]
        sdot    v11.4s, v31.16b, v26.4b[0]
        sdot    v10.4s, v31.16b, v6.4b[0]
        sdot    v21.4s, v31.16b, v28.4b[0]
        sdot    v23.4s, v31.16b, v2.4b[0]
        ldr     q31, [sp, 640]
        str     q3, [sp, 336]
        ldr     q3, [sp, 496]
        ldr     q14, [sp, 576]
        sdot    v31.4s, v27.16b, v6.4b[0]
        str     q31, [sp, 640]
        ldr     q31, [sp, 656]
        sdot    v14.4s, v27.16b, v30.4b[0]
        sdot    v3.4s, v27.16b, v29.4b[0]
        sdot    v18.4s, v27.16b, v9.4b[0]
        sdot    v5.4s, v27.16b, v26.4b[0]
        sdot    v12.4s, v27.16b, v28.4b[0]
        sdot    v31.4s, v27.16b, v25.4b[0]
        sdot    v24.4s, v27.16b, v2.4b[0]
        add     x3, x3, x29
        add     x7, x7, x29
        add     x17, x17, x29
        add     x22, x22, x29
        add     x19, x19, x29
        add     x23, x23, x29
        add     x20, x20, x29
        cmp     x29, 5
        add     x24, x24, x29
        str     q23, [sp, 704]
        str     q28, [sp, 736]
        str     q25, [sp, 528]
        stp     q31, q2, [sp, 656]
        b.cc    7f
        ldp     q27, q31, [x5, 96]
        mov     v23.16b, v26.16b
        ldp     q25, q26, [x5, 64]
        str     q3, [sp, 496]
        sdot    v0.4s, v27.16b, v30.4b[1]
        str     q0, [sp, 448]
        sdot    v19.4s, v25.16b, v30.4b[1]
        sdot    v17.4s, v26.16b, v30.4b[1]
        sdot    v14.4s, v31.16b, v30.4b[1]
        ldp     q0, q30, [sp, 480]
        sdot    v15.4s, v25.16b, v29.4b[1]
        sdot    v4.4s, v27.16b, v29.4b[1]
        ldr     q28, [sp, 336]
        sdot    v0.4s, v26.16b, v29.4b[1]
        sdot    v30.4s, v31.16b, v29.4b[1]
        ldr     q29, [sp, 752]
        str     q30, [sp, 496]
        ldr     q30, [sp, 720]
        sdot    v28.4s, v25.16b, v23.4b[1]
        sdot    v29.4s, v26.16b, v9.4b[1]
        str     q29, [sp, 752]
        mov     v29.16b, v18.16b
        sdot    v30.4s, v25.16b, v9.4b[1]
        sdot    v7.4s, v27.16b, v9.4b[1]
        sdot    v29.4s, v31.16b, v9.4b[1]
        mov     v9.16b, v28.16b
        ldr     q28, [sp, 640]
        ldr     q18, [sp, 736]
        ldr     q2, [sp, 688]
        sdot    v11.4s, v27.16b, v23.4b[1]
        sdot    v28.4s, v31.16b, v6.4b[1]
        str     q28, [sp, 640]
        ldr     q28, [sp, 608]
        sdot    v8.4s, v26.16b, v18.4b[1]
        sdot    v21.4s, v27.16b, v18.4b[1]
        sdot    v12.4s, v31.16b, v18.4b[1]
        sdot    v28.4s, v25.16b, v18.4b[1]
        str     q28, [sp, 608]
        ldr     q18, [sp, 528]
        ldr     q28, [sp, 592]
        sdot    v2.4s, v26.16b, v23.4b[1]
        sdot    v5.4s, v31.16b, v23.4b[1]
        sdot    v16.4s, v25.16b, v18.4b[1]
        sdot    v28.4s, v26.16b, v18.4b[1]
        str     q28, [sp, 592]
        ldr     q28, [sp, 624]
        ldr     q23, [sp, 544]
        str     q14, [sp, 576]
        sdot    v10.4s, v27.16b, v6.4b[1]
        sdot    v28.4s, v27.16b, v18.4b[1]
        str     q28, [sp, 624]
        ldr     q28, [sp, 656]
        ldr     q14, [sp, 752]
        sdot    v1.4s, v25.16b, v6.4b[1]
        sdot    v23.4s, v26.16b, v6.4b[1]
        sdot    v28.4s, v31.16b, v18.4b[1]
        ldr     q18, [sp, 672]
        str     q28, [sp, 656]
        ldr     q28, [sp, 704]
        str     q16, [sp, 560]
        sdot    v22.4s, v26.16b, v18.4b[1]
        str     q22, [sp, 736]
        sdot    v28.4s, v27.16b, v18.4b[1]
        ldr     q22, [sp, 496]
        mov     v27.16b, v9.16b
        ldr     q9, [sp, 448]
        sdot    v20.4s, v25.16b, v18.4b[1]
        mov     v26.16b, v7.16b
        sdot    v24.4s, v31.16b, v18.4b[1]
        add     x5, x5, 0x80
        mov     v18.16b, v29.16b
        mov     v16.16b, v30.16b
        mov     v7.16b, v0.16b
        mov     v3.16b, v2.16b
        str     q20, [sp, 528]
        str     q28, [sp, 704]
        b       5b
7:
        str     q22, [sp, 736]
        ldr     q22, [sp, 480]
        stp     q16, q14, [sp, 560]
        ldr     q16, [sp, 720]
        ldr     q23, [sp, 544]
        ldr     q27, [sp, 336]
        ldr     q2, [sp, 688]
        ldr     q14, [sp, 752]
        str     q20, [sp, 528]
        add     x5, x5, 64
        mov     v26.16b, v7.16b
        mov     v7.16b, v22.16b
        mov     v9.16b, v0.16b
        mov     v22.16b, v3.16b
        b       5b
8:
        tbz     w1, 3, 9f
        str     d31, [x6], 8
        str     d24, [x8], 8
        str     d30, [x9], 8
        str     d25, [x10], 8
        str     d29, [x11], 8
        str     d26, [x12], 8
        str     d28, [x13], 8
        str     d27, [x14], 8
        ext     v31.16b, v31.16b, v31.16b, 8
        ext     v24.16b, v24.16b, v24.16b, 8
        ext     v30.16b, v30.16b, v30.16b, 8
        ext     v25.16b, v25.16b, v25.16b, 8
        ext     v29.16b, v29.16b, v29.16b, 8
        ext     v26.16b, v26.16b, v26.16b, 8
        ext     v28.16b, v28.16b, v28.16b, 8
        ext     v27.16b, v27.16b, v27.16b, 8
9:
        mov     v31.d[1], v24.d[0]
        mov     v30.d[1], v25.d[0]
        mov     v29.d[1], v26.d[0]
        mov     v28.d[1], v27.d[0]
        tbnz    w1, 2, 13f
        tbnz    w1, 1, 14f
10:
        tbz     w1, 0, 12f
11:
        st1     {v31.b}[0], [x6]
        st1     {v31.b}[8], [x8]
        st1     {v30.b}[0], [x9]
        st1     {v30.b}[8], [x10]
        st1     {v29.b}[0], [x11]
        st1     {v29.b}[8], [x12]
        st1     {v28.b}[0], [x13]
        st1     {v28.b}[8], [x14]
12:
        add     sp, sp, 0x300
        ldp     x29, x30, [sp, 144]
        ldp     x20, x19, [sp, 128]
        ldp     x22, x21, [sp, 112]
        ldp     x24, x23, [sp, 96]
        ldp     x26, x25, [sp, 80]
        ldp     x28, x27, [sp, 64]
        ldp     d9, d8, [sp, 48]
        ldp     d11, d10, [sp, 32]
        ldp     d5, d4, [sp, 16]
        ldp     d7, d6, [sp], 160
        ret
13:
        st1     {v31.s}[0], [x6], 4
        st1     {v31.s}[2], [x8], 4
        st1     {v30.s}[0], [x9], 4
        st1     {v30.s}[2], [x10], 4
        st1     {v29.s}[0], [x11], 4
        st1     {v29.s}[2], [x12], 4
        st1     {v28.s}[0], [x13], 4
        st1     {v28.s}[2], [x14], 4
        ext     v31.16b, v31.16b, v31.16b, 4
        ext     v30.16b, v30.16b, v30.16b, 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
        tbz     w1, 1, 10b
14:
        st1     {v31.h}[0], [x6], 2
        st1     {v31.h}[4], [x8], 2
        st1     {v30.h}[0], [x9], 2
        st1     {v30.h}[4], [x10], 2
        st1     {v29.h}[0], [x11], 2
        st1     {v29.h}[4], [x12], 2
        st1     {v28.h}[0], [x13], 2
        st1     {v28.h}[4], [x14], 2
        ext     v31.16b, v31.16b, v31.16b, 2
        ext     v30.16b, v30.16b, v30.16b, 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 11b
        b       12b

END_FUNCTION xnn_qs8_gemm_minmax_ukernel_8x16c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
