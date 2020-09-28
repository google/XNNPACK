// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_igemm_minmax_ukernel_12x8c4__aarch64_neondot
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
        sub     sp, sp, 0x1b0
        ldr     x8, [sp, 616]
        cmp     x0, 2
        mov     x22, x2
        sub     x9, x2, 8
        str     x8, [sp, 32]
        ldr     x8, [sp, 592]
        ldr     x20, [sp, 608]
        ldr     x21, [sp, 600]
        and     x10, x9, 0xfffffffffffffff8
        str     x8, [sp, 16]
        add     x8, x6, x7
        csel    x12, x6, x8, lo
        cmp     x0, 3
        add     x11, x12, x7
        csel    x13, x12, x11, lo
        cmp     x0, 4
        add     x11, x13, x7
        csel    x14, x13, x11, lo
        cmp     x0, 5
        add     x11, x14, x7
        csel    x15, x14, x11, lo
        cmp     x0, 6
        add     x11, x15, x7
        csel    x16, x15, x11, lo
        cmp     x0, 7
        add     x11, x16, x7
        csel    x17, x16, x11, lo
        cmp     x0, 8
        add     x11, x17, x7
        csel    x2, x17, x11, lo
        cmp     x0, 9
        add     x11, x2, x7
        csel    x25, x2, x11, lo
        cmp     x0, 10
        add     x11, x25, x7
        csel    x19, x25, x11, lo
        lsl     x8, x9, 3
        cmp     x0, 11
        add     x11, x19, x7
        and     x8, x8, 0xffffffffffffffc0
        csel    x11, x19, x11, lo
        cmp     x0, 12
        add     x24, x8, 64
        add     x8, x11, x7
        add     x23, x10, 8
        csel    x8, x8, x11, eq
        sub     x7, x9, x10
        str     x3, [sp, 24]
1:
        ldp     q8, q23, [x5], 32
        stp     x19, x25, [sp, 56]
        mov     x25, x3
        stp     x8, x11, [sp, 40]
        mov     v31.16b, v23.16b
        mov     v10.16b, v8.16b
        mov     v11.16b, v23.16b
        mov     v30.16b, v8.16b
        mov     v29.16b, v23.16b
        mov     v27.16b, v8.16b
        mov     v25.16b, v23.16b
        mov     v22.16b, v8.16b
        mov     v4.16b, v23.16b
        mov     v20.16b, v8.16b
        mov     v19.16b, v23.16b
        mov     v18.16b, v8.16b
        mov     v17.16b, v23.16b
        mov     v16.16b, v8.16b
        mov     v28.16b, v23.16b
        mov     v0.16b, v8.16b
        mov     v1.16b, v23.16b
        mov     v26.16b, v8.16b
        mov     v2.16b, v23.16b
        mov     v3.16b, v8.16b
        mov     v12.16b, v23.16b
        mov     v13.16b, v8.16b
        stp     x2, x17, [sp, 72]
        stp     x16, x15, [sp, 88]
        stp     x14, x13, [sp, 104]
        stp     x12, x1, [sp, 120]
        str     x6, [sp, 136]
2:
        ldp     x8, x9, [x4]
        ldp     x10, x11, [x4, 16]
        ldp     x12, x13, [x4, 32]
        ldp     x14, x15, [x4, 48]
        add     x0, x8, x21
        cmp     x8, x20
        csel    x19, x8, x0, eq
        add     x8, x9, x21
        cmp     x9, x20
        csel    x0, x9, x8, eq
        add     x8, x10, x21
        cmp     x10, x20
        csel    x30, x10, x8, eq
        add     x8, x11, x21
        cmp     x11, x20
        ldp     x16, x17, [x4, 64]
        csel    x9, x11, x8, eq
        add     x8, x12, x21
        cmp     x12, x20
        csel    x29, x12, x8, eq
        add     x8, x13, x21
        cmp     x13, x20
        ldp     x2, x3, [x4, 80]
        csel    x1, x13, x8, eq
        add     x8, x14, x21
        cmp     x14, x20
        csel    x28, x14, x8, eq
        add     x8, x15, x21
        cmp     x15, x20
        csel    x6, x15, x8, eq
        add     x8, x16, x21
        cmp     x16, x20
        csel    x27, x16, x8, eq
        add     x8, x17, x21
        cmp     x17, x20
        csel    x8, x17, x8, eq
        add     x10, x2, x21
        cmp     x2, x20
        csel    x26, x2, x10, eq
        add     x10, x3, x21
        cmp     x3, x20
        csel    x3, x3, x10, eq
        cmp     x22, 8
        b.cc    5f
        mov     x16, xzr
        add     x10, x0, x23
        add     x11, x9, x23
        add     x12, x1, x23
        add     x13, x6, x23
        add     x14, x8, x23
        add     x15, x3, x23
        add     x17, x5, 32
        mov     x2, x22
        stp     q0, q12, [sp, 336]
        mov     v5.16b, v22.16b
        mov     v7.16b, v31.16b
        stp     q3, q13, [sp, 368]
        stp     q17, q19, [sp, 288]
        mov     v19.16b, v30.16b
        mov     v3.16b, v18.16b
        mov     v18.16b, v29.16b
        mov     v17.16b, v27.16b
        mov     v12.16b, v16.16b
        mov     v16.16b, v25.16b
        mov     v24.16b, v20.16b
        mov     v9.16b, v26.16b
        str     q4, [sp, 272]
        str     q28, [sp, 320]
3:
        ldr     d29, [x8, x16]
        str     q7, [sp, 224]
        ldr     d30, [x19, x16]
        ldr     d31, [x0, x16]
        ldr     d22, [x30, x16]
        ldr     d28, [x9, x16]
        ldr     d15, [x29, x16]
        ldr     d14, [x1, x16]
        ldr     d20, [x28, x16]
        ldr     d13, [x6, x16]
        ldr     d7, [x27, x16]
        stp     q29, q8, [sp, 400]
        ldp     q27, q26, [x17, -32]
        ldr     d29, [x3, x16]
        ldr     q8, [sp, 336]
        ldr     d6, [x26, x16]
        sdot    v12.4s, v27.16b, v15.4b[0]
        str     q29, [sp, 208]
        sdot    v24.4s, v27.16b, v20.4b[0]
        sdot    v8.4s, v27.16b, v28.4b[0]
        mov     v29.16b, v28.16b
        mov     v28.16b, v15.16b
        stp     q15, q14, [sp, 144]
        stp     q24, q12, [sp, 240]
        ldp     q12, q25, [sp, 400]
        ldr     q15, [sp, 208]
        ldp     q4, q0, [sp, 368]
        ldr     q24, [sp, 320]
        sdot    v9.4s, v27.16b, v22.4b[0]
        sdot    v25.4s, v27.16b, v15.4b[0]
        sdot    v0.4s, v27.16b, v30.4b[0]
        sdot    v4.4s, v27.16b, v31.4b[0]
        sdot    v3.4s, v27.16b, v14.4b[0]
        sdot    v5.4s, v27.16b, v13.4b[0]
        sdot    v17.4s, v27.16b, v7.4b[0]
        sdot    v19.4s, v27.16b, v12.4b[0]
        sdot    v10.4s, v27.16b, v6.4b[0]
        str     q25, [sp, 416]
        ldp     q27, q25, [x17], 64
        sdot    v24.4s, v26.16b, v29.4b[0]
        mov     v21.16b, v14.16b
        ldr     q14, [sp, 288]
        sdot    v8.4s, v27.16b, v29.4b[1]
        sdot    v24.4s, v25.16b, v29.4b[1]
        sdot    v11.4s, v26.16b, v12.4b[0]
        stp     q24, q8, [sp, 320]
        ldp     q24, q12, [sp, 240]
        ldr     q29, [sp, 144]
        stp     q7, q19, [sp, 176]
        mov     v19.16b, v3.16b
        mov     v3.16b, v5.16b
        ldr     q5, [sp, 352]
        sdot    v14.4s, v26.16b, v28.4b[0]
        ldr     q28, [sp, 304]
        sdot    v12.4s, v27.16b, v29.4b[1]
        sdot    v14.4s, v25.16b, v29.4b[1]
        ldr     q29, [sp, 160]
        sdot    v5.4s, v26.16b, v30.4b[0]
        sdot    v5.4s, v25.16b, v30.4b[1]
        sdot    v28.4s, v26.16b, v21.4b[0]
        str     q5, [sp, 352]
        mov     v5.16b, v3.16b
        mov     v3.16b, v19.16b
        sdot    v3.4s, v27.16b, v29.4b[1]
        sdot    v28.4s, v25.16b, v29.4b[1]
        ldr     q29, [sp, 176]
        ldr     q21, [sp, 272]
        sdot    v18.4s, v26.16b, v7.4b[0]
        ldr     q7, [sp, 224]
        ldr     q19, [sp, 192]
        sdot    v17.4s, v27.16b, v29.4b[1]
        sdot    v18.4s, v25.16b, v29.4b[1]
        ldp     q29, q8, [sp, 400]
        sdot    v2.4s, v26.16b, v31.4b[0]
        sdot    v1.4s, v26.16b, v22.4b[0]
        sdot    v21.4s, v26.16b, v20.4b[0]
        sdot    v16.4s, v26.16b, v13.4b[0]
        sdot    v7.4s, v26.16b, v6.4b[0]
        sdot    v23.4s, v26.16b, v15.4b[0]
        sub     x2, x2, 8
        sdot    v0.4s, v27.16b, v30.4b[1]
        sdot    v4.4s, v27.16b, v31.4b[1]
        sdot    v2.4s, v25.16b, v31.4b[1]
        sdot    v9.4s, v27.16b, v22.4b[1]
        sdot    v1.4s, v25.16b, v22.4b[1]
        sdot    v24.4s, v27.16b, v20.4b[1]
        sdot    v21.4s, v25.16b, v20.4b[1]
        sdot    v5.4s, v27.16b, v13.4b[1]
        sdot    v16.4s, v25.16b, v13.4b[1]
        sdot    v19.4s, v27.16b, v29.4b[1]
        sdot    v11.4s, v25.16b, v29.4b[1]
        sdot    v10.4s, v27.16b, v6.4b[1]
        sdot    v7.4s, v25.16b, v6.4b[1]
        sdot    v8.4s, v27.16b, v15.4b[1]
        sdot    v23.4s, v25.16b, v15.4b[1]
        add     x16, x16, 8
        cmp     x2, 7
        stp     q4, q0, [sp, 368]
        stp     q14, q28, [sp, 288]
        str     q21, [sp, 272]
        b.hi    3b
        mov     v30.16b, v19.16b
        mov     v29.16b, v18.16b
        mov     v27.16b, v17.16b
        mov     v25.16b, v16.16b
        ldp     q4, q17, [sp, 272]
        ldp     q19, q28, [sp, 304]
        mov     v18.16b, v3.16b
        mov     v16.16b, v12.16b
        ldp     q0, q12, [sp, 336]
        ldp     q3, q13, [sp, 368]
        add     x19, x19, x23
        add     x30, x30, x23
        add     x29, x29, x23
        add     x28, x28, x23
        add     x27, x27, x23
        add     x26, x26, x23
        add     x5, x5, x24
        mov     x16, x7
        mov     v31.16b, v7.16b
        mov     v22.16b, v5.16b
        mov     v20.16b, v24.16b
        mov     v26.16b, v9.16b
        add     x4, x4, 0x60
        cbnz    x16, 6f
4:
        subs    x25, x25, 0x60
        b.ne    2b
        b       9f
5:
        mov     x16, x22
        mov     x15, x3
        mov     x14, x8
        mov     x13, x6
        mov     x12, x1
        mov     x11, x9
        mov     x10, x0
        add     x4, x4, 0x60
        cbz     x16, 4b
6:
        stp     q2, q8, [sp, 400]
        str     q28, [sp, 320]
        str     q29, [sp, 272]
        str     q12, [sp, 352]
        mov     v12.16b, v19.16b
        ldr     d19, [x11]
        ldp     q28, q29, [x5]
        ldr     d6, [x15]
        str     q1, [sp, 240]
        mov     v1.16b, v25.16b
        sdot    v0.4s, v28.16b, v19.4b[0]
        str     q0, [sp, 336]
        ldr     q0, [sp, 416]
        ldr     d25, [x29]
        mov     v15.16b, v17.16b
        mov     v2.16b, v4.16b
        ldr     d17, [x19]
        ldr     d21, [x10]
        ldr     d24, [x30]
        ldr     d14, [x12]
        ldr     d5, [x28]
        ldr     d4, [x13]
        ldr     d9, [x27]
        ldr     d7, [x14]
        ldr     d8, [x26]
        sdot    v0.4s, v28.16b, v6.4b[0]
        sdot    v16.4s, v28.16b, v25.4b[0]
        str     q0, [sp, 416]
        sdot    v13.4s, v28.16b, v17.4b[0]
        sdot    v3.4s, v28.16b, v21.4b[0]
        sdot    v26.4s, v28.16b, v24.4b[0]
        str     q16, [sp, 256]
        sdot    v18.4s, v28.16b, v14.4b[0]
        sdot    v20.4s, v28.16b, v5.4b[0]
        sdot    v22.4s, v28.16b, v4.4b[0]
        sdot    v27.4s, v28.16b, v9.4b[0]
        sdot    v30.4s, v28.16b, v7.4b[0]
        sdot    v10.4s, v28.16b, v8.4b[0]
        mov     v16.16b, v8.16b
        ldp     q28, q8, [sp, 400]
        stp     q10, q26, [sp, 288]
        ldr     q26, [sp, 240]
        stp     q3, q13, [sp, 368]
        sdot    v28.4s, v29.16b, v21.4b[0]
        str     q28, [sp, 400]
        ldr     q28, [sp, 272]
        ldr     q13, [sp, 352]
        mov     v3.16b, v24.16b
        sdot    v26.4s, v29.16b, v24.4b[0]
        ldr     q24, [sp, 320]
        sdot    v15.4s, v29.16b, v25.4b[0]
        sdot    v12.4s, v29.16b, v14.4b[0]
        sdot    v1.4s, v29.16b, v4.4b[0]
        sdot    v28.4s, v29.16b, v9.4b[0]
        cmp     x16, 5
        mov     v10.16b, v17.16b
        sdot    v13.4s, v29.16b, v17.4b[0]
        mov     v0.16b, v21.16b
        mov     v21.16b, v19.16b
        sdot    v24.4s, v29.16b, v19.4b[0]
        mov     v19.16b, v25.16b
        mov     v17.16b, v15.16b
        mov     v25.16b, v14.16b
        mov     v15.16b, v12.16b
        mov     v12.16b, v5.16b
        sdot    v2.4s, v29.16b, v5.4b[0]
        mov     v14.16b, v4.16b
        mov     v4.16b, v1.16b
        mov     v5.16b, v9.16b
        mov     v9.16b, v28.16b
        sdot    v11.4s, v29.16b, v7.4b[0]
        sdot    v31.4s, v29.16b, v16.4b[0]
        sdot    v23.4s, v29.16b, v6.4b[0]
        str     q7, [sp, 272]
        str     q6, [sp, 320]
        b.cc    7f
        mov     v7.16b, v16.16b
        ldp     q28, q29, [x5, 32]
        ldp     q1, q16, [sp, 384]
        mov     v6.16b, v26.16b
        ldr     q26, [sp, 336]
        sdot    v13.4s, v29.16b, v10.4b[1]
        sdot    v1.4s, v28.16b, v10.4b[1]
        str     q1, [sp, 384]
        ldr     q1, [sp, 368]
        sdot    v16.4s, v29.16b, v0.4b[1]
        sdot    v6.4s, v29.16b, v3.4b[1]
        str     q16, [sp, 400]
        sdot    v1.4s, v28.16b, v0.4b[1]
        ldr     q0, [sp, 304]
        stp     q13, q1, [sp, 352]
        sdot    v26.4s, v28.16b, v21.4b[1]
        sdot    v2.4s, v29.16b, v12.4b[1]
        sdot    v0.4s, v28.16b, v3.4b[1]
        ldp     q1, q3, [sp, 256]
        str     q0, [sp, 304]
        ldr     q0, [sp, 416]
        sdot    v4.4s, v29.16b, v14.4b[1]
        sdot    v30.4s, v28.16b, v3.4b[1]
        sdot    v11.4s, v29.16b, v3.4b[1]
        ldr     q3, [sp, 320]
        ldr     q10, [sp, 288]
        sdot    v18.4s, v28.16b, v25.4b[1]
        sdot    v15.4s, v29.16b, v25.4b[1]
        sdot    v0.4s, v28.16b, v3.4b[1]
        str     q0, [sp, 416]
        sdot    v20.4s, v28.16b, v12.4b[1]
        sdot    v23.4s, v29.16b, v3.4b[1]
        mov     v25.16b, v4.16b
        mov     v4.16b, v2.16b
        mov     v0.16b, v26.16b
        ldr     q26, [sp, 304]
        ldp     q2, q8, [sp, 400]
        ldp     q12, q3, [sp, 352]
        sdot    v24.4s, v29.16b, v21.4b[1]
        sdot    v1.4s, v28.16b, v19.4b[1]
        sdot    v9.4s, v29.16b, v5.4b[1]
        sdot    v17.4s, v29.16b, v19.4b[1]
        sdot    v22.4s, v28.16b, v14.4b[1]
        sdot    v27.4s, v28.16b, v5.4b[1]
        sdot    v10.4s, v28.16b, v7.4b[1]
        sdot    v31.4s, v29.16b, v7.4b[1]
        add     x5, x5, 64
        mov     v29.16b, v9.16b
        mov     v19.16b, v15.16b
        mov     v16.16b, v1.16b
        mov     v28.16b, v24.16b
        mov     v1.16b, v6.16b
        b       8f
7:
        mov     v25.16b, v4.16b
        mov     v4.16b, v2.16b
        ldr     q16, [sp, 256]
        ldr     q0, [sp, 336]
        mov     v1.16b, v26.16b
        ldp     q10, q26, [sp, 288]
        ldr     q2, [sp, 400]
        ldr     q3, [sp, 368]
        add     x5, x5, 32
        mov     v29.16b, v9.16b
        mov     v19.16b, v15.16b
        mov     v28.16b, v24.16b
        mov     v12.16b, v13.16b
8:
        ldr     q13, [sp, 384]
        subs    x25, x25, 0x60
        b.ne    2b
9:
        ldr     x9, [sp, 32]
        mov     x8, x9
        ld1r    {v9.4s}, [x8], 4
        ld1r    {v5.4s}, [x8]
        sqrdmulh        v15.4s, v13.4s, v9.4s
        sqrdmulh        v13.4s, v3.4s, v9.4s
        sqrdmulh        v3.4s, v26.4s, v9.4s
        sqrdmulh        v26.4s, v30.4s, v9.4s
        mov     v30.16b, v31.16b
        sqrdmulh        v14.4s, v12.4s, v9.4s
        sqrdmulh        v12.4s, v2.4s, v9.4s
        sqrdmulh        v2.4s, v1.4s, v9.4s
        sqrdmulh        v1.4s, v0.4s, v9.4s
        sqrdmulh        v0.4s, v28.4s, v9.4s
        sqrdmulh        v16.4s, v16.4s, v9.4s
        sqrdmulh        v17.4s, v17.4s, v9.4s
        sqrdmulh        v18.4s, v18.4s, v9.4s
        sqrdmulh        v19.4s, v19.4s, v9.4s
        sqrdmulh        v20.4s, v20.4s, v9.4s
        sqrdmulh        v21.4s, v4.4s, v9.4s
        sqrdmulh        v22.4s, v22.4s, v9.4s
        sqrdmulh        v4.4s, v25.4s, v9.4s
        sqrdmulh        v24.4s, v27.4s, v9.4s
        sqrdmulh        v25.4s, v29.4s, v9.4s
        sqrdmulh        v27.4s, v11.4s, v9.4s
        sqrdmulh        v31.4s, v10.4s, v9.4s
        sqrdmulh        v30.4s, v30.4s, v9.4s
        sqrdmulh        v28.4s, v8.4s, v9.4s
        sqrdmulh        v29.4s, v23.4s, v9.4s
        cmeq    v9.4s, v5.4s, 0
        mvn     v10.16b, v9.16b
        and     v9.16b, v15.16b, v10.16b
        ssra    v15.4s, v9.4s, 31
        and     v9.16b, v14.16b, v10.16b
        ssra    v14.4s, v9.4s, 31
        and     v9.16b, v13.16b, v10.16b
        ssra    v13.4s, v9.4s, 31
        and     v9.16b, v12.16b, v10.16b
        ssra    v12.4s, v9.4s, 31
        and     v9.16b, v3.16b, v10.16b
        ssra    v3.4s, v9.4s, 31
        and     v9.16b, v2.16b, v10.16b
        ssra    v2.4s, v9.4s, 31
        and     v9.16b, v1.16b, v10.16b
        ssra    v1.4s, v9.4s, 31
        and     v9.16b, v0.16b, v10.16b
        ssra    v0.4s, v9.4s, 31
        and     v9.16b, v16.16b, v10.16b
        ssra    v16.4s, v9.4s, 31
        and     v9.16b, v17.16b, v10.16b
        ssra    v17.4s, v9.4s, 31
        and     v9.16b, v18.16b, v10.16b
        ssra    v18.4s, v9.4s, 31
        and     v9.16b, v19.16b, v10.16b
        ssra    v19.4s, v9.4s, 31
        and     v9.16b, v20.16b, v10.16b
        ssra    v20.4s, v9.4s, 31
        and     v9.16b, v21.16b, v10.16b
        ssra    v21.4s, v9.4s, 31
        and     v9.16b, v22.16b, v10.16b
        ssra    v22.4s, v9.4s, 31
        and     v9.16b, v4.16b, v10.16b
        ssra    v4.4s, v9.4s, 31
        and     v9.16b, v24.16b, v10.16b
        ssra    v24.4s, v9.4s, 31
        and     v9.16b, v25.16b, v10.16b
        ssra    v25.4s, v9.4s, 31
        and     v9.16b, v26.16b, v10.16b
        ssra    v26.4s, v9.4s, 31
        and     v9.16b, v27.16b, v10.16b
        ssra    v27.4s, v9.4s, 31
        and     v9.16b, v31.16b, v10.16b
        ssra    v31.4s, v9.4s, 31
        and     v9.16b, v30.16b, v10.16b
        add     x8, x9, 8
        ssra    v30.4s, v9.4s, 31
        and     v9.16b, v28.16b, v10.16b
        ssra    v28.4s, v9.4s, 31
        ld1r    {v9.8h}, [x8]
        and     v10.16b, v29.16b, v10.16b
        srshl   v15.4s, v15.4s, v5.4s
        srshl   v13.4s, v13.4s, v5.4s
        srshl   v3.4s, v3.4s, v5.4s
        srshl   v1.4s, v1.4s, v5.4s
        srshl   v16.4s, v16.4s, v5.4s
        srshl   v18.4s, v18.4s, v5.4s
        srshl   v20.4s, v20.4s, v5.4s
        srshl   v22.4s, v22.4s, v5.4s
        srshl   v24.4s, v24.4s, v5.4s
        srshl   v26.4s, v26.4s, v5.4s
        srshl   v31.4s, v31.4s, v5.4s
        ssra    v29.4s, v10.4s, 31
        srshl   v14.4s, v14.4s, v5.4s
        srshl   v12.4s, v12.4s, v5.4s
        srshl   v2.4s, v2.4s, v5.4s
        srshl   v0.4s, v0.4s, v5.4s
        srshl   v17.4s, v17.4s, v5.4s
        srshl   v19.4s, v19.4s, v5.4s
        srshl   v21.4s, v21.4s, v5.4s
        srshl   v23.4s, v4.4s, v5.4s
        srshl   v25.4s, v25.4s, v5.4s
        srshl   v27.4s, v27.4s, v5.4s
        srshl   v30.4s, v30.4s, v5.4s
        srshl   v28.4s, v28.4s, v5.4s
        sqxtn   v15.4h, v15.4s
        sqxtn   v13.4h, v13.4s
        sqxtn   v3.4h, v3.4s
        sqxtn   v1.4h, v1.4s
        sqxtn   v16.4h, v16.4s
        sqxtn   v18.4h, v18.4s
        sqxtn   v20.4h, v20.4s
        sqxtn   v22.4h, v22.4s
        sqxtn   v24.4h, v24.4s
        sqxtn   v26.4h, v26.4s
        sqxtn   v31.4h, v31.4s
        srshl   v29.4s, v29.4s, v5.4s
        sqxtn   v28.4h, v28.4s
        sqxtn2  v15.8h, v14.4s
        sqxtn2  v13.8h, v12.4s
        sqxtn2  v3.8h, v2.4s
        sqxtn2  v1.8h, v0.4s
        sqxtn2  v16.8h, v17.4s
        sqxtn2  v18.8h, v19.4s
        sqxtn2  v20.8h, v21.4s
        sqxtn2  v22.8h, v23.4s
        sqxtn2  v24.8h, v25.4s
        sqxtn2  v26.8h, v27.4s
        sqxtn2  v31.8h, v30.4s
        sqxtn2  v28.8h, v29.4s
        sqadd   v29.8h, v15.8h, v9.8h
        sqadd   v27.8h, v3.8h, v9.8h
        sqadd   v23.8h, v16.8h, v9.8h
        sqadd   v20.8h, v20.8h, v9.8h
        sqadd   v24.8h, v24.8h, v9.8h
        sqadd   v31.8h, v31.8h, v9.8h
        add     x8, x9, 10
        sqadd   v30.8h, v13.8h, v9.8h
        sqadd   v25.8h, v1.8h, v9.8h
        sqadd   v21.8h, v18.8h, v9.8h
        sqadd   v22.8h, v22.8h, v9.8h
        sqadd   v26.8h, v26.8h, v9.8h
        sqadd   v28.8h, v28.8h, v9.8h
        sqxtn   v29.8b, v29.8h
        sqxtn   v27.8b, v27.8h
        sqxtn   v23.8b, v23.8h
        sqxtn   v20.8b, v20.8h
        sqxtn   v24.8b, v24.8h
        sqxtn   v31.8b, v31.8h
        sqxtn2  v29.16b, v30.8h
        sqxtn2  v27.16b, v25.8h
        sqxtn2  v23.16b, v21.8h
        sqxtn2  v20.16b, v22.8h
        sqxtn2  v24.16b, v26.8h
        sqxtn2  v31.16b, v28.8h
        ld1r    {v28.16b}, [x8]
        add     x8, x9, 11
        ldr     x1, [sp, 128]
        ld1r    {v25.16b}, [x8]
        smax    v30.16b, v31.16b, v28.16b
        smax    v31.16b, v24.16b, v28.16b
        smax    v24.16b, v20.16b, v28.16b
        smax    v23.16b, v23.16b, v28.16b
        smax    v22.16b, v27.16b, v28.16b
        smax    v28.16b, v29.16b, v28.16b
        cmp     x1, 7
        smin    v26.16b, v30.16b, v25.16b
        smin    v27.16b, v31.16b, v25.16b
        smin    v31.16b, v24.16b, v25.16b
        smin    v30.16b, v23.16b, v25.16b
        smin    v29.16b, v22.16b, v25.16b
        smin    v28.16b, v28.16b, v25.16b
        b.ls    10f
        ldp     x8, x11, [sp, 40]
        ext     v25.16b, v26.16b, v26.16b, 8
        ldr     x12, [sp, 120]
        ldr     x6, [sp, 136]
        str     d25, [x8]
        str     d26, [x11]
        ldp     x19, x25, [sp, 56]
        ext     v25.16b, v27.16b, v27.16b, 8
        ext     v26.16b, v31.16b, v31.16b, 8
        subs    x1, x1, 8
        str     d25, [x19]
        str     d27, [x25]
        ldp     x2, x17, [sp, 72]
        ext     v25.16b, v30.16b, v30.16b, 8
        ext     v27.16b, v29.16b, v29.16b, 8
        str     d26, [x2]
        str     d31, [x17]
        ldp     x16, x15, [sp, 88]
        ext     v26.16b, v28.16b, v28.16b, 8
        str     d25, [x16]
        str     d30, [x15]
        ldp     x14, x13, [sp, 104]
        str     d27, [x14]
        str     d29, [x13]
        str     d26, [x12]
        str     d28, [x6]
        ldp     x9, x3, [sp, 16]
        add     x8, x8, x9
        add     x11, x11, x9
        add     x19, x19, x9
        add     x25, x25, x9
        add     x2, x2, x9
        add     x17, x17, x9
        add     x16, x16, x9
        add     x15, x15, x9
        add     x14, x14, x9
        add     x13, x13, x9
        add     x12, x12, x9
        add     x6, x6, x9
        sub     x4, x4, x3
        b.ne    1b
        b       13f
10:
        ldr     x8, [sp, 136]
        ldp     x10, x9, [sp, 112]
        ldp     x12, x11, [sp, 96]
        ldp     x14, x13, [sp, 80]
        ldp     x16, x15, [sp, 64]
        ldp     x0, x17, [sp, 48]
        ldr     x2, [sp, 40]
        tbnz    w1, 2, 14f
        tbnz    w1, 1, 15f
11:
        tbz     w1, 0, 13f
12:
        st1     {v26.b}[8], [x2]
        st1     {v26.b}[0], [x0]
        st1     {v27.b}[8], [x17]
        st1     {v27.b}[0], [x16]
        st1     {v31.b}[8], [x15]
        st1     {v31.b}[0], [x14]
        st1     {v30.b}[8], [x13]
        st1     {v30.b}[0], [x12]
        st1     {v29.b}[8], [x11]
        st1     {v29.b}[0], [x10]
        st1     {v28.b}[8], [x9]
        st1     {v28.b}[0], [x8]
13:
        add     sp, sp, 0x1b0
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
14:
        st1     {v26.s}[2], [x2], 4
        st1     {v26.s}[0], [x0], 4
        st1     {v27.s}[2], [x17], 4
        st1     {v27.s}[0], [x16], 4
        st1     {v31.s}[2], [x15], 4
        st1     {v31.s}[0], [x14], 4
        st1     {v30.s}[2], [x13], 4
        st1     {v30.s}[0], [x12], 4
        st1     {v29.s}[2], [x11], 4
        st1     {v29.s}[0], [x10], 4
        st1     {v28.s}[2], [x9], 4
        st1     {v28.s}[0], [x8], 4
        ext     v26.16b, v26.16b, v26.16b, 4
        ext     v27.16b, v27.16b, v27.16b, 4
        ext     v31.16b, v31.16b, v31.16b, 4
        ext     v30.16b, v30.16b, v30.16b, 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
        tbz     w1, 1, 11b
15:
        st1     {v26.h}[4], [x2], 2
        st1     {v26.h}[0], [x0], 2
        st1     {v27.h}[4], [x17], 2
        st1     {v27.h}[0], [x16], 2
        st1     {v31.h}[4], [x15], 2
        st1     {v31.h}[0], [x14], 2
        st1     {v30.h}[4], [x13], 2
        st1     {v30.h}[0], [x12], 2
        st1     {v29.h}[4], [x11], 2
        st1     {v29.h}[0], [x10], 2
        st1     {v28.h}[4], [x9], 2
        st1     {v28.h}[0], [x8], 2
        ext     v26.16b, v26.16b, v26.16b, 2
        ext     v27.16b, v27.16b, v27.16b, 2
        ext     v31.16b, v31.16b, v31.16b, 2
        ext     v30.16b, v30.16b, v30.16b, 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 12b
        b       13b

END_FUNCTION xnn_qs8_igemm_minmax_ukernel_12x8c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
