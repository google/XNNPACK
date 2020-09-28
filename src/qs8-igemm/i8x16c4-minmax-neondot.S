// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_igemm_minmax_ukernel_8x16c4__aarch64_neondot
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
        sub     sp, sp, 0x230
        ldr     x9, [sp, 744]
        cmp     x0, 2
        sub     x14, x2, 8
        mov     x8, x2
        str     x9, [sp, 24]
        add     x9, x6, x7
        csel    x21, x6, x9, lo
        cmp     x0, 3
        lsl     x9, x14, 4
        add     x10, x21, x7
        and     x2, x9, 0xffffffffffffff80
        csel    x9, x21, x10, lo
        cmp     x0, 4
        add     x10, x9, x7
        csel    x10, x9, x10, lo
        cmp     x0, 5
        add     x11, x10, x7
        csel    x11, x10, x11, lo
        cmp     x0, 6
        add     x12, x11, x7
        csel    x12, x11, x12, lo
        ldr     x16, [sp, 736]
        ldr     x17, [sp, 728]
        ldr     x19, [sp, 720]
        cmp     x0, 7
        add     x13, x12, x7
        csel    x13, x12, x13, lo
        and     x15, x14, 0xfffffffffffffff8
        cmp     x0, 8
        add     x20, x2, 0x80
        add     x2, x13, x7
        add     x0, x15, 8
        csel    x2, x2, x13, eq
        sub     x7, x14, x15
        str     x3, [sp, 16]
1:
        ldp     q23, q13, [x5]
        ldp     q14, q12, [x5, 32]
        stp     x2, x21, [sp, 32]
        add     x5, x5, 64
        mov     x21, x3
        mov     v20.16b, v12.16b
        mov     v25.16b, v14.16b
        mov     v28.16b, v13.16b
        mov     v29.16b, v13.16b
        mov     v5.16b, v23.16b
        mov     v10.16b, v14.16b
        mov     v30.16b, v13.16b
        mov     v19.16b, v23.16b
        mov     v9.16b, v12.16b
        mov     v1.16b, v14.16b
        mov     v2.16b, v13.16b
        mov     v16.16b, v23.16b
        mov     v7.16b, v12.16b
        mov     v24.16b, v14.16b
        mov     v22.16b, v13.16b
        mov     v17.16b, v23.16b
        mov     v18.16b, v12.16b
        mov     v21.16b, v14.16b
        mov     v27.16b, v13.16b
        mov     v11.16b, v23.16b
        mov     v26.16b, v12.16b
        mov     v3.16b, v14.16b
        mov     v0.16b, v23.16b
        stp     x1, x6, [sp, 48]
        str     q23, [sp, 144]
        stp     q12, q14, [sp, 208]
        str     q12, [sp, 240]
        str     q13, [sp, 512]
2:
        ldp     x14, x15, [x4]
        ldp     x1, x2, [x4, 16]
        ldp     x3, x6, [x4, 32]
        ldp     x22, x26, [x4, 48]
        add     x23, x14, x17
        cmp     x14, x16
        csel    x25, x14, x23, eq
        add     x14, x15, x17
        cmp     x15, x16
        csel    x30, x15, x14, eq
        add     x14, x1, x17
        cmp     x1, x16
        csel    x24, x1, x14, eq
        add     x14, x2, x17
        cmp     x2, x16
        csel    x28, x2, x14, eq
        add     x14, x3, x17
        cmp     x3, x16
        csel    x23, x3, x14, eq
        add     x14, x6, x17
        cmp     x6, x16
        csel    x27, x6, x14, eq
        add     x14, x22, x17
        cmp     x22, x16
        csel    x22, x22, x14, eq
        add     x14, x26, x17
        cmp     x26, x16
        csel    x26, x26, x14, eq
        cmp     x8, 8
        b.cc    5f
        ldr     q15, [sp, 144]
        ldr     q31, [sp, 512]
        mov     x1, xzr
        add     x29, x30, x0
        add     x15, x28, x0
        add     x3, x27, x0
        add     x14, x26, x0
        add     x6, x5, 64
        mov     x2, x8
        stp     q1, q7, [sp, 368]
        mov     v7.16b, v16.16b
        mov     v4.16b, v19.16b
        mov     v19.16b, v23.16b
        mov     v8.16b, v20.16b
        stp     q18, q24, [sp, 464]
        stp     q22, q0, [sp, 432]
        stp     q25, q20, [sp, 320]
        stp     q12, q14, [sp, 288]
        str     q21, [sp, 352]
        stp     q9, q11, [sp, 256]
        stp     q10, q26, [sp, 400]
        stp     q27, q13, [sp, 528]
        str     q3, [sp, 496]
3:
        ldr     d24, [x25, x1]
        ldp     q20, q21, [x6, -64]
        ldr     q18, [sp, 544]
        ldr     d26, [x23, x1]
        ldr     d9, [x27, x1]
        ldr     d6, [x26, x1]
        ldp     q23, q22, [x6, -32]
        sdot    v18.4s, v21.16b, v24.4b[0]
        ldp     q27, q3, [sp, 208]
        ldr     d25, [x30, x1]
        str     q18, [sp, 544]
        ldr     q18, [sp, 528]
        ldr     d16, [x24, x1]
        ldr     d14, [x28, x1]
        ldr     d0, [x22, x1]
        ldp     q1, q8, [sp, 432]
        ldr     q11, [sp, 272]
        sdot    v30.4s, v21.16b, v26.4b[0]
        sdot    v5.4s, v20.16b, v9.4b[0]
        sdot    v29.4s, v21.16b, v9.4b[0]
        sdot    v19.4s, v20.16b, v6.4b[0]
        sdot    v31.4s, v21.16b, v6.4b[0]
        stp     q5, q29, [sp, 144]
        stp     q30, q19, [sp, 112]
        str     q31, [sp, 512]
        ldp     q31, q29, [sp, 400]
        ldp     q5, q30, [sp, 240]
        stp     q26, q9, [sp, 64]
        sdot    v3.4s, v23.16b, v9.4b[0]
        sdot    v27.4s, v22.16b, v9.4b[0]
        ldp     q19, q9, [sp, 320]
        sdot    v18.4s, v21.16b, v25.4b[0]
        str     q18, [sp, 528]
        mov     v10.16b, v7.16b
        mov     v13.16b, v2.16b
        mov     v18.16b, v4.16b
        mov     v12.16b, v15.16b
        sdot    v8.4s, v20.16b, v24.4b[0]
        sdot    v11.4s, v20.16b, v25.4b[0]
        sdot    v17.4s, v20.16b, v16.4b[0]
        sdot    v1.4s, v21.16b, v16.4b[0]
        sdot    v10.4s, v20.16b, v14.4b[0]
        sdot    v13.4s, v21.16b, v14.4b[0]
        sdot    v18.4s, v20.16b, v26.4b[0]
        sdot    v12.4s, v20.16b, v0.4b[0]
        sdot    v28.4s, v21.16b, v0.4b[0]
        ldp     q21, q20, [x6]
        stp     q28, q17, [sp, 176]
        ldp     q2, q28, [sp, 480]
        ldp     q4, q15, [sp, 352]
        ldr     q17, [sp, 464]
        ldr     q7, [sp, 384]
        sdot    v31.4s, v23.16b, v26.4b[0]
        sdot    v5.4s, v22.16b, v26.4b[0]
        str     q0, [sp, 96]
        sdot    v19.4s, v23.16b, v0.4b[0]
        sdot    v9.4s, v22.16b, v0.4b[0]
        ldp     q26, q0, [sp, 288]
        sdot    v8.4s, v21.16b, v24.4b[1]
        sdot    v28.4s, v23.16b, v24.4b[0]
        sdot    v29.4s, v22.16b, v24.4b[0]
        sdot    v4.4s, v23.16b, v25.4b[0]
        sdot    v17.4s, v22.16b, v25.4b[0]
        sdot    v2.4s, v23.16b, v16.4b[0]
        sdot    v7.4s, v22.16b, v16.4b[0]
        sdot    v15.4s, v23.16b, v14.4b[0]
        sdot    v30.4s, v22.16b, v14.4b[0]
        sdot    v0.4s, v23.16b, v6.4b[0]
        sdot    v26.4s, v22.16b, v6.4b[0]
        ldp     q23, q22, [x6, 32]
        str     q8, [sp, 448]
        ldr     q8, [sp, 544]
        sdot    v11.4s, v21.16b, v25.4b[1]
        sdot    v28.4s, v23.16b, v24.4b[1]
        sdot    v29.4s, v22.16b, v24.4b[1]
        sdot    v8.4s, v20.16b, v24.4b[1]
        ldr     q24, [sp, 528]
        sdot    v4.4s, v23.16b, v25.4b[1]
        sdot    v17.4s, v22.16b, v25.4b[1]
        str     q28, [sp, 496]
        sdot    v24.4s, v20.16b, v25.4b[1]
        ldr     q25, [sp, 64]
        str     q29, [sp, 416]
        ldp     q29, q28, [sp, 160]
        sdot    v30.4s, v22.16b, v14.4b[1]
        sdot    v31.4s, v23.16b, v25.4b[1]
        sdot    v5.4s, v22.16b, v25.4b[1]
        str     q31, [sp, 400]
        str     q5, [sp, 240]
        ldr     q5, [sp, 144]
        ldr     q31, [sp, 80]
        str     q17, [sp, 464]
        ldr     q17, [sp, 192]
        str     q30, [sp, 256]
        sdot    v5.4s, v21.16b, v31.4b[1]
        sdot    v29.4s, v20.16b, v31.4b[1]
        sdot    v3.4s, v23.16b, v31.4b[1]
        sdot    v27.4s, v22.16b, v31.4b[1]
        ldr     q31, [sp, 96]
        ldr     q30, [sp, 112]
        sdot    v2.4s, v23.16b, v16.4b[1]
        sdot    v7.4s, v22.16b, v16.4b[1]
        sdot    v19.4s, v23.16b, v31.4b[1]
        sdot    v12.4s, v21.16b, v31.4b[1]
        sdot    v28.4s, v20.16b, v31.4b[1]
        str     q19, [sp, 320]
        ldr     q19, [sp, 128]
        sdot    v9.4s, v22.16b, v31.4b[1]
        ldr     q31, [sp, 512]
        sdot    v10.4s, v21.16b, v14.4b[1]
        sdot    v13.4s, v20.16b, v14.4b[1]
        sdot    v15.4s, v23.16b, v14.4b[1]
        sdot    v18.4s, v21.16b, v25.4b[1]
        sub     x2, x2, 8
        sdot    v17.4s, v21.16b, v16.4b[1]
        sdot    v1.4s, v20.16b, v16.4b[1]
        str     q2, [sp, 480]
        str     q7, [sp, 384]
        mov     v7.16b, v10.16b
        mov     v2.16b, v13.16b
        stp     q4, q15, [sp, 352]
        mov     v4.16b, v18.16b
        sdot    v30.4s, v20.16b, v25.4b[1]
        mov     v15.16b, v12.16b
        sdot    v19.4s, v21.16b, v6.4b[1]
        sdot    v31.4s, v20.16b, v6.4b[1]
        sdot    v0.4s, v23.16b, v6.4b[1]
        sdot    v26.4s, v22.16b, v6.4b[1]
        add     x1, x1, 8
        cmp     x2, 7
        add     x6, x6, 0x80
        str     q11, [sp, 272]
        stp     q24, q8, [sp, 528]
        str     q1, [sp, 432]
        stp     q27, q3, [sp, 208]
        str     q9, [sp, 336]
        stp     q26, q0, [sp, 288]
        b.hi    3b
        ldp     q12, q14, [sp, 288]
        ldp     q25, q20, [sp, 320]
        ldp     q9, q11, [sp, 256]
        ldp     q21, q1, [sp, 352]
        mov     v16.16b, v7.16b
        ldp     q7, q10, [sp, 384]
        ldp     q18, q24, [sp, 464]
        ldp     q26, q22, [sp, 416]
        ldp     q27, q13, [sp, 528]
        ldr     q3, [sp, 496]
        ldr     q0, [sp, 448]
        add     x25, x25, x0
        add     x24, x24, x0
        add     x23, x23, x0
        add     x22, x22, x0
        add     x5, x5, x20
        mov     x1, x7
        mov     v23.16b, v19.16b
        mov     v19.16b, v4.16b
        str     q31, [sp, 512]
        str     q15, [sp, 144]
        add     x4, x4, 64
        cbnz    x1, 6f
4:
        subs    x21, x21, 64
        b.ne    2b
        b       8f
5:
        mov     x1, x8
        mov     x14, x26
        mov     x3, x27
        mov     x15, x28
        mov     x29, x30
        add     x4, x4, 64
        cbz     x1, 4b
6:
        ldr     d8, [x25]
        ldr     d31, [x29]
        stp     q18, q24, [sp, 464]
        stp     q10, q26, [sp, 400]
        ldr     d6, [x24]
        ldp     q26, q24, [x5]
        str     q3, [sp, 496]
        ldr     d4, [x15]
        mov     v18.16b, v28.16b
        sdot    v0.4s, v26.16b, v8.4b[0]
        sdot    v22.4s, v24.16b, v6.4b[0]
        ldr     d15, [x23]
        ldr     d10, [x3]
        ldr     d3, [x22]
        ldr     d28, [x14]
        stp     q22, q0, [sp, 432]
        ldr     q0, [sp, 144]
        sdot    v13.4s, v24.16b, v8.4b[0]
        sdot    v27.4s, v24.16b, v31.4b[0]
        sdot    v29.4s, v24.16b, v10.4b[0]
        sdot    v0.4s, v26.16b, v3.4b[0]
        sdot    v11.4s, v26.16b, v31.4b[0]
        stp     q27, q13, [sp, 528]
        sdot    v17.4s, v26.16b, v6.4b[0]
        sdot    v16.4s, v26.16b, v4.4b[0]
        sdot    v19.4s, v26.16b, v15.4b[0]
        sdot    v5.4s, v26.16b, v10.4b[0]
        stp     q0, q29, [sp, 144]
        sdot    v23.4s, v26.16b, v28.4b[0]
        ldr     q26, [x5, 32]
        ldp     q29, q13, [sp, 496]
        sdot    v18.4s, v24.16b, v3.4b[0]
        stp     q18, q17, [sp, 176]
        ldr     q22, [sp, 400]
        sdot    v29.4s, v26.16b, v8.4b[0]
        str     q29, [sp, 496]
        ldp     q27, q29, [sp, 464]
        ldp     q18, q0, [sp, 224]
        sdot    v2.4s, v24.16b, v4.4b[0]
        sdot    v30.4s, v24.16b, v15.4b[0]
        sdot    v13.4s, v24.16b, v28.4b[0]
        ldr     q24, [x5, 48]
        sdot    v21.4s, v26.16b, v31.4b[0]
        sdot    v29.4s, v26.16b, v6.4b[0]
        sdot    v1.4s, v26.16b, v4.4b[0]
        sdot    v22.4s, v26.16b, v15.4b[0]
        sdot    v18.4s, v26.16b, v10.4b[0]
        sdot    v25.4s, v26.16b, v3.4b[0]
        sdot    v14.4s, v26.16b, v28.4b[0]
        ldr     q26, [sp, 416]
        ldr     q17, [sp, 208]
        cmp     x1, 5
        sdot    v27.4s, v24.16b, v31.4b[0]
        sdot    v26.4s, v24.16b, v8.4b[0]
        sdot    v7.4s, v24.16b, v6.4b[0]
        sdot    v9.4s, v24.16b, v4.4b[0]
        sdot    v0.4s, v24.16b, v15.4b[0]
        sdot    v17.4s, v24.16b, v10.4b[0]
        sdot    v20.4s, v24.16b, v3.4b[0]
        sdot    v12.4s, v24.16b, v28.4b[0]
        str     q29, [sp, 480]
        stp     q10, q28, [sp, 400]
        str     q3, [sp, 512]
        b.cc    7f
        mov     v28.16b, v30.16b
        mov     v30.16b, v21.16b
        ldp     q21, q24, [x5, 64]
        ldr     q29, [sp, 448]
        mov     v3.16b, v11.16b
        mov     v10.16b, v26.16b
        ldp     q11, q26, [x5, 96]
        sdot    v29.4s, v21.16b, v8.4b[1]
        str     q29, [sp, 448]
        ldr     q29, [sp, 544]
        sdot    v10.4s, v26.16b, v8.4b[1]
        sdot    v3.4s, v21.16b, v31.4b[1]
        sdot    v30.4s, v11.16b, v31.4b[1]
        sdot    v29.4s, v24.16b, v8.4b[1]
        str     q29, [sp, 544]
        ldr     q29, [sp, 496]
        sdot    v27.4s, v26.16b, v31.4b[1]
        sdot    v28.4s, v24.16b, v15.4b[1]
        sdot    v22.4s, v11.16b, v15.4b[1]
        sdot    v29.4s, v11.16b, v8.4b[1]
        str     q29, [sp, 496]
        ldr     q29, [sp, 528]
        ldr     q8, [sp, 480]
        sdot    v7.4s, v26.16b, v6.4b[1]
        sdot    v16.4s, v21.16b, v4.4b[1]
        sdot    v29.4s, v24.16b, v31.4b[1]
        str     q29, [sp, 528]
        ldr     q29, [sp, 192]
        ldr     q31, [sp, 432]
        sdot    v8.4s, v11.16b, v6.4b[1]
        sdot    v2.4s, v24.16b, v4.4b[1]
        sdot    v29.4s, v21.16b, v6.4b[1]
        str     q29, [sp, 192]
        sdot    v31.4s, v24.16b, v6.4b[1]
        sdot    v1.4s, v11.16b, v4.4b[1]
        sdot    v9.4s, v26.16b, v4.4b[1]
        mov     v4.16b, v28.16b
        mov     v6.16b, v22.16b
        ldp     q28, q22, [sp, 400]
        ldr     q29, [sp, 160]
        str     q27, [sp, 464]
        ldr     q27, [sp, 512]
        sdot    v5.4s, v21.16b, v28.4b[1]
        sdot    v29.4s, v24.16b, v28.4b[1]
        sdot    v18.4s, v11.16b, v28.4b[1]
        sdot    v17.4s, v26.16b, v28.4b[1]
        ldr     q28, [sp, 144]
        str     q17, [sp, 208]
        sdot    v0.4s, v26.16b, v15.4b[1]
        sdot    v13.4s, v24.16b, v22.4b[1]
        sdot    v28.4s, v21.16b, v27.4b[1]
        str     q28, [sp, 144]
        ldp     q28, q17, [sp, 176]
        stp     q18, q0, [sp, 224]
        sdot    v25.4s, v11.16b, v27.4b[1]
        sdot    v20.4s, v26.16b, v27.4b[1]
        sdot    v28.4s, v24.16b, v27.4b[1]
        str     q13, [sp, 512]
        sdot    v14.4s, v11.16b, v22.4b[1]
        mov     v11.16b, v3.16b
        ldp     q0, q18, [sp, 448]
        ldp     q27, q13, [sp, 528]
        ldr     q3, [sp, 496]
        sdot    v19.4s, v21.16b, v15.4b[1]
        sdot    v23.4s, v21.16b, v22.4b[1]
        mov     v21.16b, v30.16b
        sdot    v12.4s, v26.16b, v22.4b[1]
        mov     v26.16b, v10.16b
        add     x5, x5, 0x80
        mov     v10.16b, v6.16b
        mov     v30.16b, v4.16b
        mov     v24.16b, v8.16b
        mov     v22.16b, v31.16b
        subs    x21, x21, 64
        b.ne    2b
        b       8f
7:
        stp     q17, q18, [sp, 208]
        str     q13, [sp, 512]
        str     q0, [sp, 240]
        ldp     q29, q28, [sp, 160]
        mov     v10.16b, v22.16b
        ldp     q24, q3, [sp, 480]
        ldp     q22, q0, [sp, 432]
        ldr     q17, [sp, 192]
        mov     v18.16b, v27.16b
        ldp     q27, q13, [sp, 528]
        add     x5, x5, 64
        subs    x21, x21, 64
        b.ne    2b
8:
        ldr     x15, [sp, 24]
        mov     v6.16b, v28.16b
        mov     x14, x15
        ld1r    {v28.4s}, [x14], 4
        sqrdmulh        v31.4s, v3.4s, v28.4s
        str     q31, [sp, 544]
        sqrdmulh        v31.4s, v26.4s, v28.4s
        str     q31, [sp, 416]
        sqrdmulh        v31.4s, v27.4s, v28.4s
        str     q31, [sp, 528]
        sqrdmulh        v31.4s, v21.4s, v28.4s
        sqrdmulh        v21.4s, v29.4s, v28.4s
        ldr     q29, [sp, 224]
        str     q31, [sp, 432]
        sqrdmulh        v31.4s, v17.4s, v28.4s
        sqrdmulh        v17.4s, v22.4s, v28.4s
        mov     v22.16b, v5.16b
        mov     v27.16b, v20.16b
        sqrdmulh        v20.4s, v22.4s, v28.4s
        sqrdmulh        v22.4s, v29.4s, v28.4s
        ldr     q29, [sp, 208]
        sqrdmulh        v15.4s, v13.4s, v28.4s
        mov     v13.16b, v6.16b
        mov     v6.16b, v23.16b
        sqrdmulh        v23.4s, v29.4s, v28.4s
        sqrdmulh        v29.4s, v6.4s, v28.4s
        str     q31, [sp, 448]
        sqrdmulh        v31.4s, v24.4s, v28.4s
        ldr     q24, [sp, 144]
        mov     v26.16b, v25.16b
        ldr     q25, [sp, 240]
        str     q29, [sp, 464]
        ld1r    {v6.4s}, [x14]
        ldr     q29, [sp, 512]
        sqrdmulh        v0.4s, v0.4s, v28.4s
        sqrdmulh        v11.4s, v11.4s, v28.4s
        sqrdmulh        v8.4s, v18.4s, v28.4s
        sqrdmulh        v5.4s, v7.4s, v28.4s
        sqrdmulh        v3.4s, v16.4s, v28.4s
        sqrdmulh        v2.4s, v2.4s, v28.4s
        sqrdmulh        v1.4s, v1.4s, v28.4s
        sqrdmulh        v9.4s, v9.4s, v28.4s
        sqrdmulh        v4.4s, v19.4s, v28.4s
        sqrdmulh        v18.4s, v30.4s, v28.4s
        sqrdmulh        v16.4s, v10.4s, v28.4s
        sqrdmulh        v19.4s, v25.4s, v28.4s
        sqrdmulh        v24.4s, v24.4s, v28.4s
        sqrdmulh        v25.4s, v13.4s, v28.4s
        sqrdmulh        v26.4s, v26.4s, v28.4s
        sqrdmulh        v10.4s, v27.4s, v28.4s
        sqrdmulh        v27.4s, v29.4s, v28.4s
        sqrdmulh        v7.4s, v14.4s, v28.4s
        sqrdmulh        v30.4s, v12.4s, v28.4s
        cmeq    v28.4s, v6.4s, 0
        mvn     v28.16b, v28.16b
        ldr     q29, [sp, 544]
        str     q31, [sp, 496]
        and     v31.16b, v0.16b, v28.16b
        ssra    v0.4s, v31.4s, 31
        and     v31.16b, v15.16b, v28.16b
        ssra    v15.4s, v31.4s, 31
        str     q15, [sp, 480]
        and     v31.16b, v29.16b, v28.16b
        ldp     q12, q15, [sp, 416]
        ssra    v29.4s, v31.4s, 31
        str     q29, [sp, 544]
        ldr     q29, [sp, 528]
        and     v31.16b, v12.16b, v28.16b
        ssra    v12.4s, v31.4s, 31
        and     v31.16b, v11.16b, v28.16b
        ssra    v11.4s, v31.4s, 31
        and     v31.16b, v29.16b, v28.16b
        ldr     q14, [sp, 448]
        ssra    v29.4s, v31.4s, 31
        stp     q0, q29, [sp, 512]
        and     v31.16b, v15.16b, v28.16b
        ldr     q29, [sp, 496]
        ssra    v15.4s, v31.4s, 31
        and     v31.16b, v8.16b, v28.16b
        ssra    v8.4s, v31.4s, 31
        and     v31.16b, v14.16b, v28.16b
        ssra    v14.4s, v31.4s, 31
        and     v31.16b, v17.16b, v28.16b
        ssra    v17.4s, v31.4s, 31
        and     v31.16b, v29.16b, v28.16b
        ssra    v29.4s, v31.4s, 31
        and     v31.16b, v5.16b, v28.16b
        ssra    v5.4s, v31.4s, 31
        and     v31.16b, v3.16b, v28.16b
        ssra    v3.4s, v31.4s, 31
        and     v31.16b, v2.16b, v28.16b
        ssra    v2.4s, v31.4s, 31
        and     v31.16b, v1.16b, v28.16b
        ssra    v1.4s, v31.4s, 31
        and     v31.16b, v9.16b, v28.16b
        ssra    v9.4s, v31.4s, 31
        and     v31.16b, v4.16b, v28.16b
        ssra    v4.4s, v31.4s, 31
        and     v31.16b, v18.16b, v28.16b
        ssra    v18.4s, v31.4s, 31
        and     v31.16b, v16.16b, v28.16b
        ssra    v16.4s, v31.4s, 31
        and     v31.16b, v19.16b, v28.16b
        ssra    v19.4s, v31.4s, 31
        and     v31.16b, v20.16b, v28.16b
        ssra    v20.4s, v31.4s, 31
        and     v31.16b, v21.16b, v28.16b
        ssra    v21.4s, v31.4s, 31
        and     v31.16b, v22.16b, v28.16b
        ssra    v22.4s, v31.4s, 31
        and     v31.16b, v23.16b, v28.16b
        str     q29, [sp, 496]
        ssra    v23.4s, v31.4s, 31
        and     v31.16b, v24.16b, v28.16b
        ldr     q29, [sp, 464]
        ssra    v24.4s, v31.4s, 31
        and     v31.16b, v25.16b, v28.16b
        ssra    v25.4s, v31.4s, 31
        and     v31.16b, v26.16b, v28.16b
        srshl   v13.4s, v15.4s, v6.4s
        srshl   v15.4s, v14.4s, v6.4s
        srshl   v14.4s, v17.4s, v6.4s
        ldr     q17, [sp, 496]
        ssra    v26.4s, v31.4s, 31
        and     v31.16b, v10.16b, v28.16b
        ssra    v10.4s, v31.4s, 31
        and     v31.16b, v29.16b, v28.16b
        ssra    v29.4s, v31.4s, 31
        and     v31.16b, v27.16b, v28.16b
        ldr     q0, [sp, 544]
        str     q5, [sp, 448]
        ssra    v27.4s, v31.4s, 31
        and     v31.16b, v7.16b, v28.16b
        srshl   v17.4s, v17.4s, v6.4s
        ssra    v7.4s, v31.4s, 31
        and     v28.16b, v30.16b, v28.16b
        ldr     q31, [sp, 480]
        str     q17, [sp, 480]
        ldr     q17, [sp, 448]
        ssra    v30.4s, v28.4s, 31
        ldr     q28, [sp, 512]
        srshl   v5.4s, v0.4s, v6.4s
        ldr     q0, [sp, 528]
        srshl   v17.4s, v17.4s, v6.4s
        srshl   v28.4s, v28.4s, v6.4s
        str     q17, [sp, 496]
        srshl   v17.4s, v4.4s, v6.4s
        srshl   v20.4s, v20.4s, v6.4s
        srshl   v29.4s, v29.4s, v6.4s
        srshl   v31.4s, v31.4s, v6.4s
        srshl   v12.4s, v12.4s, v6.4s
        srshl   v11.4s, v11.4s, v6.4s
        srshl   v0.4s, v0.4s, v6.4s
        srshl   v8.4s, v8.4s, v6.4s
        srshl   v3.4s, v3.4s, v6.4s
        srshl   v2.4s, v2.4s, v6.4s
        srshl   v1.4s, v1.4s, v6.4s
        srshl   v9.4s, v9.4s, v6.4s
        srshl   v18.4s, v18.4s, v6.4s
        srshl   v4.4s, v16.4s, v6.4s
        srshl   v19.4s, v19.4s, v6.4s
        stp     q17, q20, [sp, 512]
        srshl   v21.4s, v21.4s, v6.4s
        srshl   v22.4s, v22.4s, v6.4s
        srshl   v23.4s, v23.4s, v6.4s
        srshl   v17.4s, v24.4s, v6.4s
        srshl   v25.4s, v25.4s, v6.4s
        srshl   v26.4s, v26.4s, v6.4s
        srshl   v10.4s, v10.4s, v6.4s
        str     q29, [sp, 544]
        srshl   v29.4s, v27.4s, v6.4s
        srshl   v7.4s, v7.4s, v6.4s
        srshl   v6.4s, v30.4s, v6.4s
        add     x14, x15, 8
        sqxtn   v30.4h, v28.4s
        ld1r    {v28.8h}, [x14]
        sqxtn2  v30.8h, v31.4s
        sqxtn   v24.4h, v5.4s
        sqxtn   v20.4h, v11.4s
        sqxtn   v16.4h, v13.4s
        sqxtn   v31.4h, v15.4s
        sqxtn2  v24.8h, v12.4s
        sqxtn2  v20.8h, v0.4s
        sqxtn2  v16.8h, v8.4s
        sqxtn2  v31.8h, v14.4s
        ldr     q27, [sp, 480]
        sqxtn   v3.4h, v3.4s
        sqxtn   v1.4h, v1.4s
        sqxtn   v22.4h, v22.4s
        sqxtn   v13.4h, v27.4s
        ldr     q27, [sp, 496]
        sqxtn   v26.4h, v26.4s
        sqadd   v30.8h, v30.8h, v28.8h
        sqadd   v31.8h, v31.8h, v28.8h
        sqxtn2  v13.8h, v27.4s
        sqxtn2  v3.8h, v2.4s
        sqxtn2  v1.8h, v9.4s
        ldr     q27, [sp, 512]
        sqxtn   v31.8b, v31.8h
        add     x14, x15, 10
        sqxtn   v0.4h, v27.4s
        sqxtn2  v0.8h, v18.4s
        sqxtn   v18.4h, v4.4s
        sqxtn2  v18.8h, v19.4s
        ldr     q27, [sp, 528]
        sqadd   v0.8h, v0.8h, v28.8h
        sqadd   v18.8h, v18.8h, v28.8h
        sqxtn   v19.4h, v27.4s
        sqxtn2  v19.8h, v21.4s
        sqxtn2  v22.8h, v23.4s
        sqxtn   v23.4h, v17.4s
        sqxtn2  v23.8h, v25.4s
        sqxtn2  v26.8h, v10.4s
        ldr     q27, [sp, 544]
        sqadd   v25.8h, v24.8h, v28.8h
        sqadd   v24.8h, v20.8h, v28.8h
        sqadd   v21.8h, v16.8h, v28.8h
        sqxtn   v27.4h, v27.4s
        sqxtn2  v27.8h, v29.4s
        sqxtn   v29.4h, v7.4s
        sqxtn2  v29.8h, v6.4s
        sqadd   v20.8h, v13.8h, v28.8h
        sqadd   v17.8h, v3.8h, v28.8h
        sqadd   v16.8h, v1.8h, v28.8h
        sqadd   v19.8h, v19.8h, v28.8h
        sqadd   v22.8h, v22.8h, v28.8h
        sqadd   v23.8h, v23.8h, v28.8h
        sqadd   v26.8h, v26.8h, v28.8h
        sqadd   v27.8h, v27.8h, v28.8h
        sqadd   v28.8h, v29.8h, v28.8h
        sqxtn   v29.8b, v30.8h
        sqxtn   v30.8b, v24.8h
        sqxtn2  v29.16b, v25.8h
        sqxtn2  v30.16b, v21.8h
        sqxtn   v25.8b, v17.8h
        sqxtn   v24.8b, v0.8h
        sqxtn   v21.8b, v19.8h
        sqxtn   v23.8b, v23.8h
        sqxtn   v27.8b, v27.8h
        sqxtn2  v31.16b, v20.8h
        sqxtn2  v25.16b, v16.8h
        sqxtn2  v24.16b, v18.8h
        sqxtn2  v21.16b, v22.8h
        sqxtn2  v23.16b, v26.8h
        sqxtn2  v27.16b, v28.8h
        ld1r    {v28.16b}, [x14]
        add     x14, x15, 11
        ld1r    {v22.16b}, [x14]
        ldr     x1, [sp, 48]
        smax    v27.16b, v27.16b, v28.16b
        smax    v26.16b, v23.16b, v28.16b
        smax    v23.16b, v21.16b, v28.16b
        smax    v21.16b, v24.16b, v28.16b
        smax    v20.16b, v25.16b, v28.16b
        smax    v19.16b, v31.16b, v28.16b
        smax    v18.16b, v30.16b, v28.16b
        smax    v28.16b, v29.16b, v28.16b
        cmp     x1, 0xf
        smin    v24.16b, v27.16b, v22.16b
        smin    v31.16b, v26.16b, v22.16b
        smin    v25.16b, v23.16b, v22.16b
        smin    v30.16b, v21.16b, v22.16b
        smin    v26.16b, v20.16b, v22.16b
        smin    v29.16b, v19.16b, v22.16b
        smin    v27.16b, v18.16b, v22.16b
        smin    v28.16b, v28.16b, v22.16b
        b.ls    9f
        ldp     x2, x21, [sp, 32]
        ldr     x6, [sp, 56]
        ldr     x3, [sp, 16]
        subs    x1, x1, 0x10
        str     q24, [x2]
        str     q31, [x13]
        str     q25, [x12]
        str     q30, [x11]
        str     q26, [x10]
        str     q29, [x9]
        str     q27, [x21]
        str     q28, [x6]
        add     x2, x2, x19
        add     x13, x13, x19
        add     x12, x12, x19
        add     x11, x11, x19
        add     x10, x10, x19
        add     x9, x9, x19
        add     x21, x21, x19
        add     x6, x6, x19
        sub     x4, x4, x3
        b.ne    1b
        b       14f
9:
        ldr     x8, [sp, 56]
        ldp     x15, x14, [sp, 32]
        tbnz    w1, 3, 10f
        mov     v28.d[1], v27.d[0]
        mov     v29.d[1], v26.d[0]
        mov     v30.d[1], v25.d[0]
        mov     v31.d[1], v24.d[0]
        tbnz    w1, 2, 11f
        b       12f
10:
        str     d24, [x15], 8
        str     d31, [x13], 8
        str     d25, [x12], 8
        str     d30, [x11], 8
        str     d26, [x10], 8
        str     d29, [x9], 8
        str     d27, [x14], 8
        str     d28, [x8], 8
        ext     v31.16b, v31.16b, v31.16b, 8
        ext     v24.16b, v24.16b, v24.16b, 8
        ext     v30.16b, v30.16b, v30.16b, 8
        ext     v25.16b, v25.16b, v25.16b, 8
        ext     v29.16b, v29.16b, v29.16b, 8
        ext     v26.16b, v26.16b, v26.16b, 8
        ext     v28.16b, v28.16b, v28.16b, 8
        ext     v27.16b, v27.16b, v27.16b, 8
        mov     v31.d[1], v24.d[0]
        mov     v30.d[1], v25.d[0]
        mov     v29.d[1], v26.d[0]
        mov     v28.d[1], v27.d[0]
        tbz     w1, 2, 12f
11:
        st1     {v31.s}[2], [x15], 4
        st1     {v31.s}[0], [x13], 4
        st1     {v30.s}[2], [x12], 4
        st1     {v30.s}[0], [x11], 4
        st1     {v29.s}[2], [x10], 4
        st1     {v29.s}[0], [x9], 4
        st1     {v28.s}[2], [x14], 4
        st1     {v28.s}[0], [x8], 4
        ext     v31.16b, v31.16b, v31.16b, 4
        ext     v30.16b, v30.16b, v30.16b, 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
12:
        tbnz    w1, 1, 15f
        tbz     w1, 0, 14f
13:
        st1     {v31.b}[8], [x15]
        st1     {v31.b}[0], [x13]
        st1     {v30.b}[8], [x12]
        st1     {v30.b}[0], [x11]
        st1     {v29.b}[8], [x10]
        st1     {v29.b}[0], [x9]
        st1     {v28.b}[8], [x14]
        st1     {v28.b}[0], [x8]
14:
        add     sp, sp, 0x230
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
15:
        st1     {v31.h}[4], [x15], 2
        st1     {v31.h}[0], [x13], 2
        st1     {v30.h}[4], [x12], 2
        st1     {v30.h}[0], [x11], 2
        st1     {v29.h}[4], [x10], 2
        st1     {v29.h}[0], [x9], 2
        st1     {v28.h}[4], [x14], 2
        st1     {v28.h}[0], [x8], 2
        ext     v31.16b, v31.16b, v31.16b, 2
        ext     v30.16b, v30.16b, v30.16b, 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 13b
        b       14b

END_FUNCTION xnn_qs8_igemm_minmax_ukernel_8x16c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
