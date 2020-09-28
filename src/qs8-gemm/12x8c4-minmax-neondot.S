// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_gemm_minmax_ukernel_12x8c4__aarch64_neondot
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
        sub     sp, sp, 0x1c0
        add     x9, x3, x4
        add     x8, x6, x7
        cmp     x0, 2
        csel    x8, x6, x8, lo
        csel    x27, x3, x9, lo
        cmp     x0, 3
        add     x10, x27, x4
        add     x9, x8, x7
        csel    x9, x8, x9, lo
        csel    x19, x27, x10, lo
        cmp     x0, 4
        add     x11, x19, x4
        add     x10, x9, x7
        csel    x10, x9, x10, lo
        csel    x28, x19, x11, lo
        cmp     x0, 5
        add     x12, x28, x4
        add     x11, x10, x7
        csel    x11, x10, x11, lo
        csel    x20, x28, x12, lo
        cmp     x0, 6
        add     x13, x20, x4
        add     x12, x11, x7
        csel    x12, x11, x12, lo
        csel    x29, x20, x13, lo
        cmp     x0, 7
        add     x14, x29, x4
        add     x13, x12, x7
        csel    x13, x12, x13, lo
        csel    x21, x29, x14, lo
        cmp     x0, 8
        add     x15, x21, x4
        add     x14, x13, x7
        csel    x14, x13, x14, lo
        csel    x30, x21, x15, lo
        cmp     x0, 9
        add     x16, x30, x4
        add     x15, x14, x7
        csel    x15, x14, x15, lo
        csel    x22, x30, x16, lo
        cmp     x0, 10
        add     x17, x22, x4
        add     x16, x15, x7
        csel    x16, x15, x16, lo
        csel    x26, x22, x17, lo
        str     x1, [sp, 128]
        sub     x1, x2, 8
        cmp     x0, 11
        add     x23, x26, x4
        add     x17, x16, x7
        str     x5, [sp, 152]
        mov     x5, x2
        lsl     x2, x1, 3
        csel    x17, x16, x17, lo
        csel    x23, x26, x23, lo
        cmp     x0, 12
        and     x0, x1, 0xfffffffffffffff8
        and     x2, x2, 0xffffffffffffffc0
        sub     x1, x1, x0
        add     x25, x0, 8
        add     x0, x23, x4
        str     x1, [sp, 24]
        add     x1, x17, x7
        ldr     x4, [sp, 616]
        ldr     x7, [sp, 608]
        csel    x24, x0, x23, eq
        add     x0, x2, 64
        ldr     x2, [sp, 128]
        csel    x1, x1, x17, eq
        str     x4, [sp, 40]
        str     x0, [sp, 16]
1:
        ldr     x0, [sp, 152]
        cmp     x5, 8
        ldp     q7, q20, [x0], 32
        b.cc    3f
        str     x2, [sp, 128]
        add     x2, x27, x25
        str     x2, [sp, 112]
        add     x2, x28, x25
        str     x2, [sp, 96]
        add     x2, x29, x25
        str     x2, [sp, 80]
        add     x2, x30, x25
        str     x2, [sp, 64]
        add     x2, x26, x25
        str     x2, [sp, 48]
        add     x2, x24, x25
        mov     x4, xzr
        str     x2, [sp, 32]
        str     x0, [sp, 152]
        add     x0, x0, 32
        mov     v8.16b, v7.16b
        mov     v21.16b, v7.16b
        mov     v11.16b, v7.16b
        mov     v13.16b, v7.16b
        mov     v3.16b, v7.16b
        mov     v1.16b, v7.16b
        mov     v16.16b, v7.16b
        mov     v18.16b, v7.16b
        mov     v12.16b, v7.16b
        mov     v25.16b, v20.16b
        mov     v30.16b, v7.16b
        mov     v22.16b, v20.16b
        mov     x2, x5
        mov     v14.16b, v7.16b
        stp     q20, q7, [sp, 368]
        stp     q20, q20, [sp, 336]
        stp     q20, q20, [sp, 304]
        stp     q20, q20, [sp, 400]
        stp     q20, q20, [sp, 272]
2:
        ldr     d24, [x21, x4]
        ldr     d5, [x26, x4]
        ldp     q27, q26, [x0, -32]
        str     q30, [sp, 432]
        ldr     d28, [x28, x4]
        ldr     d7, [x20, x4]
        ldr     d6, [x29, x4]
        ldr     d19, [x30, x4]
        sdot    v3.4s, v27.16b, v24.4b[0]
        sdot    v18.4s, v27.16b, v5.4b[0]
        ldr     d30, [x3, x4]
        ldr     d31, [x27, x4]
        ldr     d29, [x19, x4]
        ldr     d23, [x22, x4]
        ldr     d0, [x23, x4]
        ldr     d2, [x24, x4]
        mov     v17.16b, v24.16b
        stp     q18, q3, [sp, 240]
        ldp     q3, q24, [sp, 416]
        ldr     q9, [sp, 384]
        sdot    v21.4s, v27.16b, v28.4b[0]
        sdot    v11.4s, v27.16b, v7.4b[0]
        sdot    v13.4s, v27.16b, v6.4b[0]
        sdot    v1.4s, v27.16b, v19.4b[0]
        stp     q1, q13, [sp, 192]
        sdot    v24.4s, v27.16b, v2.4b[0]
        mov     v13.16b, v11.16b
        mov     v11.16b, v21.16b
        ldp     q15, q21, [sp, 336]
        sdot    v14.4s, v27.16b, v30.4b[0]
        sdot    v9.4s, v27.16b, v31.4b[0]
        sdot    v8.4s, v27.16b, v29.4b[0]
        sdot    v16.4s, v27.16b, v23.4b[0]
        sdot    v12.4s, v27.16b, v0.4b[0]
        str     q24, [sp, 432]
        ldp     q27, q24, [x0], 64
        ldr     q1, [sp, 400]
        str     q16, [sp, 224]
        ldp     q16, q4, [sp, 304]
        sdot    v21.4s, v26.16b, v29.4b[0]
        sdot    v1.4s, v26.16b, v17.4b[0]
        sdot    v21.4s, v24.16b, v29.4b[1]
        stp     q5, q23, [sp, 160]
        sdot    v15.4s, v26.16b, v28.4b[0]
        sdot    v16.4s, v26.16b, v19.4b[0]
        ldr     q18, [sp, 288]
        str     q21, [sp, 352]
        mov     v21.16b, v11.16b
        sdot    v1.4s, v24.16b, v17.4b[1]
        sdot    v21.4s, v27.16b, v28.4b[1]
        sdot    v15.4s, v24.16b, v28.4b[1]
        str     q1, [sp, 400]
        ldp     q28, q1, [sp, 176]
        sdot    v16.4s, v24.16b, v19.4b[1]
        str     q16, [sp, 304]
        ldr     q16, [sp, 224]
        mov     v10.16b, v8.16b
        ldr     q8, [sp, 368]
        sdot    v3.4s, v26.16b, v6.4b[0]
        sdot    v18.4s, v26.16b, v23.4b[0]
        sdot    v20.4s, v26.16b, v30.4b[0]
        ldr     q23, [sp, 272]
        sdot    v3.4s, v24.16b, v6.4b[1]
        sdot    v18.4s, v24.16b, v28.4b[1]
        sdot    v14.4s, v27.16b, v30.4b[1]
        sdot    v20.4s, v24.16b, v30.4b[1]
        ldr     q30, [sp, 432]
        mov     v11.16b, v13.16b
        ldr     q13, [sp, 208]
        str     q3, [sp, 416]
        ldr     q3, [sp, 256]
        sdot    v16.4s, v27.16b, v28.4b[1]
        str     q18, [sp, 288]
        ldr     q18, [sp, 240]
        ldr     q28, [sp, 160]
        sdot    v8.4s, v26.16b, v31.4b[0]
        sdot    v9.4s, v27.16b, v31.4b[1]
        sdot    v8.4s, v24.16b, v31.4b[1]
        sdot    v4.4s, v26.16b, v7.4b[0]
        sdot    v23.4s, v26.16b, v5.4b[0]
        sdot    v25.4s, v26.16b, v0.4b[0]
        sdot    v22.4s, v26.16b, v2.4b[0]
        stp     q8, q9, [sp, 368]
        mov     v8.16b, v10.16b
        sub     x2, x2, 8
        sdot    v8.4s, v27.16b, v29.4b[1]
        sdot    v11.4s, v27.16b, v7.4b[1]
        sdot    v4.4s, v24.16b, v7.4b[1]
        sdot    v13.4s, v27.16b, v6.4b[1]
        sdot    v3.4s, v27.16b, v17.4b[1]
        sdot    v1.4s, v27.16b, v19.4b[1]
        sdot    v18.4s, v27.16b, v28.4b[1]
        sdot    v23.4s, v24.16b, v28.4b[1]
        sdot    v12.4s, v27.16b, v0.4b[1]
        sdot    v25.4s, v24.16b, v0.4b[1]
        sdot    v30.4s, v27.16b, v2.4b[1]
        sdot    v22.4s, v24.16b, v2.4b[1]
        add     x4, x4, 8
        cmp     x2, 7
        stp     q4, q15, [sp, 320]
        str     q23, [sp, 272]
        b.hi    2b
        ldr     x0, [sp, 152]
        ldp     x2, x4, [sp, 16]
        ldr     x27, [sp, 112]
        ldr     x28, [sp, 96]
        ldr     x29, [sp, 80]
        add     x0, x0, x2
        ldr     x2, [sp, 128]
        ldr     x30, [sp, 64]
        ldr     x26, [sp, 48]
        ldr     x24, [sp, 32]
        mov     v10.16b, v25.16b
        mov     v23.16b, v12.16b
        ldp     q5, q15, [sp, 272]
        mov     v26.16b, v16.16b
        ldp     q0, q16, [sp, 304]
        ldp     q19, q12, [sp, 400]
        mov     v31.16b, v11.16b
        ldp     q17, q25, [sp, 336]
        ldp     q29, q11, [sp, 368]
        add     x3, x3, x25
        add     x19, x19, x25
        add     x20, x20, x25
        add     x21, x21, x25
        add     x22, x22, x25
        add     x23, x23, x25
        mov     v2.16b, v1.16b
        mov     v9.16b, v13.16b
        mov     v7.16b, v14.16b
        mov     v24.16b, v21.16b
        mov     v27.16b, v8.16b
        cbz     x4, 4f
        b       6f
3:
        mov     x4, x5
        mov     v22.16b, v20.16b
        mov     v30.16b, v7.16b
        mov     v10.16b, v20.16b
        mov     v23.16b, v7.16b
        mov     v5.16b, v20.16b
        mov     v18.16b, v7.16b
        mov     v15.16b, v20.16b
        mov     v26.16b, v7.16b
        mov     v0.16b, v20.16b
        mov     v2.16b, v7.16b
        mov     v19.16b, v20.16b
        mov     v3.16b, v7.16b
        mov     v12.16b, v20.16b
        mov     v9.16b, v7.16b
        mov     v16.16b, v20.16b
        mov     v31.16b, v7.16b
        mov     v17.16b, v20.16b
        mov     v24.16b, v7.16b
        mov     v25.16b, v20.16b
        mov     v27.16b, v7.16b
        mov     v29.16b, v20.16b
        mov     v11.16b, v7.16b
        cbnz    x4, 6f
4:
        str     x0, [sp, 152]
5:
        ldr     x4, [sp, 40]
        str     q30, [sp, 432]
        mov     v13.16b, v18.16b
        mov     v14.16b, v15.16b
        mov     x0, x4
        ld1r    {v28.4s}, [x0], 4
        cmp     x2, 7
        sqrdmulh        v21.4s, v7.4s, v28.4s
        str     q21, [sp, 416]
        sqrdmulh        v21.4s, v20.4s, v28.4s
        str     q21, [sp, 400]
        sqrdmulh        v21.4s, v11.4s, v28.4s
        str     q21, [sp, 384]
        sqrdmulh        v21.4s, v29.4s, v28.4s
        str     q21, [sp, 368]
        sqrdmulh        v21.4s, v27.4s, v28.4s
        str     q21, [sp, 352]
        sqrdmulh        v21.4s, v25.4s, v28.4s
        str     q21, [sp, 336]
        sqrdmulh        v21.4s, v24.4s, v28.4s
        str     q21, [sp, 320]
        sqrdmulh        v21.4s, v17.4s, v28.4s
        str     q21, [sp, 304]
        sqrdmulh        v21.4s, v31.4s, v28.4s
        str     q21, [sp, 288]
        sqrdmulh        v21.4s, v16.4s, v28.4s
        str     q21, [sp, 272]
        sqrdmulh        v21.4s, v9.4s, v28.4s
        str     q21, [sp, 256]
        sqrdmulh        v21.4s, v12.4s, v28.4s
        str     q21, [sp, 240]
        sqrdmulh        v21.4s, v3.4s, v28.4s
        str     q21, [sp, 224]
        sqrdmulh        v21.4s, v19.4s, v28.4s
        ld1r    {v8.4s}, [x0]
        str     q21, [sp, 208]
        sqrdmulh        v21.4s, v2.4s, v28.4s
        str     q21, [sp, 192]
        sqrdmulh        v21.4s, v0.4s, v28.4s
        str     q21, [sp, 176]
        sqrdmulh        v21.4s, v26.4s, v28.4s
        str     q21, [sp, 160]
        sqrdmulh        v21.4s, v18.4s, v28.4s
        sqrdmulh        v1.4s, v15.4s, v28.4s
        str     q21, [sp, 128]
        mov     v21.16b, v0.16b
        sqrdmulh        v6.4s, v5.4s, v28.4s
        mov     v18.16b, v5.16b
        sqrdmulh        v5.4s, v23.4s, v28.4s
        mov     v0.16b, v11.16b
        mov     v15.16b, v10.16b
        sqrdmulh        v4.4s, v10.4s, v28.4s
        mov     v10.16b, v9.16b
        sqrdmulh        v11.4s, v30.4s, v28.4s
        sqrdmulh        v9.4s, v22.4s, v28.4s
        cmeq    v28.4s, v8.4s, 0
        mvn     v28.16b, v28.16b
        and     v7.16b, v7.16b, v28.16b
        and     v30.16b, v20.16b, v28.16b
        stp     q30, q7, [sp, 96]
        and     v30.16b, v0.16b, v28.16b
        and     v29.16b, v29.16b, v28.16b
        stp     q29, q30, [sp, 64]
        and     v29.16b, v17.16b, v28.16b
        str     q29, [sp, 48]
        and     v29.16b, v31.16b, v28.16b
        and     v31.16b, v21.16b, v28.16b
        and     v21.16b, v26.16b, v28.16b
        ldp     q20, q26, [sp, 416]
        mov     v7.16b, v18.16b
        and     v27.16b, v27.16b, v28.16b
        and     v25.16b, v25.16b, v28.16b
        and     v24.16b, v24.16b, v28.16b
        and     v0.16b, v16.16b, v28.16b
        and     v16.16b, v10.16b, v28.16b
        and     v17.16b, v12.16b, v28.16b
        and     v18.16b, v3.16b, v28.16b
        and     v19.16b, v19.16b, v28.16b
        and     v30.16b, v2.16b, v28.16b
        and     v2.16b, v14.16b, v28.16b
        and     v3.16b, v13.16b, v28.16b
        and     v12.16b, v7.16b, v28.16b
        and     v13.16b, v23.16b, v28.16b
        and     v14.16b, v15.16b, v28.16b
        and     v15.16b, v26.16b, v28.16b
        and     v22.16b, v22.16b, v28.16b
        ldr     q28, [sp, 112]
        ldp     q23, q26, [sp, 384]
        ldp     q7, q10, [sp, 352]
        ssra    v1.4s, v2.4s, 31
        ssra    v20.4s, v28.4s, 31
        ldr     q28, [sp, 96]
        ssra    v7.4s, v27.4s, 31
        ldr     q27, [sp, 336]
        ldr     q2, [sp, 128]
        ssra    v26.4s, v28.4s, 31
        ldr     q28, [sp, 80]
        ssra    v27.4s, v25.4s, 31
        ldr     q25, [sp, 320]
        ssra    v2.4s, v3.4s, 31
        ssra    v23.4s, v28.4s, 31
        ldr     q28, [sp, 64]
        ssra    v25.4s, v24.4s, 31
        ldr     q24, [sp, 304]
        ssra    v6.4s, v12.4s, 31
        ssra    v10.4s, v28.4s, 31
        ldr     q28, [sp, 48]
        ssra    v5.4s, v13.4s, 31
        ssra    v4.4s, v14.4s, 31
        ssra    v11.4s, v15.4s, 31
        ssra    v24.4s, v28.4s, 31
        ldr     q28, [sp, 288]
        ssra    v9.4s, v22.4s, 31
        add     x0, x4, 8
        srshl   v15.4s, v20.4s, v8.4s
        ssra    v28.4s, v29.4s, 31
        ldr     q29, [sp, 272]
        srshl   v14.4s, v26.4s, v8.4s
        srshl   v13.4s, v23.4s, v8.4s
        srshl   v12.4s, v10.4s, v8.4s
        ssra    v29.4s, v0.4s, 31
        ldr     q0, [sp, 256]
        srshl   v10.4s, v7.4s, v8.4s
        srshl   v26.4s, v27.4s, v8.4s
        srshl   v25.4s, v25.4s, v8.4s
        ssra    v0.4s, v16.4s, 31
        ldr     q16, [sp, 240]
        srshl   v24.4s, v24.4s, v8.4s
        srshl   v23.4s, v28.4s, v8.4s
        srshl   v22.4s, v29.4s, v8.4s
        ssra    v16.4s, v17.4s, 31
        ldr     q17, [sp, 224]
        srshl   v20.4s, v16.4s, v8.4s
        srshl   v1.4s, v1.4s, v8.4s
        srshl   v2.4s, v2.4s, v8.4s
        ssra    v17.4s, v18.4s, 31
        ldr     q18, [sp, 208]
        srshl   v7.4s, v17.4s, v8.4s
        srshl   v3.4s, v6.4s, v8.4s
        srshl   v5.4s, v5.4s, v8.4s
        ssra    v18.4s, v19.4s, 31
        ldr     q19, [sp, 192]
        srshl   v18.4s, v18.4s, v8.4s
        srshl   v11.4s, v11.4s, v8.4s
        srshl   v9.4s, v9.4s, v8.4s
        ssra    v19.4s, v30.4s, 31
        ldr     q30, [sp, 176]
        srshl   v17.4s, v19.4s, v8.4s
        srshl   v19.4s, v4.4s, v8.4s
        sqxtn   v28.4h, v15.4s
        ssra    v30.4s, v31.4s, 31
        ldr     q31, [sp, 160]
        srshl   v16.4s, v30.4s, v8.4s
        sqxtn   v29.4h, v13.4s
        sqxtn   v30.4h, v10.4s
        ssra    v31.4s, v21.4s, 31
        srshl   v21.4s, v0.4s, v8.4s
        srshl   v0.4s, v31.4s, v8.4s
        ld1r    {v8.8h}, [x0]
        sqxtn2  v28.8h, v14.4s
        sqxtn   v31.4h, v25.4s
        sqxtn   v27.4h, v23.4s
        sqxtn2  v29.8h, v12.4s
        sqxtn2  v30.8h, v26.4s
        sqxtn2  v31.8h, v24.4s
        sqxtn2  v27.8h, v22.4s
        sqxtn   v26.4h, v21.4s
        sqxtn   v25.4h, v7.4s
        sqxtn   v24.4h, v17.4s
        sqxtn   v23.4h, v0.4s
        sqxtn   v22.4h, v2.4s
        sqxtn   v21.4h, v5.4s
        sqadd   v28.8h, v28.8h, v8.8h
        sqxtn2  v26.8h, v20.4s
        sqxtn2  v25.8h, v18.4s
        sqxtn2  v24.8h, v16.4s
        sqxtn2  v23.8h, v1.4s
        sqxtn2  v22.8h, v3.4s
        sqxtn2  v21.8h, v19.4s
        sqxtn   v20.4h, v11.4s
        sqadd   v29.8h, v29.8h, v8.8h
        sqadd   v30.8h, v30.8h, v8.8h
        sqadd   v27.8h, v27.8h, v8.8h
        sqxtn   v28.8b, v28.8h
        sqxtn2  v20.8h, v9.4s
        sqadd   v31.8h, v31.8h, v8.8h
        sqadd   v26.8h, v26.8h, v8.8h
        sqadd   v25.8h, v25.8h, v8.8h
        sqadd   v23.8h, v23.8h, v8.8h
        sqadd   v21.8h, v21.8h, v8.8h
        sqxtn2  v28.16b, v29.8h
        sqxtn   v29.8b, v30.8h
        sqxtn   v30.8b, v27.8h
        sqadd   v24.8h, v24.8h, v8.8h
        sqadd   v22.8h, v22.8h, v8.8h
        sqadd   v20.8h, v20.8h, v8.8h
        sqxtn2  v29.16b, v31.8h
        sqxtn2  v30.16b, v26.8h
        sqxtn   v31.8b, v25.8h
        add     x0, x4, 10
        sqxtn   v27.8b, v23.8h
        sqxtn   v26.8b, v21.8h
        sqxtn2  v31.16b, v24.8h
        sqxtn2  v27.16b, v22.8h
        sqxtn2  v26.16b, v20.8h
        ld1r    {v25.16b}, [x0]
        add     x0, x4, 11
        ld1r    {v24.16b}, [x0]
        smax    v28.16b, v28.16b, v25.16b
        smax    v29.16b, v29.16b, v25.16b
        smax    v30.16b, v30.16b, v25.16b
        smax    v23.16b, v31.16b, v25.16b
        smax    v22.16b, v27.16b, v25.16b
        smax    v25.16b, v26.16b, v25.16b
        smin    v26.16b, v28.16b, v24.16b
        smin    v27.16b, v29.16b, v24.16b
        smin    v31.16b, v30.16b, v24.16b
        smin    v30.16b, v23.16b, v24.16b
        smin    v29.16b, v22.16b, v24.16b
        smin    v28.16b, v25.16b, v24.16b
        b.ls    9f
        str     d26, [x6]
        ext     v26.16b, v26.16b, v26.16b, 8
        str     d26, [x8]
        ext     v26.16b, v27.16b, v27.16b, 8
        str     d27, [x9]
        ext     v27.16b, v31.16b, v31.16b, 8
        str     d26, [x10]
        ext     v26.16b, v30.16b, v30.16b, 8
        str     d31, [x11]
        ext     v31.16b, v29.16b, v29.16b, 8
        str     d27, [x12]
        ext     v27.16b, v28.16b, v28.16b, 8
        sub     x24, x24, x5
        sub     x23, x23, x5
        sub     x26, x26, x5
        sub     x22, x22, x5
        sub     x30, x30, x5
        sub     x21, x21, x5
        sub     x29, x29, x5
        sub     x20, x20, x5
        sub     x28, x28, x5
        sub     x19, x19, x5
        sub     x27, x27, x5
        sub     x3, x3, x5
        add     x6, x6, x7
        subs    x2, x2, 8
        str     d30, [x13]
        str     d26, [x14]
        str     d29, [x15]
        str     d31, [x16]
        str     d28, [x17]
        str     d27, [x1]
        add     x8, x8, x7
        add     x9, x9, x7
        add     x10, x10, x7
        add     x11, x11, x7
        add     x12, x12, x7
        add     x13, x13, x7
        add     x14, x14, x7
        add     x15, x15, x7
        add     x16, x16, x7
        add     x17, x17, x7
        add     x1, x1, x7
        b.ne    1b
        b       12f
6:
        stp     q12, q10, [sp, 416]
        ldr     d28, [x3]
        stp     q25, q29, [sp, 352]
        mov     v25.16b, v26.16b
        mov     v21.16b, v9.16b
        str     q28, [sp, 304]
        ldr     d28, [x27]
        str     q19, [sp, 400]
        ldr     d19, [x28]
        mov     v12.16b, v0.16b
        str     q28, [sp, 288]
        ldr     d28, [x19]
        ldp     q26, q4, [sp, 288]
        mov     v6.16b, v2.16b
        ldr     d1, [x20]
        str     q28, [sp, 320]
        ldp     q28, q29, [x0]
        ldr     d0, [x29]
        ldr     d10, [x21]
        ldr     d8, [x30]
        sdot    v11.4s, v28.16b, v26.4b[0]
        ldr     d14, [x22]
        ldr     d13, [x26]
        ldr     d9, [x23]
        ldr     d2, [x24]
        str     q11, [sp, 384]
        ldr     q11, [sp, 320]
        sdot    v7.4s, v28.16b, v4.4b[0]
        sdot    v24.4s, v28.16b, v19.4b[0]
        sdot    v31.4s, v28.16b, v1.4b[0]
        sdot    v27.4s, v28.16b, v11.4b[0]
        sdot    v21.4s, v28.16b, v0.4b[0]
        sdot    v3.4s, v28.16b, v10.4b[0]
        sdot    v6.4s, v28.16b, v8.4b[0]
        sdot    v25.4s, v28.16b, v14.4b[0]
        sdot    v18.4s, v28.16b, v13.4b[0]
        sdot    v23.4s, v28.16b, v9.4b[0]
        sdot    v30.4s, v28.16b, v2.4b[0]
        mov     v28.16b, v4.16b
        mov     v28.16b, v26.16b
        mov     v28.16b, v11.16b
        ldr     q28, [sp, 416]
        str     q27, [sp, 336]
        stp     q6, q21, [sp, 192]
        ldp     q27, q21, [sp, 352]
        sdot    v28.4s, v29.16b, v0.4b[0]
        str     q28, [sp, 416]
        ldr     q28, [sp, 400]
        sdot    v12.4s, v29.16b, v8.4b[0]
        sdot    v20.4s, v29.16b, v4.4b[0]
        sdot    v21.4s, v29.16b, v26.4b[0]
        sdot    v28.4s, v29.16b, v10.4b[0]
        str     q28, [sp, 400]
        ldr     q28, [sp, 432]
        sdot    v27.4s, v29.16b, v11.4b[0]
        sdot    v17.4s, v29.16b, v19.4b[0]
        sdot    v16.4s, v29.16b, v1.4b[0]
        mov     v6.16b, v0.16b
        mov     v0.16b, v12.16b
        sdot    v15.4s, v29.16b, v14.4b[0]
        sdot    v5.4s, v29.16b, v13.4b[0]
        sdot    v28.4s, v29.16b, v9.4b[0]
        sdot    v22.4s, v29.16b, v2.4b[0]
        add     x3, x3, x4
        add     x27, x27, x4
        add     x19, x19, x4
        add     x28, x28, x4
        add     x20, x20, x4
        add     x29, x29, x4
        add     x21, x21, x4
        add     x30, x30, x4
        add     x22, x22, x4
        add     x26, x26, x4
        add     x23, x23, x4
        cmp     x4, 5
        add     x24, x24, x4
        str     q25, [sp, 224]
        stp     q18, q3, [sp, 240]
        str     q9, [sp, 368]
        str     q28, [sp, 432]
        b.cc    7f
        mov     v3.16b, v11.16b
        mov     v11.16b, v19.16b
        ldp     q28, q29, [x0, 32]
        ldr     q19, [sp, 384]
        mov     v9.16b, v21.16b
        mov     v25.16b, v10.16b
        sdot    v9.4s, v29.16b, v26.4b[1]
        sdot    v19.4s, v28.16b, v26.4b[1]
        ldr     q26, [sp, 336]
        mov     v10.16b, v8.16b
        mov     v8.16b, v14.16b
        mov     v14.16b, v13.16b
        mov     v13.16b, v2.16b
        sdot    v31.4s, v28.16b, v1.4b[1]
        sdot    v16.4s, v29.16b, v1.4b[1]
        ldp     q2, q1, [sp, 192]
        mov     v18.16b, v4.16b
        sdot    v26.4s, v28.16b, v3.4b[1]
        mov     v21.16b, v27.16b
        ldp     q18, q27, [sp, 240]
        str     q19, [sp, 384]
        str     q26, [sp, 336]
        sdot    v24.4s, v28.16b, v11.4b[1]
        sdot    v17.4s, v29.16b, v11.4b[1]
        ldp     q19, q12, [sp, 400]
        sdot    v2.4s, v28.16b, v10.4b[1]
        sdot    v0.4s, v29.16b, v10.4b[1]
        ldr     q26, [sp, 224]
        ldr     q11, [sp, 368]
        ldr     q10, [sp, 432]
        sdot    v21.4s, v29.16b, v3.4b[1]
        sdot    v1.4s, v28.16b, v6.4b[1]
        sdot    v27.4s, v28.16b, v25.4b[1]
        sdot    v7.4s, v28.16b, v4.4b[1]
        sdot    v20.4s, v29.16b, v4.4b[1]
        sdot    v12.4s, v29.16b, v6.4b[1]
        sdot    v19.4s, v29.16b, v25.4b[1]
        sdot    v26.4s, v28.16b, v8.4b[1]
        sdot    v15.4s, v29.16b, v8.4b[1]
        sdot    v18.4s, v28.16b, v14.4b[1]
        sdot    v5.4s, v29.16b, v14.4b[1]
        sdot    v23.4s, v28.16b, v11.4b[1]
        sdot    v10.4s, v29.16b, v11.4b[1]
        sdot    v30.4s, v28.16b, v13.4b[1]
        sdot    v22.4s, v29.16b, v13.4b[1]
        add     x0, x0, 64
        mov     v29.16b, v9.16b
        mov     v9.16b, v1.16b
        mov     v3.16b, v27.16b
        mov     v25.16b, v21.16b
        str     x0, [sp, 152]
        b       8f
7:
        ldp     q9, q26, [sp, 208]
        ldp     q12, q10, [sp, 416]
        ldp     q18, q3, [sp, 240]
        ldr     q2, [sp, 192]
        ldr     q19, [sp, 400]
        add     x0, x0, 32
        str     x0, [sp, 152]
        mov     v29.16b, v21.16b
        mov     v25.16b, v27.16b
8:
        ldr     q27, [sp, 336]
        ldr     q11, [sp, 384]
        b       5b
9:
        tbnz    w2, 2, 13f
        tbnz    w2, 1, 14f
10:
        tbz     w2, 0, 12f
11:
        st1     {v26.b}[0], [x6]
        st1     {v26.b}[8], [x8]
        st1     {v27.b}[0], [x9]
        st1     {v27.b}[8], [x10]
        st1     {v31.b}[0], [x11]
        st1     {v31.b}[8], [x12]
        st1     {v30.b}[0], [x13]
        st1     {v30.b}[8], [x14]
        st1     {v29.b}[0], [x15]
        st1     {v29.b}[8], [x16]
        st1     {v28.b}[0], [x17]
        st1     {v28.b}[8], [x1]
12:
        add     sp, sp, 0x1c0
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
        st1     {v26.s}[0], [x6], 4
        st1     {v26.s}[2], [x8], 4
        st1     {v27.s}[0], [x9], 4
        st1     {v27.s}[2], [x10], 4
        st1     {v31.s}[0], [x11], 4
        st1     {v31.s}[2], [x12], 4
        st1     {v30.s}[0], [x13], 4
        st1     {v30.s}[2], [x14], 4
        st1     {v29.s}[0], [x15], 4
        st1     {v29.s}[2], [x16], 4
        st1     {v28.s}[0], [x17], 4
        st1     {v28.s}[2], [x1], 4
        ext     v26.16b, v26.16b, v26.16b, 4
        ext     v27.16b, v27.16b, v27.16b, 4
        ext     v31.16b, v31.16b, v31.16b, 4
        ext     v30.16b, v30.16b, v30.16b, 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
        tbz     w2, 1, 10b
14:
        st1     {v26.h}[0], [x6], 2
        st1     {v26.h}[4], [x8], 2
        st1     {v27.h}[0], [x9], 2
        st1     {v27.h}[4], [x10], 2
        st1     {v31.h}[0], [x11], 2
        st1     {v31.h}[4], [x12], 2
        st1     {v30.h}[0], [x13], 2
        st1     {v30.h}[4], [x14], 2
        st1     {v29.h}[0], [x15], 2
        st1     {v29.h}[4], [x16], 2
        st1     {v28.h}[0], [x17], 2
        st1     {v28.h}[4], [x1], 2
        ext     v26.16b, v26.16b, v26.16b, 2
        ext     v27.16b, v27.16b, v27.16b, 2
        ext     v31.16b, v31.16b, v31.16b, 2
        ext     v30.16b, v30.16b, v30.16b, 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w2, 0, 11b
        b       12b

END_FUNCTION xnn_qs8_gemm_minmax_ukernel_12x8c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
