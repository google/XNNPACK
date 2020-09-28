// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_igemm_minmax_ukernel_6x16c4__aarch64_neondot
        sub     sp, sp, 0x120
        stp     d7, d6, [sp, 128]
        stp     d5, d4, [sp, 144]
        stp     d11, d10, [sp, 160]
        stp     d9, d8, [sp, 176]
        stp     x28, x27, [sp, 192]
        stp     x26, x25, [sp, 208]
        stp     x24, x23, [sp, 224]
        stp     x22, x21, [sp, 240]
        stp     x20, x19, [sp, 256]
        stp     x29, x30, [sp, 272]
        ldp     x14, x8, [sp, 304]
        cmp     x0, 2
        sub     x13, x2, 8
        lsl     x9, x13, 4
        str     x8, [sp, 24]
        ldp     x8, x15, [sp, 288]
        and     x12, x9, 0xffffffffffffff80
        mov     x16, x2
        and     x2, x13, 0xfffffffffffffff8
        str     x8, [sp, 8]
        add     x8, x6, x7
        csel    x8, x6, x8, lo
        cmp     x0, 3
        add     x10, x8, x7
        csel    x9, x8, x10, lo
        cmp     x0, 4
        add     x10, x9, x7
        csel    x10, x9, x10, lo
        cmp     x0, 5
        add     x11, x10, x7
        csel    x11, x10, x11, lo
        cmp     x0, 6
        add     x0, x12, 0x80
        add     x12, x11, x7
        add     x17, x2, 8
        csel    x12, x12, x11, eq
        sub     x7, x13, x2
        str     x3, [sp, 16]
1:
        ldp     q7, q30, [x5]
        ldp     q31, q27, [x5, 32]
        add     x5, x5, 64
        mov     x19, x3
        mov     v25.16b, v30.16b
        mov     v6.16b, v27.16b
        mov     v26.16b, v31.16b
        mov     v24.16b, v7.16b
        mov     v23.16b, v27.16b
        mov     v22.16b, v31.16b
        mov     v21.16b, v30.16b
        mov     v20.16b, v7.16b
        mov     v19.16b, v27.16b
        mov     v18.16b, v31.16b
        mov     v17.16b, v30.16b
        mov     v16.16b, v7.16b
        mov     v0.16b, v27.16b
        mov     v1.16b, v31.16b
        mov     v2.16b, v30.16b
        mov     v3.16b, v7.16b
        mov     v12.16b, v27.16b
        mov     v13.16b, v31.16b
        mov     v14.16b, v30.16b
        mov     v15.16b, v7.16b
2:
        ldp     x13, x2, [x4]
        ldp     x3, x20, [x4, 16]
        ldp     x21, x23, [x4, 32]
        add     x22, x13, x15
        cmp     x13, x14
        csel    x22, x13, x22, eq
        add     x13, x2, x15
        cmp     x2, x14
        csel    x26, x2, x13, eq
        add     x13, x3, x15
        cmp     x3, x14
        csel    x13, x3, x13, eq
        add     x2, x20, x15
        cmp     x20, x14
        csel    x27, x20, x2, eq
        add     x2, x21, x15
        cmp     x21, x14
        csel    x3, x21, x2, eq
        add     x2, x23, x15
        cmp     x23, x14
        csel    x28, x23, x2, eq
        cmp     x16, 8
        b.cc    5f
        add     x25, x26, x17
        add     x24, x27, x17
        add     x23, x28, x17
        mov     x29, x5
        mov     x30, x22
        mov     x21, x13
        mov     x20, x3
        mov     x2, x16
        stp     q1, q13, [sp, 96]
        stp     q22, q18, [sp, 64]
        mov     v22.16b, v24.16b
        mov     v24.16b, v26.16b
        mov     v26.16b, v27.16b
        mov     v27.16b, v31.16b
        mov     v31.16b, v30.16b
        mov     v30.16b, v7.16b
        mov     v13.16b, v12.16b
        mov     v12.16b, v0.16b
        mov     v0.16b, v19.16b
        mov     v19.16b, v23.16b
        mov     v23.16b, v6.16b
3:
        ldr     d5, [x30], 8
        ldr     d4, [x26], 8
        ldr     d11, [x21], 8
        ldr     d10, [x27], 8
        ldr     d9, [x20], 8
        ldr     d8, [x28], 8
        ldp     q6, q7, [x29]
        ldp     q28, q29, [x29, 32]
        sub     x2, x2, 8
        cmp     x2, 7
        sdot    v3.4s, v6.16b, v4.4b[0]
        sdot    v16.4s, v6.16b, v11.4b[0]
        sdot    v20.4s, v6.16b, v10.4b[0]
        sdot    v22.4s, v6.16b, v9.4b[0]
        stp     q22, q20, [sp, 32]
        mov     v1.16b, v3.16b
        ldr     q3, [sp, 112]
        mov     v18.16b, v16.16b
        ldp     q20, q16, [sp, 80]
        ldr     q22, [sp, 64]
        sdot    v3.4s, v28.16b, v5.4b[0]
        sdot    v13.4s, v29.16b, v5.4b[0]
        sdot    v16.4s, v28.16b, v4.4b[0]
        sdot    v12.4s, v29.16b, v4.4b[0]
        sdot    v20.4s, v28.16b, v11.4b[0]
        sdot    v0.4s, v29.16b, v11.4b[0]
        sdot    v22.4s, v28.16b, v10.4b[0]
        sdot    v19.4s, v29.16b, v10.4b[0]
        sdot    v24.4s, v28.16b, v9.4b[0]
        sdot    v23.4s, v29.16b, v9.4b[0]
        sdot    v27.4s, v28.16b, v8.4b[0]
        sdot    v26.4s, v29.16b, v8.4b[0]
        ldp     q28, q29, [x29, 96]
        sdot    v15.4s, v6.16b, v5.4b[0]
        sdot    v14.4s, v7.16b, v5.4b[0]
        sdot    v2.4s, v7.16b, v4.4b[0]
        sdot    v20.4s, v28.16b, v11.4b[1]
        sdot    v22.4s, v28.16b, v10.4b[1]
        sdot    v17.4s, v7.16b, v11.4b[0]
        sdot    v21.4s, v7.16b, v10.4b[0]
        sdot    v25.4s, v7.16b, v9.4b[0]
        sdot    v30.4s, v6.16b, v8.4b[0]
        sdot    v31.4s, v7.16b, v8.4b[0]
        ldp     q6, q7, [x29, 64]
        str     q20, [sp, 80]
        ldr     q20, [sp, 48]
        str     q22, [sp, 64]
        ldr     q22, [sp, 32]
        sdot    v3.4s, v28.16b, v5.4b[1]
        sdot    v16.4s, v28.16b, v4.4b[1]
        str     q3, [sp, 112]
        mov     v3.16b, v1.16b
        str     q16, [sp, 96]
        mov     v16.16b, v18.16b
        sdot    v15.4s, v6.16b, v5.4b[1]
        sdot    v14.4s, v7.16b, v5.4b[1]
        sdot    v13.4s, v29.16b, v5.4b[1]
        sdot    v3.4s, v6.16b, v4.4b[1]
        sdot    v2.4s, v7.16b, v4.4b[1]
        sdot    v12.4s, v29.16b, v4.4b[1]
        sdot    v16.4s, v6.16b, v11.4b[1]
        sdot    v17.4s, v7.16b, v11.4b[1]
        sdot    v0.4s, v29.16b, v11.4b[1]
        sdot    v20.4s, v6.16b, v10.4b[1]
        sdot    v21.4s, v7.16b, v10.4b[1]
        sdot    v19.4s, v29.16b, v10.4b[1]
        sdot    v22.4s, v6.16b, v9.4b[1]
        sdot    v25.4s, v7.16b, v9.4b[1]
        sdot    v24.4s, v28.16b, v9.4b[1]
        sdot    v23.4s, v29.16b, v9.4b[1]
        sdot    v30.4s, v6.16b, v8.4b[1]
        sdot    v31.4s, v7.16b, v8.4b[1]
        sdot    v27.4s, v28.16b, v8.4b[1]
        sdot    v26.4s, v29.16b, v8.4b[1]
        add     x29, x29, 0x80
        b.hi    3b
        mov     v7.16b, v30.16b
        mov     v30.16b, v31.16b
        mov     v31.16b, v27.16b
        mov     v27.16b, v26.16b
        mov     v6.16b, v23.16b
        mov     v26.16b, v24.16b
        mov     v24.16b, v22.16b
        mov     v23.16b, v19.16b
        ldp     q22, q18, [sp, 64]
        mov     v19.16b, v0.16b
        mov     v0.16b, v12.16b
        mov     v12.16b, v13.16b
        ldp     q1, q13, [sp, 96]
        add     x22, x22, x17
        add     x13, x13, x17
        add     x3, x3, x17
        add     x5, x5, x0
        mov     x2, x7
        add     x4, x4, 0x30
        cbnz    x2, 6f
4:
        subs    x19, x19, 0x30
        b.ne    2b
        b       8f
5:
        mov     x2, x16
        mov     x23, x28
        mov     x24, x27
        mov     x25, x26
        add     x4, x4, 0x30
        cbz     x2, 4b
6:
        ldr     d5, [x22]
        ldr     d4, [x25]
        ldr     d11, [x13]
        ldr     d10, [x24]
        ldp     q28, q29, [x5]
        ldr     d9, [x3]
        ldr     d8, [x23]
        cmp     x2, 5
        sdot    v15.4s, v28.16b, v5.4b[0]
        sdot    v14.4s, v29.16b, v5.4b[0]
        sdot    v3.4s, v28.16b, v4.4b[0]
        sdot    v2.4s, v29.16b, v4.4b[0]
        sdot    v16.4s, v28.16b, v11.4b[0]
        sdot    v17.4s, v29.16b, v11.4b[0]
        sdot    v20.4s, v28.16b, v10.4b[0]
        sdot    v21.4s, v29.16b, v10.4b[0]
        sdot    v24.4s, v28.16b, v9.4b[0]
        sdot    v7.4s, v28.16b, v8.4b[0]
        sdot    v25.4s, v29.16b, v9.4b[0]
        sdot    v30.4s, v29.16b, v8.4b[0]
        ldp     q28, q29, [x5, 32]
        sdot    v13.4s, v28.16b, v5.4b[0]
        sdot    v1.4s, v28.16b, v4.4b[0]
        sdot    v18.4s, v28.16b, v11.4b[0]
        sdot    v22.4s, v28.16b, v10.4b[0]
        sdot    v26.4s, v28.16b, v9.4b[0]
        sdot    v31.4s, v28.16b, v8.4b[0]
        sdot    v12.4s, v29.16b, v5.4b[0]
        sdot    v0.4s, v29.16b, v4.4b[0]
        sdot    v19.4s, v29.16b, v11.4b[0]
        sdot    v23.4s, v29.16b, v10.4b[0]
        sdot    v6.4s, v29.16b, v9.4b[0]
        sdot    v27.4s, v29.16b, v8.4b[0]
        b.cc    7f
        stp     q15, q3, [sp, 80]
        ldp     q28, q29, [x5, 64]
        str     q13, [sp, 112]
        ldr     q13, [sp, 80]
        mov     v15.16b, v14.16b
        mov     v14.16b, v12.16b
        mov     v12.16b, v2.16b
        sdot    v13.4s, v28.16b, v5.4b[1]
        mov     v2.16b, v0.16b
        mov     v0.16b, v17.16b
        mov     v17.16b, v19.16b
        mov     v19.16b, v21.16b
        mov     v21.16b, v23.16b
        mov     v23.16b, v25.16b
        mov     v25.16b, v6.16b
        mov     v3.16b, v1.16b
        mov     v1.16b, v16.16b
        mov     v16.16b, v18.16b
        mov     v18.16b, v20.16b
        mov     v20.16b, v22.16b
        mov     v22.16b, v24.16b
        mov     v24.16b, v26.16b
        mov     v26.16b, v27.16b
        mov     v27.16b, v31.16b
        mov     v31.16b, v30.16b
        mov     v30.16b, v7.16b
        ldp     q6, q7, [x5, 96]
        str     q13, [sp, 80]
        ldr     q13, [sp, 112]
        sdot    v15.4s, v29.16b, v5.4b[1]
        sdot    v14.4s, v7.16b, v5.4b[1]
        sdot    v12.4s, v29.16b, v4.4b[1]
        sdot    v13.4s, v6.16b, v5.4b[1]
        str     q13, [sp, 112]
        ldr     q13, [sp, 96]
        sdot    v3.4s, v6.16b, v4.4b[1]
        sdot    v2.4s, v7.16b, v4.4b[1]
        sdot    v1.4s, v28.16b, v11.4b[1]
        sdot    v13.4s, v28.16b, v4.4b[1]
        str     q13, [sp, 96]
        sdot    v0.4s, v29.16b, v11.4b[1]
        sdot    v16.4s, v6.16b, v11.4b[1]
        sdot    v17.4s, v7.16b, v11.4b[1]
        sdot    v18.4s, v28.16b, v10.4b[1]
        sdot    v19.4s, v29.16b, v10.4b[1]
        sdot    v20.4s, v6.16b, v10.4b[1]
        sdot    v21.4s, v7.16b, v10.4b[1]
        sdot    v22.4s, v28.16b, v9.4b[1]
        sdot    v23.4s, v29.16b, v9.4b[1]
        sdot    v24.4s, v6.16b, v9.4b[1]
        sdot    v25.4s, v7.16b, v9.4b[1]
        sdot    v30.4s, v28.16b, v8.4b[1]
        sdot    v31.4s, v29.16b, v8.4b[1]
        sdot    v27.4s, v6.16b, v8.4b[1]
        sdot    v26.4s, v7.16b, v8.4b[1]
        mov     v6.16b, v25.16b
        mov     v25.16b, v23.16b
        mov     v23.16b, v21.16b
        mov     v21.16b, v19.16b
        mov     v19.16b, v17.16b
        mov     v17.16b, v0.16b
        mov     v0.16b, v2.16b
        mov     v2.16b, v12.16b
        mov     v12.16b, v14.16b
        mov     v14.16b, v15.16b
        mov     v7.16b, v30.16b
        mov     v30.16b, v31.16b
        mov     v31.16b, v27.16b
        mov     v27.16b, v26.16b
        mov     v26.16b, v24.16b
        mov     v24.16b, v22.16b
        mov     v22.16b, v20.16b
        mov     v20.16b, v18.16b
        mov     v18.16b, v16.16b
        mov     v16.16b, v1.16b
        mov     v1.16b, v3.16b
        ldp     q15, q3, [sp, 80]
        ldr     q13, [sp, 112]
        add     x5, x5, 0x80
        subs    x19, x19, 0x30
        b.ne    2b
        b       8f
7:
        add     x5, x5, 64
        subs    x19, x19, 0x30
        b.ne    2b
8:
        ldr     x2, [sp, 24]
        mov     v10.16b, v27.16b
        cmp     x1, 0xf
        mov     x13, x2
        ld1r    {v9.4s}, [x13], 4
        ld1r    {v8.4s}, [x13]
        sqrdmulh        v15.4s, v15.4s, v9.4s
        sqrdmulh        v14.4s, v14.4s, v9.4s
        sqrdmulh        v13.4s, v13.4s, v9.4s
        sqrdmulh        v12.4s, v12.4s, v9.4s
        sqrdmulh        v3.4s, v3.4s, v9.4s
        sqrdmulh        v2.4s, v2.4s, v9.4s
        sqrdmulh        v1.4s, v1.4s, v9.4s
        sqrdmulh        v0.4s, v0.4s, v9.4s
        sqrdmulh        v16.4s, v16.4s, v9.4s
        sqrdmulh        v17.4s, v17.4s, v9.4s
        sqrdmulh        v18.4s, v18.4s, v9.4s
        sqrdmulh        v19.4s, v19.4s, v9.4s
        sqrdmulh        v20.4s, v20.4s, v9.4s
        sqrdmulh        v21.4s, v21.4s, v9.4s
        sqrdmulh        v22.4s, v22.4s, v9.4s
        sqrdmulh        v23.4s, v23.4s, v9.4s
        sqrdmulh        v24.4s, v24.4s, v9.4s
        sqrdmulh        v25.4s, v25.4s, v9.4s
        sqrdmulh        v26.4s, v26.4s, v9.4s
        sqrdmulh        v27.4s, v6.4s, v9.4s
        sqrdmulh        v28.4s, v7.4s, v9.4s
        sqrdmulh        v29.4s, v30.4s, v9.4s
        sqrdmulh        v30.4s, v31.4s, v9.4s
        sqrdmulh        v31.4s, v10.4s, v9.4s
        cmeq    v9.4s, v8.4s, 0
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
        and     v9.16b, v23.16b, v10.16b
        ssra    v23.4s, v9.4s, 31
        and     v9.16b, v24.16b, v10.16b
        ssra    v24.4s, v9.4s, 31
        and     v9.16b, v25.16b, v10.16b
        ssra    v25.4s, v9.4s, 31
        and     v9.16b, v26.16b, v10.16b
        ssra    v26.4s, v9.4s, 31
        and     v9.16b, v27.16b, v10.16b
        ssra    v27.4s, v9.4s, 31
        and     v9.16b, v28.16b, v10.16b
        ssra    v28.4s, v9.4s, 31
        and     v9.16b, v29.16b, v10.16b
        add     x13, x2, 8
        ssra    v29.4s, v9.4s, 31
        and     v9.16b, v30.16b, v10.16b
        ssra    v30.4s, v9.4s, 31
        ld1r    {v9.8h}, [x13]
        and     v10.16b, v31.16b, v10.16b
        srshl   v15.4s, v15.4s, v8.4s
        srshl   v13.4s, v13.4s, v8.4s
        srshl   v3.4s, v3.4s, v8.4s
        srshl   v1.4s, v1.4s, v8.4s
        srshl   v16.4s, v16.4s, v8.4s
        srshl   v18.4s, v18.4s, v8.4s
        srshl   v20.4s, v20.4s, v8.4s
        srshl   v22.4s, v22.4s, v8.4s
        srshl   v24.4s, v24.4s, v8.4s
        srshl   v26.4s, v26.4s, v8.4s
        srshl   v28.4s, v28.4s, v8.4s
        ssra    v31.4s, v10.4s, 31
        srshl   v14.4s, v14.4s, v8.4s
        srshl   v12.4s, v12.4s, v8.4s
        srshl   v2.4s, v2.4s, v8.4s
        srshl   v0.4s, v0.4s, v8.4s
        srshl   v17.4s, v17.4s, v8.4s
        srshl   v19.4s, v19.4s, v8.4s
        srshl   v21.4s, v21.4s, v8.4s
        srshl   v23.4s, v23.4s, v8.4s
        srshl   v25.4s, v25.4s, v8.4s
        srshl   v27.4s, v27.4s, v8.4s
        srshl   v29.4s, v29.4s, v8.4s
        srshl   v30.4s, v30.4s, v8.4s
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
        sqxtn   v28.4h, v28.4s
        srshl   v31.4s, v31.4s, v8.4s
        sqxtn   v30.4h, v30.4s
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
        sqxtn2  v28.8h, v29.4s
        sqxtn2  v30.8h, v31.4s
        sqadd   v29.8h, v15.8h, v9.8h
        sqadd   v27.8h, v3.8h, v9.8h
        sqadd   v23.8h, v16.8h, v9.8h
        sqadd   v20.8h, v20.8h, v9.8h
        sqadd   v24.8h, v24.8h, v9.8h
        sqadd   v28.8h, v28.8h, v9.8h
        add     x13, x2, 10
        sqadd   v31.8h, v13.8h, v9.8h
        sqadd   v25.8h, v1.8h, v9.8h
        sqadd   v21.8h, v18.8h, v9.8h
        sqadd   v22.8h, v22.8h, v9.8h
        sqadd   v26.8h, v26.8h, v9.8h
        sqadd   v30.8h, v30.8h, v9.8h
        sqxtn   v29.8b, v29.8h
        sqxtn   v27.8b, v27.8h
        sqxtn   v23.8b, v23.8h
        sqxtn   v20.8b, v20.8h
        sqxtn   v24.8b, v24.8h
        sqxtn   v28.8b, v28.8h
        sqxtn2  v29.16b, v31.8h
        sqxtn2  v27.16b, v25.8h
        sqxtn2  v23.16b, v21.8h
        sqxtn2  v20.16b, v22.8h
        sqxtn2  v24.16b, v26.8h
        sqxtn2  v28.16b, v30.8h
        ld1r    {v30.16b}, [x13]
        add     x13, x2, 11
        ld1r    {v25.16b}, [x13]
        smax    v28.16b, v28.16b, v30.16b
        smax    v31.16b, v24.16b, v30.16b
        smax    v24.16b, v20.16b, v30.16b
        smax    v23.16b, v23.16b, v30.16b
        smax    v22.16b, v27.16b, v30.16b
        smax    v21.16b, v29.16b, v30.16b
        smin    v26.16b, v28.16b, v25.16b
        smin    v30.16b, v31.16b, v25.16b
        smin    v27.16b, v24.16b, v25.16b
        smin    v29.16b, v23.16b, v25.16b
        smin    v31.16b, v22.16b, v25.16b
        smin    v28.16b, v21.16b, v25.16b
        b.ls    9f
        str     q26, [x12]
        str     q30, [x11]
        str     q27, [x10]
        str     q29, [x9]
        str     q31, [x8]
        str     q28, [x6]
        ldp     x13, x3, [sp, 8]
        subs    x1, x1, 0x10
        add     x12, x12, x13
        add     x11, x11, x13
        add     x10, x10, x13
        add     x9, x9, x13
        add     x8, x8, x13
        add     x6, x6, x13
        sub     x4, x4, x3
        b.ne    1b
        b       14f
9:
        tbnz    w1, 3, 10f
        mov     v28.d[1], v31.d[0]
        mov     v29.d[1], v27.d[0]
        mov     v30.d[1], v26.d[0]
        tbnz    w1, 2, 11f
        b       12f
10:
        str     d26, [x12], 8
        str     d30, [x11], 8
        str     d27, [x10], 8
        str     d29, [x9], 8
        str     d31, [x8], 8
        str     d28, [x6], 8
        ext     v30.16b, v30.16b, v30.16b, 8
        ext     v26.16b, v26.16b, v26.16b, 8
        ext     v29.16b, v29.16b, v29.16b, 8
        ext     v27.16b, v27.16b, v27.16b, 8
        ext     v28.16b, v28.16b, v28.16b, 8
        ext     v31.16b, v31.16b, v31.16b, 8
        mov     v30.d[1], v26.d[0]
        mov     v29.d[1], v27.d[0]
        mov     v28.d[1], v31.d[0]
        tbz     w1, 2, 12f
11:
        st1     {v30.s}[2], [x12], 4
        st1     {v30.s}[0], [x11], 4
        st1     {v29.s}[2], [x10], 4
        st1     {v29.s}[0], [x9], 4
        st1     {v28.s}[2], [x8], 4
        st1     {v28.s}[0], [x6], 4
        ext     v30.16b, v30.16b, v30.16b, 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
12:
        tbnz    w1, 1, 15f
        tbz     w1, 0, 14f
13:
        st1     {v30.b}[8], [x12]
        st1     {v30.b}[0], [x11]
        st1     {v29.b}[8], [x10]
        st1     {v29.b}[0], [x9]
        st1     {v28.b}[8], [x8]
        st1     {v28.b}[0], [x6]
14:
        ldp     x29, x30, [sp, 272]
        ldp     x20, x19, [sp, 256]
        ldp     x22, x21, [sp, 240]
        ldp     x24, x23, [sp, 224]
        ldp     x26, x25, [sp, 208]
        ldp     x28, x27, [sp, 192]
        ldp     d9, d8, [sp, 176]
        ldp     d11, d10, [sp, 160]
        ldp     d5, d4, [sp, 144]
        ldp     d7, d6, [sp, 128]
        add     sp, sp, 0x120
        ret
15:
        st1     {v30.h}[4], [x12], 2
        st1     {v30.h}[0], [x11], 2
        st1     {v29.h}[4], [x10], 2
        st1     {v29.h}[0], [x9], 2
        st1     {v28.h}[4], [x8], 2
        st1     {v28.h}[0], [x6], 2
        ext     v30.16b, v30.16b, v30.16b, 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 13b
        b       14b

END_FUNCTION xnn_qs8_igemm_minmax_ukernel_6x16c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

