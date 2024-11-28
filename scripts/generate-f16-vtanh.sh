#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### x86 AVX ###################################
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-u8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-u16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-u24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-u32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-u8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-u16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-u24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-u32.c &

tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-u8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-u16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-u24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-u32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-u8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-u16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-u24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-u32.c &

tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u32.c &

tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=8  -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9t2-u8.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=16 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9t2-u16.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=24 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9t2-u24.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=32 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9t2-u32.c &

tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=8  -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9t2-u8.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=16 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9t2-u16.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=24 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9t2-u24.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=32 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9t2-u32.c &

tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-u8.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-u16.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-u24.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-u32.c &

tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-u8.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-u16.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-u24.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-u32.c &

tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-u8.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-u16.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-u24.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-u32.c &

tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-u8.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-u16.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-u24.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-u32.c &

wait
