#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### x86 AVX ###################################
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-div-x80.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=40 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=48 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=56 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=64 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=72 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=80 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2ts-rcp-x80.c &

tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-div-x80.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=40 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=48 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=56 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=64 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=72 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=80 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2ts-rcp-x80.c &

tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-x80.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=40 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=48 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=56 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=64 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=72 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=80 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-x80.c &

tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=8  -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9-x8.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=16 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9-x16.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=24 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9-x24.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=32 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9-x32.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=40 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9-x40.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=48 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9-x48.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=56 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9-x56.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=64 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9-x64.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=72 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9-x72.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=80 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-polynomial-p19h9-x80.c &

tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=8  -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9-x8.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=16 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9-x16.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=24 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9-x24.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=32 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9-x32.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=40 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9-x40.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=48 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9-x48.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=56 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9-x56.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=64 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9-x64.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=72 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9-x72.c &
tools/xngen src/f16-vtanh/avx-polynomial.c.in -D P=19 -D H=9 -D BATCH_TILE=80 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-polynomial-p19h9-x80.c &

tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-x8.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-x16.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-x24.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-x32.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=40 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-x40.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=48 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-x48.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=56 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-x56.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=64 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-x64.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=72 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-x72.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=80 -D SAT=MINMAX -D DIV=DIV      -o src/f16-vtanh/gen/f16-vtanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div-x80.c &

tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-x8.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-x16.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-x24.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-x32.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=40 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-x40.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=48 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-x48.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=56 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-x56.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=64 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-x64.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=72 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-x72.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=80 -D SAT=MINMAX -D DIV=NR1FMA   -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma-x80.c &

tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-x8.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-x16.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-x24.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-x32.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=40 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-x40.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=48 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-x48.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=56 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-x56.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=64 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-x64.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=72 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-x72.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=80 -D SAT=MINMAX -D DIV=NR1RECPS -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps-x80.c &

tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=8  -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-x8.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=16 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-x16.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=24 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-x24.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=32 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-x32.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=40 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-x40.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=48 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-x48.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=56 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-x56.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=64 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-x64.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=72 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-x72.c &
tools/xngen src/f16-vtanh/neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D PS=0 -D BATCH_TILE=80 -D SAT=MINMAX -D DIV=RECPEADJ -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj-x80.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f16-vtanh.yaml --output test/f16-vtanh.cc &

wait
