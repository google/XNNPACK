#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### x86 AVX ###################################
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-div-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-div-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-div-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=32 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-div-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=40 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-div-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=48 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-div-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=56 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-div-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=64 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-div-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=72 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-div-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=80 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-div-x80.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=8  -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-rcp-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=16 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-rcp-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=24 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-rcp-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=32 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-rcp-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=40 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-rcp-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=48 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-rcp-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=56 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-rcp-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=64 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-rcp-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=72 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-rcp-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=80 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=0 -o src/f16-vtanh/gen/f16-vtanh-f16c-expm1minus-rr1-p3h2-rcp-x80.c &

tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-div-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-div-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-div-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=32 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-div-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=40 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-div-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=48 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-div-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=56 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-div-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=64 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-div-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=72 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-div-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=80 -D DIV=DIV -D SAT=MINMAX -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-div-x80.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=8  -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-rcp-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=16 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-rcp-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=24 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-rcp-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=32 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-rcp-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=40 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-rcp-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=48 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-rcp-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=56 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-rcp-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=64 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-rcp-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=72 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-rcp-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=80 -D DIV=RCP -D SAT=SELECT -D AVX=1 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-fma3-expm1minus-rr1-p3h2-rcp-x80.c &

tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-div-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-div-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-div-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=32 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-div-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=40 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-div-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=48 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-div-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=56 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-div-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=64 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-div-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=72 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-div-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=80 -D DIV=DIV -D SAT=MINMAX -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-div-x80.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=8  -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-rcp-x8.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=16 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-rcp-x16.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=24 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-rcp-x24.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=32 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-rcp-x32.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=40 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-rcp-x40.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=48 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-rcp-x48.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=56 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-rcp-x56.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=64 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-rcp-x64.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=72 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-rcp-x72.c &
tools/xngen src/f16-vtanh/avx-expm1minus.c.in -D P=3 -D H=2 -D BATCH_TILE=80 -D DIV=RCP -D SAT=SELECT -D AVX=2 -D FMA=3 -o src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2-rcp-x80.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f16-vtanh.yaml --output test/f16-vtanh.cc &

wait
