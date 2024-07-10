#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-rsum/scalar.c.in -D CHANNEL_TILE=1  -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-scalar-u1.c &
tools/xngen src/qs8-rsum/scalar.c.in -D CHANNEL_TILE=2  -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-scalar-u2.c &
tools/xngen src/qs8-rsum/scalar.c.in -D CHANNEL_TILE=4  -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-scalar-u4.c &

################################## ARM NEON ###################################
tools/xngen src/qs8-rsum/neon.c.in   -D CHANNEL_TILE=16 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-neon-u16.c &
tools/xngen src/qs8-rsum/neon.c.in   -D CHANNEL_TILE=32 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-neon-u32.c &
tools/xngen src/qs8-rsum/neon.c.in   -D CHANNEL_TILE=32 -D ACCUMULATORS=2 -o src/qs8-rsum/gen/qs8-rsum-neon-u32-acc2.c &
tools/xngen src/qs8-rsum/neon.c.in   -D CHANNEL_TILE=64 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-neon-u64.c &
tools/xngen src/qs8-rsum/neon.c.in   -D CHANNEL_TILE=64 -D ACCUMULATORS=2 -o src/qs8-rsum/gen/qs8-rsum-neon-u64-acc2.c &
tools/xngen src/qs8-rsum/neon.c.in   -D CHANNEL_TILE=64 -D ACCUMULATORS=4 -o src/qs8-rsum/gen/qs8-rsum-neon-u64-acc4.c &

################################## ARM NEONDOT ################################
tools/xngen src/qs8-rsum/neondot.c.in -D CHANNEL_TILE=16 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-neondot-u16.c &
tools/xngen src/qs8-rsum/neondot.c.in -D CHANNEL_TILE=32 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-neondot-u32.c &
tools/xngen src/qs8-rsum/neondot.c.in -D CHANNEL_TILE=32 -D ACCUMULATORS=2 -o src/qs8-rsum/gen/qs8-rsum-neondot-u32-acc2.c &
tools/xngen src/qs8-rsum/neondot.c.in -D CHANNEL_TILE=64 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-neondot-u64.c &
tools/xngen src/qs8-rsum/neondot.c.in -D CHANNEL_TILE=64 -D ACCUMULATORS=2 -o src/qs8-rsum/gen/qs8-rsum-neondot-u64-acc2.c &
tools/xngen src/qs8-rsum/neondot.c.in -D CHANNEL_TILE=64 -D ACCUMULATORS=4 -o src/qs8-rsum/gen/qs8-rsum-neondot-u64-acc4.c &

################################### x86 SSSE3 #################################
tools/xngen src/qs8-rsum/ssse3.c.in -D CHANNEL_TILE=16 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-ssse3-u16.c &
tools/xngen src/qs8-rsum/ssse3.c.in -D CHANNEL_TILE=32 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-ssse3-u32.c &
tools/xngen src/qs8-rsum/ssse3.c.in -D CHANNEL_TILE=32 -D ACCUMULATORS=2 -o src/qs8-rsum/gen/qs8-rsum-ssse3-u32-acc2.c &
tools/xngen src/qs8-rsum/ssse3.c.in -D CHANNEL_TILE=64 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-ssse3-u64.c &
tools/xngen src/qs8-rsum/ssse3.c.in -D CHANNEL_TILE=64 -D ACCUMULATORS=2 -o src/qs8-rsum/gen/qs8-rsum-ssse3-u64-acc2.c &
tools/xngen src/qs8-rsum/ssse3.c.in -D CHANNEL_TILE=64 -D ACCUMULATORS=4 -o src/qs8-rsum/gen/qs8-rsum-ssse3-u64-acc4.c &

################################### x86 AVX2 ##################################
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=32  -D ACCUMULATORS=1 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avx2-u32.c &
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=64  -D ACCUMULATORS=1 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avx2-u64.c &
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=64  -D ACCUMULATORS=2 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avx2-u64-acc2.c &
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=1 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avx2-u128.c &
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=2 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avx2-u128-acc2.c &
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=4 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avx2-u128-acc4.c &

################################### x86 AVX256SKX ##################################
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=32  -D ACCUMULATORS=1 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256skx-u32.c &
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=64  -D ACCUMULATORS=1 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256skx-u64.c &
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=64  -D ACCUMULATORS=2 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256skx-u64-acc2.c &
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=1 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256skx-u128.c &
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=2 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256skx-u128-acc2.c &
tools/xngen src/qs8-rsum/avx2.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=4 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256skx-u128-acc4.c &

################################### x86 AVX512SKX ##############################
tools/xngen src/qs8-rsum/avx512skx.c.in -D CHANNEL_TILE=64  -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-avx512skx-u64.c &
tools/xngen src/qs8-rsum/avx512skx.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-avx512skx-u128.c &
tools/xngen src/qs8-rsum/avx512skx.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=2 -o src/qs8-rsum/gen/qs8-rsum-avx512skx-u128-acc2.c &
tools/xngen src/qs8-rsum/avx512skx.c.in -D CHANNEL_TILE=256 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-avx512skx-u256.c &
tools/xngen src/qs8-rsum/avx512skx.c.in -D CHANNEL_TILE=256 -D ACCUMULATORS=2 -o src/qs8-rsum/gen/qs8-rsum-avx512skx-u256-acc2.c &
tools/xngen src/qs8-rsum/avx512skx.c.in -D CHANNEL_TILE=256 -D ACCUMULATORS=4 -o src/qs8-rsum/gen/qs8-rsum-avx512skx-u256-acc4.c &

################################### x86 AVX512VNNI #############################
tools/xngen src/qs8-rsum/avx512vnni.c.in -D CHANNEL_TILE=64  -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-avx512vnni-u64.c &
tools/xngen src/qs8-rsum/avx512vnni.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-avx512vnni-u128.c &
tools/xngen src/qs8-rsum/avx512vnni.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=2 -o src/qs8-rsum/gen/qs8-rsum-avx512vnni-u128-acc2.c &
tools/xngen src/qs8-rsum/avx512vnni.c.in -D CHANNEL_TILE=256 -D ACCUMULATORS=1 -o src/qs8-rsum/gen/qs8-rsum-avx512vnni-u256.c &
tools/xngen src/qs8-rsum/avx512vnni.c.in -D CHANNEL_TILE=256 -D ACCUMULATORS=2 -o src/qs8-rsum/gen/qs8-rsum-avx512vnni-u256-acc2.c &
tools/xngen src/qs8-rsum/avx512vnni.c.in -D CHANNEL_TILE=256 -D ACCUMULATORS=4 -o src/qs8-rsum/gen/qs8-rsum-avx512vnni-u256-acc4.c &

################################### x86 AVXVNNI #############################
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=32  -D ACCUMULATORS=1 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avxvnni-u32.c &
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=64  -D ACCUMULATORS=1 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avxvnni-u64.c &
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=64  -D ACCUMULATORS=2 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avxvnni-u64-acc2.c &
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=1 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avxvnni-u128.c &
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=2 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avxvnni-u128-acc2.c &
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=4 -D AVX=2 -o src/qs8-rsum/gen/qs8-rsum-avxvnni-u128-acc4.c &

################################### x86 AVX256VNNI #############################
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=32  -D ACCUMULATORS=1 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256vnni-u32.c &
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=64  -D ACCUMULATORS=1 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256vnni-u64.c &
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=64  -D ACCUMULATORS=2 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256vnni-u64-acc2.c &
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=1 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256vnni-u128.c &
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=2 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256vnni-u128-acc2.c &
tools/xngen src/qs8-rsum/avxvnni.c.in -D CHANNEL_TILE=128 -D ACCUMULATORS=4 -D AVX=10 -o src/qs8-rsum/gen/qs8-rsum-avx256vnni-u128-acc4.c &

wait
