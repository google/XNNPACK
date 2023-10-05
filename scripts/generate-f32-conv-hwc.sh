#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-conv-hwc/3x3s2p1c3-neon-x1.c.in -D CHANNEL_TILE=4 -D HEIGHT_TILE=2 -D FMA=1 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p1c3x4-aarch64-neonfma-2x1.c &
tools/xngen src/f32-conv-hwc/3x3s2p1c3-neon-x1.c.in -D CHANNEL_TILE=8 -D HEIGHT_TILE=2 -D FMA=1 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p1c3x8-aarch64-neonfma-2x1.c &
tools/xngen src/f32-conv-hwc/3x3s2p1c3-neon-x2.c.in -D CHANNEL_TILE=4 -D HEIGHT_TILE=2 -D FMA=1 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p1c3x4-aarch64-neonfma-2x2.c &
tools/xngen src/f32-conv-hwc/3x3s2p1c3-neon-x2.c.in -D CHANNEL_TILE=8 -D HEIGHT_TILE=2 -D FMA=1 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p1c3x8-aarch64-neonfma-2x2.c &

tools/xngen src/f32-conv-hwc/3x3s2p0p1c3-neon-x1.c.in -D CHANNEL_TILE=4 -D HEIGHT_TILE=2 -D FMA=1 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p0p1c3x4-aarch64-neonfma-2x1.c &
tools/xngen src/f32-conv-hwc/3x3s2p0p1c3-neon-x1.c.in -D CHANNEL_TILE=8 -D HEIGHT_TILE=2 -D FMA=1 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p0p1c3x8-aarch64-neonfma-2x1.c &
tools/xngen src/f32-conv-hwc/3x3s2p0p1c3-neon-x2.c.in -D CHANNEL_TILE=4 -D HEIGHT_TILE=2 -D FMA=1 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p0p1c3x4-aarch64-neonfma-2x2.c &
tools/xngen src/f32-conv-hwc/3x3s2p0p1c3-neon-x2.c.in -D CHANNEL_TILE=8 -D HEIGHT_TILE=2 -D FMA=1 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p0p1c3x8-aarch64-neonfma-2x2.c &

tools/xngen src/f32-conv-hwc/3x3s2p1c3-neon-x1.c.in -D CHANNEL_TILE=4 -D HEIGHT_TILE=2 -D FMA=0 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p1c3x4-neon-2x1.c &
tools/xngen src/f32-conv-hwc/3x3s2p1c3-neon-x1.c.in -D CHANNEL_TILE=8 -D HEIGHT_TILE=2 -D FMA=0 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p1c3x8-neon-2x1.c &
tools/xngen src/f32-conv-hwc/3x3s2p1c3-neon-x2.c.in -D CHANNEL_TILE=4 -D HEIGHT_TILE=2 -D FMA=0 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p1c3x4-neon-2x2.c &
tools/xngen src/f32-conv-hwc/3x3s2p1c3-neon-x2.c.in -D CHANNEL_TILE=8 -D HEIGHT_TILE=2 -D FMA=0 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p1c3x8-neon-2x2.c &

tools/xngen src/f32-conv-hwc/3x3s2p0p1c3-neon-x1.c.in -D CHANNEL_TILE=4 -D HEIGHT_TILE=2 -D FMA=0 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p0p1c3x4-neon-2x1.c &
tools/xngen src/f32-conv-hwc/3x3s2p0p1c3-neon-x1.c.in -D CHANNEL_TILE=8 -D HEIGHT_TILE=2 -D FMA=0 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p0p1c3x8-neon-2x1.c &
tools/xngen src/f32-conv-hwc/3x3s2p0p1c3-neon-x2.c.in -D CHANNEL_TILE=4 -D HEIGHT_TILE=2 -D FMA=0 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p0p1c3x4-neon-2x2.c &
tools/xngen src/f32-conv-hwc/3x3s2p0p1c3-neon-x2.c.in -D CHANNEL_TILE=8 -D HEIGHT_TILE=2 -D FMA=0 -o src/f32-conv-hwc/gen/f32-conv-hwc-3x3s2p0p1c3x8-neon-2x2.c &

wait
