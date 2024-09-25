#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

### Tests for Rsum micro-kernels
tools/generate-rdsum-benchmark.py  --spec test/f32-rdsum.yaml --output bench/f32-rdsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/f16-f32acc-rdsum.yaml --output bench/f16-f32acc-rdsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/qs8-rdsum-minmax-fp32.yaml --output bench/qs8-rdsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/qu8-rdsum.yaml --output bench/qu8-rdsum.cc &

tools/generate-rdsum-benchmark.py  --spec test/f16-rsum.yaml --output bench/f16-rsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/f16-f32acc-rsum.yaml --output bench/f16-f32acc-rsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/f32-rsum.yaml --output bench/f32-rsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/qs8-rsum.yaml --output bench/qs8-rsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/qu8-rsum.yaml --output bench/qu8-rsum.cc &
wait
