#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

### Tests for VUnary micro-kernels
tools/generate-vunary-benchmark.py --spec test/f16-vabs.yaml --output bench/f16-vabs.cc &
tools/generate-vunary-benchmark.py --spec test/f16-vclamp.yaml --output bench/f16-vclamp.cc &
tools/generate-vunary-benchmark.py --spec test/f16-velu.yaml --output bench/f16-velu.cc &
tools/generate-vunary-benchmark.py --spec test/f16-vneg.yaml --output bench/f16-vneg.cc &
tools/generate-vunary-benchmark.py --spec test/f16-vrndd.yaml  --output bench/f16-vrndd.cc &
tools/generate-vunary-benchmark.py --spec test/f16-vrndne.yaml --output bench/f16-vrndne.cc &
tools/generate-vunary-benchmark.py --spec test/f16-vrndu.yaml  --output bench/f16-vrndu.cc &
tools/generate-vunary-benchmark.py --spec test/f16-vrndz.yaml  --output bench/f16-vrndz.cc &
tools/generate-vunary-benchmark.py --spec test/f16-vrsqrt.yaml --output bench/f16-vrsqrt.cc &
tools/generate-vunary-benchmark.py --spec test/f16-vsigmoid.yaml --output bench/f16-vsigmoid.cc &
tools/generate-vunary-benchmark.py --spec test/f16-vsqr.yaml --output bench/f16-vsqr.cc &
tools/generate-vunary-benchmark.py --spec test/f16-vsqrt.yaml --output bench/f16-vsqrt.cc &
tools/generate-vunary-benchmark.py --spec test/f16-vtanh.yaml --output bench/f16-vtanh.cc &

tools/generate-vunary-benchmark.py --spec test/f32-vabs.yaml --output bench/f32-vabs.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vclamp.yaml --output bench/f32-vclamp.cc &
tools/generate-vunary-benchmark.py --spec test/f32-velu.yaml --output bench/f32-velu.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vgelu.yaml --output bench/f32-vgelu.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vneg.yaml --output bench/f32-vneg.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vrelu.yaml --output bench/f32-vrelu.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vrndd.yaml  --output bench/f32-vrndd.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vrndne.yaml --output bench/f32-vrndne.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vrndu.yaml  --output bench/f32-vrndu.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vrndz.yaml  --output bench/f32-vrndz.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vrsqrt.yaml --output bench/f32-vrsqrt.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vsigmoid.yaml --output bench/f32-vsigmoid.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vsqr.yaml --output bench/f32-vsqr.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vsqrt.yaml --output bench/f32-vsqrt.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vtanh.yaml --output bench/f32-vtanh.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vlog.yaml --output bench/f32-vlog.cc &

### Tests for VLRelu micro-kernels
tools/generate-vunary-benchmark.py --spec test/f16-vlrelu.yaml --output bench/f16-vlrelu.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vlrelu.yaml --output bench/f32-vlrelu.cc &

### Tests for VHSwish micro-kernels
tools/generate-vunary-benchmark.py --spec test/f16-vhswish.yaml --output bench/f16-vhswish.cc &
tools/generate-vunary-benchmark.py --spec test/f32-vhswish.yaml --output bench/f32-vhswish.cc &

### Tests for Rsum micro-kernels
tools/generate-rdsum-benchmark.py  --spec test/f32-rdsum.yaml --output bench/f32-rdsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/f16-f32acc-rdsum.yaml --output bench/f16-f32acc-rdsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/qs8-rdsum-minmax-fp32.yaml --output bench/qs8-rdsum.cc &

tools/generate-rdsum-benchmark.py  --spec test/f16-rsum.yaml --output bench/f16-rsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/f16-f32acc-rsum.yaml --output bench/f16-f32acc-rsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/f32-rsum.yaml --output bench/f32-rsum.cc &
tools/generate-rdsum-benchmark.py  --spec test/qs8-rsum.yaml --output bench/qs8-rsum.cc &
wait
