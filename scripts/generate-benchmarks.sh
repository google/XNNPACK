#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

### Tests for VUnary micro-kernels
tools/generate-vunary-benchmark.py --ukernel f16-vabs --output bench/f16-vabs.cc &
tools/generate-vunary-benchmark.py --ukernel f16-vclamp --output bench/f16-vclamp.cc &
tools/generate-vunary-benchmark.py --ukernel f16-velu --output bench/f16-velu.cc &
tools/generate-vunary-benchmark.py --ukernel f16-vneg --output bench/f16-vneg.cc &
tools/generate-vunary-benchmark.py --ukernel f16-vrndd  --output bench/f16-vrndd.cc &
tools/generate-vunary-benchmark.py --ukernel f16-vrndne --output bench/f16-vrndne.cc &
tools/generate-vunary-benchmark.py --ukernel f16-vrndu  --output bench/f16-vrndu.cc &
tools/generate-vunary-benchmark.py --ukernel f16-vrndz  --output bench/f16-vrndz.cc &
tools/generate-vunary-benchmark.py --ukernel f16-vrsqrt --output bench/f16-vrsqrt.cc &
tools/generate-vunary-benchmark.py --ukernel f16-vsigmoid --output bench/f16-vsigmoid.cc &
tools/generate-vunary-benchmark.py --ukernel f16-vsqr --output bench/f16-vsqr.cc &
tools/generate-vunary-benchmark.py --ukernel f16-vsqrt --output bench/f16-vsqrt.cc &
tools/generate-vunary-benchmark.py --ukernel f16-vtanh --output bench/f16-vtanh.cc &

tools/generate-vunary-benchmark.py --ukernel f32-vabs --output bench/f32-vabs.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vclamp --output bench/f32-vclamp.cc &
tools/generate-vunary-benchmark.py --ukernel f32-velu --output bench/f32-velu.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vgelu --output bench/f32-vgelu.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vneg --output bench/f32-vneg.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vrelu --output bench/f32-vrelu.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vrndd  --output bench/f32-vrndd.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vrndne --output bench/f32-vrndne.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vrndu  --output bench/f32-vrndu.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vrndz  --output bench/f32-vrndz.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vrsqrt --output bench/f32-vrsqrt.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vsigmoid --output bench/f32-vsigmoid.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vsqr --output bench/f32-vsqr.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vsqrt --output bench/f32-vsqrt.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vtanh --output bench/f32-vtanh.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vlog --output bench/f32-vlog.cc &

### Tests for VLRelu micro-kernels
tools/generate-vunary-benchmark.py --ukernel f16-vlrelu --output bench/f16-vlrelu.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vlrelu --output bench/f32-vlrelu.cc &

### Tests for VHSwish micro-kernels
tools/generate-vunary-benchmark.py --ukernel f16-vhswish --output bench/f16-vhswish.cc &
tools/generate-vunary-benchmark.py --ukernel f32-vhswish --output bench/f32-vhswish.cc &

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
