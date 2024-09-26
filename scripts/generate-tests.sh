#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

### Tests for GAvgPool micro-kernels
tools/generate-gavgpool-test.py --spec test/f16-gavgpool-minmax.yaml --output test/f16-gavgpool-minmax.cc &
tools/generate-gavgpool-test.py --spec test/f32-gavgpool-minmax.yaml --output test/f32-gavgpool-minmax.cc &
tools/generate-gavgpool-test.py --spec test/qs8-gavgpool-minmax-fp32.yaml --output test/qs8-gavgpool-minmax-fp32.cc &
tools/generate-gavgpool-test.py --spec test/qs8-gavgpool-minmax-rndnu.yaml --output test/qs8-gavgpool-minmax-rndnu.cc &
tools/generate-gavgpool-test.py --spec test/qu8-gavgpool-minmax-fp32.yaml --output test/qu8-gavgpool-minmax-fp32.cc &
tools/generate-gavgpool-test.py --spec test/qu8-gavgpool-minmax-rndnu.yaml --output test/qu8-gavgpool-minmax-rndnu.cc &

### Tests for GAvgPool CW layout micro-kernels
tools/generate-gavgpool-cw-test.py --spec test/f16-gavgpool-cw.yaml --output test/f16-gavgpool-cw.cc &
tools/generate-gavgpool-cw-test.py --spec test/f32-gavgpool-cw.yaml --output test/f32-gavgpool-cw.cc &

### Tests for ArgMaxPool micro-kernels
tools/generate-argmaxpool-test.py --spec test/f32-argmaxpool.yaml --output test/f32-argmaxpool.cc &

### Tests for GEMM micro-kernels
tools/generate-gemm-test.py --spec test/bf16-gemm-minmax.yaml --output-test test/bf16-gemm-minmax.cc &

tools/generate-gemm-test.py --spec test/f16-gemm-minmax.yaml        --output-test test/f16-gemm-minmax.cc --output-bench bench/f16-gemm-minmax.cc &
tools/generate-gemm-test.py --spec test/f16-f32acc-gemm-minmax.yaml --output-test test/f16-f32acc-gemm-minmax.cc &

tools/generate-gemm-test.py --spec test/f32-gemm.yaml            --output-test test/f32-gemm.cc           --output-test test/f32-gemm-2.cc &
tools/generate-gemm-test.py --spec test/f32-gemm-relu.yaml       --output-test test/f32-gemm-relu.cc      --output-test test/f32-gemm-relu-2.cc &
tools/generate-gemm-test.py --spec test/f32-gemm-minmax.yaml     --output-test test/f32-gemm-minmax.cc    --output-test test/f32-gemm-minmax-2.cc --output-bench bench/f32-gemm-minmax.cc &
tools/generate-gemm-test.py --spec test/f32-gemminc-minmax.yaml  --output-test test/f32-gemminc-minmax.cc --output-test test/f32-gemminc-minmax-2.cc &
tools/generate-gemm-test.py --spec test/f32-gemm-goi-minmax.yaml --output-test test/f32-gemm-goi-minmax.cc --output-bench bench/f32-gemm-goi-minmax.cc &

tools/generate-gemm-test.py --spec test/f32-qc4w-gemm-minmax.yaml --output-test test/f32-qc4w-gemm-minmax.cc &

tools/generate-gemm-test.py --spec test/f32-qc8w-gemm.yaml        --output-test test/f32-qc8w-gemm.cc        &
tools/generate-gemm-test.py --spec test/f32-qc8w-gemm-relu.yaml   --output-test test/f32-qc8w-gemm-relu.cc   &
tools/generate-gemm-test.py --spec test/f32-qc8w-gemm-minmax.yaml --output-test test/f32-qc8w-gemm-minmax.cc &

tools/generate-gemm-test.py --spec test/qu8-gemm-minmax-fp32.yaml --output-test test/qu8-gemm-minmax-fp32.cc --output-test test/qu8-gemm-minmax-fp32-2.cc --output-bench bench/qu8-gemm-fp32.cc &
tools/generate-gemm-test.py --spec test/qu8-gemm-minmax-rndnu.yaml --output-test test/qu8-gemm-minmax-rndnu.cc --output-test test/qu8-gemm-minmax-rndnu-2.cc --output-bench bench/qu8-gemm-rndnu.cc &

tools/generate-gemm-test.py --spec test/qd8-f16-qc4w-gemm-minmax.yaml --output-test test/qd8-f16-qc4w-gemm-minmax.cc  --output-test test/qd8-f16-qc4w-gemm-minmax-2.cc  --output-test test/qd8-f16-qc4w-gemm-minmax-3.cc  --output-test test/qd8-f16-qc4w-gemm-minmax-4.cc --output-bench bench/qd8-f16-qc4w-gemm.cc &
tools/generate-gemm-test.py --spec test/qd8-f16-qb4w-gemm-minmax.yaml --output-test test/qd8-f16-qb4w-gemm-minmax.cc --output-bench bench/qd8-f16-qb4w-gemm.cc &
tools/generate-gemm-test.py --spec test/qd8-f16-qc8w-gemm-minmax.yaml --output-test test/qd8-f16-qc8w-gemm-minmax.cc --output-test test/qd8-f16-qc8w-gemm-minmax-2.cc --output-test test/qd8-f16-qc8w-gemm-minmax-3.cc --output-test test/qd8-f16-qc8w-gemm-minmax-4.cc --output-bench bench/qd8-f16-qc8w-gemm.cc &
tools/generate-gemm-test.py --spec test/qd8-f32-qc8w-gemm-minmax.yaml --output-test test/qd8-f32-qc8w-gemm-minmax.cc  --output-test test/qd8-f32-qc8w-gemm-minmax-2.cc  --output-test test/qd8-f32-qc8w-gemm-minmax-3.cc  --output-test test/qd8-f32-qc8w-gemm-minmax-4.cc --output-bench bench/qd8-f32-qc8w-gemm.cc &
tools/generate-gemm-test.py --spec test/qd8-f32-qc4w-gemm-minmax.yaml --output-test test/qd8-f32-qc4w-gemm-minmax.cc  --output-test test/qd8-f32-qc4w-gemm-minmax-2.cc  --output-test test/qd8-f32-qc4w-gemm-minmax-3.cc  --output-test test/qd8-f32-qc4w-gemm-minmax-4.cc --output-bench bench/qd8-f32-qc4w-gemm.cc &
tools/generate-gemm-test.py --spec test/qd8-f32-qb4w-gemm-minmax.yaml --output-test test/qd8-f32-qb4w-gemm-minmax.cc --output-bench bench/qd8-f32-qb4w-gemm.cc &

tools/generate-gemm-test.py --spec test/qp8-f32-qc4w-gemm-minmax.yaml --output-test test/qp8-f32-qc4w-gemm-minmax.cc --output-bench bench/qp8-f32-qc4w-gemm.cc &
tools/generate-gemm-test.py --spec test/qp8-f32-qb4w-gemm-minmax.yaml --output-test test/qp8-f32-qb4w-gemm-minmax.cc --output-bench bench/qp8-f32-qb4w-gemm.cc &

tools/generate-gemm-test.py --spec test/qs8-qc8w-gemm-minmax-fp32.yaml --output-test test/qs8-qc8w-gemm-minmax-fp32.cc --output-test test/qs8-qc8w-gemm-minmax-fp32-2.cc --output-test test/qs8-qc8w-gemm-minmax-fp32-3.cc --output-bench bench/qs8-qc8w-gemm-fp32.cc &

### Tests for IGEMM micro-kernels
tools/generate-gemm-test.py --spec test/f16-igemm-minmax.yaml --output-test test/f16-igemm-minmax.cc &
tools/generate-gemm-test.py --spec test/f16-f32acc-igemm-minmax.yaml --output-test test/f16-f32acc-igemm-minmax.cc &

tools/generate-gemm-test.py --spec test/f32-igemm.yaml --output-test test/f32-igemm.cc --output-test test/f32-igemm-2.cc &
tools/generate-gemm-test.py --spec test/f32-igemm-relu.yaml --output-test test/f32-igemm-relu.cc --output-test test/f32-igemm-relu-2.cc &
tools/generate-gemm-test.py --spec test/f32-igemm-minmax.yaml --output-test test/f32-igemm-minmax.cc --output-test test/f32-igemm-minmax-2.cc &

tools/generate-gemm-test.py --spec test/qd8-f16-qc8w-igemm-minmax.yaml --output-test test/qd8-f16-qc8w-igemm-minmax.cc --output-test test/qd8-f16-qc8w-igemm-minmax-2.cc --output-test test/qd8-f16-qc8w-igemm-minmax-3.cc --output-test test/qd8-f16-qc8w-igemm-minmax-4.cc &
tools/generate-gemm-test.py --spec test/qd8-f32-qc8w-igemm-minmax.yaml --output-test test/qd8-f32-qc8w-igemm-minmax.cc --output-test test/qd8-f32-qc8w-igemm-minmax-2.cc --output-test test/qd8-f32-qc8w-igemm-minmax-3.cc &

tools/generate-gemm-test.py --spec test/qu8-igemm-minmax-fp32.yaml --output-test test/qu8-igemm-minmax-fp32.cc --output-test test/qu8-igemm-minmax-fp32-2.cc &
tools/generate-gemm-test.py --spec test/qu8-igemm-minmax-rndnu.yaml --output-test test/qu8-igemm-minmax-rndnu.cc --output-test test/qu8-igemm-minmax-rndnu-2.cc &

tools/generate-gemm-test.py --spec test/qs8-qc8w-igemm-minmax-fp32.yaml --output-test test/qs8-qc8w-igemm-minmax-fp32.cc --output-test test/qs8-qc8w-igemm-minmax-fp32-2.cc --output-test test/qs8-qc8w-igemm-minmax-fp32-3.cc &

### Tests for PPMM micro-kernels
tools/generate-gemm-test.py --spec test/f32-ppmm-minmax.yaml --output-test test/f32-ppmm-minmax.cc &

### Tests for SPMM micro-kernels
tools/generate-spmm-test.py --spec test/f16-spmm-minmax.yaml --output-test test/f16-spmm-minmax.cc &
tools/generate-spmm-test.py --spec test/f32-spmm-minmax.yaml --output-test test/f32-spmm-minmax.cc  --output-test test/f32-spmm-minmax-2.cc  --output-test test/f32-spmm-minmax-3.cc  --output-test test/f32-spmm-minmax-4.cc --output-bench bench/f32-spmm.cc &

### Tests for VBinary micro-kernels
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f16-vadd --output test/f16-vadd.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f16-vdiv --output test/f16-vdiv.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f16-vmax --output test/f16-vmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f16-vmin --output test/f16-vmin.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f16-vmul --output test/f16-vmul.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f16-vprelu --output test/f16-vprelu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f16-vsqrdiff --output test/f16-vsqrdiff.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f16-vsub --output test/f16-vsub.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f16-vaddc --output test/f16-vaddc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f16-vdivc --output test/f16-vdivc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f16-vrdivc --output test/f16-vrdivc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f16-vmaxc --output test/f16-vmaxc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f16-vminc --output test/f16-vminc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f16-vmulc --output test/f16-vmulc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f16-vpreluc --output test/f16-vpreluc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f16-vrpreluc --output test/f16-vrpreluc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f16-vsqrdiffc --output test/f16-vsqrdiffc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f16-vsubc --output test/f16-vsubc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f16-vrsubc --output test/f16-vrsubc.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f32-vadd        --output test/f32-vadd.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f32-vcopysign   --output test/f32-vcopysign.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f32-vdiv        --output test/f32-vdiv.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f32-vmax        --output test/f32-vmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f32-vmin        --output test/f32-vmin.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f32-vmul        --output test/f32-vmul.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f32-vprelu      --output test/f32-vprelu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f32-vsqrdiff    --output test/f32-vsqrdiff.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel f32-vsub        --output test/f32-vsub.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vaddc         --output test/f32-vaddc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vcopysignc    --output test/f32-vcopysignc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vdivc         --output test/f32-vdivc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vmaxc         --output test/f32-vmaxc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vminc         --output test/f32-vminc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vmulc         --output test/f32-vmulc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vpreluc       --output test/f32-vpreluc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vrpreluc      --output test/f32-vrpreluc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vrcopysignc   --output test/f32-vrcopysignc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vrdivc        --output test/f32-vrdivc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vrsubc        --output test/f32-vrsubc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vsqrdiffc     --output test/f32-vsqrdiffc.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel f32-vsubc         --output test/f32-vsubc.cc &

tools/generate-vbinary-test.py --tester VCMulMicrokernelTester --ukernel f16-vcmul --output test/f16-vcmul.cc &
tools/generate-vbinary-test.py --tester VCMulMicrokernelTester --ukernel f32-vcmul --output test/f32-vcmul.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel qs8-vadd-minmax  --output test/qs8-vadd-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel qu8-vadd-minmax  --output test/qu8-vadd-minmax.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel qs8-vaddc-minmax --output test/qs8-vaddc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel qu8-vaddc-minmax --output test/qu8-vaddc-minmax.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel qs8-vmul-minmax-fp32  --output test/qs8-vmul-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel qs8-vmul-minmax-rndnu  --output test/qs8-vmul-minmax-rndnu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel qu8-vmul-minmax-fp32  --output test/qu8-vmul-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel qu8-vmul-minmax-rndnu  --output test/qu8-vmul-minmax-rndnu.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel qs8-vmulc-minmax-fp32 --output test/qs8-vmulc-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel qs8-vmulc-minmax-rndnu --output test/qs8-vmulc-minmax-rndnu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel qu8-vmulc-minmax-fp32 --output test/qu8-vmulc-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel qu8-vmulc-minmax-rndnu --output test/qu8-vmulc-minmax-rndnu.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --ukernel s32-vmul   --output test/s32-vmul.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester --broadcast_b --ukernel s32-vmulc    --output test/s32-vmulc.cc &

### Tests for VUnary micro-kernels
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel bf16-vabs --output test/bf16-vabs.cc &

tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vclamp --output test/f16-vclamp.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-velu --output test/f16-velu.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vabs --output test/f16-vabs.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vneg --output test/f16-vneg.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vsqr --output test/f16-vsqr.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vrndne --output test/f16-vrndne.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vrndz  --output test/f16-vrndz.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vrndu  --output test/f16-vrndu.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vrndd  --output test/f16-vrndd.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vrsqrt --output test/f16-vrsqrt.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vsigmoid --output test/f16-vsigmoid.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vsqrt --output test/f16-vsqrt.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vtanh --output test/f16-vtanh.cc &

tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vabs --output test/f32-vabs.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vclamp --output test/f32-vclamp.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-velu --output test/f32-velu.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vgelu --output test/f32-vgelu.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vexp --output test/f32-vexp.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vlog --output test/f32-vlog.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vneg --output test/f32-vneg.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vrelu --output test/f32-vrelu.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vrndd  --output test/f32-vrndd.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vrndne --output test/f32-vrndne.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vrndu  --output test/f32-vrndu.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vrndz  --output test/f32-vrndz.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vrsqrt --output test/f32-vrsqrt.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vsigmoid --output test/f32-vsigmoid.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vsqr --output test/f32-vsqr.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vsqrt --output test/f32-vsqrt.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vtanh --output test/f32-vtanh.cc &

tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel s8-vclamp --output test/s8-vclamp.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel u8-vclamp --output test/u8-vclamp.cc &

### Tests for VLRelu micro-kernels
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vlrelu --output test/f16-vlrelu.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vlrelu --output test/f32-vlrelu.cc &
tools/generate-vunary-test.py --tester VLReLUMicrokernelTester --ukernel qs8-vlrelu --output test/qs8-vlrelu.cc &
tools/generate-vunary-test.py --tester VLReLUMicrokernelTester --ukernel qu8-vlrelu --output test/qu8-vlrelu.cc &

### Tests for Reduce micro-kernels
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f16-rmax.yaml --output test/f16-rmax.cc &
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f16-rmin.yaml --output test/f16-rmin.cc &
tools/generate-reduce-test.py --tester RSumMicrokernelTester --spec test/f16-rsum.yaml --output test/f16-rsum.cc &
tools/generate-reduce-test.py --tester RSumMicrokernelTester --spec test/f16-f32acc-rsum.yaml --output test/f16-f32acc-rsum.cc &
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f16-rminmax.yaml --output test/f16-rminmax.cc &

tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f32-rmax.yaml --output test/f32-rmax.cc &
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f32-rmin.yaml --output test/f32-rmin.cc &
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f32-rminmax.yaml --output test/f32-rminmax.cc &

tools/generate-reduce-test.py --tester RSumMicrokernelTester --spec test/qs8-rsum.yaml --output test/qs8-rsum.cc &
tools/generate-reduce-test.py --tester RSumMicrokernelTester --spec test/qu8-rsum.yaml --output test/qu8-rsum.cc &
tools/generate-reduce-test.py --tester RSumMicrokernelTester --spec test/f32-rsum.yaml --output test/f32-rsum.cc &

tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/u8-rmax.yaml --output test/u8-rmax.cc &

tools/generate-rdsum-test.py --spec test/f16-f32acc-rdsum.yaml --output test/f16-f32acc-rdsum.cc &
tools/generate-rdsum-test.py --spec test/f32-rdsum.yaml --output test/f32-rdsum.cc &
tools/generate-rdsum-test.py --spec test/qs8-rdsum-minmax-fp32.yaml --output test/qs8-rdsum-minmax-fp32.cc &
tools/generate-rdsum-test.py --spec test/qu8-rdsum.yaml --output test/qu8-rdsum.cc &

### Tests for LUT micro-kernels
tools/generate-lut-test.py --spec test/x8-lut.yaml --output test/x8-lut.cc &

### Tests for Conv HWC layout micro-kernels
tools/generate-conv-hwc-test.py --spec test/f32-conv-hwc.yaml --output test/f32-conv-hwc.cc &

### Tests for Conv HWC2CHW layout micro-kernels
tools/generate-conv-hwc2chw-test.py --spec test/f16-conv-hwc2chw.yaml --output test/f16-conv-hwc2chw.cc &
tools/generate-conv-hwc2chw-test.py --spec test/f32-conv-hwc2chw.yaml --output test/f32-conv-hwc2chw.cc &

### Tests for DWConv micro-kernels
tools/generate-dwconv-unipass-test.py --ukernel f16-dwconv-minmax-unipass --output test/f16-dwconv-minmax-unipass.cc &
tools/generate-dwconv-multipass-test.py --ukernel f16-dwconv-minmax-multipass --output test/f16-dwconv-minmax-multipass.cc &

tools/generate-dwconv-unipass-test.py --ukernel f32-dwconv-unipass --output test/f32-dwconv-unipass.cc &
tools/generate-dwconv-unipass-test.py --ukernel f32-dwconv-minmax-unipass --output test/f32-dwconv-minmax-unipass.cc &
tools/generate-dwconv-multipass-test.py --ukernel f32-dwconv-multipass --output test/f32-dwconv-multipass.cc &
tools/generate-dwconv-multipass-test.py --ukernel f32-dwconv-minmax-multipass --output test/f32-dwconv-minmax-multipass.cc &

tools/generate-dwconv-unipass-test.py --ukernel qs8-qc8w-dwconv-minmax-unipass-fp32 --output test/qs8-qc8w-dwconv-minmax-unipass-fp32.cc &
tools/generate-dwconv-unipass-test.py --ukernel qs8-dwconv-minmax-unipass-fp32 --output test/qs8-dwconv-minmax-unipass-fp32.cc &
tools/generate-dwconv-unipass-test.py --ukernel qu8-dwconv-minmax-unipass-fp32 --output test/qu8-dwconv-minmax-unipass-fp32.cc &

tools/generate-dwconv-unipass-test.py --ukernel qs8-dwconv-minmax-unipass-rndnu --output test/qs8-dwconv-minmax-unipass-rndnu.cc &
tools/generate-dwconv-unipass-test.py --ukernel qu8-dwconv-minmax-unipass-rndnu --output test/qu8-dwconv-minmax-unipass-rndnu.cc &

tools/generate-dwconv-multipass-test.py --ukernel qs8-qc8w-dwconv-minmax-multipass-fp32 --output test/qs8-qc8w-dwconv-minmax-multipass-fp32.cc &
tools/generate-dwconv-multipass-test.py --ukernel qs8-dwconv-minmax-multipass-fp32 --output test/qs8-dwconv-minmax-multipass-fp32.cc &
tools/generate-dwconv-multipass-test.py --ukernel qu8-dwconv-minmax-multipass-fp32 --output test/qu8-dwconv-minmax-multipass-fp32.cc &

tools/generate-dwconv-multipass-test.py --ukernel qs8-dwconv-minmax-multipass-rndnu --output test/qs8-dwconv-minmax-multipass-rndnu.cc &
tools/generate-dwconv-multipass-test.py --ukernel qu8-dwconv-minmax-multipass-rndnu --output test/qu8-dwconv-minmax-multipass-rndnu.cc &

### Tests for DWConv CHW layout micro-kernels
tools/generate-dwconv2d-chw-test.py --spec test/f16-dwconv2d-chw.yaml --output test/f16-dwconv2d-chw.cc &
tools/generate-dwconv2d-chw-test.py --spec test/f32-dwconv2d-chw.yaml --output test/f32-dwconv2d-chw.cc &

### Tests for VHSwish micro-kernels
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f16-vhswish --output test/f16-vhswish.cc &
tools/generate-vunary-test.py --tester VUnaryMicrokernelTester --ukernel f32-vhswish --output test/f32-vhswish.cc &
tools/generate-vhswish-test.py --spec test/qs8-vhswish.yaml --output test/qs8-vhswish.cc &
tools/generate-vhswish-test.py --spec test/qu8-vhswish.yaml --output test/qu8-vhswish.cc &

### Tests for IBilinear micro-kernels
tools/generate-ibilinear-test.py --spec test/f16-ibilinear.yaml --output test/f16-ibilinear.cc &
tools/generate-ibilinear-test.py --spec test/f32-ibilinear.yaml --output test/f32-ibilinear.cc &
tools/generate-ibilinear-test.py --spec test/s8-ibilinear.yaml --output test/s8-ibilinear.cc &
tools/generate-ibilinear-test.py --spec test/u8-ibilinear.yaml --output test/u8-ibilinear.cc &

### Tests for IBilinear CHW layout micro-kernels
tools/generate-ibilinear-chw-test.py --spec test/f16-ibilinear-chw.yaml --output test/f16-ibilinear-chw.cc &
tools/generate-ibilinear-chw-test.py --spec test/f32-ibilinear-chw.yaml --output test/f32-ibilinear-chw.cc &

### Tests for PRelu micro-kernels
tools/generate-prelu-test.py --spec test/f16-prelu.yaml --output test/f16-prelu.cc &
tools/generate-prelu-test.py --spec test/f32-prelu.yaml --output test/f32-prelu.cc &

### Tests for RAddExpMinusMax micro-kernels
tools/generate-raddexpminusmax-test.py --spec test/f32-raddexpminusmax.yaml --output test/f32-raddexpminusmax.cc &

### Tests for RAddExtExp micro-kernels
tools/generate-raddextexp-test.py --spec test/f32-raddextexp.yaml --output test/f32-raddextexp.cc &

### Tests for RAddStoreExpMinusMax micro-kernels
tools/generate-raddstoreexpminusmax-test.py --spec test/f16-raddstoreexpminusmax.yaml --output test/f16-raddstoreexpminusmax.cc &
tools/generate-raddstoreexpminusmax-test.py --spec test/f32-raddstoreexpminusmax.yaml --output test/f32-raddstoreexpminusmax.cc &

### Tests for VScaleExtExp micro-kernels
tools/generate-vscaleextexp-test.py --spec test/f32-vscaleextexp.yaml --output test/f32-vscaleextexp.cc &

### Tests for VScaleExpMinusMax micro-kernels
tools/generate-vscaleexpminusmax-test.py --spec test/f32-vscaleexpminusmax.yaml --output test/f32-vscaleexpminusmax.cc &

### Tests for VMulCAddC micro-kernels
tools/generate-vmulcaddc-test.py --spec test/f16-vmulcaddc-minmax.yaml --output test/f16-vmulcaddc-minmax.cc &
tools/generate-vmulcaddc-test.py --spec test/f32-vmulcaddc-minmax.yaml --output test/f32-vmulcaddc-minmax.cc &

### Tests for the portable SIMD wrappers.
tools/xngen test/f32-simd.cc.in -D ARCH=scalar -D ARCH_MACRO="" -D TEST_REQUIRES="" -o test/f32-simd-scalar.cc &
tools/xngen test/f32-simd.cc.in -D ARCH=sse2 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_SSE2 -o test/f32-simd-sse2.cc &
tools/xngen test/f32-simd.cc.in -D ARCH=avx -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX -o test/f32-simd-avx.cc &
tools/xngen test/f32-simd.cc.in -D ARCH=avx2 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX2 -o test/f32-simd-avx2.cc &
tools/xngen test/f32-simd.cc.in -D ARCH=fma3 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_FMA3 -o test/f32-simd-fma3.cc &
tools/xngen test/f32-simd.cc.in -D ARCH=avx512f -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX512F -o test/f32-simd-avx512f.cc &
tools/xngen test/f32-simd.cc.in -D ARCH=neon -D ARCH_MACRO="XNN_ARCH_ARM || XNN_ARCH_ARM64" -D TEST_REQUIRES=TEST_REQUIRES_ARM_NEON -o test/f32-simd-neon.cc &
tools/xngen test/f32-simd.cc.in -D ARCH=wasmsimd -D ARCH_MACRO="XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD" -D TEST_REQUIRES="" -o test/f32-simd-wasmsimd.cc &
tools/xngen test/f32-simd.cc.in -D ARCH=hvx -D ARCH_MACRO=XNN_ARCH_HEXAGON -D TEST_REQUIRES=TEST_REQUIRES_HVX -o test/f32-simd-hvx.cc &

tools/xngen test/f16-simd.cc.in -D ARCH=scalar -D ARCH_MACRO="" -D TEST_REQUIRES="" -o test/f16-simd-scalar.cc &

tools/xngen test/s16-simd.cc.in -D ARCH=scalar -D ARCH_MACRO="" -D TEST_REQUIRES="" -o test/s16-simd-scalar.cc &
tools/xngen test/s16-simd.cc.in -D ARCH=neon -D ARCH_MACRO="XNN_ARCH_ARM || XNN_ARCH_ARM64" -D TEST_REQUIRES=TEST_REQUIRES_ARM_NEON -o test/s16-simd-neon.cc &
tools/xngen test/s16-simd.cc.in -D ARCH=sse41 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_SSE41 -o test/s16-simd-sse41.cc &
tools/xngen test/s16-simd.cc.in -D ARCH=avx2 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX2 -o test/s16-simd-avx2.cc &
tools/xngen test/s16-simd.cc.in -D ARCH=avx512skx -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX512SKX -o test/s16-simd-avx512skx.cc &
tools/xngen test/s16-simd.cc.in -D ARCH=wasmsimd -D ARCH_MACRO="XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD" -D TEST_REQUIRES="" -o test/s16-simd-wasmsimd.cc &

tools/xngen test/s32-simd.cc.in -D ARCH=scalar -D ARCH_MACRO="" -D TEST_REQUIRES="" -o test/s32-simd-scalar.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=neon -D ARCH_MACRO="XNN_ARCH_ARM || XNN_ARCH_ARM64" -D TEST_REQUIRES=TEST_REQUIRES_ARM_NEON -o test/s32-simd-neon.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=sse41 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_SSE41 -o test/s32-simd-sse41.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=avx2 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX2 -o test/s32-simd-avx2.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=avx512f -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX512F -o test/s32-simd-avx512f.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=wasmsimd -D ARCH_MACRO="XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD" -D TEST_REQUIRES="" -o test/s32-simd-wasmsimd.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=hvx -D ARCH_MACRO="XNN_ENABLE_HVX && XNN_ARCH_HEXAGON" -D TEST_REQUIRES=TEST_REQUIRES_HVX -o test/s32-simd-hvx.cc &

tools/xngen test/u32-simd.cc.in -D ARCH=scalar -D ARCH_MACRO="" -D TEST_REQUIRES="" -o test/u32-simd-scalar.cc &
tools/xngen test/u32-simd.cc.in -D ARCH=neon -D ARCH_MACRO="XNN_ARCH_ARM || XNN_ARCH_ARM64" -D TEST_REQUIRES=TEST_REQUIRES_ARM_NEON -o test/u32-simd-neon.cc &
tools/xngen test/u32-simd.cc.in -D ARCH=sse41 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_SSE41 -o test/u32-simd-sse41.cc &
tools/xngen test/u32-simd.cc.in -D ARCH=avx2 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX2 -o test/u32-simd-avx2.cc &
tools/xngen test/u32-simd.cc.in -D ARCH=avx512f -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX512F -o test/u32-simd-avx512f.cc &
tools/xngen test/u32-simd.cc.in -D ARCH=wasmsimd -D ARCH_MACRO="XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD" -D TEST_REQUIRES="" -o test/u32-simd-wasmsimd.cc &

tools/xngen test/s8-simd.cc.in -D ARCH=scalar -D ARCH_MACRO="" -D TEST_REQUIRES="" -o test/s8-simd-scalar.cc &

wait
