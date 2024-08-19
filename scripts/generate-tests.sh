#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

### Tests for packing micro-kernels
tools/generate-pack-test.py --spec test/x32-packx.yaml --output test/x32-packx.cc &

### Tests for Pack quantized micro-kernels
tools/generate-packq-test.py --spec test/x8-packq.yaml --output test/x8-packq.cc --output-bench bench/x8-packq.cc &

### Tests for Pack Weights micro-kernels
tools/generate-packw-test.py --spec test/x8-packw.yaml --output test/x8-packw.cc --output-bench bench/x8-packw.cc &
tools/generate-packw-test.py --spec test/x16-packw.yaml --output test/x16-packw.cc --output-bench bench/x16-packw.cc &
tools/generate-packw-test.py --spec test/x32-packw.yaml --output test/x32-packw.cc --output-bench bench/x32-packw.cc &

### Tests for Zero Bias micro-kernels
tools/generate-packb-test.py --spec test/x32-zerob.yaml --output test/x32-zerob.cc &

### Tests for Pack Bias micro-kernels
tools/generate-packb-test.py --spec test/x32-packb.yaml --output test/x32-packb.cc &

### Tests for MaxPool micro-kernels
tools/generate-maxpool-test.py --spec test/f32-maxpool-minmax.yaml --output test/f32-maxpool-minmax.cc &
tools/generate-maxpool-test.py --spec test/f16-maxpool-minmax.yaml --output test/f16-maxpool-minmax.cc &
tools/generate-maxpool-test.py --spec test/s8-maxpool-minmax.yaml --output test/s8-maxpool-minmax.cc &
tools/generate-maxpool-test.py --spec test/u8-maxpool-minmax.yaml --output test/u8-maxpool-minmax.cc &

### Tests for AvgPool micro-kernels
tools/generate-avgpool-test.py --spec test/f16-avgpool-minmax.yaml --output test/f16-avgpool-minmax.cc &
tools/generate-avgpool-test.py --spec test/f32-avgpool-minmax.yaml --output test/f32-avgpool-minmax.cc &
tools/generate-avgpool-test.py --spec test/qu8-avgpool-minmax-fp32.yaml --output test/qu8-avgpool-minmax-fp32.cc &

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

### Tests for PAvgPool micro-kernels
tools/generate-avgpool-test.py --spec test/f16-pavgpool-minmax.yaml --output test/f16-pavgpool-minmax.cc &
tools/generate-avgpool-test.py --spec test/f32-pavgpool-minmax.yaml --output test/f32-pavgpool-minmax.cc &

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
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vadd-minmax.yaml --output test/f16-vadd-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vdiv-minmax.yaml --output test/f16-vdiv-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vmax.yaml --output test/f16-vmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vmin.yaml --output test/f16-vmin.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vmul-minmax.yaml --output test/f16-vmul-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vsqrdiff.yaml --output test/f16-vsqrdiff.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vsub-minmax.yaml --output test/f16-vsub-minmax.cc &

tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vaddc-minmax.yaml --output test/f16-vaddc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vdivc-minmax.yaml --output test/f16-vdivc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vrdivc-minmax.yaml --output test/f16-vrdivc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vmaxc.yaml --output test/f16-vmaxc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vminc.yaml --output test/f16-vminc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vmulc-minmax.yaml --output test/f16-vmulc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vsqrdiffc.yaml --output test/f16-vsqrdiffc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vsubc-minmax.yaml --output test/f16-vsubc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vrsubc-minmax.yaml --output test/f16-vrsubc-minmax.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vadd-minmax.yaml --output test/f32-vadd-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vadd-relu.yaml   --output test/f32-vadd-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vadd.yaml        --output test/f32-vadd.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vcopysign.yaml   --output test/f32-vcopysign.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vdiv-minmax.yaml --output test/f32-vdiv-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vdiv-relu.yaml   --output test/f32-vdiv-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vdiv.yaml        --output test/f32-vdiv.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vmax.yaml        --output test/f32-vmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vmin.yaml        --output test/f32-vmin.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vmul-minmax.yaml --output test/f32-vmul-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vmul-relu.yaml   --output test/f32-vmul-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vmul.yaml        --output test/f32-vmul.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vsqrdiff.yaml    --output test/f32-vsqrdiff.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vsub-minmax.yaml --output test/f32-vsub-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vsub-relu.yaml   --output test/f32-vsub-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vsub.yaml        --output test/f32-vsub.cc &

tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vaddc-minmax.yaml  --output test/f32-vaddc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vaddc-relu.yaml    --output test/f32-vaddc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vaddc.yaml         --output test/f32-vaddc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vcopysignc.yaml    --output test/f32-vcopysignc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vdivc-minmax.yaml  --output test/f32-vdivc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vdivc-relu.yaml    --output test/f32-vdivc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vdivc.yaml         --output test/f32-vdivc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vmaxc.yaml         --output test/f32-vmaxc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vminc.yaml         --output test/f32-vminc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vmulc-minmax.yaml  --output test/f32-vmulc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vmulc-relu.yaml    --output test/f32-vmulc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vmulc.yaml         --output test/f32-vmulc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrcopysignc.yaml   --output test/f32-vrcopysignc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrdivc-minmax.yaml --output test/f32-vrdivc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrdivc-relu.yaml   --output test/f32-vrdivc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrdivc.yaml        --output test/f32-vrdivc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrsubc-minmax.yaml --output test/f32-vrsubc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrsubc-relu.yaml   --output test/f32-vrsubc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrsubc.yaml        --output test/f32-vrsubc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vsqrdiffc.yaml     --output test/f32-vsqrdiffc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vsubc-minmax.yaml  --output test/f32-vsubc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vsubc-relu.yaml    --output test/f32-vsubc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vsubc.yaml         --output test/f32-vsubc.cc &

tools/generate-vbinary-test.py --tester VCMulMicrokernelTester --spec test/f16-vcmul.yaml --output test/f16-vcmul.cc &
tools/generate-vbinary-test.py --tester VCMulMicrokernelTester --spec test/f32-vcmul.yaml --output test/f32-vcmul.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/qs8-vadd-minmax.yaml  --output test/qs8-vadd-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/qu8-vadd-minmax.yaml  --output test/qu8-vadd-minmax.cc &

tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/qs8-vaddc-minmax.yaml --output test/qs8-vaddc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/qu8-vaddc-minmax.yaml --output test/qu8-vaddc-minmax.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/qs8-vmul-minmax-fp32.yaml  --output test/qs8-vmul-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/qs8-vmul-minmax-rndnu.yaml  --output test/qs8-vmul-minmax-rndnu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/qu8-vmul-minmax-fp32.yaml  --output test/qu8-vmul-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/qu8-vmul-minmax-rndnu.yaml  --output test/qu8-vmul-minmax-rndnu.cc &

tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/qs8-vmulc-minmax-fp32.yaml --output test/qs8-vmulc-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/qs8-vmulc-minmax-rndnu.yaml --output test/qs8-vmulc-minmax-rndnu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/qu8-vmulc-minmax-fp32.yaml --output test/qu8-vmulc-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/qu8-vmulc-minmax-rndnu.yaml --output test/qu8-vmulc-minmax-rndnu.cc &

tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/s32-vmul.yaml   --output test/s32-vmul.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/s32-vmulc.yaml    --output test/s32-vmulc.cc &

### Tests for VUnary micro-kernels
tools/generate-vunary-test.py --spec test/bf16-vabs.yaml --output test/bf16-vabs.cc &

tools/generate-vunary-test.py --spec test/f16-vclamp.yaml --output test/f16-vclamp.cc &
tools/generate-vunary-test.py --spec test/f16-velu.yaml --output test/f16-velu.cc &
tools/generate-vunary-test.py --spec test/f16-vabs.yaml --output test/f16-vabs.cc &
tools/generate-vunary-test.py --spec test/f16-vneg.yaml --output test/f16-vneg.cc &
tools/generate-vunary-test.py --spec test/f16-vsqr.yaml --output test/f16-vsqr.cc &
tools/generate-vunary-test.py --spec test/f16-vrndne.yaml --output test/f16-vrndne.cc &
tools/generate-vunary-test.py --spec test/f16-vrndz.yaml  --output test/f16-vrndz.cc &
tools/generate-vunary-test.py --spec test/f16-vrndu.yaml  --output test/f16-vrndu.cc &
tools/generate-vunary-test.py --spec test/f16-vrndd.yaml  --output test/f16-vrndd.cc &
tools/generate-vunary-test.py --spec test/f16-vrsqrt.yaml --output test/f16-vrsqrt.cc &
tools/generate-vunary-test.py --spec test/f16-vsigmoid.yaml --output test/f16-vsigmoid.cc &
tools/generate-vunary-test.py --spec test/f16-vsqrt.yaml --output test/f16-vsqrt.cc &
tools/generate-vunary-test.py --spec test/f16-vtanh.yaml --output test/f16-vtanh.cc &

tools/generate-vunary-test.py --spec test/f32-vabs.yaml --output test/f32-vabs.cc &
tools/generate-vunary-test.py --spec test/f32-vclamp.yaml --output test/f32-vclamp.cc &
tools/generate-vunary-test.py --spec test/f32-velu.yaml --output test/f32-velu.cc &
tools/generate-vunary-test.py --spec test/f32-vgelu.yaml --output test/f32-vgelu.cc &
tools/generate-vunary-test.py --spec test/f32-vexp.yaml --output test/f32-vexp.cc &
tools/generate-vunary-test.py --spec test/f32-vlog.yaml --output test/f32-vlog.cc &
tools/generate-vunary-test.py --spec test/f32-vneg.yaml --output test/f32-vneg.cc &
tools/generate-vunary-test.py --spec test/f32-vrelu.yaml --output test/f32-vrelu.cc &
tools/generate-vunary-test.py --spec test/f32-vrndd.yaml  --output test/f32-vrndd.cc &
tools/generate-vunary-test.py --spec test/f32-vrndne.yaml --output test/f32-vrndne.cc &
tools/generate-vunary-test.py --spec test/f32-vrndu.yaml  --output test/f32-vrndu.cc &
tools/generate-vunary-test.py --spec test/f32-vrndz.yaml  --output test/f32-vrndz.cc &
tools/generate-vunary-test.py --spec test/f32-vrsqrt.yaml --output test/f32-vrsqrt.cc &
tools/generate-vunary-test.py --spec test/f32-vsigmoid.yaml --output test/f32-vsigmoid.cc &
tools/generate-vunary-test.py --spec test/f32-vsqr.yaml --output test/f32-vsqr.cc &
tools/generate-vunary-test.py --spec test/f32-vsqrt.yaml --output test/f32-vsqrt.cc &
tools/generate-vunary-test.py --spec test/f32-vtanh.yaml --output test/f32-vtanh.cc &

tools/generate-vunary-test.py --spec test/s8-vclamp.yaml --output test/s8-vclamp.cc &
tools/generate-vunary-test.py --spec test/u8-vclamp.yaml --output test/u8-vclamp.cc &

tools/generate-vunary-test.py --spec test/u64-u32-vsqrtshift.yaml --output test/u64-u32-vsqrtshift.cc &

### Tests for VLRelu micro-kernels
tools/generate-vunary-test.py --spec test/f16-vlrelu.yaml --output test/f16-vlrelu.cc &
tools/generate-vunary-test.py --spec test/f32-vlrelu.yaml --output test/f32-vlrelu.cc &
tools/generate-vlrelu-test.py --spec test/qs8-vlrelu.yaml --output test/qs8-vlrelu.cc &
tools/generate-vlrelu-test.py --spec test/qu8-vlrelu.yaml --output test/qu8-vlrelu.cc &

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
tools/generate-reduce-test.py --tester RSumMicrokernelTester --spec test/f32-rsum.yaml --output test/f32-rsum.cc &

tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/u8-rmax.yaml --output test/u8-rmax.cc &

tools/generate-rdsum-test.py --spec test/f16-f32acc-rdsum.yaml --output test/f16-f32acc-rdsum.cc &
tools/generate-rdsum-test.py --spec test/f32-rdsum.yaml --output test/f32-rdsum.cc &
tools/generate-rdsum-test.py --spec test/qs8-rdsum-minmax-fp32.yaml --output test/qs8-rdsum-minmax-fp32.cc &

### Tests for Fill micro-kernels
tools/generate-fill-test.py --spec test/xx-fill.yaml --output test/xx-fill.cc &

### Tests for Pad micro-kernels
tools/generate-pad-test.py --spec test/xx-pad.yaml --output test/xx-pad.cc &

### Tests for Transpose micro-kernels
tools/generate-transpose-test.py --spec test/x8-transpose.yaml  --output test/x8-transpose.cc &
tools/generate-transpose-test.py --spec test/x16-transpose.yaml --output test/x16-transpose.cc &
tools/generate-transpose-test.py --spec test/x24-transpose.yaml --output test/x24-transpose.cc &
tools/generate-transpose-test.py --spec test/x32-transpose.yaml --output test/x32-transpose.cc &
tools/generate-transpose-test.py --spec test/x64-transpose.yaml --output test/x64-transpose.cc &
tools/generate-transpose-test.py --spec test/xx-transposev.yaml  --output test/xx-transposev.cc &

### Tests for LUT micro-kernels
tools/generate-lut-test.py --spec test/x8-lut.yaml --output test/x8-lut.cc &

### Tests for Conv HWC layout micro-kernels
tools/generate-conv-hwc-test.py --spec test/f32-conv-hwc.yaml --output test/f32-conv-hwc.cc &

### Tests for Conv HWC2CHW layout micro-kernels
tools/generate-conv-hwc2chw-test.py --spec test/f16-conv-hwc2chw.yaml --output test/f16-conv-hwc2chw.cc &
tools/generate-conv-hwc2chw-test.py --spec test/f32-conv-hwc2chw.yaml --output test/f32-conv-hwc2chw.cc &

### Tests for DWConv micro-kernels
tools/generate-dwconv-unipass-test.py --spec test/f16-dwconv-minmax-unipass.yaml --output test/f16-dwconv-minmax-unipass.cc &
tools/generate-dwconv-multipass-test.py --spec test/f16-dwconv-minmax-multipass.yaml --output test/f16-dwconv-minmax-multipass.cc &

tools/generate-dwconv-unipass-test.py --spec test/f32-dwconv-unipass.yaml --output test/f32-dwconv-unipass.cc &
tools/generate-dwconv-unipass-test.py --spec test/f32-dwconv-minmax-unipass.yaml --output test/f32-dwconv-minmax-unipass.cc &
tools/generate-dwconv-multipass-test.py --spec test/f32-dwconv-multipass.yaml --output test/f32-dwconv-multipass.cc &
tools/generate-dwconv-multipass-test.py --spec test/f32-dwconv-minmax-multipass.yaml --output test/f32-dwconv-minmax-multipass.cc &

tools/generate-dwconv-unipass-test.py --spec test/qs8-qc8w-dwconv-minmax-unipass-fp32.yaml --output test/qs8-qc8w-dwconv-minmax-unipass-fp32.cc &
tools/generate-dwconv-unipass-test.py --spec test/qs8-dwconv-minmax-unipass-fp32.yaml --output test/qs8-dwconv-minmax-unipass-fp32.cc &
tools/generate-dwconv-unipass-test.py --spec test/qu8-dwconv-minmax-unipass-fp32.yaml --output test/qu8-dwconv-minmax-unipass-fp32.cc &

tools/generate-dwconv-unipass-test.py --spec test/qs8-dwconv-minmax-unipass-rndnu.yaml --output test/qs8-dwconv-minmax-unipass-rndnu.cc &
tools/generate-dwconv-unipass-test.py --spec test/qu8-dwconv-minmax-unipass-rndnu.yaml --output test/qu8-dwconv-minmax-unipass-rndnu.cc &

tools/generate-dwconv-multipass-test.py --spec test/qs8-qc8w-dwconv-minmax-multipass-fp32.yaml --output test/qs8-qc8w-dwconv-minmax-multipass-fp32.cc &
tools/generate-dwconv-multipass-test.py --spec test/qs8-dwconv-minmax-multipass-fp32.yaml --output test/qs8-dwconv-minmax-multipass-fp32.cc &
tools/generate-dwconv-multipass-test.py --spec test/qu8-dwconv-minmax-multipass-fp32.yaml --output test/qu8-dwconv-minmax-multipass-fp32.cc &

tools/generate-dwconv-multipass-test.py --spec test/qs8-dwconv-minmax-multipass-rndnu.yaml --output test/qs8-dwconv-minmax-multipass-rndnu.cc &
tools/generate-dwconv-multipass-test.py --spec test/qu8-dwconv-minmax-multipass-rndnu.yaml --output test/qu8-dwconv-minmax-multipass-rndnu.cc &

### Tests for DWConv CHW layout micro-kernels
tools/generate-dwconv2d-chw-test.py --spec test/f16-dwconv2d-chw.yaml --output test/f16-dwconv2d-chw.cc &
tools/generate-dwconv2d-chw-test.py --spec test/f32-dwconv2d-chw.yaml --output test/f32-dwconv2d-chw.cc &

### Tests for VConvert micro-kernels
tools/generate-vcvt-test.py --spec test/qs8-vcvt.yaml --output test/qs8-vcvt.cc --output-bench bench/qs8-vcvt.cc &
tools/generate-vcvt-test.py --spec test/qu8-vcvt.yaml --output test/qu8-vcvt.cc --output-bench bench/qu8-vcvt.cc &
tools/generate-vcvt-test.py --spec test/qs8-f16-vcvt.yaml --output test/qs8-f16-vcvt.cc --output-bench bench/qs8-f16-vcvt.cc &
tools/generate-vcvt-test.py --spec test/qs8-f32-vcvt.yaml --output test/qs8-f32-vcvt.cc --output-bench bench/qs8-f32-vcvt.cc &
tools/generate-vcvt-test.py --spec test/qu8-f32-vcvt.yaml --output test/qu8-f32-vcvt.cc --output-bench bench/qu8-f32-vcvt.cc &
tools/generate-vcvt-test.py --spec test/f16-qs8-vcvt.yaml --output test/f16-qs8-vcvt.cc --output-bench bench/f16-qs8-vcvt.cc &
tools/generate-vcvt-test.py --spec test/f32-qs8-vcvt.yaml --output test/f32-qs8-vcvt.cc --output-bench bench/f32-qs8-vcvt.cc &
tools/generate-vcvt-test.py --spec test/f32-qu8-vcvt.yaml --output test/f32-qu8-vcvt.cc --output-bench bench/f32-qu8-vcvt.cc &
tools/generate-vcvt-test.py --spec test/f32-f16-vcvt.yaml --output test/f32-f16-vcvt.cc --output-bench bench/f32-f16-vcvt.cc &
tools/generate-vcvt-test.py --spec test/f16-f32-vcvt.yaml --output test/f16-f32-vcvt.cc --output-bench bench/f16-f32-vcvt.cc &
tools/generate-vcvt-test.py --spec test/qs16-qs8-vcvt.yaml --output test/qs16-qs8-vcvt.cc --output-bench bench/qs16-qs8-vcvt.cc &

### Tests for VLShift micro-kernels
tools/generate-vlshift-test.py --spec test/i16-vlshift.yaml --output test/i16-vlshift.cc &

### Tests for VLog micro-kernels
tools/generate-vlog-test.py --spec test/u32-vlog.yaml --output test/u32-vlog.cc &

### Tests for VHSwish micro-kernels
tools/generate-vunary-test.py --spec test/f16-vhswish.yaml --output test/f16-vhswish.cc &
tools/generate-vunary-test.py --spec test/f32-vhswish.yaml --output test/f32-vhswish.cc &
tools/generate-vhswish-test.py --spec test/qs8-vhswish.yaml --output test/qs8-vhswish.cc &
tools/generate-vhswish-test.py --spec test/qu8-vhswish.yaml --output test/qu8-vhswish.cc &

### Tests for Window micro-kernels
tools/generate-window-test.py --spec test/s16-window.yaml --output test/s16-window.cc &

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

### Tests for FFTR micro-kernels
tools/generate-fftr-test.py --spec test/cs16-fftr.yaml --output test/cs16-fftr.cc &

### Tests for BFly4 micro-kernels
tools/generate-bfly4-test.py --spec test/cs16-bfly4.yaml --output test/cs16-bfly4.cc &

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

### Tests for RMaxAbs micro-kernels
tools/generate-rmaxabs-test.py --spec test/s16-rmaxabs.yaml --output test/s16-rmaxabs.cc &

### Tests for VMulCAddC micro-kernels
tools/generate-vmulcaddc-test.py --spec test/f16-vmulcaddc-minmax.yaml --output test/f16-vmulcaddc-minmax.cc &
tools/generate-vmulcaddc-test.py --spec test/f32-vmulcaddc-minmax.yaml --output test/f32-vmulcaddc-minmax.cc &

### Tests for VSquareAbs micro-kernels
tools/generate-vsquareabs-test.py --spec test/cs16-vsquareabs.yaml --output test/cs16-vsquareabs.cc &

### Tests for FilterBank accumulate micro-kernels
tools/generate-filterbank-accumulate-test.py --spec test/u32-filterbank-accumulate.yaml --output test/u32-filterbank-accumulate.cc &

### Tests for FilterBank subtract micro-kernels
tools/generate-filterbank-subtract-test.py --spec test/u32-filterbank-subtract.yaml --output test/u32-filterbank-subtract.cc &

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
tools/xngen test/s16-simd.cc.in -D ARCH=scalar -D ARCH_MACRO="" -D TEST_REQUIRES="" -o test/s16-simd-scalar.cc &
tools/xngen test/s16-simd.cc.in -D ARCH=sse41 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_SSE41 -o test/s16-simd-sse41.cc &
tools/xngen test/s16-simd.cc.in -D ARCH=avx2 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX2 -o test/s16-simd-avx2.cc &
tools/xngen test/s16-simd.cc.in -D ARCH=avx512bw -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX512F -o test/s16-simd-avx512bw.cc &
tools/xngen test/s16-simd.cc.in -D ARCH=wasmsimd -D ARCH_MACRO="XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD" -D TEST_REQUIRES="" -o test/s16-simd-wasmsimd.cc &

tools/xngen test/s32-simd.cc.in -D ARCH=scalar -D ARCH_MACRO="" -D TEST_REQUIRES="" -o test/s32-simd-scalar.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=neon -D ARCH_MACRO="XNN_ARCH_ARM || XNN_ARCH_ARM64" -D TEST_REQUIRES=TEST_REQUIRES_ARM_NEON -o test/s32-simd-neon.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=scalar -D ARCH_MACRO="" -D TEST_REQUIRES="" -o test/s32-simd-scalar.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=sse41 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_SSE41 -o test/s32-simd-sse41.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=avx2 -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX2 -o test/s32-simd-avx2.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=avx512f -D ARCH_MACRO="XNN_ARCH_X86 || XNN_ARCH_X86_64" -D TEST_REQUIRES=TEST_REQUIRES_X86_AVX512F -o test/s32-simd-avx512f.cc &
tools/xngen test/s32-simd.cc.in -D ARCH=wasmsimd -D ARCH_MACRO="XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD" -D TEST_REQUIRES="" -o test/s32-simd-wasmsimd.cc &

tools/xngen test/s8-simd.cc.in -D ARCH=scalar -D ARCH_MACRO="" -D TEST_REQUIRES="" -o test/s8-simd-scalar.cc &

wait
