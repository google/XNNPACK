#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

### Tests for Q8 micro-kernels
tools/generate-gemm-test.py --spec test/q8-gemm.yaml --output test/q8-gemm.cc
tools/generate-gemm-test.py --spec test/q8-igemm.yaml --output test/q8-igemm.cc
tools/generate-dwconv-test.py --spec test/q8-dwconv.yaml --output test/q8-dwconv.cc

### Tests for U8 micro-kernels
tools/generate-clamp-test.py --spec test/u8-clamp.yaml --output test/u8-clamp.cc

### Tests for packing micro-kernels
tools/generate-pack-test.py --spec test/x32-packx.yaml --output test/x32-packx.cc

### Tests for MaxPool micro-kernels
tools/generate-maxpool-test.py --spec test/u8-maxpool.yaml --output test/u8-maxpool.cc
tools/generate-maxpool-test.py --spec test/f32-maxpool.yaml --output test/f32-maxpool.cc

### Tests for AvgPool micro-kernels
tools/generate-avgpool-test.py --spec test/q8-avgpool.yaml --output test/q8-avgpool.cc
tools/generate-avgpool-test.py --spec test/f32-avgpool.yaml --output test/f32-avgpool.cc

### Tests for PAvgPool micro-kernels
tools/generate-avgpool-test.py --spec test/f32-pavgpool.yaml --output test/f32-pavgpool.cc

### Tests for ArgMaxPool micro-kernels
tools/generate-argmaxpool-test.py --spec test/f32-argmaxpool.yaml --output test/f32-argmaxpool.cc
