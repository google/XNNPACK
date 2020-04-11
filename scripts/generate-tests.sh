#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

### Tests for Q8 micro-kernels
tools/generate-gemm-test.py --spec test/q8-gemm-minmax.yaml --output test/q8-gemm-minmax.cc
tools/generate-gemm-test.py --spec test/q8-igemm-minmax.yaml --output test/q8-igemm-minmax.cc
tools/generate-dwconv-test.py --spec test/q8-dwconv-minmax.yaml --output test/q8-dwconv-minmax.cc

### Tests for U8 micro-kernels
tools/generate-clamp-test.py --spec test/u8-clamp.yaml --output test/u8-clamp.cc

### Tests for packing micro-kernels
tools/generate-pack-test.py --spec test/x32-packx.yaml --output test/x32-packx.cc

### Tests for MaxPool micro-kernels
tools/generate-maxpool-test.py --spec test/u8-maxpool-minmax.yaml --output test/u8-maxpool-minmax.cc
tools/generate-maxpool-test.py --spec test/f32-maxpool-minmax.yaml --output test/f32-maxpool-minmax.cc

### Tests for AvgPool micro-kernels
tools/generate-avgpool-test.py --spec test/q8-avgpool-minmax.yaml --output test/q8-avgpool-minmax.cc
tools/generate-avgpool-test.py --spec test/f32-avgpool-minmax.yaml --output test/f32-avgpool-minmax.cc

### Tests for PAvgPool micro-kernels
tools/generate-avgpool-test.py --spec test/f32-pavgpool-minmax.yaml --output test/f32-pavgpool-minmax.cc

### Tests for ArgMaxPool micro-kernels
tools/generate-argmaxpool-test.py --spec test/f32-argmaxpool.yaml --output test/f32-argmaxpool.cc
