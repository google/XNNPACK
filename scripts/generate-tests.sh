#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

### Tests for U8 micro-kernels
tools/generate-vunary-test.py --spec test/u8-vclamp.yaml --output test/u8-vclamp.cc

### Tests for packing micro-kernels
tools/generate-pack-test.py --spec test/x32-packx.yaml --output test/x32-packx.cc

### Tests for MaxPool micro-kernels
tools/generate-maxpool-test.py --spec test/u8-maxpool-minmax.yaml --output test/u8-maxpool-minmax.cc
tools/generate-maxpool-test.py --spec test/f32-maxpool-minmax.yaml --output test/f32-maxpool-minmax.cc

### Tests for AvgPool micro-kernels
tools/generate-avgpool-test.py --spec test/qu8-avgpool-minmax.yaml --output test/qu8-avgpool-minmax.cc
tools/generate-avgpool-test.py --spec test/f32-avgpool-minmax.yaml --output test/f32-avgpool-minmax.cc

### Tests for GAvgPool micro-kernels
tools/generate-gavgpool-test.py --spec test/f16-gavgpool-minmax.yaml --output test/f16-gavgpool-minmax.cc
tools/generate-gavgpool-test.py --spec test/f32-gavgpool-minmax.yaml --output test/f32-gavgpool-minmax.cc

### Tests for PAvgPool micro-kernels
tools/generate-avgpool-test.py --spec test/f32-pavgpool-minmax.yaml --output test/f32-pavgpool-minmax.cc

### Tests for ArgMaxPool micro-kernels
tools/generate-argmaxpool-test.py --spec test/f32-argmaxpool.yaml --output test/f32-argmaxpool.cc
