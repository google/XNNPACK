#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

tools/generate-enum.py --enum xnn_operator_type --spec src/operator-strings.yaml --output_src src/operator-strings.c --output_hdr src/xnnpack/operator-type.h &
tools/generate-enum.py --enum xnn_ukernel_type --spec src/ukernel-strings.yaml --output_src src/ukernel-strings.c --output_hdr src/xnnpack/ukernel-type.h &

wait
