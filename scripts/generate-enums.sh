#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

tools/generate-enum.py --enum xnn_operator_type --spec src/enums/operator-type.yaml --output_src src/enums/operator-type.c --output_hdr src/xnnpack/operator-type.h &
tools/generate-enum.py --enum xnn_microkernel_type --spec src/enums/microkernel-type.yaml --output_src src/enums/microkernel-type.c --output_hdr src/xnnpack/microkernel-type.h &
tools/generate-enum.py --debug --enum xnn_node_type --spec src/enums/node-type.yaml --output_src src/enums/node-type.c --output_hdr src/xnnpack/node-type.h &

wait
