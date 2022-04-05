#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

############################### xnn_operator_type ##############################
tools/generate-enum-strings.py --spec src/operator-strings.yaml --output=src/operator-strings.c --enum operator &
tools/generate-enum.py --spec src/operator-strings.yaml --output=src/xnnpack/operator-type.h --enum operator &


############################### xnn_ukernel_type ##############################
tools/generate-enum-strings.py --spec src/ukernel-strings.yaml --output=src/ukernel-strings.c --enum ukernel &
tools/generate-enum.py --spec src/ukernel-strings.yaml --output=src/xnnpack/ukernel-type.h --enum ukernel &

wait
