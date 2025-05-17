# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description:
#   XNNPACK - optimized floating-point neural network operators library

OPERATOR_SRCS = [
    "src/operator-delete.c",
    "src/operator-run.c",
    "src/operators/argmax-pooling-nhwc.c",
    "src/operators/average-pooling-nhwc.c",
    "src/operators/batch-matrix-multiply-nc.c",
    "src/operators/binary-elementwise-nd.c",
    "src/operators/constant-pad-nd.c",
    "src/operators/convolution-nchw.c",
    "src/operators/convolution-nhwc.c",
    "src/operators/deconvolution-nhwc.c",
    "src/operators/dynamic-fully-connected-nc.c",
    "src/operators/fully-connected-nc.c",
    "src/operators/max-pooling-nhwc.c",
    "src/operators/pack-lh.c",
    "src/operators/reduce-nd.c",
    "src/operators/resize-bilinear-nchw.c",
    "src/operators/resize-bilinear-nhwc.c",
    "src/operators/rope-nthc.c",
    "src/operators/slice-nd.c",
    "src/operators/softmax-nc.c",
    "src/operators/transpose-nd.c",
    "src/operators/unary-elementwise-nc.c",
    "src/operators/unpooling-nhwc.c",
]

SUBGRAPH_SRCS = [
    "src/memory-planner.c",
    "src/runtime.c",
    "src/subgraph.c",
    "src/subgraph/argmax-pooling-2d.c",
    "src/subgraph/average-pooling-2d.c",
    "src/subgraph/batch-matrix-multiply.c",
    "src/subgraph/binary.c",
    "src/subgraph/concatenate.c",
    "src/subgraph/convolution-2d.c",
    "src/subgraph/copy.c",
    "src/subgraph/deconvolution-2d.c",
    "src/subgraph/deprecated.c",
    "src/subgraph/depth-to-space-2d.c",
    "src/subgraph/depthwise-convolution-2d.c",
    "src/subgraph/even-split.c",
    "src/subgraph/fully-connected-sparse.c",
    "src/subgraph/fully-connected.c",
    "src/subgraph/max-pooling-2d.c",
    "src/subgraph/pack-lh.c",
    "src/subgraph/reshape-helpers.c",
    "src/subgraph/rope.c",
    "src/subgraph/softmax.c",
    "src/subgraph/space-to-depth-2d.c",
    "src/subgraph/static-constant-pad.c",
    "src/subgraph/static-reduce.c",
    "src/subgraph/static-resize-bilinear-2d.c",
    "src/subgraph/static-slice.c",
    "src/subgraph/static-transpose.c",
    "src/subgraph/subgraph-utils.c",
    "src/subgraph/unary.c",
    "src/subgraph/unpooling-2d.c",
    "src/subgraph/validation.c",
    "src/tensor.c",
]

TABLE_SRCS = [
    "src/tables/exp2-k-over-64.c",
    "src/tables/exp2-k-over-2048.c",
    "src/tables/exp2minus-k-over-4.c",
    "src/tables/exp2minus-k-over-8.c",
    "src/tables/exp2minus-k-over-16.c",
    "src/tables/exp2minus-k-over-32.c",
    "src/tables/exp2minus-k-over-64.c",
    "src/tables/exp2minus-k-over-2048.c",
    "src/tables/vlog.c",
]

LOGGING_SRCS = [
    "src/enums/allocation-type.c",
    "src/enums/datatype-strings.c",
    "src/enums/microkernel-type.c",
    "src/enums/node-type.c",
    "src/enums/operator-type.c",
    "src/log.c",
]
