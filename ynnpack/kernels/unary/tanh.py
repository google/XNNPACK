# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Definition of tanh kernels."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_offset", Float(32)),
    Scalar("output_multiplier", Float(32)),
)
@operator_name("tanh")
def tanh_fp32(a, x, output_offset, output_multiplier):
  va = load(a)
  return store(multiply_add(tanh(va), output_multiplier, output_offset), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    Scalar("output_offset", Float(64)),
    Scalar("output_multiplier", Float(64)),
)
@operator_name("tanh")
def tanh_fp64(a, x, output_offset, output_multiplier):
  va = load(a)
  return store(multiply_add(tanh(va), output_multiplier, output_offset), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_offset", Float(32)),
    Scalar("output_multiplier", Float(32)),
)
@operator_name("tanh")
@kernel_flags("unary_flag::precision_approx")
def approx_tanh_fp32(a, x, output_offset, output_multiplier):
  va = load(a)
  return store(
      multiply_add(approx_tanh(va), output_multiplier, output_offset), x
  )
