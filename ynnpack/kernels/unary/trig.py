# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This algorithm is a scalar implementation of the rational polynomial
# approximation of `sin(x)/cos(x)` from
# `src/f32-vsin/rational-5-4.c.in`.

"""Definition of sin/cos kernels."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
# pylint: disable=g-unsafe-pickle-load
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_offset", Float(32)),
    Scalar("output_multiplier", Float(32)),
)
@operator_name("sin")
def sin_fp32(a, x, output_offset, output_multiplier):
  return store(multiply_add(sin(load(a)), output_multiplier, output_offset), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    Scalar("output_offset", Float(64)),
    Scalar("output_multiplier", Float(64)),
)
@operator_name("sin")
def sin_fp64(a, x, output_offset, output_multiplier):
  return store(multiply_add(sin(load(a)), output_multiplier, output_offset), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_offset", Float(32)),
    Scalar("output_multiplier", Float(32)),
)
@operator_name("cos")
def cos_fp32(a, x, output_offset, output_multiplier):
  return store(multiply_add(cos(load(a)), output_multiplier, output_offset), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    Scalar("output_offset", Float(64)),
    Scalar("output_multiplier", Float(64)),
)
@operator_name("cos")
def cos_fp64(a, x, output_offset, output_multiplier):
  return store(multiply_add(cos(load(a)), output_multiplier, output_offset), x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@operator_name("tan")
def tan_fp32(a, x):
  return store(tan(load(a)), x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@operator_name("tan")
def tan_fp64(a, x):
  return store(tan(load(a)), x)
