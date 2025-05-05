#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from gemm_compiler import generate_bf16_f32_gemm_microkernels as bf16_f32
from gemm_compiler import generate_f32_gemm_microkernels as f32
from gemm_compiler import generate_qd8_f32_qc4w_gemm_microkernels as qd8_f32_qc4w
from gemm_compiler import generate_qd8_f32_qc8w_gemm_microkernels as qd8_f32_qc8w
from gemm_compiler import generate_qs8_qc4w_gemm_microkernels as qs8_qc4w
from gemm_compiler import generate_qs8_qc8w_gemm_microkernels as qs8_qc8w


def main(_):
  """Generates all assembly gemm microkernels."""

  bf16_f32.generate_bf16_f32_gemm_microkernels()
  f32.generate_f32_gemm_microkernels()
  qd8_f32_qc4w.generate_qd8_f32_qc4w_gemm_microkernels()
  qd8_f32_qc8w.generate_qd8_f32_qc8w_gemm_microkernels()
  qs8_qc4w.generate_qs8_qc4w_gemm_microkernels()
  qs8_qc8w.generate_qs8_qc8w_gemm_microkernels()


if __name__ == "__main__":
  main(sys.argv[1:])
