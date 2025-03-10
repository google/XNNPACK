#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc


class BaseArchitecture:
  """Base architecture for GEMM microkernel generation."""

  def __init__(self, m: int, n: int):
    """Initializes the `BaseArchitecture` with the given number of rows and columns."""
    self._m = m
    self._n = n // self.n_step()
    self._c = 1

  @property
  def m(self) -> int:
    return self._m

  @property
  def n(self) -> int:
    return self._n

  @property
  def c(self) -> int:
    raise NotImplementedError

  def labels(self):
    return [
        'zero',
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine',
        'ten',
        'eleven',
    ]

  @abc.abstractmethod
  def adjust_kc(self) -> str:
    """Returns an assembly string computing the adjusted `kc`."""
    raise NotImplementedError

  @abc.abstractmethod
  def outer_loop_prepare(self) -> str:
    """Returns an assembly string preparing the outer loop."""
    raise NotImplementedError

  @abc.abstractmethod
  def init_accumulators(self) -> str:
    """Returns an assembly string initializing the accumulators."""
    raise NotImplementedError

  @abc.abstractmethod
  def register_bytes(self) -> int:
    raise NotImplementedError

  @abc.abstractmethod
  def isa(self) -> str:
    """Returns a string identifying the ISA of this `BaseArchitecture`."""
    raise NotImplementedError

  @abc.abstractmethod
  def dequantize(self) -> str:
    """Returns an assembly string dequantizing the accumulators."""
    raise NotImplementedError

  @abc.abstractmethod
  def store(self) -> str:
    """Returns an assembly string storing the results."""
    raise NotImplementedError

  @abc.abstractmethod
  def astride_register(self):
    """Returns the register containing a_stride."""
    raise NotImplementedError

  @abc.abstractmethod
  def compute_asm(self) -> dict[str, list[str]]:
    raise NotImplementedError

  @abc.abstractmethod
  def input_asm(self) -> dict[str, list[str]]:
    raise NotImplementedError

  @abc.abstractmethod
  def weights_asm(self) -> dict[str, list[str]]:
    raise NotImplementedError

  @abc.abstractmethod
  def w_registers(self) -> list[str]:
    raise NotImplementedError

  @abc.abstractmethod
  def a_registers(self, idx) -> str:
    raise NotImplementedError

  @abc.abstractmethod
  def kc_register(self):
    """Returns the register containing kc, the number of channels (the reduction dimensions)."""
    raise NotImplementedError

  @abc.abstractmethod
  def k_register(self):
    """Returns the register containing k, the current channel being processed."""
    raise NotImplementedError

  @abc.abstractmethod
  def cm_stride_register(self):
    """Returns the register containing cm_stride."""
    raise NotImplementedError

  @abc.abstractmethod
  def am_registers(self):
    """Returns the registers containing the pointers to each row of A (LHS)."""
    raise NotImplementedError

  @abc.abstractmethod
  def a_ptr_register(self):
    """Returns the register containing the A pointer."""
    raise NotImplementedError

  @abc.abstractmethod
  def c_ptr_register(self):
    """Returns the register containing the C pointer."""
    raise NotImplementedError

  @abc.abstractmethod
  def cm_registers(self):
    """Returns the registers containing the pointers to each row of C (Output)."""
    raise NotImplementedError

  @abc.abstractmethod
  def acc_registers(self):
    """Returns the accumulator registers."""
    raise NotImplementedError

  @abc.abstractmethod
  def w_ptr_register(self):
    """Returns the register containing the weight's pointer."""
    raise NotImplementedError

  @abc.abstractmethod
  def min_register(self):
    """Returns the register containing the min value for clamping."""
    raise NotImplementedError

  @abc.abstractmethod
  def max_register(self):
    """Returns the register containing the max value for clamping."""
    raise NotImplementedError

  @abc.abstractmethod
  def nc_register(self):
    """Returns the register containing nc, the number of output rows processed per iteration."""
    raise NotImplementedError

  @abc.abstractmethod
  def mr_register(self):
    """Returns the register containing mr, the number of input rows processed per kernel call."""
    raise NotImplementedError

  @abc.abstractmethod
  def tmp_gp_registers(self):
    """Returns some general purpose registers which may be used for storing temporary data."""
    raise NotImplementedError

  @abc.abstractmethod
  def jump_to_label(self, label):
    """Jump to the given label."""
    raise NotImplementedError

  @abc.abstractmethod
  def function_name(self):
    """Returns the microkernel name."""
    raise NotImplementedError

  @abc.abstractmethod
  def header(self):
    """Returns the assembly header."""
    raise NotImplementedError

  @abc.abstractmethod
  def input_output_register_setup(self):
    """Setup the input (A) and output (C) registers."""
    raise NotImplementedError

  @abc.abstractmethod
  def max_m_before_spilling(self):
    """How large can M be before spilling A and C registers to the stack."""
    raise NotImplementedError

  @abc.abstractmethod
  def read_a_registers(self):
    """Read the A registers from the stack."""
    raise NotImplementedError

  @abc.abstractmethod
  def increment_ptr(self, ptr, step):
    """Increment the given pointer by step bytes."""
    raise NotImplementedError

  @abc.abstractmethod
  def initialize_k_register(self):
    """Initialized the general purpose register for inner loop control."""
    raise NotImplementedError

  @abc.abstractmethod
  def cmp_k_and_jump_if_less(self, label):
    """If k is less than kc, then do another iteration of the inner loop."""
    raise NotImplementedError

  @abc.abstractmethod
  def epilogue(self):
    """Returns the function epilogue."""
    raise NotImplementedError

  @abc.abstractmethod
  def inner_loop(self):
    """Returns the assemebly for the microkernel's inner loop."""
    raise NotImplementedError

  @abc.abstractmethod
  def n_step(self) -> int:
    """Returns the n step for the microkernel's inner loop."""
    raise NotImplementedError

  @abc.abstractmethod
  def prefix(self) -> str:
    """Returns the register name prefix."""
    raise NotImplementedError

  @abc.abstractmethod
  def clamp_min(self, reg: str, prefix: str, other_reg: str):
    """Returns an assembly string computing the minimum clamp."""
    raise NotImplementedError

  @abc.abstractmethod
  def clamp_max(self, reg: str, prefix: str, other_reg: str):
    """Returns an assembly string computing the maximum clamp."""
    raise NotImplementedError

  def clamp(self) -> str:
    """Returns an assembly string computing the minimum/maximum clamp."""
    acc_registers = self.acc_registers()
    asm_string = ''
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        asm_string += self.clamp_min(
            reg=acc_registers[self.m * nr + mr], prefix=self.prefix()
        )
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        asm_string += self.clamp_max(
            reg=acc_registers[self.m * nr + mr], prefix=self.prefix()
        )
    return asm_string
