# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base class for x86 dot kernel generators.

Handles accumulating C tiles to the output.
"""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import dot_base, indent


class x86(dot_base):
  def __init__(self, arch, dot_type, c_type, bits, tile_shape):
    super().__init__(arch, dot_type)
    self.bits = bits
    self.tile_shape = tile_shape
    self.c_type = c_type

  def header(self):
    return """
#include <immintrin.h>

""" + super().header()

  def _mm(self, bits=0):
    if bits == 0:
      bits = self.bits
    if bits <= 128:
      return "_mm"
    else:
      return f"_mm{bits}"

  def init_c_tile(self, i, j):
    if self.c_type == "float":
      return f"__m{self.bits} c_{i}_{j} = {self._mm()}_setzero_ps();\n"
    elif self.c_type == "int32_t":
      return f"__m{self.bits}i c_{i}_{j} = {self._mm()}_setzero_si{self.bits}();\n"
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def add_c_tile(self, i, j):
    if self.c_type == "float":
      io_bits = min(self.bits, self.block_shape[1] * 32)
      load = f"{self._mm(io_bits)}_loadu_ps({self.c_in_ptr(i, j)})"
      if io_bits < self.bits:
        # The tile is smaller than a vector, do a smaller load and cast.
        load = f"{self._mm()}_castps{io_bits}_ps{self.bits}({load})"
      return f"c_{i}_{j} = {self._mm()}_add_ps(c_{i}_{j}, {load});\n"
    elif self.c_type == "int32_t":
      return (
          f"c_{i}_{j} = {self._mm()}_add_epi32(c_{i}_{j},"
          f" {self._mm()}_loadu_si{self.bits}({self.c_in_ptr(i, j, f'__m{self.bits}i')}));\n"
      )
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def store_c_tile(self, i, j):
    if self.c_type == "float":
      io_bits = min(self.block_shape[1] * 32, self.bits)
      c_ij = f"c_{i}_{j}"
      if io_bits < self.bits:
        # The tile is smaller than a vector, cast and do a smaller store.
        c_ij = f"{self._mm()}_castps{self.bits}_ps{io_bits}({c_ij})"
      return f"{self._mm(io_bits)}_storeu_ps({self.c_out_ptr(i, j)}, {c_ij});\n"
    elif self.c_type == "int32_t":
      return (
          f"{self._mm()}_storeu_si{self.bits}({self.c_out_ptr(i, j, f'__m{self.bits}i')},"
          f" c_{i}_{j});\n"
      )
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def shift_c_tiles(self, n):
    assert n % self.tile_shape[1] == 0
    result = ""

    # Shift the accumulator registers down and the pointers up.
    for i in range(0, self.block_shape[0]):
      for j in range(0, n, self.tile_shape[1]):
        result += f"c_{i}_{j} = c_{i}_{j + n};\n"

    result += f"N -= {n};\n"
    result += (
        f"C_in = C_in ? offset_bytes(C_in, {n} * sizeof({self.c_type})) :"
        " nullptr;\n"
    )
    result += f"C_out = offset_bytes(C_out, {n} * sizeof({self.c_type}));\n"
    return result

  def add_c_block_vectors(self, n):
    assert(n % (self.bits//32) == 0)
    result = self.add_c_tiles(n)

    result += self.shift_c_tiles(n)
    return result

  def add_c_block_vector_tail(self):
    raise NotImplementedError()

  def add_c_block_tail(self):
    result = ""
    n = self.block_shape[1] // 2
    while n >= self.bits//32:
      # This might be a whole vector.
      result += f"if (N & {n}) {{\n"
      result += indent(self.add_c_block_vectors(n), "  ")
      result += "\n}\n"
      n //= 2
    result += "if (N > 0) {\n"
    result += indent(self.add_c_block_vector_tail(), "  ")
    result += "\n}\n"

    return result


class x86_sse2(x86):
  def add_c_block_vector_tail(self):
    result = ""
    result += f"assert(N < {self.bits//32});\n"

    result += "if (N & 2) {\n"

    # Load all of the output and add it, before writing anything.
    add_c_tiles = ""
    for i in range(0, self.block_shape[0]):
      add_c_tiles += (
          f"c_{i}_0 = {self._mm()}_add_ps(c_{i}_0,"
          f" _mm_loadl_pi(_mm_setzero_ps(), {self.c_in_ptr(i, 0, '__m64')}));\n"
      )
    result += "  if (C_in) {\n"
    result += indent(add_c_tiles, "    ") + "\n"
    result += "  }\n"

    for i in reversed(range(0, self.block_shape[0], self.tile_shape[0])):
      result += f"  _mm_storel_pi({self.c_out_ptr(i, 0, '__m64')}, c_{i}_0);\n"
      result += f"  c_{i}_0 = _mm_movehl_ps(c_{i}_0, c_{i}_0);\n"
    result += (
        f"  C_in = C_in ? offset_bytes(C_in, 2 * sizeof({self.c_type})) :"
        " nullptr;\n"
    )
    result += f"  C_out = offset_bytes(C_out, 2 * sizeof({self.c_type}));\n"
    result += "}\n"
    result += "if (N & 1) {\n"
    # Load all of the output and add it, before writing anything.
    add_c_tiles = ""
    for i in range(0, self.block_shape[0]):
      add_c_tiles += (
          f"c_{i}_0 = {self._mm()}_add_ss(c_{i}_0,"
          f" _mm_load_ss({self.c_in_ptr(i, 0)}));\n"
      )
    result += "  if (C_in) {\n"
    result += indent(add_c_tiles, "    ") + "\n"
    result += "  }\n"

    for i in reversed(range(0, self.block_shape[0], self.tile_shape[0])):
      result += f"  _mm_store_ss({self.c_out_ptr(i, 0)}, c_{i}_0);\n"
    result += "}\n"
    result += f"N = 0;\n"
    return result


class x86_avx(x86):
  def header(self):
    return """
#include <immintrin.h>

""" + super().header() + """
static const int32_t mask_table[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
"""

  def add_c_block_vector_tail(self):
    result = ""
    result += f"assert(N < {self.bits//32});\n"

    # Load all of the output and add it, before writing anything.
    result += f"const __m{self.bits}i mask = {self._mm()}_loadu_si{self.bits}((const __m{self.bits}i*) &mask_table[8 - N]);\n"

    add_c_tiles = ""
    for i in range(0, self.block_shape[0]):
      if self.c_type == "float":
        add_c_tiles += (
            f"c_{i}_0 = {self._mm()}_add_ps(c_{i}_0,"
            f" {self._mm()}_maskload_ps({self.c_in_ptr(i, 0)}, mask));\n"
        )
      elif self.c_type == "int32_t":
        add_c_tiles += (
            f"c_{i}_0 = {self._mm()}_add_epi32(c_{i}_0,"
            f" {self._mm()}_maskload_epi32({self.c_in_ptr(i, 0)}, mask));\n"
        )
      else:
        raise ValueError(f"Unsupported c_type: {self.c_type}")

    result += "if (C_in) {\n"
    result += indent(add_c_tiles, "  ") + "\n"
    result += "}\n"

    for i in reversed(range(0, self.block_shape[0], self.tile_shape[0])):
      if self.c_type == "float":
        result += (
            f"{self._mm()}_maskstore_ps({self.c_out_ptr(i, 0)}, mask,"
            f" c_{i}_0);\n"
        )
      elif self.c_type == "int32_t":
        result += (
            f"{self._mm()}_maskstore_epi32({self.c_out_ptr(i, 0)}, mask,"
            f" c_{i}_0);\n"
        )
      else:
        raise ValueError(f"Unsupported c_type: {self.c_type}")

    result += f"N = 0;\n"
    return result


class x86_avx512(x86):
  def add_c_block_vector_tail(self):
    result = ""
    result += f"assert(N < {self.bits//32});\n"

    result += f"const __mmask16 mask = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << N) - 1) >> 0));\n"

    add_c_tiles = ""
    # Load all of the output and add it, before writing anything.
    for i in range(0, self.block_shape[0]):
      if self.c_type == "float":
        add_c_tiles += (
            f"c_{i}_0 = {self._mm()}_add_ps(c_{i}_0,"
            f" {self._mm()}_mask_loadu_ps(c_{i}_0, mask,"
            f" {self.c_in_ptr(i, 0)}));\n"
        )
      elif self.c_type == "int32_t":
        add_c_tiles += (
            f"c_{i}_0 = {self._mm()}_add_epi32(c_{i}_0,"
            f" {self._mm()}_mask_loadu_epi32(c_{i}_0, mask,"
            f" {self.c_in_ptr(i, 0)}));\n"
        )
      else:
        raise ValueError(f"Unsupported c_type: {self.c_type}")
    result += "if (C_in) {\n"
    result += indent(add_c_tiles, "  ") + "\n"
    result += "}\n"

    for i in reversed(range(0, self.block_shape[0], self.tile_shape[0])):
      if self.c_type == "float":
        result += (
            f"{self._mm()}_mask_storeu_ps({self.c_out_ptr(i, 0)}, mask,"
            f" c_{i}_0);\n"
        )
      elif self.c_type == "int32_t":
        result += (
            f"{self._mm()}_mask_storeu_epi32({self.c_out_ptr(i, 0)}, mask,"
            f" c_{i}_0);\n"
        )
      else:
        raise ValueError(f"Unsupported c_type: {self.c_type}")
    result += "N = 0;\n"
    return result
