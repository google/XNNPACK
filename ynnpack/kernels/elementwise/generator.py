"""Generate optimized YNNPACK elementwise kernels."""

from collections.abc import Sequence
import importlib
import sys

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.arm import *  # pylint: disable=wildcard-import
from ynnpack.kernels.elementwise.x86 import *  # pylint: disable=wildcard-import

arch_to_target = {
    "x86_sse2": X86(["SSE2"]),
    "x86_sse41": X86(["SSE41"]),
    "x86_avx": X86(["AVX"]),
    "x86_avx2": X86(["AVX2"]),
    "x86_fma3": X86(["FMA3"]),
    "x86_f16c": X86(["F16C"]),
    "x86_avx2_fma3": X86(["AVX2", "FMA3"]),
    "x86_avx512f": X86(["AVX512F"]),
    "x86_avx512bw": X86(["AVX512BW"]),
    "arm_neon": ARM(["NEON"]),
    "arm_neonfp16": ARM(["NEONFP16"]),
    "arm_neon_fma": ARM(["NEON", "FMA"]),
}


def generate(module, argv: Sequence[str]) -> None:

  output_src = argv[1]
  output_inc = argv[2]
  target = arch_to_target[argv[3]]

  src = ""
  inc = ""

  src += target.header

  for name_shape in argv[4:]:
    name, shape = name_shape.split(",")
    vectorize, unroll = shape.split("x")
    fn = module[name]
    assert fn
    src_i, inc_i = target.compile_function(
        name, fn, [(int(unroll), int(vectorize))]
    )

    src += src_i
    inc += inc_i

  with open(output_src, "w") as f:
    f.write(src)
  with open(output_inc, "w") as f:
    f.write(inc)
