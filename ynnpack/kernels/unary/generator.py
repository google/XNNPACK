from collections.abc import Sequence
import sys

from ynnpack.kernels.elementwise.generator import generate
from ynnpack.kernels.unary.exp import *
from ynnpack.kernels.unary.kernels import *
from ynnpack.kernels.unary.sine_cosine import *
from ynnpack.kernels.unary.tanh import *


def main(argv: Sequence[str]) -> None:
  generate(globals(), argv)


if __name__ == "__main__":
  main(sys.argv)
