from collections.abc import Sequence
import sys

from ynnpack.kernels.elementwise.generator import generate
from ynnpack.kernels.unary.convert import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.exp import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.kernels import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.sigmoid import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.sine_cosine import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.tanh import *  # pylint: disable=wildcard-import


def main(argv: Sequence[str]) -> None:
  generate(globals(), argv)


if __name__ == "__main__":
  main(sys.argv)
