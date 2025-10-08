from collections.abc import Sequence
import sys

from ynnpack.kernels.elementwise.generator import generate
from ynnpack.kernels.ternary.convert import *  # pylint: disable=wildcard-import
from ynnpack.kernels.ternary.kernels import *  # pylint: disable=wildcard-import


def main(argv: Sequence[str]) -> None:
  generate(globals(), argv)


if __name__ == "__main__":
  main(sys.argv)
