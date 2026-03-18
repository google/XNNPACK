from collections.abc import Sequence
import sys

from ynnpack.kernels.binary.kernels import *
from ynnpack.kernels.elementwise.generator import generate


def main(argv: Sequence[str]) -> None:
  generate(globals(), argv)


if __name__ == "__main__":
  main(sys.argv)
