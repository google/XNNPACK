// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <utility>

#include <gtest/gtest.h>

#include "slice-operator-tester.h"

constexpr size_t kDim1 = 4;
constexpr size_t kDim2 = 5;
constexpr size_t kDim3 = 7;
constexpr size_t kDim4 = 11;
constexpr size_t kDim5 = 13;
constexpr size_t kDim6 = 17;

// For each dimension test 3 cases:
// 1. offset = 0, size = full dimension
// 2. offset = 1, size = full dimension - 1 (skip first and last 2 elements)
// 3. offset > 0 && offset < size, size = 1 (copy 1 element from the middle)
constexpr std::array<std::pair<size_t, size_t>, 3> kDim1TestCases = {{
    {0, kDim1},
    {1, kDim1 - 3},
    {kDim1 - 1, 1},
}};
constexpr std::array<std::pair<size_t, size_t>, 3> kDim2TestCases = {{
    {0, kDim2},
    {1, kDim2 - 3},
    {kDim2 - 2, 1},
}};
constexpr std::array<std::pair<size_t, size_t>, 3> kDim3TestCases = {{
    {0, kDim3},
    {1, kDim3 - 3},
    {kDim3 - 3, 1},
}};
constexpr std::array<std::pair<size_t, size_t>, 3> kDim4TestCases = {{
    {0, kDim4},
    {1, kDim4 - 3},
    {kDim4 - 4, 1},
}};
constexpr std::array<std::pair<size_t, size_t>, 3> kDim5TestCases = {{
    {0, kDim5},
    {1, kDim5 - 3},
    {kDim5 - 5, 1},
}};
constexpr std::array<std::pair<size_t, size_t>, 3> kDim6TestCases = {{
    {0, kDim6},
    {1, kDim6 - 3},
    {kDim6 - 6, 1},
}};

TEST(SLICE_ND_X8, 1d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    SliceOperatorTester()
        .input_shape({kDim1})
        .offsets({dim1_offset})
        .sizes({dim1_size})
        .TestX8();
  }
}

TEST(SLICE_ND_X8, 2d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      SliceOperatorTester()
          .input_shape({kDim1, kDim2})
          .offsets({dim1_offset, dim2_offset})
          .sizes({dim1_size, dim2_size})
          .TestX8();
    }
  }
}

TEST(SLICE_ND_X8, 3d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        SliceOperatorTester()
            .input_shape({kDim1, kDim2, kDim3})
            .offsets({dim1_offset, dim2_offset, dim3_offset})
            .sizes({dim1_size, dim2_size, dim3_size})
            .TestX8();
      }
    }
  }
}

TEST(SLICE_ND_X8, 4d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        for (const auto& dim4_offset_size : kDim4TestCases) {
          const size_t dim4_offset = dim4_offset_size.first;
          const size_t dim4_size = dim4_offset_size.second;
          SliceOperatorTester()
              .input_shape({kDim1, kDim2, kDim3, kDim4})
              .offsets({dim1_offset, dim2_offset, dim3_offset, dim4_offset})
              .sizes({dim1_size, dim2_size, dim3_size, dim4_size})
              .TestX8();
        }
      }
    }
  }
}

TEST(SLICE_ND_X8, 5d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        for (const auto& dim4_offset_size : kDim4TestCases) {
          const size_t dim4_offset = dim4_offset_size.first;
          const size_t dim4_size = dim4_offset_size.second;
          for (const auto& dim5_offset_size : kDim5TestCases) {
            const size_t dim5_offset = dim5_offset_size.first;
            const size_t dim5_size = dim5_offset_size.second;
            SliceOperatorTester()
                .input_shape({kDim1, kDim2, kDim3, kDim4, kDim5})
                .offsets({dim1_offset, dim2_offset, dim3_offset, dim4_offset, dim5_offset})
                .sizes({dim1_size, dim2_size, dim3_size, dim4_size, dim5_size})
                .TestX8();
          }
        }
      }
    }
  }
}

TEST(SLICE_ND_X8, 6d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        for (const auto& dim4_offset_size : kDim4TestCases) {
          const size_t dim4_offset = dim4_offset_size.first;
          const size_t dim4_size = dim4_offset_size.second;
          for (const auto& dim5_offset_size : kDim5TestCases) {
            const size_t dim5_offset = dim5_offset_size.first;
            const size_t dim5_size = dim5_offset_size.second;
            for (const auto& dim6_offset_size : kDim6TestCases) {
              const size_t dim6_offset = dim6_offset_size.first;
              const size_t dim6_size = dim6_offset_size.second;
              SliceOperatorTester()
                  .input_shape({kDim1, kDim2, kDim3, kDim4, kDim5, kDim6})
                  .offsets({dim1_offset, dim2_offset, dim3_offset, dim4_offset, dim5_offset, dim6_offset})
                  .sizes({dim1_size, dim2_size, dim3_size, dim4_size, dim5_size, dim6_size})
                  .TestX8();
            }
          }
        }
      }
    }
  }
}

TEST(SLICE_ND_X16, 1d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    SliceOperatorTester()
        .input_shape({kDim1})
        .offsets({dim1_offset})
        .sizes({dim1_size})
        .TestX16();
  }
}

TEST(SLICE_ND_X16, 2d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      SliceOperatorTester()
          .input_shape({kDim1, kDim2})
          .offsets({dim1_offset, dim2_offset})
          .sizes({dim1_size, dim2_size})
          .TestX16();
    }
  }
}

TEST(SLICE_ND_X16, 3d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        SliceOperatorTester()
            .input_shape({kDim1, kDim2, kDim3})
            .offsets({dim1_offset, dim2_offset, dim3_offset})
            .sizes({dim1_size, dim2_size, dim3_size})
            .TestX16();
      }
    }
  }
}

TEST(SLICE_ND_X16, 4d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        for (const auto& dim4_offset_size : kDim4TestCases) {
          const size_t dim4_offset = dim4_offset_size.first;
          const size_t dim4_size = dim4_offset_size.second;
          SliceOperatorTester()
              .input_shape({kDim1, kDim2, kDim3, kDim4})
              .offsets({dim1_offset, dim2_offset, dim3_offset, dim4_offset})
              .sizes({dim1_size, dim2_size, dim3_size, dim4_size})
              .TestX16();
        }
      }
    }
  }
}

TEST(SLICE_ND_X16, 5d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        for (const auto& dim4_offset_size : kDim4TestCases) {
          const size_t dim4_offset = dim4_offset_size.first;
          const size_t dim4_size = dim4_offset_size.second;
          for (const auto& dim5_offset_size : kDim5TestCases) {
            const size_t dim5_offset = dim5_offset_size.first;
            const size_t dim5_size = dim5_offset_size.second;
            SliceOperatorTester()
                .input_shape({kDim1, kDim2, kDim3, kDim4, kDim5})
                .offsets({dim1_offset, dim2_offset, dim3_offset, dim4_offset, dim5_offset})
                .sizes({dim1_size, dim2_size, dim3_size, dim4_size, dim5_size})
                .TestX16();
          }
        }
      }
    }
  }
}

TEST(SLICE_ND_X16, 6d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        for (const auto& dim4_offset_size : kDim4TestCases) {
          const size_t dim4_offset = dim4_offset_size.first;
          const size_t dim4_size = dim4_offset_size.second;
          for (const auto& dim5_offset_size : kDim5TestCases) {
            const size_t dim5_offset = dim5_offset_size.first;
            const size_t dim5_size = dim5_offset_size.second;
            for (const auto& dim6_offset_size : kDim6TestCases) {
              const size_t dim6_offset = dim6_offset_size.first;
              const size_t dim6_size = dim6_offset_size.second;
              SliceOperatorTester()
                  .input_shape({kDim1, kDim2, kDim3, kDim4, kDim5, kDim6})
                  .offsets({dim1_offset, dim2_offset, dim3_offset, dim4_offset, dim5_offset, dim6_offset})
                  .sizes({dim1_size, dim2_size, dim3_size, dim4_size, dim5_size, dim6_size})
                  .TestX16();
            }
          }
        }
      }
    }
  }
}

TEST(SLICE_ND_X32, 1d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    SliceOperatorTester()
        .input_shape({kDim1})
        .offsets({dim1_offset})
        .sizes({dim1_size})
        .TestX32();
  }
}

TEST(SLICE_ND_X32, 2d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
          SliceOperatorTester()
              .input_shape({kDim1, kDim2})
              .offsets({dim1_offset, dim2_offset})
              .sizes({dim1_size, dim2_size})
              .TestX32();
    }
  }
}

TEST(SLICE_ND_X32, 3d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        SliceOperatorTester()
            .input_shape({kDim1, kDim2, kDim3})
            .offsets({dim1_offset, dim2_offset, dim3_offset})
            .sizes({dim1_size, dim2_size, dim3_size})
            .TestX32();
      }
    }
  }
}

TEST(SLICE_ND_X32, 4d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        for (const auto& dim4_offset_size : kDim4TestCases) {
          const size_t dim4_offset = dim4_offset_size.first;
          const size_t dim4_size = dim4_offset_size.second;
          SliceOperatorTester()
              .input_shape({kDim1, kDim2, kDim3, kDim4})
              .offsets({dim1_offset, dim2_offset, dim3_offset, dim4_offset})
              .sizes({dim1_size, dim2_size, dim3_size, dim4_size})
              .TestX32();
        }
      }
    }
  }
}

TEST(SLICE_ND_X32, 5d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        for (const auto& dim4_offset_size : kDim4TestCases) {
          const size_t dim4_offset = dim4_offset_size.first;
          const size_t dim4_size = dim4_offset_size.second;
          for (const auto& dim5_offset_size : kDim5TestCases) {
            const size_t dim5_offset = dim5_offset_size.first;
            const size_t dim5_size = dim5_offset_size.second;
            SliceOperatorTester()
                .input_shape({kDim1, kDim2, kDim3, kDim4, kDim5})
                .offsets({dim1_offset, dim2_offset, dim3_offset, dim4_offset, dim5_offset})
                .sizes({dim1_size, dim2_size, dim3_size, dim4_size, dim5_size})
                .TestX32();
          }
        }
      }
    }
  }
}

TEST(SLICE_ND_X32, 6d) {
  for (const auto& dim1_offset_size : kDim1TestCases) {
    const size_t dim1_offset = dim1_offset_size.first;
    const size_t dim1_size = dim1_offset_size.second;
    for (const auto& dim2_offset_size : kDim2TestCases) {
      const size_t dim2_offset = dim2_offset_size.first;
      const size_t dim2_size = dim2_offset_size.second;
      for (const auto& dim3_offset_size : kDim3TestCases) {
        const size_t dim3_offset = dim3_offset_size.first;
        const size_t dim3_size = dim3_offset_size.second;
        for (const auto& dim4_offset_size : kDim4TestCases) {
          const size_t dim4_offset = dim4_offset_size.first;
          const size_t dim4_size = dim4_offset_size.second;
          for (const auto& dim5_offset_size : kDim5TestCases) {
            const size_t dim5_offset = dim5_offset_size.first;
            const size_t dim5_size = dim5_offset_size.second;
            for (const auto& dim6_offset_size : kDim6TestCases) {
              const size_t dim6_offset = dim6_offset_size.first;
              const size_t dim6_size = dim6_offset_size.second;
              SliceOperatorTester()
                  .input_shape({kDim1, kDim2, kDim3, kDim4, kDim5, kDim6})
                  .offsets({dim1_offset, dim2_offset, dim3_offset, dim4_offset, dim5_offset, dim6_offset})
                  .sizes({dim1_size, dim2_size, dim3_size, dim4_size, dim5_size, dim6_size})
                  .TestX32();
            }
          }
        }
      }
    }
  }
}
