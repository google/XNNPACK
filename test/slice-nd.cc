// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cstddef>
#include <utility>

#include <gtest/gtest.h>
#include "slice-operator-tester.h"

constexpr size_t kDim1 = 4;
constexpr size_t kDim2 = 5;
constexpr size_t kDim3 = 6;
constexpr size_t kDim4 = 7;
constexpr size_t kDim5 = 8;
constexpr size_t kDim6 = 9;

// For each dimension test 3 cases:
// 1. offset = 0, size = full dimension
// 2. offset = 1, size = full dimension - 3 (skip first and last 2 elements)
// 3. offset > 0 && offset < size, size = 1 (copy 1 element from the middle)
static_assert(kDim1 > 3, "kDim1 must be more than 3");
constexpr std::array<std::pair<size_t, size_t>, 3> kDim1TestCases = {{
    {0, kDim1},
    {1, kDim1 - 3},
    {kDim1 - 1, 1},
}};
static_assert(kDim2 > 3, "kDim2 must be more than 3");
constexpr std::array<std::pair<size_t, size_t>, 3> kDim2TestCases = {{
    {0, kDim2},
    {1, kDim2 - 3},
    {kDim2 - 2, 1},
}};
static_assert(kDim3 > 3, "kDim3 must be more than 3");
constexpr std::array<std::pair<size_t, size_t>, 3> kDim3TestCases = {{
    {0, kDim3},
    {1, kDim3 - 3},
    {kDim3 - 3, 1},
}};
static_assert(kDim4 > 4, "kDim4 must be more than 4");
constexpr std::array<std::pair<size_t, size_t>, 3> kDim4TestCases = {{
    {0, kDim4},
    {1, kDim4 - 3},
    {kDim4 - 4, 1},
}};
static_assert(kDim5 > 5, "kDim5 must be more than 5");
constexpr std::array<std::pair<size_t, size_t>, 3> kDim5TestCases = {{
    {0, kDim5},
    {1, kDim5 - 3},
    {kDim5 - 5, 1},
}};
static_assert(kDim6 > 6, "kDim6 must be more than 6");
constexpr std::array<std::pair<size_t, size_t>, 3> kDim6TestCases = {{
    {0, kDim6},
    {1, kDim6 - 3},
    {kDim6 - 6, 1},
}};

TEST(SLICE_ND_X8, slice_1d) {
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

TEST(SLICE_ND_X8, slice_2d) {
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

TEST(SLICE_ND_X8, slice_3d) {
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

TEST(SLICE_ND_X8, slice_4d) {
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

TEST(SLICE_ND_X8, slice_5d) {
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

TEST(SLICE_ND_X8, slice_6d) {
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

TEST(SLICE_ND_X16, slice_1d) {
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

TEST(SLICE_ND_X16, slice_2d) {
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

TEST(SLICE_ND_X16, slice_3d) {
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

TEST(SLICE_ND_X16, slice_4d) {
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

TEST(SLICE_ND_X16, slice_5d) {
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

TEST(SLICE_ND_X16, slice_6d) {
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

TEST(SLICE_ND_X32, slice_1d) {
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

TEST(SLICE_ND_X32, slice_2d) {
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

TEST(SLICE_ND_X32, slice_3d) {
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

TEST(SLICE_ND_X32, slice_4d) {
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

TEST(SLICE_ND_X32, slice_5d) {
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

TEST(SLICE_ND_X32, slice_6d) {
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
