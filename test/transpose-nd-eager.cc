// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "transpose-operator-tester.h"

TEST(TRANSPOSE_ND_X8, Run1D) {
  TransposeOperatorTester()
      .num_dims(1)
      .shape({713})
      .perm({0})
      .TestRunX8();
}

TEST(TRANSPOSE_ND_X8, Run2D) {
  std::vector<size_t> perm{0,1};
  do {
    TransposeOperatorTester()
        .num_dims(2)
        .shape({37, 113})
        .perm(perm)
        .TestRunX8();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X8, Run3D) {
  std::vector<size_t> perm{0,1,2};
  do {
    TransposeOperatorTester()
        .num_dims(3)
        .shape({5, 7, 11})
        .perm(perm)
        .TestRunX8();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X8, Run4D) {
  std::vector<size_t> perm{0,1,2,3};
  do {
    TransposeOperatorTester()
        .num_dims(4)
        .shape({5,7,11,13})
        .perm(perm)
        .TestRunX8();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X8, Run5D) {
  std::vector<size_t> perm{0,1,2,3,4};
  do {
    TransposeOperatorTester()
        .num_dims(5)
        .shape({3,5,7,11,13})
        .perm(perm)
        .TestRunX8();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X8, Run6D) {
  std::vector<size_t> perm{0,1,2,3,4,5};
  do {
    TransposeOperatorTester()
        .num_dims(6)
        .shape({2,3,5,7,11,13})
        .perm(perm)
        .TestRunX8();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X16, Run1D) {
  TransposeOperatorTester()
      .num_dims(1)
      .shape({713})
      .perm({0})
      .TestRunX16();
}

TEST(TRANSPOSE_ND_X16, Run2D) {
  std::vector<size_t> perm{0,1};
  do {
    TransposeOperatorTester()
        .num_dims(2)
        .shape({37, 113})
        .perm(perm)
        .TestRunX16();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X16, Run3D) {
  std::vector<size_t> perm{0,1,2};
  do {
    TransposeOperatorTester()
        .num_dims(3)
        .shape({5, 7, 11})
        .perm(perm)
        .TestRunX16();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X16, Run4D) {
  std::vector<size_t> perm{0,1,2,3};
  do {
    TransposeOperatorTester()
        .num_dims(4)
        .shape({5,7,11,13})
        .perm(perm)
        .TestRunX16();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X16, Run6D) {
  std::vector<size_t> perm{0,1,2,3,4,5};
  do {
    TransposeOperatorTester()
        .num_dims(6)
        .shape({2,3,5,7,11,13})
        .perm(perm)
        .TestRunX16();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, Run1D) {
  TransposeOperatorTester()
      .num_dims(1)
      .shape({713})
      .perm({0})
      .TestRunX32();
}

TEST(TRANSPOSE_ND_X32, Run2D) {
  std::vector<size_t> perm{0,1};
  do {
    TransposeOperatorTester()
        .num_dims(2)
        .shape({37, 113})
        .perm(perm)
        .TestRunX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, Run3D) {
  std::vector<size_t> perm{0,1,2};
  do {
    TransposeOperatorTester()
        .num_dims(3)
        .shape({5, 7, 11})
        .perm(perm)
        .TestRunX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, Run4D) {
  std::vector<size_t> perm{0,1,2,3};
  do {
    TransposeOperatorTester()
        .num_dims(4)
        .shape({5,7,11,13})
        .perm(perm)
        .TestRunX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, Run5D) {
  std::vector<size_t> perm{0,1,2,3,4};
  do {
    TransposeOperatorTester()
        .num_dims(5)
        .shape({3,5,7,11,13})
        .perm(perm)
        .TestRunX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, Run6D) {
  std::vector<size_t> perm{0,1,2,3,4,5};
  do {
    TransposeOperatorTester()
        .num_dims(6)
        .shape({2,3,5,7,11,13})
        .perm(perm)
        .TestRunX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X64, Run1D) {
  TransposeOperatorTester()
      .num_dims(1)
      .shape({713})
      .perm({0})
      .TestRunX64();
}

TEST(TRANSPOSE_ND_X64, Run2D) {
  std::vector<size_t> perm{0,1};
  do {
    TransposeOperatorTester()
        .num_dims(2)
        .shape({37, 113})
        .perm(perm)
        .TestRunX64();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X64, Run3D) {
  std::vector<size_t> perm{0,1,2};
  do {
    TransposeOperatorTester()
        .num_dims(3)
        .shape({5, 7, 11})
        .perm(perm)
        .TestRunX64();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X64, Run4D) {
  std::vector<size_t> perm{0,1,2,3};
  do {
    TransposeOperatorTester()
        .num_dims(4)
        .shape({5,7,11,13})
        .perm(perm)
        .TestRunX64();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X64, Run5D) {
  std::vector<size_t> perm{0,1,2,3,4};
  do {
    TransposeOperatorTester()
        .num_dims(5)
        .shape({3,5,7,11,13})
        .perm(perm)
        .TestRunX64();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X64, Run6D) {
  std::vector<size_t> perm{0,1,2,3,4,5};
  do {
    TransposeOperatorTester()
        .num_dims(6)
        .shape({2,3,5,7,11,13})
        .perm(perm)
        .TestRunX64();
  } while (std::next_permutation(perm.begin(), perm.end()));
}
