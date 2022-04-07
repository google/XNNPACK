#include <gtest/gtest.h>

#include <algorithm>

#include "transpose-operator-tester.h"

TEST(TRANSPOSE_ND_X32, 1D) {
  TransposeOperatorTester()
      .num_dims(1)
      .shape({713})
      .perm({0})
      .TestX32();
}

TEST(TRANSPOSE_ND_X32, 2D) {
  std::vector<size_t> perm{0,1};
  do {
    TransposeOperatorTester()
        .num_dims(2)
        .shape({37, 113})
        .perm(perm)
        .TestX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, 3D) {
  std::vector<size_t> perm{0,1,2};
  do {
    TransposeOperatorTester()
        .num_dims(3)
        .shape({5, 7, 11})
        .perm(perm)
        .TestX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, 4D) {
  std::vector<size_t> perm{0,1,2,3};
  do {
    TransposeOperatorTester()
        .num_dims(4)
        .shape({5,7,11,13})
        .perm(perm)
        .TestX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, 5D) {
  std::vector<size_t> perm{0,1,2,3,4};
  do {
    TransposeOperatorTester()
        .num_dims(5)
        .shape({3,5,7,11,13})
        .perm(perm)
        .TestX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TRANSPOSE_ND_X32, 6D) {
  std::vector<size_t> perm{0,1,2,3,4,5};
  do {
    TransposeOperatorTester()
        .num_dims(6)
        .shape({2,3,5,7,11,13})
        .perm(perm)
        .TestX32();
  } while (std::next_permutation(perm.begin(), perm.end()));
}
