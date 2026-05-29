// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <climits>
#include <cstddef>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "test/operators/constant-pad-operator-tester.h"

constexpr size_t kDim1 = 2;
constexpr size_t kDim2 = 3;
constexpr size_t kDim3 = 2;
constexpr size_t kDim4 = 3;
constexpr size_t kDim5 = 2;
constexpr size_t kDim6 = 3;
constexpr size_t kDim1PrePad = kDim1 / 2;
constexpr size_t kDim1PostPad = kDim1 / 2 + 1;
constexpr size_t kDim2PrePad = kDim2 / 2;
constexpr size_t kDim2PostPad = kDim2 / 2 + 1;
constexpr size_t kDim3PrePad = kDim3 / 2;
constexpr size_t kDim3PostPad = kDim3 / 2 + 1;
constexpr size_t kDim4PrePad = kDim4 / 2;
constexpr size_t kDim4PostPad = kDim4 / 2 + 1;
constexpr size_t kDim5PrePad = kDim5 / 2;
constexpr size_t kDim5PostPad = kDim5 / 2 + 1;
constexpr size_t kDim6PrePad = kDim6 / 2;
constexpr size_t kDim6PostPad = kDim6 / 2 + 1;

TEST(CONSTANT_PAD_ND_X8, constant_pad_0d) {
  ConstantPadOperatorTester().TestX8();
}

TEST(CONSTANT_PAD_ND_X8, constant_pad_1d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      ConstantPadOperatorTester()
          .input_shape({kDim1})
          .pre_paddings({dim1_pre_pad})
          .post_paddings({dim1_post_pad})
          .TestX8();
    }
  }
}

TEST(CONSTANT_PAD_ND_X8, constant_pad_2d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          ConstantPadOperatorTester()
              .input_shape({kDim1, kDim2})
              .pre_paddings({dim1_pre_pad, dim2_pre_pad})
              .post_paddings({dim1_post_pad, dim2_post_pad})
              .TestX8();
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X8, constant_pad_3d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              ConstantPadOperatorTester()
                  .input_shape({kDim1, kDim2, kDim3})
                  .pre_paddings({dim1_pre_pad, dim2_pre_pad, dim3_pre_pad})
                  .post_paddings({dim1_post_pad, dim2_post_pad, dim3_post_pad})
                  .TestX8();
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X8, constant_pad_4d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              for (size_t dim4_pre_pad = 0; dim4_pre_pad <= kDim4PrePad;
                   dim4_pre_pad += kDim4PrePad) {
                for (size_t dim4_post_pad = 0; dim4_post_pad <= kDim4PostPad;
                     dim4_post_pad += kDim4PostPad) {
                  ConstantPadOperatorTester()
                      .input_shape({kDim1, kDim2, kDim3, kDim4})
                      .pre_paddings({dim1_pre_pad, dim2_pre_pad, dim3_pre_pad,
                                     dim4_pre_pad})
                      .post_paddings({dim1_post_pad, dim2_post_pad,
                                      dim3_post_pad, dim4_post_pad})
                      .TestX8();
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X8, constant_pad_5d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              for (size_t dim4_pre_pad = 0; dim4_pre_pad <= kDim4PrePad;
                   dim4_pre_pad += kDim4PrePad) {
                for (size_t dim4_post_pad = 0; dim4_post_pad <= kDim4PostPad;
                     dim4_post_pad += kDim4PostPad) {
                  for (size_t dim5_pre_pad = 0; dim5_pre_pad <= kDim5PrePad;
                       dim5_pre_pad += kDim5PrePad) {
                    for (size_t dim5_post_pad = 0;
                         dim5_post_pad <= kDim5PostPad;
                         dim5_post_pad += kDim5PostPad) {
                      ConstantPadOperatorTester()
                          .input_shape({kDim1, kDim2, kDim3, kDim4, kDim5})
                          .pre_paddings({dim1_pre_pad, dim2_pre_pad,
                                         dim3_pre_pad, dim4_pre_pad,
                                         dim5_pre_pad})
                          .post_paddings({dim1_post_pad, dim2_post_pad,
                                          dim3_post_pad, dim4_post_pad,
                                          dim5_post_pad})
                          .TestX8();
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X8, constant_pad_6d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              for (size_t dim4_pre_pad = 0; dim4_pre_pad <= kDim4PrePad;
                   dim4_pre_pad += kDim4PrePad) {
                for (size_t dim4_post_pad = 0; dim4_post_pad <= kDim4PostPad;
                     dim4_post_pad += kDim4PostPad) {
                  for (size_t dim5_pre_pad = 0; dim5_pre_pad <= kDim5PrePad;
                       dim5_pre_pad += kDim5PrePad) {
                    for (size_t dim5_post_pad = 0;
                         dim5_post_pad <= kDim5PostPad;
                         dim5_post_pad += kDim5PostPad) {
                      for (size_t dim6_pre_pad = 0; dim6_pre_pad <= kDim6PrePad;
                           dim6_pre_pad += kDim6PrePad) {
                        for (size_t dim6_post_pad = 0;
                             dim6_post_pad <= kDim6PostPad;
                             dim6_post_pad += kDim6PostPad) {
                          ConstantPadOperatorTester()
                              .input_shape(
                                  {kDim1, kDim2, kDim3, kDim4, kDim5, kDim6})
                              .pre_paddings({dim1_pre_pad, dim2_pre_pad,
                                             dim3_pre_pad, dim4_pre_pad,
                                             dim5_pre_pad, dim6_pre_pad})
                              .post_paddings({dim1_post_pad, dim2_post_pad,
                                              dim3_post_pad, dim4_post_pad,
                                              dim5_post_pad, dim6_post_pad})
                              .TestX8();
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X16, constant_pad_0d) {
  ConstantPadOperatorTester().TestX16();
}

TEST(CONSTANT_PAD_ND_X16, constant_pad_1d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      ConstantPadOperatorTester()
          .input_shape({kDim1})
          .pre_paddings({dim1_pre_pad})
          .post_paddings({dim1_post_pad})
          .TestX16();
    }
  }
}

TEST(CONSTANT_PAD_ND_X16, constant_pad_2d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          ConstantPadOperatorTester()
              .input_shape({kDim1, kDim2})
              .pre_paddings({dim1_pre_pad, dim2_pre_pad})
              .post_paddings({dim1_post_pad, dim2_post_pad})
              .TestX16();
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X16, constant_pad_3d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              ConstantPadOperatorTester()
                  .input_shape({kDim1, kDim2, kDim3})
                  .pre_paddings({dim1_pre_pad, dim2_pre_pad, dim3_pre_pad})
                  .post_paddings({dim1_post_pad, dim2_post_pad, dim3_post_pad})
                  .TestX16();
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X16, constant_pad_4d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              for (size_t dim4_pre_pad = 0; dim4_pre_pad <= kDim4PrePad;
                   dim4_pre_pad += kDim4PrePad) {
                for (size_t dim4_post_pad = 0; dim4_post_pad <= kDim4PostPad;
                     dim4_post_pad += kDim4PostPad) {
                  ConstantPadOperatorTester()
                      .input_shape({kDim1, kDim2, kDim3, kDim4})
                      .pre_paddings({dim1_pre_pad, dim2_pre_pad, dim3_pre_pad,
                                     dim4_pre_pad})
                      .post_paddings({dim1_post_pad, dim2_post_pad,
                                      dim3_post_pad, dim4_post_pad})
                      .TestX16();
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X16, constant_pad_5d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              for (size_t dim4_pre_pad = 0; dim4_pre_pad <= kDim4PrePad;
                   dim4_pre_pad += kDim4PrePad) {
                for (size_t dim4_post_pad = 0; dim4_post_pad <= kDim4PostPad;
                     dim4_post_pad += kDim4PostPad) {
                  for (size_t dim5_pre_pad = 0; dim5_pre_pad <= kDim5PrePad;
                       dim5_pre_pad += kDim5PrePad) {
                    for (size_t dim5_post_pad = 0;
                         dim5_post_pad <= kDim5PostPad;
                         dim5_post_pad += kDim5PostPad) {
                      ConstantPadOperatorTester()
                          .input_shape({kDim1, kDim2, kDim3, kDim4, kDim5})
                          .pre_paddings({dim1_pre_pad, dim2_pre_pad,
                                         dim3_pre_pad, dim4_pre_pad,
                                         dim5_pre_pad})
                          .post_paddings({dim1_post_pad, dim2_post_pad,
                                          dim3_post_pad, dim4_post_pad,
                                          dim5_post_pad})
                          .TestX16();
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X16, constant_pad_6d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              for (size_t dim4_pre_pad = 0; dim4_pre_pad <= kDim4PrePad;
                   dim4_pre_pad += kDim4PrePad) {
                for (size_t dim4_post_pad = 0; dim4_post_pad <= kDim4PostPad;
                     dim4_post_pad += kDim4PostPad) {
                  for (size_t dim5_pre_pad = 0; dim5_pre_pad <= kDim5PrePad;
                       dim5_pre_pad += kDim5PrePad) {
                    for (size_t dim5_post_pad = 0;
                         dim5_post_pad <= kDim5PostPad;
                         dim5_post_pad += kDim5PostPad) {
                      for (size_t dim6_pre_pad = 0; dim6_pre_pad <= kDim6PrePad;
                           dim6_pre_pad += kDim6PrePad) {
                        for (size_t dim6_post_pad = 0;
                             dim6_post_pad <= kDim6PostPad;
                             dim6_post_pad += kDim6PostPad) {
                          ConstantPadOperatorTester()
                              .input_shape(
                                  {kDim1, kDim2, kDim3, kDim4, kDim5, kDim6})
                              .pre_paddings({dim1_pre_pad, dim2_pre_pad,
                                             dim3_pre_pad, dim4_pre_pad,
                                             dim5_pre_pad, dim6_pre_pad})
                              .post_paddings({dim1_post_pad, dim2_post_pad,
                                              dim3_post_pad, dim4_post_pad,
                                              dim5_post_pad, dim6_post_pad})
                              .TestX16();
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X32, constant_pad_0d) {
  ConstantPadOperatorTester().TestX32();
}

TEST(CONSTANT_PAD_ND_X32, constant_pad_1d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      ConstantPadOperatorTester()
          .input_shape({kDim1})
          .pre_paddings({dim1_pre_pad})
          .post_paddings({dim1_post_pad})
          .TestX32();
    }
  }
}

TEST(CONSTANT_PAD_ND_X32, constant_pad_2d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          ConstantPadOperatorTester()
              .input_shape({kDim1, kDim2})
              .pre_paddings({dim1_pre_pad, dim2_pre_pad})
              .post_paddings({dim1_post_pad, dim2_post_pad})
              .TestX32();
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X32, constant_pad_3d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              ConstantPadOperatorTester()
                  .input_shape({kDim1, kDim2, kDim3})
                  .pre_paddings({dim1_pre_pad, dim2_pre_pad, dim3_pre_pad})
                  .post_paddings({dim1_post_pad, dim2_post_pad, dim3_post_pad})
                  .TestX32();
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X32, constant_pad_4d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              for (size_t dim4_pre_pad = 0; dim4_pre_pad <= kDim4PrePad;
                   dim4_pre_pad += kDim4PrePad) {
                for (size_t dim4_post_pad = 0; dim4_post_pad <= kDim4PostPad;
                     dim4_post_pad += kDim4PostPad) {
                  ConstantPadOperatorTester()
                      .input_shape({kDim1, kDim2, kDim3, kDim4})
                      .pre_paddings({dim1_pre_pad, dim2_pre_pad, dim3_pre_pad,
                                     dim4_pre_pad})
                      .post_paddings({dim1_post_pad, dim2_post_pad,
                                      dim3_post_pad, dim4_post_pad})
                      .TestX32();
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X32, constant_pad_5d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              for (size_t dim4_pre_pad = 0; dim4_pre_pad <= kDim4PrePad;
                   dim4_pre_pad += kDim4PrePad) {
                for (size_t dim4_post_pad = 0; dim4_post_pad <= kDim4PostPad;
                     dim4_post_pad += kDim4PostPad) {
                  for (size_t dim5_pre_pad = 0; dim5_pre_pad <= kDim5PrePad;
                       dim5_pre_pad += kDim5PrePad) {
                    for (size_t dim5_post_pad = 0;
                         dim5_post_pad <= kDim5PostPad;
                         dim5_post_pad += kDim5PostPad) {
                      ConstantPadOperatorTester()
                          .input_shape({kDim1, kDim2, kDim3, kDim4, kDim5})
                          .pre_paddings({dim1_pre_pad, dim2_pre_pad,
                                         dim3_pre_pad, dim4_pre_pad,
                                         dim5_pre_pad})
                          .post_paddings({dim1_post_pad, dim2_post_pad,
                                          dim3_post_pad, dim4_post_pad,
                                          dim5_post_pad})
                          .TestX32();
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X32, constant_pad_6d) {
  for (size_t dim1_pre_pad = 0; dim1_pre_pad <= kDim1PrePad;
       dim1_pre_pad += kDim1PrePad) {
    for (size_t dim1_post_pad = 0; dim1_post_pad <= kDim1PostPad;
         dim1_post_pad += kDim1PostPad) {
      for (size_t dim2_pre_pad = 0; dim2_pre_pad <= kDim2PrePad;
           dim2_pre_pad += kDim2PrePad) {
        for (size_t dim2_post_pad = 0; dim2_post_pad <= kDim2PostPad;
             dim2_post_pad += kDim2PostPad) {
          for (size_t dim3_pre_pad = 0; dim3_pre_pad <= kDim3PrePad;
               dim3_pre_pad += kDim3PrePad) {
            for (size_t dim3_post_pad = 0; dim3_post_pad <= kDim3PostPad;
                 dim3_post_pad += kDim3PostPad) {
              for (size_t dim4_pre_pad = 0; dim4_pre_pad <= kDim4PrePad;
                   dim4_pre_pad += kDim4PrePad) {
                for (size_t dim4_post_pad = 0; dim4_post_pad <= kDim4PostPad;
                     dim4_post_pad += kDim4PostPad) {
                  for (size_t dim5_pre_pad = 0; dim5_pre_pad <= kDim5PrePad;
                       dim5_pre_pad += kDim5PrePad) {
                    for (size_t dim5_post_pad = 0;
                         dim5_post_pad <= kDim5PostPad;
                         dim5_post_pad += kDim5PostPad) {
                      for (size_t dim6_pre_pad = 0; dim6_pre_pad <= kDim6PrePad;
                           dim6_pre_pad += kDim6PrePad) {
                        for (size_t dim6_post_pad = 0;
                             dim6_post_pad <= kDim6PostPad;
                             dim6_post_pad += kDim6PostPad) {
                          ConstantPadOperatorTester()
                              .input_shape(
                                  {kDim1, kDim2, kDim3, kDim4, kDim5, kDim6})
                              .pre_paddings({dim1_pre_pad, dim2_pre_pad,
                                             dim3_pre_pad, dim4_pre_pad,
                                             dim5_pre_pad, dim6_pre_pad})
                              .post_paddings({dim1_post_pad, dim2_post_pad,
                                              dim3_post_pad, dim4_post_pad,
                                              dim5_post_pad, dim6_post_pad})
                              .TestX32();
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(CONSTANT_PAD_ND_X32, reshape_output_dimension_overflow) {
  // pre_padding + input_dim would overflow size_t; verify that reshape returns
  // xnn_status_invalid_parameter rather than silently wrapping around.
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));

  xnn_operator_t op = nullptr;
  const uint32_t padding_value = 0;
  ASSERT_EQ(xnn_status_success,
            xnn_create_constant_pad_nd_x32(&padding_value, /*flags=*/0, &op));
  ASSERT_NE(nullptr, op);

  const size_t input_shape[]  = {2};
  const size_t pre_paddings[]  = {SIZE_MAX - 1};
  const size_t post_paddings[] = {0};

  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_reshape_constant_pad_nd_x32(op, /*num_dims=*/1, input_shape,
                                            pre_paddings, post_paddings,
                                            /*threadpool=*/nullptr));
  xnn_delete_operator(op);
}
