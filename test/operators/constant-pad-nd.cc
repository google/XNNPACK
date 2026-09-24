// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include <gtest/gtest.h>
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

TEST(CONSTANT_PAD_ND_X32, output_size_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  xnn_operator_t constant_pad_op = nullptr;
  const uint32_t padding_value = 0;
  ASSERT_EQ(xnn_status_success,
            xnn_create_constant_pad_nd_x32(
                &padding_value, 0, &constant_pad_op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      constant_pad_op, xnn_delete_operator);

  const size_t input_shape[2] = {SIZE_MAX / 2, 3};
  const size_t pre_padding[2] = {0, 0};
  const size_t post_padding[2] = {0, 0};
  EXPECT_EQ(
      xnn_status_out_of_memory,
      xnn_reshape_constant_pad_nd_x32(
          constant_pad_op, 2, input_shape, pre_padding, post_padding,
          nullptr));
}

TEST(CONSTANT_PAD_ND_X32, stride_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  xnn_operator_t constant_pad_op = nullptr;
  const uint32_t padding_value = 0;
  ASSERT_EQ(xnn_status_success,
            xnn_create_constant_pad_nd_x32(
                &padding_value, 0, &constant_pad_op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      constant_pad_op, xnn_delete_operator);

  const size_t input_shape[2] = {1, (SIZE_MAX / 4) + 1};
  const size_t pre_padding[2] = {0, 0};
  const size_t post_padding[2] = {0, 0};
  EXPECT_EQ(
      xnn_status_out_of_memory,
      xnn_reshape_constant_pad_nd_x32(
          constant_pad_op, 2, input_shape, pre_padding, post_padding,
          nullptr));
}

TEST(CONSTANT_PAD_ND_X8, null_operator_out) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const uint8_t padding_value = 0;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_create_constant_pad_nd_x8(&padding_value, 0, nullptr));
}

TEST(CONSTANT_PAD_ND_X8, null_padding_value) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_operator_t op = nullptr;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_create_constant_pad_nd_x8(nullptr, 0, &op));
  EXPECT_EQ(nullptr, op);

  const size_t shape[1] = {1};
  const size_t pre[1] = {0};
  const size_t post[1] = {0};
  uint8_t in = 0, out = 0;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_run_constant_pad_nd_x8(0, 1, shape, pre, post, &in, &out,
                                       nullptr, nullptr));
}

TEST(CONSTANT_PAD_ND_X16, null_operator_out) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const uint16_t padding_value = 0;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_create_constant_pad_nd_x16(&padding_value, 0, nullptr));
}

TEST(CONSTANT_PAD_ND_X16, null_padding_value) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_operator_t op = nullptr;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_create_constant_pad_nd_x16(nullptr, 0, &op));
  EXPECT_EQ(nullptr, op);

  const size_t shape[1] = {1};
  const size_t pre[1] = {0};
  const size_t post[1] = {0};
  uint16_t in = 0, out = 0;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_run_constant_pad_nd_x16(0, 1, shape, pre, post, &in, &out,
                                        nullptr, nullptr));
}

TEST(CONSTANT_PAD_ND_X32, null_operator_out) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const uint32_t padding_value = 0;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_create_constant_pad_nd_x32(&padding_value, 0, nullptr));
}

TEST(CONSTANT_PAD_ND_X32, null_padding_value) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_operator_t op = nullptr;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_create_constant_pad_nd_x32(nullptr, 0, &op));
  EXPECT_EQ(nullptr, op);

  const size_t shape[1] = {1};
  const size_t pre[1] = {0};
  const size_t post[1] = {0};
  uint32_t in = 0, out = 0;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_run_constant_pad_nd_x32(0, 1, shape, pre, post, &in, &out,
                                        nullptr, nullptr));
}

TEST(CONSTANT_PAD_ND_X8, null_operator) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const size_t shape[1] = {1};
  const size_t pre[1] = {0};
  const size_t post[1] = {0};
  uint8_t in = 0, out = 0;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_reshape_constant_pad_nd_x8(nullptr, 1, shape, pre, post,
                                           nullptr));
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_setup_constant_pad_nd_x8(nullptr, &in, &out));
}

TEST(CONSTANT_PAD_ND_X8, null_parameters_in_reshape) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const uint8_t padding_value = 0;
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_constant_pad_nd_x8(&padding_value, 0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  const size_t shape[1] = {1};
  const size_t pre[1] = {0};
  const size_t post[1] = {0};
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_reshape_constant_pad_nd_x8(op, 1, nullptr, pre, post, nullptr));
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_reshape_constant_pad_nd_x8(op, 1, shape, nullptr, post,
                                           nullptr));
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_reshape_constant_pad_nd_x8(op, 1, shape, pre, nullptr,
                                           nullptr));
}

TEST(CONSTANT_PAD_ND_X8, null_pointers_in_setup) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const uint8_t padding_value = 0;
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_constant_pad_nd_x8(&padding_value, 0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  const size_t shape[1] = {1};
  const size_t pre[1] = {0};
  const size_t post[1] = {0};
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_constant_pad_nd_x8(op, 1, shape, pre, post, nullptr));

  uint8_t buffer = 0;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_setup_constant_pad_nd_x8(op, nullptr, &buffer));
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_setup_constant_pad_nd_x8(op, &buffer, nullptr));
}

TEST(CONSTANT_PAD_ND_X8, unsupported_num_dims) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const uint8_t padding_value = 0;
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_constant_pad_nd_x8(&padding_value, 0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  const size_t shape[XNN_MAX_TENSOR_DIMS + 1] = {1};
  const size_t pre[XNN_MAX_TENSOR_DIMS + 1] = {0};
  const size_t post[XNN_MAX_TENSOR_DIMS + 1] = {0};
  EXPECT_EQ(xnn_status_unsupported_parameter,
            xnn_reshape_constant_pad_nd_x8(
                op, XNN_MAX_TENSOR_DIMS + 1, shape, pre, post, nullptr));
}

TEST(CONSTANT_PAD_ND_X8, padded_dim_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const uint8_t padding_value = 0;
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_constant_pad_nd_x8(&padding_value, 0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  const size_t shape[1] = {SIZE_MAX};
  const size_t pre[1] = {1};
  const size_t post[1] = {0};
  const enum xnn_status status1 =
      xnn_reshape_constant_pad_nd_x8(op, 1, shape, pre, post, nullptr);
  EXPECT_TRUE(status1 == xnn_status_out_of_memory ||
              status1 == xnn_status_unsupported_parameter);

  const size_t shape2[1] = {1};
  const size_t pre2[1] = {SIZE_MAX};
  const size_t post2[1] = {1};
  const enum xnn_status status2 =
      xnn_reshape_constant_pad_nd_x8(op, 1, shape2, pre2, post2, nullptr);
  EXPECT_TRUE(status2 == xnn_status_out_of_memory ||
              status2 == xnn_status_unsupported_parameter);
}

TEST(CONSTANT_PAD_ND_X8, squeezed_dim_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const uint8_t padding_value = 0;
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_constant_pad_nd_x8(&padding_value, 0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  const size_t shape[2] = {SIZE_MAX / 2 + 1, 2};
  const size_t pre[2] = {0, 0};
  const size_t post[2] = {0, 0};
  const enum xnn_status status =
      xnn_reshape_constant_pad_nd_x8(op, 2, shape, pre, post, nullptr);
  EXPECT_TRUE(status == xnn_status_out_of_memory ||
              status == xnn_status_unsupported_parameter);
}

TEST(CONSTANT_PAD_ND_X8, output_size_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const uint8_t padding_value = 0;
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_constant_pad_nd_x8(&padding_value, 0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  const size_t shape[2] = {SIZE_MAX / 2 + 1, 3};
  const size_t pre[2] = {1, 0};
  const size_t post[2] = {0, 0};
  const enum xnn_status status =
      xnn_reshape_constant_pad_nd_x8(op, 2, shape, pre, post, nullptr);
  EXPECT_TRUE(status == xnn_status_out_of_memory ||
              status == xnn_status_unsupported_parameter);
}
