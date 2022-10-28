// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "resize-bilinear-operator-tester.h"

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_centers_upscale_y) {
  for (size_t input_height = 2; input_height <= 3; input_height++) {
    for (size_t output_height = input_height + 1; output_height < 15; output_height *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_centers_upscale_x) {
  for (size_t input_width = 2; input_width <= 3; input_width++) {
    for (size_t output_width = input_width + 1; output_width < 15; output_width *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_centers_upscale) {
  for (size_t output_height = 3; output_height <= 5; output_height += 2) {
    for (size_t output_width = 3; output_width <= 5; output_width += 2) {
      ResizeBilinearOperatorTester()
        .input_size(2, 2)
        .output_size(output_height, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_centers_downscale_y) {
  for (size_t output_height = 1; output_height <= 3; output_height++) {
    for (size_t input_height = output_height + 1; input_height < 15; input_height *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_centers_downscale_x) {
  for (size_t output_width = 2; output_width <= 3; output_width++) {
    for (size_t input_width = output_width + 1; input_width < 15; input_width *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_centers_downscale) {
  for (size_t input_height = 3; input_height <= 5; input_height += 2) {
    for (size_t input_width = 3; input_width <= 5; input_width += 2) {
      ResizeBilinearOperatorTester()
        .input_size(input_height, input_width)
        .output_size(2, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_centers_identical_size) {
  for (size_t height = 2; height < 10; height *= 3) {
    for (size_t width = 2; width < 10; width *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(height, width)
        .output_size(height, width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_centers_varying_channels) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .input_size(input_size, input_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_centers_with_input_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .input_size(input_size, input_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .input_pixel_stride(23)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_centers_with_output_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .input_size(input_size, input_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .output_pixel_stride(29)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_centers_varying_batch_size) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
        ResizeBilinearOperatorTester()
          .batch_size(batch_size)
          .input_size(input_size, input_size)
          .output_size(output_size, output_size)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_corners_upscale_y) {
  for (size_t input_height = 2; input_height <= 3; input_height++) {
    for (size_t output_height = input_height + 1; output_height < 15; output_height *= 3) {
      ResizeBilinearOperatorTester()
        .align_corners(true)
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_corners_upscale_x) {
  for (size_t input_width = 2; input_width <= 3; input_width++) {
    for (size_t output_width = input_width + 1; output_width < 15; output_width *= 3) {
      ResizeBilinearOperatorTester()
        .align_corners(true)
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_corners_upscale) {
  for (size_t output_height = 3; output_height <= 5; output_height += 2) {
    for (size_t output_width = 3; output_width <= 5; output_width += 2) {
      ResizeBilinearOperatorTester()
        .align_corners(true)
        .input_size(2, 2)
        .output_size(output_height, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_corners_downscale_y) {
  for (size_t output_height = 2; output_height <= 3; output_height++) {
    for (size_t input_height = output_height + 1; input_height < 15; input_height *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_corners_downscale_x) {
  for (size_t output_width = 2; output_width <= 3; output_width++) {
    for (size_t input_width = output_width + 1; input_width < 15; input_width *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_corners_downscale) {
  for (size_t input_height = 3; input_height <= 5; input_height += 2) {
    for (size_t input_width = 3; input_width <= 5; input_width += 2) {
      ResizeBilinearOperatorTester()
        .input_size(input_height, input_width)
        .output_size(2, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_corners_identical_size) {
  for (size_t height = 2; height < 10; height *= 3) {
    for (size_t width = 2; width < 10; width *= 3) {
      ResizeBilinearOperatorTester()
        .align_corners(true)
        .input_size(height, width)
        .output_size(height, width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_corners_varying_channels) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .align_corners(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_corners_with_input_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .align_corners(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .input_pixel_stride(23)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_corners_with_output_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .align_corners(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .output_pixel_stride(29)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, aligned_corners_varying_batch_size) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
        ResizeBilinearOperatorTester()
          .align_corners(true)
          .batch_size(batch_size)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, tf_mode_aligned_centers_upscale_y) {
  for (size_t input_height = 2; input_height <= 3; input_height++) {
    for (size_t output_height = input_height + 1; output_height < 15; output_height *= 3) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, tf_mode_aligned_centers_upscale_x) {
  for (size_t input_width = 2; input_width <= 3; input_width++) {
    for (size_t output_width = input_width + 1; output_width < 15; output_width *= 3) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, tf_mode_aligned_centers_upscale) {
  for (size_t output_height = 3; output_height <= 5; output_height += 2) {
    for (size_t output_width = 3; output_width <= 5; output_width += 2) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(2, 2)
        .output_size(output_height, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, tf_mode_aligned_centers_downscale_y) {
  for (size_t output_height = 1; output_height <= 3; output_height++) {
    for (size_t input_height = output_height + 1; input_height < 15; input_height *= 3) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, tf_mode_aligned_centers_downscale_x) {
  for (size_t output_width = 1; output_width <= 3; output_width++) {
    for (size_t input_width = output_width + 1; input_width < 15; input_width *= 3) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, tf_mode_aligned_centers_downscale) {
  for (size_t input_height = 3; input_height <= 5; input_height += 2) {
    for (size_t input_width = 3; input_width <= 5; input_width += 2) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(input_height, input_width)
        .output_size(2, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, tf_mode_aligned_centers_identical_size) {
  for (size_t height = 2; height < 10; height *= 3) {
    for (size_t width = 2; width < 10; width *= 3) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(height, width)
        .output_size(height, width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF16();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, tf_mode_aligned_centers_varying_channels) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .tf_legacy_mode(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, tf_mode_aligned_centers_with_input_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .tf_legacy_mode(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .input_pixel_stride(23)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, tf_mode_aligned_centers_with_output_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .tf_legacy_mode(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .output_pixel_stride(29)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F16, tf_mode_aligned_centers_varying_batch_size) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
        ResizeBilinearOperatorTester()
          .tf_legacy_mode(true)
          .batch_size(batch_size)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .iterations(3)
          .TestNCHWxF16();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_centers_upscale_y) {
  for (size_t input_height = 2; input_height <= 3; input_height++) {
    for (size_t output_height = input_height + 1; output_height < 15; output_height *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_centers_upscale_x) {
  for (size_t input_width = 2; input_width <= 3; input_width++) {
    for (size_t output_width = input_width + 1; output_width < 15; output_width *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_centers_upscale) {
  for (size_t output_height = 3; output_height <= 5; output_height += 2) {
    for (size_t output_width = 3; output_width <= 5; output_width += 2) {
      ResizeBilinearOperatorTester()
        .input_size(2, 2)
        .output_size(output_height, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_centers_downscale_y) {
  for (size_t output_height = 1; output_height <= 3; output_height++) {
    for (size_t input_height = output_height + 1; input_height < 15; input_height *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_centers_downscale_x) {
  for (size_t output_width = 2; output_width <= 3; output_width++) {
    for (size_t input_width = output_width + 1; input_width < 15; input_width *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_centers_downscale) {
  for (size_t input_height = 3; input_height <= 5; input_height += 2) {
    for (size_t input_width = 3; input_width <= 5; input_width += 2) {
      ResizeBilinearOperatorTester()
        .input_size(input_height, input_width)
        .output_size(2, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_centers_identical_size) {
  for (size_t height = 2; height < 10; height *= 3) {
    for (size_t width = 2; width < 10; width *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(height, width)
        .output_size(height, width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_centers_varying_channels) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .input_size(input_size, input_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_centers_with_input_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .input_size(input_size, input_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .input_pixel_stride(23)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_centers_with_output_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .input_size(input_size, input_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .output_pixel_stride(29)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_centers_varying_batch_size) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
        ResizeBilinearOperatorTester()
          .batch_size(batch_size)
          .input_size(input_size, input_size)
          .output_size(output_size, output_size)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_corners_upscale_y) {
  for (size_t input_height = 2; input_height <= 3; input_height++) {
    for (size_t output_height = input_height + 1; output_height < 15; output_height *= 3) {
      ResizeBilinearOperatorTester()
        .align_corners(true)
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_corners_upscale_x) {
  for (size_t input_width = 2; input_width <= 3; input_width++) {
    for (size_t output_width = input_width + 1; output_width < 15; output_width *= 3) {
      ResizeBilinearOperatorTester()
        .align_corners(true)
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_corners_upscale) {
  for (size_t output_height = 3; output_height <= 5; output_height += 2) {
    for (size_t output_width = 3; output_width <= 5; output_width += 2) {
      ResizeBilinearOperatorTester()
        .align_corners(true)
        .input_size(2, 2)
        .output_size(output_height, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_corners_downscale_y) {
  for (size_t output_height = 2; output_height <= 3; output_height++) {
    for (size_t input_height = output_height + 1; input_height < 15; input_height *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_corners_downscale_x) {
  for (size_t output_width = 2; output_width <= 3; output_width++) {
    for (size_t input_width = output_width + 1; input_width < 15; input_width *= 3) {
      ResizeBilinearOperatorTester()
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_corners_downscale) {
  for (size_t input_height = 3; input_height <= 5; input_height += 2) {
    for (size_t input_width = 3; input_width <= 5; input_width += 2) {
      ResizeBilinearOperatorTester()
        .input_size(input_height, input_width)
        .output_size(2, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_corners_identical_size) {
  for (size_t height = 2; height < 10; height *= 3) {
    for (size_t width = 2; width < 10; width *= 3) {
      ResizeBilinearOperatorTester()
        .align_corners(true)
        .input_size(height, width)
        .output_size(height, width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_corners_varying_channels) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .align_corners(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_corners_with_input_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .align_corners(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .input_pixel_stride(23)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_corners_with_output_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .align_corners(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .output_pixel_stride(29)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, aligned_corners_varying_batch_size) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
        ResizeBilinearOperatorTester()
          .align_corners(true)
          .batch_size(batch_size)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, tf_mode_aligned_centers_upscale_y) {
  for (size_t input_height = 2; input_height <= 3; input_height++) {
    for (size_t output_height = input_height + 1; output_height < 15; output_height *= 3) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, tf_mode_aligned_centers_upscale_x) {
  for (size_t input_width = 2; input_width <= 3; input_width++) {
    for (size_t output_width = input_width + 1; output_width < 15; output_width *= 3) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, tf_mode_aligned_centers_upscale) {
  for (size_t output_height = 3; output_height <= 5; output_height += 2) {
    for (size_t output_width = 3; output_width <= 5; output_width += 2) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(2, 2)
        .output_size(output_height, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, tf_mode_aligned_centers_downscale_y) {
  for (size_t output_height = 1; output_height <= 3; output_height++) {
    for (size_t input_height = output_height + 1; input_height < 15; input_height *= 3) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(input_height, 2)
        .output_size(output_height, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, tf_mode_aligned_centers_downscale_x) {
  for (size_t output_width = 1; output_width <= 3; output_width++) {
    for (size_t input_width = output_width + 1; input_width < 15; input_width *= 3) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(2, input_width)
        .output_size(2, output_width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, tf_mode_aligned_centers_downscale) {
  for (size_t input_height = 3; input_height <= 5; input_height += 2) {
    for (size_t input_width = 3; input_width <= 5; input_width += 2) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(input_height, input_width)
        .output_size(2, 2)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, tf_mode_aligned_centers_identical_size) {
  for (size_t height = 2; height < 10; height *= 3) {
    for (size_t width = 2; width < 10; width *= 3) {
      ResizeBilinearOperatorTester()
        .tf_legacy_mode(true)
        .input_size(height, width)
        .output_size(height, width)
        .channels(17)
        .iterations(3)
        .TestNCHWxF32();
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, tf_mode_aligned_centers_varying_channels) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .tf_legacy_mode(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, tf_mode_aligned_centers_with_input_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .tf_legacy_mode(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .input_pixel_stride(23)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, tf_mode_aligned_centers_with_output_stride) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t channels = 15; channels <= 19; channels++) {
        ResizeBilinearOperatorTester()
          .tf_legacy_mode(true)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .channels(channels)
          .output_pixel_stride(29)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}

TEST(RESIZE_BILINEAR_NCHW_F32, tf_mode_aligned_centers_varying_batch_size) {
  for (size_t input_size = 2; input_size <= 6; input_size += 2) {
    for (size_t output_size = 2; output_size <= 6; output_size += 2) {
      for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
        ResizeBilinearOperatorTester()
          .tf_legacy_mode(true)
          .batch_size(batch_size)
          .input_size(output_size, output_size)
          .output_size(output_size, output_size)
          .iterations(3)
          .TestNCHWxF32();
      }
    }
  }
}
