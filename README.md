# XNNPACK

XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms. XNNPACK is not intended for direct use by deep learning practitioners and researchers; instead it provides low-level performance primitives for accelerating high-level machine learning frameworks, such as [TensorFlow Lite](https://www.tensorflow.org/lite), [TensorFlow.js](https://www.tensorflow.org/js), [PyTorch](https://pytorch.org/), and [MediaPipe](https://mediapipe.dev).

## Supported Architectures

- ARM64 on Android, Linux, and iOS (including WatchOS and tvOS)
- ARMv7 (with NEON) on Android, Linux, and iOS (including WatchOS)
- WebAssembly MVP
- WebAssembly SIMD (experimental)
- x86 and x86-64 (up to AVX512) on Android, Linux, macOS, and iOS simulator

## Operator Coverage

XNNPACK implements the following neural network operators:

- 2D Convolution (including grouped and depthwise)
- 2D Deconvolution (AKA Transposed Convolution)
- 2D Average Pooling
- 2D Max Pooling
- 2D ArgMax Pooling (Max Pooling + indices)
- 2D Unpooling
- 2D Bilinear Resize
- Add (including broadcasting, two inputs only)
- Subtract (including broadcasting)
- Divide (including broadcasting)
- Maximum (including broadcasting)
- Minimum (including broadcasting)
- Multiply (including broadcasting)
- Global Average Pooling
- Channel Shuffle
- Fully Connected
- Clamp (includes ReLU and ReLU6)
- HardSwish
- Sigmoid
- Softmax
- PReLU

All operators in XNNPACK support NHWC layout, but additionally allow custom stride along the **C**hannel dimension. Thus, operators can consume a subset of channels in the input tensor, and produce a subset of channels in the output tensor, providing a zero-cost Channel Split and Channel Concatenation operations.

## Performance

### Mobile phones

The table below presents **single-threaded** performance of XNNPACK library on three generations of MobileNet models and three generations of Pixel phones.

| Model              | Pixel, ms | Pixel 2, ms | Pixel 3a, ms |
| ------------------ | :-------: | :---------: | :----------: |
| MobileNet v1 1.0X  |    82     |      86     |      88      |
| MobileNet v2 1.0X  |    49     |      53     |      55      |
| MobileNet v3 Large |    39     |      42     |      44      |
| MobileNet v3 Small |    12     |      14     |      14      |

The following table presents **multi-threaded** (using as many threads as there are big cores) performance of XNNPACK library on three generations of MobileNet models and three generations of Pixel phones.

| Model              | Pixel, ms | Pixel 2, ms | Pixel 3a, ms |
| ------------------ | :-------: | :---------: | :----------: |
| MobileNet v1 1.0X  |    43     |      27     |      46      |
| MobileNet v2 1.0X  |    26     |      18     |      28      |
| MobileNet v3 Large |    22     |      16     |      24      |
| MobileNet v3 Small |     7     |       6     |       8      |

Benchmarked on March 27, 2020 with `end2end_bench --benchmark_min_time=5` on an Android/ARM64 build with Android NDK r21 (`bazel build -c opt --config android_arm64 :end2end_bench`) and neural network models with randomized weights and inputs.

### Raspberry Pi

The table below presents **multi-threaded** performance of XNNPACK library on three generations of MobileNet models and three generations of Raspberry Pi boards.

| Model              | RPi 2 (BCM2836), ms | RPi 3+ (BCM2837B0), ms | RPi 4 (BCM2711), ms |
| ------------------ | :-----------------: | :--------------------: | :-----------------: |
| MobileNet v1 1.0X  |         341         |          115           |          75         |
| MobileNet v2 1.0X  |         197         |           79           |          44         |
| MobileNet v3 Large |         165         |           67           |          41         |
| MobileNet v3 Small |          53         |           23           |          14         |

Benchmarked on February 12, 2020 with `end2end-bench --benchmark_min_time=5` on a Raspbian Buster build with CMake (`./scripts/build-local.sh`) and neural network models with randomized weights and inputs.

## Publications

- Marat Dukhan "The Indirect Convolution Algorithm". Presented on [Efficient Deep Learning for Compute Vision (ECV) 2019](https://sites.google.com/corp/view/ecv2019/) workshop ([slides](https://drive.google.com/file/d/1ZayB3By5ZxxQIRtN7UDq_JvPg1IYd3Ac/view), [paper on ArXiv](https://arxiv.org/abs/1907.02129)).
- Erich Elsen, Marat Dukhan, Trevor Gale, Karen Simonyan "Fast Sparse ConvNets".
  [Paper on ArXiv](https://arxiv.org/abs/1911.09723), [pre-trained sparse
  models](https://github.com/google-research/google-research/tree/master/fastconvnets).
- Marat Dukhan, Artsiom Ablavatski "The Two-Pass Softmax Algorithm".
  [Paper on ArXiv](https://arxiv.org/abs/2001.04438).

## Ecosystem

### Machine Learning Frameworks

- [TensorFlow.js WebAssembly backend](https://blog.tensorflow.org/2020/03/introducing-webassembly-backend-for-tensorflow-js.html).
- [MediaPipe for Web](https://developers.googleblog.com/2020/01/mediapipe-on-web.html).
- TensorFlow Lite through the [XNNPACK delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/xnnpack).
- [PyTorch](https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/xnnpack).

## Acknowledgements

XNNPACK is a based on [QNNPACK](https://github.com/pytorch/QNNPACK) library. Unlike QNNPACK, XNNPACK focuses entirely on floating-point operators, and its API is no longer compatible with QNNPACK.
