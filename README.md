# XNNPACK

XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms. XNNPACK is not intended for direct use by deep learning practitioners and researchers; instead it provides low-level performance primitives for accelerating high-level machine learning frameworks, such as [MediaPipe](https://mediapipe.dev), [TensorFlow Lite](https://www.tensorflow.org/lite), and [TensorFlow.js](https://www.tensorflow.org/js).

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
| MobileNet v1 1.0X  |    81     |      89     |      88      |
| MobileNet v2 1.0X  |    48     |      55     |      54      |
| MobileNet v3 Large |    40     |      44     |      44      |
| MobileNet v3 Small |    12     |      14     |      14      |

The following table presents **multi-threaded** (using as many threads as there are big cores) performance of XNNPACK library on three generations of MobileNet models and three generations of Pixel phones.

| Model              | Pixel, ms | Pixel 2, ms | Pixel 3a, ms |
| ------------------ | :-------: | :---------: | :----------: |
| MobileNet v1 1.0X  |    45     |      27     |      46      |
| MobileNet v2 1.0X  |    28     |      18     |      28      |
| MobileNet v3 Large |    23     |      16     |      24      |
| MobileNet v3 Small |     7     |       6     |       8      |

Benchmarked on January 9, 2020 with `end2end_bench --benchmark_min_time=5` on an Android/ARM64 build (`bazel build -c opt --config android_arm64 :end2end_bench`) and neural network models with randomized weights and inputs.

### Raspberry Pi

The table below presents **multi-threaded** performance of XNNPACK library on three generations of MobileNet models and three generations of Raspberry Pi boards.

| Model              | RPi 2 (BCM2836), ms | RPi 3+ (BCM2837B0), ms | RPi 4 (BCM2711), ms |
| ------------------ | :-----------------: | :--------------------: | :-----------------: |
| MobileNet v1 1.0X  |         380         |          115           |          76         |
| MobileNet v2 1.0X  |         217         |           80           |          45         |
| MobileNet v3 Large |         180         |           67           |          41         |
| MobileNet v3 Small |          57         |           23           |          15         |

Benchmarked on January 9, 2020 with `end2end-bench --benchmark_min_time=5` on a Raspbian Buster build with CMake (`./scripts/build-local.sh`) and neural network models with randomized weights and inputs.

## Publications

- Marat Dukhan "The Indirect Convolution Algorithm". Presented on [Efficient Deep Learning for Compute Vision (ECV) 2019](https://sites.google.com/corp/view/ecv2019/) workshop ([slides](https://drive.google.com/file/d/1ZayB3By5ZxxQIRtN7UDq_JvPg1IYd3Ac/view), [paper on ArXiv](https://arxiv.org/abs/1907.02129)).
- Erich Elsen, Marat Dukhan, Trevor Gale, Karen Simonyan "Fast Sparse ConvNets".
  [Paper on ArXiv](https://arxiv.org/abs/1911.09723), [pre-trained sparse
  models](https://github.com/google-research/google-research/tree/master/fastconvnets).
- Marat Dukhan, Artsiom Ablavatski "The Two-Pass Softmax Algorithm".
  [Paper on ArXiv](https://arxiv.org/abs/2001.04438).

## Acknowledgements

XNNPACK is a based on [QNNPACK](https://github.com/pytorch/QNNPACK) library. Unlike QNNPACK, XNNPACK focuses entirely on floating-point operators, and its API is no longer compatible with QNNPACK.
