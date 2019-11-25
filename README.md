# XNNPACK

XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 (SSE2 level) platforms. XNNPACK is not intended for direct use by deep learning practitioners and researchers; instead it provides low-level performance primitives for accelerating high-level machine learning frameworks, such as [MediaPipe](https://mediapipe.dev), [TensorFlow Lite](https://www.tensorflow.org/lite), and [TensorFlow.js](https://www.tensorflow.org/js).

## Supported Architectures

- ARM64 on Android and Linux
- ARMv7 (with NEON) on Android and Linux
- WebAssembly MVP
- WebAssembly SIMD (experimental)
- x86 and x86-64 (up to AVX2) on Android, Linux, and macOS

## Operator Coverage

XNNPACK implements the following neural network operators:

- 2D Convolution (including grouped and depthwise)
- 2D Deconvolution (AKA Transposed Convolution)
- 2D Average Pooling
- 2D Max Pooling
- 2D ArgMax Pooling (Max Pooling + indices)
- 2D Unpooling
- 2D Bilinear Resize
- Add (tensors of same shape)
- Multiply (including broadcasting)
- Global Average Pooling
- Channel Shuffle
- Fully Connected
- Clamp (includes ReLU and ReLU6)
- HardSwish
- PReLU

All operators in XNNPACK support NHWC layout, but additionally allow custom stride along the **C**hannel dimension. Thus, operators can consume a subset of channels in the input tensor, and produce a subset of channels in the output tensor, providing a zero-cost Channel Split and Channel Concatenation operations.

## Performance

The table below presents single-threaded performance of XNNPACK library on two generations of MobileNet models and three generations of Pixel phones.

| Model              | Pixel, ms | Pixel 2, ms | Pixel 3a, ms |
| ------------------ | :-------: | :---------: | :----------: |
| MobileNet v1 1.0X  |    81     |      93     |      88      |
| MobileNet v2 1.0X  |    48     |      58     |      54      |

Benchmarked on October 9, 2019 with `end2end_bench --benchmark_min_time=5` on an Android/ARM64 build (`bazel build -c opt --config android_arm64 :end2end_bench`) and neural network models with randomized weights and inputs.

## Publications

- Marat Dukhan "The Indirect Convolution Algorithm". Presented on [Efficient Deep Learning for Compute Vision (ECV) 2019](https://sites.google.com/corp/view/ecv2019/) workshop ([slides](https://drive.google.com/file/d/1ZayB3By5ZxxQIRtN7UDq_JvPg1IYd3Ac/view), [paper on ArXiv](https://arxiv.org/abs/1907.02129)).
- Erich Elsen, Marat Dukhan, Trevor Gale, Karen Simonyan "Fast Sparse ConvNets".
  [Paper on ArXiv](https://arxiv.org/abs/1911.09723), [pre-trained sparse
  models](https://github.com/google-research/google-research/tree/master/fastconvnets).

## Acknowledgements

XNNPACK is a based on [QNNPACK](https://github.com/pytorch/QNNPACK) library. Unlike QNNPACK, XNNPACK focuses entirely on floating-point operators, and its API is no longer compatible with QNNPACK.
