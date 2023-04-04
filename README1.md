# XNNPACK

XNNPACK is a highly optimized solution for neural network inference on ARM, x86, WebAssembly, and RISC-V platforms. XNNPACK is not intended for direct use by deep learning practitioners and researchers; instead it provides low-level performance primitives for accelerating high-level machine learning frameworks, such as [TensorFlow Lite](https://www.tensorflow.org/lite), [TensorFlow.js](https://www.tensorflow.org/js), [PyTorch](https://pytorch.org/), [ONNX Runtime](https://onnxruntime.ai), and [MediaPipe](https://mediapipe.dev).

## Supported Architectures

- ARM64 on Android, iOS, macOS, Linux, and Windows
- ARMv7 (with NEON) on Android
- ARMv6 (with VFPv2) on Linux
- x86 and x86-64 (up to AVX512) on Windows, Linux, macOS, Android, and iOS simulator
- WebAssembly MVP
- WebAssembly SIMD
- [WebAssembly Relaxed SIMD](https://github.com/WebAssembly/relaxed-simd) (experimental)
- RISC-V (RV32GC and RV64GC)

## Operator Coverage

XNNPACK implements the following neural network operators:

- 2D Convolution (including grouped and depthwise)
- 2D Deconvolution (AKA Transposed Convolution)
- 2D Average Pooling
- 2D Max Pooling
- 2D ArgMax Pooling (Max Pooling + indices)
- 2D Unpooling
- 2D Bilinear Resize
- 2D Depth-to-Space (AKA Pixel Shuffle)
- Add (including broadcasting, two inputs only)
- Subtract (including broadcasting)
- Divide (including broadcasting)
- Maximum (including broadcasting)
- Minimum (including broadcasting)
- Multiply (including broadcasting)
- Squared Difference (including broadcasting)
- Global Average Pooling
- Channel Shuffle
- Fully Connected
- Abs (absolute value)
- Bankers' Rounding (rounding to nearest, ties to even)
- Ceiling (rounding to integer above)
- Clamp (includes ReLU and ReLU6)
- Convert (includes fixed-point and half-precision quantization and
  dequantization)
- Copy
- ELU
- Floor (rounding to integer below)
- HardSwish
- Leaky ReLU
- Negate
- Sigmoid
- Softmax
- Square
- Tanh
- Transpose
- Truncation (rounding to integer towards zero)
- PReLU

All operators in XNNPACK support NHWC layout, but additionally allow custom stride along the **C**hannel dimension. Thus, operators can consume a subset of channels in the input tensor, and produce a subset of channels in the output tensor, providing a zero-cost Channel Split and Channel Concatenation operations.

## Performance

### Mobile phones

The table below presents **single-threaded** performance of XNNPACK library on three generations of MobileNet models and three generations of Pixel phones.

| Model                   | Pixel, ms | Pixel 2, ms | Pixel 3a, ms |
| ----------------------- | :-------: | :---------: | :----------: |
| FP32 MobileNet v1 1.0X  |    82     |      86     |      88      |
| FP32 MobileNet v2 1.0X  |    49     |      53     |      55      |
| FP32 MobileNet v3 Large |    39     |      42     |      44      |
| FP32 MobileNet v3 Small |    12     |      14     |      14      |

The following table presents **multi-threaded** (using as many threads as there are big cores) performance of XNNPACK library on three generations of MobileNet models and three generations of Pixel phones.

| Model                   | Pixel, ms | Pixel 2, ms | Pixel 3a, ms |
| ----------------------- | :-------: | :---------: | :----------: |
| FP32 MobileNet v1 1.0X  |    43     |      27     |      46      |
| FP32 MobileNet v2 1.0X  |    26     |      18     |      28      |
| FP32 MobileNet v3 Large |    22     |      16     |      24      |
| FP32 MobileNet v3 Small |     7     |       6     |       8      |

Benchmarked on March 27, 2020 with `end2end_bench --benchmark_min_time=5` on an Android/ARM64 build with Android NDK r21 (`bazel build -c opt --config android_arm64 :end2end_bench`) and neural network models with randomized weights and inputs.

### Raspberry Pi

The table below presents **multi-threaded** performance of XNNPACK library on three generations of MobileNet models and three generations of Raspberry Pi boards.

| Model                   | RPi Zero W (BCM2835), ms | RPi 2 (BCM2836), ms | RPi 3+ (BCM2837B0), ms | RPi 4 (BCM2711), ms | RPi 4 (BCM2711, ARM64), ms |
| ----------------------- | :----------------------: | :-----------------: | :--------------------: | :-----------------: | :------------------------: |
| FP32 MobileNet v1 1.0X  |          3919            |         302         |          114           |          72         |             77             |
| FP32 MobileNet v2 1.0X  |          1987            |         191         |           79           |          41         |             46             |
| FP32 MobileNet v3 Large |          1658            |         161         |           67           |          38         |             40             |
| FP32 MobileNet v3 Small |           474            |          50         |           22           |          13         |             15             |
| INT8 MobileNet v1 1.0X  |          2589            |         128         |           46           |          29         |             24             |
| INT8 MobileNet v2 1.0X  |          1495            |          82         |           30           |          20         |             17             |

Benchmarked on Feb 8, 2022 with `end2end-bench --benchmark_min_time=5` on a Raspbian Buster build with CMake (`./scripts/build-local.sh`) and neural network models with randomized weights and inputs. INT8 inference was evaluated on per-channel quantization schema.

## Minimum build requirements

- C11
- C++14
- Python 3

## Publications

- Marat Dukhan "The Indirect Convolution Algorithm". Presented on [Efficient Deep Learning for Compute Vision (ECV) 2019](https://sites.google.com/corp/view/ecv2019/) workshop ([slides](https://drive.google.com/file/d/1ZayB3By5ZxxQIRtN7UDq_JvPg1IYd3Ac/view), [paper on ArXiv](https://arxiv.org/abs/1907.02129)).
- Erich Elsen, Marat Dukhan, Trevor Gale, Karen Simonyan "Fast Sparse ConvNets".
  [Paper on ArXiv](https://arxiv.org/abs/1911.09723), [pre-trained sparse
  models](https://github.com/google-research/google-research/tree/master/fastconvnets).
- Marat Dukhan, Artsiom Ablavatski "The Two-Pass Softmax Algorithm".
  [Paper on ArXiv](https://arxiv.org/abs/2001.04438).
- Yury Pisarchyk, Juhyun Lee "Efficient Memory Management for Deep Neural Net Inference".
  [Paper on ArXiv](https://arxiv.org/abs/2001.03288).

## Ecosystem

### Machine Learning Frameworks

- [TensorFlow Lite](https://blog.tensorflow.org/2020/07/accelerating-tensorflow-lite-xnnpack-integration.html).
- [TensorFlow.js WebAssembly backend](https://blog.tensorflow.org/2020/03/introducing-webassembly-backend-for-tensorflow-js.html).
- [PyTorch Mobile](https://pytorch.org/mobile).
- [ONNX Runtime Mobile](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html)
- [MediaPipe for the Web](https://developers.googleblog.com/2020/01/mediapipe-on-web.html).
- [Alibaba HALO (Heterogeneity-Aware Lowering and Optimization)](https://github.com/alibaba/heterogeneity-aware-lowering-and-optimization)
- [Samsung ONE (On-device Neural Engine)](https://github.com/Samsung/ONE)

## Acknowledgements

XNNPACK is a based on [QNNPACK](https://github.com/pytorch/QNNPACK) library. Over time its codebase diverged a lot, and XNNPACK API is no longer compatible with QNNPACK.
