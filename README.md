# XNNPACK

XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 (SSE2 level) platforms. XNNPACK is not intended for direct use by deep learning practitioners researchers; instead it provides low-level performance primitives for accelerating high-level machine learning frameworks, such as [MediaPipe](https://mediapipe.dev), [TensorFlow Lite](https://www.tensorflow.org/lite), and [TensorFlow.js](https://www.tensorflow.org/js).

## Supported Architectures

- ARM on Android, Linux, and iOS
- ARM64 on Android, Linux, and iOS
- WebAssembly MVP
- WebAssembly SIMD (experimental)
- x86 and x86-64 (up to SSE2 only) on Android, Linux, and Mac

## Operator Coverage

XNNPACK implements the following neural network operators:

- 2D Convolution (including grouped and depthwise)
- 2D Deconvolution (AKA Transposed Convolution)
- 2D Average Pooling
- 2D Max Pooling
- 2D ArgMax Pooling (Max Pooling + indices)
- 2D Unpooling
- Add (tensors of same shape)
- Global Average Pooling
- Channel Shuffle
- Clamp (includes ReLU and ReLU6)
- HardSwish
- PReLU

All operators in XNNPACK support NHWC layout, but additionally allow custom stride along the **C**hannel dimension. Thus, operators can consume a subset of channels in the input tensor, and produce a subset of channels in the output tensor, providing a zero-cost Channel Split and Channel Concatenation operations.

## Acknowledgements

XNNPACK is a based on [QNNPACK](https://github.com/pytorch/QNNPACK) library. However, unlike QNNPACK, XNNPACK focuses entirely on floating-point operators, and its API is no longer compatible with QNNPACK.
