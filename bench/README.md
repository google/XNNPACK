# XNNPACK Microkernel Benchmarks

This directory contains microkernel benchmarks for XNNPACK.
Some of these benchmarks need shapes provided via the command line.

You can use standard shell expansion to pass shapes from the provided text
files:

```bash
# Run all GEMM layers for MobileNet V2
bazel run -c opt //bench:f32_gemm_bench -- $(cat bench/models/mobilenet_v2.txt)

# Run a custom Convolution shape (H W KH KW PH PW S D GCin GCout)
bazel run -c opt //bench:f32_conv_hwc_bench -- 224 224 3 3 1 1 2 1 3 32
```

## Shape Formats

The benchmarks expect positional arguments in the following formats:

| Benchmark Type | Format | Example (Positional Args) |
| :--- | :--- | :--- |
| **GEMM** | `M N K` | `12544 64 64` |
| **Batch GEMM** | `B M N K` | `12 384 64 384` |
| **Convolution** | `H W KH KW PH PW S D GCin GCout` | `224 224 3 3 1 1 2 1 3 32` |
| **Depthwise Conv** | `H W KH KW PH PW S D G` | `112 112 3 3 2 2 1 1 32` |
| **Sparse GEMM (SpMM)** | `M N K` | `12544 16 32` |
| **Deconvolution** | `H W Cout` | `224 224 64` |

## Model Library

Representative shapes for various models are provided in the `models/`
subdirectory.

### Common Models (`models/`)
- **MobileNet**: `mobilenet_v1.txt`, `mobilenet_v2.txt`, `mobilenet_v3_small.txt`, `mobilenet_v3_large.txt` (and variants for `_conv.txt`, `_dwconv.txt`, `_spmm.txt`)
- **ResNet**: `resnet18_gemm.txt`, `resnet50.txt`
- **ShuffleNet**: `shufflenet_v1_g*.txt`, `shufflenet_v2_x*.txt`
- **Inception**: `inception_v3_gemm.txt`
- **LLM / Attention**: `llm_gemm.txt`, `attention_bgemm.txt`, `sd1x_diffusion_bgemm.txt`

## Benchmark-Specific Flags

### GEMM Block Size
For blockwise quantization kernels (names ending in `_qb`), you can specify a
uniform block size for all cases using the `--block_size` flag:

```bash
bazel run -c opt ///bench:qd8_f32_qc8w_gemm_bench -- --block_size=128 $(cat bench/models/llm_gemm.txt)
```
*Default value is 32.*
