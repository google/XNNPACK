# Microkernel naming conventions

This documents deciphers XNNPACK's microkernels naming convention.

## General conventions

Microkernel function names follow this convention:

`xnn_<datatype>_<microkernel><activation?>_ukernel_<parameters>__<arch>_u<unroll>`

Where `<datatype>` can be:

-   `cs16`
-   `f16` - 16-bit half precision float
-   `f32` - 32-bit single precision float
-   `qc8`
-   `qs8` - quantized signed 8 bit
-   `qu8` - quantized unsigned 8 bit
-   `s16`
-   `u32`
-   `x8`
-   `x16`
-   `x24`
-   `x32`
-   `xx`

`<microkernel>` is the type of microkernel, such as:

-   `gemm`
-   `igemm`
-   `avgpool`

`<activation>` if supported for the microkernel is activation that is fused into
the microkernel:

-   `linear`
-   `minmax`
-   `relu`

`<parameters>` are microkernel specific, and can mean different things depending
on the microkernel (see below for details).

`<arch>` is the architecture the microkernel is optimized for, and can contain
further subdivisions for additional instruction sets supported on the specified
architecture, or processor information:

-   `scalar`
-   `aarch32_neon_cortex_a55`
-   `neonv8_mlal`
-   `wasm`
-   `avx512`
-   `avx512skx`

`<unroll>` is the unroll factor, in elements, along the innermost loop of the
microkernel.

## GEMM and IGEMM microkernels

GEMM refers to general matrix multiplication. IGEMM is a modification of GEMM,
stands for indirect. Instead of reading matrix A directly, an IGEMM microkernel
reads pointers to A (one level of indirection). See
[The Indirect Convolution Algorithm](https://arxiv.org/abs/1907.02129) for
details.

In the context of convolution operator, we decide whether to use GEMM or IGEMM
based on parameters like stride, padding, kernel size, etc.

The `<parameters>` for GEMM and IGEMM microkernels represent the `mr` and `nr`
of the microkernel. You can think of it as the number of rows and columns of the
output calculated by the microkernel.

E.g. `xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a7` processes 32
elements of the output matrix.

## DWCONV microkernels

These microkernels come in 2 varieties, uni-pass and multi-pass.

Uni-pass have `XpYc` in their name, where `X` is the kernel tile, and `Y` is the
channel tile. `p` stands for primary, `c` for channel.

Multi-pass have `UfVmWlXcYsZr` in their name, where `U` is the first pass tile,
`V` is the middle pass tile, `W` is the last pass tile, `X` is the channel tile,
`Y` is the channel subtile, and `Z` is the channel round. `f` stands for first,
`m` for middle, `l` for last, `c` for channel, `s` for subtile, `r` for round.
The kernel size must be at least `W+1`, the middle pass runs for as many
iterations as possible, and the last pass handles the remainder (at least 1).
`c`, `s`, `r`, affects the tiling of channels. We run as many tiles of `c` as
possible, followed by rounds of `s`. We determine how many tiles of `c` to run
based on rounding the number of channels up to `r`. `r` is determined based on
the natural tiling size of the microarchitecture (e.g. SSE/AVX) and the number
of elements we can read OOB (`XNN_EXTRA_BYTES`).

## Average Pooling and Global Average Pooling

These microkernels come in 2 varieties, uni-pass and multi-pass.

Uni-pass have `Cx` in their name, where `C` is a number. This microkernel
processes up to and including `C` elements.

Multi-pass have `CpDx` in their name, where `C` and `D` are numbers. This
microkernel processes `D` elements in the first pass, and middle pass (which can
run multiple times), and up to `C` elements in the last pass.

E.g. `xnn_f32_avgpool_minmax_ukernel_9x__neon_c4` can process up to 9 elements.
