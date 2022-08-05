# Microkernel naming conventions

This documents deciphers XNNPACK's microkernels naming convention.

## General conventions

Microkernel function names follow this convention:

`xnn_<datatype>_<microkernel><activation?>_ukernel_<parameters>__<arch>`

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

## GEMM and IGEMM microkernels

The `<parameters>` for GEMM and IGEMM microkernels represent the `mr` and `nr`
of the microkernel. You can think of it as the number of rows and columns of the
output calculated by the microkernel.

E.g. `xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a7` processes 32
elements of the output matrix.

## Average Pooling and Global Average Pooling

These microkernels come in 2 varieties, uni-pass and multi-pass.

Uni-pass have `Cx` in their name, where `C` is a number. This microkernel
processes up to and including `C` elements.

Multi-pass have `CpDx` in their name, where `C` and `D` are numbers. This
microkernel processes `D` elements in the first pass, and middle pass (which can
run multiple times), and up to `C` elements in the last pass.

E.g. `xnn_f32_avgpool_minmax_ukernel_9x__neon_c4` can process up to 9 elements.
