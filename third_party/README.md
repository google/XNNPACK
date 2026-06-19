This directory contains sources and build definitions for other projects.

* `BUILD`/ files are for Bazel, which will fetch dependencies automatically.
* Anything containing a `README.xnnpack` is for use with the Chromium toolchain
   and `DEPS` (see `BUILD.md` for more details.)

## Security Critical libraries

The stand-alone build of XNNPACK with GN and the Chromium toolchain is intended
for debugging and integration, not distribution. As a result, _security-critical
sub-projects may not be up to date_. When integrated into a larger product (i.e.
Chromium), the versions in the outer DEPS file will be resolved instead. These
should always be up-to-date.

## Updating a third_party library

In general:
1. Update `//DEPS` with the desired revision
2. Run `gclient sync`
3. Build for a variety of platforms
