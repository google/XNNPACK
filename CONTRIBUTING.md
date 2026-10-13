# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Development & PR Guidelines

Before contributing, please read [AGENTS.md](AGENTS.md) for architectural
guidelines, code generation workflows, and pull request requirements.

In particular:
- **Atomic Scope**: Keep pull requests focused on a single operator or
  microkernel family.
- **Code Generation**: Never edit auto-generated files directly; update
  templates (`.c.in`) or specifications (`.yaml`) and run the generators.
- **Benchmarking**: Performance and microkernel changes require reproducible
  benchmark results from `bench/` on target hardware.
- **No Speculative Sweeps**: Avoid mass defensive hardening or redundant
  internal validation sweeps.
- **Minimal Comments**: Omit comments on self-explanatory code or validation
  checks (only comment non-obvious invariants), and keep PR descriptions
  concise (1-3 sentences plus test/benchmark results).

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).
