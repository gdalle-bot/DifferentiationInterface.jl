# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.11.0...main)

### Changed

- **Breaking:** scenarios are now call-based. A `Scenario` stores one call to a DifferentiationInterface operator, `op(f, pre_backend_args..., backend, post_backend_args...)`, together with its reference `Result`. It is easiest to create with the `@scenario` macro, by writing the call with `missing` in place of the backend. Each scenario tests a single operator, so the operators with and without value, in-place and out-of-place, are now separate scenarios.
- **Breaking:** scenarios are tested as `TripleScenario`s: preparation happens on one scenario (by default, on zero inputs), execution on another, and a third "flush" scenario is executed between two executions to check that preparation keeps no stale state.
- **Breaking:** all scenario generators (`default_scenarios`, `sparse_scenarios`, `complex_scenarios`, `complex_sparse_scenarios`, `empty_scenarios`, `allocfree_scenarios`, `static_scenarios`, `component_scenarios`, `gpu_scenarios`) return `TripleScenario`s, while keeping their keyword arguments. Most of them add preparation mismatch scenarios unless `include_mismatched=false`.
- Correctness tests are stricter: every order returned by an operator must be present in the reference result and is compared, tuples of tangents must have the right length, output buffers must be returned as is in every execution, and every preparation (none, strict, non-strict, same point) is tested.
- The sparsity check is automatic for sparse backends and sparse reference matrices, and compares structural patterns instead of numbers of nonzeros.
- Test `value_gradient_and_hvp(!)` instead of the deprecated `gradient_and_hvp(!)`, which requires DifferentiationInterface v0.7.22.

### Added

- `custom_zero`, an extensible function used to build preparation inputs.
- `default_scenarios(; modifiers)` to add scenarios transformed by custom modifiers.

### Removed

- **Breaking:** the `sparsity` keyword of `test_differentiation` (the check is automatic), `zero(::Scenario)` (replaced by `zero_der`), `same_function` and `no_matrices` (replaced by filtering on `has_matrix`).

### Migration

| DifferentiationInterfaceTest v0.11                       | DifferentiationInterfaceTest v0.12                                                         |
|:-------------------------------------------------------- |:------------------------------------------------------------------------------------------ |
| `Scenario{:gradient,:out}(f, x; res1=g)`                 | `@scenario gradient(f, missing, x) Result(; order1=g)`                                      |
| `Scenario{:gradient,:in}(f, x; res1=g)`                  | `@scenario gradient!(f, similar(g), missing, x) Result(; order1=g)`                        |
| value and gradient tested by the same scenario           | `@scenario value_and_gradient(f, missing, x) Result(f(x), g, nothing)`, or `DifferentiationInterfaceTest.all_variants` |
| `Scenario{:pushforward,:out}(f!, y, x, (dx,); res1=(dy,))` | `@scenario pushforward(f!, y, missing, x, (dx,)) Result(; order1=(dy,))`                  |
| `Scenario{:hessian,:out}(f, x; res1=g, res2=H)`          | `@scenario value_gradient_and_hessian(f, missing, x) Result(f(x), g, H)`                   |
| `Scenario{:gradient,:out}(f, x, Constant(c); res1=g)`    | `@scenario gradient(f, missing, x, Constant(c)) Result(; order1=g)`                        |
| `prep_args=(; x=x_prep, contexts=())`                    | `TripleScenario(s; prep=@scenario gradient(f, missing, x_prep))`                           |
| `scen.f`, `scen.x`, `scen.y`, `scen.res1`                | `DifferentiationInterfaceTest.func(s)`, `primal_input(s)`, `primal_output(s)`, `result(s)` |
| `zero(scen)`                                             | `DifferentiationInterfaceTest.zero_der(scen)`                                              |
| `test_differentiation(...; sparsity=true)`               | `test_differentiation(...)`                                                                |

## [0.11.0](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.10.4...DifferentiationInterfaceTest-v0.11.0)

### Changed

- Benchmarking and type stability analysis are now implemented in package extensions, which depend on Chairmarks and JET respectively
- Benchmarking no longer returns a `DataFrame` but a Tables.jl-compatible object.

## [0.10.4](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.10.3...DifferentiationInterfaceTest-v0.10.4)

### Removed

- Remove neural network tests ([#914](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/914))

## [0.10.3](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.10.2...DifferentiationInterfaceTest-v0.10.3)

### Fixed

- Update JET (to v0.11) & JLArrays compat ([#877](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/877))
- Allow JET v0.10 but disable type stability tests on 1.12 ([#841](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/841))

## [0.10.2](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.10.1...DifferentiationInterfaceTest-v0.10.2)

### Fixed

- Refactor test loops ([#848](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/848))

## [0.10.1](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.10.0...DifferentiationInterfaceTest-v0.10.1)

### Added

- Improve support for empty inputs (still not guaranteed) ([#835](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/835))
- Compute Scenario results with a reference backend ([#839](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/839))

### Fixed

- Put test deps into `test/Project.toml` ([#840](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/840))
- Set up `pre-commit` ([#837](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/837))
- Bump compat for SparseConnectivityTracer v1 ([#823](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/823))

## [0.10.0](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.9.6...DifferentiationInterfaceTest-v0.10.0)

### Changed

- Specify preparation arguments in DIT Scenario ([#786](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/786))

### Removed

- Remove scenario lists from public API ([#796](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/796))

## [0.9.6](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.9.5...DifferentiationInterfaceTest-v0.9.6) - 2025-03-28

### Added

- Add new ConstantOrCache context ([#749](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/749))
- Support nested tuples of arrays as Caches ([#748](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/748))
- Test type consistency between preparation and execution ([#745](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/745))
