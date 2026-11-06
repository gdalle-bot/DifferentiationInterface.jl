# Tutorial

We present a typical workflow with DifferentiationInterfaceTest.jl, building on the tutorial of the [DifferentiationInterface.jl documentation](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface) (which we encourage you to read first).

```@repl tuto
import Chairmarks
using DataFrames
using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using Zygote: Zygote
```

## Introduction

The AD backends we want to compare are [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [Zygote.jl](https://github.com/FluxML/Zygote.jl).

```@example tuto
backends = [AutoForwardDiff(), AutoZygote()]
```

To do that, we are going to take gradients of a simple function:

```@example tuto
f(x::AbstractArray) = sum(sin, x)
```

Of course we know the true gradient mapping:

```@example tuto
∇f(x::AbstractArray) = cos.(x)
```

DifferentiationInterfaceTest.jl relies on so-called [`Scenario`](@ref)s, each of which encapsulates one call to a DifferentiationInterface operator, together with the reference [`Result`](@ref) it should return.
The easiest way to create a scenario is the [`@scenario`](@ref) macro, where you write the call itself with `missing` in place of the backend:

```@example tuto
xv = rand(Float32, 3)
s = @scenario value_and_gradient(f, missing, xv) Result(f(xv), ∇f(xv), nothing)
```

The [`Result`](@ref) stores the reference outputs by differentiation order: here the value and the gradient.
Every order returned by the operator must be provided, and all of them are compared.

From such a scenario, [`DifferentiationInterfaceTest.all_variants`](@ref) generates the other operators of the same family: with or without the value, in-place or out-of-place.

```@example tuto
xm = rand(Float64, 3, 2)
scenarios = vcat(
    DifferentiationInterfaceTest.all_variants(s),
    DifferentiationInterfaceTest.all_variants(
        @scenario value_and_gradient(f, missing, xm) Result(f(xm), ∇f(xm), nothing)
    ),
)
```

When testing, each scenario is wrapped into a [`TripleScenario`](@ref), which prepares the operator on zero inputs, then executes it twice on the actual inputs, with a "flush" call on other inputs in between.
This makes sure that the preparation result can be reused safely.

## Testing

The main entry point for testing is the function [`test_differentiation`](@ref).
It has many options, but the main ingredients are the following:

```@repl tuto
test_differentiation(
    backends,  # the backends you want to compare
    scenarios;  # the scenarios you defined,
    correctness=true,  # compares values against the reference
    type_stability=:none,  # checks type stability with JET.jl
    detailed=true,  # prints a detailed test set
)
```

## Benchmarking

Once you are confident that your backends give the correct answers, you probably want to compare their performance.
This is made easy by the [`benchmark_differentiation`](@ref) function, whose syntax should feel familiar:

```@example tuto
table = benchmark_differentiation(backends, scenarios);
```

The resulting object is a table, which can easily be converted into a `DataFrame` from [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl).
Its columns correspond to the fields of [`DifferentiationBenchmarkDataRow`](@ref).

```@example tuto
df = DataFrame(table)
```
