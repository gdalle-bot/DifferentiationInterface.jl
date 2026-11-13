module CorrectnessTests

using ADTypes
using DifferentiationInterface
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest: Scenario, TripleScenario, Result, @scenario
using ForwardDiff: ForwardDiff
using SparseArrays
using SparseConnectivityTracer
using SparseMatrixColorings
using Test

include("../basics/counting_testset.jl")

kwargs = (; isapprox, atol = 0, rtol = 1.0e-8, scenario_intact = true, reprepare = true)
backend = AutoForwardDiff()

## Canonical scenarios for every operator family

f_num(x::Number, c) = [sin(c * x), cos(x)]
f_num_der(x, c) = [c * cos(c * x), -sin(x)]
f_num_der2(x, c) = [-c^2 * sin(c * x), -cos(x)]

f_arr(x::AbstractArray, c) = c * sum(abs2, x) + prod(x)
f_arr_grad(x, c) = 2c .* x .+ [x[2] * x[3], x[1] * x[3], x[1] * x[2]]
f_arr_hess(x, c) = [2c x[3] x[2]; x[3] 2c x[1]; x[2] x[1] 2c]

f_vec!(y, x, c) = (y .= c .* x .^ 2; nothing)
f_vec_jac(x, c) = Matrix(spdiagm(2c .* x))

c = 3.0
x_num = 0.5
x = [1.0, 2.0, 3.0]
tx = ([1.0, 0.0, -1.0], [0.0, 2.0, 1.0])
ty = ([1.0, 1.0, 0.0], [0.0, -1.0, 2.0])

canonical_scenarios = [
    (
        @scenario value_and_derivative(f_num, missing, x_num, Constant(c)) Result(
            f_num(x_num, c), f_num_der(x_num, c), nothing
        )
    ),
    (
        @scenario value_derivative_and_second_derivative(f_num, missing, x_num, Constant(c)) Result(
            f_num(x_num, c), f_num_der(x_num, c), f_num_der2(x_num, c)
        )
    ),
    (
        @scenario value_and_gradient(f_arr, missing, x, Constant(c)) Result(
            f_arr(x, c), f_arr_grad(x, c), nothing
        )
    ),
    (
        @scenario value_gradient_and_hessian(f_arr, missing, x, Constant(c)) Result(
            f_arr(x, c), f_arr_grad(x, c), f_arr_hess(x, c)
        )
    ),
    (
        @scenario value_gradient_and_hvp(f_arr, missing, x, tx, Constant(c)) Result(
            f_arr(x, c), f_arr_grad(x, c), map(dx -> f_arr_hess(x, c) * dx, tx)
        )
    ),
    (
        @scenario value_and_jacobian(f_vec!, zeros(3), missing, x, Constant(c)) Result(
            c .* x .^ 2, f_vec_jac(x, c), nothing
        )
    ),
    (
        @scenario value_and_pushforward(f_vec!, zeros(3), missing, x, tx, Constant(c)) Result(
            c .* x .^ 2, map(dx -> f_vec_jac(x, c) * dx, tx), nothing
        )
    ),
    (
        @scenario value_and_pullback(f_vec!, zeros(3), missing, x, ty, Constant(c)) Result(
            c .* x .^ 2, map(dy -> f_vec_jac(x, c)' * dy, ty), nothing
        )
    ),
]

# all outputs are arrays
all_scenarios = mapreduce(s -> DIT.all_variants(s; include_inplace = true), vcat, canonical_scenarios)

n_candidates(op) = DIT.needs_tangent(op) ? 4 : 3

@testset "Correct scenarios" begin
    @test length(all_scenarios) == 8 * 4
    @test sort(unique(map(DIT.operator_symbol ∘ DIT.operator, all_scenarios))) ==
        sort(vcat(DIT.FIRST_ORDER, DIT.SECOND_ORDER))
    @testset "$s" for s in all_scenarios
        counts = count_tests() do
            DIT.test_correctness(backend, TripleScenario(s); kwargs...)
        end
        @test counts.failures == 0
        @test counts.passes > 0
    end
end

## Mutation tests: every returned order is compared in every execution

corrupt(x::Number) = 2x + 1
corrupt(x::AbstractArray) = corrupt.(x)
corrupt(x::Tuple) = (corrupt(first(x)), Base.tail(x)...)

function corrupt_order(r::Result, k)
    return Result(
        k == 0 ? corrupt(r.order0) : r.order0,
        k == 1 ? corrupt(r.order1) : r.order1,
        k == 2 ? corrupt(r.order2) : r.order2,
    )
end

function drop_tangent(r::Result, k)
    return Result(
        r.order0, k == 1 ? Base.front(r.order1) : r.order1, k == 2 ? Base.front(r.order2) : r.order2
    )
end

function with_result(s::Scenario, r::Result)
    return Scenario(
        DIT.operator(s), DIT.func(s), DIT.pre_backend_args(s), DIT.post_backend_args(s); result = r
    )
end

@testset "Corrupted results are detected" begin
    @testset "$s" for s in all_scenarios
        op = DIT.operator(s)
        for k in DIT.returned_orders(op)
            wrong = with_result(s, corrupt_order(DIT.result(s), k))
            counts = count_tests() do
                DIT.test_correctness(backend, TripleScenario(wrong); kwargs...)
            end
            # one failure per execution (2) per preparation candidate
            @test counts.failures == 2 * n_candidates(op)
            if DIT.getorder(DIT.result(s), k) isa Tuple
                wrong = with_result(s, drop_tangent(DIT.result(s), k))
                counts = count_tests() do
                    DIT.test_correctness(backend, TripleScenario(wrong); kwargs...)
                end
                @test counts.failures == 2 * n_candidates(op)
            end
        end
    end
end

## Preparation

@testset "Re-preparation" begin
    f!(y, x) = (y .= x .^ 2; nothing)
    x = [1.0, 2.0, 3.0]
    exec = @scenario value_and_jacobian(f!, zeros(3), missing, x) Result(
        x .^ 2, Matrix(spdiagm(2x)), nothing
    )
    prep = @scenario value_and_jacobian(f!, zeros(2), missing, zeros(2))
    ts = TripleScenario(exec; prep)
    @test DIT.should_reprepare(ts)
    counts = count_tests() do
        DIT.test_correctness(backend, ts; kwargs...)
    end
    @test counts.failures == 0
    counts = count_tests() do
        DIT.test_correctness(backend, ts; kwargs..., reprepare = false)
    end
    @test counts.failures > 0
end

struct OtherFunction{F}
    f::F
end
(of::OtherFunction)(args...) = of.f(args...)

@testset "Preparation mismatch" begin
    exec = @scenario value_and_gradient(f_arr, missing, x, Constant(c)) Result(
        f_arr(x, c), f_arr_grad(x, c), nothing
    )
    prep = Scenario(
        value_and_gradient, OtherFunction(f_arr), (), DIT.post_backend_args(exec)
    )
    counts = count_tests() do
        DIT.test_correctness(backend, TripleScenario(exec; prep, mismatch = true); kwargs...)
    end
    @test counts == (; passes = 2, failures = 0)
    # a scenario wrongly declared as a mismatch is detected
    counts = count_tests() do
        DIT.test_correctness(backend, TripleScenario(exec; mismatch = true); kwargs...)
    end
    @test counts == (; passes = 1, failures = 1)
end

## Missing reference

@testset "Missing reference" begin
    s = @scenario gradient(f_arr, missing, x, Constant(c))
    @test_throws ArgumentError(
        "Correctness tests require a reference result for the executed scenario"
    ) DIT.test_correctness(backend, TripleScenario(s); kwargs...)
end

## Scenario intact

function f_mutating(x, c)
    c[2] += 1  # does not affect the output
    return sum(abs2, x) * c[1]
end

@testset "Scenario intact" begin
    s = @scenario value_and_gradient(f_mutating, missing, x, Constant([1.0, 0.0])) Result(
        sum(abs2, x), 2x, nothing
    )
    counts = count_tests() do
        DIT.test_correctness(backend, TripleScenario(s); kwargs...)
    end
    @test counts.failures == 1
    counts = count_tests() do
        DIT.test_correctness(backend, TripleScenario(s); kwargs..., scenario_intact = false)
    end
    @test counts.failures == 0
end

## Sparsity

@testset "Sparsity" begin
    sparse_backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    f!(y, x) = (y .= x .^ 2; nothing)
    x = [1.0, 2.0, 3.0]
    jac = spdiagm(2x)
    s = @scenario value_and_jacobian(f!, zeros(3), missing, x) Result(x .^ 2, jac, nothing)

    sparse_counts = count_tests() do
        DIT.test_correctness(sparse_backend, TripleScenario(s); kwargs...)
    end
    @test sparse_counts.failures == 0
    dense_counts = count_tests() do
        DIT.test_correctness(backend, TripleScenario(s); kwargs...)
    end
    @test dense_counts.failures == 0
    # sparse type and pattern, for each execution and preparation candidate
    @test sparse_counts.passes == dense_counts.passes + 2 * 2 * 3

    # the pattern must match exactly, even with an explicit zero
    jac_wrong = sparse([1, 2, 3, 1], [1, 2, 3, 2], [2.0, 4.0, 6.0, 0.0])
    s_wrong = @scenario value_and_jacobian(f!, zeros(3), missing, x) Result(
        x .^ 2, jac_wrong, nothing
    )
    counts = count_tests() do
        DIT.test_correctness(sparse_backend, TripleScenario(s_wrong); kwargs...)
    end
    @test counts.failures == 2 * 3
end

end
