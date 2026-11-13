module ModifiersTests

using ADTypes
using DifferentiationInterface
using DifferentiationInterface: AutoZeroForward
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest: Scenario, TripleScenario, Result, @scenario
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using SparseArrays
using SparseConnectivityTracer
using SparseMatrixColorings
using Test

include("counting_testset.jl")

kwargs = (; isapprox, atol = 0, rtol = 1.0e-8, scenario_intact = true, reprepare = true)

f_num(x::Number) = [sin(2x), cos(x)]
f_arr(x::AbstractArray) = sum(abs2, x) + prod(x)
f_arr_grad(x) = 2 .* x .+ [x[2] * x[3], x[1] * x[3], x[1] * x[2]]
f_arr_hess(x) = [2 x[3] x[2]; x[3] 2 x[1]; x[2] x[1] 2]
f_vec!(y, x) = (y .= x .^ 2; nothing)
f_vec_jac(x) = [i == j ? 2x[i] : 0.0 for i in 1:3, j in 1:3]

x_num = 0.5
x = [1.0, 2.0, 3.0]
dx = ([1.0, 0.0, -1.0],)
dy = ([0.0, 1.0, 2.0],)

canonical = [
    (
        @scenario value_derivative_and_second_derivative(f_num, missing, x_num) Result(
            f_num(x_num), [2cos(2x_num), -sin(x_num)], [-4sin(2x_num), -cos(x_num)]
        )
    ),
    (@scenario value_gradient_and_hessian(f_arr, missing, x) Result(f_arr(x), f_arr_grad(x), f_arr_hess(x))),
    (
        @scenario value_gradient_and_hvp(f_arr, missing, x, dx) Result(
            f_arr(x), f_arr_grad(x), map(t -> f_arr_hess(x) * t, dx)
        )
    ),
    (@scenario value_and_jacobian(f_vec!, zeros(3), missing, x) Result(x .^ 2, f_vec_jac(x), nothing)),
    (
        @scenario value_and_pushforward(f_vec!, zeros(3), missing, x, dx) Result(
            x .^ 2, map(t -> f_vec_jac(x) * t, dx), nothing
        )
    ),
    (
        @scenario value_and_pullback(f_vec!, zeros(3), missing, x, dy) Result(
            x .^ 2, map(t -> f_vec_jac(x)' * t, dy), nothing
        )
    ),
]

# all outputs are arrays
scens = map(TripleScenario, mapreduce(s -> DIT.all_variants(s; include_inplace = true), vcat, canonical))
n_tangent_scens = count(ts -> DIT.needs_tangent(DIT.operator(ts)), scens)

function correctness_counts(backend, scens; rtol = 1.0e-8)
    return count_tests() do
        for ts in scens
            DIT.test_correctness(backend, ts; kwargs..., rtol)
        end
    end
end

@testset "Original scenarios" begin
    @test length(scens) == 6 * 4
    @test correctness_counts(AutoForwardDiff(), scens).failures == 0
end

# closures store their outputs in `Float64` buffers, which fails with dual numbers,
# and nested finite differences are not accurate enough for second order
@testset "$m" for (m, n, backend, rtol) in [
        (DIT.batchify, n_tangent_scens, AutoForwardDiff(), 1.0e-8),
        (
            s -> DIT.closurify(filter(ts -> DIT.order(DIT.operator(ts)) == 1, s)),
            count(ts -> DIT.order(DIT.operator(ts)) == 1, scens), AutoFiniteDiff(), 1.0e-3,
        ),
        (DIT.constantify, length(scens), AutoForwardDiff(), 1.0e-8),
        (s -> DIT.cachify(s; use_tuples = false), length(scens), AutoForwardDiff(), 1.0e-8),
        (s -> DIT.cachify(s; use_tuples = true), length(scens), AutoForwardDiff(), 1.0e-8),
        (DIT.constantorcachify, length(scens), AutoForwardDiff(), 1.0e-8),
    ]
    new_scens = m(scens)
    @test length(new_scens) == n
    counts = correctness_counts(backend, new_scens; rtol)
    @test counts.failures == 0
    @test counts.passes > 0
    for ts in new_scens
        # the function is shared by all three scenarios
        f = DIT.func(DIT.exec_scenario(ts))
        @test DIT.func(DIT.prep_scenario(ts)) === f
        @test DIT.func(DIT.flush_scenario(ts)) === f
    end
end

@testset "Closurify" begin
    for (ts, new_ts) in zip(scens, DIT.closurify(scens))
        s, new_s = DIT.exec_scenario(ts), DIT.exec_scenario(new_ts)
        # outputs are scaled by 3 + 4
        for k in DIT.returned_orders(DIT.operator(s))
            v, new_v = DIT.getorder(DIT.result(s), k), DIT.getorder(DIT.result(new_s), k)
            @test all(map((a, b) -> 7a ≈ b, v isa Tuple ? v : (v,), new_v isa Tuple ? new_v : (new_v,)))
        end
        if !DIT.has_inplace_function(s)
            @test DIT.func(new_s)(DIT.primal_input(s)) ≈ 7 * DIT.func(s)(DIT.primal_input(s))
        end
    end
end

@testset "Batchify" begin
    for ts in DIT.batchify(scens)
        s = DIT.exec_scenario(ts)
        @test length(DIT.tangents(s)) == 2
        @test length(DIT.tangents(DIT.prep_scenario(ts))) == 2
        @test DIT.tangents(s)[2] == -DIT.tangents(s)[1]
        if DIT.is_inplace(DIT.operator(s))
            @test length(last(DIT.output_buffers(s))) == 2
        end
    end
    @test_throws ArgumentError DIT.batchify(first(scens))
end

@testset "Contexts" begin
    for ts in DIT.constantify(scens)
        exec_context = only(DIT.contexts(DIT.exec_scenario(ts)))
        prep_context = only(DIT.contexts(DIT.prep_scenario(ts)))
        @test exec_context isa Constant
        @test DI.unwrap(exec_context) == 3.0
        # preparation does not use the same constant
        @test DI.unwrap(prep_context) == 0.0
    end
    for ts in DIT.cachify(scens; use_tuples = false)
        @test only(DIT.contexts(DIT.exec_scenario(ts))) isa Cache
    end
    # modifiers which add contexts do not apply to scenarios which already have some
    constantified = DIT.constantify(scens)
    @test isempty(DIT.constantify(constantified))
    @test isempty(DIT.cachify(constantified; use_tuples = false))
    @test_throws ArgumentError DIT.constantify(first(constantified))
end

@testset "Names" begin
    s = @scenario "name" value_and_gradient(f_arr, missing, x) Result(f_arr(x), f_arr_grad(x), nothing)
    @test string(DIT.constantify(s)) == "name [constantified]"
    @test string(DIT.batchify(DIT.zero_der(@scenario "a" pushforward(f_vec!, zeros(3), missing, x, dx) Result(; order1 = dx)))) ==
        "a [zero derivative] [batchified]"
end

@testset "Zero derivatives" begin
    first_order = filter(ts -> DIT.order(DIT.operator(ts)) == 1, scens)
    @test correctness_counts(AutoZeroForward(), first_order).failures > 0
    zero_scens = DIT.zero_der(first_order)
    @test correctness_counts(AutoZeroForward(), zero_scens).failures == 0
    @test correctness_counts(AutoForwardDiff(), zero_scens).failures > 0
end

@testset "Change function" begin
    g(x) = f_arr(x)
    s = @scenario "name" value_and_gradient(f_arr, missing, x) Result(f_arr(x), f_arr_grad(x), nothing)
    new_s = DIT.change_function(s, g)
    @test DIT.func(new_s) === g
    @test string(new_s) == "name [new function]"
    new_ts = DIT.change_function(TripleScenario(s), g)
    @test DIT.func(DIT.prep_scenario(new_ts)) === g
    @test DIT.func(new_ts) === g
    # mismatch scenarios remain mismatched
    mismatched = DIT.change_function(DIT.mismatch(s), g)
    @test DIT.func(DIT.prep_scenario(mismatched)) === g
    @test DIT.func(mismatched) === g
    @test isnothing(DIT.func(DIT.exec_scenario(mismatched)))
    # the function of a mismatch scenario can be wrapped, like `scen.f` before
    @test DIT.func(DIT.mismatch(s)) === DIT.func(s)
end

@testset "Compute results" begin
    for ts in scens
        zero_ts = DIT.zero_der(ts)
        new_ts = DIT.compute_results(zero_ts, AutoForwardDiff())
        op = DIT.operator(ts)
        counts = count_tests() do
            DIT.test_approx(
                op, DIT.result(DIT.exec_scenario(new_ts)), DIT.result(DIT.exec_scenario(ts));
                isapprox, atol = 0, rtol = 1.0e-8
            )
        end
        @test counts.failures == 0
        @test DIT.prep_scenario(new_ts) === DIT.prep_scenario(zero_ts)
    end
end

@testset "Mismatch $kind" for kind in (:function, :input)
    mismatched = map(s -> DIT.mismatch(s; kind), scens)
    @test all(DIT.is_mismatch, mismatched)
    counts = correctness_counts(AutoForwardDiff(), mismatched)
    @test counts == (; passes = 2 * length(scens), failures = 0)
    @test_throws ArgumentError DIT.mismatch(first(scens); kind = :other)
end

@testset "Map arrays" begin
    immutable_copy(x) = copy(x)
    mutable_copy(x) = copy(x)
    m = DIT.map_arrays(immutable_copy, mutable_copy; func = f -> f)
    for ts in DIT.cachify(scens; use_tuples = true)
        new_ts = m(ts)
        s, new_s = DIT.exec_scenario(ts), DIT.exec_scenario(new_ts)
        @test DIT.primal_input(new_s) == DIT.primal_input(s)
        if DIT.primal_input(s) isa AbstractArray
            @test DIT.primal_input(new_s) !== DIT.primal_input(s)
        end
        @test DIT.func(new_s) === DIT.func(s)
        # caches with tuples remain caches
        c = only(DIT.contexts(new_s))
        @test c isa Cache
        @test DI.unwrap(c).useful_cache[1] !== DI.unwrap(only(DIT.contexts(s))).useful_cache[1]
        for (b, new_b) in zip(DIT.output_buffers(s), DIT.output_buffers(new_s))
            @test new_b !== b
        end
    end
    counts = correctness_counts(AutoForwardDiff(), m(DIT.constantify(scens)))
    @test counts.failures == 0
    @test counts.passes > 0
end

@testset "Buffers are not modified" begin
    # buffers are never read, and scaling them could destroy their sparsity pattern
    f!(y, x) = (y .= x .^ 2; nothing)
    x = [1.0, 2.0, 3.0]
    jac = spdiagm(2x)
    buffer = DIT.custom_zero(jac)  # explicit zeros in the structural pattern
    s = Scenario(jacobian!, f!, (zeros(3), buffer), (x,); result = Result(x .^ 2, jac, nothing))
    for m in (DIT.zero_der, DIT.closurify, DIT.constantify, DIT.constantorcachify)
        @test only(DIT.output_buffers(m(s))) === buffer
    end
    sparse_backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    counts = correctness_counts(sparse_backend, [DIT.constantify(TripleScenario(s))])
    @test counts.failures == 0
    @test counts.passes > 0
end

@testset "Filters" begin
    @test DIT.has_matrix(@scenario value_and_gradient(f_arr, missing, ones(2, 2)) Result(6.0, ones(2, 2), nothing))
    @test !DIT.has_matrix(first(scens))
    @test DIT.function_place(scens[end]) == :in
    @test DIT.operator_place(DIT.inplace(DIT.exec_scenario(scens[end]))) == :in
end

end
