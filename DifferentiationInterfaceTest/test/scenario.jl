using DifferentiationInterface
using DifferentiationInterfaceTest
using DifferentiationInterfaceTest: default_scenarios, sparse_scenarios
import DifferentiationInterfaceTest as DIT
using ForwardDiff: ForwardDiff
using Test

@testset "Naming" begin
    scen = @scenario "My pretty little scenario" gradient(sum, missing, zeros(10)) Result(;
        order1 = ones(10)
    )
    @test string(scen) == "My pretty little scenario"

    testset = test_differentiation(
        AutoForwardDiff(), [scen]; testset_name = "My amazing test set"
    )

    data = benchmark_differentiation(
        AutoForwardDiff(), [scen]; testset_name = "My amazing test set"
    )
end;

@testset "Compute results" begin
    for ts in default_scenarios()
        DIT.is_mismatch(ts) && continue
        new_ts = DIT.compute_results(DIT.zero_der(ts), AutoForwardDiff())
        DIT.test_approx(
            DIT.operator(ts),
            DIT.result(DIT.exec_scenario(new_ts)),
            DIT.result(DIT.exec_scenario(ts));
            isapprox,
            atol = 0,
            rtol = sqrt(eps()),
        )
    end
end;

@testset "Coverage of default scenarios" begin
    # number of calls per (operator, operator place, function place) in the legacy generators
    function new_counts(scens)
        key(ts) = (DIT.operator_symbol(ts), DIT.operator_place(ts), DIT.function_place(ts))
        scens = filter(!DIT.is_mismatch, scens)
        return sort(collect(Dict(k => count(ts -> key(ts) == k, scens) for k in unique(map(key, scens)))))
    end
    all_kwargs = (;
        include_closurified = true,
        include_constantified = true,
        include_cachified = true,
        include_constantorcachified = true,
        use_tuples = true,
    )
    scens = default_scenarios()
    @test new_counts(scens) == [(:derivative, :in, :in) => 4, (:derivative, :in, :out) => 4, (:derivative, :out, :in) => 4, (:derivative, :out, :out) => 6, (:gradient, :in, :out) => 4, (:gradient, :out, :out) => 4, (:hessian, :in, :out) => 4, (:hessian, :out, :out) => 4, (:hvp, :in, :out) => 8, (:hvp, :out, :out) => 8, (:jacobian, :in, :in) => 10, (:jacobian, :in, :out) => 10, (:jacobian, :out, :in) => 10, (:jacobian, :out, :out) => 10, (:pullback, :in, :in) => 20, (:pullback, :in, :out) => 28, (:pullback, :out, :in) => 28, (:pullback, :out, :out) => 40, (:pushforward, :in, :in) => 28, (:pushforward, :in, :out) => 28, (:pushforward, :out, :in) => 28, (:pushforward, :out, :out) => 40, (:second_derivative, :in, :out) => 4, (:second_derivative, :out, :out) => 6]
    # one mismatch scenario for each scenario
    @test count(DIT.is_mismatch, scens) == count(!DIT.is_mismatch, scens)
    scens = default_scenarios(; all_kwargs...)
    @test new_counts(scens) == [(:derivative, :in, :in) => 20, (:derivative, :in, :out) => 20, (:derivative, :out, :in) => 20, (:derivative, :out, :out) => 30, (:gradient, :in, :out) => 20, (:gradient, :out, :out) => 20, (:hessian, :in, :out) => 20, (:hessian, :out, :out) => 20, (:hvp, :in, :out) => 40, (:hvp, :out, :out) => 40, (:jacobian, :in, :in) => 50, (:jacobian, :in, :out) => 50, (:jacobian, :out, :in) => 50, (:jacobian, :out, :out) => 50, (:pullback, :in, :in) => 100, (:pullback, :in, :out) => 140, (:pullback, :out, :in) => 140, (:pullback, :out, :out) => 200, (:pushforward, :in, :in) => 140, (:pushforward, :in, :out) => 140, (:pushforward, :out, :in) => 140, (:pushforward, :out, :out) => 200, (:second_derivative, :in, :out) => 20, (:second_derivative, :out, :out) => 30]
    @test count(DIT.is_mismatch, scens) == count(!DIT.is_mismatch, scens)
    # scenarios with smaller preparation inputs
    scens = default_scenarios(; include_normal = false, include_smaller = true, include_mismatched = false)
    @test new_counts(scens) == [(:derivative, :in, :in) => 4, (:derivative, :in, :out) => 4, (:derivative, :out, :in) => 4, (:derivative, :out, :out) => 6, (:gradient, :in, :out) => 4, (:gradient, :out, :out) => 4, (:hessian, :in, :out) => 4, (:hessian, :out, :out) => 4, (:hvp, :in, :out) => 4, (:hvp, :out, :out) => 4, (:jacobian, :in, :in) => 10, (:jacobian, :in, :out) => 10, (:jacobian, :out, :in) => 10, (:jacobian, :out, :out) => 10, (:pullback, :in, :in) => 10, (:pullback, :in, :out) => 14, (:pullback, :out, :in) => 14, (:pullback, :out, :out) => 20, (:pushforward, :in, :in) => 14, (:pushforward, :in, :out) => 14, (:pushforward, :out, :in) => 14, (:pushforward, :out, :out) => 20, (:second_derivative, :in, :out) => 4, (:second_derivative, :out, :out) => 6]
end;

@testset "Coverage of other scenarios" begin
    # number of calls per (operator, operator place, function place) in the legacy generators
    function new_counts(scens)
        key(ts) = (DIT.operator_symbol(ts), DIT.operator_place(ts), DIT.function_place(ts))
        scens = filter(!DIT.is_mismatch, scens)
        return sort(collect(Dict(k => count(ts -> key(ts) == k, scens) for k in unique(map(key, scens)))))
    end
    @test new_counts(DIT.sparse_scenarios()) == [(:hessian, :in, :out) => 10, (:hessian, :out, :out) => 10, (:jacobian, :in, :in) => 14, (:jacobian, :in, :out) => 14, (:jacobian, :out, :in) => 14, (:jacobian, :out, :out) => 14]
    @test new_counts(DIT.sparse_scenarios(; band_sizes = [], include_constantified = true, include_cachified = true, include_constantorcachified = true, use_tuples = true)) == [(:hessian, :in, :out) => 16, (:hessian, :out, :out) => 16, (:jacobian, :in, :in) => 32, (:jacobian, :in, :out) => 32, (:jacobian, :out, :in) => 32, (:jacobian, :out, :out) => 32]
    @test new_counts(DIT.complex_scenarios()) == [(:derivative, :in, :in) => 2, (:derivative, :in, :out) => 2, (:derivative, :out, :in) => 2, (:derivative, :out, :out) => 4, (:gradient, :in, :out) => 6, (:gradient, :out, :out) => 6, (:jacobian, :in, :in) => 2, (:jacobian, :in, :out) => 2, (:jacobian, :out, :in) => 2, (:jacobian, :out, :out) => 2, (:pullback, :in, :in) => 2, (:pullback, :in, :out) => 8, (:pullback, :out, :in) => 4, (:pullback, :out, :out) => 12, (:pushforward, :in, :in) => 4, (:pushforward, :in, :out) => 4, (:pushforward, :out, :in) => 4, (:pushforward, :out, :out) => 8]
    @test new_counts(DIT.complex_sparse_scenarios()) == [(:jacobian, :in, :in) => 2, (:jacobian, :in, :out) => 2, (:jacobian, :out, :in) => 2, (:jacobian, :out, :out) => 2]
    @test new_counts(DIT.empty_scenarios()) == [(:derivative, :out, :in) => 2, (:derivative, :out, :out) => 2, (:gradient, :out, :out) => 2, (:jacobian, :out, :in) => 2, (:jacobian, :out, :out) => 2]
    @test new_counts(DIT.allocfree_scenarios()) == [(:derivative, :out, :out) => 2, (:gradient, :in, :out) => 2, (:jacobian, :in, :in) => 2, (:pullback, :in, :in) => 2, (:pullback, :in, :out) => 2, (:pullback, :out, :out) => 2, (:pushforward, :in, :in) => 2, (:pushforward, :out, :out) => 4]
end;
