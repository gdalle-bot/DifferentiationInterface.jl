using DifferentiationInterface
using DifferentiationInterfaceTest
using DifferentiationInterfaceTest: default_scenarios
import DifferentiationInterfaceTest as DIT
using ForwardDiff: ForwardDiff
using Test

@testset "Naming" begin
    scen = LegacyScenario{:gradient, :out}(
        sum, zeros(10); res1 = ones(10), name = "My pretty little scenario"
    )
    @test string(scen) == "My pretty little scenario"

    testset = test_differentiation(
        AutoForwardDiff(), [scen]; testset_name = "My amazing test set"
    )

    data = benchmark_differentiation(
        AutoForwardDiff(), [scen]; testset_name = "My amazing test set"
    )
end;

@testset "Compute results (legacy)" begin
    scens = DifferentiationInterfaceTest.legacy_default_scenarios()
    new_scens = map(s -> compute_results(s, AutoForwardDiff()), scens)

    isapprox_robust(x, y) = isapprox(x, y)
    isapprox_robust(x::Nothing, y::Nothing) = true
    isapprox_robust(x::NTuple, y::NTuple) = all(map(isapprox, x, y))

    for (sa, sb) in zip(scens, new_scens)
        @test isapprox_robust(sa.res1, sb.res1)
        @test isapprox_robust(sa.res2, sb.res2)
    end
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
    # each legacy scenario tested the operator with and without value
    function legacy_counts(scens)
        return Dict(
            k => 2 * count(s -> (DIT.operator(s), DIT.operator_place(s), DIT.function_place(s)) == k, scens)
                for k in unique(map(s -> (DIT.operator(s), DIT.operator_place(s), DIT.function_place(s)), scens))
        )
    end
    function new_counts(scens)
        key(ts) = (DIT.operator_symbol(ts), DIT.operator_place(ts), DIT.function_place(ts))
        return Dict(k => count(ts -> key(ts) == k, scens) for k in unique(map(key, scens)))
    end
    all_kwargs = (;
        include_closurified = true,
        include_constantified = true,
        include_cachified = true,
        include_constantorcachified = true,
        use_tuples = true,
    )
    for kwargs in [(;), all_kwargs]
        all_scens = default_scenarios(; kwargs...)
        scens = filter(!DIT.is_mismatch, all_scens)
        @test new_counts(scens) == legacy_counts(DIT.legacy_default_scenarios(; kwargs...))
        # one mismatch scenario for each scenario
        @test count(DIT.is_mismatch, all_scens) == length(scens)
    end
    # scenarios with smaller preparation inputs
    scens = default_scenarios(; include_normal = false, include_smaller = true, include_mismatched = false)
    @test new_counts(scens) ==
        legacy_counts(DIT.legacy_default_scenarios(; include_normal = false, include_smaller = true))
    @test all(ts -> !DIT.is_mismatch(ts), scens)
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
