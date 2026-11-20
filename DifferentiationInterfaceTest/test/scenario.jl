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
