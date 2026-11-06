module ChecksTests

using ADTypes
import Chairmarks
using DataFrames: DataFrame
using DifferentiationInterface
using DifferentiationInterface: AutoZeroForward
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest:
    Scenario, TripleScenario, Result, @scenario, test_differentiation,
    benchmark_differentiation
using ForwardDiff: ForwardDiff
import JET
using Test

include("../basics/counting_testset.jl")

f_scalar(x) = sum(abs2, x)
f_num(x::Number) = 2x

const UNSTABLE = Ref{Any}(1.0)
f_unstable(x) = sum(abs2, x) * UNSTABLE[]

const STORE = Float64[]
f_alloc(x::Number) = (push!(STORE, x); 2x)  # may allocate when resizing

struct FakeBackend <: ADTypes.AbstractADType end
ADTypes.mode(::FakeBackend) = ADTypes.ForwardMode()

x = [1.0, 2.0]

stable_scens = DIT.all_variants(
    @scenario value_and_gradient(f_scalar, missing, x) Result(5.0, 2x, nothing)
)
unstable_scens = DIT.all_variants(
    @scenario value_and_gradient(f_unstable, missing, x) Result(5.0, 2x, nothing)
)

# type stability checks are disabled on Julia >= 1.12 in the rest of the test suite
if VERSION < v"1.12-"
    @testset "Type stability" begin
        # a fixed chunk size is necessary for type-stable preparation
        for subset in (:prepared, :full)
            counts = count_tests() do
                test_differentiation(
                    AutoForwardDiff(; chunksize = 2), stable_scens; correctness = false,
                    type_stability = subset, logging = false
                )
            end
            @test counts.failures == 0
            @test counts.passes == length(stable_scens) * (subset == :full ? 3 : 1)

            counts = count_tests() do
                test_differentiation(
                    AutoForwardDiff(; chunksize = 2), unstable_scens; correctness = false,
                    type_stability = subset, logging = false
                )
            end
            @test counts.failures > 0
        end
    end
end

@testset "Allocations" begin
    # the value is needed to make sure that the function is called
    s = @scenario value_and_derivative(f_num, missing, 1.0) Result(2.0, 2.0, nothing)
    counts = count_tests() do
        test_differentiation(
            AutoZeroForward(), [s]; correctness = false, allocations = :prepared,
            logging = false
        )
    end
    @test counts == (; passes = 1, failures = 0)

    s = @scenario value_and_derivative(f_alloc, missing, 1.0) Result(2.0, 2.0, nothing)
    counts = count_tests() do
        test_differentiation(
            AutoZeroForward(), [s]; correctness = false, allocations = :prepared,
            logging = false
        )
    end
    @test counts == (; passes = 0, failures = 1)
end

@testset "Benchmark" begin
    data_prepared = benchmark_differentiation(
        AutoForwardDiff(), stable_scens; benchmark = :prepared, benchmark_seconds = 0.01,
        logging = false
    ) |> DataFrame
    @test size(data_prepared, 1) == length(stable_scens)
    @test all(data_prepared[!, :prepared])
    @test Set(data_prepared[!, :operator]) ==
        Set([:gradient, :gradient!, :value_and_gradient, :value_and_gradient!])

    data_full = benchmark_differentiation(
        AutoForwardDiff(), stable_scens; benchmark = :full, benchmark_seconds = 0.01,
        logging = false
    ) |> DataFrame
    @test size(data_full, 1) == 3 * length(stable_scens)
    @test count(isnothing, data_full[!, :prepared]) == length(stable_scens)
    @test count(==(:prepare_gradient), data_full[!, :operator]) == length(stable_scens)
    @test all(>=(0), data_full[!, :calls])
    # operators call the function, preparation may not
    @test all(row -> isnothing(row.prepared) || row.calls > 0, eachrow(data_full))
    @test all(>(0), data_full[!, :time])
    @test all(s -> s isa TripleScenario, data_full[!, :scenario])

    # benchmark errors are recorded as NaN
    data_fake = benchmark_differentiation(
        FakeBackend(), stable_scens; benchmark_test = false, logging = false
    ) |> DataFrame
    @test all(isnan, data_fake[!, :time])
    @test all(==(-1), data_fake[!, :calls])
end

@testset "Mismatch scenarios are only tested for correctness" begin
    s = @scenario value_and_gradient(f_scalar, missing, x) Result(5.0, 2x, nothing)
    mismatch = TripleScenario(
        s; prep = Scenario(value_and_gradient, f_unstable, (), (x,)), mismatch = true
    )
    data = benchmark_differentiation(
        AutoForwardDiff(), [mismatch]; benchmark_seconds = 0.01, logging = false
    ) |> DataFrame
    @test size(data, 1) == 0
end

end
