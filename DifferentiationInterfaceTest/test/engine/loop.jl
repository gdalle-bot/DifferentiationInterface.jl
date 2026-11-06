module LoopTests

using ADTypes
using DifferentiationInterface
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest:
    Scenario, TripleScenario, Result, @scenario, test_differentiation, FIRST_ORDER
using ForwardDiff: ForwardDiff
using SparseArrays
using SparseConnectivityTracer
using SparseMatrixColorings
using Test
using Zygote: Zygote

include("../basics/counting_testset.jl")

f_scalar(x) = sum(abs2, x)
f_vec!(y, x) = (y .= 2 .* x; nothing)

x = [1.0, 2.0]
dx = ([1.0, 0.0],)

scens = vcat(
    DIT.all_variants(@scenario value_and_gradient(f_scalar, missing, x) Result(5.0, 2x, nothing)),
    DIT.all_variants(
        @scenario value_gradient_and_hvp(f_scalar, missing, x, dx) Result(5.0, 2x, (2 .* dx))
    ),
    DIT.all_variants(
        @scenario value_and_jacobian(f_vec!, zeros(2), missing, x) Result(
            2x, sparse([2.0 0.0; 0.0 2.0]), nothing
        )
    ),
)

@testset "Call-based scenarios" begin
    counts = count_tests() do
        test_differentiation(AutoForwardDiff(), scens; logging = false)
    end
    @test counts.failures == 0
    @test counts.passes > 0

    # scenarios can be given as `TripleScenario`s, also in a mixed vector
    triple_counts = count_tests() do
        test_differentiation(
            AutoForwardDiff(),
            [TripleScenario(scens[1]), scens[2:end]...];
            logging = false,
        )
    end
    @test triple_counts == counts
end

@testset "Mixed with legacy scenarios" begin
    legacy = DIT.LegacyScenario{:gradient, :out}(f_scalar, x; res1 = 2x)
    counts_new = count_tests() do
        test_differentiation(AutoForwardDiff(), scens; logging = false)
    end
    counts_legacy = count_tests() do
        test_differentiation(AutoForwardDiff(), [legacy]; logging = false)
    end
    counts_mixed = count_tests() do
        test_differentiation(AutoForwardDiff(), [legacy, scens...]; logging = false)
    end
    @test counts_mixed.failures == 0
    @test counts_mixed.passes == counts_new.passes + counts_legacy.passes
end

@testset "Exclusion" begin
    counts_all = count_tests() do
        test_differentiation(AutoForwardDiff(), scens; logging = false)
    end
    counts_hvp = count_tests() do
        test_differentiation(AutoForwardDiff(), scens; excluded = FIRST_ORDER, logging = false)
    end
    counts_first = count_tests() do
        test_differentiation(AutoForwardDiff(), scens; excluded = [:hvp], logging = false)
    end
    @test 0 < counts_hvp.passes < counts_all.passes
    @test counts_hvp.passes + counts_first.passes == counts_all.passes
    counts_none = count_tests() do
        test_differentiation(
            AutoForwardDiff(), scens; excluded = [:gradient, :hvp, :jacobian], logging = false
        )
    end
    @test counts_none == (; passes = 0, failures = 0)
end

@testset "Compatibility" begin
    # no in-place functions with Zygote
    jacobian_scens = filter(s -> DIT.operator_symbol(DIT.operator(s)) == :jacobian, scens)
    counts = count_tests() do
        test_differentiation(AutoZygote(), jacobian_scens; logging = false)
    end
    @test counts == (; passes = 0, failures = 0)

    # sparse backends only for Jacobians and Hessians
    sparse_backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    counts_sparse = count_tests() do
        test_differentiation(sparse_backend, scens; logging = false)
    end
    counts_jacobian = count_tests() do
        test_differentiation(sparse_backend, jacobian_scens; logging = false)
    end
    @test counts_sparse.failures == 0
    @test counts_sparse == counts_jacobian

    # second-order backends only for second-order operators
    counts = count_tests() do
        test_differentiation(
            SecondOrder(AutoForwardDiff(), AutoForwardDiff()), scens; excluded = [:hvp],
            logging = false
        )
    end
    @test counts == (; passes = 0, failures = 0)
end

@testset "Batch size adaptation" begin
    # a chunk size larger than the input would error without adaptation
    counts = count_tests() do
        test_differentiation(AutoForwardDiff(; chunksize = 10), scens; logging = false)
    end
    @test counts.failures == 0
end

end
