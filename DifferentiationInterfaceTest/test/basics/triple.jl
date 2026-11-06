module TripleTests

using ADTypes
using DifferentiationInterface
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest: Scenario, TripleScenario, Result, @scenario
using ForwardDiff: ForwardDiff
using Test

f_scalar(x, c) = c * sum(abs2, x)
f_vec!(y, x) = (y .= 2 .* x; nothing)

struct WithBuffer
    buffer::Vector{Float64}
end
(w::WithBuffer)(x) = sum(abs2, x)

@testset "Default preparation" begin
    x, c = [1.0, 2.0], Constant(3.0)
    exec = @scenario value_and_gradient(f_scalar, missing, x, c) Result(15.0, [6.0, 12.0], nothing)
    ts = TripleScenario(exec)
    @test DIT.exec_scenario(ts) === exec
    @test !DIT.is_mismatch(ts)
    @test DIT.operator(ts) === value_and_gradient
    for s in (DIT.prep_scenario(ts), DIT.flush_scenario(ts))
        @test DIT.operator(s) === value_and_gradient
        @test isnothing(DIT.result(s))
        @test DIT.primal_input(s) == [0.0, 0.0]
        @test DIT.primal_input(s) !== x
        @test only(DIT.contexts(s)) isa Constant
        @test DI.unwrap(only(DIT.contexts(s))) == 0.0
    end
    # the original scenario is left untouched
    @test x == [1.0, 2.0]
    @test !DIT.should_reprepare(ts)
    @test TripleScenario(ts) === ts

    # output buffers are kept as is
    g = [0.0, 0.0]
    ts_buf = TripleScenario(@scenario gradient!(f_scalar, g, missing, x, c) Result(; order1 = 6x))
    @test only(DIT.output_buffers(DIT.prep_scenario(ts_buf))) === g
    @test string(ts) == string(exec)

    # the function is shared, also after a deep copy
    w = WithBuffer([0.0])
    ts = TripleScenario(@scenario gradient(w, missing, x) Result(; order1 = 2x))
    @test DIT.func(DIT.prep_scenario(ts)) === w
    @test DIT.func(DIT.flush_scenario(ts)) === w
    ts_copy = deepcopy(ts)
    @test DIT.func(DIT.prep_scenario(ts_copy)) === DIT.func(DIT.exec_scenario(ts_copy))
    @test DIT.func(DIT.prep_scenario(ts_copy)) !== w
end

@testset "Custom preparation" begin
    x, y = [1.0, 2.0, 3.0], zeros(3)
    jac = [2.0 0 0; 0 2.0 0; 0 0 2.0]
    exec = @scenario value_and_jacobian(f_vec!, y, missing, x) Result(2x, jac, nothing)
    prep = @scenario value_and_jacobian(f_vec!, zeros(2), missing, zeros(2))
    ts = TripleScenario(exec; prep)
    @test DIT.should_reprepare(ts)
    # the flush scenario has the size of the execution scenario
    @test size(DIT.primal_input(DIT.flush_scenario(ts))) == size(x)

    prep_y = @scenario value_and_jacobian(f_vec!, zeros(2), missing, zeros(3))
    @test DIT.should_reprepare(TripleScenario(exec; prep = prep_y))

    @test string(TripleScenario(exec; mismatch = true)) == string(exec) * " [mismatch]"
end

@testset "Validation" begin
    x = [1.0, 2.0]
    @test_throws ArgumentError TripleScenario(@scenario gradient(sum, missing, x))
    exec = @scenario gradient(sum, missing, x) Result(; order1 = [1.0, 1.0])
    @test_throws ArgumentError TripleScenario(
        exec; prep = @scenario value_and_gradient(sum, missing, x)
    )
    @test_throws ArgumentError TripleScenario(
        exec; flush = @scenario gradient!(sum, [0.0, 0.0], missing, x)
    )
end

@testset "Batch size adaptation" begin
    backend = AutoForwardDiff(; chunksize = 100)
    x = [1.0, 2.0, 3.0]
    exec = @scenario gradient(sum, missing, x) Result(6.0, ones(3), nothing)
    @test DIT.adapt_batchsize(backend, TripleScenario(exec)) == AutoForwardDiff(; chunksize = 3)
    prep = @scenario gradient(sum, missing, zeros(2))
    @test DIT.adapt_batchsize(backend, TripleScenario(exec; prep)) ==
        AutoForwardDiff(; chunksize = 2)
    exec = @scenario derivative(x -> [x, x], missing, 1.0) Result([1.0, 1.0], [1.0, 1.0], nothing)
    @test DIT.adapt_batchsize(backend, TripleScenario(exec)) == AutoForwardDiff(; chunksize = 2)
    exec = @scenario derivative(sin, missing, 1.0) Result(sin(1.0), cos(1.0), nothing)
    @test DIT.adapt_batchsize(backend, TripleScenario(exec)) == backend
end

end
