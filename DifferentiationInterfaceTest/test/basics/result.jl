module ResultTests

using DifferentiationInterface
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest: Result
using Test

include("counting_testset.jl")

approx_kwargs = (; isapprox, atol = 0, rtol = 1.0e-8)

@testset "Missing reference orders" begin
    @test isnothing(DIT.check_result(gradient, Result(; order1 = [1.0])))
    @test_throws ArgumentError DIT.check_result(value_and_gradient, Result(; order1 = [1.0]))
    @test_throws ArgumentError DIT.check_result(hessian, Result(; order1 = [1.0]))
    @test_throws ArgumentError DIT.test_approx(
        value_and_gradient, Result(1.0, [1.0], nothing), Result(; order1 = [1.0]);
        approx_kwargs...
    )
end

# number type which cannot be promoted with an expression, like `Tracker.TrackedReal`
struct Unpromotable <: Real
    x::Float64
end
Base.promote_rule(::Type{Unpromotable}, ::Type{Expr}) = error("cannot promote")
Base.isapprox(a::Unpromotable, b::Real; kwargs...) = isapprox(a.x, b; kwargs...)

@testset "Unpromotable numbers" begin
    reference = Result(1.0, [2.0], nothing)
    @test count_tests(
        () -> DIT.test_approx(
            value_and_gradient, Result(Unpromotable(1.0), [2.0], nothing), reference;
            approx_kwargs...
        )
    ) == (; passes = 2, failures = 0)
end

@testset "Only returned orders are compared" begin
    reference = Result(1.0, [2.0], nothing)
    wrong_value = Result(-1.0, [2.0], nothing)
    @test count_tests(() -> DIT.test_approx(gradient, wrong_value, reference; approx_kwargs...)) ==
        (; passes = 1, failures = 0)
    @test count_tests(
        () -> DIT.test_approx(value_and_gradient, wrong_value, reference; approx_kwargs...)
    ) == (; passes = 1, failures = 1)
end

@testset "Every returned order is compared" begin
    reference = Result(1.0, [2.0], [3.0;;])
    @test count_tests(
        () -> DIT.test_approx(
            value_gradient_and_hessian, reference, reference; approx_kwargs...
        )
    ) == (; passes = 3, failures = 0)
    for k in 0:2
        wrong = Result(
            k == 0 ? -1.0 : 1.0, k == 1 ? [-2.0] : [2.0], k == 2 ? [-3.0;;] : [3.0;;]
        )
        @test count_tests(
            () -> DIT.test_approx(
                value_gradient_and_hessian, wrong, reference; approx_kwargs...
            )
        ) == (; passes = 2, failures = 1)
    end
end

@testset "Tuples of tangents" begin
    reference = Result(; order1 = ([1.0], [2.0]))
    @test count_tests(
        () -> DIT.test_approx(pushforward, reference, reference; approx_kwargs...)
    ) == (; passes = 4, failures = 0)
    # one wrong tangent
    @test count_tests(
        () -> DIT.test_approx(
            pushforward, Result(; order1 = ([1.0], [-2.0])), reference; approx_kwargs...
        )
    ) == (; passes = 3, failures = 1)
    # missing tangent: no silent truncation
    @test count_tests(
        () -> DIT.test_approx(
            pushforward, Result(; order1 = ([1.0],)), reference; approx_kwargs...
        )
    ).failures == 1
    # extra tangent
    @test count_tests(
        () -> DIT.test_approx(
            pushforward, Result(; order1 = ([1.0], [2.0], [3.0])), reference;
            approx_kwargs...
        )
    ).failures == 1
    # not a tuple
    @test count_tests(
        () -> DIT.test_approx(pushforward, Result(; order1 = [1.0]), reference; approx_kwargs...)
    ).failures == 1
end

@testset "Same objects" begin
    grad, hess = [1.0], [1.0;;]
    input = DIT.result_from_input(value_gradient_and_hessian!, (), (grad, hess))
    @test input.order0 === nothing
    @test input.order1 === grad
    @test input.order2 === hess
    @test count_tests(
        () -> DIT.test_same(value_gradient_and_hessian!, Result(1.0, grad, hess), input)
    ) == (; passes = 2, failures = 0)
    @test count_tests(
        () -> DIT.test_same(value_gradient_and_hessian!, Result(1.0, copy(grad), hess), input)
    ) == (; passes = 1, failures = 1)

    y, ty = [1.0], ([2.0],)
    input = DIT.result_from_input(value_and_pushforward!, (y,), (ty,))
    @test input.order0 === y
    @test input.order1 === ty
    @test count_tests(
        () -> DIT.test_same(value_and_pushforward!, Result(copy(y), ty, nothing), input)
    ) == (; passes = 1, failures = 1)
    # the primal output is not returned without value
    input = DIT.result_from_input(pushforward!, (y,), (ty,))
    @test input.order0 === nothing
    # no buffers for out-of-place operators
    input = DIT.result_from_input(value_and_pushforward, (y,), ())
    @test input == Result(; order0 = y)
end

@testset "Equality" begin
    @test Result(1.0, [2.0], nothing) == Result(1.0, [2.0], nothing)
    @test Result(1.0, [2.0], nothing) != Result(1.0, [3.0], nothing)
    @test hash(Result(1.0, [2.0], nothing)) == hash(Result(1.0, [2.0], nothing))
    @test DIT.getorder(Result(1.0, 2.0, 3.0), 2) == 3.0
    @test_throws ArgumentError DIT.getorder(Result(), 3)
end

end
