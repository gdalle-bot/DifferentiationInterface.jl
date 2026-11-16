module VariantsTests

using DifferentiationInterface
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest: Scenario, Result, @scenario
using ForwardDiff: ForwardDiff
using Test

f_scalar(x) = sum(abs2, x)
f_vec!(y, x) = (y .= 2 .* x; nothing)

x = [1.0, 2.0]
y = zeros(2)
dx = ([1.0, 0.0], [0.0, 1.0])

canonical_scenarios = [
    (@scenario value_and_gradient(f_scalar, missing, x) Result(5.0, 2x, nothing)),
    (@scenario value_gradient_and_hessian(f_scalar, missing, x) Result(5.0, 2x, [2.0 0; 0 2])),
    (@scenario value_gradient_and_hvp(f_scalar, missing, x, dx) Result(5.0, 2x, (2 .* dx))),
    (@scenario value_and_pushforward(f_vec!, y, missing, x, dx) Result(2x, (2 .* dx), nothing)),
    (@scenario value_and_jacobian(f_vec!, y, missing, x) Result(2x, [2.0 0; 0 2], nothing)),
    # scalar outputs
    (@scenario value_and_derivative(sin, missing, 1.0) Result(sin(1.0), cos(1.0), nothing)),
    (
        @scenario value_derivative_and_second_derivative(sin, missing, 1.0) Result(
            sin(1.0), cos(1.0), -sin(1.0)
        )
    ),
]

@testset "All variants of $s" for s in canonical_scenarios
    # scalar outputs cannot be overwritten
    include_inplace = !(DIT.result(s).order1 isa Number)
    variants = DIT.all_variants(s; include_inplace)
    std = DIT.standard(DIT.operator(s))
    @test allunique(map(DIT.operator, variants))
    @test all(v -> DIT.standard(DIT.operator(v)) === std, variants)
    @test length(variants) == (include_inplace ? 4 : 2)
    @test !any(v -> DIT.is_inplace(DIT.operator(v)), DIT.all_variants(s; include_inplace = false))
    for v in variants
        @test DIT.result(v) === DIT.result(s)
        @test DIT.func(v) === DIT.func(s)
        @test DIT.post_backend_args(v) === DIT.post_backend_args(s)
        # the primal output of in-place functions is copied
        @test DIT.value_args(v) == DIT.value_args(s)
        @test all(map(!==, DIT.value_args(v), DIT.value_args(s)))
        @test length(DIT.output_buffers(v)) == DIT.n_buffers(DIT.operator(v))
        # the variant can actually be executed
        op = DIT.operator(v)
        output = op(
            DIT.func(v), DIT.pre_backend_args(v)..., AutoForwardDiff(), DIT.post_backend_args(v)...
        )
        DIT.test_approx(
            op, DIT.result_from_output(op, output), DIT.result(v); isapprox, atol = 0, rtol = 1.0e-8
        )
    end
end

@testset "Variant round trips" begin
    s = canonical_scenarios[2]
    @test DIT.operator(DIT.inplace(s)) === value_gradient_and_hessian!
    @test DIT.operator(DIT.without_value(DIT.inplace(s))) === hessian!
    @test length(DIT.output_buffers(DIT.without_value(DIT.inplace(s)))) == 1
    @test DIT.operator(DIT.with_value(DIT.outofplace(DIT.inplace(s)))) ===
        value_gradient_and_hessian
end

@testset "Errors" begin
    s = @scenario gradient(f_scalar, missing, x) Result(; order1 = 2x)
    # the value is missing from the result
    @test_throws ArgumentError DIT.with_value(s)
    @test_throws ArgumentError DIT.all_variants(s; include_inplace = false)
    @test_throws ArgumentError DIT.change_operator(s, hessian)
    @test_throws ArgumentError DIT.inplace(@scenario gradient(f_scalar, missing, x))
end

end
