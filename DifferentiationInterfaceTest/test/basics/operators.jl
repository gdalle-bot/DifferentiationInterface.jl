module OperatorsTests

using DifferentiationInterface
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest: Result
using ForwardDiff: ForwardDiff
using Test

@testset "Operator traits" begin
    @test length(DIT.ALL_OPERATORS) == 32
    @test allunique(DIT.ALL_OPERATORS)
    @test length(unique(map(DIT.standard, DIT.ALL_OPERATORS))) == 8

    @testset "$op" for op in DIT.ALL_OPERATORS
        std = DIT.standard(op)
        name = string(nameof(op))

        @test DIT.standard(std) === std
        @test !DIT.has_value(std)
        @test !DIT.is_inplace(std)
        @test DIT.operator_symbol(op) === nameof(std)
        @test DIT.order(op) == (DIT.operator_symbol(op) in DIT.SECOND_ORDER ? 2 : 1)

        # variants stay in the same family
        for variant in (DIT.with_value, DIT.without_value, DIT.inplace, DIT.outofplace)
            @test DIT.standard(variant(op)) === std
        end
        @test DIT.has_value(DIT.with_value(op))
        @test !DIT.has_value(DIT.without_value(op))
        @test DIT.is_inplace(DIT.inplace(op))
        @test !DIT.is_inplace(DIT.outofplace(op))
        @test DIT.is_inplace(DIT.with_value(op)) == DIT.is_inplace(op)
        @test DIT.has_value(DIT.inplace(op)) == DIT.has_value(op)

        # traits agree with names
        @test endswith(name, "!") == DIT.is_inplace(op)
        @test startswith(name, "value") == DIT.has_value(op)
        @test DIT.needs_tangent(op) == (DIT.operator_symbol(op) in (:pushforward, :pullback, :hvp))

        # preparation
        std_name = string(nameof(std))
        @test DIT.preparator(op) === getfield(DI, Symbol("prepare_", std_name))
        @test DIT.preparator!(op) === getfield(DI, Symbol("prepare!_", std_name))
        if DIT.needs_tangent(op)
            @test DIT.preparator_same_point(op) ===
                getfield(DI, Symbol("prepare_", std_name, "_same_point"))
        else
            @test isnothing(DIT.preparator_same_point(op))
        end
        @test DIT.prep_type(op) <: DI.Prep

        # outputs
        orders = DIT.returned_orders(op)
        @test issorted(orders)
        @test (0 in orders) == DIT.has_value(op)
        @test DIT.order(op) in orders
        @test DIT.n_buffers(op) == (DIT.is_inplace(op) ? count(!iszero, orders) : 0)
    end
end

## Check traits against actual DI calls

f_numvec(x::Number) = [sin(x), cos(2x)]
f_scalar(x::AbstractArray) = sum(abs2, x) + prod(x)
f_vec(x::AbstractArray) = abs2.(x) .+ x[1]
f_vec!(y, x) = (y .= f_vec(x); nothing)

x_num = 0.5
x_arr = [1.0, 2.0, 3.0]
tx_in = ([1.0, -1.0, 0.0], [0.0, 2.0, 1.0])
tx_out = ([1.0, 0.0, 0.5], [3.0, -1.0, 1.0])

calls = [
    (pushforward, f_vec, (), (x_arr, tx_in)),
    (pullback, f_vec, (), (x_arr, tx_out)),
    (derivative, f_numvec, (), (x_num,)),
    (gradient, f_scalar, (), (x_arr,)),
    (jacobian, f_vec, (), (x_arr,)),
    (second_derivative, f_numvec, (), (x_num,)),
    (hvp, f_scalar, (), (x_arr, tx_in)),
    (hessian, f_scalar, (), (x_arr,)),
    # in-place functions
    (pushforward, f_vec!, (similar(x_arr),), (x_arr, tx_in)),
    (pullback, f_vec!, (similar(x_arr),), (x_arr, tx_out)),
    (jacobian, f_vec!, (similar(x_arr),), (x_arr,)),
]

@testset "Operator outputs" begin
    backend = AutoForwardDiff()
    @testset "$std - $f" for (std, f, value_args, post) in calls
        val_op = DIT.with_value(std)
        reference = DIT.result_from_output(
            val_op, val_op(f, map(copy, value_args)..., backend, post...)
        )
        variants = (std, DIT.with_value(std), DIT.inplace(std), DIT.with_value(DIT.inplace(std)))
        for op in variants
            @test isnothing(DIT.check_result(op, reference))
            new_value_args = map(copy, value_args)
            buffers = DIT.make_buffers(op, reference)
            @test length(buffers) == DIT.n_buffers(op)
            output = op(f, new_value_args..., buffers..., backend, post...)
            computed = DIT.result_from_output(op, output)
            DIT.test_approx(op, computed, reference; isapprox, atol = 0, rtol = 1.0e-10)
            # includes the primal output of in-place functions, if it is returned
            DIT.test_same(op, computed, DIT.result_from_input(op, new_value_args, buffers))
        end
    end
end

@testset "Output arity" begin
    @test_throws ArgumentError DIT.result_from_output(value_and_gradient, 1.0)
    @test_throws ArgumentError DIT.result_from_output(
        value_gradient_and_hessian, (1.0, [1.0])
    )
    @test DIT.result_from_output(gradient, [1.0]) == Result(; order1 = [1.0])
    @test DIT.result_from_output(hessian!, [1.0;;]) == Result(; order2 = [1.0;;])
end

end
