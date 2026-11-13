"""
    Result(order0, order1, order2)
    Result(; order0=nothing, order1=nothing, order2=nothing)

Reference outputs of a differentiation operator, sorted by differentiation order.

| operator family     | `order0` | `order1`                       | `order2`               |
|:------------------- |:-------- |:------------------------------ |:---------------------- |
| `pushforward`       | value    | tuple of output tangents       | -                      |
| `pullback`          | value    | tuple of input cotangents      | -                      |
| `derivative`        | value    | derivative                     | -                      |
| `gradient`          | value    | gradient                       | -                      |
| `jacobian`          | value    | Jacobian                       | -                      |
| `second_derivative` | value    | derivative                     | second derivative      |
| `hvp`               | value    | gradient                       | tuple of HVPs          |
| `hessian`           | value    | gradient                       | Hessian                |

A `Result` may contain more orders than a given operator returns (e.g. the value for `gradient`): only the orders returned by the operator are compared.
However, every order returned by the operator must be provided.

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct Result{O0, O1, O2}
    "primal value"
    order0::O0 = nothing
    "first-order derivative"
    order1::O1 = nothing
    "second-order derivative"
    order2::O2 = nothing
end

function Base.:(==)(r1::Result, r2::Result)
    return r1.order0 == r2.order0 && r1.order1 == r2.order1 && r1.order2 == r2.order2
end

Base.hash(r::Result, h::UInt) = hash(r.order2, hash(r.order1, hash(r.order0, hash(:Result, h))))

function getorder(r::Result, k::Integer)
    if k == 0
        return r.order0
    elseif k == 1
        return r.order1
    elseif k == 2
        return r.order2
    else
        throw(ArgumentError("Differentiation order must be 0, 1 or 2, got $k"))
    end
end

function result_from_orders(orders::NTuple{N, Integer}, values::NTuple{N, Any}) where {N}
    function get_value(k)
        i = findfirst(==(k), orders)
        return isnothing(i) ? nothing : values[i]
    end
    return Result(get_value(0), get_value(1), get_value(2))
end

"""
    check_result(op, r::Result)

Throw an `ArgumentError` if `r` lacks one of the orders returned by `op`.
"""
function check_result(op, r::Result)
    for k in returned_orders(op)
        if isnothing(getorder(r, k))
            throw(
                ArgumentError(
                    "Operator `$op` returns order $k but the reference `Result` does not contain it",
                ),
            )
        end
    end
    return nothing
end

"""
    result_from_output(op, output)

Sort the `output` of `op` into a [`Result`](@ref).
"""
function result_from_output(op, output)
    orders = returned_orders(op)
    values = length(orders) == 1 ? (output,) : output
    if !(values isa Tuple) || length(values) != length(orders)
        throw(
            ArgumentError(
                "Operator `$op` should return $(length(orders)) objects, got $(typeof(output))",
            ),
        )
    end
    return result_from_orders(orders, values)
end

"""
    result_from_input(op, value_args, buffers)

Sort the inputs of `op` which should be returned as outputs into a [`Result`](@ref).

- `value_args` contains the primal output `y` for in-place functions `f!(y, x)`, and is empty otherwise.
- `buffers` contains the output buffers of in-place operators, and is empty otherwise.
"""
function result_from_input(op, value_args::Tuple, buffers::Tuple)
    @assert length(buffers) == n_buffers(op)
    @assert length(value_args) <= 1
    orders = filter(!iszero, returned_orders(op))
    r = is_inplace(op) ? result_from_orders(orders, buffers) : Result()
    if has_value(op) && !isempty(value_args)
        r = Result(only(value_args), r.order1, r.order2)
    end
    return r
end

"""
    test_approx(op, computed::Result, reference::Result; isapprox, atol, rtol)

Test that every order returned by `op` is approximately equal between `computed` and `reference`.

Tuples (of tangents) are compared element by element, after checking that their lengths match.
"""
function test_approx(
        op, computed::Result, reference::Result; isapprox::Function, atol::Real, rtol::Real
    )
    check_result(op, reference)
    for k in returned_orders(op)
        compare_approx(getorder(computed, k), getorder(reference, k); isapprox, atol, rtol)
    end
    return nothing
end

function compare_approx(computed, reference; isapprox::Function, atol::Real, rtol::Real)
    @test isapprox(computed, reference; atol, rtol)
    return nothing
end

function compare_approx(computed, reference::Tuple; isapprox::Function, atol::Real, rtol::Real)
    @test computed isa Tuple
    computed isa Tuple || return nothing
    @test length(computed) == length(reference)
    for i in 1:min(length(computed), length(reference))
        compare_approx(computed[i], reference[i]; isapprox, atol, rtol)
    end
    return nothing
end

"""
    test_same(op, output::Result, input::Result)

Test that every order returned by `op` which was given as an input (see [`result_from_input`](@ref)) is returned as the same object.
"""
function test_same(op, output::Result, input::Result)
    for k in returned_orders(op)
        in_k = getorder(input, k)
        isnothing(in_k) && continue
        @test getorder(output, k) === in_k
    end
    return nothing
end
