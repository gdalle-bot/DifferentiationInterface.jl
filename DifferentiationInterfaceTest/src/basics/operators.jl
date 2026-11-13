#=
Traits of DifferentiationInterface operators, defined on the operator functions themselves.

Each operator family (e.g. gradient) has four variants:

- `op` (out-of-place, without value), e.g. `gradient`
- `op!` (in-place, without value), e.g. `gradient!`
- `val_op` (out-of-place, with value), e.g. `value_and_gradient`
- `val_op!` (in-place, with value), e.g. `value_and_gradient!`
=#

"""
    standard(op)

Return the out-of-place operator without value in the family of `op`, e.g. `gradient` for `value_and_gradient!`.
"""
function standard end

"""
    order(op)

Return the differentiation order of `op` (`1` or `2`).
"""
function order end

"""
    needs_tangent(op)

Return `true` if `op` takes a tuple of tangents as input (pushforward, pullback, HVP).
"""
function needs_tangent end

"""
    has_value(op)

Return `true` if `op` also returns the primal value.
"""
function has_value end

"""
    is_inplace(op)

Return `true` if `op` overwrites output buffers.
"""
function is_inplace end

"""
    with_value(op)

Return the variant of `op` which also returns the primal value.
"""
function with_value end

"""
    without_value(op)

Return the variant of `op` which does not return the primal value.
"""
function without_value end

"""
    inplace(op)

Return the variant of `op` which overwrites output buffers.
"""
function inplace end

"""
    outofplace(op)

Return the variant of `op` which allocates its outputs.
"""
function outofplace end

"""
    preparator(op)

Return the preparation function for `op`, e.g. `prepare_gradient`.
"""
function preparator end

"""
    preparator!(op)

Return the re-preparation function for `op`, e.g. `prepare!_gradient`.
"""
function preparator! end

"""
    preparator_same_point(op)

Return the same-point preparation function for `op`, e.g. `prepare_pushforward_same_point`, or `nothing` if it does not exist.
"""
function preparator_same_point end

"""
    prep_type(op)

Return the abstract type of preparation results for `op`, e.g. `GradientPrep`.
"""
function prep_type end

const OPERATOR_FAMILIES = [
    # (op, val_op, order, needs_tangent, prep type)
    (:pushforward, :value_and_pushforward, 1, true, :PushforwardPrep),
    (:pullback, :value_and_pullback, 1, true, :PullbackPrep),
    (:derivative, :value_and_derivative, 1, false, :DerivativePrep),
    (:gradient, :value_and_gradient, 1, false, :GradientPrep),
    (:jacobian, :value_and_jacobian, 1, false, :JacobianPrep),
    (
        :second_derivative,
        :value_derivative_and_second_derivative,
        2,
        false,
        :SecondDerivativePrep,
    ),
    (:hvp, :value_gradient_and_hvp, 2, true, :HVPPrep),
    (:hessian, :value_gradient_and_hessian, 2, false, :HessianPrep),
]

for (op, val_op, ord, tang, P) in OPERATOR_FAMILIES
    op! = Symbol(op, "!")
    val_op! = Symbol(val_op, "!")
    prep_op = Symbol("prepare_", op)
    prep_op! = Symbol("prepare!_", op)
    prep_op_same = tang ? Symbol("prepare_", op, "_same_point") : :nothing
    family = :(Union{typeof($op), typeof($op!), typeof($val_op), typeof($val_op!)})

    @eval begin
        standard(::$family) = $op
        order(::$family) = $ord
        needs_tangent(::$family) = $tang
        preparator(::$family) = $prep_op
        preparator!(::$family) = $prep_op!
        preparator_same_point(::$family) = $prep_op_same
        prep_type(::$family) = $P

        has_value(::Union{typeof($op), typeof($op!)}) = false
        has_value(::Union{typeof($val_op), typeof($val_op!)}) = true

        is_inplace(::Union{typeof($op), typeof($val_op)}) = false
        is_inplace(::Union{typeof($op!), typeof($val_op!)}) = true

        with_value(::Union{typeof($op), typeof($val_op)}) = $val_op
        with_value(::Union{typeof($op!), typeof($val_op!)}) = $val_op!

        without_value(::Union{typeof($op), typeof($val_op)}) = $op
        without_value(::Union{typeof($op!), typeof($val_op!)}) = $op!

        inplace(::Union{typeof($op), typeof($op!)}) = $op!
        inplace(::Union{typeof($val_op), typeof($val_op!)}) = $val_op!

        outofplace(::Union{typeof($op), typeof($op!)}) = $op
        outofplace(::Union{typeof($val_op), typeof($val_op!)}) = $val_op
    end
end

"""
    ALL_OPERATORS

Tuple of all DifferentiationInterface operators supported by call-based scenarios.
"""
const ALL_OPERATORS = Tuple(
    Iterators.flatten(
        map(OPERATOR_FAMILIES) do (op, val_op)
            map(
                s -> getfield(@__MODULE__, s),
                (op, Symbol(op, "!"), val_op, Symbol(val_op, "!")),
            )
        end,
    ),
)

"""
    operator_symbol(op)

Return the name of the operator family of `op` as a `Symbol`, e.g. `:gradient` for `value_and_gradient!`.

This is used to filter operators with the `excluded` keyword of [`test_differentiation`](@ref).
"""
operator_symbol(op) = nameof(standard(op))

"""
    returned_orders(op)

Return the tuple of differentiation orders returned by `op`, in the order in which they are returned (order `0` is the primal value).
"""
function returned_orders(op)
    if has_value(op)
        return order(op) == 1 ? (0, 1) : (0, 1, 2)
    else
        return (order(op),)
    end
end

"""
    n_buffers(op)

Return the number of output buffers taken by `op` (zero if it is out-of-place).

The primal value is never written to a buffer: for in-place functions `f!(y, x)`, `y` is passed separately before the buffers.
"""
n_buffers(op) = is_inplace(op) ? count(!iszero, returned_orders(op)) : 0
