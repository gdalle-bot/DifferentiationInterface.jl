"""
    Scenario

Store a test scenario: a call to a DifferentiationInterface operator, together with its reference result.

A scenario represents the call

    operator(func, pre_backend_args..., backend, post_backend_args...)

for any `backend`, and it is easiest to create with the [`@scenario`](@ref) macro.

# Constructors

    Scenario(operator, func, pre_backend_args, post_backend_args; result=nothing, name=nothing)
    Scenario(; operator, func, pre_backend_args=(), post_backend_args, result=nothing, name=nothing)

The arguments are checked against the conventions of DifferentiationInterface:

- `pre_backend_args` contains the primal output `y` (only for in-place functions `f!(y, x)`), followed by the output buffers (only for in-place operators).
- `post_backend_args` contains the primal input `x`, the tuple of tangents (only for `pushforward`, `pullback` and `hvp`) and the contexts.
- `result` must contain every differentiation order returned by `operator`, see [`Result`](@ref). It can only be `nothing` for scenarios which are not executed, such as the preparation part of a [`TripleScenario`](@ref).

# Fields

$(TYPEDFIELDS)
"""
struct Scenario{O, F, A1 <: Tuple, A2 <: Tuple, R <: Union{Result, Nothing}}
    "DifferentiationInterface operator, e.g. `value_and_gradient!`"
    operator::O
    "function to differentiate"
    func::F
    "arguments between the function and the backend"
    pre_backend_args::A1
    "arguments after the backend"
    post_backend_args::A2
    "reference result of the operator"
    result::R
    "name for display in test sets and benchmark tables"
    name::Union{String, Nothing}

    function Scenario(
            operator::O,
            func::F,
            pre_backend_args::A1,
            post_backend_args::A2;
            result::R = nothing,
            name::Union{AbstractString, Nothing} = nothing,
        ) where {O, F, A1 <: Tuple, A2 <: Tuple, R <: Union{Result, Nothing}}
        check_scenario(operator, pre_backend_args, post_backend_args, result)
        return new{O, F, A1, A2, R}(
            operator,
            func,
            pre_backend_args,
            post_backend_args,
            result,
            isnothing(name) ? nothing : String(name),
        )
    end
end

function Scenario(;
        operator, func, pre_backend_args = (), post_backend_args, result = nothing, name = nothing
    )
    return Scenario(operator, func, pre_backend_args, post_backend_args; result, name)
end

const INPLACE_FUNCTION_FAMILIES = (:pushforward, :pullback, :derivative, :jacobian)

function check_scenario(op, pre_backend_args::Tuple, post_backend_args::Tuple, result)
    if !(op in ALL_OPERATORS)
        throw(ArgumentError("`$op` is not a supported DifferentiationInterface operator"))
    end
    nb = n_buffers(op)
    if !(nb <= length(pre_backend_args) <= nb + 1)
        throw(
            ArgumentError(
                "Operator `$op` takes $nb output buffer(s) before the backend, plus `y` for in-place functions, but got $(length(pre_backend_args)) argument(s)",
            ),
        )
    end
    if length(pre_backend_args) == nb + 1 && !(operator_symbol(op) in INPLACE_FUNCTION_FAMILIES)
        throw(ArgumentError("Operator `$op` does not support in-place functions"))
    end
    n_active = needs_tangent(op) ? 2 : 1
    if length(post_backend_args) < n_active
        throw(
            ArgumentError(
                "Operator `$op` needs at least $n_active argument(s) after the backend, got $(length(post_backend_args))",
            ),
        )
    end
    if needs_tangent(op) && !(post_backend_args[2] isa NTuple)
        throw(ArgumentError("Operator `$op` needs a tuple of tangents after the input `x`"))
    end
    for c in post_backend_args[(n_active + 1):end]
        if !(c isa Context)
            throw(ArgumentError("Extra arguments must be contexts, got $(typeof(c))"))
        end
    end
    !isnothing(result) && check_result(op, result)
    return nothing
end

## Accessors

operator(s::Scenario) = s.operator
func(s::Scenario) = s.func
pre_backend_args(s::Scenario) = s.pre_backend_args
post_backend_args(s::Scenario) = s.post_backend_args
result(s::Scenario) = s.result
name(s::Scenario) = s.name

"""
    has_inplace_function(s::Scenario)

Return `true` if the function of `s` has the form `f!(y, x)`.
"""
has_inplace_function(s::Scenario) = length(pre_backend_args(s)) > n_buffers(operator(s))

"""
    value_args(s::Scenario)

Return the tuple `(y,)` for in-place functions `f!(y, x)`, and `()` otherwise.
"""
value_args(s::Scenario) = has_inplace_function(s) ? (first(pre_backend_args(s)),) : ()

"""
    output_buffers(s::Scenario)

Return the tuple of output buffers of the in-place operator in `s`, or `()` for out-of-place operators.
"""
function output_buffers(s::Scenario)
    return has_inplace_function(s) ? Base.tail(pre_backend_args(s)) : pre_backend_args(s)
end

"""
    primal_inputs(s::Scenario)

Return the tuple of active primal inputs of `s` (for now, there is only one).
"""
primal_inputs(s::Scenario) = (first(post_backend_args(s)),)

"""
    primal_input(s::Scenario)

Return the only active primal input of `s`, or throw an error if there are several.
"""
function primal_input(s::Scenario)
    xs = primal_inputs(s)
    if length(xs) != 1
        throw(
            ArgumentError(
                "This operation is only supported with a single active argument, because it is unclear how to resize several ones",
            ),
        )
    end
    return only(xs)
end

"""
    tangents(s::Scenario)

Return the tuple of tangents of `s`, or `nothing` if its operator does not need them.
"""
tangents(s::Scenario) = needs_tangent(operator(s)) ? post_backend_args(s)[2] : nothing

"""
    contexts(s::Scenario)

Return the tuple of contexts of `s`.
"""
function contexts(s::Scenario)
    post = post_backend_args(s)
    return needs_tangent(operator(s)) ? Base.tail(Base.tail(post)) : Base.tail(post)
end

"""
    primal_output(s::Scenario)

Return the primal output `y` of `s` if it is known, and `nothing` otherwise.
"""
function primal_output(s::Scenario)
    if has_inplace_function(s)
        return only(value_args(s))
    elseif !isnothing(result(s))
        return result(s).order0
    else
        return nothing
    end
end

"""
    protected_args(s::Scenario)

Return the arguments of `s` which should not be modified by the operator: the primal input, the tangents and the contexts except caches.
"""
function protected_args(s::Scenario)
    return (
        primal_inputs(s)...,
        something(tangents(s), ()),
        filter(c -> !(c isa Union{Cache, ConstantOrCache}), contexts(s))...,
    )
end

## Display

function Base.show(io::IO, s::Scenario)
    if !isnothing(name(s))
        print(io, name(s))
    else
        print(io, nameof(operator(s)), "(", func(s))
        for a in pre_backend_args(s)
            print(io, ", ::", typeof(a))
        end
        print(io, ", backend")
        for a in post_backend_args(s)
            print(io, ", ::", typeof(a))
        end
        print(io, ")")
    end
    return nothing
end

## Macro

"""
    @scenario [name] operator(func, pre_backend_args..., missing, post_backend_args...) [result]

Create a [`Scenario`](@ref) by writing the operator call it represents, with `missing` in place of the backend.

# Example

```julia
@scenario "my scenario" value_and_gradient(sum, missing, [1.0, 2.0]) Result(3.0, [1.0, 1.0], nothing)
```
"""
macro scenario(args...)
    return scenario_expr(args...)
end

is_backend_placeholder(a) = a === :missing

function is_scenario_call(ex)
    return Meta.isexpr(ex, :call) && length(ex.args) >= 3 &&
        any(is_backend_placeholder, ex.args[3:end])
end

function scenario_expr(args...)
    i = findfirst(is_scenario_call, args)
    if isnothing(i)
        throw(
            ArgumentError(
                "`@scenario` expects an operator call with `missing` in place of the backend, e.g. `gradient(f, missing, x)`",
            ),
        )
    elseif i > 2 || length(args) - i > 1
        throw(ArgumentError("`@scenario` expects the syntax `@scenario [name] call [result]`"))
    end
    name = i == 2 ? args[1] : nothing
    result = length(args) > i ? args[i + 1] : nothing
    call = args[i]
    if any(a -> Meta.isexpr(a, (:parameters, :kw)), call.args)
        throw(ArgumentError("`@scenario` does not support keyword arguments in the operator call"))
    end
    op, f, rest = call.args[1], call.args[2], call.args[3:end]
    b = findall(is_backend_placeholder, rest)
    if length(b) != 1
        throw(ArgumentError("`@scenario` expects exactly one `missing` in place of the backend"))
    end
    pre = rest[1:(only(b) - 1)]
    post = rest[(only(b) + 1):end]
    return :(
        Scenario(
            $(esc(op)),
            $(esc(f)),
            ($(map(esc, pre)...),),
            ($(map(esc, post)...),);
            result = $(esc(result)),
            name = $(esc(name)),
        )
    )
end
