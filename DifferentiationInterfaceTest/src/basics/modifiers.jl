#=
Hook-based scenario modifiers.

A modifier is a callable struct which transforms scenarios. It only implements the hooks it needs:

- `applies_to(m, s)`: whether `m` can modify the scenario `s`
- `modify_func(m, f, ref)`: new function, possibly using the primal output of the reference scenario `ref`
- `modify_contexts(m, contexts, ref)`: new contexts
- `modify_tangents(m, t)`: new tuple of tangents
- `modify_output(m, k, x)`: new output of order `k`, applied to reference results and to output buffers
- `name_suffix(m)`: suffix added to the scenario name

When applied to a `TripleScenario`, the function is modified once and shared between the three scenarios, and the contexts of the preparation and flush scenarios are zeroed.
=#

"""
    ScenarioModifier

Abstract supertype for scenario modifiers, which can be applied to a [`Scenario`](@ref), a [`TripleScenario`](@ref) or a vector of scenarios (keeping only those to which the modifier applies).
"""
abstract type ScenarioModifier end

applies_to(::ScenarioModifier, ::Scenario) = true
applies_to(m::ScenarioModifier, ts::TripleScenario) = applies_to(m, exec_scenario(ts))
modify_func(::ScenarioModifier, f, ::Scenario) = f
modify_contexts(::ScenarioModifier, contexts::Tuple, ::Scenario) = contexts
modify_tangents(::ScenarioModifier, t) = t
modify_output(::ScenarioModifier, k::Integer, x) = x
name_suffix(m::ScenarioModifier) = lowercase(string(nameof(typeof(m))))

function modify_scenario(
        m::ScenarioModifier, s::Scenario, ref::Scenario, new_func; zero_contexts::Bool
    )
    op = operator(s)
    new_value_args = map(y -> modify_output(m, 0, y), value_args(s))
    new_buffers = if is_inplace(op)
        map(
            (k, b) -> modify_output(m, k, b),
            filter(!iszero, returned_orders(op)),
            output_buffers(s),
        )
    else
        ()
    end
    t = tangents(s)
    new_tangents = isnothing(t) ? () : (modify_tangents(m, t),)
    new_contexts = modify_contexts(m, contexts(s), ref)
    if zero_contexts
        new_contexts = map(custom_zero, new_contexts)
    end
    r = result(s)
    new_result = if isnothing(r)
        nothing
    else
        Result(
            map((0, 1, 2)) do k
                v = getorder(r, k)
                isnothing(v) ? nothing : modify_output(m, k, v)
            end...,
        )
    end
    new_name = isnothing(name(s)) ? nothing : string(name(s), " [", name_suffix(m), "]")
    return Scenario(
        op,
        new_func,
        (new_value_args..., new_buffers...),
        (primal_inputs(s)..., new_tangents..., new_contexts...);
        result = new_result,
        name = new_name,
    )
end

function (m::ScenarioModifier)(s::Scenario)
    if !applies_to(m, s)
        throw(ArgumentError("Modifier `$m` does not apply to scenario `$s`"))
    end
    return modify_scenario(m, s, s, modify_func(m, func(s), s); zero_contexts = false)
end

function (m::ScenarioModifier)(ts::TripleScenario)
    exec = exec_scenario(ts)
    if !applies_to(m, exec)
        throw(ArgumentError("Modifier `$m` does not apply to scenario `$ts`"))
    end
    new_func = modify_func(m, func(exec), exec)
    function modify_other(s)
        other_func = func(s) === func(exec) ? new_func : modify_func(m, func(s), exec)
        return modify_scenario(m, s, exec, other_func; zero_contexts = true)
    end
    return TripleScenario(
        modify_scenario(m, exec, exec, new_func; zero_contexts = false);
        prep = modify_other(prep_scenario(ts)),
        flush = modify_other(flush_scenario(ts)),
        mismatch = is_mismatch(ts),
    )
end

function (m::ScenarioModifier)(scens::AbstractVector)
    return [m(s) for s in scens if applies_to(m, s)]
end

const AnyNewScenario = Union{Scenario, TripleScenario}

function_place(s::Scenario) = has_inplace_function(s) ? :in : :out
operator_place(s::Scenario) = is_inplace(operator(s)) ? :in : :out
function_place(ts::TripleScenario) = function_place(exec_scenario(ts))
operator_place(ts::TripleScenario) = operator_place(exec_scenario(ts))

## Zero derivatives

struct ZeroDer <: ScenarioModifier end

modify_output(::ZeroDer, k::Integer, x) = iszero(k) ? x : custom_zero(x)
name_suffix(::ZeroDer) = "zero derivative"

"""
    zero_der(s)

Return a scenario identical to `s` except that the reference derivatives (of order 1 and 2) are zero.

Useful to test backends which return zero derivatives, like `DifferentiationInterface.AutoZeroForward`.
"""
zero_der(s::AnyNewScenario) = ZeroDer()(s)
zero_der(scens::AbstractVector) = ZeroDer()(scens)

## Change function

struct ChangeFunction{F} <: ScenarioModifier
    f::F
end

modify_func(m::ChangeFunction, _, ::Scenario) = m.f
name_suffix(::ChangeFunction) = "new function"

change_function(s::AnyNewScenario, new_f) = ChangeFunction(new_f)(s)

## Batchify

struct Batchify <: ScenarioModifier end

function applies_to(::Batchify, s::Scenario)
    return needs_tangent(operator(s)) && length(tangents(s)) == 1
end
modify_tangents(::Batchify, t::Tuple) = (only(t), -only(t))
modify_output(::Batchify, k::Integer, x) = x
modify_output(::Batchify, k::Integer, x::Tuple) = (only(x), -only(x))
name_suffix(::Batchify) = "batchified"

batchify(s::AnyNewScenario) = Batchify()(s)
batchify(scens::AbstractVector) = Batchify()(scens)

## Closurify

struct Closurify <: ScenarioModifier
    a::Float64
    b::Vector{Float64}
end

Closurify() = Closurify(3.0, [4.0])

function applies_to(::Closurify, s::Scenario)
    return isempty(contexts(s)) && !isnothing(primal_output(s))
end
function modify_func(m::Closurify, f, ref::Scenario)
    x_buffer = [custom_zero(primal_input(ref))]
    y_buffer = [custom_zero(primal_output(ref))]
    return WritableClosure{function_place(ref)}(f, x_buffer, y_buffer, m.a, m.b)
end
modify_output(m::Closurify, k::Integer, x) = mymultiply(x, m.a + only(m.b))
name_suffix(::Closurify) = "closurified"

closurify(s::AnyNewScenario) = Closurify()(s)
closurify(scens::AbstractVector) = Closurify()(scens)

## Constantify

struct Constantify <: ScenarioModifier
    a::Float64
end

Constantify() = Constantify(3.0)

applies_to(::Constantify, s::Scenario) = isempty(contexts(s))
modify_func(::Constantify, f, ref::Scenario) = MultiplyByConstant{function_place(ref)}(f)
modify_contexts(m::Constantify, ::Tuple, ::Scenario) = (Constant(m.a),)
modify_output(m::Constantify, k::Integer, x) = mymultiply(x, m.a)
name_suffix(::Constantify) = "constantified"

constantify(s::AnyNewScenario) = Constantify()(s)
constantify(scens::AbstractVector) = Constantify()(scens)

## Cachify

struct Cachify <: ScenarioModifier
    use_tuples::Bool
end

function applies_to(::Cachify, s::Scenario)
    return isempty(contexts(s)) && !isnothing(primal_output(s))
end
modify_func(::Cachify, f, ref::Scenario) = StoreInCache{function_place(ref)}(f)
function modify_contexts(m::Cachify, ::Tuple, ref::Scenario)
    y = primal_output(ref)
    y_cache = y isa Number ? [zero(y)] : similar(y)
    if m.use_tuples
        return (Cache((; useful_cache = (y_cache,), useless_cache = similar(y_cache))),)
    else
        return (Cache(y_cache),)
    end
end
name_suffix(::Cachify) = "cachified"

cachify(s::AnyNewScenario; use_tuples::Bool) = Cachify(use_tuples)(s)
cachify(scens::AbstractVector; use_tuples::Bool) = Cachify(use_tuples)(scens)

## ConstantOrCachify

struct ConstantOrCachify <: ScenarioModifier
    a::Float64
    b::Vector{Float64}
end

ConstantOrCachify() = ConstantOrCachify(3.0, [4.0])

function applies_to(::ConstantOrCachify, s::Scenario)
    return isempty(contexts(s)) && !isnothing(primal_output(s))
end
function modify_func(::ConstantOrCachify, f, ref::Scenario)
    return MultiplyByConstantAndStoreInCache{function_place(ref)}(f)
end
function modify_contexts(m::ConstantOrCachify, ::Tuple, ref::Scenario)
    y = primal_output(ref)
    cache = y isa Number ? [zero(y)] : similar(y)
    return (ConstantOrCache((; cache, constant = (; m.a, b = copy(m.b)))),)
end
modify_output(m::ConstantOrCachify, k::Integer, x) = mymultiply(x, m.a + only(m.b))
name_suffix(::ConstantOrCachify) = "constantorcachified"

constantorcachify(s::AnyNewScenario) = ConstantOrCachify()(s)
constantorcachify(scens::AbstractVector) = ConstantOrCachify()(scens)

## Preparation mismatch

mismatched_eltype(x::Number) = Float32(x)
mismatched_eltype(x::AbstractArray) = map(mismatched_eltype, x)

"""
    mismatch(s; kind=:function)

Return a [`TripleScenario`](@ref) whose execution is incompatible with its (default) preparation, which must trigger a `DifferentiationInterface.PreparationMismatchError`.

- `kind = :function`: the execution function is `nothing`
- `kind = :input`: the execution input has a different element type

Preparation is done with the original function and input types, so that it does not need to be compiled again.
"""
function mismatch(s::Scenario; kind::Symbol = :function)
    new_func, new_post = if kind == :function
        nothing, post_backend_args(s)
    elseif kind == :input
        func(s),
            (mismatched_eltype(primal_input(s)), Base.tail(post_backend_args(s))...)
    else
        throw(ArgumentError("Unknown mismatch kind `$kind`"))
    end
    exec = Scenario(
        operator(s), new_func, pre_backend_args(s), new_post; result = result(s), name = name(s)
    )
    prep = zero_inputs(s)
    return TripleScenario(exec; prep, flush = prep, mismatch = true)
end

mismatch(ts::TripleScenario; kind::Symbol = :function) = mismatch(exec_scenario(ts); kind)

## Filters

"""
    has_matrix(s)

Return `true` if the primal input or output of `s` is a matrix.
"""
function has_matrix(s::Scenario)
    return primal_input(s) isa AbstractMatrix || primal_output(s) isa AbstractMatrix
end
has_matrix(ts::TripleScenario) = has_matrix(exec_scenario(ts))

## Results computed with a backend

"""
    compute_results(s, backend)

Return a scenario identical to `s` but whose reference result is computed with `backend`, using the operator with value in the family of `s`.
"""
function compute_results(s::Scenario, backend::AbstractADType)
    op = outofplace(with_value(operator(s)))
    new_s = deepcopy(s)
    output = op(
        func(new_s), map(mysimilar, value_args(new_s))..., backend, post_backend_args(new_s)...
    )
    return Scenario(
        operator(s),
        func(s),
        pre_backend_args(s),
        post_backend_args(s);
        result = result_from_output(op, output),
        name = name(s),
    )
end

function compute_results(ts::TripleScenario, backend::AbstractADType)
    return TripleScenario(
        compute_results(exec_scenario(ts), backend);
        prep = prep_scenario(ts),
        flush = flush_scenario(ts),
        mismatch = is_mismatch(ts),
    )
end
