#=
Hook-based scenario modifiers.

A modifier is a callable struct which transforms scenarios, by implementing only the hooks it needs (see their docstrings).
Every hook returns a new object and must not modify its arguments in-place.
=#

"""
    ScenarioModifier

Abstract supertype for scenario modifiers, which can be applied to a [`Scenario`](@ref), a [`TripleScenario`](@ref) or a vector of scenarios (keeping only those to which the modifier applies).

A modifier implements some of the following hooks, whose default versions leave the scenario unchanged:

- [`applies_to`](@ref)
- [`modify_func`](@ref)
- [`modify_contexts`](@ref)
- [`modify_inputs`](@ref)
- [`modify_tangents`](@ref)
- [`modify_output`](@ref)
- [`modify_buffer`](@ref)
- [`name_suffix`](@ref)

Each hook receives the reference scenario `ref`, which is the scenario being modified, or the execution scenario when modifying the preparation and flush scenarios of a `TripleScenario`.
When applied to a `TripleScenario`, the function is modified once and shared between the three scenarios (unless they had different functions to begin with), and the contexts of the preparation and flush scenarios are zeroed.
"""
abstract type ScenarioModifier end

"""
    applies_to(m::ScenarioModifier, ref::Scenario)

Return `true` if `m` can modify the scenario `ref` (`true` by default).
"""
applies_to(::ScenarioModifier, ::Scenario) = true
applies_to(m::ScenarioModifier, ts::TripleScenario) = applies_to(m, exec_scenario(ts))

"""
    modify_func(m::ScenarioModifier, f, ref::Scenario)

Return the new function replacing `f` (`f` itself by default).
"""
modify_func(::ScenarioModifier, @nospecialize(f), @nospecialize(::Scenario)) = f

"""
    modify_contexts(m::ScenarioModifier, contexts::Tuple, ref::Scenario)

Return the new tuple of contexts replacing `contexts` (`contexts` itself by default).
"""
modify_contexts(::ScenarioModifier, @nospecialize(contexts::Tuple), @nospecialize(::Scenario)) = contexts

"""
    modify_inputs(m::ScenarioModifier, xs::Tuple, ref::Scenario)

Return the new tuple of primal inputs replacing `xs` (`xs` itself by default).
"""
modify_inputs(::ScenarioModifier, @nospecialize(xs::Tuple), @nospecialize(::Scenario)) = xs

"""
    modify_tangents(m::ScenarioModifier, t::Tuple, ref::Scenario)

Return the new tuple of tangents replacing `t`, for operators which take tangents (`t` itself by default).
"""
modify_tangents(::ScenarioModifier, t::Tuple, @nospecialize(::Scenario)) = t

"""
    modify_output(m::ScenarioModifier, k::Integer, x, ref::Scenario)

Return the new reference output of order `k` replacing `x` in the result of the scenario (`x` itself by default).
"""
modify_output(::ScenarioModifier, k::Integer, x, @nospecialize(::Scenario)) = x

"""
    modify_buffer(m::ScenarioModifier, k::Integer, x, ref::Scenario)

Return the new buffer of order `k` replacing `x` (`x` itself by default), where the buffer of order `0` is the primal output `y` of an in-place function.

Buffers are never read, so only modifiers which change their structure need this hook.
"""
modify_buffer(::ScenarioModifier, k::Integer, x, @nospecialize(::Scenario)) = x

"""
    name_suffix(m::ScenarioModifier)

Return the suffix appended to the name of modified scenarios (the lowercase name of the type of `m` by default).
"""
name_suffix(m::ScenarioModifier) = lowercase(string(nameof(typeof(m))))

function modify_scenario(
        @nospecialize(m::ScenarioModifier),
        @nospecialize(s::Scenario),
        @nospecialize(ref::Scenario),
        @nospecialize(new_func);
        zero_contexts::Bool,
    )
    op = operator(s)
    new_value_args = map(y -> modify_buffer(m, 0, y, ref), value_args(s))
    new_buffers = if is_inplace(op)
        map(
            (k, b) -> modify_buffer(m, k, b, ref),
            filter(!iszero, returned_orders(op)),
            output_buffers(s),
        )
    else
        ()
    end
    t = tangents(s)
    new_tangents = isnothing(t) ? () : (modify_tangents(m, t, ref),)
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
                isnothing(v) ? nothing : modify_output(m, k, v, ref)
            end...,
        )
    end
    new_name = isnothing(name(s)) ? nothing : string(name(s), " [", name_suffix(m), "]")
    return Scenario(
        op,
        new_func,
        (new_value_args..., new_buffers...),
        (modify_inputs(m, primal_inputs(s), ref)..., new_tangents..., new_contexts...);
        result = new_result,
        name = new_name,
    )
end

function (m::ScenarioModifier)(@nospecialize(s::Scenario))
    if !applies_to(m, s)
        throw(ArgumentError("Modifier `$m` does not apply to scenario `$s`"))
    end
    return modify_scenario(m, s, s, modify_func(m, func(s), s); zero_contexts = false)
end

function (m::ScenarioModifier)(@nospecialize(ts::TripleScenario))
    exec = exec_scenario(ts)
    if !applies_to(m, exec)
        throw(ArgumentError("Modifier `$m` does not apply to scenario `$ts`"))
    end
    new_func = modify_func(m, func(exec), exec)
    function modify_other(s::Scenario)
        # preparation may be tied to the function object (e.g. a closure with buffers):
        # if `s` shares its function with `exec`, it must share the new function too,
        # otherwise its own function is modified separately
        other_func = func(s) === func(exec) ? new_func : modify_func(m, func(s), exec)
        # `exec` is the reference because only it is guaranteed to know its primal output,
        # and the contexts are zeroed because preparation must not depend on their values
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

modify_output(::ZeroDer, k::Integer, x, @nospecialize(::Scenario)) = iszero(k) ? x : custom_zero(x)
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

# the `nothing` function of mismatch scenarios is kept, so that they remain mismatched
modify_func(m::ChangeFunction, @nospecialize(f), @nospecialize(::Scenario)) = isnothing(f) ? f : m.f
name_suffix(::ChangeFunction) = "new function"

change_function(s::AnyNewScenario, new_f) = ChangeFunction(new_f)(s)

## Batchify

struct Batchify <: ScenarioModifier end

function applies_to(::Batchify, @nospecialize(s::Scenario))
    return needs_tangent(operator(s)) && length(tangents(s)) == 1
end
modify_tangents(::Batchify, t::Tuple, @nospecialize(::Scenario)) = (only(t), -only(t))
# only the highest order is a tuple of tangents, a primal output could be a tuple too
function modify_output(::Batchify, k::Integer, x, @nospecialize(ref::Scenario))
    return k == order(operator(ref)) ? (only(x), -only(x)) : x
end
modify_buffer(m::Batchify, k::Integer, x, @nospecialize(ref::Scenario)) = modify_output(m, k, x, ref)
name_suffix(::Batchify) = "batchified"

batchify(s::AnyNewScenario) = Batchify()(s)
batchify(scens::AbstractVector) = Batchify()(scens)

## Closurify

struct Closurify <: ScenarioModifier
    a::Float64
    b::Vector{Float64}
end

Closurify() = Closurify(3.0, [4.0])

function applies_to(::Closurify, @nospecialize(s::Scenario))
    return isempty(contexts(s)) && !isnothing(primal_output(s))
end
function modify_func(m::Closurify, @nospecialize(f), @nospecialize(ref::Scenario))
    x_buffer = [custom_zero(primal_input(ref))]
    y_buffer = [custom_zero(primal_output(ref))]
    return WritableClosure{function_place(ref)}(f, x_buffer, y_buffer, m.a, m.b)
end
modify_output(m::Closurify, k::Integer, x, @nospecialize(::Scenario)) = mymultiply(x, m.a + only(m.b))
name_suffix(::Closurify) = "closurified"

closurify(s::AnyNewScenario) = Closurify()(s)
closurify(scens::AbstractVector) = Closurify()(scens)

## Constantify

struct Constantify <: ScenarioModifier
    a::Float64
end

Constantify() = Constantify(3.0)

applies_to(::Constantify, @nospecialize(s::Scenario)) = isempty(contexts(s))
modify_func(::Constantify, @nospecialize(f), @nospecialize(ref::Scenario)) = MultiplyByConstant{function_place(ref)}(f)
modify_contexts(m::Constantify, @nospecialize(::Tuple), @nospecialize(::Scenario)) = (Constant(m.a),)
modify_output(m::Constantify, k::Integer, x, @nospecialize(::Scenario)) = mymultiply(x, m.a)
name_suffix(::Constantify) = "constantified"

constantify(s::AnyNewScenario) = Constantify()(s)
constantify(scens::AbstractVector) = Constantify()(scens)

## Cachify

struct Cachify <: ScenarioModifier
    use_tuples::Bool
end

function applies_to(::Cachify, @nospecialize(s::Scenario))
    return isempty(contexts(s)) && !isnothing(primal_output(s))
end
modify_func(::Cachify, @nospecialize(f), @nospecialize(ref::Scenario)) = StoreInCache{function_place(ref)}(f)
function modify_contexts(m::Cachify, @nospecialize(::Tuple), @nospecialize(ref::Scenario))
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

function applies_to(::ConstantOrCachify, @nospecialize(s::Scenario))
    return isempty(contexts(s)) && !isnothing(primal_output(s))
end
function modify_func(::ConstantOrCachify, @nospecialize(f), @nospecialize(ref::Scenario))
    return MultiplyByConstantAndStoreInCache{function_place(ref)}(f)
end
function modify_contexts(m::ConstantOrCachify, @nospecialize(::Tuple), @nospecialize(ref::Scenario))
    y = primal_output(ref)
    cache = y isa Number ? [zero(y)] : similar(y)
    return (ConstantOrCache((; cache, constant = (; m.a, b = copy(m.b)))),)
end
modify_output(m::ConstantOrCachify, k::Integer, x, @nospecialize(::Scenario)) = mymultiply(x, m.a + only(m.b))
name_suffix(::ConstantOrCachify) = "constantorcachified"

constantorcachify(s::AnyNewScenario) = ConstantOrCachify()(s)
constantorcachify(scens::AbstractVector) = ConstantOrCachify()(scens)

## Map arrays

struct MapArrays{F, I, M} <: ScenarioModifier
    func::F
    immutable::I
    mutable::M
end

map_array(f, x::AbstractArray) = f(x)
map_array(f, x::Union{Tuple, NamedTuple}) = map(y -> map_array(f, y), x)
map_array(f, x) = x

map_context(m::MapArrays, c::Constant) = Constant(map_array(m.immutable, unwrap(c)))
map_context(m::MapArrays, c::Cache) = Cache(map_array(m.mutable, unwrap(c)))
map_context(m::MapArrays, c::ConstantOrCache) = ConstantOrCache(map_array(m.mutable, unwrap(c)))

modify_func(m::MapArrays, @nospecialize(f), @nospecialize(::Scenario)) = m.func(f)
function modify_inputs(m::MapArrays, @nospecialize(xs::Tuple), @nospecialize(::Scenario))
    return map_array(m.immutable, xs)
end
modify_tangents(m::MapArrays, t::Tuple, @nospecialize(::Scenario)) = map_array(m.immutable, t)
function modify_contexts(m::MapArrays, @nospecialize(contexts::Tuple), @nospecialize(::Scenario))
    return map(c -> map_context(m, c), contexts)
end
modify_output(m::MapArrays, k::Integer, x, @nospecialize(::Scenario)) = map_array(m.immutable, x)
modify_buffer(m::MapArrays, k::Integer, x, @nospecialize(::Scenario)) = map_array(m.mutable, x)
name_suffix(::MapArrays) = "mapped arrays"

"""
    map_arrays(immutable, mutable; func=identity)

Return a scenario modifier which converts every array in a scenario:

- `immutable` is applied to primal inputs, tangents, constant contexts and reference results;
- `mutable` is applied to the primal output of in-place functions, output buffers and caches;
- `func` is applied to the function.

Useful to test array types defined in other packages, for instance static or GPU arrays.
"""
map_arrays(immutable, mutable; func = identity) = MapArrays(func, immutable, mutable)

## Preparation mismatch

mismatched_eltype(x::Float32) = Float64(x)
mismatched_eltype(x::Real) = Float32(x)
mismatched_eltype(x::Complex) = complex(mismatched_eltype(real(x)), mismatched_eltype(imag(x)))
mismatched_eltype(x::AbstractArray) = map(mismatched_eltype, x)

"""
    mismatch(s; kind=:function)

Return a [`TripleScenario`](@ref) whose execution is incompatible with its (default) preparation, which must trigger a `DifferentiationInterface.PreparationMismatchError`.

- `kind = :function`: the execution function is `nothing`
- `kind = :input`: the execution input has a different element type (`Float32` instead of `Float64` and vice versa)

Preparation is done with the original function and input types, so that it does not need to be compiled again.
"""
function mismatch(@nospecialize(s::Scenario); kind::Symbol = :function)
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
function compute_results(@nospecialize(s::Scenario), backend::AbstractADType)
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
