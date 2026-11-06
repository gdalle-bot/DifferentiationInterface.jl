"""
    zero_inputs(s::Scenario)

Return a scenario with the same operator and function as `s`, where every argument except the output buffers is replaced by its [`custom_zero`](@ref).

The result is set to `nothing`, since the zeroed scenario is only used for preparation and flushing.
The function is shared with `s`, because preparation may be tied to the function object.
The output buffers are kept, because they are never read and their structure (e.g. a sparsity pattern) may matter.
"""
function zero_inputs(@nospecialize(s::Scenario))
    return Scenario(
        operator(s),
        func(s),
        (map(custom_zero, value_args(s))..., output_buffers(s)...),
        map(custom_zero, post_backend_args(s));
        result = nothing,
        name = nothing,
    )
end

"""
    TripleScenario

Store three scenarios sharing the same operator, to test preparation:

1. `prep`: the scenario on which preparation is performed;
2. `exec`: the scenario which is executed and compared to its reference result, before and after the flush;
3. `flush`: the scenario executed in between, to check that preparation does not keep stale state.

If `mismatch` is `true`, preparing on `prep` and executing on `exec` must throw a `DifferentiationInterface.PreparationMismatchError`.

# Constructors

    TripleScenario(exec::Scenario; prep=zero_inputs(exec), flush=zero_inputs(exec), mismatch=false)
    TripleScenario(ts::TripleScenario)

# Fields

$(TYPEDFIELDS)
"""
struct TripleScenario{S1 <: Scenario, S2 <: Scenario, S3 <: Scenario}
    "scenario used for preparation"
    prep::S1
    "scenario executed and compared to its reference result"
    exec::S2
    "scenario executed between two executions of `exec`"
    flush::S3
    "whether preparing on `prep` and executing on `exec` must throw a preparation mismatch error"
    mismatch::Bool

    function TripleScenario(
            @nospecialize(exec::Scenario);
            @nospecialize(prep::Scenario = zero_inputs(exec)),
            @nospecialize(flush::Scenario = zero_inputs(exec)),
            mismatch::Bool = false,
        )
        if isnothing(result(exec))
            throw(ArgumentError("The executed scenario must have a reference result"))
        end
        if !(operator(prep) === operator(exec) === operator(flush))
            throw(ArgumentError("The three scenarios must have the same operator"))
        end
        return new{typeof(prep), typeof(exec), typeof(flush)}(prep, exec, flush, mismatch)
    end
end

TripleScenario(ts::TripleScenario) = ts

prep_scenario(ts::TripleScenario) = ts.prep
exec_scenario(ts::TripleScenario) = ts.exec
flush_scenario(ts::TripleScenario) = ts.flush
is_mismatch(ts::TripleScenario) = ts.mismatch

operator(ts::TripleScenario) = operator(exec_scenario(ts))
name(ts::TripleScenario) = name(exec_scenario(ts))
"""
    func(ts::TripleScenario)

Return the function of `ts`, taken from its preparation scenario (the execution function of a mismatch scenario may be `nothing`).
"""
func(ts::TripleScenario) = func(prep_scenario(ts))
result(ts::TripleScenario) = result(exec_scenario(ts))
primal_input(ts::TripleScenario) = primal_input(exec_scenario(ts))
primal_output(ts::TripleScenario) = primal_output(exec_scenario(ts))

function Base.show(io::IO, ts::TripleScenario)
    show(io, exec_scenario(ts))
    is_mismatch(ts) && print(io, " [mismatch]")
    return nothing
end

"""
    should_reprepare(ts::TripleScenario)

Return `true` if the primal input or the primal output of the preparation scenario does not have the same size as in the execution scenario.
"""
function should_reprepare(ts::TripleScenario)
    prep, exec = prep_scenario(ts), exec_scenario(ts)
    x_prep, x_exec = primal_input(prep), primal_input(exec)
    y_prep, y_exec = map(mysize, value_args(prep)), map(mysize, value_args(exec))
    return !isequal(mysize(x_prep), mysize(x_exec)) || !isequal(y_prep, y_exec)
end

"""
    adapt_batchsize(backend, ts::TripleScenario)

Cap the batch size of `backend` (if it has one) so that it does not exceed the input or output dimensions of `ts`.
"""
function adapt_batchsize(backend::AbstractADType, ts::TripleScenario)
    prep, exec = prep_scenario(ts), exec_scenario(ts)
    x_prep, x_exec = primal_input(prep), primal_input(exec)
    y_exec = primal_output(exec)
    y_prep = has_inplace_function(prep) ? primal_output(prep) : y_exec
    Bmax = if x_exec isa AbstractArray && y_exec isa AbstractArray
        min(length(x_exec), length(y_exec), length(x_prep), length(y_prep))
    elseif x_exec isa AbstractArray
        min(length(x_exec), length(x_prep))
    elseif y_exec isa AbstractArray
        min(length(y_exec), length(y_prep))
    else
        typemax(Int)
    end
    return DI.threshold_batchsize(backend, Bmax)
end
