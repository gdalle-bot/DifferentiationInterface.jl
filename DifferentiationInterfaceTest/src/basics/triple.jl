"""
    zero_inputs(s::Scenario)

Return a scenario with the same operator and function as `s`, where every argument except the output buffers is replaced by its [`custom_zero`](@ref).

The result is set to `nothing`, since the zeroed scenario is only used for preparation and flushing.
The function is shared with `s`, because preparation may be tied to the function object.
The output buffers are kept, because they are never read and their structure (e.g. a sparsity pattern) may matter.
"""
function zero_inputs(s::Scenario)
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
2. `exec`: the scenario which is executed (and compared to its reference result, if any), before and after the flush;
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
    "scenario executed (and compared to its reference result, if any)"
    exec::S2
    "scenario executed between two executions of `exec`"
    flush::S3
    "whether preparing on `prep` and executing on `exec` must throw a preparation mismatch error"
    mismatch::Bool

    function TripleScenario(
            exec::S2;
            prep::S1 = zero_inputs(exec),
            flush::S3 = zero_inputs(exec),
            mismatch::Bool = false,
        ) where {S1 <: Scenario, S2 <: Scenario, S3 <: Scenario}
        if !(operator(prep) === operator(exec) === operator(flush))
            throw(ArgumentError("The three scenarios must have the same operator"))
        end
        return new{S1, S2, S3}(prep, exec, flush, mismatch)
    end
end

TripleScenario(ts::TripleScenario) = ts

prep_scenario(ts::TripleScenario) = ts.prep
exec_scenario(ts::TripleScenario) = ts.exec
flush_scenario(ts::TripleScenario) = ts.flush
is_mismatch(ts::TripleScenario) = ts.mismatch

operator(ts::TripleScenario) = operator(exec_scenario(ts))
name(ts::TripleScenario) = name(exec_scenario(ts))

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
    xsize_prep, xsize_exec = mysize(primal_input(prep)), mysize(primal_input(exec))
    ysizes_prep, ysizes_exec = map(mysize, value_args(prep)), map(mysize, value_args(exec))
    return !isequal(xsize_prep, xsize_exec) || !isequal(ysizes_prep, ysizes_exec)
end

"""
    adapt_batchsize(backend, ts::TripleScenario)

Cap the batch size of `backend` (if it has one) so that it does not exceed the lengths of the array inputs and known array outputs of `ts`.
"""
function adapt_batchsize(backend::AbstractADType, ts::TripleScenario)
    prep, exec = prep_scenario(ts), exec_scenario(ts)
    # primal outputs are `nothing` when unknown
    primals = (primal_input(prep), primal_input(exec), primal_output(prep), primal_output(exec))
    Bmax = minimum(length, Iterators.filter(p -> p isa AbstractArray, primals); init = typemax(Int))
    return DI.threshold_batchsize(backend, Bmax)
end
