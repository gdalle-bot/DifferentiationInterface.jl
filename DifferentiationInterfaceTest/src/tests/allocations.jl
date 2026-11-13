function test_noallocs(skip::Bool, func, args...)
    possible_allocations = check_allocs(func, typeof.(args); ignore_throw = true)
    return @test isempty(possible_allocations) skip = skip
end

"""
    prepared_calls(backend, ts::TripleScenario)

Return the preparation call, unprepared call and prepared call of `ts` as `(f, args)` pairs, with a deep copy of `ts` and fresh output buffers.

Preparation is adapted to the execution inputs when their sizes differ.
"""
function prepared_calls(backend::AbstractADType, ts::TripleScenario)
    new_ts = deepcopy(ts)
    prep_s, exec_s = prep_scenario(new_ts), exec_scenario(new_ts)
    op = operator(exec_s)
    preparation = (
        preparator(op),
        (func(prep_s), value_args(prep_s)..., backend, post_backend_args(prep_s)...),
    )
    prep = prepare_scenario(prep_s, backend; strict = Val(true))
    if should_reprepare(new_ts)
        prep = reprepare_scenario(exec_s, prep, backend)
    end
    pre = map(mysimilar, pre_backend_args(exec_s))
    post = post_backend_args(exec_s)
    unprepared = (op, (func(exec_s), pre..., backend, post...))
    prepared = (op, (func(exec_s), pre..., prep, backend, post...))
    return (; preparation, unprepared, prepared)
end

"""
    test_alloccheck(backend, ts::TripleScenario; subset, skip)

Check with AllocCheck.jl that the calls of `ts` cannot allocate: the prepared operator (if `subset` is `:prepared` or `:full`), the unprepared operator and the preparation (if `subset` is `:full`).
"""
function test_alloccheck(backend::AbstractADType, ts::TripleScenario; subset::Symbol, skip::Bool)
    calls = prepared_calls(backend, ts)
    if subset == :full
        test_noallocs(skip, first(calls.preparation), last(calls.preparation)...)
        test_noallocs(skip, first(calls.unprepared), last(calls.unprepared)...)
    end
    if subset in (:prepared, :full)
        test_noallocs(skip, first(calls.prepared), last(calls.prepared)...)
    end
    return nothing
end
