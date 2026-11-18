#=
Generic correctness tests for call-based scenarios.
=#

"""
    call_scenario(s::Scenario, prep_tup::Tuple, backend)

Call the operator of `s` with fresh copies of its pre-backend arguments, and with the preparation result in `prep_tup` (if it is not empty).

Return the output and the tuples of value arguments and buffers that were passed.
"""
function call_scenario(s::Scenario, prep_tup::Tuple, backend::AbstractADType)
    value_in = map(mysimilar, value_args(s))
    buffers_in = map(mysimilar, output_buffers(s))
    output = operator(s)(
        func(s), value_in..., buffers_in..., prep_tup..., backend, post_backend_args(s)...
    )
    return output, value_in, buffers_in
end

"""
    prepare_scenario(s::Scenario, backend; strict)

Prepare the operator of `s` on its inputs, with fresh copies of its value arguments.
"""
function prepare_scenario(s::Scenario, backend::AbstractADType; strict::Val)
    return preparator(operator(s))(
        func(s), map(mysimilar, value_args(s))..., backend, post_backend_args(s)...; strict
    )
end

"""
    reprepare_scenario(s::Scenario, old_prep, backend)

Adapt `old_prep` to the inputs of `s`, with fresh copies of its value arguments.
"""
function reprepare_scenario(s::Scenario, old_prep, backend::AbstractADType)
    return preparator!(operator(s))(
        func(s), map(mysimilar, value_args(s))..., old_prep, backend, post_backend_args(s)...
    )
end

"""
    prepare_scenario_same_point(s::Scenario, backend)

Prepare the operator of `s` at the inputs of `s`, with fresh copies of its value arguments.
"""
function prepare_scenario_same_point(s::Scenario, backend::AbstractADType)
    return preparator_same_point(operator(s))(
        func(s),
        map(mysimilar, value_args(s))...,
        backend,
        primal_inputs(s)...,
        tangents(s),
        contexts(s)...,
    )
end

"""
    zero_tangents(s::Scenario)

Return a copy of `s` where the tangents (if any) are replaced by their [`custom_zero`](@ref), which ensures that (re)preparation does not depend on their values.
"""
function zero_tangents(s::Scenario)
    needs_tangent(operator(s)) || return s
    x, t, rest... = post_backend_args(s)
    return Scenario(
        operator(s), func(s), pre_backend_args(s), (x, custom_zero(t), rest...); result = nothing
    )
end

"""
    preparation_candidates(backend, ts::TripleScenario; reprepare)

Return a tuple of pairs `kind => prep_tup`, where `prep_tup` contains zero or one preparation result to test the operator with:

- `:unprepared`: no preparation;
- `:strict`: strict preparation on `prep`, possibly adapted to `exec`;
- `:nonstrict`: non-strict preparation on `prep`, possibly adapted to `exec`;
- `:same_point`: same-point preparation on `exec`, for operators that take tangents.

Tangents are zeroed for repreparation and same-point preparation (see [`zero_tangents`](@ref)).
Mismatch scenarios are never reprepared, since repreparation would remove the mismatch.
"""
function preparation_candidates(backend::AbstractADType, ts::TripleScenario; reprepare::Bool)
    prep_s, exec_s = prep_scenario(ts), zero_tangents(exec_scenario(ts))
    op = operator(ts)
    do_reprepare = reprepare && !is_mismatch(ts) && should_reprepare(ts)
    strict_prep, nonstrict_prep = map((Val(true), Val(false))) do strict
        prep = prepare_scenario(prep_s, backend; strict)
        do_reprepare ? reprepare_scenario(exec_s, prep, backend) : prep
    end
    candidates = (
        :unprepared => (), :strict => (strict_prep,), :nonstrict => (nonstrict_prep,),
    )
    if needs_tangent(op)
        candidates = (
            candidates..., :same_point => (prepare_scenario_same_point(exec_s, backend),),
        )
    end
    foreach(candidates) do (_, prep_tup)
        foreach(prep -> @test(prep isa prep_type(op)), prep_tup)
    end
    return candidates
end

sparsity_pattern(A::AbstractSparseMatrix) = (findnz(A)[1:2], size(A))

"""
    test_sparsity(backend, op, computed::Result, reference::Result)

For sparse backends and sparse reference Jacobians or Hessians, test that the computed matrix is sparse and has the same structural pattern as the reference.
"""
function test_sparsity(backend::AbstractADType, op, computed::Result, reference::Result)
    backend isa AutoSparse || return nothing
    operator_symbol(op) in (:jacobian, :hessian) || return nothing
    k = order(op)
    ref = getorder(reference, k)
    ref isa AbstractSparseMatrix || return nothing
    comp = getorder(computed, k)
    @test comp isa AbstractSparseMatrix
    comp isa AbstractSparseMatrix || return nothing
    @test sparsity_pattern(comp) == sparsity_pattern(ref)
    return nothing
end

"""
    test_correctness(backend, ts::TripleScenario; isapprox, atol, rtol, scenario_intact, reprepare)

Test the operator of `ts` with every preparation candidate (see [`preparation_candidates`](@ref)).

For each candidate, `exec` is executed twice, with a call on `flush` in between when the operator is prepared.
Each execution is compared to the reference result, and output buffers must be returned as is.

If `ts` is a mismatch scenario, executing with the strict preparation must throw a `PreparationMismatchError` instead, and the non-strict preparation is skipped (its behavior is undefined).
"""
function test_correctness(
        backend::AbstractADType,
        ts::TripleScenario;
        isapprox::Function,
        atol::Real,
        rtol::Real,
        scenario_intact::Bool,
        reprepare::Bool,
        sparsity::Bool = false,  # automatic for call-based scenarios
    )
    new_ts = deepcopy(ts)
    exec_s, flush_s = exec_scenario(new_ts), flush_scenario(new_ts)
    op = operator(new_ts)
    reference = result(exec_s)
    if isnothing(reference)
        throw(ArgumentError("Correctness tests require a reference result for the executed scenario"))
    end
    mismatch = is_mismatch(new_ts)
    foreach(preparation_candidates(backend, new_ts; reprepare)) do (kind, prep_tup)
        if mismatch && kind === :strict
            @test_throws PreparationMismatchError call_scenario(exec_s, prep_tup, backend)
        elseif !(mismatch && kind === :nonstrict)
            out1, value_in1, buffers_in1 = call_scenario(exec_s, prep_tup, backend)
            isempty(prep_tup) || call_scenario(flush_s, prep_tup, backend)
            out2, value_in2, buffers_in2 = call_scenario(exec_s, prep_tup, backend)
            for (out, value_in, buffers_in) in
                ((out1, value_in1, buffers_in1), (out2, value_in2, buffers_in2))
                computed = result_from_output(op, out)
                test_approx(op, computed, reference; isapprox, atol, rtol)
                test_same(op, computed, result_from_input(op, value_in, buffers_in))
                test_sparsity(backend, op, computed, reference)
            end
        end
    end
    if scenario_intact
        @test protected_args(exec_s) == protected_args(exec_scenario(ts))
    end
    return nothing
end
