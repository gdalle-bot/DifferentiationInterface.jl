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

Prepare the operator of `s` on its inputs.
"""
function prepare_scenario(s::Scenario, backend::AbstractADType; strict::Val)
    return preparator(operator(s))(
        func(s), value_args(s)..., backend, post_backend_args(s)...; strict
    )
end

"""
    reprepare_scenario(s::Scenario, old_prep, backend)

Adapt `old_prep` to the inputs of `s`.
"""
function reprepare_scenario(s::Scenario, old_prep, backend::AbstractADType)
    return preparator!(operator(s))(
        func(s), value_args(s)..., old_prep, backend, post_backend_args(s)...
    )
end

"""
    prepare_scenario_same_point(s::Scenario, backend)

Prepare the operator of `s` at the inputs of `s`, but with zero tangents.
"""
function prepare_scenario_same_point(s::Scenario, backend::AbstractADType)
    return preparator_same_point(operator(s))(
        func(s),
        value_args(s)...,
        backend,
        primal_inputs(s)...,
        custom_zero(tangents(s)),
        contexts(s)...,
    )
end

"""
    preparation_candidates(backend, ts::TripleScenario; reprepare)

Return a vector of tuples, each containing zero or one preparation result to test the operator with:

- no preparation;
- strict preparation on `prep`, possibly adapted to `exec`;
- non-strict preparation on `prep`, possibly adapted to `exec`;
- same-point preparation on `exec`, for operators that take tangents.
"""
function preparation_candidates(backend::AbstractADType, ts::TripleScenario; reprepare::Bool)
    prep_s, exec_s = prep_scenario(ts), exec_scenario(ts)
    op = operator(ts)
    preps = map((Val(true), Val(false))) do strict
        prep = prepare_scenario(prep_s, backend; strict)
        if reprepare && should_reprepare(ts)
            prep = reprepare_scenario(exec_s, prep, backend)
        end
        prep
    end
    if needs_tangent(op)
        preps = (preps..., prepare_scenario_same_point(exec_s, backend))
    end
    for prep in preps
        @test prep isa prep_type(op)
    end
    return [(), map(tuple, preps)...]
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

If `ts` is a mismatch scenario, test that executing with strict preparation throws a `PreparationMismatchError` instead.
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
    if is_mismatch(new_ts)
        return test_mismatch(backend, new_ts)
    end
    exec_s, flush_s = exec_scenario(new_ts), flush_scenario(new_ts)
    op = operator(new_ts)
    reference = result(exec_s)
    for prep_tup in preparation_candidates(backend, new_ts; reprepare)
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
    if scenario_intact
        @test protected_args(exec_s) == protected_args(exec_scenario(ts))
    end
    return nothing
end

"""
    test_mismatch(backend, ts::TripleScenario)

Test that preparing on the `prep` scenario of `ts` and executing on its `exec` scenario throws a `PreparationMismatchError`.
"""
function test_mismatch(backend::AbstractADType, ts::TripleScenario)
    prep = prepare_scenario(prep_scenario(ts), backend; strict = Val(true))
    @test prep isa prep_type(operator(ts))
    @test_throws PreparationMismatchError call_scenario(exec_scenario(ts), (prep,), backend)
    return nothing
end
