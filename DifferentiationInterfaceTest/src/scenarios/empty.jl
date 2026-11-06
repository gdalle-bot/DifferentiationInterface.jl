make_empty(t::Number) = typeof(t)[]
function make_empty!(y::AbstractArray, t::Number)
    @assert isempty(y)
    return nothing
end

"""
    empty_scenarios(; include_mismatched=true)

Create a vector of [`TripleScenario`](@ref)s with empty inputs or outputs, for out-of-place operators.
"""
function empty_scenarios(; include_mismatched = true)
    canonical = [
        canonical_scenario(value_and_derivative, make_empty, 1.0; res1 = Float64[]),
        canonical_scenario!(value_and_derivative, make_empty!, Float64[], 1.0; res1 = Float64[]),
        canonical_scenario(value_and_gradient, sum, Float64[]; res1 = Float64[]),
        canonical_scenario(value_and_jacobian, copy, Float64[]; res1 = float.(I(0))),
        canonical_scenario!(value_and_jacobian, copyto!, Float64[], Float64[]; res1 = float.(I(0))),
    ]
    scens = TripleScenario[
        TripleScenario(v) for s in canonical for v in place_variants(s; inplace_op = false)
    ]
    include_mismatched && add_mismatched!(scens)
    return scens
end
