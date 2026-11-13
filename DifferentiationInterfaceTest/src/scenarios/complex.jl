square_only(x::AbstractVector) = only(x)^2
abs2_only(x::AbstractVector) = abs2(only(x))

function complex_gradient_canonical(f, x::AbstractVector; grad)
    dy = 1.0
    return [
        canonical_scenario(value_and_gradient, f, x; res1 = grad) => true,
        canonical_scenario(value_and_pullback, f, x, (dy,); res1 = (grad,)) => true,
    ]
end

function complex_holomorphic_gradient_scenarios()
    # http://arxiv.org/abs/2409.06752
    x = [1.0 + im]
    return complex_gradient_canonical(square_only, x; grad = 2 * conj(x))
end

function complex_gradient_scenarios()
    x = [1.0 + im]
    return complex_gradient_canonical(abs2_only, x; grad = 2 * x)
end

"""
    complex_scenarios(; include_mismatched=true)

Create a vector of first-order [`TripleScenario`](@ref)s with complex-valued array types and holomorphic functions only.
"""
function complex_scenarios(; include_mismatched = true)
    x_ = 0.42 + im
    dx_ = 3.14 + im
    dy_ = -1 / 12 + im

    x_6 = float.(1:6) .+ im
    dx_6 = float.(-1:-1:-6) .+ im

    dy_2 = [-3.0, 4.0] .+ im
    dy_6 = float.(-5:2:5) .+ im
    dy_12 = float.(-11:2:11) .+ im

    canonical = vcat(
        # one argument
        num_to_num_canonical(x_; dx = dx_, dy = dy_),
        num_to_vec_canonical_onearg(x_; dx = dx_, dy = dy_2),
        arr_to_num_canonical_onearg(x_6; dx = dx_6, dy = dy_),
        vec_to_vec_canonical_onearg(x_6; dx = dx_6, dy = dy_12),
        # two arguments
        num_to_vec_canonical_twoarg(x_; dx = dx_, dy = dy_6),
        vec_to_vec_canonical_twoarg(x_6; dx = dx_6, dy = dy_12),
        # complex gradients
        complex_gradient_scenarios(),
        complex_holomorphic_gradient_scenarios(),
    )
    canonical = filter(p -> order(operator(first(p))) == 1, canonical)
    scens = TripleScenario[TripleScenario(v) for p in canonical for v in tagged_variants(p)]
    include_mismatched && add_mismatched!(scens)
    return scens
end

"""
    complex_sparse_scenarios(; include_mismatched=true)

Create a vector of Jacobian [`TripleScenario`](@ref)s with complex-valued array types and holomorphic functions only.
"""
function complex_sparse_scenarios(; include_mismatched = true)
    x_6 = float.(1:6) .+ float.(1:6) .^ 2 .* im
    scens = TripleScenario[sparse_vec_to_vec_scenarios(x_6)...]
    include_mismatched && add_mismatched!(scens)
    return scens
end
