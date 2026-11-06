function identity_scenarios(x::Number; dx::Number, dy::Number)
    f = identity
    canonical = [
        canonical_scenario(value_and_pushforward, f, x, (dx,); res1 = (dx,)),
        canonical_scenario(value_and_pullback, f, x, (dy,); res1 = (dy,)),
        canonical_scenario(value_and_derivative, f, x; res1 = oneunit(x)),
    ]
    return mapreduce(s -> place_variants(s; inplace_op = false), vcat, canonical)
end

function sum_scenarios(x::AbstractArray; dx::AbstractArray, dy::Number)
    f = sum
    grad = similar(x)
    grad .= oneunit(eltype(x))
    pushforward_scen = canonical_scenario(value_and_pushforward, f, x, (dx,); res1 = (sum(dx),))
    pullback_scen = canonical_scenario(value_and_pullback, f, x, (dy,); res1 = ((similar(x) .= dy),))
    gradient_scen = canonical_scenario(value_and_gradient, f, x; res1 = grad)
    return vcat(
        place_variants(pushforward_scen; inplace_op = false),
        place_variants(pullback_scen; inplace_op = true),
        place_variants(gradient_scen; inplace_op = true),
    )
end

function copyto!_scenarios(x::AbstractArray; dx::AbstractArray, dy::AbstractArray)
    f! = copyto!
    y = similar(x)
    f!(y, x)
    jac = Matrix(Diagonal(ones(eltype(x), length(x))))
    canonical = [
        canonical_scenario!(value_and_pushforward, f!, y, x, (dx,); res1 = (dx,)),
        canonical_scenario!(value_and_pullback, f!, y, x, (dy,); res1 = (dy,)),
        canonical_scenario!(value_and_jacobian, f!, y, x; res1 = jac),
    ]
    return mapreduce(s -> place_variants(s; inplace_op = true), vcat, canonical)
end

"""
    allocfree_scenarios()

Create a vector of [`TripleScenario`](@ref)s with functions and operators that do not allocate.

!!! warning

    At the moment, second-order scenarios are excluded.
"""
function allocfree_scenarios()
    x_ = 0.42
    dx_ = 3.14
    dy_ = -1 / 12

    x_6 = float.(1:6)
    dx_6 = float.(-1:-1:-6)
    dy_6 = float.(-5:2:5)

    scens = vcat(
        identity_scenarios(x_; dx = dx_, dy = dy_), #
        sum_scenarios(x_6; dx = dx_6, dy = dy_),
        copyto!_scenarios(x_6; dx = dx_6, dy = dy_6),
    )
    return map(TripleScenario, scens)
end
