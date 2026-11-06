module DifferentiationInterfaceTestComponentArraysExt

using ComponentArrays: ComponentVector
using DifferentiationInterface: value_and_gradient, value_and_pullback, value_and_pushforward
import DifferentiationInterfaceTest as DIT
using LinearAlgebra: dot

## Vector to scalar

function comp_to_num(x::ComponentVector)::Number
    return sum(sin.(x.a)) + sum(cos.(x.b))
end

comp_to_num_gradient(x) = ComponentVector(; a = cos.(x.a), b = (-sin.(x.b)))

function comp_to_num_pushforward(x, dx)
    g = comp_to_num_gradient(x)
    return dot(g, dx)
end

function comp_to_num_pullback(x, dy)
    return comp_to_num_gradient(x) .* dy
end

function comp_to_num_canonical_onearg(x::ComponentVector; dx::AbstractVector, dy::Number)
    f = comp_to_num
    return [
        DIT.canonical_scenario(
            value_and_pullback, f, x, (dy,); res1 = (comp_to_num_pullback(x, dy),)
        ),
        DIT.canonical_scenario(value_and_gradient, f, x; res1 = comp_to_num_gradient(x)),
        DIT.canonical_scenario(
            value_and_pushforward, f, x, (dx,); res1 = (comp_to_num_pushforward(x, dx),)
        ),
    ]
end

## Gather

function DIT.component_scenarios(; include_mismatched = true)
    dy_ = -1 / 12

    x_comp = ComponentVector(; a = float.(1:4), b = float.(5:6))
    dx_comp = ComponentVector(; a = float.(4:-1:1), b = float.(6:-1:5))

    canonical = comp_to_num_canonical_onearg(x_comp; dx = dx_comp, dy = dy_)
    scens = DIT.TripleScenario[
        DIT.TripleScenario(v) for s in canonical for v in DIT.all_variants(s)
    ]
    include_mismatched && DIT.add_mismatched!(scens)
    return scens
end

end
