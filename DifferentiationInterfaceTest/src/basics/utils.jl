"""
    custom_zero(x)

Return an object with the same structure as `x` but filled with zeros.

It is used to build the default preparation and flush inputs of a scenario, which ensures that preparation does not depend on the values of the inputs.

Methods are defined for numbers, arrays, tuples, named tuples, DifferentiationInterface contexts and `nothing`.
To use other types inside scenarios, define a method for `DifferentiationInterfaceTest.custom_zero`.
"""
custom_zero(x::Number) = zero(x)
custom_zero(x::AbstractArray{<:Number}) = zero(x)
custom_zero(x::AbstractArray) = map(custom_zero, x)
custom_zero(x::Union{Tuple, NamedTuple}) = map(custom_zero, x)
custom_zero(::Nothing) = nothing
custom_zero(c::Context) = DI.maker(c)(custom_zero(unwrap(c)))

function custom_zero(x)
    throw(
        ArgumentError(
            "No method of `DifferentiationInterfaceTest.custom_zero` for objects of type $(typeof(x)). Please define one to use them in scenarios.",
        ),
    )
end
