module DifferentiationInterfaceTestStaticArraysExt

import DifferentiationInterfaceTest as DIT
using StaticArrays: MMatrix, MVector, SMatrix, SVector
using PrecompileTools: @compile_workload

static_num_to_vec(x::Number) = sin.(SVector(1, 2) .* x)
static_num_to_mat(x::Number) = hcat(static_num_to_vec(x), static_num_to_vec(3x))

static_func(f) = f
static_func(::typeof(DIT.num_to_vec)) = static_num_to_vec
static_func(::typeof(DIT.num_to_mat)) = static_num_to_mat

mystatic(x::AbstractVector{T}) where {T} = convert(SVector{length(x), T}, x)
function mystatic(x::AbstractMatrix{T}) where {T}
    return convert(SMatrix{size(x, 1), size(x, 2), T, length(x)}, x)
end

mymutablestatic(x::AbstractVector{T}) where {T} = convert(MVector{length(x), T}, x)
function mymutablestatic(x::AbstractMatrix{T}) where {T}
    return convert(MMatrix{size(x, 1), size(x, 2), T, length(x)}, x)
end

function DIT.static_scenarios(args...; kwargs...)
    scens = DIT.default_scenarios(args...; kwargs...)
    return DIT.map_arrays(mystatic, mymutablestatic; func = static_func)(scens)
end

@compile_workload begin
    DIT.static_scenarios(; include_constantified = true, include_cachified = true)
end

end
