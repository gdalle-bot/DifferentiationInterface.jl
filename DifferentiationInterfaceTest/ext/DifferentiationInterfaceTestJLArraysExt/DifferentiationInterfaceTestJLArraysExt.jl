module DifferentiationInterfaceTestJLArraysExt

import DifferentiationInterfaceTest as DIT
using JLArrays: jl
using PrecompileTools: @compile_workload

jl_num_to_vec(x::Number) = sin.(jl([1, 2]) .* x)
jl_num_to_mat(x::Number) = hcat(jl_num_to_vec(x), jl_num_to_vec(3x))

jl_func(f) = f
jl_func(::typeof(DIT.num_to_vec)) = jl_num_to_vec
jl_func(::typeof(DIT.num_to_mat)) = jl_num_to_mat

function DIT.gpu_scenarios(args...; kwargs...)
    scens = DIT.default_scenarios(args...; kwargs...)
    return DIT.map_arrays(jl, jl; func = jl_func)(scens)
end

@compile_workload begin
    DIT.gpu_scenarios(; include_constantified = true, include_cachified = true)
end

end
