#=
Constraints on the scenarios:
- type-stable
- GPU-compatible (no scalar indexing)
- vary shapes to be tricky
=#

first_half(v::AbstractVector) = @view v[1:(length(v) ÷ 2)]
second_half(v::AbstractVector) = @view v[(length(v) ÷ 2 + 1):end]

## Number to number

num_to_num(x::Number)::Number = sin(x)

num_to_num_derivative(x) = cos(x)
num_to_num_second_derivative(x) = -sin(x)
num_to_num_pushforward(x, dx) = num_to_num_derivative(x) * dx
num_to_num_pullback(x, dy) = conj(num_to_num_derivative(x)) * dy

function num_to_num_scenarios(x::Number; dx::Number, dy::Number)
    f = num_to_num
    y = f(x)
    dy_from_dx = num_to_num_pushforward(x, dx)
    dx_from_dy = num_to_num_pullback(x, dy)
    der = num_to_num_derivative(x)
    der2 = num_to_num_second_derivative(x)

    # everyone out of place
    scens = LegacyScenario[
        LegacyScenario{:pushforward, :out}(f, x, (dx,); res1 = (dy_from_dx,)),
        LegacyScenario{:pullback, :out}(f, x, (dy,); res1 = (dx_from_dy,)),
        LegacyScenario{:derivative, :out}(f, x; res1 = der),
        LegacyScenario{:second_derivative, :out}(f, x; res1 = der, res2 = der2),
    ]
    return scens
end

onevec_to_onevec(x) = sin.(x)

function onevec_to_onevec!(y, x)
    map!(sin, y, x)
    return nothing
end

function onevec_to_onevec_scenarios_onearg(x::Number; dx::Number, dy::Number)
    f = num_to_num
    y = f(x)
    dy_from_dx = num_to_num_pushforward(x, dx)
    dx_from_dy = num_to_num_pullback(x, dy)
    der = num_to_num_derivative(x)

    jac = fill(der, 1, 1)

    scens = LegacyScenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                LegacyScenario{:pushforward, pl_op}(
                    onevec_to_onevec, [x], ([dx],); res1 = ([dy_from_dx],)
                ),
                LegacyScenario{:pullback, pl_op}(
                    onevec_to_onevec, [x], ([dy],); res1 = ([dx_from_dy],)
                ),
                LegacyScenario{:jacobian, pl_op}(onevec_to_onevec, [x]; res1 = jac),
            ],
        )
    end

    return scens
end

function onevec_to_onevec_scenarios_twoarg(x::Number; dx::Number, dy::Number)
    f = num_to_num
    y = f(x)
    dy_from_dx = num_to_num_pushforward(x, dx)
    dx_from_dy = num_to_num_pullback(x, dy)
    der = num_to_num_derivative(x)

    jac = fill(der, 1, 1)

    scens = LegacyScenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                LegacyScenario{:pushforward, pl_op}(
                    onevec_to_onevec!, [y], [x], ([dx],); res1 = ([dy_from_dx],)
                ),
                LegacyScenario{:pullback, pl_op}(
                    onevec_to_onevec!, [y], [x], ([dy],); res1 = ([dx_from_dy],)
                ),
                LegacyScenario{:jacobian, pl_op}(onevec_to_onevec!, [y], [x]; res1 = jac),
            ],
        )
    end

    return scens
end

## Number to vector

num_to_vec(x::Number) = sin.([1, 2] .* x)
num_to_vec_derivative(x) = [1, 2] .* cos.([1, 2] .* x)
num_to_vec_second_derivative(x) = [1, 2] .^ 2 .* (.-sin.([1, 2] .* x))
num_to_vec_pushforward(x, dx) = dx .* num_to_vec_derivative(x)
num_to_vec_pullback(x, dy) = sum(conj(num_to_vec_derivative(x)) .* dy)

function num_to_vec!(y::AbstractVector, x::Number)
    n = length(y)
    y[1:(n ÷ 2)] .= sin(x)
    y[((n ÷ 2) + 1):end] .= sin(2x)
    return nothing
end

function num_to_vec!_derivative(x; y)
    n = length(y)
    return vcat(fill(cos(x), n ÷ 2), fill(2cos(2x), n - n ÷ 2))
end

num_to_vec!_pushforward(x, dx; y) = dx .* num_to_vec!_derivative(x; y)

function num_to_vec!_pullback(x, dy)
    return sum(conj(num_to_vec!_derivative(x; y = similar(dy))) .* dy)
end

function num_to_vec_scenarios_onearg(x::Number; dx::Number, dy::AbstractArray)
    f = num_to_vec
    dy_from_dx = num_to_vec_pushforward(x, dx)
    dx_from_dy = num_to_vec_pullback(x, dy)
    der = num_to_vec_derivative(x)
    der2 = num_to_vec_second_derivative(x)

    # pullback stays out of place
    scens = LegacyScenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                LegacyScenario{:pushforward, pl_op}(f, x, (dx,); res1 = (dy_from_dx,)),
                LegacyScenario{:derivative, pl_op}(f, x; res1 = der),
                LegacyScenario{:second_derivative, pl_op}(f, x; res1 = der, res2 = der2),
            ],
        )
    end
    for pl_op in (:out,)
        append!(scens, [LegacyScenario{:pullback, pl_op}(f, x, (dy,); res1 = (dx_from_dy,))])
    end
    return scens
end

function num_to_vec_scenarios_twoarg(x::Number; dx::Number, dy::AbstractArray)
    f! = num_to_vec!
    y = similar(dy)
    f!(y, x)
    dy_from_dx = num_to_vec!_pushforward(x, dx; y)
    dx_from_dy = num_to_vec!_pullback(x, dy)
    der = num_to_vec!_derivative(x; y)

    # pullback stays out of place
    scens = LegacyScenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                LegacyScenario{:pushforward, pl_op}(f!, y, x, (dx,); res1 = (dy_from_dx,)),
                LegacyScenario{:derivative, pl_op}(f!, y, x; res1 = der),
            ],
        )
    end
    for pl_op in (:out,)
        append!(scens, [LegacyScenario{:pullback, pl_op}(f!, y, x, (dy,); res1 = (dx_from_dy,))])
    end
    return scens
end

## Number to matrix

num_to_mat(x::Number) = hcat(num_to_vec(x), num_to_vec(3x))

num_to_mat_derivative(x) = hcat(num_to_vec_derivative(x), 3 .* num_to_vec_derivative(3x))
function num_to_mat_second_derivative(x)
    return hcat(num_to_vec_second_derivative(x), 9 .* num_to_vec_second_derivative(3x))
end
function num_to_mat_pushforward(x, dx)
    return hcat(num_to_vec_pushforward(x, dx), 3 .* num_to_vec_pushforward(3x, dx))
end
function num_to_mat_pullback(x, dy)
    return num_to_vec_pullback(x, dy[:, 1]) + 3 * num_to_vec_pullback(3x, dy[:, 2])
end

function num_to_mat!(y::AbstractMatrix, x::Number)
    num_to_vec!(view(y, :, 1), x)
    num_to_vec!(view(y, :, 2), 3x)
    return nothing
end

function num_to_mat!_derivative(x; y)
    return hcat(
        num_to_vec!_derivative(x; y = y[:, 1]), 3 .* num_to_vec!_derivative(3x; y = y[:, 2])
    )
end
function num_to_mat!_pushforward(x, dx; y)
    return hcat(
        num_to_vec!_pushforward(x, dx; y = y[:, 1]),
        3 .* num_to_vec!_pushforward(3x, dx; y = y[:, 2]),
    )
end
function num_to_mat!_pullback(x, dy)
    return num_to_vec!_pullback(x, dy[:, 1]) + 3 * num_to_vec!_pullback(3x, dy[:, 2])
end

function num_to_mat_scenarios_onearg(x::Number; dx::Number, dy::AbstractArray)
    f = num_to_mat
    dy_from_dx = num_to_mat_pushforward(x, dx)
    dx_from_dy = num_to_mat_pullback(x, dy)
    der = num_to_mat_derivative(x)
    der2 = num_to_mat_second_derivative(x)

    # pullback stays out of place
    scens = LegacyScenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                LegacyScenario{:pushforward, pl_op}(f, x, (dx,); res1 = (dy_from_dx,)),
                LegacyScenario{:derivative, pl_op}(f, x; res1 = der),
                LegacyScenario{:second_derivative, pl_op}(f, x; res1 = der, res2 = der2),
            ],
        )
    end
    for pl_op in (:out,)
        append!(scens, [LegacyScenario{:pullback, pl_op}(f, x, (dy,); res1 = (dx_from_dy,))])
    end
    return scens
end

function num_to_mat_scenarios_twoarg(x::Number; dx::Number, dy::AbstractArray)
    f! = num_to_mat!
    y = similar(dy)
    f!(y, x)
    dy_from_dx = num_to_mat!_pushforward(x, dx; y)
    dx_from_dy = num_to_mat!_pullback(x, dy)
    der = num_to_mat!_derivative(x; y)

    # pullback stays out of place
    scens = LegacyScenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                LegacyScenario{:pushforward, pl_op}(f!, y, x, (dx,); res1 = (dy_from_dx,)),
                LegacyScenario{:derivative, pl_op}(f!, y, x; res1 = der),
            ],
        )
    end
    for pl_op in (:out,)
        append!(scens, [LegacyScenario{:pullback, pl_op}(f!, y, x, (dy,); res1 = (dx_from_dy,))])
    end
    return scens
end

## Array to number

const α = 4
const β = 6

arr_to_num_linalg(x::AbstractArray) = sum(vec(x .^ α) .* transpose(vec(x .^ β)))

function arr_to_num_no_linalg(x::AbstractArray)
    n = length(x)
    s = zero(eltype(x))
    for i in 1:n, j in 1:n
        s += x[i]^α * x[j]^β
    end
    return s
end

function arr_to_num_gradient(x0)
    x = Array(x0)  # GPU arrays don't like indexing
    g = similar(x)
    for k in eachindex(g, x)
        g[k] = (
            α * x[k]^(α - 1) * sum(x[j]^β for j in eachindex(x) if j != k) +
                β * x[k]^(β - 1) * sum(x[i]^α for i in eachindex(x) if i != k) +
                (α + β) * x[k]^(α + β - 1)
        )
    end
    return conj(convert(typeof(x0), g))
end

function arr_to_num_hessian(x0)
    x = Array(x0)  # GPU arrays don't like indexing
    H = similar(x, length(x), length(x))
    for k in axes(H, 1), l in axes(H, 2)
        if k == l
            H[k, k] = (
                α * (α - 1) * x[k]^(α - 2) * sum(x[j]^β for j in eachindex(x) if j != k) +
                    β * (β - 1) * x[k]^(β - 2) * sum(x[i]^α for i in eachindex(x) if i != k) +
                    (α + β) * (α + β - 1) * x[k]^(α + β - 2)
            )
        else
            H[k, l] = α * β * (x[k]^(α - 1) * x[l]^(β - 1) + x[k]^(β - 1) * x[l]^(α - 1))
        end
    end
    return convert(typeof(similar(x0, length(x0), length(x0))), H)
end

arr_to_num_pushforward(x, dx) = sum(conj.(arr_to_num_gradient(x)) .* dx)
arr_to_num_pullback(x, dy) = arr_to_num_gradient(x) .* dy
arr_to_num_hvp(x, dx) = reshape(arr_to_num_hessian(x) * vec(dx), size(x))

function arr_to_num_scenarios_onearg(
        x::AbstractArray; dx::AbstractArray, dy::Number, linalg = true
    )
    f = linalg ? arr_to_num_linalg : arr_to_num_no_linalg
    dy_from_dx = arr_to_num_pushforward(x, dx)
    dx_from_dy = arr_to_num_pullback(x, dy)
    grad = arr_to_num_gradient(x)
    dg = arr_to_num_hvp(x, dx)
    hess = arr_to_num_hessian(x)

    # pushforward stays out of place
    scens = LegacyScenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                LegacyScenario{:pullback, pl_op}(f, x, (dy,); res1 = (dx_from_dy,)),
                LegacyScenario{:gradient, pl_op}(f, x; res1 = grad),
                LegacyScenario{:hvp, pl_op}(f, x, (dx,); res1 = grad, res2 = (dg,)),
                LegacyScenario{:hessian, pl_op}(f, x; res1 = grad, res2 = hess),
            ],
        )
    end
    for pl_op in (:out,)
        append!(scens, [LegacyScenario{:pushforward, pl_op}(f, x, (dx,); res1 = (dy_from_dx,))])
    end
    return scens
end

## Array to array

function all_array_to_array_scenarios(f, x; dx, dy, dy_from_dx, dx_from_dy, jac)
    scens = LegacyScenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                LegacyScenario{:pushforward, pl_op}(f, x, (dx,); res1 = (dy_from_dx,)),
                LegacyScenario{:pullback, pl_op}(f, x, (dy,); res1 = (dx_from_dy,)),
                LegacyScenario{:jacobian, pl_op}(f, x; res1 = jac),
            ],
        )
    end
    return scens
end

function all_array_to_array_scenarios(f!, y, x; dx, dy, dy_from_dx, dx_from_dy, jac)
    scens = LegacyScenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                LegacyScenario{:pushforward, pl_op}(f!, y, x, (dx,); res1 = (dy_from_dx,)),
                LegacyScenario{:pullback, pl_op}(f!, y, x, (dy,); res1 = (dx_from_dy,)),
                LegacyScenario{:jacobian, pl_op}(f!, y, x; res1 = jac),
            ],
        )
    end
    return scens
end

### Vector to vector

vec_to_vec(x::AbstractVector)::AbstractVector = vcat(sin.(x), cos.(x))

function vec_to_vec!(y::AbstractVector, x::AbstractVector)
    y[1:length(x)] .= sin.(x)
    y[(length(x) + 1):(2length(x))] .= cos.(x)
    return nothing
end

vec_to_vec_pushforward(x, dx) = vcat(cos.(x) .* dx, -sin.(x) .* dx)
function vec_to_vec_pullback(x, dy)
    return conj(cos.(x)) .* first_half(dy) .- conj(sin.(x)) .* second_half(dy)
end
vec_to_vec_jacobian(x) = vcat(Diagonal(cos.(x)), Diagonal(-sin.(x)))

function vec_to_vec_scenarios_onearg(
        x::AbstractVector; dx::AbstractVector, dy::AbstractVector
    )
    f = vec_to_vec
    dy_from_dx = vec_to_vec_pushforward(x, dx)
    dx_from_dy = vec_to_vec_pullback(x, dy)
    jac = vec_to_vec_jacobian(x)

    return all_array_to_array_scenarios(f, x; dx, dy, dy_from_dx, dx_from_dy, jac)
end

function vec_to_vec_scenarios_twoarg(
        x::AbstractVector; dx::AbstractVector, dy::AbstractVector
    )
    f! = vec_to_vec!
    y = similar(vec_to_vec(x))
    f!(y, x)
    dy_from_dx = vec_to_vec_pushforward(x, dx)
    dx_from_dy = vec_to_vec_pullback(x, dy)
    jac = vec_to_vec_jacobian(x)

    return all_array_to_array_scenarios(f!, y, x; dx, dy, dy_from_dx, dx_from_dy, jac)
end

### Vector to matrix

vec_to_mat(x::AbstractVector)::AbstractMatrix = hcat(sin.(x), cos.(x))

function vec_to_mat!(y::AbstractMatrix, x::AbstractVector)
    y[:, 1] .= sin.(x)
    y[:, 2] .= cos.(x)
    return nothing
end

vec_to_mat_pushforward(x, dx) = hcat(cos.(x) .* dx, -sin.(x) .* dx)
vec_to_mat_pullback(x, dy) = conj(cos.(x)) .* dy[:, 1] .- conj(sin.(x)) .* dy[:, 2]
vec_to_mat_jacobian(x) = vcat(Diagonal(cos.(x)), Diagonal(-sin.(x)))

function vec_to_mat_scenarios_onearg(
        x::AbstractVector; dx::AbstractVector, dy::AbstractMatrix
    )
    f = vec_to_mat
    dy_from_dx = vec_to_mat_pushforward(x, dx)
    dx_from_dy = vec_to_mat_pullback(x, dy)
    jac = vec_to_mat_jacobian(x)

    return all_array_to_array_scenarios(f, x; dx, dy, dy_from_dx, dx_from_dy, jac)
end

function vec_to_mat_scenarios_twoarg(
        x::AbstractVector; dx::AbstractVector, dy::AbstractMatrix
    )
    f! = vec_to_mat!
    y = similar(vec_to_mat(x))
    f!(y, x)
    dy_from_dx = vec_to_mat_pushforward(x, dx)
    dx_from_dy = vec_to_mat_pullback(x, dy)
    jac = vec_to_mat_jacobian(x)

    return all_array_to_array_scenarios(f!, y, x; dx, dy, dy_from_dx, dx_from_dy, jac)
end

### Matrix to vector

mat_to_vec(x::AbstractMatrix)::AbstractVector = vcat(vec(sin.(x)), vec(cos.(x)))

function mat_to_vec!(y::AbstractVector, x::AbstractMatrix)
    n = length(x)
    y[1:n] .= sin.(getindex.(Ref(x), 1:n))
    y[(n + 1):(2n)] .= cos.(getindex.(Ref(x), 1:n))
    return nothing
end

function mat_to_vec_pushforward(x, dx)
    return vcat(vec(cos.(x) .* dx), vec(-sin.(x) .* dx))
end

function mat_to_vec_pullback(x, dy)
    return conj(cos.(x)) .* reshape(first_half(dy), size(x)) .-
        conj(sin.(x)) .* reshape(second_half(dy), size(x))
end

mat_to_vec_jacobian(x) = vcat(Diagonal(vec(cos.(x))), Diagonal(vec(-sin.(x))))

function mat_to_vec_scenarios_onearg(
        x::AbstractMatrix; dx::AbstractMatrix, dy::AbstractVector
    )
    f = mat_to_vec
    dy_from_dx = mat_to_vec_pushforward(x, dx)
    dx_from_dy = mat_to_vec_pullback(x, dy)
    jac = mat_to_vec_jacobian(x)

    return all_array_to_array_scenarios(f, x; dx, dy, dy_from_dx, dx_from_dy, jac)
end

function mat_to_vec_scenarios_twoarg(
        x::AbstractMatrix; dx::AbstractMatrix, dy::AbstractVector
    )
    f! = mat_to_vec!
    y = similar(mat_to_vec(x))
    f!(y, x)
    dy_from_dx = mat_to_vec_pushforward(x, dx)
    dx_from_dy = mat_to_vec_pullback(x, dy)
    jac = mat_to_vec_jacobian(x)

    return all_array_to_array_scenarios(f!, y, x; dx, dy, dy_from_dx, dx_from_dy, jac)
end

### Matrix to matrix

mat_to_mat(x::AbstractMatrix)::AbstractMatrix = hcat(vec(sin.(x)), vec(cos.(x)))

function mat_to_mat!(y::AbstractMatrix, x::AbstractMatrix)
    n = length(x)
    y[:, 1] .= sin.(getindex.(Ref(x), 1:n))
    y[:, 2] .= cos.(getindex.(Ref(x), 1:n))
    return nothing
end

function mat_to_mat_pushforward(x, dx)
    return hcat(vec(cos.(x) .* dx), vec(-sin.(x) .* dx))
end

function mat_to_mat_pullback(x, dy)
    return conj(cos.(x)) .* reshape(dy[:, 1], size(x)) .-
        conj(sin.(x)) .* reshape(dy[:, 2], size(x))
end

mat_to_mat_jacobian(x) = vcat(Diagonal(vec(cos.(x))), Diagonal(vec(-sin.(x))))

function mat_to_mat_scenarios_onearg(
        x::AbstractMatrix; dx::AbstractMatrix, dy::AbstractMatrix
    )
    f = mat_to_mat
    dy_from_dx = mat_to_mat_pushforward(x, dx)
    dx_from_dy = mat_to_mat_pullback(x, dy)
    jac = mat_to_mat_jacobian(x)

    return all_array_to_array_scenarios(f, x; dx, dy, dy_from_dx, dx_from_dy, jac)
end

function mat_to_mat_scenarios_twoarg(
        x::AbstractMatrix; dx::AbstractMatrix, dy::AbstractMatrix
    )
    f! = mat_to_mat!
    y = similar(mat_to_mat(x))
    f!(y, x)
    dy_from_dx = mat_to_mat_pushforward(x, dx)
    dx_from_dy = mat_to_mat_pullback(x, dy)
    jac = mat_to_mat_jacobian(x)

    return all_array_to_array_scenarios(f!, y, x; dx, dy, dy_from_dx, dx_from_dy, jac)
end

## Gather

"""
    legacy_default_scenarios()

Create a vector of [`LegacyScenario`](@ref)s with standard array types.

Only used by the array type extensions until they are ported to call-based scenarios.
"""
function legacy_default_scenarios(;
        linalg = true,
        include_normal = true,
        include_batchified = true,
        include_closurified = false,
        include_constantified = false,
        include_cachified = false,
        include_constantorcachified = false,
        use_tuples = false,
        include_smaller = false,
    )
    x_ = 0.42
    dx_ = 3.14
    dy_ = -1 / 12

    x_6 = float.(1:6)
    dx_6 = float.(-1:-1:-6)

    x_2_3 = float.(reshape(1:6, 2, 3))
    dx_2_3 = float.(reshape(-1:-1:-6, 2, 3))

    dy_2 = [-3.0, 4.0]
    dy_6 = float.(-5:2:5)
    dy_12 = float.(-11:2:11)
    dy_2_2 = [-3.0 4.0; 7.0 -1.0]
    dy_2_3 = float.(reshape(-5:2:5, 2, 3))
    dy_6_2 = float.(reshape(-11:2:11, 6, 2))

    scens = vcat(
        # one argument
        num_to_num_scenarios(x_; dx = dx_, dy = dy_),
        onevec_to_onevec_scenarios_onearg(x_; dx = dx_, dy = dy_),
        num_to_vec_scenarios_onearg(x_; dx = dx_, dy = dy_2),
        num_to_mat_scenarios_onearg(x_; dx = dx_, dy = dy_2_2),
        arr_to_num_scenarios_onearg(x_6; dx = dx_6, dy = dy_, linalg),
        arr_to_num_scenarios_onearg(x_2_3; dx = dx_2_3, dy = dy_, linalg),
        vec_to_vec_scenarios_onearg(x_6; dx = dx_6, dy = dy_12),
        vec_to_mat_scenarios_onearg(x_6; dx = dx_6, dy = dy_6_2),
        mat_to_vec_scenarios_onearg(x_2_3; dx = dx_2_3, dy = dy_12),
        mat_to_mat_scenarios_onearg(x_2_3; dx = dx_2_3, dy = dy_6_2),
        # two arguments
        onevec_to_onevec_scenarios_twoarg(x_; dx = dx_, dy = dy_),
        num_to_vec_scenarios_twoarg(x_; dx = dx_, dy = dy_6),
        num_to_mat_scenarios_twoarg(x_; dx = dx_, dy = dy_6_2),
        vec_to_vec_scenarios_twoarg(x_6; dx = dx_6, dy = dy_12),
        vec_to_mat_scenarios_twoarg(x_6; dx = dx_6, dy = dy_6_2),
        mat_to_vec_scenarios_twoarg(x_2_3; dx = dx_2_3, dy = dy_12),
        mat_to_mat_scenarios_twoarg(x_2_3; dx = dx_2_3, dy = dy_6_2),
    )

    smallerscens = vcat(
        # one argument
        num_to_num_scenarios(x_; dx = dx_, dy = dy_),
        onevec_to_onevec_scenarios_onearg(x_; dx = dx_, dy = dy_),
        num_to_vec_scenarios_onearg(x_; dx = dx_, dy = dy_2),
        num_to_mat_scenarios_onearg(x_; dx = dx_, dy = dy_2_2),
        arr_to_num_scenarios_onearg(x_6[1:3]; dx = dx_6[1:3], dy = dy_, linalg),
        arr_to_num_scenarios_onearg(x_2_3[1:1, 1:2]; dx = dx_2_3[1:1, 1:2], dy = dy_, linalg),
        vec_to_vec_scenarios_onearg(x_6[1:3]; dx = dx_6[1:3], dy = dy_12[1:6]),
        vec_to_mat_scenarios_onearg(x_6[1:3]; dx = dx_6[1:3], dy = dy_6_2[1:3, :]),
        mat_to_vec_scenarios_onearg(x_2_3[1:1, 1:2]; dx = dx_2_3[1:1, 1:2], dy = dy_12[1:4]),
        mat_to_mat_scenarios_onearg(
            x_2_3[1:1, 1:2]; dx = dx_2_3[1:1, 1:2], dy = dy_6_2[1:2, :]
        ),
        # two arguments
        onevec_to_onevec_scenarios_twoarg(x_; dx = dx_, dy = dy_),
        num_to_vec_scenarios_twoarg(x_; dx = dx_, dy = dy_6[1:3]),
        num_to_mat_scenarios_twoarg(x_; dx = dx_, dy = dy_6_2[1:3, :]),
        vec_to_vec_scenarios_twoarg(x_6[1:3]; dx = dx_6[1:3], dy = dy_12[1:6]),
        vec_to_mat_scenarios_twoarg(x_6[1:3]; dx = dx_6[1:3], dy = dy_6_2[1:3, :]),
        mat_to_vec_scenarios_twoarg(x_2_3[1:1, 1:2]; dx = dx_2_3[1:1, 1:2], dy = dy_12[1:4]),
        mat_to_mat_scenarios_twoarg(
            x_2_3[1:1, 1:2]; dx = dx_2_3[1:1, 1:2], dy = dy_6_2[1:2, :]
        ),
    )

    scens_smaller_prep = map(scens, smallerscens) do s1, s2
        LegacyScenario{operator(s1), operator_place(s1), function_place(s1)}(;
            f = s1.f,
            y = s1.y,
            x = s1.x,
            t = s1.t,
            contexts = s1.contexts,
            res1 = s1.res1,
            res2 = s1.res2,
            name = isnothing(s1.name) ? nothing : s1.name * " [smaller prep]",
            prep_args = s2.prep_args,
        )
    end

    include_batchified && append!(scens, batchify(scens))

    final_scens = LegacyScenario[]
    include_normal && append!(final_scens, scens)
    include_closurified && append!(final_scens, closurify(scens))
    include_constantified && append!(final_scens, constantify(scens))
    include_cachified && append!(final_scens, cachify(scens; use_tuples = use_tuples))
    include_constantorcachified && append!(final_scens, constantorcachify(scens))
    include_smaller && append!(final_scens, scens_smaller_prep)

    return final_scens
end

## Call-based scenarios

make_post(x, t) = isnothing(t) ? (x,) : (x, t)

"""
    add_mismatched!(scens)

Append one preparation mismatch scenario for each scenario of `scens`.
"""
function add_mismatched!(scens::AbstractVector)
    return append!(scens, map(ts -> mismatch(ts; kind = :function), scens))
end

"""
    canonical_scenario(op, f, x, [t]; res1, res2=nothing)

Create the canonical scenario of an operator family for an out-of-place function `f`: out-of-place operator with value, whose primal output is computed by calling `f`.
"""
function canonical_scenario(op, f, x, t = nothing; res1, res2 = nothing)
    y = f(x)
    return Scenario(op, f, (), make_post(x, t); result = Result(y, res1, res2))
end

"""
    canonical_scenario!(op, f!, y, x, [t]; res1)

Create the canonical scenario of an operator family for an in-place function `f!(y, x)`.
"""
function canonical_scenario!(op, f!, y, x, t = nothing; res1)
    return Scenario(op, f!, (copy(y),), make_post(x, t); result = Result(copy(y), res1, nothing))
end

function num_to_num_canonical(x::Number; dx::Number, dy::Number)
    f = num_to_num
    der = num_to_num_derivative(x)
    return [
        canonical_scenario(value_and_pushforward, f, x, (dx,); res1 = (num_to_num_pushforward(x, dx),)),
        canonical_scenario(value_and_pullback, f, x, (dy,); res1 = (num_to_num_pullback(x, dy),)),
        canonical_scenario(value_and_derivative, f, x; res1 = der),
        canonical_scenario(
            value_derivative_and_second_derivative, f, x; res1 = der, res2 = num_to_num_second_derivative(x)
        ),
    ]
end

function onevec_to_onevec_canonical_onearg(x::Number; dx::Number, dy::Number)
    f = onevec_to_onevec
    return [
        canonical_scenario(value_and_pushforward, f, [x], ([dx],); res1 = ([num_to_num_pushforward(x, dx)],)),
        canonical_scenario(value_and_pullback, f, [x], ([dy],); res1 = ([num_to_num_pullback(x, dy)],)),
        canonical_scenario(value_and_jacobian, f, [x]; res1 = fill(num_to_num_derivative(x), 1, 1)),
    ]
end

function onevec_to_onevec_canonical_twoarg(x::Number; dx::Number, dy::Number)
    f! = onevec_to_onevec!
    y = [num_to_num(x)]
    return [
        canonical_scenario!(value_and_pushforward, f!, y, [x], ([dx],); res1 = ([num_to_num_pushforward(x, dx)],)),
        canonical_scenario!(value_and_pullback, f!, y, [x], ([dy],); res1 = ([num_to_num_pullback(x, dy)],)),
        canonical_scenario!(value_and_jacobian, f!, y, [x]; res1 = fill(num_to_num_derivative(x), 1, 1)),
    ]
end

function num_to_arr_canonical_onearg(f, x::Number; dx::Number, dy::AbstractArray, der, der2, pushforward, pullback)
    return [
        canonical_scenario(value_and_pushforward, f, x, (dx,); res1 = (pushforward(x, dx),)),
        canonical_scenario(value_and_derivative, f, x; res1 = der(x)),
        canonical_scenario(value_derivative_and_second_derivative, f, x; res1 = der(x), res2 = der2(x)),
        canonical_scenario(value_and_pullback, f, x, (dy,); res1 = (pullback(x, dy),)),
    ]
end

function num_to_arr_canonical_twoarg(f!, x::Number; dx::Number, dy::AbstractArray, der, pushforward, pullback)
    y = similar(dy)
    f!(y, x)
    return [
        canonical_scenario!(value_and_pushforward, f!, y, x, (dx,); res1 = (pushforward(x, dx; y),)),
        canonical_scenario!(value_and_derivative, f!, y, x; res1 = der(x; y)),
        canonical_scenario!(value_and_pullback, f!, y, x, (dy,); res1 = (pullback(x, dy),)),
    ]
end

function num_to_vec_canonical_onearg(x::Number; dx::Number, dy::AbstractArray)
    return num_to_arr_canonical_onearg(
        num_to_vec, x; dx, dy,
        der = num_to_vec_derivative, der2 = num_to_vec_second_derivative,
        pushforward = num_to_vec_pushforward, pullback = num_to_vec_pullback,
    )
end

function num_to_vec_canonical_twoarg(x::Number; dx::Number, dy::AbstractArray)
    return num_to_arr_canonical_twoarg(
        num_to_vec!, x; dx, dy,
        der = num_to_vec!_derivative, pushforward = num_to_vec!_pushforward,
        pullback = num_to_vec!_pullback,
    )
end

function num_to_mat_canonical_onearg(x::Number; dx::Number, dy::AbstractArray)
    return num_to_arr_canonical_onearg(
        num_to_mat, x; dx, dy,
        der = num_to_mat_derivative, der2 = num_to_mat_second_derivative,
        pushforward = num_to_mat_pushforward, pullback = num_to_mat_pullback,
    )
end

function num_to_mat_canonical_twoarg(x::Number; dx::Number, dy::AbstractArray)
    return num_to_arr_canonical_twoarg(
        num_to_mat!, x; dx, dy,
        der = num_to_mat!_derivative, pushforward = num_to_mat!_pushforward,
        pullback = num_to_mat!_pullback,
    )
end

function arr_to_num_canonical_onearg(x::AbstractArray; dx::AbstractArray, dy::Number, linalg = true)
    f = linalg ? arr_to_num_linalg : arr_to_num_no_linalg
    grad = arr_to_num_gradient(x)
    return [
        canonical_scenario(value_and_pullback, f, x, (dy,); res1 = (arr_to_num_pullback(x, dy),)),
        canonical_scenario(value_and_gradient, f, x; res1 = grad),
        canonical_scenario(value_gradient_and_hvp, f, x, (dx,); res1 = grad, res2 = (arr_to_num_hvp(x, dx),)),
        canonical_scenario(value_gradient_and_hessian, f, x; res1 = grad, res2 = arr_to_num_hessian(x)),
        canonical_scenario(value_and_pushforward, f, x, (dx,); res1 = (arr_to_num_pushforward(x, dx),)),
    ]
end

function arr_to_arr_canonical(f, x; dx, dy, dy_from_dx, dx_from_dy, jac)
    return [
        canonical_scenario(value_and_pushforward, f, x, (dx,); res1 = (dy_from_dx,)),
        canonical_scenario(value_and_pullback, f, x, (dy,); res1 = (dx_from_dy,)),
        canonical_scenario(value_and_jacobian, f, x; res1 = jac),
    ]
end

function arr_to_arr_canonical!(f!, y, x; dx, dy, dy_from_dx, dx_from_dy, jac)
    return [
        canonical_scenario!(value_and_pushforward, f!, y, x, (dx,); res1 = (dy_from_dx,)),
        canonical_scenario!(value_and_pullback, f!, y, x, (dy,); res1 = (dx_from_dy,)),
        canonical_scenario!(value_and_jacobian, f!, y, x; res1 = jac),
    ]
end

for (name, f, f!, pushforward, pullback, jacobian) in [
        (:vec_to_vec, :vec_to_vec, :vec_to_vec!, :vec_to_vec_pushforward, :vec_to_vec_pullback, :vec_to_vec_jacobian),
        (:vec_to_mat, :vec_to_mat, :vec_to_mat!, :vec_to_mat_pushforward, :vec_to_mat_pullback, :vec_to_mat_jacobian),
        (:mat_to_vec, :mat_to_vec, :mat_to_vec!, :mat_to_vec_pushforward, :mat_to_vec_pullback, :mat_to_vec_jacobian),
        (:mat_to_mat, :mat_to_mat, :mat_to_mat!, :mat_to_mat_pushforward, :mat_to_mat_pullback, :mat_to_mat_jacobian),
    ]
    onearg = Symbol(name, "_canonical_onearg")
    twoarg = Symbol(name, "_canonical_twoarg")
    @eval function $onearg(x::AbstractArray; dx::AbstractArray, dy::AbstractArray)
        return arr_to_arr_canonical(
            $f, x; dx, dy, dy_from_dx = $pushforward(x, dx), dx_from_dy = $pullback(x, dy),
            jac = $jacobian(x),
        )
    end
    @eval function $twoarg(x::AbstractArray; dx::AbstractArray, dy::AbstractArray)
        y = similar($f(x))
        $f!(y, x)
        return arr_to_arr_canonical!(
            $f!, y, x; dx, dy, dy_from_dx = $pushforward(x, dx), dx_from_dy = $pullback(x, dy),
            jac = $jacobian(x),
        )
    end
end

function default_canonical_scenarios(; linalg, smaller::Bool)
    x_ = 0.42
    dx_ = 3.14
    dy_ = -1 / 12

    x_6 = float.(1:6)
    dx_6 = float.(-1:-1:-6)

    x_2_3 = float.(reshape(1:6, 2, 3))
    dx_2_3 = float.(reshape(-1:-1:-6, 2, 3))

    dy_2 = [-3.0, 4.0]
    dy_6 = float.(-5:2:5)
    dy_12 = float.(-11:2:11)
    dy_2_2 = [-3.0 4.0; 7.0 -1.0]
    dy_6_2 = float.(reshape(-11:2:11, 6, 2))

    if !smaller
        return vcat(
            # one argument
            num_to_num_canonical(x_; dx = dx_, dy = dy_),
            onevec_to_onevec_canonical_onearg(x_; dx = dx_, dy = dy_),
            num_to_vec_canonical_onearg(x_; dx = dx_, dy = dy_2),
            num_to_mat_canonical_onearg(x_; dx = dx_, dy = dy_2_2),
            arr_to_num_canonical_onearg(x_6; dx = dx_6, dy = dy_, linalg),
            arr_to_num_canonical_onearg(x_2_3; dx = dx_2_3, dy = dy_, linalg),
            vec_to_vec_canonical_onearg(x_6; dx = dx_6, dy = dy_12),
            vec_to_mat_canonical_onearg(x_6; dx = dx_6, dy = dy_6_2),
            mat_to_vec_canonical_onearg(x_2_3; dx = dx_2_3, dy = dy_12),
            mat_to_mat_canonical_onearg(x_2_3; dx = dx_2_3, dy = dy_6_2),
            # two arguments
            onevec_to_onevec_canonical_twoarg(x_; dx = dx_, dy = dy_),
            num_to_vec_canonical_twoarg(x_; dx = dx_, dy = dy_6),
            num_to_mat_canonical_twoarg(x_; dx = dx_, dy = dy_6_2),
            vec_to_vec_canonical_twoarg(x_6; dx = dx_6, dy = dy_12),
            vec_to_mat_canonical_twoarg(x_6; dx = dx_6, dy = dy_6_2),
            mat_to_vec_canonical_twoarg(x_2_3; dx = dx_2_3, dy = dy_12),
            mat_to_mat_canonical_twoarg(x_2_3; dx = dx_2_3, dy = dy_6_2),
        )
    else
        return vcat(
            # one argument
            num_to_num_canonical(x_; dx = dx_, dy = dy_),
            onevec_to_onevec_canonical_onearg(x_; dx = dx_, dy = dy_),
            num_to_vec_canonical_onearg(x_; dx = dx_, dy = dy_2),
            num_to_mat_canonical_onearg(x_; dx = dx_, dy = dy_2_2),
            arr_to_num_canonical_onearg(x_6[1:3]; dx = dx_6[1:3], dy = dy_, linalg),
            arr_to_num_canonical_onearg(x_2_3[1:1, 1:2]; dx = dx_2_3[1:1, 1:2], dy = dy_, linalg),
            vec_to_vec_canonical_onearg(x_6[1:3]; dx = dx_6[1:3], dy = dy_12[1:6]),
            vec_to_mat_canonical_onearg(x_6[1:3]; dx = dx_6[1:3], dy = dy_6_2[1:3, :]),
            mat_to_vec_canonical_onearg(x_2_3[1:1, 1:2]; dx = dx_2_3[1:1, 1:2], dy = dy_12[1:4]),
            mat_to_mat_canonical_onearg(x_2_3[1:1, 1:2]; dx = dx_2_3[1:1, 1:2], dy = dy_6_2[1:2, :]),
            # two arguments
            onevec_to_onevec_canonical_twoarg(x_; dx = dx_, dy = dy_),
            num_to_vec_canonical_twoarg(x_; dx = dx_, dy = dy_6[1:3]),
            num_to_mat_canonical_twoarg(x_; dx = dx_, dy = dy_6_2[1:3, :]),
            vec_to_vec_canonical_twoarg(x_6[1:3]; dx = dx_6[1:3], dy = dy_12[1:6]),
            vec_to_mat_canonical_twoarg(x_6[1:3]; dx = dx_6[1:3], dy = dy_6_2[1:3, :]),
            mat_to_vec_canonical_twoarg(x_2_3[1:1, 1:2]; dx = dx_2_3[1:1, 1:2], dy = dy_12[1:4]),
            mat_to_mat_canonical_twoarg(x_2_3[1:1, 1:2]; dx = dx_2_3[1:1, 1:2], dy = dy_6_2[1:2, :]),
        )
    end
end

"""
    default_scenarios(; kwargs...)

Create a vector of [`TripleScenario`](@ref)s with standard array types, covering every operator variant.

# Keyword arguments

- `linalg=true`: whether array-to-number functions use linear algebra
- `include_normal=true`: include the base scenarios
- `include_batchified=true`: add scenarios with two tangents for `pushforward`, `pullback` and `hvp`
- `include_closurified=false`, `include_constantified=false`, `include_cachified=false`, `include_constantorcachified=false`: add scenarios transformed by the corresponding modifiers
- `use_tuples=false`: whether cachified scenarios store their caches in tuples
- `include_smaller=false`: add scenarios prepared on smaller inputs
- `include_mismatched=true`: add scenarios whose preparation does not match their execution
- `modifiers=()`: additional scenario modifiers whose outputs are added
"""
function default_scenarios(;
        linalg = true,
        include_normal = true,
        include_batchified = true,
        include_closurified = false,
        include_constantified = false,
        include_cachified = false,
        include_constantorcachified = false,
        use_tuples = false,
        include_smaller = false,
        include_mismatched = true,
        modifiers = (),
    )
    canonical = default_canonical_scenarios(; linalg, smaller = false)
    scens = map(TripleScenario, mapreduce(all_variants, vcat, canonical))
    smaller_scens = if include_smaller
        smaller_canonical = default_canonical_scenarios(; linalg, smaller = true)
        smaller_variants = mapreduce(all_variants, vcat, smaller_canonical)
        map(scens, smaller_variants) do ts, small
            @assert operator(ts) === operator(small)
            TripleScenario(
                exec_scenario(ts); prep = zero_inputs(small)
            )
        end
    else
        TripleScenario[]
    end
    include_batchified && append!(scens, batchify(scens))

    final_scens = TripleScenario[]
    include_normal && append!(final_scens, scens)
    include_closurified && append!(final_scens, closurify(scens))
    include_constantified && append!(final_scens, constantify(scens))
    include_cachified && append!(final_scens, cachify(scens; use_tuples))
    include_constantorcachified && append!(final_scens, constantorcachify(scens))
    for m in modifiers
        append!(final_scens, m(scens))
    end
    include_mismatched && add_mismatched!(final_scens)
    include_smaller && append!(final_scens, smaller_scens)
    return final_scens
end
