#=
Function wrappers used by scenario modifiers, for out-of-place (`:out`) or in-place (`:in`) functions.
=#

abstract type FunctionModifier end

struct WritableClosure{pl_fun, F, X, Y} <: FunctionModifier
    f::F
    x_buffer::Vector{X}
    y_buffer::Vector{Y}
    a::Float64
    b::Vector{Float64}
end

function WritableClosure{pl_fun}(
        f::F, x_buffer::Vector{X}, y_buffer::Vector{Y}, a, b
    ) where {pl_fun, F, X, Y}
    return WritableClosure{pl_fun, F, X, Y}(f, x_buffer, y_buffer, a, b)
end

Base.show(io::IO, f::WritableClosure) = print(io, "WritableClosure($(f.f))")

function (mc::WritableClosure{:out})(x)
    (; f, x_buffer, y_buffer, a, b) = mc
    x_buffer[1] = copy(x)
    y_buffer[1] = (a + only(b)) * f(x)
    return copy(y_buffer[1])
end

function (mc::WritableClosure{:in})(y, x)
    (; f, x_buffer, y_buffer, a, b) = mc
    x_buffer[1] = copy(x)
    f(y_buffer[1], x_buffer[1])
    y_buffer[1] .*= (a + only(b))
    copyto!(y, y_buffer[1])
    return nothing
end

struct MultiplyByConstant{pl_fun, F} <: FunctionModifier
    f::F
end

MultiplyByConstant{pl_fun}(f::F) where {pl_fun, F} = MultiplyByConstant{pl_fun, F}(f)

Base.show(io::IO, f::MultiplyByConstant) = print(io, "MultiplyByConstant($(f.f))")

function (mc::MultiplyByConstant{:out})(x, a)
    y = a * mc.f(x)
    return y
end

function (mc::MultiplyByConstant{:in})(y, x, a)
    mc.f(y, x)
    y .*= a
    return nothing
end

struct StoreInCache{pl_fun, F} <: FunctionModifier
    f::F
end

function StoreInCache{pl_fun}(f::F) where {pl_fun, F}
    return StoreInCache{pl_fun, F}(f)
end

Base.show(io::IO, f::StoreInCache) = print(io, "StoreInCache($(f.f))")

(sc::StoreInCache{:out})(x, y_cache::NamedTuple) = sc(x, y_cache.useful_cache)
(sc::StoreInCache{:in})(y, x, y_cache::NamedTuple) = sc(y, x, y_cache.useful_cache)
(sc::StoreInCache{:out})(x, y_cache::Tuple) = sc(x, first(y_cache))
(sc::StoreInCache{:in})(y, x, y_cache::Tuple) = sc(y, x, first(y_cache))

function (sc::StoreInCache{:out})(x, y_cache)  # no annotation otherwise Zygote.Buffer cries
    y = sc.f(x)
    if y isa Number
        y_cache[1] = y
        return y_cache[1]
    else
        copyto!(y_cache, y)
        return copy(y_cache)
    end
end

function (sc::StoreInCache{:in})(y, x, y_cache)
    sc.f(y_cache, x)
    copyto!(y, y_cache)
    return nothing
end

struct MultiplyByConstantAndStoreInCache{pl_fun, F} <: FunctionModifier
    f::F
end

function MultiplyByConstantAndStoreInCache{pl_fun}(f::F) where {pl_fun, F}
    return MultiplyByConstantAndStoreInCache{pl_fun, F}(f)
end

function Base.show(io::IO, f::MultiplyByConstantAndStoreInCache)
    return print(io, "MultiplyByConstantAndStoreInCache($(f.f))")
end

function (sc::MultiplyByConstantAndStoreInCache{:out})(x, constantorcache)
    (; constant, cache) = constantorcache
    (; a, b) = constant
    y = (a + only(b)) * sc.f(x)
    if eltype(y) == eltype(cache)
        newcache = cache
    else
        # poor man's PreallocationTools
        newcache = similar(cache, eltype(y))
    end
    if y isa Number
        newcache[1] = y
        return newcache[1]
    else
        copyto!(newcache, y)
        return copy(newcache)
    end
end

function (sc::MultiplyByConstantAndStoreInCache{:in})(y, x, constantorcache)
    (; constant, cache) = constantorcache
    (; a, b) = constant
    if eltype(y) == eltype(cache)
        newcache = cache
    else
        # poor man's PreallocationTools
        newcache = similar(cache, eltype(y))
    end
    sc.f(newcache, x)
    newcache .*= (a + only(b))
    copyto!(y, newcache)
    return nothing
end
