module UtilsTests

using DifferentiationInterface
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest: custom_zero
using SparseArrays
using Test

struct MyParameters
    a::Float64
end

DIT.custom_zero(::MyParameters) = MyParameters(0.0)

@testset "custom_zero" begin
    @test custom_zero(2.0) === 0.0
    @test custom_zero(3) === 0
    @test custom_zero(1.0 + 2.0im) === 0.0 + 0.0im
    @test custom_zero([1.0, 2.0]) == [0.0, 0.0]
    A = sparse([1, 2, 1], [1, 2, 2], [1.0, 2.0, 0.0])
    A0 = custom_zero(A)
    @test A0 isa SparseMatrixCSC
    @test iszero(A0)
    @test findnz(A0)[1:2] == findnz(A)[1:2]  # explicit zeros are kept
    @test A == sparse([1, 2, 1], [1, 2, 2], [1.0, 2.0, 0.0])
    @test custom_zero(sparsevec([2], [1.0], 3)) == sparsevec([2], [0.0], 3)
    @test custom_zero([[1.0], [2.0, 3.0]]) == [[0.0], [0.0, 0.0]]
    @test custom_zero(([1.0], 2.0)) == ([0.0], 0.0)
    @test custom_zero((; a = [1.0], b = 2.0)) == (; a = [0.0], b = 0.0)
    @test custom_zero(nothing) === nothing

    # the original is left untouched
    x = [1.0, 2.0]
    @test custom_zero(x) !== x
    @test x == [1.0, 2.0]

    # contexts keep their type
    @test custom_zero(Constant(2.0)) isa Constant
    @test DI.unwrap(custom_zero(Constant(2.0))) == 0.0
    @test custom_zero(Cache([1.0])) isa Cache
    @test DI.unwrap(custom_zero(Cache((; a = [1.0])))) == (; a = [0.0])
    @test custom_zero(ConstantOrCache([1.0])) isa ConstantOrCache

    # errors for unknown types
    @test_throws ArgumentError custom_zero(:a)
    @test_throws ArgumentError custom_zero(Constant("a"))

    # user extension
    @test custom_zero(MyParameters(2.0)) == MyParameters(0.0)
    @test DI.unwrap(custom_zero(Constant(MyParameters(2.0)))) == MyParameters(0.0)
end

end
